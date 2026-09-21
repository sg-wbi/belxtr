import json
import multiprocessing as mp
import os
import time  # noqa: F401

# import time
import datasets
import hydra
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, is_bf16_available, set_seed
from datasets import Split
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

# from tqdm import tqdm
from src import utils
from src.index import BaseAcceleratedIndex, IndexConfig, get_index
from src.model import (
    BaseRetriever,
    MultiVectorRetriever,
    get_model,
    get_optimizer,
    load_checkpoint,
)
from src.task import Task, get_task
from src.task.collator import MultiVectorRetrieverCollator

# datasets.disable_caching()  # you need a bit more code but loading from cached `map` is a pain for large datasets
# # https://github.com/pytorch/pytorch/issues/145755
# warnings.filterwarnings(
#     "ignore",
#     message="Skipping serialization of skipfiles_inline_module_allowlist value",
# )


# @utils.timeit
def mine_negatives(
    batch: dict,
    model: BaseRetriever,
    index: BaseAcceleratedIndex,
    kb: datasets.Dataset,
    task: Task,
    name_idx_to_entity_idx: dict[int, int] | None = None,
):
    model.eval()

    is_multivector = isinstance(model, MultiVectorRetriever)

    model_kwargs = {
        "query_input_ids": batch["input_ids"],
        "query_attention_mask": batch["attention_mask"],
        "query_subword_mask": batch["query_subword_mask"],
    }
    if is_multivector:
        assert isinstance(task.collator, MultiVectorRetrieverCollator)
        if task.collator.qe and not task.collator.qe_negative_mining:
            # NOTE: using [MASK] embeddings makes training ~2x slower
            model_kwargs["query_subword_mask"] = (
                batch["query_subword_mask"] - batch["expansion_subword_mask"]
            )

    with torch.no_grad():
        query = model(**model_kwargs)

    search_kwargs = {
        "queries": query["query_embedding"],
        "topk": task.collator.topk_train,
    }
    if is_multivector:
        search_kwargs["lengths"] = query["query_subword_mask"].sum(-1)
        search_kwargs["token_topk"] = task.eval.token_topk_predict

    retrieval = index.search(**search_kwargs)

    labels = task.collator.build_labels(
        gold_idxs=batch["idx"],
        candidates_idxs=retrieval["candidates_ids"],
        name_idx_to_entity_idx=name_idx_to_entity_idx,
        qe_idxs=batch.get("qe_idx"),
    )

    candidates = task.collator.collate_kb([kb[kb_idx] for kb_idx in labels["idxs"]])

    return candidates, labels


def display_bar(bar, epoch: int, step: int, values: dict):
    pbar_items = []
    keys = ["loss", "ret", "qe"]
    for k in keys:
        v = values.get(k)
        if v is None:
            continue
        pbar_items.append(f"{k}:{round(v.item(),2)}")
    pbar_str = f"Epoch:{epoch} - step:{step} - {' - '.join(pbar_items)}"
    bar.set_description(pbar_str)


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig):  # noqa: C901
    # cache_dir = utils.get_cache_dir(cfg.cache_dir, add_branch=False)
    # datasets.config.HF_DATASETS_CACHE = Path(cache_dir)

    assert cfg.run is not None, "`run` cannot be `None`"

    project_dir = utils.get_project_dir(cfg, run=cfg.run, subdir=cfg.subfolder)
    logger.add(os.path.join(project_dir, "run.log"))

    set_seed(cfg.hps.seed)
    num_proc = min(mp.cpu_count(), cfg.num_proc)

    accelerator = Accelerator(
        mixed_precision="bf16" if is_bf16_available() else cfg.hps.mixed_precision,
        device_placement=False,
        gradient_accumulation_steps=cfg.hps.gradient_accumulation_steps,
        log_with="tensorboard",
        project_config=ProjectConfiguration(
            project_dir=project_dir,
            automatic_checkpoint_naming=True,
            total_limit=5,
        ),
        # https://pytorch.org/data/main/torchdata.stateful_dataloader.html
        # dataloader_config=DataLoaderConfiguration(use_stateful_dataloader=True),
        # kwargs_handlers=[
        #     DistributedDataParallelKwargs(
        #         static_graph=True, find_unused_parameters=False, bucket_cap_mb=100
        #     )
        # ],
    )

    # ValueError: value should be one of int, float, str, bool, or torch.Tensor
    # accelerator.init_trackers("logs", utils.get_hps_as_dict(cfg))

    device = accelerator.device

    is_main_process = accelerator.is_local_main_process
    progress_bar = (
        accelerator.is_local_main_process if not cfg.disable_progress_bar else False
    )

    utils.log_where(f"Run: {project_dir}", condition=is_main_process)

    task = get_task(accelerator=accelerator, hps=cfg.hps)

    utils.log_where("Load KB and datasets", condition=is_main_process)
    with accelerator.main_process_first():
        data = task.load_data(
            train=cfg.hps.train.split("+"),
            test=cfg.hps.test.split("+"),
            train_on_dev=cfg.hps.train_on_dev,
            train_on_test=cfg.hps.train_on_test,
            kb_entity_based=cfg.hps.kb_entity_based,
            num_proc=num_proc,
        )

    if "name_idx_to_id" in data and is_main_process:
        with open(os.path.join(project_dir, "name_idx_to_id.json"), "w") as fp:
            json.dump(data["name_idx_to_id"], fp)
        with open(os.path.join(project_dir, "name_idx_to_name.json"), "w") as fp:
            json.dump(data["name_idx_to_name"], fp)

    ds = data["ds"]
    kb = data["kb"]

    dl_kwargs = {
        "num_workers": 1,
        "pin_memory": torch.cuda.is_available(),
        "shuffle": True,
        "batch_size": cfg.hps.per_gpu_batch_size_train,
        "collate_fn": task.collator.collate_query,
    }
    train_dl = DataLoader(dataset=ds[Split.TRAIN], **dl_kwargs)
    dl_kwargs.update(
        {
            "shuffle": False,
            "batch_size": cfg.per_gpu_batch_size_predict,
            "collate_fn": task.collator.collate_query,
        }
    )

    dev_dl = DataLoader(dataset=ds[Split.VALIDATION], **dl_kwargs)

    utils.log_where("Initialize model", condition=is_main_process)
    with accelerator.main_process_first():
        model = get_model(
            model_name_or_path=cfg.hps.model,
            multivector=cfg.hps.multivector,
            multilabel=not cfg.hps.kb_entity_based,
            project_size=cfg.hps.project_size,
            mode=cfg.hps.mode,
            metric=cfg.hps.metric,
            share_weights=cfg.hps.share_weights,
            scale_logits=cfg.hps.scale_logits,
            token_topk_train=cfg.hps.token_topk_train,  # only for multivector=True
            smooth_pool=cfg.hps.smooth_pool,  # only for multivector=True
            qe=cfg.hps.qe_train,
            loss_weights={"w": cfg.hps.loss_weight, "w_qe": cfg.hps.loss_weight_qe},
        )

    if model.base_vocab_size != len(task.tokenizer):
        model.resize_token_embeddings(len(task.tokenizer))

    model = load_checkpoint(
        model=model, project_dir=project_dir, is_main_process=is_main_process
    )
    model.to(device)

    index = get_index(
        accelerator=accelerator,
        config=IndexConfig(multivector=cfg.hps.multivector),
        directory=os.path.join(project_dir, "checkpoint", "last", "index"),
    )

    utils.log_where(
        f"index_refresh={cfg.hps.index_refresh}",
        level="debug",
        condition=is_main_process,
    )
    index_scheduler = utils.IndexRefreshScheduler(
        refresh_schedule=str(cfg.hps.index_refresh),
    )

    # https://huggingface.co/docs/accelerate/v1.2.1/en/concept_guides/performance#learning-rates
    optimizer = get_optimizer(
        model=model,
        lr=cfg.hps.lr * accelerator.num_processes,
        lr_qe=cfg.hps.lr_qe * accelerator.num_processes,
        weight_decay=cfg.hps.weight_decay,
        amsgrad=cfg.hps.amsgrad,
    )

    num_gradient_updates = utils.get_num_gradient_updates(
        train_dl=train_dl,
        max_epochs=cfg.hps.max_epochs,
        gradient_accumulation_steps=cfg.hps.gradient_accumulation_steps,
    )

    lr_scheduler = utils.get_lr_scheduler(
        lr_schedule=cfg.hps.lr_schedule,
        optimizer=optimizer,
        warmup_steps=int(num_gradient_updates * cfg.hps.warmup_proportion),
        num_training_steps=num_gradient_updates,
    )

    train_dl, dev_dl, model, optimizer, lr_scheduler = accelerator.prepare(
        train_dl, dev_dl, model, optimizer, lr_scheduler
    )

    kb_slice = kb.select(
        np.array_split(np.arange(len(kb)), accelerator.num_processes)[
            accelerator.local_process_index
        ]
    )

    kb_dl = DataLoader(
        dataset=kb_slice,
        shuffle=False,
        collate_fn=task.collator.collate_kb,
        batch_size=index.config.batch_size,
        num_workers=num_proc,
        pin_memory=torch.cuda.is_available(),
    )

    utils.log_where("Train", condition=is_main_process)
    utils.log_where(
        f"per_gpu_batch_size_train={cfg.hps.per_gpu_batch_size_train}, "
        f"topk_train={cfg.hps.topk_train} => "
        f"max_candidates={cfg.hps.per_gpu_batch_size_train * cfg.hps.topk_train}",
        condition=is_main_process,
    )
    if not index.saved:
        index.build(
            model=accelerator.unwrap_model(model),
            dl=kb_dl,
            progress_bar=progress_bar,
        )
        index.save(progress_bar=progress_bar)
    else:
        index.load(progress_bar=progress_bar)

    # import time
    #
    # start = time.time()
    #
    # task.eval.run(
    #     model=model,
    #     index=index,
    #     dl=dev_dl,
    #     kb=kb,
    #     progress_bar=progress_bar,
    #     step=-1,
    #     name_idx_to_entity_idx=data.get("name_idx_to_entity_idx"),
    # )
    #
    # end = time.time() - start
    #
    # print(end)

    backward_step = 0
    forward_step = 0
    for epoch in range(cfg.hps.max_epochs):
        for batch in (bar := tqdm(train_dl, disable=not progress_bar)):
            # t_mine_start = time.time()
            candidates, labels = mine_negatives(
                batch=batch,
                model=accelerator.unwrap_model(model),
                index=index,
                kb=kb,
                task=task,
                name_idx_to_entity_idx=data.get("name_idx_to_entity_idx"),
            )

            # torch.cuda.synchronize()
            # mining_time = time.time() - t_mine_start

            batch.update(labels)

            forward_kwargs = {
                "query_input_ids": batch["input_ids"],
                "query_attention_mask": batch["attention_mask"],
                "query_subword_mask": batch["query_subword_mask"],
                "candidates_input_ids": candidates["input_ids"],
                "candidates_attention_mask": candidates["attention_mask"],
                "labels": batch["retriever_labels"],
            }

            if cfg.hps.multivector:
                forward_kwargs.update(
                    {
                        "candidates_subword_mask": candidates[
                            "candidates_subword_mask"
                        ],
                        "train": True,
                        "gather_idxs": batch.get("gather_idxs"),
                    }
                )

                if cfg.hps.qe_train:
                    forward_kwargs.update(
                        {
                            "expansion_subword_mask": batch["expansion_subword_mask"],
                            "expansion_input_ids": batch["expansion_input_ids"],
                        }
                    )

            model.train()
            with accelerator.accumulate(model):
                # t0 = time.time()
                out = model(**forward_kwargs)
                # forward_time = time.time() - t0

                # t0 = time.time()
                loss = out["loss"]
                forward_step += 1
                accelerator.backward(loss)
                # torch.cuda.synchronize()
                # backward_time = time.time() - t0

                # t0 = time.time()
                optimizer.step()
                if not accelerator.optimizer_step_was_skipped:
                    lr_scheduler.step()
                # gradient_norms = get_gradients_norm(model)
                optimizer.zero_grad()
                # torch.cuda.synchronize()
                # step_time = time.time() - t0

            # if forward_step % 50 == 0:
            #     utils.log_where(
            #         f"Mining: {mining_time:.3f}s", condition=is_main_process
            #     )
            #     utils.log_where(
            #         f"Mining: {mining_time:.3f}s | Forward: {forward_time:.3f}s | Backward: {backward_time:.3f}s | Step: {step_time:.3f}s",
            #         condition=is_main_process,
            #     )

            if accelerator.sync_gradients:
                backward_step += 1
                display_bar(bar=bar, epoch=epoch, step=backward_step, values=out)
                accelerator.log(out, step=backward_step)

            if index_scheduler.is_time_to_refresh(forward_step) and dev_dl is not None:
                task.evaluate(
                    accelerator=accelerator,
                    model=model,
                    index=index,
                    dev_dl=dev_dl,
                    kb_dl=kb_dl,
                    progress_bar=progress_bar,
                    epoch=epoch,
                    backward_step=backward_step,
                    is_main_process=is_main_process,
                    project_dir=project_dir,
                    kb=kb,
                    name_idx_to_entity_idx=data.get("name_idx_to_entity_idx"),
                )

        if dev_dl is not None:
            task.evaluate(
                accelerator=accelerator,
                model=model,
                index=index,
                dev_dl=dev_dl,
                kb_dl=kb_dl,
                progress_bar=progress_bar,
                epoch=epoch,
                backward_step=backward_step,
                is_main_process=is_main_process,
                project_dir=project_dir,
                kb=kb,
                name_idx_to_entity_idx=data.get("name_idx_to_entity_idx"),
            )

    accelerator.end_training()
    utils.log_where("End training", condition=is_main_process)


if __name__ == "__main__":
    main()
