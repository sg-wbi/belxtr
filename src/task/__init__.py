import dataclasses
import os
from collections.abc import Mapping

import datasets
from accelerate import Accelerator
from belb import BelbConfig, load_config, load_corpus, load_kb
from datasets.formatting.formatting import LazyBatch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from src.index import BaseAcceleratedIndex
from src.model import BaseRetriever, RetrieverTokenizer
from src.task.collator import (
    BaseRetrieverCollator,
    # MultiVectorRetrieverCollator,
    # SingleVectorRetrieverCollator,
    get_collator,
)
from src.task.ds import DatasetTransform
from src.task.evaluate import AcceleratedEvaluator
from src.task.kb import KbTransform
from src.utils import SaveBestCallback, log_where


def load_pairs(
    train: list[str],
    config: BelbConfig,
    test: list[str] | None = None,
    train_on_dev: bool = False,
    train_on_test: bool = False,
    num_proc: int = 1,
    kb_entity_based: bool = False,
):
    config = config if config is not None else load_config()

    if train_on_test and not test:
        raise ValueError("Specify `test` when `train_on_test` is True")

    test_pairs = test or []
    pairs_to_load = set(train) | set(test_pairs)

    corpora: dict = {"train": [], "validation": [], "test": []}
    kb = None

    for name in pairs_to_load:
        pairing = config.name_to_pair[name]

        if kb is None:
            kb = load_kb(
                name=pairing.kb,
                config=config,
                kb_kwargs=pairing.kb_args,
                names=True,
                num_proc=num_proc,
            )

        corpus = load_corpus(
            name=pairing.corpus,
            config=config,
            corpus_kwargs=pairing.corpus_args,
            num_proc=num_proc,
        )

        for split, ds in corpus.items():
            # 1. Specific evaluation target
            if name in test_pairs and split == "test":
                corpora["test"].append(ds)

            # 2. Training data logic
            elif name in train:
                if (
                    split == "train"
                    or (split == "validation" and train_on_dev)
                    or (split == "test" and train_on_test)
                ):
                    corpora["train"].append(ds)
                elif split == "validation":
                    corpora["validation"].append(ds)
                elif split == "test" and not test_pairs:
                    corpora["test"].append(ds)

    # Filter out empty splits and concatenate
    ds_dict = {k: datasets.concatenate_datasets(v) for k, v in corpora.items() if v}
    return {"corpus": datasets.DatasetDict(ds_dict), "kb": kb}


def _batch_add_species_to_label(batch: LazyBatch, column: str, column_fallback: str):
    labels = []
    for i, label in enumerate(batch["label"]):
        species = batch[column][i]
        if species is None:
            species = batch[column_fallback][i]

        if species is not None:
            species = species if isinstance(species, str) else ",".join(species)
            label = f"{label} [{species}]"

        labels.append(label)

    batch["label"] = labels

    return batch


@dataclasses.dataclass
class Task:
    tokenizer: PreTrainedTokenizerBase
    collator: BaseRetrieverCollator
    kbt: KbTransform
    dst: DatasetTransform
    eval: AcceleratedEvaluator

    def __post_init__(self):
        self.save_callback = SaveBestCallback()

    def load_data(
        self,
        train: list[str],
        test: list[str] | None,
        train_on_dev: bool = False,
        train_on_test: bool = False,
        kb_entity_based: bool = True,
        config_dir: str | None = None,
        num_proc: int = 1,
    ):
        is_cellosaurus = "cellosaurus" in train[0]
        is_ncbi_gene = "ncbi-gene" in train[0]

        config = load_config(config_dir=config_dir)

        data = load_pairs(
            config=config,
            train=train,
            test=test,
            train_on_dev=train_on_dev,
            train_on_test=train_on_test,
            num_proc=num_proc,
            kb_entity_based=kb_entity_based,
        )

        corpus = data["corpus"]
        kb = data["kb"]

        entities = kb["entities"]

        id_to_idx = dict(zip(entities["id"], entities["idx"]))

        out: dict = {"raw": {"kb": entities, "ds": corpus}}

        names = None
        if not kb_entity_based:
            names = kb["names"]
            out["raw"]["names"] = names
            name_idx_to_entity_idx = dict(zip(names["idx"], names["entity_idx"]))
            out["name_idx_to_entity_idx"] = name_idx_to_entity_idx

            name_idx_to_id = dict(zip(names["idx"], names["id"]))
            out["name_idx_to_id"] = name_idx_to_id

            name_idx_to_name = dict(zip(names["idx"], names["name"]))
            out["name_idx_to_name"] = name_idx_to_name
        else:
            if is_ncbi_gene or is_cellosaurus:
                if is_ncbi_gene:
                    fn_kwargs = {
                        "column": "taxonomy_common_name",
                        "column_fallback": "taxonomy_scientific_name",
                    }
                else:
                    fn_kwargs = {
                        "column": "taxonomy_common_names",
                        "column_fallback": "taxonomy_scientific_names",
                    }
                entities = entities.map(
                    _batch_add_species_to_label,
                    fn_kwargs=fn_kwargs,
                    batched=True,
                    num_proc=num_proc,
                    desc="KB (entity-based): add species to label",
                )

        out["kb"] = self.kbt.apply(
            ds=entities if kb_entity_based else names,
            entity_based=kb_entity_based,
            num_proc=num_proc,
        )

        out["ds"] = self.dst.apply(
            ds=out["raw"]["ds"],
            kb=out["kb"],
            qe=getattr(self.collator, "qe", False),
            id_to_idx=id_to_idx,
            num_proc=num_proc,
        )

        return out

    def evaluate(
        self,
        kb: datasets.Dataset,
        dev_dl: DataLoader,
        project_dir: str,
        accelerator: Accelerator,
        model: BaseRetriever,
        kb_dl: DataLoader,
        progress_bar: bool,
        epoch: int,
        backward_step: int,
        is_main_process: bool,
        index: BaseAcceleratedIndex,
        name_idx_to_entity_idx: dict[int, int] | None = None,
    ):
        log_where(
            f"Epoch:{epoch} - step:{backward_step} - refresh index & evaluate",
            condition=is_main_process,
        )
        index.build(
            model=accelerator.unwrap_model(model),
            dl=kb_dl,
            progress_bar=progress_bar,
        )
        index.save(
            directory=os.path.join(project_dir, "checkpoint", "last", "index"),
            progress_bar=progress_bar,
        )

        accelerator.save_model(
            model=model,
            save_directory=os.path.join(project_dir, "checkpoint", "last"),
            safe_serialization=False,
        )
        metrics = self.eval.run(
            model=model,
            index=index,
            dl=dev_dl,
            kb=kb,
            progress_bar=progress_bar,
            step=backward_step,
            name_idx_to_entity_idx=name_idx_to_entity_idx,
        )
        if accelerator.is_local_main_process:
            self.save_callback.save_metrics(
                path=os.path.join(project_dir, "metrics.jsonl"),
                metrics=metrics,
                epoch=epoch,
                step=backward_step,
            )

        if self.save_callback.save(
            metric=metrics[self.eval.main_metric],
            progress_bar=progress_bar,
        ):
            accelerator.save_model(
                model=model,
                save_directory=os.path.join(project_dir, "checkpoint", "best"),
                safe_serialization=False,
            )
            index.save(
                directory=os.path.join(project_dir, "checkpoint", "best", "index"),
                progress_bar=progress_bar,
            )


def _get_collator_kwargs(hps: Mapping):
    kwargs = {
        "context_size": hps["context_size"],
        "topk_train": hps["topk_train"],
        "multilabel": not hps["kb_entity_based"],
    }
    if hps["multivector"]:
        kwargs.update(
            {
                "query_cls": hps["query_cls"],
                "query_prefix": hps["query_prefix"],
                "query_punctuation": hps["query_punctuation"],
                "candidate_aliases": hps["candidate_aliases"],
                "candidate_cls": hps["candidate_cls"],
                "candidate_prefix": hps["candidate_prefix"],
                "candidate_punctuation": hps["candidate_punctuation"],
                "qe": hps["qe"],
                "qe_train": hps["qe_train"],
                "qe_attend_mask": hps["qe_attend_mask"],
                "qe_negative_mining": hps["qe_negative_mining"],
            }
        )

    return kwargs


def get_task(
    accelerator: Accelerator,
    hps: Mapping,
    allow_missing: bool = False,
    topk_predict: int = 100,
) -> Task:
    tokenizer = RetrieverTokenizer.from_pretrained(hps["model"])

    collator = get_collator(
        tokenizer=tokenizer,
        multivector=hps["multivector"],
        **_get_collator_kwargs(hps),
    )

    kbt = KbTransform(tokenizer=tokenizer, candidate_size=hps["candidate_size"])

    dst = DatasetTransform(
        tokenizer=tokenizer, qe_sim_threshold=hps["qe_sim_threshold"]
    )

    evaluator = AcceleratedEvaluator(
        accelerator=accelerator,
        collator=collator,
        topk_predict=topk_predict,
        token_topk_predict=hps["token_topk_predict"],
    )

    task = Task(
        tokenizer=tokenizer, collator=collator, kbt=kbt, dst=dst, eval=evaluator
    )

    return task
