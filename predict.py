import argparse
import gzip
import json
import multiprocessing as mp
import os

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import is_bf16_available
from belb import CORPUS_FEATURES, load_config, load_corpus
from belb.corpus.parsing import parse_pubtator_example
from bioc import pubtator
from datasets import Dataset, DatasetDict, Features
from datasets.formatting.formatting import LazyRow
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.index import IndexConfig, get_index
from src.model import get_model, load_checkpoint
from src.task import get_task


def parse_args() -> argparse.Namespace:
    """CLI"""
    parser = argparse.ArgumentParser(description="predict")
    parser.add_argument(
        "--models_dir",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=("belb", "biored"),
        type=str,
        help="benchmark",
    )

    return parser.parse_args()


def _filter_annotations(row: LazyRow, entity_type: str):
    row["annotations"] = [
        a for a in row["annotations"] if a["type"].upper() == entity_type.upper()
    ]
    return row


def _stream_biored_aioner(path: str) -> dict:
    with open(path) as fp:
        docs = pubtator.load(fp)
        for doc in docs:
            # apparently if `id` is missing, `annotations` are parsed as `relations`
            for r in doc.relations:
                a = pubtator.PubTatorAnn(
                    pmid=r.pmid,
                    start=r.type,
                    end=r.id1,
                    type=r.neg,
                    text=r.id2,
                    id="-1",
                )

                if a.type.lower() == "cellline":
                    a.type = "Cell-line"
                doc.add_annotation(a)
            yield parse_pubtator_example(doc)


def load_biored_aioner(path: str, entity_type: str) -> DatasetDict:
    ds = Dataset.from_generator(
        _stream_biored_aioner,
        gen_kwargs={"path": path},
        features=Features(CORPUS_FEATURES),
    )

    ds = ds.map(
        _filter_annotations,
        fn_kwargs={"entity_type": entity_type},
        desc=f"Filter annotations by entity type `{entity_type}`",
    )

    ds = ds.filter(
        lambda e: len(e["annotations"]) > 0, desc="Remove examples w/o annotations"
    )

    return DatasetDict({"test": ds})


def get_biored_pair_to_run(models_dir: str):
    return {
        "biored-disease.ctd-diseases": os.path.join(models_dir, "disease"),
        "biored-chemical.ctd-chemicals": os.path.join(models_dir, "chemical"),
        "biored-species.ncbi-taxonomy": os.path.join(models_dir, "species"),
        "biored-gene.ncbi-gene": os.path.join(models_dir, "gene"),
        "biored-cell-line.cellosaurus": os.path.join(models_dir, "cell-line"),
    }


def get_belb_pair_to_run(models_dir: str):
    return {
        "bc5cdr-disease.ctd-diseases": os.path.join(
            models_dir, "bc5cdr-disease.ctd-diseases-bs4-top32-me5"
        ),
        "bc5cdr-chemical.ctd-chemicals": os.path.join(
            models_dir, "bc5cdr-chemical.ctd-chemicals-bs4-top32-me5"
        ),
        "nlm-chem.ctd-chemicals": os.path.join(
            models_dir, "nlm-chem.ctd-chemicals-bs4-top32-me5"
        ),
        "linnaeus.ncbi-taxonomy": os.path.join(
            models_dir, "linnaeus.ncbi-taxonomy-bs4-top32-me5"
        ),
        "s800.ncbi-taxonomy": os.path.join(
            models_dir, "s800.ncbi-taxonomy-bs4-top32-me5"
        ),
        "gnormplus.ncbi-gene-gnormplus": os.path.join(
            models_dir, "gnormplus.ncbi-gene-gnormplus-bs4-top32-me5"
        ),
        "nlm-gene.ncbi-gene-nlm-gene": os.path.join(
            models_dir, "nlm-gene.ncbi-gene-nlm-gene-bs4-top32-me5"
        ),
        "bioid-cell-line.cellosaurus": os.path.join(
            models_dir, "bioid-cell-line.cellosaurus-bs4-top32-me5"
        ),
    }


def main(args: argparse.Namespace):
    num_proc = min(mp.cpu_count(), 40)

    MODELS_DIR = args.models_dir

    accelerator = Accelerator(
        mixed_precision="bf16" if is_bf16_available() else "fp16",
        device_placement=False,
    )

    device = accelerator.device

    belb_config = load_config()

    if args.target == "belb":
        output_dir = "./data/belb/candidates/"
        pair_to_run = get_belb_pair_to_run(os.path.join(MODELS_DIR, "belb"))
        checkpoint = "best"
    elif args.target == "biored":
        output_dir = "./data/biored/output"
        pair_to_run = get_biored_pair_to_run(os.path.join(MODELS_DIR, "biored"))
        checkpoint = "best"

    os.makedirs(output_dir, exist_ok=True)

    for pair_name, run_dir in pair_to_run.items():
        if args.target == "belb":
            pairing = belb_config.name_to_pair[pair_name]
            corpus = load_corpus(
                name=pairing.corpus,
                config=belb_config,
                corpus_kwargs=pairing.corpus_args,
            )
            corpus.pop("train")
            corpus.pop("validation")
        else:
            entity_type = pair_name.split(".")[0].split("-", maxsplit=1)[1]
            corpus = load_biored_aioner(
                path="./data/biored/input/test-aioner-biomedbertcfr.pubtator",
                entity_type=entity_type,
            )

        with open(os.path.join(run_dir, "name_idx_to_id.json")) as fp:
            name_idx_to_id = {int(k): v for k, v in json.load(fp).items()}

        with open(os.path.join(run_dir, "name_idx_to_name.json")) as fp:
            name_idx_to_name = {int(k): v for k, v in json.load(fp).items()}

        cfg = OmegaConf.load(os.path.join(run_dir, "conf.yaml"))

        task = get_task(accelerator=accelerator, hps=cfg.hps)

        ds = task.dst.apply(
            ds=corpus,
            qe=cfg.hps.qe,
            names=list(name_idx_to_name.values()),
            num_proc=num_proc,
        )

        dl_kwargs = {
            "num_workers": 1,
            "pin_memory": torch.cuda.is_available(),
            "shuffle": False,
            "batch_size": cfg.hps.per_gpu_batch_size_train,
            "collate_fn": task.collator.collate_query,
        }
        test_dl = DataLoader(dataset=ds["test"], **dl_kwargs)

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
        )

        if model.base_vocab_size != len(task.tokenizer):
            model.resize_token_embeddings(len(task.tokenizer))

        model = load_checkpoint(
            model=model,
            project_dir=run_dir,
            checkpoint=checkpoint,
            is_main_process=True,
        )
        model.to(device)

        index = get_index(
            accelerator=accelerator,
            config=IndexConfig(multivector=cfg.hps.multivector),
            directory=os.path.join(run_dir, "checkpoint", "best", "index"),
        )

        index.load(progress_bar=True)

        pred: dict = {}
        for batch in tqdm(test_dl, desc="Predict"):
            model_kwargs = {
                "query_input_ids": batch["input_ids"],
                "query_attention_mask": batch["attention_mask"],
                "query_subword_mask": batch["query_subword_mask"],
            }

            with torch.no_grad():
                query = model(**model_kwargs)

            search_kwargs = {
                "queries": query["query_embedding"],
                "topk": task.eval.topk_predict,
                "lengths": query["query_subword_mask"].sum(-1),
                "token_topk": task.eval.token_topk_predict,
            }

            retrieval = index.search(**search_kwargs)

            y_pred = retrieval["candidates_ids"]

            y_scores = retrieval["candidates_scores"].tolist()

            y_names = np.vectorize(name_idx_to_name.get)(y_pred).tolist()

            y_pred = np.vectorize(name_idx_to_id.get)(y_pred).tolist()

            aids = batch["aid"]

            for i in range(len(aids)):
                aid = aids[i]

                pred[aid] = {}
                pred[aid]["gold"] = batch["ids"][i]
                pred[aid]["candidates"] = []
                for p, n, s in zip(y_pred[i], y_names[i], y_scores[i]):
                    pred[aid]["candidates"].append({"id": p, "name": n, "score": s})

        if args.target == "belb":
            filename = pair_name.split(".")[0]
        else:
            filename = entity_type

        with gzip.open(os.path.join(output_dir, f"{filename}.json.gz"), "w") as fp:
            fp.write(json.dumps(pred).encode("utf-8"))


if __name__ == "__main__":
    main(parse_args())
