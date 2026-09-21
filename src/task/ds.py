from __future__ import annotations

import dataclasses
import os

import bm25s
import datasets
import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import rapidfuzz
from datasets.fingerprint import Hasher
from datasets.formatting.formatting import LazyBatch, LazyRow
from loguru import logger
from tqdm import tqdm

from src.model import RetrieverTokenizer
from src.utils.preprocess import STOPWORDS


def get_char_ngrams(texts, n=3, separator=" "):
    """
    Takes a list of strings and returns a list of strings, where each
    string contains its n-grams separated by a specific separator.
    """
    if isinstance(texts, str):
        texts = [texts]

    processed_texts = []
    for text in texts:
        ngrams = []

        if len(text) < n:
            ngrams.append(text)
        else:
            for i in range(len(text) - n + 1):
                ngrams.append(text[i : i + n])

        # Join n-grams back into a single string separated by the chosen separator
        processed_texts.append(separator.join(ngrams))

    return processed_texts


def tokenize(texts: list[str], ngrams: int = 3):
    return bm25s.tokenize(
        get_char_ngrams([rapidfuzz.utils.default_process(t) for t in texts], n=ngrams),
        token_pattern=r"\S+",
        stopwords=False,
    )


def _arrow_explode_rows(
    ds: datasets.Dataset,
    columns_to_keep=["id", "text"],
    ann_fields_to_keep=["start", "end", "ids"],
):
    table = ds.data.table
    ann_col = table["annotations"].combine_chunks()

    # 1. Map parent rows to flattened indices
    parent_indices = pc.list_parent_indices(ann_col)

    # 2. Flatten annotations and extract specific fields
    flat_ann_structs = pc.list_flatten(ann_col)

    new_data = {}

    # 3. Add broadcasted global columns (drop everything else by omission)
    for col in columns_to_keep:
        new_data[col] = pc.take(table[col].combine_chunks(), parent_indices)

    # 4. Add flattened annotation fields (using original names)
    for field_name in ann_fields_to_keep:
        new_data[field_name] = flat_ann_structs.field(field_name)

    new_ds = datasets.Dataset(pa.table(new_data))
    return new_ds


def _batch_map_to_index(batch: LazyBatch, id_to_idx: dict[str, int]) -> LazyBatch:
    batch_ids, batch_idxs = [], []
    for ids in batch["ids"]:
        i = sorted(ids)[0]
        batch_ids.append(i)
        batch_idxs.append(id_to_idx[i])

    batch["id"] = batch_ids
    batch["idx"] = batch_idxs
    return batch


def _duckdb_left_join(
    ds: datasets.Dataset,
    kb: datasets.Dataset,
) -> datasets.Dataset:
    conn = duckdb.connect()
    conn.register("kb", kb.data.table)

    preserve_order = ds.split == datasets.Split.TEST
    if preserve_order:
        ds_with_index = ds.add_column("__original_index", list(range(len(ds))))
        conn.register("ds", ds_with_index.data.table)
    else:
        conn.register("ds", ds.data.table)

    cols = ["ds.*", "kb.label as label", "kb.entity as entity"]
    if "taxonomy_common_name" in kb.column_names:
        cols.append("kb.taxonomy_common_name as taxonomy_common_name")
    if "taxonomy_scientific_name" in kb.column_names:
        cols.append("kb.taxonomy_scientific_name as taxonomy_scientific_name")

    # cols = [
    #     "ds.*",
    #     "kb.label_input_ids as expansion_input_ids",
    #     "kb.label_idxs as expansion_idxs",
    # ]

    query = f"""
        SELECT 
            {", ".join(cols)}
        FROM ds 
        LEFT JOIN kb ON ds.id = kb.id
    """
    if preserve_order:
        query += "\nORDER BY ds.__original_index"

    result = conn.execute(query).to_arrow_table()

    jds = datasets.Dataset(result)

    if preserve_order:
        jds = jds.remove_columns(["__original_index"])
    return jds


def _preprocess_name(name: str) -> str:
    return "".join(sorted(rapidfuzz.utils.default_process(name).split()))


def _rm_stopwords(name: str):
    return " ".join([s for s in name.split() if s.lower() not in STOPWORDS])


def _batch_add_mention(batch: LazyBatch):
    mentions = []
    for start, end, text in zip(batch["start"], batch["end"], batch["text"]):
        mentions.append(text[start:end])

    batch["mention"] = mentions

    return batch


def _row_add_expansion_train(
    row: LazyRow, name_lookup: dict, idx_lookup: dict
) -> LazyRow:
    scored = rapidfuzz.process.extract(
        row["mention"],
        name_lookup[row["id"]],
        scorer=rapidfuzz.fuzz.partial_ratio,
        processor=_preprocess_name,
    )

    qe = set()
    qe_idx = []
    for n in scored[:2]:
        qe.add(_rm_stopwords(n[0]))
        qe_idx.append(idx_lookup[n[0]])

    row["qe"] = qe
    row["qe_idx"] = qe_idx

    return row


def _batch_add_expansion_test(batch: LazyBatch, retriever):
    query_tokens = tokenize(batch["mention"])
    batch_results, _ = retriever.retrieve(query_tokens, k=5)

    expansion = []
    for b in batch_results:
        qe = set()
        for r in b[:2]:
            qe.add(_rm_stopwords(r["text"]))
        expansion.append(qe)

    batch["qe"] = expansion

    return batch


def _batch_build_input_ids(
    batch: LazyBatch,
    tokenizer: RetrieverTokenizer,
):
    tokenizer_kwargs = {
        "is_split_into_words": False,
        "return_attention_mask": False,
        "return_token_type_ids": False,
        "return_offsets_mapping": True,
        "add_special_tokens": False,
    }
    query_start_token = tokenizer.query_start_token
    query_end_token = tokenizer.query_end_token
    # space_token_id = tokenizer.space_token_id

    queries, mentions = [], []
    for q, start, end in zip(
        batch["text"],
        batch["start"],
        batch["end"],
    ):
        mention = q[start:end]
        mentions.append(mention)

        query = (
            q[:start]
            + f" {query_start_token} "
            + mention
            + f" {query_end_token} "
            + q[end:]
        )
        queries.append(query)

    tokenizer_out = tokenizer(queries, **tokenizer_kwargs)

    batch["input_ids"] = tokenizer_out.input_ids

    mentions_input_ids = tokenizer(mentions, **tokenizer_kwargs).input_ids
    qe_input_ids = []

    punctuation_ids = tokenizer.punctuation_ids
    if "qe" in batch:
        for i, qe in enumerate(batch["qe"]):
            qids = [
                i
                for sublist in tokenizer(qe, **tokenizer_kwargs).input_ids
                for i in sublist
            ]
            mids = mentions_input_ids[i]
            qids = set(i for i in qids if i not in mids)
            qids = [i for i in qids if i not in punctuation_ids]
            qe_input_ids.append(qids)

        batch["qe_input_ids"] = qe_input_ids
        # batch["qe_num_tokens"] = qe_num_tokens

    return batch


@dataclasses.dataclass
class DatasetTransform:
    tokenizer: RetrieverTokenizer
    qe_sim_threshold: float = 0.9

    def get_fingerprint(self, ds: datasets.DatasetDict, extra: dict | None = None):
        hasher = Hasher()
        hasher.update(
            {name: split._fingerprint for name, split in ds.items()} | extra
            if extra is not None
            else {}
        )
        return hasher.hexdigest()

    def _model_agnostic_preprocess(
        self,
        ds: datasets.DatasetDict,
        kb: datasets.Dataset | None = None,
        qe_train: bool = True,
        id_to_idx: dict | None = None,
        names: list[str] | None = None,
        num_proc: int = 1,
    ):
        logger.debug("Dataset: convert annotations to queries (explode rows)")

        split_name = str(next(iter(ds)))
        base_dir = os.path.dirname(ds[split_name].cache_files[0]["filename"])

        extra = {
            "columns_to_keep": ["id", "text"],
            "ann_fields_to_keep": ["start", "end", "ids"],
        }
        cache_dir = os.path.join(base_dir, self.get_fingerprint(ds=ds, extra=extra))

        if os.path.exists(cache_dir):
            ds = datasets.load_from_disk(cache_dir)
        else:
            # explode document with mentions into queries (one mention per document)
            ds = datasets.DatasetDict(
                {
                    str(name): _arrow_explode_rows(ds=ds, **extra)
                    for name, ds in tqdm(ds.items(), desc="Dataset: explode rows")
                }
            )
            ds.save_to_disk(cache_dir)

        if id_to_idx is not None:
            logger.debug("Dataset: map annotations ids to KB index")
            ds = ds.map(
                _batch_map_to_index,
                fn_kwargs={"id_to_idx": id_to_idx},
                desc="Dataset: `id`->`idx`",
                batched=True,
                remove_columns=["ids"],
            )

        if qe_train:
            ds = ds.map(
                _batch_add_mention, desc="Dataset: extract mention", batched=True
            )

            if "train" in ds:
                ids = set(ds["train"]["id"])
                subset = kb.filter(
                    lambda row: row["id"] in ids,
                    num_proc=num_proc,
                    desc="Dataset: collect names for query expansion",
                )
                name_lookup = (
                    subset.to_pandas().groupby("id")["name"].apply(list).to_dict()
                )
                idx_lookup = dict(zip(subset["name"], subset["idx"]))
                ds["train"] = ds["train"].map(
                    _row_add_expansion_train,
                    fn_kwargs={"name_lookup": name_lookup, "idx_lookup": idx_lookup},
                    desc="Dataset: add query expansion (train)",
                    batched=False,
                    num_proc=1,
                )

            names = kb["name"] if kb is not None else names
            metadata = [{"text": n} for n in names]
            retriever = bm25s.BM25(corpus=metadata)

            corpus_tokens = tokenize(texts=[m["text"] for m in metadata])
            retriever.index(corpus_tokens)

            for split in ["validation", "test"]:
                if split not in ds:
                    continue
                ds[split] = ds[split].map(
                    _batch_add_expansion_test,
                    fn_kwargs={"retriever": retriever},
                    desc="Dataset: add query expansion (test)",
                    batched=True,
                    remove_columns="mention",
                    num_proc=1,
                )

        return ds

    def apply(
        self,
        ds: datasets.DatasetDict,
        kb: datasets.Dataset | None = None,
        qe: bool = True,
        id_to_idx: dict | None = None,
        names: list[str] | None = None,
        num_proc: int = 1,
    ) -> datasets.DatasetDict:
        ds = self._model_agnostic_preprocess(
            ds=ds,
            kb=kb,
            qe_train=qe,
            id_to_idx=id_to_idx,
            names=names,
            num_proc=num_proc,
        )

        split_name = str(next(iter(ds)))
        features = ds[split_name].info.features.copy()
        features["input_ids"] = datasets.Sequence(datasets.Value("int32"))

        if qe:
            features["qe_input_ids"] = datasets.Sequence(datasets.Value("int32"))
            # features["qe_num_tokens"] = datasets.Value("int32")

        ds = ds.map(
            _batch_build_input_ids,
            fn_kwargs={
                "tokenizer": self.tokenizer,
            },
            desc="Dataset: subword tokenize",
            features=datasets.Features(features),
            batched=True,
            num_proc=num_proc,
        )

        return ds
