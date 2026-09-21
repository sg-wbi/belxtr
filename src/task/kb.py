import dataclasses
from collections import defaultdict

import datasets
from datasets.formatting.formatting import LazyBatch, LazyRow
from loguru import logger

from src.model import RetrieverTokenizer

SEPARATOR_TOKEN = ","


def _batch_entity_subword_tokenize(
    batch: LazyBatch,
    tokenizer: RetrieverTokenizer,
):
    kwargs = {
        "is_split_into_words": False,
        "truncation": False,
        "add_special_tokens": False,
        "return_attention_mask": False,
        "return_offsets_mapping": False,
    }

    out = {}
    out["label_input_ids"] = tokenizer(batch["label"], **kwargs).input_ids

    out["aliases_input_ids"] = [
        tokenizer(aliases, **kwargs).input_ids if aliases is not None else []
        for aliases in batch["aliases"]
    ]

    if "species" in batch:
        batch["species_input_ids"] = tokenizer(batch["species"], **kwargs)

    return out


def _batch_name_subword_tokenize(
    batch: LazyBatch,
    tokenizer: RetrieverTokenizer,
    candidate_size: int,
):
    kwargs = {
        "is_split_into_words": False,
        "truncation": True,
        "add_special_tokens": True,
        "return_attention_mask": False,
        "return_offsets_mapping": False,
        "max_length": candidate_size,
    }

    batch["input_ids"] = tokenizer(batch["name"], **kwargs).input_ids
    return batch


@dataclasses.dataclass
class EntityBuilder:
    column_to_prefix: dict[str, int]
    separator_token_id: int
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    candidate_start_token_id: int | None = None
    candidate_size: int = 128
    tolerance: int = 5

    def __post_init__(self):
        self.budget = self.candidate_size
        self.allocated = 0
        self.data = defaultdict(list)
        self.columns = ["label"] + list(self.column_to_prefix)

    def add_prefix(self):
        for t in [self.bos_token_id, self.candidate_start_token_id]:
            if t is not None:
                self.allocated += 1
                self.data["input_ids"].append(t)
                for c in self.columns:
                    self.data[f"{c}_idxs"].append(0)

    def add_label(self, row: LazyRow):
        label_ids = row["label_input_ids"]
        self.data["input_ids"].extend(label_ids)
        title_len = len(label_ids)
        for c in self.columns:
            values = [1 if c == "label" else 0] * title_len
            self.data[f"{c}_idxs"].extend(values)

        self.allocated += title_len

    def finalize(self):
        if self.eos_token_id is not None:
            self.data["input_ids"].append(self.eos_token_id)
            for c in self.columns:
                self.data[f"{c}_idxs"].append(0)

        for c in self.columns:
            key = f"{c}_idxs"
            assert len(self.data[key]) == len(self.data["input_ids"])
            self.data[key] = [idx for idx, v in enumerate(self.data[key]) if v != 0]

        if "entity" not in self.data:
            self.data["entity"] = []

        return dict(self.data)

    def add_column(
        self,
        column: str,
        input_ids: list[int],
        text: str,
        is_first: bool,
    ):
        # Append tokens
        toadd_len = len(input_ids)
        self.data["input_ids"].extend(input_ids)
        self.data["entity"].append(text)
        for c in self.columns:
            value = int(c == column)
            if is_first:
                self.data[f"{c}_idxs"].extend([0] + [value] * (toadd_len - 1))
            else:
                self.data[f"{c}_idxs"].extend([value] * toadd_len)

        self.allocated += toadd_len

    def add_separator(self, column: str):
        self.data["input_ids"].append(self.separator_token_id)

        for c in self.columns:
            value = int(c == column)
            self.data[f"{c}_idxs"].append(value)

        self.allocated += 1

    def build(self, row: LazyRow):
        self.add_prefix()
        self.add_label(row)

        for column, prefix in self.column_to_prefix.items():
            ids = row[f"{column}_input_ids"]
            if not ids:
                continue

            is_multi = len(ids) > 1

            for idx, v in enumerate(ids):
                is_first = idx == 0
                is_last = idx == len(ids) - 1

                input_ids = [prefix] + v if is_first else v

                # Budget check
                if self.allocated + len(input_ids) > self.budget + self.tolerance:
                    return self.finalize()

                self.add_column(
                    column=column,
                    input_ids=input_ids,
                    text=row[column][idx],
                    is_first=is_first,
                )

                # Separator if needed
                if is_multi and not is_last:
                    self.add_separator(column=column)

        return self.finalize()


def _row_build_entity(
    row: LazyRow,
    column_to_prefix: dict[str, int],
    separator_token_id: int,
    bos_token_id: int | None = None,
    eos_token_id: int | None = None,
    candidate_start_token_id: int | None = None,
    candidate_size: int = 128,
    tolerance: int = 5,
    # expansion_columns: list[str] = ["title", "description"],
):
    builder = EntityBuilder(
        column_to_prefix=column_to_prefix,
        separator_token_id=separator_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        candidate_start_token_id=candidate_start_token_id,
        candidate_size=candidate_size,
        tolerance=tolerance,
        # expansion_columns=expansion_columns,
    )

    return builder.build(row)


def _batch_add_species_to_label(batch: LazyBatch):
    labels = []
    for i, input_ids in enumerate(batch["label_input_ids"]):
        species_input_ids = batch["species_input_ids"][i]
        labels.append(input_ids + species_input_ids)

    batch["label_input_ids"] = labels

    return batch


@dataclasses.dataclass
class KbTransform:
    tokenizer: RetrieverTokenizer
    candidate_size: int = 128
    tolerance: int = 5

    def __post_init__(self):
        self.separator_token_id = self.tokenizer.vocab[SEPARATOR_TOKEN]
        self.column_to_prefix = {
            "aliases": self.tokenizer.vocab[
                self.tokenizer.available_special_tokens.pop()
            ]
        }

    def apply(
        self,
        ds: datasets.Dataset,
        entity_based: bool = True,
        num_proc: int = 1,
    ):
        if entity_based:
            features = {
                "idx": datasets.Value("int64"),
                "id": datasets.Value("string"),
                "label": datasets.Value("string"),
                "aliases": datasets.Sequence(datasets.Value("string")),
            }
            if "species" in ds.column_names:
                features["species"] = datasets.Value("string")

            remove_columns = [c for c in ds.column_names if c not in features]
            ds = ds.remove_columns(remove_columns)

            features["label_input_ids"] = datasets.Sequence(datasets.Value("int32"))
            features["aliases_input_ids"] = datasets.Sequence(
                datasets.Sequence(datasets.Value("int32"))
            )

            if "species" in ds.column_names:
                features["species_input_ids"] = datasets.Sequence(
                    datasets.Value("int32")
                )

            logger.debug("KB: subword tokenize")
            ds = ds.map(
                _batch_entity_subword_tokenize,
                fn_kwargs={"tokenizer": self.tokenizer},
                desc="KB: subword tokenize",
                batched=True,
                num_proc=num_proc,
                features=datasets.Features(features),
            )

            if "species" in ds.column_names:
                ds = ds.map(
                    _batch_add_species_to_label,
                    desc="KB: add species to label",
                    batched=True,
                    num_proc=num_proc,
                )

            logger.debug("KB: build entities")
            features.update(
                {
                    "entity": datasets.Sequence(datasets.Value("string")),
                    "input_ids": datasets.Sequence(datasets.Value("int32")),
                    "label_idxs": datasets.Sequence(datasets.Value("int32")),
                    "aliases_idxs": datasets.Sequence(datasets.Value("int32")),
                }
            )
            ds = ds.map(
                _row_build_entity,
                fn_kwargs={
                    "column_to_prefix": self.column_to_prefix,
                    "bos_token_id": self.tokenizer.bos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "candidate_start_token_id": self.tokenizer.candidate_start_token_id,
                    "separator_token_id": self.separator_token_id,
                    "candidate_size": self.candidate_size,
                    "tolerance": self.tolerance,
                },
                desc=f"KB: build entities: truncate to {self.candidate_size} tokens",
                num_proc=num_proc,
                features=datasets.Features(features),
            )
        else:
            features = {
                "idx": datasets.Value("int64"),
                "id": datasets.Value("string"),
                "name": datasets.Value("string"),
            }
            if "species" in ds.column_names:
                features["species"] = datasets.Value("string")

            remove_columns = [c for c in ds.column_names if c not in features]
            ds = ds.remove_columns(remove_columns)

            features["input_ids"] = datasets.Sequence(datasets.Value("int32"))
            logger.debug("KB: subword tokenize")
            ds = ds.map(
                _batch_name_subword_tokenize,
                fn_kwargs={
                    "tokenizer": self.tokenizer,
                    "candidate_size": self.candidate_size,
                },
                desc="KB: subword tokenize",
                batched=True,
                num_proc=num_proc,
                features=datasets.Features(features),
            )

        return ds
