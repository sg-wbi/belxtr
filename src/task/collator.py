from __future__ import annotations

import dataclasses
import itertools
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch

from src.model import RetrieverTokenizer


def split_on_token(lst, token):
    return [
        list(group)
        for k, group in itertools.groupby(lst, lambda x: x == token)
        if not k
    ]


def row_wise_isin(src: np.ndarray, trg: np.ndarray) -> np.ndarray:
    """
    Given two 2d arryas compute row-wise isin
    See comment here:
    https://stackoverflow.com/questions/67870579/rowwise-numpy-isin-for-2d-arrays

    >>> y_true = np.asarray([ [11457], [8740], [2779] ])
    >>> y_pred = np.asarray([ [6791, 8742], [8735, 5054], [ 299, 2779] ])
    >>> row_wise_isin(y_pred, y_true)
    array([[False, False],
           [False, False],
           [False,  True]])
    """
    return (src[:, :, None] == trg[:, None, :]).any(-1)


@dataclasses.dataclass
class BaseRetrieverCollator(ABC):
    tokenizer: RetrieverTokenizer
    context_size: int = -1
    topk_train: int = 32
    multilabel: bool = False

    def __post_init__(self):
        self.pad_token_type_id = getattr(self.tokenizer, "pad_token_type_id")
        assert self.pad_token_type_id is not None
        self.pad_kwargs = {
            "padding": True,
            "max_length": None,
            "pad_to_multiple_of": None,
            "return_tensors": "np",
        }

        self.input_ids_budget = (
            self.tokenizer.model_max_length
            - int(self.tokenizer.bos_token_id is not None)
            - int(self.tokenizer.eos_token_id is not None)
        )

        self.entity_idx_to_name_idxs = {}

    @abstractmethod
    def collate_query(
        self,
        features: list[dict[str, Any]],
        train: bool = False,
    ) -> dict[str, Any]:
        pass

    @abstractmethod
    def collate_kb(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        pass

    def _entity_build_labels(
        self,
        gold_idxs: np.ndarray,
        candidates_idxs: np.ndarray,
    ):
        retriever_candidates_list = []
        for i, _id in enumerate(gold_idxs):
            positive = [_id]
            hard_negatives = candidates_idxs[i][candidates_idxs[i] != _id].tolist()

            candidates = positive + hard_negatives
            retriever_candidates_list.append(candidates[: self.topk_train])
            retriever_candidates = np.asarray(retriever_candidates_list)

        candidates_pool = set(np.unique(retriever_candidates))
        cid_to_cidx = {int(_id): idx for idx, _id in enumerate(sorted(candidates_pool))}
        retriever_labels = np.zeros((len(gold_idxs), len(cid_to_cidx)), dtype=int)
        for i, _id in enumerate(gold_idxs):
            retriever_labels[i, cid_to_cidx[_id]] = 1

        return {
            "idxs": list(cid_to_cidx),
            "retriever_labels": torch.as_tensor(retriever_labels),
        }

    def _inject_expansion(self, candidates_idxs: np.ndarray, qe_idxs: np.ndarray):
        # Create a copy so we don't mutate the original candidate matrix
        result = candidates_idxs.copy()

        # Process row by row
        for i in range(len(qe_idxs)):
            row_a = np.asarray(qe_idxs[i])
            row_cand = result[i]

            # Find elements in row_a that are NOT in row_cand
            # np.isin returns True for elements that ARE there, so we invert it with ~
            missing_elements = row_a[~np.isin(row_a, row_cand)]

            # FIXME: this doesn't check if another positive is there
            if len(missing_elements) > 0:
                # Replace elements starting from the end of the row
                # e.g., if 2 elements are missing, replace result[i, -2:]
                result[i, -len(missing_elements) :] = missing_elements

        return result

    def _name_build_labels_all(
        self,
        gold_idxs: np.ndarray,
        candidates_idxs: np.ndarray,
        name_idx_to_entity_idx: dict[int, int],
        qe_idxs: np.ndarray | None = None,
    ):
        gold_idxs = np.asarray(gold_idxs)

        if not len(self.entity_idx_to_name_idxs):
            mapping = {}
            for name_idx, entity_idx in name_idx_to_entity_idx.items():
                if entity_idx not in mapping:
                    mapping[entity_idx] = []
                mapping[entity_idx].append(name_idx)
            self.entity_idx_to_name_idxs = mapping

        if getattr(self, "qe_train", False):
            candidates_idxs = self._inject_expansion(
                candidates_idxs=candidates_idxs, qe_idxs=qe_idxs
            )

        names_idxs = np.unique(candidates_idxs)

        entity_idxs = np.vectorize(name_idx_to_entity_idx.get)(names_idxs)
        entity_idxs = entity_idxs[np.newaxis, :].repeat(gold_idxs.shape[0], 0)

        retriever_labels = row_wise_isin(entity_idxs, gold_idxs[:, np.newaxis])
        retriever_labels = retriever_labels.astype(int)

        return {
            "idxs": np.unique(candidates_idxs),
            "retriever_labels": torch.as_tensor(retriever_labels),
        }

    def _name_build_labels_topk(
        self,
        gold_idxs: np.ndarray,
        candidates_idxs: np.ndarray,
        name_idx_to_entity_idx: dict[int, int],
    ):
        gold_idxs = np.asarray(gold_idxs)

        if not len(self.entity_idx_to_name_idxs):
            mapping = {}
            for name_idx, entity_idx in name_idx_to_entity_idx.items():
                if entity_idx not in mapping:
                    mapping[entity_idx] = []
                mapping[entity_idx].append(name_idx)
            self.entity_idx_to_name_idxs = mapping

        entity_idxs = np.vectorize(name_idx_to_entity_idx.get)(candidates_idxs)
        retriever_labels = row_wise_isin(entity_idxs, gold_idxs[:, np.newaxis])
        retriever_labels = retriever_labels.astype(int)

        idxs = np.unique(candidates_idxs)

        batch_idxs = {int(j): i for i, j in enumerate(idxs)}

        gather_idxs = np.vectorize(batch_idxs.get)(candidates_idxs)

        return {
            "idxs": idxs,
            "retriever_labels": torch.as_tensor(retriever_labels),
            "gather_idxs": gather_idxs,
        }

    def _name_build_labels(
        self,
        gold_idxs: np.ndarray,
        candidates_idxs: np.ndarray,
        name_idx_to_entity_idx: dict[int, int],
        qe_idxs: np.ndarray | None = None,
    ):
        topk_only = False

        if topk_only:
            return self._name_build_labels_topk(
                gold_idxs=gold_idxs,
                candidates_idxs=candidates_idxs,
                name_idx_to_entity_idx=name_idx_to_entity_idx,
            )
        else:
            return self._name_build_labels_all(
                gold_idxs=gold_idxs,
                candidates_idxs=candidates_idxs,
                name_idx_to_entity_idx=name_idx_to_entity_idx,
                qe_idxs=qe_idxs,
            )

    def build_labels(
        self,
        gold_idxs: np.ndarray,
        candidates_idxs: np.ndarray,
        name_idx_to_entity_idx: dict[int, int] | None = None,
        qe_idxs: np.ndarray | None = None,
    ):
        if self.multilabel:
            assert name_idx_to_entity_idx is not None
            return self._name_build_labels(
                gold_idxs=gold_idxs,
                candidates_idxs=candidates_idxs,
                name_idx_to_entity_idx=name_idx_to_entity_idx,
                qe_idxs=qe_idxs,
            )
        else:
            return self._entity_build_labels(
                gold_idxs=gold_idxs, candidates_idxs=candidates_idxs
            )

    def pad(self, *pad_args, **pad_kwargs):
        if not hasattr(self.tokenizer, "deprecation_warnings"):
            return self.tokenizer.pad(*pad_args, **pad_kwargs)

        warning_state = self.tokenizer.deprecation_warnings.get(
            "Asking-to-pad-a-fast-tokenizer", False
        )
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        try:
            padded = self.tokenizer.pad(
                *pad_args, **pad_kwargs, return_attention_mask=False
            )
        finally:
            self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = (
                warning_state
            )

        return padded

    def _get_mention_boundaries(
        self, features: list[dict[str, Any]]
    ) -> list[tuple[int, int]]:
        boundaries = []
        for f in features:
            start = f["input_ids"].index(self.tokenizer.query_start_token_id)
            end = f["input_ids"].index(self.tokenizer.query_end_token_id)
            boundaries.append((start, end))
        return boundaries

    def _prepare_input_ids(
        self,
        features: list[dict[str, Any]],
        boundaries: list[tuple[int, int]],
        qe: bool = False,
    ):
        mask_token_id = self.tokenizer.mask_token_id

        num_marker_tokens = 0
        if self.tokenizer.bos_token_id:
            num_marker_tokens += 1
        if self.tokenizer.eos_token_id:
            num_marker_tokens += 1

        for i, feature in enumerate(features):
            input_ids = feature["input_ids"]
            start, end = boundaries[i]

            input_ids_budget = (
                self.input_ids_budget
                - (end - start - 1)  # start is `query_start_token_id`
                - num_marker_tokens  # start/end marker token
            )

            if qe:
                num_mask_tokens = len(feature["qe_input_ids"])
                input_ids_budget -= num_mask_tokens

            if self.context_size == -1:
                side_budget = input_ids_budget // 2

                # TODO: this favors left context
                left_budget = min(len(input_ids[:start]), side_budget)

                right_budget = min(
                    len(input_ids[end + 1 :]),  # end is `query_end_token_id`
                    side_budget,
                )
            else:
                left_budget = right_budget = self.context_size

            value = feature["input_ids"]

            # truncate
            value = value[
                max(0, start - left_budget) : min(
                    end
                    + right_budget
                    + 1,  # add because `end` is position of `query_end_token_id` but it is already accounted for in `input_ids_budget`
                    len(input_ids),
                )
            ]

            if self.tokenizer.bos_token_id:
                value = [self.tokenizer.bos_token_id] + value

            if self.tokenizer.eos_token_id:
                value.append(self.tokenizer.eos_token_id)

            if qe:
                value.extend([mask_token_id] * num_mask_tokens)

            feature["input_ids"] = value

        return features


@dataclasses.dataclass
class SingleVectorRetrieverCollator(BaseRetrieverCollator):
    def collate_kb(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self.pad(
            encoded_inputs=[{"input_ids": f["input_ids"]} for f in features],
            **self.pad_kwargs,
        )
        batch["attention_mask"] = (batch["input_ids"] != self.pad_token_type_id).astype(
            int
        )

        if "id" in features[0]:
            batch["id"] = np.asarray([f["id"] for f in features])

        if "idx" in features[0]:
            batch["idx"] = np.asarray([f["idx"] for f in features])
        return batch

    def collate_query(
        self,
        features: list[dict[str, Any]],
        train: bool = False,
    ) -> dict[str, Any]:
        boundaries = self._get_mention_boundaries(features)

        features = self._prepare_input_ids(
            features=features, boundaries=boundaries, qe=False
        )

        batch = self.pad(
            encoded_inputs=[{"input_ids": f["input_ids"]} for f in features],
            **self.pad_kwargs,
        )
        batch["attention_mask"] = (batch["input_ids"] != self.pad_token_type_id).astype(
            int
        )

        query_subword_mask = np.zeros_like(batch["input_ids"], dtype=np.int64)
        _, starts = np.where(batch["input_ids"] == self.tokenizer.query_start_token_id)
        _, ends = np.where(batch["input_ids"] == self.tokenizer.query_end_token_id)
        for i, (s, e) in enumerate(zip(starts, ends)):
            query_subword_mask[i, s] = 1
            query_subword_mask[i, e] = 1

        batch["query_subword_mask"] = query_subword_mask

        batch["id"] = np.asarray([f["id"] for f in features])
        batch["aid"] = np.asarray(
            [f"{f['id']}-{f['start']}-{f['end']}" for f in features]
        )
        if "idx" in features[0]:
            batch["idx"] = np.asarray([f["idx"] for f in features])

        return batch


@dataclasses.dataclass
class MultiVectorRetrieverCollator(BaseRetrieverCollator):
    query_cls: bool = False
    query_prefix: bool = False
    query_punctuation: bool = True

    candidate_description: bool = True
    candidate_aliases: bool = True
    candidate_cls: bool = False
    candidate_prefix: bool = False
    candidate_punctuation: bool = False

    qe: bool = True
    qe_train: bool = False
    qe_attend_mask: bool = False
    qe_negative_mining: bool = False

    def collate_kb(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        feature_names = list(features[0])

        batch = self.pad(
            encoded_inputs=[{"input_ids": f["input_ids"]} for f in features],
            **self.pad_kwargs,
        )
        batch["attention_mask"] = batch["input_ids"] != self.pad_token_type_id
        batch["attention_mask"] = batch["attention_mask"].astype(int)

        # "description": self.candidate_description,
        kwargs = {
            "input_ids": batch["input_ids"],
            "cls_": self.candidate_cls,
            "prefix": self.candidate_prefix,
            "punctuation": self.candidate_punctuation,
        }

        if self.multilabel:
            kwargs["base_masks"] = {"label": np.ones_like(batch["input_ids"])}
            kwargs["aliases"] = False
        else:
            base_masks = {}
            # for column in ["label", "description", "aliases"]:
            for column in ["label", "aliases"]:
                key = f"{column}_idxs"
                mask = np.zeros_like(batch["input_ids"])
                for i in range(batch["input_ids"].shape[0]):
                    mask[i, features[i][key]] = 1
                base_masks[column] = mask

            kwargs["base_masks"] = base_masks
            kwargs["aliases"] = self.candidate_aliases

        csm = self.tokenizer.get_candidates_subword_mask(**kwargs)
        batch["candidates_subword_mask"] = csm

        if "idx" in feature_names:
            batch["idx"] = np.asarray([f["idx"] for f in features])

        if "id" in features[0]:
            batch["id"] = np.asarray([f["id"] for f in features])

        return batch

    def collate_query(
        self,
        features: list[dict[str, Any]],
    ) -> dict[str, Any]:
        boundaries = self._get_mention_boundaries(features)

        features = self._prepare_input_ids(
            features=features,
            boundaries=boundaries,
            qe=self.qe,
        )

        batch = self.pad(
            encoded_inputs=[{"input_ids": f["input_ids"]} for f in features],
            **self.pad_kwargs,
        )

        batch["attention_mask"] = self._get_attention_mask(batch["input_ids"])

        base_masks = {
            "query": self._build_query_mask(batch["input_ids"]),
        }
        batch["query_subword_mask"] = self.tokenizer.get_query_subword_mask(
            input_ids=batch["input_ids"],
            base_masks=base_masks,
            cls_=self.query_cls,
            prefix=self.query_prefix,
            punctuation=self.query_punctuation,
        )

        # used to remove [MASK] tokens from query in negative mining
        if self.qe or self.qe_train:
            qe_mask = batch["input_ids"] == self.tokenizer.mask_token_id
            batch["expansion_subword_mask"] = qe_mask

        if self.qe_train:
            qe_labels = np.empty(
                (qe_mask.shape[0], qe_mask.sum(-1).max()), dtype=np.int64
            )
            qe_labels.fill(-100)
            for i, f in enumerate(features):
                labels = f["qe_input_ids"]
                qe_labels[i, : len(labels)] = labels

            batch["expansion_input_ids"] = qe_labels

        batch["id"] = [f["id"] for f in features]
        batch["aid"] = [f"{f['id']}-{f['start']}-{f['end']}" for f in features]

        if "idx" in features[0]:
            batch["idx"] = [f["idx"] for f in features]

        if "ids" in features[0]:
            batch["ids"] = [f["ids"] for f in features]

        if "qe_idx" in features[0]:
            batch["qe_idx"] = [f["qe_idx"] for f in features]

        batch["text"] = [f["text"] for f in features]
        batch["start"] = [f["start"] for f in features]
        batch["end"] = [f["end"] for f in features]

        return batch

    def _build_query_mask(self, input_ids: np.ndarray) -> np.ndarray:
        _, starts = (input_ids == self.tokenizer.query_start_token_id).nonzero()
        _, ends = (input_ids == self.tokenizer.query_end_token_id).nonzero()

        query_mask = np.zeros_like(input_ids)
        for i in range(query_mask.shape[0]):
            query_mask[i, starts[i] + 1 : ends[i]] = 1

        return query_mask

    def _get_attention_mask(self, input_ids: np.ndarray):
        attention_mask = (input_ids != self.tokenizer.pad_token_type_id).astype(int)
        if not self.qe_attend_mask:  # this is False
            attention_mask[input_ids == self.tokenizer.mask_token_id] = 0
        return attention_mask


def get_collator(
    tokenizer: RetrieverTokenizer,
    multivector: bool = True,
    **kwargs,
) -> BaseRetrieverCollator:
    if multivector:
        return MultiVectorRetrieverCollator(tokenizer=tokenizer, **kwargs)
    return SingleVectorRetrieverCollator(tokenizer=tokenizer, **kwargs)
