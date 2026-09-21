from __future__ import annotations

import dataclasses
import json

import datasets
import numpy as np
import torch
from accelerate import Accelerator
from loguru import logger
from tqdm import tqdm

from src.index import BaseAcceleratedIndex
from src.model import BaseRetriever, MultiVectorRetriever
from src.task.collator import BaseRetrieverCollator
from src.task.metrics import AccuracyMetric, RecallAtKMetric
from src.utils import batch_to_device

RECALL_KS = [5, 10, 20, 32, 40, 50, 64, 100]


@dataclasses.dataclass
class AcceleratedEvaluator:
    accelerator: Accelerator
    collator: BaseRetrieverCollator
    main_metric: str = "recall@1"
    topk_predict: int = 100
    token_topk_predict: int = 2048
    # qe_predict: bool = True

    def __post_init__(self):
        self._metrics = None
        self.recall_ks = [k for k in RECALL_KS if k <= self.topk_predict]

    @property
    def metrics(self) -> dict[str, AccuracyMetric]:
        if self._metrics is None:
            self._metrics = {
                "recall@1": AccuracyMetric(),
            }
            for k in self.recall_ks:
                self._metrics[f"recall@{k}"] = RecallAtKMetric(k=k)
        return self._metrics

    def load_candidates(
        self,
        kb: datasets.Dataset,
        collator: BaseRetrieverCollator,
        candidates_ids: np.ndarray,
        device: torch.device = torch.device("cpu"),
    ) -> dict:
        assert (
            candidates_ids.ndim == 2
        ), "`candidates_ids` must be in form: batch_size x num_candidates"

        # fill the `topk` budget with the i-th ranked candidate for each query
        # equivalnet of `topk / num_query` query-specific candidates
        # # https://stackoverflow.com/a/15637512
        # # Uniques are returned in order of appearance. This does NOT sort.
        # # Significantly faster than numpy.unique for long enough sequences. Includes NA values.
        # idxs = pd.unique(candidates_ids.T.flatten()).tolist()
        # idxs = idxs[:topk]

        cid_to_cidx = {
            v: k for k, v in dict(enumerate(np.unique(candidates_ids).tolist())).items()
        }

        rows = [kb[i] for i in list(cid_to_cidx)]

        candidates = self.collator.collate_kb(rows)

        candidates = batch_to_device(candidates, device=device)

        candidates["query_to_candidates"] = torch.as_tensor(
            np.vstack(
                [[cid_to_cidx[cid] for cid in cids] for cids in candidates_ids.tolist()]
            )
        )

        return candidates

    def _end_evaluation(self) -> dict[str, float]:
        metrics = {}
        for k, v in self.metrics.items():
            metrics[k] = round(v.compute() * 100, 2)
            v.reset()
        return metrics

    def _add_runtime_params(self, metrics: dict) -> dict:
        metrics.update(
            {
                "token_topk_predict": self.token_topk_predict,
            }
        )
        return metrics

    def run(
        self,
        dl: torch.utils.data.DataLoader,
        model: BaseRetriever,
        index: BaseAcceleratedIndex,
        kb: datasets.Dataset,
        step: int,
        progress_bar: bool = False,
        name_idx_to_entity_idx: dict[int, int] | None = None,
    ):
        if progress_bar:
            logger.info("Start evaluation")

        model = self.accelerator.unwrap_model(model)
        model.eval()
        device = self.accelerator.device

        is_multivector = isinstance(model, MultiVectorRetriever)

        for batch in tqdm(dl, desc="Evaluate", disable=not progress_bar):
            y_true = torch.as_tensor(batch["idx"], device=device)
            y_true_all = self.accelerator.gather_for_metrics(y_true).cpu().numpy()

            model_kwargs = {
                "query_input_ids": batch["input_ids"],
                "query_attention_mask": batch["attention_mask"],
                "query_subword_mask": batch["query_subword_mask"],
            }
            # if is_multivector and not self.qe_predict:
            #     model_kwargs["query_subword_mask"] = (
            #         batch["query_subword_mask"] - batch["expansion_subword_mask"]
            #     )

            with torch.no_grad():
                query = model(**model_kwargs)

            search_kwargs = {
                "queries": query["query_embedding"],
                "topk": self.topk_predict,
            }
            if is_multivector:
                search_kwargs["lengths"] = query["query_subword_mask"].sum(-1)
                search_kwargs["token_topk"] = self.token_topk_predict

            retrieval = index.search(**search_kwargs)
            candidates_ids = retrieval["candidates_ids"]

            if name_idx_to_entity_idx is not None:
                candidates_ids = np.vectorize(name_idx_to_entity_idx.get)(
                    candidates_ids
                )

            y_pred_ret_all = self.accelerator.gather_for_metrics(
                torch.as_tensor(candidates_ids, device=device)
            )
            y_pred_ret_all = y_pred_ret_all.cpu().numpy()
            self.metrics["recall@1"].update(
                y_pred=y_pred_ret_all[:, 0], y_true=y_true_all
            )
            for k in self.recall_ks:
                self.metrics[f"recall@{k}"].update(
                    y_pred=y_pred_ret_all, y_true=y_true_all[..., np.newaxis]
                )

        self.accelerator.wait_for_everyone()

        metrics = self._end_evaluation()
        self.accelerator.log(metrics, step=step)

        if progress_bar:
            logger.info("Evaluation end: {}", json.dumps(metrics, indent=1))

        metrics = self._add_runtime_params(metrics)

        return metrics
