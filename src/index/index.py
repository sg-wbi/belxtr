import math
import os
import pickle
from abc import ABC, abstractmethod

import numpy as np
import polars as pl
import torch
from accelerate import Accelerator
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.index.config import IndexConfig
# from src.index.search import ApproximateSearch, ExactSearch
from src.index.search import  ExactSearch
from src.model.retriever import (
    BaseRetriever,
    MultiVectorRetriever,
    SingleVectorRetriever,
)
from src.utils import distributed as dist_utils
from src.utils import timeit  # noqa: F401


def _serialize_uids(uids: list[list[int]]):
    return torch.tensor(list(pickle.dumps(uids)), dtype=torch.uint8).cuda()


def _deserialize_uids(uids):
    return [pickle.loads(x.cpu().numpy().tobytes()) for x in uids]


def _init_search_output(shape: tuple):
    indices = np.empty(shape)
    indices.fill(-1)
    indices = indices.astype(int)
    scores = np.empty(shape)
    scores.fill(-1)
    return scores, indices


class BaseAcceleratedIndex(ABC):
    def __init__(
        self,
        accelerator: Accelerator,
        config: IndexConfig = IndexConfig(),
        directory: str | None = None,
    ):
        self.directory = directory
        if self.directory is not None:
            os.makedirs(self.directory, exist_ok=True)

        self.accelerator = accelerator

        self.rank = self.accelerator.local_process_index
        self.world_size = self.accelerator.num_processes
        self.device = self.accelerator.device

        self.config = config

        if self.config.ann is not None:
            raise NotImplemented()
            # self.searcher = ApproximateSearch(
            #     metric=self.config.metric,
            #     device=self.device,
            #     force_cpu=config.force_cpu,
            #     method=config.ann,
            #     code_size=config.code_size,
            #     nlist=config.nlist,
            #     nprobe=config.nprobe,
            #     use_cuvs=config.use_cuvs,
            # )
        else:
            self.searcher = ExactSearch(
                metric=self.config.metric,
                device=self.device,
                force_cpu=config.force_cpu,
            )

    @property
    def saved(self) -> bool:
        if self.directory is not None:
            return (
                os.path.exists(self.directory) and len(os.listdir(self.directory)) > 0
            )
        return False

    def build(
        self,
        model: BaseRetriever,
        dl: DataLoader,
        progress_bar: bool = False,
    ):
        if progress_bar:
            logger.info("Build index")

        model.eval()
        ids, vectors = self._build(
            model=model,
            dl=dl,
            progress_bar=progress_bar,
        )

        self.searcher.initialize(vectors=vectors, ids=ids, progress_bar=progress_bar)

        # search requires all processes to have loaded the index
        self.accelerator.wait_for_everyone()

    @abstractmethod
    def _build(
        self,
        model: BaseRetriever,
        dl: DataLoader,
        progress_bar: bool = False,
    ) -> tuple[list[int], torch.Tensor]:
        pass

    def _save_shard(
        self, directory: str, shard_id: int, ids: list[int], vectors: torch.Tensor
    ):
        ids_shard_path = os.path.join(directory, f"ids_{shard_id}.txt")
        with open(ids_shard_path, "w") as fp:
            fp.write(",".join([str(i) for i in ids]))

        vectors_shard_path = os.path.join(directory, f"vectors_{shard_id}.pt")
        torch.save(vectors.cpu(), vectors_shard_path)

    def save(
        self,
        directory: str | None = None,
        num_shards: int | None = None,
        progress_bar: bool = False,
    ):
        directory = directory if directory is not None else self.directory
        os.makedirs(directory, exist_ok=True)

        assert directory is not None, "`directory` cannot be None"

        # 'num_shards' = 6 ** 5 : pick a number s.t. %2 and %3 == 0
        num_shards = num_shards if num_shards is not None else 6 * self.world_size

        assert (
            num_shards % self.world_size == 0
        ), "# of workers must be a multiple of shards to save!"
        shards_per_worker = num_shards // self.world_size

        assert self.searcher.vectors is not None, "Call `.build()` first"
        num_vectors = self.searcher.vectors.shape[1]
        vectors_per_shard = math.ceil(num_vectors / shards_per_worker)
        for shard_ind, shard_start in enumerate(
            tqdm(
                range(0, num_vectors, vectors_per_shard),
                desc="Index - save data",
                disable=not progress_bar,
            )
        ):
            shard_end = min(shard_start + vectors_per_shard, num_vectors)
            # get global shard number
            shard_id = shard_ind + self.rank * shards_per_worker

            self._save_shard(
                directory=directory,
                shard_id=shard_id,
                ids=self.searcher.ids[shard_start:shard_end],
                vectors=self.searcher.vectors[:, shard_start:shard_end].clone(),
            )

        # if isinstance(self.searcher, ApproximateSearch):
        #     raise NotImplementedError()
            # path = os.path.join(
            #     directory,
            #     f"{self.searcher.name}_rank{self.rank}_ws{self.world_size}.faiss",
            # )
            # self.searcher.save(path=path)

    def _load_shard(
        self, directory: str, shard_id: int
    ) -> tuple[list[int], torch.Tensor]:
        ids_shard_path = os.path.join(directory, f"ids_{shard_id}.txt")
        with open(ids_shard_path) as fp:
            ids_shard = [int(i) for i in fp.read().split(",")]

        vectors_shard_path = os.path.join(directory, f"vectors_{shard_id}.pt")
        vectors_shard = torch.load(vectors_shard_path, map_location="cpu")
        vectors_shard = vectors_shard.to(torch.float16)
        return ids_shard, vectors_shard

    def _load_ids_and_vectors(
        self, progress_bar: bool = False
    ) -> tuple[list[int], torch.Tensor]:
        assert self.directory is not None, "`directory` cannot be None"
        num_shards = len([f for f in os.listdir(self.directory) if f.endswith(".pt")])
        assert (
            num_shards % self.world_size == 0
        ), f"`world_size={self.world_size}` must be a multiple of `num_shards={num_shards}`"

        shards_per_worker = num_shards // self.world_size
        vectors_list: list[torch.Tensor] = []
        ids = []
        for shard_id in tqdm(
            range(self.rank * shards_per_worker, (self.rank + 1) * shards_per_worker),
            desc="Index - load data",
            disable=not progress_bar,
        ):
            ids_shard, vectors_shard = self._load_shard(
                directory=self.directory, shard_id=shard_id
            )
            ids.extend(ids_shard)
            vectors_list.append(vectors_shard)

        vectors = torch.concat(vectors_list, dim=1)
        num_vectors = vectors.shape[1]
        assert num_vectors == len(
            ids
        ), f"# of vectors {num_vectors} != # of uids {len(ids)}"

        return ids, vectors

    def _load_exact_search(self, progress_bar: bool = False):
        ids, vectors = self._load_ids_and_vectors(progress_bar)
        self.searcher.initialize(ids=ids, vectors=vectors, progress_bar=progress_bar)

    def _load_approximate_search(self, progress_bar: bool = False):
        assert self.directory is not None
        files = [
            f
            for f in os.listdir(self.directory)
            if f.endswith(".faiss") and f"_ws{self.world_size}" in f
        ]
        if len(files) != self.world_size:
            if progress_bar:
                logger.info(
                    "# of ANN shards ({}) != # of GPUs ({}): recompute",
                    len(files),
                    self.world_size,
                )
            ids, vectors = self._load_ids_and_vectors(progress_bar=progress_bar)
            self.searcher.initialize(
                vectors=vectors, ids=ids, progress_bar=progress_bar
            )
            path = os.path.join(
                self.directory,
                f"{self.searcher.name}_rank{self.rank}_ws{self.world_size}.faiss",
            )
            self.searcher.save(path)
        else:
            files = [f for f in files if f"_rank{self.rank}_ws{self.world_size}" in f]
            assert len(files) == 1, f"Found {len(files)} ({files}) but expected one"
            self.searcher.load(os.path.join(self.directory, files[0]))

    def load(self, progress_bar: bool = False):
        if progress_bar:
            logger.info("Load index")

        if isinstance(self.searcher, ExactSearch):
            self._load_exact_search(progress_bar)
        else:
            self._load_approximate_search(progress_bar)

        # search requires all processes to have loaded the index
        self.accelerator.wait_for_everyone()

    def _reshape_output(
        self,
        batch_size: int,
        max_len: int,
        topk: int,
        scores_flat: np.ndarray,
        uids_flat: np.ndarray,
        lengths: torch.Tensor | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if batch_size == 1:
            if lengths is not None:
                scores = scores_flat[np.newaxis, : lengths[0]]
                uids = uids_flat[np.newaxis, : lengths[0]]
            else:
                scores = scores_flat[np.newaxis, :]
                uids = uids_flat[np.newaxis, :]
            return scores, uids
        else:
            if lengths is not None:
                scores, uids = _init_search_output(
                    shape=(int(batch_size), int(max_len), topk)
                )
                start = 0
                for i in range(batch_size):
                    end = start + int(lengths[i])
                    # in case there is an empty query
                    # i.e. NED with slice with no mentions
                    if lengths[i] > 0:
                        uids[i, : lengths[i]] = uids_flat[start:end]
                        scores[i, : lengths[i]] = scores_flat[start:end]
                    start = end
            else:
                uids = uids_flat.reshape((batch_size, max_len, topk))  # type: ignore
                scores = scores_flat.reshape((batch_size, max_len, topk))  # type: ignore
            return scores, uids

    @torch.no_grad()
    def _search_tokens(
        self,
        queries: torch.Tensor,
        topk: int,
        lengths: torch.Tensor | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_size, max_len, hidden_size = queries.shape

        queries_flat = self._flatten_queries(queries, lengths=lengths)

        scores_flat, uids_flat = self.searcher.search(queries=queries_flat, topk=topk)

        return self._reshape_output(
            batch_size=batch_size,
            max_len=int(max_len),
            topk=topk,
            scores_flat=scores_flat.cpu().numpy(),
            uids_flat=np.asarray(uids_flat),
            lengths=lengths,
        )

    def _gather_uids(self, uids: list[list[int]], allsizes: np.ndarray):
        batched_uids = [
            uids[allsizes[k] : allsizes[k + 1]] for k in range(len(allsizes) - 1)
        ]
        serialized_uids = [_serialize_uids(x) for x in batched_uids]
        serialized_gather_uids = [
            dist_utils.varsize_gather(serialized_uids[k], dst=k, dim=0)
            for k in range(self.world_size)
        ]
        serialized_rank_uids = serialized_gather_uids[self.rank]
        rank_uids = _deserialize_uids(serialized_rank_uids)
        # merge_uids: list = [[] for _ in range(uids.size(0))]
        merge_uids: list = [[] for _ in range(len(uids))]
        for _uids in rank_uids:
            for k, x in enumerate(_uids):
                merge_uids[k].extend(x)
        return merge_uids

    def _gather_scores(self, scores: torch.Tensor, allsizes: np.ndarray):
        batched_scores = [
            scores[allsizes[k] : allsizes[k + 1]] for k in range(len(allsizes) - 1)
        ]
        gather_scores = [
            dist_utils.varsize_gather(batched_scores[k], dst=k, dim=1)
            for k in range(self.world_size)
        ]
        rank_scores = gather_scores[self.rank]
        return torch.cat(rank_scores, dim=1).cpu()

    def _flatten_queries(
        self, queries: torch.Tensor, lengths: torch.Tensor | None = None
    ):
        if lengths is not None:
            queries_flat = torch.vstack(
                [queries[i, : lengths[i], :] for i in range(queries.shape[0])]
            )
        else:
            queries_flat = queries.reshape(-1, queries.shape[-1])

        return queries_flat

    @torch.no_grad()
    def _dist_search_tokens(
        self, queries: torch.Tensor, topk: int, lengths: torch.Tensor | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Conducts exhaustive search of the k-nearest neighbours
        using the inner product metric.
        """

        batch_size, max_len, hidden_size = queries.shape

        queries_flat = self._flatten_queries(queries, lengths=lengths)

        allqueries = dist_utils.varsize_all_gather(queries_flat)
        # compute scores for the part of the index located on each process
        allscores, alluids = self.searcher.search(queries=allqueries, topk=topk)

        allsizes = np.cumsum([0] + dist_utils.get_varsize(queries_flat).cpu().tolist())

        rank_indices = self._gather_uids(uids=alluids, allsizes=allsizes)

        rank_scores = self._gather_scores(scores=allscores, allsizes=allsizes)

        _, subindices = torch.topk(rank_scores, topk, dim=1)
        subindices = subindices.tolist()  # type: ignore

        # Extract topk scores and associated ids
        scores_flat = np.asarray(
            [[rank_scores[k][j] for j in idx] for k, idx in enumerate(subindices)]
        )

        uids_flat = np.asarray(
            [[rank_indices[k][j] for j in idx] for k, idx in enumerate(subindices)]
        )

        return self._reshape_output(
            batch_size=batch_size,
            max_len=int(max_len),
            topk=topk,
            scores_flat=scores_flat,
            uids_flat=uids_flat,
            lengths=lengths,
        )

    @torch.no_grad()
    def search_tokens(
        self,
        queries: torch.Tensor,
        lengths: torch.Tensor | None = None,
        topk: int = 100,
    ):
        if self.world_size == 1:
            scores, indices = self._search_tokens(
                queries=queries, topk=topk, lengths=lengths
            )
        else:
            scores, indices = self._dist_search_tokens(
                queries=queries, topk=topk, lengths=lengths
            )
        return scores, indices

    @torch.no_grad()
    def search(
        self,
        queries: torch.Tensor,
        topk: int = 100,
        **kwargs,
    ) -> dict:
        raise NotImplementedError()


class AcceleratedMultiVectorIndex(BaseAcceleratedIndex):
    def _build(
        self,
        model: MultiVectorRetriever,
        dl: DataLoader,
        progress_bar: bool = False,
    ) -> tuple[list[int], torch.Tensor]:
        vectors_list: list[torch.Tensor] = []
        ids = []
        for batch in tqdm(dl, desc="Index - build vectors", disable=not progress_bar):
            if not isinstance(batch["idx"], np.ndarray):
                batch["idx"] = batch["idx"].cpu().numpy()

            candidates_subword_mask = batch["candidates_subword_mask"]

            with torch.no_grad():
                retriever_out = model(
                    candidates_input_ids=batch["input_ids"],
                    candidates_attention_mask=batch["attention_mask"],
                    candidates_subword_mask=candidates_subword_mask,
                )

            batch_vectors = retriever_out["candidates_embedding"].to(torch.float16)
            if self.config.ann is not None:
                batch_vectors = batch_vectors.cpu()

            sliced_mask = retriever_out["candidates_subword_mask"].cpu().numpy()
            batch_ids = batch["idx"][:, np.newaxis].repeat(sliced_mask.shape[1], -1)
            flat_batch_ids = batch_ids[sliced_mask == 1].tolist()
            flat_batch_vectors = batch_vectors.view(-1, batch_vectors.shape[-1])
            flat_batch_vectors = flat_batch_vectors[sliced_mask.flatten() == 1, :].T

            assert (
                len(flat_batch_ids) == flat_batch_vectors.shape[1]
            ), f"# of vectors {len(flat_batch_vectors.shape[1])} != # of ids {len(flat_batch_ids)}"

            ids.extend(flat_batch_ids)
            vectors_list.append(flat_batch_vectors)

        vectors = torch.concat(vectors_list, dim=1)
        assert vectors.shape[1] == len(
            ids
        ), f"# of vectors {vectors.shape[1]} != # of ids {len(ids)}"
        return ids, vectors

    @torch.no_grad()
    def aggregate_scores(
        self, tokens_scores: np.ndarray, tokens_ids: np.ndarray, topk: int = 100
    ):
        batch_size = tokens_ids.shape[0]

        candidates_scores, candidates_ids = _init_search_output((batch_size, topk))

        for i in range(batch_size):
            i_mask = tokens_ids[i, :, 0] != -1

            i_indices = tokens_ids[i, i_mask, :]
            i_scores = tokens_scores[i, i_mask, :]

            idx_to_cand = dict(
                enumerate(pl.DataFrame({"ids": i_indices.flatten()})["ids"].unique())
            )

            cand_to_idx = {v: k for k, v in idx_to_cand.items()}

            i_idxs = np.vectorize(lambda x: cand_to_idx.get(x, -1))(i_indices)

            i_tokens = np.arange(i_indices.shape[0])[:, np.newaxis].repeat(
                i_indices.shape[1], axis=-1
            )

            df = pl.DataFrame(
                {
                    "token": i_tokens.reshape(-1),
                    "score": i_scores.reshape(-1),
                    "index": i_idxs.reshape(-1),
                }
            )

            df = df.group_by(["token", "index"]).agg(pl.max("score"))

            # pre-fill with lowest score for each token
            # this is the assignment of the estimated missing score (see XTR paper for details).
            c_scores = i_scores.min(-1)[..., np.newaxis].repeat(
                len(cand_to_idx), axis=-1
            )

            # # assign score to candidites that were retrieved
            c_scores[df["token"], df["index"]] = df["score"]

            # candidate score is mean over tokens
            c_scores = c_scores.mean(0)

            # sort by score (maximum first)
            sorting = np.argsort(-c_scores)

            sorted_topk_scores = c_scores[sorting[:topk]]
            candidates_scores[i, : len(sorted_topk_scores)] = sorted_topk_scores

            sorted_topk_indices = np.vectorize(lambda x: idx_to_cand.get(x, -1))(
                sorting[:topk]
            )
            candidates_ids[i, : len(sorted_topk_indices)] = sorted_topk_indices

        return candidates_scores, candidates_ids

    @torch.no_grad()
    def search(
        self,
        queries: torch.Tensor,
        topk: int = 100,
        **kwargs,
    ) -> dict:
        token_topk = kwargs.get("token_topk", 100)
        return_candidates = kwargs.get("return_candidates", True)
        lengths = kwargs.get("lengths", None)

        out = {}

        tokens_scores, tokens_ids = self.search_tokens(
            queries=queries, lengths=lengths, topk=token_topk
        )
        out["tokens_scores"] = tokens_scores
        out["tokens_ids"] = tokens_ids
        if return_candidates:
            candidates_scores, candidates_ids = self.aggregate_scores(
                tokens_scores=tokens_scores, tokens_ids=tokens_ids, topk=topk
            )
            out["candidates_scores"] = candidates_scores
            out["candidates_ids"] = candidates_ids

        return out


class AcceleratedSingleVectorIndex(BaseAcceleratedIndex):
    def _build(
        self,
        model: SingleVectorRetriever,
        dl: DataLoader,
        progress_bar: bool = False,
    ) -> tuple[list[int], torch.Tensor]:
        vectors_list: list[torch.Tensor] = []
        ids = []
        for batch in tqdm(dl, desc="Index - build vectors", disable=not progress_bar):
            if not isinstance(batch["idx"], np.ndarray):
                batch["idx"] = batch["idx"].cpu().numpy()

            # For single vector we don't necessarily need subword mask,
            # but we follow the same forward pattern for consistency.
            with torch.no_grad():
                retriever_out = model(
                    candidates_input_ids=batch["input_ids"],
                    candidates_attention_mask=batch["attention_mask"],
                )

            batch_vectors = retriever_out["candidates_embedding"].to(torch.float16)
            if self.config.ann is not None:
                batch_vectors = batch_vectors.cpu()

            batch_vectors = batch_vectors.squeeze(1)

            ids.extend(batch["idx"].tolist())
            vectors_list.append(batch_vectors.T)

        vectors = torch.concat(vectors_list, dim=1)
        assert vectors.shape[1] == len(
            ids
        ), f"# of vectors {vectors.shape[1]} != # of ids {len(ids)}"
        return ids, vectors

    @torch.no_grad()
    def search(
        self,
        queries: torch.Tensor,
        topk: int = 100,
        **kwargs,
    ) -> dict:
        # In single vector mode, token_topk is essentially the topk
        # and lengths should be handled by the pooling (usually queries.shape[1] == 1 or we pool here)

        # Ensure queries are 3D (batch_size, 1, hidden_size)
        if queries.ndim == 2:
            queries = queries.unsqueeze(1)

        assert (
            queries.ndim == 3 and queries.shape[1] == 1
        ), f"Expected 3D queries with 1 token for single vector search, got {queries.shape}"

        scores, indices = self.search_tokens(queries=queries, lengths=None, topk=topk)

        # Reshape to (batch_size, topk) as expected for single vector
        scores = scores.squeeze(1)
        indices = indices.squeeze(1)

        out = {
            "candidates_scores": scores,
            "candidates_ids": indices,
        }
        return out


def get_index(
    accelerator: Accelerator,
    config: IndexConfig = IndexConfig(),
    directory: str | None = None,
) -> BaseAcceleratedIndex:
    if config.multivector:
        return AcceleratedMultiVectorIndex(
            accelerator=accelerator, config=config, directory=directory
        )
    else:
        return AcceleratedSingleVectorIndex(
            accelerator=accelerator, config=config, directory=directory
        )
