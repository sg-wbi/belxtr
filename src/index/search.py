import math
from typing import Union, get_args

# import faiss
# https://github.com/facebookresearch/faiss/blob/main/contrib/torch_utils.py
# import faiss.contrib.torch_utils
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm


class BaseSearher:
    def __init__(
        self,
        metric: str = "cos",
        device: torch.device = torch.device("cpu"),
        force_cpu: bool = False,
    ):
        self.metric = metric
        self.device = device
        self.force_cpu = force_cpu
        self.vectors: torch.Tensor | None = None
        self.ids: list[int] | None = None


class ExactSearch(BaseSearher):
    def initialize(
        self, vectors: torch.Tensor, ids: list[int], progress_bar: bool = False
    ):
        self.vectors = vectors
        self.ids = ids
        if self.force_cpu:
            self.vectors = self.vectors.to("cpu")
        else:
            self.vectors = self.vectors.to(self.device)

    def _negative_l2_squared_batched(
        self, queries: torch.Tensor, batch_size: int = 2**16
    ):
        all_scores = []
        num_vectors = self.vectors.shape[1]

        for start in range(0, num_vectors, batch_size):
            end = min(start + batch_size, num_vectors)
            vector_batch = self.vectors[:, start:end]
            squared_dist = torch.sum(
                (queries[:, :, None] - vector_batch[None, :, :]) ** 2, dim=1
            )
            all_scores.append(-squared_dist)

        return torch.cat(all_scores, dim=1)

    def search(
        self, queries: torch.Tensor, topk: int, batch_size: int = 2**16
    ) -> tuple[torch.Tensor, list[list[int]]]:
        """
        Computes the distance matrix for the query embeddings and embeddings chunk
        and returns the k-nearest neighbours and corresponding scores.
        """

        assert self.vectors is not None, "Index was not initialized"
        assert self.ids is not None, "Index was not initialized"

        if self.metric in ["cos", "ip"]:
            scores = torch.matmul(queries.to(torch.float16), self.vectors)
        else:
            # elif self.metric == "l2":
            #     # Compute negative squared L2 distances (similarity scores)
            #     scores = self._negative_l2_squared_batched(
            #         queries=queries.to(torch.float16), batch_size=batch_size
            #     )
            raise NotImplementedError(f"Invalid metric {self.metric}")

        scores, idxs = torch.topk(scores, topk, dim=1)
        ids = [[self.ids[i] for i in inds] for inds in idxs.tolist()]
        return scores.half(), ids


def _cast_to_torch32(vectors: torch.Tensor) -> torch.Tensor:
    """
    Converts a torch tensor to a contiguous float 32 torch tensor.
    """
    return vectors.to(torch.float32).contiguous()


def _cast_to_numpy(vectors: torch.Tensor) -> np.ndarray:
    """
    Converts a torch tensor to a contiguous numpy float 32 ndarray.
    """
    return (
        vectors.cpu().to(dtype=torch.float16).numpy().astype("float32").copy(order="C")
    )


# CPUFaissIndex = Union[
#     faiss.IndexIVFFlat,
#     faiss.IndexIVFPQ,
#     faiss.IndexIVFScalarQuantizer,
# ]
#
# GPUFaissIndex = Union[
#     faiss.GpuIndexIVFFlat,
#     faiss.GpuIndexIVFPQ,
#     faiss.GpuIndexIVFScalarQuantizer,
# ]
#
#
# GPUFaissIndexConfig = Union[
#     faiss.GpuIndexIVFFlatConfig,
#     faiss.GpuIndexIVFPQConfig,
#     faiss.GpuIndexIVFScalarQuantizerConfig,
# ]


# class ApproximateSearch(BaseSearher):
#     def __init__(
#         self,
#         method: str,
#         code_size: int = 16,
#         nlist: int | None = None,
#         nprobe: int | None = None,
#         use_cuvs: bool = False,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.method = method
#
#         self.code_size = code_size
#         self.nprobe = nprobe
#         self.nbits = 8
#         self.nlist = nlist
#         self.use_cuvs = use_cuvs
#
#         self.index: faiss.Index | None = None
#
#         self.gpu_resources: faiss.StandardGpuResources | None = None
#         if self.device.type == "cuda":
#             self.gpu_resources = faiss.StandardGpuResources()
#
#         self._invalid_result_warned = False
#
#     @property
#     def name(self):
#         return f"{self.method}{self.code_size}" if "q" in self.method else self.method
#
#     def initialize(
#         self,
#         vectors: torch.Tensor,
#         ids: list[int],
#         progress_bar: bool = False,
#     ):
#         self.vectors = vectors
#         self.ids = ids
#
#         num_vectors = vectors.shape[1]
#         vector_size = vectors.shape[0]
#
#         nlist = (
#             self.nlist
#             if self.nlist is not None
#             # https://github.com/facebookresearch/faiss/issues/112#issuecomment-299781486
#             else math.floor(4 * math.sqrt(num_vectors))
#         )
#
#         self.index = (
#             self._get_cpu_index(vector_size=vector_size)
#             if self.device.type == "cpu"
#             else self._get_gpu_index(nlist=nlist, vector_size=vector_size)
#         )
#
#         # flat index doesn't need training
#         if not self.index.is_trained:
#             self.train(
#                 vectors=vectors,
#                 nlist=nlist,
#                 device=self.device,
#                 progress_bar=progress_bar,
#             )
#
#         if self.index.ntotal == 0:
#             device = self.device
#
#             if self.force_cpu:
#                 device = torch.device("cpu")
#                 self.index = faiss.index_gpu_to_cpu(self.index)
#
#             self.populate(
#                 vectors=vectors, ids=ids, device=device, progress_bar=progress_bar
#             )
#
#         nprobe = (
#             self.nprobe
#             if self.nprobe is not None
#             else math.floor(math.sqrt(self.index.ntotal))
#         )
#         if isinstance(self.index, get_args(GPUFaissIndex)):
#             # GPU IVF index only supports nprobe selection up to 2048
#             self.index.nprobe = min(nprobe, 2048)
#         else:
#             self.index.nprobe = nprobe
#
#     def train(
#         self,
#         vectors: torch.Tensor,
#         nlist: int,
#         device: torch.device = torch.device("cpu"),
#         progress_bar: bool = False,
#     ):
#         assert self.index is not None
#
#         num_vectors = vectors.shape[1]
#         if progress_bar:
#             logger.info("ANN: train")
#
#         self.index.reset()
#
#         # https://github.com/facebookresearch/faiss/wiki/FAQ#can-i-ignore-warning-clustering-xxx-points-to-yyy-centroids
#         size = int(nlist * faiss.ClusteringParameters().max_points_per_centroid)
#         if size < num_vectors:
#             idxs = np.random.RandomState(123).choice(
#                 list(range(num_vectors)), size=size, replace=False
#             )
#         else:
#             idxs = np.arange(num_vectors)
#
#         if isinstance(self.index, get_args(GPUFaissIndex)):
#             self.index.train(_cast_to_torch32(vectors[:, idxs].to(device).T))
#         else:
#             self.index.train(_cast_to_numpy(vectors[:, idxs].T))
#
#     def populate(
#         self,
#         vectors: torch.Tensor,
#         ids: list[int],
#         device: torch.device = torch.device("cpu"),
#         progress_bar: bool = False,
#     ):
#         assert self.index is not None
#         num_vectors = vectors.shape[1]
#
#         chunk_size = num_vectors // 3
#
#         chunks = [
#             (vectors[:, 0:chunk_size], ids[0:chunk_size]),
#             (
#                 vectors[:, chunk_size : 2 * chunk_size],
#                 ids[chunk_size : 2 * chunk_size],
#             ),
#             (
#                 vectors[:, 2 * chunk_size : num_vectors],
#                 ids[2 * chunk_size : num_vectors],
#             ),
#         ]
#
#         for vectors, ids in tqdm(
#             chunks, desc="ANN - populate", disable=not progress_bar
#         ):
#             if isinstance(self.index, get_args(GPUFaissIndex)):
#                 self.index.add_with_ids(
#                     _cast_to_torch32(vectors.to(device).T), torch.tensor(ids)
#                 )
#             else:
#                 self.index.add_with_ids(_cast_to_numpy(vectors.T), np.asarray(ids))
#
#     def save(
#         self,
#         path: str,
#         progress_bar: bool = False,
#     ):
#         if progress_bar:
#             logger.info("ANN: save data")
#
#         faiss.write_index(faiss.index_gpu_to_cpu(self.index), path)
#
#     def load(self, path: str, progress_bar: bool = False):
#         if progress_bar:
#             logger.info("ANN: load data")
#
#         self.index = faiss.read_index(path)
#         self.index.nprobe = (
#             self.nprobe
#             if self.nprobe is not None
#             else math.floor(math.sqrt(self.index.ntotal))
#         )
#         if self.device.type == "cuda" and not self.force_cpu:
#             self.index = faiss.index_cpu_to_gpu(
#                 self.gpu_resources,
#                 torch.cuda.current_device(),
#                 self.index,
#                 self._get_gpu_cloner_options(),
#             )
#             # GPU IVF index only supports nprobe selection up to 2048
#             self.index.nprobe = min(self.index.nprobe, 2048)
#
#     def search(
#         self, queries: torch.tensor, topk: int
#     ) -> tuple[torch.Tensor, list[list[int]]]:
#         """
#         Computes the distance matrix for the query embeddings and embeddings chunk
#         and returns the k-nearest neighbours and corresponding scores.
#         """
#         assert self.index is not None, "ANN: index was not initialized"
#
#         if isinstance(self.index, get_args(GPUFaissIndex)):
#             # GPU IVF index only supports nprobe selection up to 2048
#             self.index.nprobe = min(self.index.nprobe, 2048)
#             scores, uids = self.index.search(_cast_to_torch32(queries), topk)
#         else:
#             np_scores, uids = self.index.search(_cast_to_numpy(queries), topk)
#             scores = torch.as_tensor(np_scores, device=self.device)
#
#         uids = uids.tolist()
#
#         if (np.asarray(uids) == -1).any() and not self._invalid_result_warned:
#             logger.warning(
#                 "`topk`>>`nlist`: this returns -1. Try increasing `nprobe` "
#                 "See: https://github.com/facebookresearch/faiss/wiki/FAQ#what-does-it-mean-when-a-search-returns--1-ids"
#             )
#             self._invalid_result_warned = True
#
#         return (scores.half(), uids)
#
#     def _get_cpu_index(self, vector_size: int):
#         raise NotImplementedError()
#
#     def _set_gpu_index_config(
#         self, index_config: GPUFaissIndexConfig
#     ) -> GPUFaissIndexConfig:
#         """
#         Returns the GPU config options for GPU indexes.
#         """
#         index_config.device = torch.cuda.current_device()
#         index_config.useFloat16LookupTables = True
#         index_config.indicesOptions = faiss.INDICES_32_BIT
#         # https://github.com/facebookresearch/faiss/wiki/GPU-Faiss-with-cuVS-usage
#         index_config.use_cuvs = self.use_cuvs
#         return index_config
#
#     def _get_gpu_cloner_options(self) -> faiss.GpuMultipleClonerOptions:
#         """
#         Returns the GPU cloner options neccessary when moving a CPU index to the GPU.
#         """
#         cloner_opts = faiss.GpuClonerOptions()
#         cloner_opts.useFloat16 = True
#         cloner_opts.usePrecomputed = False
#         cloner_opts.indicesOptions = faiss.INDICES_32_BIT
#         # https://github.com/facebookresearch/faiss/wiki/GPU-Faiss-with-cuVS-usage
#         cloner_opts.use_cuvs = self.use_cuvs
#         return cloner_opts
#
#     def _get_similarity_metric(self):
#         if self.metric in ["cos", "ip"]:
#             return faiss.METRIC_INNER_PRODUCT
#         else:
#             # elif self.metric == "l2":
#             #     return faiss.METRIC_L2
#             raise NotImplementedError(f"Invalid metric {self.metric}")
#
#     def _get_gpu_index(self, nlist: int, vector_size: int) -> GPUFaissIndexConfig:
#         """
#         Instantiates and returns the selected GPU index class.
#         """
#
#         metric = self._get_similarity_metric()
#         # if self.method == "flat":
#         #     index = faiss.GpuIndexFlatIP(
#         #         self.gpu_resources,
#         #         self.vector_size,
#         #         self._set_gpu_index_config(faiss.GpuIndexFlatConfig()),
#         #     )
#         if self.method == "ivfflat":
#             index = faiss.GpuIndexIVFFlat(
#                 self.gpu_resources,
#                 vector_size,
#                 nlist,
#                 metric,
#                 self._set_gpu_index_config(faiss.GpuIndexIVFFlatConfig()),
#             )
#         # elif self.method == "pq":
#         #     index = faiss.index_cpu_to_gpu(
#         #         self.gpu_resources,
#         #         self.device,
#         #         faiss.index_factory(
#         #             self.vector_size,
#         #             "PQ" + str(self.config.code_size),
#         #             faiss.METRIC_INNER_PRODUCT,
#         #         ),
#         #         self._get_gpu_cloner_options(),
#         #     )
#         elif self.method == "ivfpq":
#             # this doesn't work
#             # opq_matrix = faiss.OPQMatrix(vector_size, self.code_size)
#             # opq_matrix.niter = 10
#             index = faiss.GpuIndexIVFPQ(
#                 self.gpu_resources,
#                 vector_size,
#                 nlist,
#                 self.code_size,
#                 self.nbits,
#                 metric,
#                 self._set_gpu_index_config(faiss.GpuIndexIVFPQConfig()),
#             )
#             # index = faiss.IndexPreTransform(opq_matrix, index)
#         elif self.method == "ivfsq":
#             index = faiss.GpuIndexIVFScalarQuantizer(
#                 self.gpu_resources,
#                 vector_size,
#                 nlist,
#                 faiss.ScalarQuantizer.QT_4bit,
#                 metric,
#                 True,
#                 self._set_gpu_index_config(faiss.GpuIndexIVFScalarQuantizerConfig()),
#             )
#         else:
#             raise ValueError("Unsupported index type")
#
#         return index
