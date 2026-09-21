from dataclasses import dataclass

# from src.index.search import ANN_TYPES
from src.model.retriever import METRICS

# BITS_PER_CODE: int = 8
# CHUNK_SPLIT: int = 3

# https://github.com/facebookresearch/faiss/wiki/Faiss-indexes#indexes-based-on-product-quantization-codes
SUPPORTED_NBITS = [8, 12, 16]

ANN_TYPES = [
    "ivfflat",
    "ivfpq",
    "ivfsq",
    # "flat", # `add_with_ids` not supported
    # "pq", # `add_with_ids` not supported
]


@dataclass
class IndexConfig:
    metric: str = "cos"
    multivector: bool = True  # relevant only in case of multivector=False

    pooling: str = "cls"
    ann: str | None = None
    code_size: int = 64  # PQ options: number of sub-vectors to split original ones into
    # opq: bool = False # Beneficial for unbalanced vectors with uneven data distributions.
    nlist: int | None = (
        None  # IVF options: how many Voronoi cells (must be >= k* which is 2**n_bits when combined w/ PQ)
    )
    nprobe: int | None = None
    nbits = 8  # the number of centroids assigned to each subspace as k_ = 2**n_bits:  An n_bits of 11 leaves us with 2048 centroids per subspace.
    # probe_factor: int = 1
    use_cuvs: bool = False
    force_cpu: bool = False

    batch_size: int = 1024

    def __post_init__(self):
        if self.metric not in METRICS:
            raise ValueError(
                f"Invalid similarity metric `{self.metric}`, must be one of {tuple(METRICS)}"
            )

        if self.ann is not None and self.ann not in ANN_TYPES:
            raise ValueError(
                f"Invalid index type {self.ann}: must be `None` or one of `{tuple(ANN_TYPES)}`"
            )

        if self.nbits not in SUPPORTED_NBITS:
            raise ValueError(f"`nbits` must be one of {tuple(SUPPORTED_NBITS)}")
