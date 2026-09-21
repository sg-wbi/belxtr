# ruff: noqa: F401
import dataclasses
import importlib.resources as pkg_resources
import os
import time
from collections import deque
from collections.abc import Mapping
from functools import wraps
from importlib.abc import Traversable
from itertools import chain, islice

import datasets
import git
from omegaconf import DictConfig, OmegaConf

from .bm25 import BM25Scorer
from .chars import SPECIAL_CHARS
from .preprocess import (
    RE_BRACKETS_JUNK,
    RE_PUNCT,
    RE_PUNCT_EXTRA,
    RE_WHITESPACES,
    TOKEN_DELIMITER,
)
from .train import (
    IndexRefreshScheduler,
    SaveBestCallback,
    batch_to_device,
    get_gradients_norm,
    get_lr_scheduler,
    get_num_gradient_updates,
    get_optimizer,
    log_where,
)

ASSETS: Traversable = pkg_resources.files("src") / "assets"


GIT_REPOSITORY = git.Repo(search_parent_directories=True)

try:
    import flash_attn  # noqa:  F401

    FLASH_ATTENTION_INSTALLED = True
except Exception:
    FLASH_ATTENTION_INSTALLED = False


def get_cache_dir(path: str, add_branch: bool = False):
    if add_branch:
        return os.path.join(path, GIT_REPOSITORY.active_branch.name)
    else:
        return path


def recursive_flatten_list(lst: list):
    return [
        item
        for sublist in lst
        for item in (
            recursive_flatten_list(sublist) if isinstance(sublist, list) else [sublist]
        )
    ]


def flatten_cfg(cfg: DictConfig) -> DictConfig:
    """
    Flattens a nested DictConfig so that all leaf keys are promoted to the top level.

    Example:
        foo:
          bar:
            value: 10
          baz:
            other: 20

    → becomes

        value: 10
        other: 20

    If duplicate leaf keys are found, raises a ValueError.
    """
    flat = {}

    def _collect(d: Mapping):
        for k, v in d.items():
            if isinstance(v, Mapping):
                _collect(v)
            else:
                if k in flat:
                    raise ValueError(f"Duplicate key found during flattening: '{k}'")
                flat[k] = v

    _collect(OmegaConf.to_container(cfg, resolve=True))
    return OmegaConf.create(flat)


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function `{func.__name__}` took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


def get_hps_as_dict(cfg: DictConfig) -> dict:
    hps = OmegaConf.to_container(cfg.hps)
    for k, v in hps.items():
        if isinstance(v, list):
            hps[k] = ",".join([str(x) for x in v])

    commit = GIT_REPOSITORY.head.object.hexsha
    branch = GIT_REPOSITORY.active_branch.name

    # if repo.is_dirty():
    #     raise RuntimeError("That's not how we do things here son: commit your changes")
    hps["commit"] = commit
    hps["branch"] = branch

    return hps  # type: ignore


def get_project_dir(
    cfg: DictConfig,
    subdir: str | None = None,
    run: str | None = None,
) -> str:
    hps = get_hps_as_dict(cfg)

    path = [cfg.project_dir]
    if subdir is not None:
        path.append(subdir)

    if run is not None:
        path.append(run)
    else:
        path.append(datasets.fingerprint.Hasher.hash(hps))

    project_dir = os.path.join(*path)
    os.makedirs(project_dir, exist_ok=True)

    cfg.hps.commit = hps["commit"]
    cfg.hps.branch = hps["branch"]

    OmegaConf.save(config=cfg, f=os.path.join(project_dir, "conf.yaml"))

    return project_dir


def windowed(seq, n, fillvalue=None, step=1):
    """Return a sliding window of width *n* over the given iterable.

        >>> all_windows = windowed([1, 2, 3, 4, 5], 3)
        >>> list(all_windows)
        [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

    When the window is larger than the iterable, *fillvalue* is used in place
    of missing values:

        >>> list(windowed([1, 2, 3], 4))
        [(1, 2, 3, None)]

    Each window will advance in increments of *step*:

        >>> list(windowed([1, 2, 3, 4, 5, 6], 3, fillvalue='!', step=2))
        [(1, 2, 3), (3, 4, 5), (5, 6, '!')]

    To slide into the iterable's items, use :func:`chain` to add filler items
    to the left:

        >>> iterable = [1, 2, 3, 4]
        >>> n = 3
        >>> padding = [None] * (n - 1)
        >>> list(windowed(chain(padding, iterable), 3))
        [(None, None, 1), (None, 1, 2), (1, 2, 3), (2, 3, 4)]
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if n == 0:
        yield ()
        return
    if step < 1:
        raise ValueError("step must be >= 1")

    iterable = iter(seq)

    # Generate first window
    window = deque(islice(iterable, n), maxlen=n)

    # Deal with the first window not being full
    if not window:
        return
    if len(window) < n:
        yield tuple(window) + ((fillvalue,) * (n - len(window)))
        return
    yield tuple(window)

    # Create the filler for the next windows. The padding ensures
    # we have just enough elements to fill the last window.
    padding = (fillvalue,) * (n - 1 if step >= n else step - 1)
    filler = map(window.append, chain(iterable, padding))

    # Generate the rest of the windows
    for _ in islice(filler, step - 1, None, step):
        yield tuple(window)
