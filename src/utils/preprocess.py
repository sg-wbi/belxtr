import itertools
import re
import string
import sys
from collections import defaultdict

import nltk
import rapidfuzz
from nltk.corpus import stopwords

# from collections import defaultdict
# import rapidfuzz
# from syntok import segmenter
# from unidecode import unidecode
from src.utils.chars import SPECIAL_CHARS

# def get_stopwords(langs: str | list[str]) -> frozenset[str]:
#     if isinstance(langs, str):
#         langs = [langs]
#     return
#

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

STOPWORDS = frozenset(stopwords.words("english"))


TOKEN_DELIMITER = " "

RE_BRACKETS_JUNK = re.compile(r"[\(|\[\{]\W*[\)\]\}]")

RE_PUNCT = re.compile("|".join(re.escape(p) for p in string.punctuation))

RE_PUNCT_EXTRA = re.compile("|".join(re.escape(p) for p in SPECIAL_CHARS))

RE_WHITESPACES = re.compile(r"(\s)+", re.UNICODE)
ALL_CHARS = (chr(i) for i in range(sys.maxunicode))
CATEGORIES = {"Cc"}
# CONTROL_CHARS = "".join(c for c in ALL_CHARS if unicodedata.category(c) in CATEGORIES)
# or equivalently and much more efficiently
CONTROL_CHARS = "".join(map(chr, itertools.chain(range(0x00, 0x20), range(0x7F, 0xA0))))

CONTROL_CHAR_RE = re.compile("[%s]" % re.escape(CONTROL_CHARS))


def downsample_strings_with_target(
    target: str,
    strings: list[str],
    threshold: float = 0.9,
    scorer: str = "partial_ratio",
) -> list[str]:
    assert 0 < threshold < 1, "threshold must be in [0, 1]"

    assert hasattr(rapidfuzz.fuzz, scorer), f"`rapidfuzz` has no scorer `{scorer}`"
    threshold = threshold * 100
    scorer_fn = getattr(rapidfuzz.fuzz, scorer)
    scores = rapidfuzz.process.cdist(
        queries=[target],
        choices=strings,
        scorer=scorer_fn,
        processor=lambda x: rapidfuzz.utils.default_process(x).replace(" ", ""),
    )[0].tolist()

    return [x for x, s in zip(strings, scores) if s <= threshold]


def downsample_strings_with_clustering(
    strings: list[str],
    threshold: float = 0.8,
    num_strings: int = 10,
    keep_longest: bool = False,
):
    # base case: already short enough
    if len(strings) <= num_strings:
        return strings

    processed_aliases = defaultdict(list)
    for a in strings:
        pa = "".join(rapidfuzz.utils.default_process(a).split())
        processed_aliases[pa].append(a)

    queries = sorted(processed_aliases, key=lambda x: len(x))
    out = rapidfuzz.process.cdist(
        queries=queries, choices=queries, workers=1, scorer=rapidfuzz.fuzz.ratio
    )

    clusters = {}
    assigned = set()
    assert 0 < threshold < 1, "threshold must be in [0, 1]"
    effective_threshold = threshold * 100

    for i in range(out.shape[0]):
        if i in assigned:
            continue
        idxs = [
            j
            for j in (out[i] >= effective_threshold).nonzero()[0].tolist()
            if j not in assigned
        ]
        if idxs:
            clusters[i] = [a for j in idxs for a in processed_aliases[queries[j]]]
        assigned.update(idxs)

    if not clusters:  # safeguard: no clusters formed
        return sorted(strings)[:num_strings]

    # Deterministic representative: shortest string in each cluster
    fn = max if keep_longest else min
    new_strings = [fn(cluster, key=len) for cluster in clusters.values()]

    # If clustering did not reduce the number of strings, lower threshold
    if len(new_strings) == len(strings) and threshold > 0.1:
        return downsample_strings_with_clustering(
            new_strings, threshold=threshold - 0.05, num_strings=num_strings
        )

    return downsample_strings_with_clustering(
        new_strings, threshold=threshold, num_strings=num_strings
    )
