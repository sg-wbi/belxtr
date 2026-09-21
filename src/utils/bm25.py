import math
from collections.abc import Iterator

from tqdm import tqdm


class BM25Scorer:
    """
    Modified from:
        - https://github.com/HITsz-TMG/Sparse-Retrieval-Fewshot-EL/blob/main/bm25.py
        - https://aclanthology.org/2023.emnlp-main.789/

    Implementation of Best Matching 25 ranking function.

    Attributes
    ----------
    corpus_size : int
        Size of corpus (number of documents).
    avgdl : float
        Average length of document in `corpus`.
    doc_freqs : list of dicts of int
        Dictionary with terms frequencies for each document in `corpus`. Words used as keys and frequencies as values.
    idf : dict
        Dictionary with inversed documents frequencies for whole `corpus`. Words used as keys and frequencies as values.
    doc_len : list of int
        List of document lengths.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        corpus_size: int | None = None,
    ):
        """
        Parameters
        ----------
        corpus : list of list of str
            Given corpus.

        """

        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = corpus_size if corpus_size is not None else 0
        self.avgdl = 0.0
        self.idf = {}
        self.doc_freqs = {}
        self.doc_len = {}

    def initialize(self, corpus: Iterator[dict], tokens_column: str = "tokens"):
        """Calculates frequencies of terms in documents and in corpus. Also computes inverse document frequencies."""

        nd = {}  # word -> number of documents with word
        num_doc = 0
        count_corpus = self.corpus_size == 0
        for row in tqdm(
            corpus,
            desc="Initialize BM25",
            total=self.corpus_size if not count_corpus else None,
        ):
            if count_corpus:
                self.corpus_size += 1
            try:
                idx = row["idx"]
            except KeyError:
                raise KeyError("`corpus` must contain `dict`s with `idx` key")

            try:
                document = row[tokens_column]
            except KeyError:
                raise KeyError(
                    f"`corpus` must contain `dict`s with `{tokens_column}` key"
                )

            self.doc_len[idx] = len(document)
            num_doc += len(document)
            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs[idx] = frequencies

            for word, freq in frequencies.items():
                if word not in nd:
                    nd[word] = 0
                nd[word] += 1

        self.avgdl = num_doc / self.corpus_size
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0.0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = float(idf_sum) / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_tokens_scores(self, tokens: list[str], index: int):
        tokens_scores = {}
        doc_freqs = self.doc_freqs[index]
        for token in tokens:
            if token not in doc_freqs:
                continue
            score = (
                self.idf[token]
                * doc_freqs[token]
                * (self.k1 + 1)
                / (
                    doc_freqs[token]
                    + self.k1 * (1 - self.b + self.b * self.doc_len[index] / self.avgdl)
                )
            )
            # score = self.idf[word] * doc_freqs[word]
            if token not in tokens_scores:
                tokens_scores[token] = score
            else:
                tokens_scores[token] = max(tokens_scores[token], score)
        word_score_tuples = [(word, score) for word, score in tokens_scores.items()]
        word_score_tuples = sorted(word_score_tuples, key=lambda x: x[1], reverse=True)
        return word_score_tuples
