from functools import cached_property

import numpy as np
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from src.utils.chars import HYPHENS, PUNCT, QUOTES

BERT_BASED_MODELS = [
    "lightonai/colbertv2.0",
    "google-bert/bert-base-uncased",
    "jhu-clsp/ettin-encoder-17m",
    "jhu-clsp/ettin-encoder-32m",
    "jhu-clsp/ettin-encoder-68m",
    "jhu-clsp/ettin-encoder-150m",
]

MODEL_TO_TOKENS = {model: ("[unused0]", "[unused1]") for model in BERT_BASED_MODELS}

MODEL_MAX_LEN = {"microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext": 512}


def _get_special_tokens(tokenizer: PreTrainedTokenizerBase):
    model_name_or_path = tokenizer.name_or_path

    config = AutoConfig.from_pretrained(model_name_or_path)
    additional_special_tokens = []
    if config.model_type in ["bert", "modernbert"]:
        fallback_query_start_token = "[unused0]"
        fallback_candidate_token = "[unused1]"
        # https://github.com/huggingface/transformers/issues/4683

        additional_special_tokens = sorted(
            [k for k in tokenizer.vocab if k.startswith("[unused")]
        )
        if len(additional_special_tokens) == 0:
            additional_special_tokens = [f"[unused{i}]" for i in range(10)]
    else:
        raise ValueError(
            f"Cannot determine tokenizer metadata for model type `{config.model_type}` (inferred from `{model_name_or_path}`)"
        )

    try:
        query_start_token, candidate_start_token = MODEL_TO_TOKENS[model_name_or_path]
    except KeyError:
        query_start_token = fallback_query_start_token
        candidate_start_token = fallback_candidate_token
        # logger.warning(
        #     "Could not determine `query` and `document` token prefix for model `{}`: use default `{}` (Q) and `{}` (D)",
        #     model_name_or_path,
        #     query_prefix,
        #     document_prefix,
        # )

    special_tokens = {
        "candidate_start_token": candidate_start_token,
        "query_start_token": query_start_token,
    }
    reserved_special_tokens = {v for v in special_tokens.values() if v is not None}
    available_special_tokens = set(additional_special_tokens) - reserved_special_tokens
    available_special_tokens = sorted(available_special_tokens)
    special_tokens["query_end_token"] = next(iter(available_special_tokens))
    available_special_tokens.remove(special_tokens["query_end_token"])
    special_tokens["available_special_tokens"] = available_special_tokens

    return special_tokens


class RetrieverTokenizer:
    """Wrapper that adds custom attributes with type safety"""

    def __init__(
        self,
        base_tokenizer: PreTrainedTokenizerBase,
        query_start_token: str,
        query_end_token: str,
        candidate_start_token: str,
        available_special_tokens: list[str],
    ):
        # Don't call super().__init__() - we're wrapping, not inheriting behavior
        self._tokenizer = base_tokenizer
        # Assign all custom attributes
        self.query_start_token = query_start_token
        self.query_end_token = query_end_token
        self.candidate_start_token = candidate_start_token
        self.available_special_tokens = available_special_tokens

    @cached_property
    def bos_token_id(self) -> int:
        return self._tokenizer.bos_token_id or self._tokenizer.cls_token_id

    @cached_property
    def eos_token_id(self) -> int:
        return self._tokenizer.eos_token_id or self._tokenizer.sep_token_id

    @cached_property
    def unk_token_id(self) -> int:
        return self._tokenizer.vocab["[UNK]"]

    @cached_property
    def mask_token_id(self) -> int:
        return self._tokenizer.vocab["[MASK]"]

    @cached_property
    def space_token_id(self) -> int | None:
        # for T5 model
        return self._tokenizer.vocab.get("▁")

    @cached_property
    def query_start_token_id(self) -> int:
        return self._tokenizer.vocab[self.query_start_token]

    @cached_property
    def query_end_token_id(self) -> int:
        return self._tokenizer.vocab[self.query_end_token]

    @cached_property
    def candidate_start_token_id(self) -> int:
        return self._tokenizer.vocab[self.candidate_start_token]

    @cached_property
    def subset_special_ids(self) -> list[int]:
        tok = self._tokenizer
        subset_special_ids = tok.all_special_ids.copy()
        if self.space_token_id is not None:
            subset_special_ids.append(self.space_token_id)
        if "[MASK]" in subset_special_ids:
            subset_special_ids.remove("[MASK]")

        return subset_special_ids

    @cached_property
    def punctuation_ids(self) -> list[int]:
        punctuations = PUNCT | HYPHENS | QUOTES
        return list(
            set(
                i
                for p in punctuations
                for i in self._tokenizer.encode(p, add_special_tokens=False)
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name: str, *args, **kwargs):
        bt = AutoTokenizer.from_pretrained(pretrained_model_name, *args, **kwargs)

        if pretrained_model_name in MODEL_MAX_LEN:
            bt.model_max_length = MODEL_MAX_LEN[pretrained_model_name]

        kwargs = _get_special_tokens(bt)

        new_tokens = []
        for v in kwargs.values():
            if isinstance(v, str):
                new_tokens.append(v)
            else:
                new_tokens.extend(v)
        bt.add_tokens(new_tokens)

        kwargs["base_tokenizer"] = bt
        return cls(**kwargs)

    def get_query_subword_mask(
        self,
        base_masks: dict[str, np.ndarray],
        input_ids: np.ndarray,
        cls_: bool,
        prefix: bool,
        punctuation: bool = True,
        first_subword: bool = False,
    ):
        base_mask = base_masks["query"].copy()

        skiplist = self.subset_special_ids.copy()

        base_mask[input_ids == self.mask_token_id] = 1
        skiplist.remove(self.mask_token_id)

        if cls_:
            base_mask[:, 0] = 1
            skiplist.remove(self.bos_token_id)
        if prefix:
            base_mask[input_ids == prefix] = 1
            skiplist.remove(self.query_start_token_id)
        special_ids_mask = self._build_from_skiplist(array=input_ids, skiplist=skiplist)

        masks = [base_mask, special_ids_mask]
        if not punctuation:
            punctuation_mask = self._build_from_skiplist(
                array=input_ids, skiplist=self.punctuation_ids
            )
            masks.append(punctuation_mask)

        return np.all(masks, axis=0).astype(int)

    def get_expansion_subword_mask(self, input_ids: np.ndarray):
        return (input_ids == self.tokenizer.mask_token_id).astype(int)

    def get_candidates_subword_mask(
        self,
        base_masks: dict[str, np.ndarray],
        input_ids: np.ndarray,
        # description: bool,
        cls_: bool,
        prefix: bool,
        aliases: bool = True,
        punctuation: bool = False,
        # first_only: bool | None = None,
    ):
        # first_only = first_only or self.first_only

        base_mask = self._get_candidate_base_mask(
            base_masks=base_masks,
            # description=description,
            aliases=aliases,
        )

        skiplist = self.subset_special_ids.copy()
        if cls_:
            base_mask[:, 0] = 1
            skiplist.remove(self.bos_token_id)
        if prefix:
            base_mask[:, 1] = 1
            skiplist.remove(self.candidate_start_token_id)

        special_ids_mask = self._build_from_skiplist(array=input_ids, skiplist=skiplist)

        masks = [base_mask, special_ids_mask]
        if not punctuation:
            punctuation_mask = self._build_from_skiplist(
                array=input_ids, skiplist=self.punctuation_ids
            )
            # # keep punctuation in title
            # # 124330:{"wikipedia_id": 600744, "wikipedia_title": "!!!", "description": "!!!"}
            # # keep always the emebddings for title even if theyr are puncutation
            # # so we don't end up with empty embeddings
            # puncutation_title_mask = np.any(
            #     [base_masks["label"], punctuation_mask], axis=0
            # ).astype(int)
            # masks.append(puncutation_title_mask)
            masks.append(punctuation_mask)

        return np.all(masks, axis=0).astype(int)

    def _build_from_skiplist(
        self, array: np.ndarray, skiplist: list[int]
    ) -> np.ndarray:
        assert array.ndim == 2
        return np.asarray([[int(x not in skiplist) for x in a] for a in array.tolist()])

    def _get_candidate_base_mask(
        self,
        base_masks: dict[str, np.ndarray],
        # description: bool,
        aliases: bool,
    ):
        base_mask = base_masks["label"].copy()
        # try:
        #     if description:
        #         base_mask += base_masks["description"]
        # except KeyError:
        #     raise KeyError(
        #         "`description` wasn't included into the entity representation"
        #     )

        try:
            if aliases:
                base_mask += base_masks["aliases"]
        except KeyError:
            raise KeyError("`aliases` wasn't included into the entity representation")

        return base_mask

    def __getattr__(self, name: str):
        # Delegate everything else to the wrapped tokenizer
        return getattr(self._tokenizer, name)

    def __call__(self, *args, **kwargs):
        return self._tokenizer(*args, **kwargs)

    def __len__(self):
        return len(self._tokenizer)

    def __reduce__(self):
        """Make the tokenizer picklable for datasets.map"""
        return (
            self.__class__,
            (
                self._tokenizer,
                self.query_start_token,
                self.query_end_token,
                self.candidate_start_token,
                self.available_special_tokens,
            ),
        )
