import copy
import dataclasses
import os

import torch
from accelerate.utils import is_bf16_available
from loguru import logger
from sentence_transformers import SentenceTransformer
from sentence_transformers.models.Dense import Dense
from sentence_transformers.util import is_sentence_transformer_model
from transformers import AutoConfig, AutoModel, PretrainedConfig

from src.model.moderncolbert import (
    get_retriever as _moderncolbert_get_retriever,
)
from src.model.qe import QueryExpansionCL, QueryExpansionMLM
from src.utils import FLASH_ATTENTION_INSTALLED, log_where
from src.utils.func import normalize, padded_slice_with_mask, smooth_max
from src.utils.train import get_parameters_groups

INPUT_TYPES = ["query", "candidate"]

MODES = ["xtr", "colbert"]
METRICS = ["cos", "ip"]


class LogitsScaler(torch.nn.Module):
    def __init__(self, value: float = 0.07, max_value: float = 100.0):
        super().__init__()
        self._value = torch.nn.Parameter(torch.log(torch.tensor(1.0 / value)))
        self.max_value = max_value

    @property
    def value(self) -> torch.Tensor:
        return self._value.exp().clamp(max=self.max_value)


class TokensPooler(torch.nn.Module):
    def __init__(self, smooth: bool = False):
        super().__init__()
        self.smooth = smooth
        if self.smooth:
            self._temperature = torch.nn.Parameter(torch.tensor(0.05))

    @property
    def temperature(self):
        return self._temperature.clamp(max=1)

    def smax(
        self, inputs: torch.Tensor, dim: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert mask is not None, "`mask` cannot be None with `smooth=True`"
        return smooth_max(
            inputs=inputs, dim=dim, temperature=self.temperature, mask=mask
        )

    def pool(
        self, inputs: torch.Tensor, dim: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.smooth:
            return self.smax(inputs=inputs, dim=dim, mask=mask)

        return inputs.max(dim=dim).values


def get_model_kwargs(model_name_or_path: str) -> dict:
    kwargs = {}
    config = AutoConfig.from_pretrained(model_name_or_path)
    if config.model_type == "modernbert" and FLASH_ATTENTION_INSTALLED:
        kwargs.update(
            {
                "attn_implementation": "flash_attention_2",
                "torch_dtype": torch.bfloat16 if is_bf16_available() else torch.float16,
            }
        )
    elif config.model_type in ["bert", "roberta"]:
        kwargs.update({"add_pooling_layer": False})

    return kwargs


def get_model_hidden_size(config: PretrainedConfig):
    hidden_size = getattr(config, "hidden_size")
    if hidden_size is None:
        hidden_size = getattr(config, "d_model")
    if hidden_size is None:
        raise ValueError(f"Cannot determine `hidden_size` of `{config._name_or_path}`")

    return hidden_size


def get_model_from_transformers(model_name_or_path: str, embd_size: int | None = None):
    if embd_size is None:
        embd_size = 128

    model_kwargs = get_model_kwargs(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
    hidden_size = get_model_hidden_size(model.config)

    out = {
        "encoder": model,
        "hidden_size": hidden_size,
        "search_layer": torch.nn.Linear(hidden_size, embd_size, bias=False),
    }

    return out


def get_model_from_sentence_transformers(
    model_name_or_path: str, embd_size: int | None = None
):
    if embd_size is None:
        embd_size = 128

    model_kwargs = get_model_kwargs(model_name_or_path)
    model = SentenceTransformer(model_name_or_path, model_kwargs=model_kwargs)

    encoder = model[0].auto_model
    hidden_size = get_model_hidden_size(encoder.config)

    if isinstance(model[1], Dense):
        parts = []
        activation_function = model[1]._modules.get("activation_function")
        if activation_function is not None and not isinstance(
            activation_function, torch.nn.modules.linear.Identity
        ):
            parts.append(activation_function)
        parts.append(model[1]._modules["linear"])
        search_layer = torch.nn.Sequential(*parts)
        logger.debug(
            "`{}` has pretrained embedding layer: {}", model_name_or_path, search_layer
        )
    else:
        search_layer = torch.nn.Linear(hidden_size, embd_size, bias=False)

    out = {
        "encoder": encoder,
        "hidden_size": hidden_size,
        "search_layer": search_layer,
    }

    return out


@dataclasses.dataclass
class RetrieverConfig:
    model_name_or_path: str
    multivector: bool = True
    multilabel: bool = False  # relevant only in case of multivector=False
    project_size: int = -1
    mode: str = "xtr"
    metric: str = "cos"
    share_weights: bool = True
    loss_weights: dict | None = None

    def __post_init__(self):
        if self.multivector:
            assert self.mode in MODES
        assert self.metric in METRICS


class Encoder(torch.nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        logits_scaler: LogitsScaler | None = None,
        project_size: int = 128,
        share_weights: bool = True,
    ):
        super().__init__()
        if model_name_or_path == "lightonai/GTE-ModernColBERT-v1":
            retriever = _moderncolbert_get_retriever(retriever=model_name_or_path)
        else:
            if is_sentence_transformer_model(model_name_or_path=model_name_or_path):
                retriever = get_model_from_sentence_transformers(
                    model_name_or_path=model_name_or_path,
                    embd_size=project_size if project_size > 0 else None,
                )
            else:
                retriever = get_model_from_transformers(
                    model_name_or_path=model_name_or_path,
                    embd_size=project_size if project_size > 0 else None,
                )

        self.hidden_size = retriever["hidden_size"]
        self.query_encoder = retriever["encoder"]

        self.query_search_layer = None
        self.candidate_search_layer = None

        if project_size > 0:
            self.query_search_layer = retriever["search_layer"]

        if share_weights:
            self.candidate_encoder = self.query_encoder
            if project_size > 0:
                self.candidate_search_layer = self.query_search_layer
        else:
            self.candidate_encoder = copy.deepcopy(self.query_encoder)
            if project_size > 0:
                self.candidate_search_layer = copy.deepcopy(self.query_search_layer)

        self.logits_scaler = logits_scaler

    def forward_hidden_state(
        self,
        input_type: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        assert (
            input_type in INPUT_TYPES
        ), f"Invalid input type `{input_type}`: must be one of {INPUT_TYPES}"

        if not isinstance(input_ids, torch.Tensor) or not isinstance(
            attention_mask, torch.Tensor
        ):
            input_ids = torch.as_tensor(input_ids, device=self.device)
            attention_mask = torch.as_tensor(attention_mask, device=self.device)

        encoder_out = (
            self.query_encoder(input_ids=input_ids, attention_mask=attention_mask)
            if input_type == "query"
            else self.candidate_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
        )
        return encoder_out.last_hidden_state


class BaseRetriever(torch.nn.Module):
    def __init__(
        self,
        config: RetrieverConfig,
        logits_scaler: LogitsScaler | None = None,
    ):
        super().__init__()
        self.config = config
        self.encoder = Encoder(
            model_name_or_path=config.model_name_or_path,
            project_size=config.project_size,
            share_weights=config.share_weights,
            logits_scaler=logits_scaler,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def base_vocab_size(self) -> int:
        return self.encoder.query_encoder.config.vocab_size

    def resize_token_embeddings(self, size: int):
        self.encoder.query_encoder.resize_token_embeddings(size)
        self.encoder.candidate_encoder.resize_token_embeddings(size)

    def forward_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        gather_idxs: torch.Tensor | None = None,
    ):
        if self.encoder.logits_scaler is not None:
            logits = logits * self.encoder.logits_scaler.value

        labels = torch.as_tensor(labels, device=self.device)

        if gather_idxs is not None:
            logits = torch.gather(
                logits, dim=1, index=torch.as_tensor(gather_idxs, device=logits.device)
            )

        if self.config.multilabel:
            probs = torch.softmax(logits, dim=-1)
            losses = probs * labels
            losses = losses.sum(dim=-1)  # sum all positive scores
            losses = losses[losses > 0]  # filter sets with at least one positives
            losses = torch.clamp(losses, min=1e-9, max=1)  # for numerical stability
            losses = -torch.log(losses)  # for negative log likelihood
            loss = losses.sum() if len(losses) == 0 else losses.mean()
        else:
            labels = torch.as_tensor(labels, device=self.device)
            loss = torch.nn.CrossEntropyLoss()(
                logits,
                labels.nonzero()[:, 1],
            )
        return {"loss": loss, "logits": logits}


class SingleVectorRetriever(BaseRetriever):
    def pool(self, batch_vectors: torch.Tensor) -> torch.Tensor:
        return batch_vectors[:, 0, :]

    # def pool_query(
    #     self,
    #     embd: torch.Tensor,
    #     subword_mask: torch.Tensor | None = None,
    # ):
    #     if self.config.multilabel and subword_mask is not None:
    #         embd, _ = padded_slice_with_mask(embd, subword_mask)
    #         return embd.mean(1)
    #     else:
    #         return self.pool(embd)

    def forward_embedding(
        self,
        input_type: str,
        hidden_state: torch.Tensor,
        subword_mask: torch.Tensor | None = None,
    ):
        hidden_state = self.pool(hidden_state)

        if input_type == "query":
            if self.encoder.query_search_layer:
                embed = self.encoder.query_search_layer(hidden_state)
            else:
                embed = hidden_state
        else:
            if self.encoder.candidate_search_layer:
                embed = self.encoder.candidate_search_layer(hidden_state)
            else:
                embed = hidden_state

        # if input_type == "query":
        #     embed = self.pool_query(hidden_state, subword_mask)
        #     if self.query_search_layer:
        #         embed = self.query_search_layer(embed)
        # else:
        #     embed = (
        #         self.candidate_search_layer(hidden_state)
        #         if self.candidate_search_layer
        #         else hidden_state
        #     )
        #     embed = self.pool(embed)

        if self.config.metric == "cos":
            embed = normalize(embed)

        return embed.unsqueeze(1)

    def forward_inputs(
        self,
        input_type: str,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
        subword_mask: torch.Tensor | None = None,
    ):
        if hidden_state is None:
            hidden_state = self.encoder.forward_hidden_state(
                input_type=input_type,
                input_ids=torch.as_tensor(input_ids, device=self.device),
                attention_mask=torch.as_tensor(attention_mask, device=self.device),
            )

        embd = self.forward_embedding(
            input_type=input_type, hidden_state=hidden_state, subword_mask=subword_mask
        )
        return {"embedding": embd, "hidden_state": hidden_state}

    def forward_logits(
        self,
        query_embedding: torch.Tensor,
        candidates_embedding: torch.Tensor,
    ):
        if self.config.metric in ["cos", "ip"]:
            logits = torch.matmul(
                query_embedding.squeeze(1), candidates_embedding.squeeze(1).T
            )
        else:
            raise NotImplementedError(f"Invalid metric {self.config.metric}")

        return {"logits": logits}

    def forward(
        self,
        query_input_ids: torch.Tensor | None = None,
        query_attention_mask: torch.Tensor | None = None,
        query_hidden_state: torch.Tensor | None = None,
        query_embedding: torch.Tensor | None = None,
        query_subword_mask: torch.Tensor | None = None,
        candidates_input_ids: torch.Tensor | None = None,
        candidates_attention_mask: torch.Tensor | None = None,
        candidates_hidden_state: torch.Tensor | None = None,
        candidates_embedding: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ):
        out = {}
        q_inputs = {
            "input_ids": query_input_ids,
            "attention_mask": query_attention_mask,
            "hidden_state": query_hidden_state,
        }
        if (
            q_inputs["input_ids"] is not None and q_inputs["attention_mask"] is not None
        ) or q_inputs["hidden_state"] is not None:
            q_inputs["subword_mask"] = query_subword_mask
            query_out = self.forward_inputs(input_type="query", **q_inputs)
            for k, v in query_out.items():
                out[f"query_{k}"] = v

        c_inputs = {
            "input_ids": candidates_input_ids,
            "attention_mask": candidates_attention_mask,
            "hidden_state": candidates_hidden_state,
        }
        if (
            c_inputs["input_ids"] is not None and c_inputs["attention_mask"] is not None
        ) or c_inputs["hidden_state"] is not None:
            candidates_out = self.forward_inputs(input_type="candidate", **c_inputs)
            for k, v in candidates_out.items():
                out[f"candidates_{k}"] = v

        logits_inputs = {
            "query_embedding": out.get("query_embedding", query_embedding),
            "candidates_embedding": out.get(
                "candidates_embedding", candidates_embedding
            ),
        }
        if all(v is not None for v in logits_inputs.values()):
            out.update(self.forward_logits(**logits_inputs))

        loss_inputs = {"logits": out.get("logits", logits), "labels": labels}
        if all(v is not None for v in loss_inputs.values()):
            out.update(self.forward_loss(**loss_inputs))
        return out


class MultiVectorRetriever(BaseRetriever):
    def __init__(
        self,
        config: RetrieverConfig,
        logits_scaler: LogitsScaler | None = None,
        tokens_pooler: TokensPooler | None = None,
        token_topk_train: int | None = None,
        qe: bool = False,
    ):
        super().__init__(config=config, logits_scaler=logits_scaler)

        self.encoder.tokens_pooler = tokens_pooler
        self.token_topk_train = token_topk_train
        # self.qe = QueryExpansion(config.model_name_or_path) if qe else None
        self.qe = QueryExpansionCL() if qe else None

    def forward_embedding(
        self,
        input_type: str,
        hidden_state: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        if input_type == "query" and self.encoder.query_search_layer:
            embd = self.encoder.query_search_layer(hidden_state)
        elif input_type == "candidate" and self.encoder.candidate_search_layer:
            embd = self.encoder.candidate_search_layer(hidden_state)
        else:
            embd = hidden_state

        if mask is not None:
            embd = embd * mask.unsqueeze(-1)

        if self.config.metric == "cos":
            embd = normalize(embd)

        return embd

    def forward_inputs(
        self,
        input_type: str,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
        subword_mask: torch.Tensor | None = None,
    ):
        out = {}
        if hidden_state is None:
            hidden_state = self.encoder.forward_hidden_state(
                input_type=input_type,
                input_ids=torch.as_tensor(input_ids, device=self.device),
                attention_mask=torch.as_tensor(attention_mask, device=self.device),
            )
            out["hidden_state_"] = hidden_state

        if subword_mask is not None:
            hidden_state, mask = padded_slice_with_mask(
                tensor=hidden_state, mask=subword_mask
            )
            out["hidden_state"] = hidden_state
            out["subword_mask"] = mask
        else:
            mask = attention_mask

        embd = self.forward_embedding(
            input_type=input_type, hidden_state=hidden_state, mask=mask
        )
        out["embedding"] = embd
        return out

    def forward_tokens_logits(
        self,
        query_embedding: torch.Tensor,
        candidates_embedding: torch.Tensor,
    ):
        if self.config.metric in ["cos", "ip"]:
            tokens_logits = torch.einsum(
                "bnh,cmh->bcnm", query_embedding, candidates_embedding
            )
        else:
            raise NotImplementedError(f"Invalid metric {self.config.metric}")

        return {"tokens_logits": tokens_logits}

    def get_min_value(
        self, tokens_logits: torch.Tensor, token_topk_train: int | None = None
    ):
        token_topk_train = (
            token_topk_train if token_topk_train is not None else self.token_topk_train
        )
        assert token_topk_train is not None, "`token_topk_train` cannot be None"

        scores = tokens_logits.flatten()

        if len(scores) < token_topk_train:
            token_topk_train = len(scores)

        out = scores.topk(token_topk_train).values.min()

        return out

    def _xtr_forward_logits(
        self,
        tokens_logits: torch.Tensor,
        query_lengths: torch.Tensor,
        token_topk_train: int | None = None,
        min_value: float | None = None,
    ):
        if min_value is None:
            min_value = self.get_min_value(
                tokens_logits=tokens_logits, token_topk_train=token_topk_train
            )
        topk_mask = tokens_logits >= min_value

        # NOTE: this is wrong
        # logits = (tokens_logits * topk_mask).sum(-1).sum(-1)

        logits = self.encoder.tokens_pooler.pool(
            inputs=tokens_logits * topk_mask, mask=topk_mask, dim=-1
        ).sum(-1)

        logits = logits / query_lengths.unsqueeze(-1)

        # NOTE: from the paper: "candidate scores are normalized by how many tokens would be retrieved by topk"
        # logits = logits / (topk_mask.sum(-1).sum(-1) + 1e-9)
        # NOTE: loss doens't decrease if I do this
        # my hunch is that this is wrong but in the original paper works becuasue documents are long
        # if you take the max a score for a document is the sum of the query-token-scores
        # however `topk_mask` may have multiple query-tokens active, so the document score is reduced
        # disproportionately w.r.t. to the numbert of acutal query-tokens.

        return {"logits": logits, "topk_mask": topk_mask, "min_value": min_value}

    def forward_logits(
        self,
        tokens_logits: torch.Tensor,
        query_lengths: torch.Tensor,
        train: bool,
        token_topk_train: int | None = None,
        min_value: float | None = None,
    ):
        if not train or self.config.mode == "colbert":
            logits = tokens_logits.max(-1).values.sum(-1)
            logits = logits / query_lengths.unsqueeze(-1)
            return {"logits": logits}
        elif train and self.config.mode == "xtr":
            return self._xtr_forward_logits(
                tokens_logits=tokens_logits,
                query_lengths=query_lengths,
                token_topk_train=token_topk_train,
                min_value=min_value,
            )
        else:
            raise NotImplementedError(f"Invalid scoring {self.config.mode}")

    def forward(
        self,
        query_input_ids: torch.Tensor | None = None,
        query_attention_mask: torch.Tensor | None = None,
        query_hidden_state: torch.Tensor | None = None,
        query_embedding: torch.Tensor | None = None,
        query_subword_mask: torch.Tensor | None = None,
        candidates_input_ids: torch.Tensor | None = None,
        candidates_attention_mask: torch.Tensor | None = None,
        candidates_subword_mask: torch.Tensor | None = None,
        candidates_hidden_state: torch.Tensor | None = None,
        candidates_embedding: torch.Tensor | None = None,
        tokens_logits: torch.Tensor | None = None,
        query_lengths: torch.Tensor | None = None,
        train: bool | None = None,
        token_topk_train: int | None = None,
        min_value: float | None = None,
        logits: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        expansion_subword_mask: torch.Tensor | None = None,
        expansion_input_ids: torch.Tensor | None = None,
        gather_idxs: torch.Tensor | None = None,
    ):
        out = {}
        q_inputs = {
            "input_ids": query_input_ids,
            "attention_mask": query_attention_mask,
            "hidden_state": query_hidden_state,
        }
        if (
            q_inputs["input_ids"] is not None and q_inputs["attention_mask"] is not None
        ) or q_inputs["hidden_state"] is not None:
            q_inputs["subword_mask"] = query_subword_mask
            query_out = self.forward_inputs(input_type="query", **q_inputs)
            for k, v in query_out.items():
                out[f"query_{k}"] = v

        c_inputs = {
            "input_ids": candidates_input_ids,
            "attention_mask": candidates_attention_mask,
            "hidden_state": candidates_hidden_state,
        }
        if (
            c_inputs["input_ids"] is not None and c_inputs["attention_mask"] is not None
        ) or c_inputs["hidden_state"] is not None:
            c_inputs["subword_mask"] = candidates_subword_mask
            candidates_out = self.forward_inputs(input_type="candidate", **c_inputs)
            for k, v in candidates_out.items():
                out[f"candidates_{k}"] = v

        tokens_logits_inputs = {
            "query_embedding": out.get("query_embedding", query_embedding),
            "candidates_embedding": out.get(
                "candidates_embedding", candidates_embedding
            ),
        }
        if all(v is not None for v in tokens_logits_inputs.values()):
            out.update(self.forward_tokens_logits(**tokens_logits_inputs))

        logits_inputs = {
            "tokens_logits": out.get("tokens_logits", tokens_logits),
            "train": train,
            "token_topk_train": token_topk_train,
            "min_value": min_value,
        }
        if query_subword_mask is not None:
            logits_inputs["query_lengths"] = torch.as_tensor(
                query_subword_mask.sum(-1), device=self.device
            )
        if logits_inputs.get("tokens_logits") is not None:
            out.update(self.forward_logits(**logits_inputs))

        loss_inputs = {"logits": out.get("logits", logits), "labels": labels}
        if all(v is not None for v in loss_inputs.values()):
            loss_inputs["gather_idxs"] = gather_idxs
            out.update(self.forward_loss(**loss_inputs))

        qe_inputs = {
            "tokens_logits": out.get("tokens_logits"),
            "query_embedding": out.get("query_embedding"),
            "query_subword_mask": query_subword_mask,
            "candidates_input_ids": candidates_input_ids,
            "candidates_subword_mask": candidates_subword_mask,
            "expansion_input_ids": expansion_input_ids,
            "expansion_subword_mask": expansion_subword_mask,
            "labels": labels,
        }
        if self.qe is not None and all(v is not None for v in qe_inputs.values()):
            if self.encoder.logits_scaler is not None:
                qe_inputs["tokens_logits"] = (
                    qe_inputs["tokens_logits"] * self.encoder.logits_scaler.value
                )

            qe_out = self.qe(**qe_inputs)
            for k, v in qe_out.items():
                out[f"qe_{k}"] = v

        if "qe_loss" in out:
            loss_r = out["loss"] * self.config.loss_weights["w"]
            loss_qe = out["qe_loss"] * self.config.loss_weights["w_qe"]
            loss = loss_r + loss_qe
            out["loss"] = loss
            out["ret"] = loss_r.detach()
            out["qe"] = loss_qe.detach()

        return out


def get_model(
    model_name_or_path: str,
    multivector: bool = True,
    multilabel: bool = False,
    project_size: int = 128,
    mode: str = "xtr",
    metric: str = "cos",
    token_topk_train: int | None = None,
    share_weights: bool = True,
    scale_logits: bool = True,
    smooth_pool: bool = False,
    qe: bool = False,
    loss_weights: dict | None = None,
) -> BaseRetriever:
    config = RetrieverConfig(
        model_name_or_path=model_name_or_path,
        share_weights=share_weights,
        project_size=project_size,
        multivector=multivector,
        mode=mode,
        metric=metric,
        multilabel=multilabel,
        loss_weights=loss_weights,
    )
    logits_scaler = (
        LogitsScaler(value=0.07, max_value=100.0)
        if (scale_logits and metric == "cos")
        else None
    )

    if multivector:
        tokens_pooler = TokensPooler(smooth=smooth_pool)
        model = MultiVectorRetriever(
            config=config,
            token_topk_train=token_topk_train,
            logits_scaler=logits_scaler,
            tokens_pooler=tokens_pooler,
            qe=qe,
        )
    else:
        model = SingleVectorRetriever(
            config=config,
            logits_scaler=logits_scaler,
        )
    return model


def get_optimizer(
    model: BaseRetriever,
    lr: float = 3e-6,
    lr_qe: float = 1e-5,
    weight_decay: float = 0.0,
    amsgrad: bool = False,
):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []
    optimizer_grouped_parameters.extend(
        get_parameters_groups(
            model=model.encoder,
            # model=model,
            lr=lr,
            weight_decay=weight_decay,
            no_decay=no_decay,
        )
    )
    if getattr(model, "qe", None) is not None:
        optimizer_grouped_parameters.extend(
            get_parameters_groups(
                model=model.qe,
                lr=lr_qe,
                weight_decay=weight_decay,
                no_decay=no_decay,
            )
        )

    optimizer = torch.optim.AdamW(
        params=optimizer_grouped_parameters, lr=lr, amsgrad=amsgrad
    )
    return optimizer


def load_checkpoint(
    model: BaseRetriever,
    project_dir: str,
    checkpoint: str = "last",
    is_main_process: bool = False,
):
    checkpoint_path = os.path.join(
        project_dir, "checkpoint", checkpoint, "pytorch_model.bin"
    )
    if os.path.exists(checkpoint_path):
        log_where("Load model checkpoint", condition=is_main_process)
        model.load_state_dict(torch.load(checkpoint_path))
    return model
