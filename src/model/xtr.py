import json
import os

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, PretrainedConfig, T5EncoderModel
from transformers.modeling_utils import PreTrainedModel


def load_file_path(model_name_or_path: str, filename: str) -> str | None:
    # If file is local
    file_path = os.path.join(model_name_or_path, filename)
    if os.path.exists(file_path):
        return file_path

    # If file is remote
    try:
        return hf_hub_download(
            model_name_or_path,
            filename=filename,
        )
    except Exception:
        return None


class XtrConfig(PretrainedConfig):
    """Model config."""

    model_type = "xtr"

    def __init__(self, model_name_or_path=None):
        t5_config = AutoConfig.from_pretrained(model_name_or_path)

        # overwrite config
        self.__class__ = t5_config.__class__
        self.__dict__ = t5_config.__dict__

        # load config for linear layer
        linear_config_path = load_file_path(model_name_or_path, "2_Dense/config.json")
        with open(linear_config_path) as f:
            self.__dict__.update(json.load(f))


class XtrPreTrainedModel(PreTrainedModel):
    """
    A simple interface for downloading and loading pretrained models.
    """

    config_class = XtrConfig  # type: ignore
    base_model_prefix = "xtr"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]  # type: ignore


class XtrModel(XtrPreTrainedModel):
    """ """

    def __init__(self, model_name_or_path: str):
        config = XtrConfig(model_name_or_path)
        super().__init__(config)
        self.config = config
        self.encoder = self._load_encoder(model_name_or_path)
        self.linear_layer = self._load_linear_layer(model_name_or_path)

    def _load_encoder(self, model_name_or_path=None):
        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]

        t5_encoder = T5EncoderModel.from_pretrained(
            model_name_or_path, use_safetensors=True
        )

        return t5_encoder

    def _load_linear_layer(self, model_name_or_path=None):
        linear_path = load_file_path(model_name_or_path, "2_Dense/pytorch_model.bin")
        linear_weight = torch.load(linear_path)
        linear_layer = torch.nn.Linear(
            self.config.in_features, self.config.out_features
        )
        linear_layer.weight = torch.nn.Parameter(linear_weight["linear.weight"])
        linear_layer.bias = torch.nn.Parameter(torch.zeros(self.config.out_features))
        return linear_layer

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
    ):
        def pass_through(model_output, attention_mask):
            token_embeddings = model_output[
                0
            ]  # First element of model_output contains all token embeddings
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            return token_embeddings * input_mask_expanded

        model_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Perform pooling
        embeddings = pass_through(model_output, attention_mask)

        # Apply linear layer
        embeddings = self.linear_layer(embeddings)

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=2)

        return embeddings


def get_retriever(model_name_or_path: str) -> dict:
    model = XtrModel(model_name_or_path=model_name_or_path)

    return {
        "encoder": model.encoder,
        "search_layer": model.linear_layer,
        "hidden_size": model.config.d_model,
    }


##################################
# GRAVEYARD
##################################

#
#
# def merge_masks(retriever_mask: np.ndarray, reranking_mask: np.ndarray):
#     a = np.empty((retriever_mask.shape[0], retriever_mask.sum(-1).max()))
#     a.fill(-1)
#     rows, columns = retriever_mask.nonzero()
#     for i in range(retriever_mask.shape[0]):
#         js = columns[rows == i]
#         a[i, : len(js)] = js
#
#     b = np.empty((reranking_mask.shape[0], reranking_mask.sum(-1).max()))
#     b.fill(-2)
#     rows, columns = reranking_mask.nonzero()
#     for i in range(reranking_mask.shape[0]):
#         js = columns[rows == i]
#         b[i, : len(js)] = js
#
#     # https://stackoverflow.com/a/67870684 (see comment)
#     out = (a[:, :, None] == b[:, None, :]).any(-1).astype(int)
#
#     return out
#
# def _forward_compare_pooled(
#     self,
#     query: torch.Tensor,
#     candidates: torch.Tensor,
#     candidates_indices: torch.Tensor | None = None,
#     candidates_lengths: torch.Tensor | None = None,
#     candidates_attention_mask: torch.Tensor | None = None,
# ):
#     """Compare (via attention) every token in the query with the `pooled` representation of every candidate."""
#     # candidates_hidden_state.shape != candidates_attention_mask.shape if candidates_indices is not None
#     if candidates_indices is not None:
#         assert (
#             candidates_lengths is not None
#         ), "`candidates_lengths` cannot be `None` if `candidates_indices` is not None`"
#         mask = self._get_mask_from_lengths(candidates_lengths).to(candidates.device)
#     else:
#         assert (
#             candidates_attention_mask is not None
#         ), "`candidates_attention_mask` cannot be `None` if `candidates_indices` is None`"
#         mask = candidates_attention_mask
#
#     values = self.pool(hidden_state=candidates, attention_mask=mask)
#
#     batch_size, num_tokens, hidden_size = query.shape
#     assert (
#         hidden_size == values.shape[-1]
#     ), "`query.shape[-1]` must match `values.shape[-1]` (embedding size)"
#     packed = torch.cat(
#         [
#             query.unsqueeze(1).view(-1, 1, hidden_size),
#             values.unsqueeze(0).expand(batch_size * num_tokens, -1, -1),
#         ],
#         dim=1,
#     )
#
#     out = self.attention_layer(packed)
#
#     # first item in the sequence dimension is the query token
#     query_compare = out[:, 0, :].reshape(query.shape)
#     candidates_compare = out[:, 1:, :].reshape(candidates.shape)
#
#     return query_compare, candidates_compare
#
# def forward_compare(
#     self,
#     query: torch.Tensor,
#     candidates: torch.Tensor,
#     candidates_indices: torch.Tensor | None = None,
#     candidates_lengths: torch.Tensor | None = None,
#     candidates_attention_mask: torch.Tensor | None = None,
#     token_scores: torch.Tensor | None = None,
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     if self.reranking == "pooled":
#         query_compare, candidates_compare = self._forward_compare_pooled(
#             query=query,
#             candidates=candidates,
#             candidates_indices=candidates_indices,
#             candidates_lengths=candidates_lengths,
#             candidates_attention_mask=candidates_attention_mask,
#         )
#     elif self.reranking == "tokens":
#         query_compare, candidates_compare = self._forward_compare_tokens(
#             query=query, candidates=candidates, token_scores=token_scores
#         )
#
#     return query_compare, candidates_compare
