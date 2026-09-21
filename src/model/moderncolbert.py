import json
import os

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate.utils import is_bf16_available
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoConfig, ModernBertModel, PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from src.utils import FLASH_ATTENTION_INSTALLED


def load_file_path(retriever: str, filename: str) -> str | None:
    # If file is local
    file_path = os.path.join(retriever, filename)
    if os.path.exists(file_path):
        return file_path

    # If file is remote
    try:
        return hf_hub_download(
            retriever,
            filename=filename,
        )
    except Exception:
        return None


class ModernColBertConfig(PretrainedConfig):
    """Model config."""

    model_type = "modernbert"

    def __init__(self, retriever=None):
        modernbert_config = AutoConfig.from_pretrained(retriever)

        # overwrite config
        self.__class__ = modernbert_config.__class__
        self.__dict__ = modernbert_config.__dict__

        # load config for linear layer
        linear_config_path = load_file_path(retriever, "1_Dense/config.json")
        with open(linear_config_path) as f:
            self.__dict__.update(json.load(f))


class ModernColBertPreTrainedModel(PreTrainedModel):
    """
    A simple interface for downloading and loading pretrained models.
    """

    config_class = ModernColBertConfig  # type: ignore
    base_model_prefix = "modernbert"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]  # type: ignore


class ModernColBertModel(ModernColBertPreTrainedModel):
    """ """

    def __init__(self, retriever: str):
        config = ModernColBertConfig(retriever)
        super().__init__(config)
        self.config = config
        self.encoder = self._load_encoder(retriever)
        self.linear_layer = self._load_linear_layer(retriever)

    def _load_encoder(self, retriever=None):
        kwargs = {"use_safetensors": True}
        if FLASH_ATTENTION_INSTALLED:
            kwargs.update(
                {
                    "attn_implementation": "flash_attention_2",
                    "torch_dtype": torch.bfloat16
                    if is_bf16_available()
                    else torch.float16,
                }
            )

        encoder = ModernBertModel.from_pretrained(retriever, **kwargs)

        return encoder

    def _load_linear_layer(self, retriever=None):
        linear_path = load_file_path(retriever, "1_Dense/model.safetensors")
        linear_weight = load_file(linear_path)
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


def get_retriever(retriever: str) -> dict:
    model = ModernColBertModel(retriever=retriever)

    return {
        "encoder": model.encoder,
        "search_layer": model.linear_layer,
        "hidden_size": model.config.hidden_size,
    }
