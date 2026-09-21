import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM

from src.utils.func import padded_slice_with_mask


def get_mlm_head(model_name_or_path: str):
    model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
    model_type = model.config.model_type
    out = {}
    if model_type == "bert":
        out["head"] = model.cls.predictions.transform
        out["decoder"] = model.cls.predictions.decoder
    elif model_type == "modernbert":
        out["head"] = model.head
        out["decoder"] = model.decoder
    else:
        raise ValueError(
            f"Cannot determine language modeling head of model type `{model_type}` (`{model_name_or_path}`)"
        )

    return out


class PermutationInvariantLMLoss(torch.nn.Module):
    def __init__(self, ignore_index=-100, eps=1e-8):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits, targets):
        """
        Args:
            logits:  Tensor of shape (batch_size, num_masks, vocab_size)
                     Supports any dynamic number of [MASK] tokens.
            targets: Tensor of shape (batch_size, num_targets)
                     Contains target token IDs, padded with `ignore_index` (-100).
        """
        batch_size, num_masks, vocab_size = logits.shape
        _, num_targets = targets.shape

        # 1. Identify valid elements vs padding tokens
        valid_mask = targets != self.ignore_index  # Shape: (batch_size, num_targets)

        # 2. Neutralize padding for torch.gather execution
        # Replace -100 with 0. The probability calculated for index 0 will be ignored later.
        safe_targets = torch.where(valid_mask, targets, torch.zeros_like(targets))

        # 3. Compute vocabulary probability distributions
        probs = F.softmax(logits, dim=-1)  # Shape: (batch_size, num_masks, vocab_size)

        # 4. Extract token probabilities for all targets across all slots
        # Expand targets from (B, T) -> (B, 1, T) -> (B, M, T)
        expanded_targets = safe_targets.unsqueeze(1).expand(-1, num_masks, -1)

        # Gather probabilities from the vocab axis (dim=2)
        # target_probs shape: (batch_size, num_masks, num_targets)
        target_probs = torch.gather(probs, 2, expanded_targets)

        # 5. Permutation-Invariant Pooling
        # Max-pool along the mask slot axis (dim=1).
        # "Which mask did the best job of capturing this target?"
        # Shape: (batch_size, num_targets)
        pooled_target_probs, _ = torch.max(target_probs, dim=1)

        # With a smooth approximation (LogSumExp along the mask dimension):
        # This allows all slots to learn simultaneously
        # pooled_target_probs = torch.logsumexp(target_probs, dim=1)

        # pooled_target_probs = smooth_max(
        #     target_probs,
        #     dim=1,
        #     temperature=self.temperature,
        #     mask=valid_mask.unsqueeze(1),
        # )

        # 6. Apply Negative Log Likelihood
        loss_elements = -torch.log(
            pooled_target_probs + self.eps
        )  # Shape: (batch_size, num_targets)

        # 7. Mask out the losses belonging to padded (-100) labels
        masked_loss = loss_elements * valid_mask.float()

        # 8. Average the loss across actual valid labels only
        total_valid_tokens = valid_mask.sum().clamp(min=1)
        return masked_loss.sum() / total_valid_tokens


class QueryExpansionMLM(torch.nn.Module):
    def __init__(self, model_name_or_path: str, head: bool = True):
        super().__init__()
        params = get_mlm_head(model_name_or_path)
        self.head = params["head"] if head else None
        self.decoder = params["decoder"]
        self.loss = PermutationInvariantLMLoss()

    def forward(
        self,
        hidden_state: torch.Tensor,
        subword_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden_state, _ = padded_slice_with_mask(hidden_state, subword_mask)

        out = {}
        if self.head is not None:
            hidden_state = self.head(hidden_state)

        logits = self.decoder(hidden_state)
        out["logits"] = logits

        if labels is not None:
            labels = torch.as_tensor(labels, device=logits.device)
            loss = self.loss(logits=logits, targets=labels)
            # loss = torch.nn.CrossEntropyLoss()(logits, labels)
            out["loss"] = loss

        return out
