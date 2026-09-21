import torch
import torch.nn.functional as F


def normalize(inputs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(inputs, p=2, dim=dim)


def padded_slice_with_mask(
    tensor: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    assert (
        tuple(mask.shape) == tuple(tensor.shape[:2])
    ), f"`tensor` and `mask` must have equal first two dimension: found {tensor.shape} and {mask.shape}"

    B, S = mask.shape
    device = tensor.device

    mask_t = torch.as_tensor(mask, device=device, dtype=torch.bool)
    lengths = mask_t.sum(dim=1)  # [B]
    Lmax = int(lengths.max().item())
    if Lmax == 0:
        return torch.zeros(
            (B, 0, *tensor.shape[2:]), dtype=tensor.dtype, device=device
        ), torch.zeros((B, 0), dtype=torch.bool, device=device)

    # Flatten trailing dims
    t_flat = tensor.view(B, S, -1)  # [B, S, F]
    F = t_flat.shape[-1]

    # Position of each True within its row
    pos = mask_t.cumsum(dim=1) - 1  # [B, S]

    # Prepare output tensors
    out_flat = torch.zeros((B, Lmax, F), dtype=tensor.dtype, device=device)
    updated_mask = torch.zeros((B, Lmax), dtype=torch.int32, device=device)

    # Indices of all True elements
    b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, S)[mask_t]  # [Ntrue]
    p_idx = pos[mask_t]  # [Ntrue]
    src = t_flat[mask_t]  # [Ntrue, F]

    # Scatter values
    out_flat[b_idx, p_idx] = src
    updated_mask[b_idx, p_idx] = 1

    # Reshape back if original tensor had trailing dims
    sliced = out_flat.view(B, Lmax, *tensor.shape[2:])
    return sliced, updated_mask


def masked_softmax(inputs: torch.Tensor, mask: torch.Tensor, dim: int):
    try:
        torch.broadcast_shapes(inputs.shape, mask.shape)
    except RuntimeError as e:
        raise ValueError(
            f"Mask shape {mask.shape} cannot broadcast to inputs shape {inputs.shape}"
        ) from e

    masked_logits = inputs.masked_fill(~mask, float("-inf"))
    probs = F.softmax(masked_logits, dim=dim)
    probs = torch.nan_to_num(probs, nan=0.0)
    return probs


def smooth_max(
    inputs: torch.Tensor,
    dim: int,
    temperature: torch.Tensor | float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    weights = masked_softmax(inputs=inputs / temperature, mask=mask, dim=dim)
    return (inputs * weights).sum(dim)
