import numpy as np
import torch
import torch.nn.functional as F


def get_relative_expansion_idxs(
    query_subword_mask: torch.Tensor, expansion_subword_mask: torch.Tensor
):
    max_lens = query_subword_mask.sum(-1)

    lengths = expansion_subword_mask.sum(-1)

    idxs = []
    for i, length in enumerate(lengths):
        idxs.append(list(range(max_lens[i] - length, max_lens[i])))

    return idxs


def get_dispersive_loss(query_embedding, mask_idxs):
    pair_sums = []

    for i, idxs in enumerate(mask_idxs):
        if len(idxs) > 1:
            emb = query_embedding[i, idxs, :]
            emb = F.normalize(emb, p=2, dim=-1)

            # Pairwise cosine similarities
            sim = emb @ emb.T

            # Keep each pair once; exclude diagonal
            triu = torch.triu_indices(
                len(idxs),
                len(idxs),
                offset=1,
                device=emb.device,
            )
            pair_sims = sim[triu[0], triu[1]]

            pair_sums.append(pair_sims.mean())

    if pair_sums:
        return torch.stack(pair_sums).mean()
    else:
        return query_embedding.new_tensor(0.0)


class QueryExpansionCL(torch.nn.Module):
    def forward(
        self,
        tokens_logits: torch.Tensor,
        query_embedding: torch.Tensor,
        query_subword_mask: torch.Tensor,
        expansion_subword_mask: torch.Tensor,
        candidates_input_ids: torch.Tensor,
        candidates_subword_mask: torch.Tensor,
        expansion_input_ids: torch.Tensor,
        labels: torch.Tensor,
    ):
        mask_idxs = get_relative_expansion_idxs(
            query_subword_mask, expansion_subword_mask
        )

        disperse_loss = get_dispersive_loss(query_embedding, mask_idxs)

        if all(len(idxs) == 0 for idxs in mask_idxs):
            loss = torch.tensor(0.0, device=tokens_logits.device, requires_grad=True)
            return {"loss": loss}

        losses = []
        for batch_idx, exp_ids in enumerate(expansion_input_ids):
            if not mask_idxs[batch_idx]:
                continue

            cand_idxs = labels[batch_idx].nonzero().flatten().tolist()

            for cand_idx in cand_idxs:
                cand_subword_idxs = candidates_subword_mask[cand_idx].nonzero()[0]

                cand_subword_ids = candidates_input_ids[cand_idx, cand_subword_idxs]

                cand_subword_logits = tokens_logits[
                    batch_idx, cand_idx, mask_idxs[batch_idx], : len(cand_subword_idxs)
                ]

                cand_exp_mask = np.isin(cand_subword_ids, exp_ids)

                cand_exp_mask = torch.as_tensor(
                    cand_exp_mask, device=tokens_logits.device
                )

                cand_max_indices = (
                    cand_subword_logits.masked_fill(~cand_exp_mask, float("-inf"))
                    .max(-1)
                    .indices
                )

                cand_subword_probs = F.log_softmax(cand_subword_logits, dim=-1)

                cand_exp_probs = cand_subword_probs[
                    torch.arange(len(cand_max_indices)), cand_max_indices
                ]

                query_loss = cand_exp_probs.mean()

                # if torch.isnan(query_loss):
                #     breakpoint()

                losses.append(query_loss)

        loss = -1 * torch.hstack(losses).mean()

        return {"loss": loss + disperse_loss}
