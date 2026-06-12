"""Losses adapted from the public GPO-V LLaDA-V demo."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_NEGATIVE_IDS = [
    10248,
    29327,
    55206,
    105947,
    78149,
    16103,
    3260,
    96359,
    74696,
    40,
    3972,
    6034,
]


class PositionTokenAnchorLoss(nn.Module):
    """GPO-V anchor loss plus global refusal-token suppression.

    Args:
        target_dict: Mapping from response position to allowed anchor token ids.
        negative_ids: Token ids to suppress globally across the response logits.
        weight_target: Weight for exact anchor targets.
        weight_negative: Weight for negative/refusal token suppression.
    """

    def __init__(
        self,
        target_dict: dict[int, list[int]],
        negative_ids: list[int] | None = None,
        weight_target: float = 1.0,
        weight_negative: float = 10.0,
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__()
        self.weight_target = weight_target
        self.weight_negative = weight_negative
        self.target_positions = {
            int(pos): torch.tensor(list(set(ids)), device=device, dtype=torch.long)
            for pos, ids in target_dict.items()
        }
        self.negative_ids = (
            torch.tensor(list(set(negative_ids)), device=device, dtype=torch.long)
            if negative_ids
            else None
        )

    def forward(self, logits: torch.Tensor) -> tuple[torch.Tensor, dict[int, tuple[int, float, bool]]]:
        """Return scalar loss and per-anchor monitoring information."""
        batch_size, seq_len, _ = logits.shape
        target_loss = torch.tensor(0.0, device=logits.device)

        for pos, ids in self.target_positions.items():
            if pos >= seq_len:
                continue
            pos_logits = logits[:, pos, :]
            if ids.numel() == 1:
                targets = ids.view(1).expand(batch_size).to(logits.device)
            else:
                targets = ids[:1].expand(batch_size).to(logits.device)
            target_loss = target_loss + F.cross_entropy(pos_logits, targets)

        if self.negative_ids is not None:
            neg_ids = self.negative_ids.to(logits.device)
            probs = F.softmax(logits, dim=-1)
            negative_loss = probs.index_select(dim=-1, index=neg_ids).sum(dim=-1).mean()
        else:
            negative_loss = torch.tensor(0.0, device=logits.device)

        loss = self.weight_target * target_loss + self.weight_negative * negative_loss

        monitor = {}
        with torch.no_grad():
            for pos, ids in self.target_positions.items():
                if pos >= seq_len:
                    continue
                pos_probs = F.softmax(logits[0, pos, :], dim=-1)
                best_id = int(torch.argmax(pos_probs).item())
                best_prob = float(pos_probs[best_id].item())
                monitor[pos] = (best_id, best_prob, best_id in ids.tolist())

        return loss, monitor

