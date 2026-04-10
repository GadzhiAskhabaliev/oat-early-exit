"""
Early exit for autoregressive OAT token generation.

The gate maps the last-layer hidden state (after RMSNorm, before the vocab head)
to a scalar logit P(stop | context). At inference, generation can stop early when
sigmoid(logit) exceeds a threshold (after a minimum number of tokens), trading
latency for possible quality loss on harder trajectories.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EarlyExitGate(nn.Module):
    """
    Small MLP: n_emb -> hidden -> 1. Applied per time step to hidden states during
    training (auxiliary BCE) and at inference to the causal LM state before sampling
    the *next* token.
    """

    def __init__(
        self,
        n_emb: int,
        hidden_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        h = hidden_mult * n_emb
        self.net = nn.Sequential(
            nn.Linear(n_emb, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, T, n_emb) or (B, n_emb)
        Returns:
            logits: (B, T, 1) or (B, 1), raw (no sigmoid)
        """
        return self.net(h)


def early_exit_supervision_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Weak default labels: fire the gate only on the last valid position (end of sequence).
    Shape (seq_len,) with 1.0 at last index, 0 elsewhere.
    """
    t = torch.zeros(seq_len, device=device, dtype=torch.float32)
    t[-1] = 1.0
    return t
