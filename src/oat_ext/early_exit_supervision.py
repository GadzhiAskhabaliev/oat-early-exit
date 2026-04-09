"""
Reconstruction-based targets for EarlyExitGate (offline / auxiliary training).

For each prefix length k = 1..L of OAT discrete tokens, decode with the frozen
tokenizer and measure MSE against ground-truth continuous actions. If MSE is below
a threshold, the prefix is considered sufficient — label 1 at time step k-1 for
the gate (aligned with teacher-forcing hidden h[:, k-1] that predicts token k).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def mse_per_prefix(
    action_tokenizer: nn.Module,
    gt_actions: torch.Tensor,
    tokens: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        action_tokenizer: OATTok (or compatible) with ``detokenize``.
        gt_actions: (B, T, Da) ground-truth actions in the same space as detokenize output.
        tokens: (B, L) integer OAT token indices from ``tokenize(gt_actions)``.

    Returns:
        mse: (B, L) where mse[b, k] = MSE( detokenize(tokens[b, :k+1]), gt_actions[b] )
        i.e. column k corresponds to prefix length (k+1) tokens (k is 0-based column).
    """
    if tokens.dim() != 2:
        raise ValueError(f"tokens must be (B, L), got {tokens.shape}")
    B, L = tokens.shape
    device = gt_actions.device
    dtype = gt_actions.dtype
    mses = []
    for k in range(1, L + 1):
        partial = tokens[:, :k].contiguous()
        recon = action_tokenizer.detokenize(partial)
        err = (recon.to(dtype) - gt_actions) ** 2
        mses.append(err.mean(dim=(1, 2)))
    return torch.stack(mses, dim=1).to(device=device, dtype=dtype)


def reconstruction_labels(
    mse_per_prefix_matrix: torch.Tensor,
    mse_threshold: float,
) -> torch.Tensor:
    """
    Args:
        mse_per_prefix_matrix: (B, L) from :func:`mse_per_prefix`.
        mse_threshold: scalar; MSE below this → label 1 (early exit allowed).

    Returns:
        labels: (B, L) float {0., 1.}
    """
    return (mse_per_prefix_matrix < mse_threshold).float()


@torch.no_grad()
def batch_early_exit_stats(
    generated_length: torch.Tensor,
    full_length: int,
) -> dict:
    """
    Summarize how often generation stopped early (proxy metric for logging).

    Args:
        generated_length: (B,) int number of OAT tokens produced per sequence.
        full_length: nominal maximum (e.g. latent_horizon).

    Returns:
        dict with ``exit_rate``, ``mean_tokens``, ``mean_speedup``.
    """
    early = generated_length < full_length
    return {
        "early_exit_rate": float(early.float().mean().item()),
        "mean_tokens": float(generated_length.float().mean().item()),
        "mean_speedup_vs_full": float((full_length - generated_length.float()).mean().item()),
    }
