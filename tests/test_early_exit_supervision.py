import pytest
import torch

from oat_ext.early_exit_supervision import (
    batch_early_exit_stats,
    mse_per_prefix,
    reconstruction_labels,
)


class _FakeTokenizer:
    """Returns constant reconstructions so MSE vs gt is predictable."""

    def detokenize(self, partial: torch.Tensor) -> torch.Tensor:
        B = partial.shape[0]
        return torch.zeros(B, 4, 3, device=partial.device, dtype=torch.float32)


def test_mse_per_prefix_and_labels():
    tok = _FakeTokenizer()
    B, L = 2, 4
    gt = torch.zeros(B, 4, 3)
    tokens = torch.zeros(B, L, dtype=torch.long)
    m = mse_per_prefix(tok, gt, tokens)
    assert m.shape == (B, L)
    assert (m == 0).all()
    lab = reconstruction_labels(m, mse_threshold=0.01)
    assert lab.shape == m.shape
    assert (lab == 1).all()


def test_early_exit_stats():
    s = batch_early_exit_stats(
        torch.tensor([8, 6, 8]),
        full_length=8,
    )
    assert s["early_exit_rate"] == pytest.approx(1 / 3)
    assert s["mean_tokens"] == pytest.approx(22 / 3)
