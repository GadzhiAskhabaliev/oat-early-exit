import torch

from oat_ext.early_exit import EarlyExitGate, early_exit_supervision_mask


def test_early_exit_gate_shapes():
    g = EarlyExitGate(n_emb=64, hidden_mult=2, dropout=0.0)
    b, t, d = 3, 7, 64
    h = torch.randn(b, t, d)
    out = g(h)
    assert out.shape == (b, t, 1)
    out2 = g(h[:, -1])
    assert out2.shape == (b, 1)


def test_supervision_mask():
    m = early_exit_supervision_mask(5, device=torch.device("cpu"))
    assert m.shape == (5,)
    assert m[-1].item() == 1.0 and m[0].item() == 0.0
