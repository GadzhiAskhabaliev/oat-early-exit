import torch

from oat_ext.dynamic_patching import patch_size_from_entropy, softmax_entropy


def test_softmax_entropy_shape():
    x = torch.randn(2, 3, 4)
    e = softmax_entropy(x, dim=-1)
    assert e.shape == (2, 3)


def test_patch_size_from_entropy():
    ent = torch.tensor(0.9)
    assert patch_size_from_entropy(ent, entropy_threshold=0.5, base_patch_size=2) == 3
    ent_low = torch.tensor(0.1)
    assert patch_size_from_entropy(ent_low, entropy_threshold=0.5, base_patch_size=2) == 2
