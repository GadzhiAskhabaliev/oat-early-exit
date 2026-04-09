"""BLT-подобный динамический патчинг действий (заготовка для интеграции в OATTok)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def softmax_entropy(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Энтропия распределения после softmax по указанной оси."""
    p = F.softmax(logits, dim=dim)
    log_p = F.log_softmax(logits, dim=dim)
    return -(p * log_p).sum(dim=dim)


def patch_size_from_entropy(
    entropy_scalar: torch.Tensor,
    entropy_threshold: float,
    base_patch_size: int,
    max_extra: int = 1,
) -> int:
    """Пример: при высокой энтропии увеличиваем «дробность» (размер патча base или base+1)."""
    high = entropy_scalar.item() > entropy_threshold
    return base_patch_size + (max_extra if high else 0)
