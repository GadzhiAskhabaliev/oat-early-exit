"""Иерархический токенизатор действий — заготовка под H-Net / многоуровневые масштабы."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class HierarchicalLevelsSpec:
    """Описание уровней (заполнить под выбранную архитектуру OAT)."""

    level_names: Sequence[str] = ("coarse", "fine")
    # coarse_horizon, fine_horizon, fusion — добавятся при интеграции
