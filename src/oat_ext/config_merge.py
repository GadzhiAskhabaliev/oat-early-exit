"""Merge a base OmegaConf from OAT with flat overrides for ablation experiments."""

from __future__ import annotations

from typing import Any, Mapping

from omegaconf import OmegaConf


def merge_with_overrides(
    base: Any,
    overrides: Mapping[str, Any] | None = None,
) -> Any:
    """Return a copy of `base` with `overrides` merged in (nested keys via dotted paths; see OmegaConf.merge)."""
    if overrides is None or len(overrides) == 0:
        return OmegaConf.create(base) if not OmegaConf.is_config(base) else base
    extra = OmegaConf.create(overrides)
    return OmegaConf.merge(OmegaConf.create(base), extra)
