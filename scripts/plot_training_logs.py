#!/usr/bin/env python3
"""Plot train / validation loss curves from OAT policy `logs.json` (JSONL, one object per line)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from viz_lab import ACCENT_CYAN, ACCENT_TEAL, apply_tech_theme


def _read_epoch_metrics(path: Path) -> tuple[list[int], list[float], list[int], list[float]]:
    """Use the last JSON record per epoch (mid-epoch lines carry noisy batch losses)."""
    last_by_ep: dict[int, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ep = row.get("epoch")
            if ep is None:
                continue
            last_by_ep[int(ep)] = row
    epochs = sorted(last_by_ep)
    epochs_t: list[int] = []
    train_t: list[float] = []
    epochs_v: list[int] = []
    val_v: list[float] = []
    for ep in epochs:
        row = last_by_ep[ep]
        if "train_loss" in row:
            epochs_t.append(ep)
            train_t.append(float(row["train_loss"]))
        if "val_loss" in row:
            epochs_v.append(ep)
            val_v.append(float(row["val_loss"]))
    return epochs_t, train_t, epochs_v, val_v


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logs", type=Path, required=True, help="Path to logs.json (JSONL)")
    p.add_argument("--out", type=Path, required=True, help="Output PNG path")
    args = p.parse_args()
    if not args.logs.is_file():
        raise SystemExit(f"logs file not found: {args.logs}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("matplotlib required: pip install matplotlib") from e

    apply_tech_theme()
    et, yt, ev, yv = _read_epoch_metrics(args.logs)
    if not yt and not yv:
        raise SystemExit("No train_loss / val_loss rows found in logs")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
    if et and yt:
        ax.plot(et, yt, "-", color=ACCENT_CYAN, linewidth=2.0, label="Train loss (epoch mean)", marker="o", markersize=3)
    if ev and yv:
        ax.plot(ev, yv, "-", color=ACCENT_TEAL, linewidth=2.0, label="Validation loss", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("OAT policy training (LIBERO-style run)")
    ax.legend(loc="upper right")
    ax.grid(True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
