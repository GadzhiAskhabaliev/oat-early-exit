#!/usr/bin/env python3
"""Plot a compact summary from `eval_log.json` produced by `eval_policy_sim.py`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from viz_lab import ACCENT_TEAL, apply_tech_theme


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-log", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if not args.eval_log.is_file():
        raise SystemExit(f"eval log not found: {args.eval_log}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("matplotlib required: pip install matplotlib") from e

    data = json.loads(args.eval_log.read_text())
    key = "mean_success_rate_mean"
    if key not in data:
        raise SystemExit(f"Missing {key!r} in eval JSON (keys: {list(data.keys())[:12]}...)")

    rate = float(data[key])
    n_test = None
    resolved = data.get("env_runner_resolved") or {}
    if isinstance(resolved, dict):
        n_test = resolved.get("n_test") or resolved.get("num_test")

    apply_tech_theme()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 3.8), constrained_layout=True)
    ax.barh([0], [rate], height=0.45, color=ACCENT_TEAL, edgecolor="#30363d", linewidth=1)
    ax.set_xlim(0, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("Mean success rate")
    title = "LIBERO simulation eval (aggregated)"
    if n_test is not None:
        title += f" — n_test={n_test}"
    ax.set_title(title)
    ax.axvline(rate, color="#58d6ff", linestyle=":", alpha=0.7)
    pct = rate * 100.0
    ax.text(
        min(rate + 0.04, 0.92),
        0,
        f"{pct:.1f}%",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#f0f6fc",
    )
    ax.grid(True, axis="x")
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
