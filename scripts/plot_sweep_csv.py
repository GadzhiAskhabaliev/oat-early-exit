#!/usr/bin/env python3
"""Plot proxy metrics from sweep_early_exit.py CSV (English labels, dark tech theme)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from viz_lab import ACCENT_AMBER, ACCENT_CYAN, ACCENT_TEAL, apply_tech_theme


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("matplotlib is required: pip install matplotlib") from e

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, type=Path, help="Sweep CSV path")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG (default: docs/assets/<stem>_sweep.png next to CSV if under fixtures, else beside CSV)",
    )
    args = p.parse_args()
    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")

    rows: list[dict[str, str]] = []
    with args.csv.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        raise SystemExit("CSV has no data rows")

    def fcol(key: str) -> list[float]:
        return [float(row[key]) for row in rows]

    th = fcol("threshold")
    mode = rows[0].get("mode", "?")
    exit_rate = fcol("early_exit_rate")
    saved = fcol("mean_tokens_saved_vs_full")
    mse = fcol("proxy_action_mse_vs_gt")
    mean_tok = fcol("mean_generated_tokens")

    apply_tech_theme()
    out = args.out
    if out is None:
        out = args.csv.with_name(f"{args.csv.stem}_sweep.png")

    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.0), constrained_layout=True)
    fig.suptitle(
        f"Early-exit threshold sweep — mode={mode}",
        fontsize=13,
        fontweight="bold",
        color="#f0f6fc",
    )

    ax00, ax01 = axes[0]
    ax10, ax11 = axes[1]

    ax00.plot(th, exit_rate, "o-", color=ACCENT_CYAN, linewidth=2.0, markersize=7)
    ax00.set_xlabel("Threshold")
    ax00.set_ylabel("Early-exit rate")
    ax00.set_ylim(-0.05, 1.05)
    ax00.set_title("How often decoding stops early")
    ax00.grid(True)

    ax01.plot(th, saved, "o-", color=ACCENT_TEAL, linewidth=2.0, markersize=7)
    ax01.set_xlabel("Threshold")
    ax01.set_ylabel("Mean tokens saved vs full horizon")
    ax01.set_title("Compute savings (token budget)")
    ax01.grid(True)

    ax10.plot(th, mse, "o-", color=ACCENT_AMBER, linewidth=2.0, markersize=7)
    ax10.set_xlabel("Threshold")
    ax10.set_ylabel("Proxy action MSE vs ground truth")
    ax10.set_title("Offline reconstruction quality (proxy)")
    ax10.grid(True)

    sc = ax11.scatter(saved, mse, c=th, cmap="plasma", s=120, edgecolors="#30363d", linewidths=1)
    for x, y, t in zip(saved, mse, th):
        ax11.annotate(f"{t:.2f}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8, color="#8b949e")
    ax11.set_xlabel("Mean tokens saved vs full")
    ax11.set_ylabel("Proxy action MSE vs GT")
    ax11.set_title("Pareto-style trade-off (colored by threshold)")
    cb = fig.colorbar(sc, ax=ax11, shrink=0.75, label="Threshold")
    cb.ax.yaxis.label.set_color("#e6edf3")
    ax11.grid(True)

    fig.text(
        0.5,
        0.01,
        f"Sequences per point: n={len(rows)} | mean gen. tokens: min={min(mean_tok):.2f} max={max(mean_tok):.2f}",
        ha="center",
        fontsize=9,
        color="#8b949e",
    )

    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
