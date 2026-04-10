#!/usr/bin/env python3
"""Plot proxy metrics from sweep_early_exit.py CSV output."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required: pip install matplotlib"
        ) from e

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, type=Path, help="Sweep CSV path")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: <csv_stem>_plots.png next to CSV)",
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

    out = args.out or args.csv.with_name(f"{args.csv.stem}_plots.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
    fig.suptitle(f"Early-exit sweep (mode={mode})", fontsize=11, fontweight="bold")

    ax0, ax1, ax2 = axes
    ax0.plot(th, exit_rate, "o-", color="#6366f1", linewidth=1.5, markersize=6)
    ax0.set_xlabel("Threshold")
    ax0.set_ylabel("Early-exit rate")
    ax0.set_ylim(-0.05, 1.05)
    ax0.grid(True, alpha=0.3)

    ax1.plot(th, saved, "o-", color="#14b8a6", linewidth=1.5, markersize=6)
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Mean tokens saved vs full")
    ax1.grid(True, alpha=0.3)

    ax2.plot(th, mse, "o-", color="#f97316", linewidth=1.5, markersize=6)
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Proxy action MSE vs GT")
    ax2.grid(True, alpha=0.3)

    fig.text(
        0.5,
        0.02,
        f"mean seq. len: min={min(mean_tok):.2f} max={max(mean_tok):.2f} | n={len(rows)} points",
        ha="center",
        fontsize=8,
        color="gray",
    )

    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
