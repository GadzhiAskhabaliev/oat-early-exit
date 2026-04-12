#!/usr/bin/env python3
"""
Write a synthetic sweep CSV under docs/assets/fixtures/ and render figure_early_exit_sweep.png.

Run from repo root:
  python scripts/generate_report_assets.py

This only refreshes the **early-exit sweep** panel (demo proxy metrics). It does **not** overwrite
`figure_training_curves.png` or `figure_eval_summary.png` — use real `logs.json` / `eval_log.json`
with `plot_training_logs.py`, `plot_eval_log.py`, or `./scripts/plot_real_bundle.sh`.
"""

from __future__ import annotations

import csv
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
FIXTURES = ASSETS / "fixtures"


def _write_demo_sweep_csv(path: Path) -> None:
    """Synthetic but schema-identical sweep (max-prob style) for layout demos."""
    rows = []
    for i, th in enumerate([0.55, 0.65, 0.75, 0.85, 0.92]):
        exit_r = min(0.95, 0.15 + i * 0.17)
        saved = 2.0 + i * 1.1 + 0.15 * math.sin(i)
        mse = 0.022 - i * 0.0025 + 0.002 * math.cos(i)
        rows.append(
            {
                "mode": "maxprob",
                "threshold": th,
                "batches": 50,
                "sequences": 1600,
                "early_exit_rate": exit_r,
                "mean_generated_tokens": 14.5 - saved * 0.4,
                "mean_tokens_saved_vs_full": saved,
                "proxy_action_mse_vs_gt": max(0.004, mse),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)

    sweep = FIXTURES / "demo_sweep_maxprob.csv"
    _write_demo_sweep_csv(sweep)

    py = sys.executable
    cmd = [
        py,
        str(ROOT / "scripts/plot_sweep_csv.py"),
        "--csv",
        str(sweep),
        "--out",
        str(ASSETS / "figure_early_exit_sweep.png"),
    ]
    print("==>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))
    print("Done. Updated:", ASSETS / "figure_early_exit_sweep.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
