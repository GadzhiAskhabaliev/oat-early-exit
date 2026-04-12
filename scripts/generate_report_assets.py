#!/usr/bin/env python3
"""
Write illustrative fixtures under docs/assets/fixtures/ and render PNG figures for README.

Run from repo root:
  python scripts/generate_report_assets.py

Figures are committed so the README renders on GitHub without local training artifacts.
Replace fixtures with your real logs.json / eval_log.json / sweep CSV and re-run to refresh.
"""

from __future__ import annotations

import csv
import json
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


def _write_demo_logs_jsonl(path: Path) -> None:
    """Synthetic epoch-end JSONL matching TrainPolicyWorkspace logging keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ep in range(30):
            train = 3.2 * math.exp(-0.09 * ep) + 0.35 + 0.04 * math.sin(ep * 0.7)
            val = 3.0 * math.exp(-0.085 * ep) + 0.42 + 0.05 * math.sin(ep * 0.5 + 1)
            if ep >= 26:
                val = 2.22 + 0.02 * (ep - 28)
            row = {
                "train_loss": round(train, 5),
                "val_loss": round(val, 5),
                "epoch": ep,
                "global_step": (ep + 1) * 120 - 1,
                "lr": 3e-4 * (0.5 ** (ep // 10)),
            }
            f.write(json.dumps(row) + "\n")


def _write_demo_eval_json(path: Path) -> None:
    payload = {
        "checkpoint": "third_party/oat/output/manual/train30_20260411_134306/checkpoints/latest.ckpt",
        "mean_success_rate_mean": 0.222857,
        "num_exp": 1,
        "env_runner_resolved": {
            "n_test": 350,
            "n_parallel_envs": 12,
            "n_test_vis": 0,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)

    sweep = FIXTURES / "demo_sweep_maxprob.csv"
    logs = FIXTURES / "demo_logs.jsonl"
    ev = FIXTURES / "demo_eval_log.json"
    _write_demo_sweep_csv(sweep)
    _write_demo_logs_jsonl(logs)
    _write_demo_eval_json(ev)

    py = sys.executable
    steps = [
        [py, str(ROOT / "scripts/plot_sweep_csv.py"), "--csv", str(sweep), "--out", str(ASSETS / "figure_early_exit_sweep.png")],
        [py, str(ROOT / "scripts/plot_training_logs.py"), "--logs", str(logs), "--out", str(ASSETS / "figure_training_curves.png")],
        [py, str(ROOT / "scripts/plot_eval_log.py"), "--eval-log", str(ev), "--out", str(ASSETS / "figure_eval_summary.png")],
    ]
    for cmd in steps:
        print("==>", " ".join(cmd))
        subprocess.check_call(cmd, cwd=str(ROOT))
    print("Done. Figures under", ASSETS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
