<div align="center">

# Results, Benchmarks & Figures

**What to capture after remote training (e.g. Vast) and how to wire it into the README / report**

</div>

---

## 1. Artifact layout

| Location | Keep in git? | Purpose |
|----------|----------------|---------|
| `experiments/runs/*.csv` | Usually **no** (large / machine-specific); `.gitignore` | Raw sweep outputs from `sweep_early_exit.py` / `vast_run_early_exit.sh` |
| `checkpoints/early_exit_gate.pt` | **No** | Trained gate weights |
| `docs/assets/` | **Yes** (small PNG/SVG) | Figures embedded in `README.md` or PDF report |
| Report appendix | **Yes** | Tables exported from CSV (or link to Zenodo for heavy bundles) |

After a Vast run, **copy** the sweep CSV and a short **run manifest** (YAML or markdown snippet) into a safe place before the instance is destroyed.

### 1.1 Regenerate the synthetic sweep figure

`docs/assets/fixtures/demo_sweep_maxprob.csv` plus `figure_early_exit_sweep.png` can be rebuilt anytime:

```bash
pip install matplotlib
python scripts/generate_report_assets.py
```

That script **does not** overwrite training or eval PNGs. For real curves, use `scripts/plot_training_logs.py`, `plot_eval_log.py`, or `scripts/plot_real_bundle.sh` (see root `README.md`).

---

## 2. Sweep CSV schema (`sweep_early_exit.py`)

Each row is one `(mode, threshold)` setting:

| Column | Meaning |
|--------|---------|
| `mode` | `gate` or `maxprob` |
| `threshold` | Gate sigmoid threshold or max-probability cutoff |
| `batches` / `sequences` | How much data was averaged (document when comparing runs) |
| `early_exit_rate` | Fraction of sequences that stopped before full horizon |
| `mean_generated_tokens` | Average decoded length |
| `mean_tokens_saved_vs_full` | `full_len - mean_generated_tokens` |
| `proxy_action_mse_vs_gt` | Batch-mean MSE between detokenized prediction and GT action |

Use the same policy checkpoint and comparable `max_batches` when comparing gate vs maxprob.

---

## 3. Recommended visualizations

| Plot | X | Y | Story |
|------|---|---|--------|
| **Trade-off A** | `threshold` | `early_exit_rate` | How often you actually exit early |
| **Trade-off B** | `threshold` | `mean_tokens_saved_vs_full` | Compute savings vs threshold |
| **Trade-off C** | `mean_tokens_saved_vs_full` | `proxy_action_mse_vs_gt` | Proxy quality vs savings (Pareto-style) |
| **Training curves** | epoch | `train_bce`, `val_bce` | Gate learning sanity (from offline train logs) |

Optional for the full paper-style story:

- **Prefix MSE curve** (Protocol A in the experiment template): prefix length \(k\) vs mean reconstruction MSE.
- **LIBERO bar chart**: success rate and mean latency per method (baseline vs ablations).

---

## 4. Quick plot from CSV

Install plotting dependency (once):

```bash
pip install matplotlib
```

From repo root:

```bash
python scripts/plot_sweep_csv.py \
  --csv experiments/runs/sweep_gate_trained.csv \
  --out docs/assets/sweep_gate_trained.png
```

Check the figure into `docs/assets/` and reference it in `README.md`:

```markdown
![Sweep trade-offs](docs/assets/sweep_gate_trained.png)
```

---

## 5. README “benchmark strip” (suggested)

After you have numbers, add a compact table under **System snapshot** or a new **Benchmarks** section:

| Setting | Threshold | Early-exit rate | Mean tokens | Proxy MSE | Notes |
|---------|-----------|-----------------|-------------|-----------|--------|
| Baseline | — | 0% | 8.0 | … | Full horizon |
| Gate | 0.85 | … | … | … | Vast run `YYYY-MM-DD` |
| Max-prob | 0.95 | … | … | … | Same checkpoint |

Link the CSV path or a frozen copy in `docs/assets/` for reproducibility.

---

## 6. What improves the narrative (beyond plots)

- **Run manifest**: GPU model, batch counts, commit hash, policy checkpoint path (even if redacted), `mse-threshold`, epochs.
- **Negative result**: threshold too aggressive → success or MSE collapses; show the failure mode.
- **Side-by-side**: gate vs maxprob at **matched** early-exit rate (interpolate thresholds), not only matched numeric threshold.

---

## Related

- [`experiments-section-template.md`](experiments-section-template.md) — where these numbers land in the written report.
- [`early-exit.md`](early-exit.md) — metric definitions and protocols.
- [`../scripts/vast_run_early_exit.sh`](../scripts/vast_run_early_exit.sh) — default outputs: `checkpoints/early_exit_gate.pt`, `experiments/runs/sweep_gate_trained.csv`.

---

## 7. Filled snapshot for this repo

Concrete paths, eval settings, and PNG list for the **April 2026** LIBERO policy run are written in **`experiments-section-template.md` → section *Filled snapshot (this project, April 2026)***. Use that block when pasting into the course PDF; keep this file as the **schema / plotting** reference.
