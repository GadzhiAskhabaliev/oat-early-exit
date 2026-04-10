<div align="center">

# OAT · Adaptive Early Exit

**Learned and heuristic early stopping for autoregressive OAT action tokens**  
*Extends [OAT](https://github.com/Chaoqi-LIU/oat) with a BLT/H-Net–style compute–quality trade-off at decode time.*

<!-- Badges: CI expects repo at github.com/criptyn97/mipt-lab-project — change paths if your remote differs. -->

[![CI](https://github.com/criptyn97/mipt-lab-project/actions/workflows/ci.yml/badge.svg)](https://github.com/criptyn97/mipt-lab-project/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)  
[![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)](https://github.com/criptyn97/mipt-lab-project/actions/workflows/ci.yml)
[![uv](https://img.shields.io/badge/package%20manager-uv-5A0FC8?logo=uv&logoColor=white)](https://github.com/astral-sh/uv)
[![Hydra](https://img.shields.io/badge/config-Hydra-89b8cd)](https://hydra.cc/)
[![Docs](https://img.shields.io/badge/docs-early--exit-111827?style=flat&logo=readthedocs&logoColor=white)](docs/early-exit.md)
[![Upstream OAT](https://img.shields.io/badge/upstream-OAT-111827?style=flat&logo=github)](https://github.com/Chaoqi-LIU/oat)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/criptyn97/mipt-lab-project/pulls)

```text
  obs ──► encoder ──► LM + KV cache ──► logits ──► sample
                           │                    │
                           └──── early-exit ◄───┘
                                 gate | max-p
```

</div>

---

## System snapshot

| Layer | Role |
|--------|------|
| **Policy** | OAT autoregressive tokenizer over discrete action tokens |
| **Extension** | `EarlyExitGate` (MLP on `ln_f` hidden) or `max_prob` heuristic |
| **Objective** | Shorter generations when confident / well-reconstructed; full horizon when not |
| **Artifacts** | `src/oat_ext/*`, patched `third_party/oat` transformer + policy |

---

## Repository map

| Path | Contents |
|------|----------|
| `src/oat_ext/` | `EarlyExitGate`, supervision helpers, hooks for dynamic patching / hierarchy |
| `scripts/` | Install, offline gate training, threshold sweeps, `vast_run_early_exit.sh`, `plot_sweep_csv.py`, eval helpers |
| `configs/hydra_overrides/` | Optional `.env` snippets for baseline / ablation Hydra merges |
| `tests/` | `pytest` for `oat_ext` (see `pytest.ini` → `pythonpath = src`) |
| `docs/` | `early-exit.md`, `experiments-section-template.md`, `results-and-visuals.md` |
| `third_party/oat/` | Vendored OAT with local modifications |

---

## Quick start

```bash
# 1) OAT environment + deps
./scripts/install_oat.sh

# 2) LIBERO-10 zarr (if running full pipeline)
./scripts/download_libero10_zarr.sh
```

If `uv` is not on PATH:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

---

## Execution graph

```mermaid
flowchart TB
  A[Baseline OAT train] --> B[Optional: joint early-exit loss]
  A --> C[Offline gate: reconstruction labels]
  C --> D[Threshold / mode sweep]
  B --> D
  D --> E[Proxy metrics CSV]
  D --> F[LIBERO eval optional]
```

1. **Train baseline** — upstream configs (GPU recommended; requires LIBERO-10 zarr under `third_party/oat/data/` per `train_baseline.sh` comments):

   ```bash
   ./scripts/train_baseline.sh
   ```

   Smoke run:

   ```bash
   ./scripts/train_baseline.sh training.num_epochs=1 training.val_every=1 dataloader.batch_size=4
   ```

2. **Train `EarlyExitGate` offline** (reconstruction-supervised).  
   Default output for [`vast_run_early_exit.sh`](scripts/vast_run_early_exit.sh) is `<repo>/checkpoints/early_exit_gate.pt`; below uses the same path so it matches the Vast section and [docs/results-and-visuals.md](docs/results-and-visuals.md) (you can use `experiments/checkpoints/` instead if you prefer).

   ```bash
   cd third_party/oat
   uv run python ../../scripts/train_early_exit_offline.py \
     --checkpoint /path/to/policy.ckpt \
     --mse-threshold 0.015 \
     --epochs 3 \
     --max-batches 100 \
     --out-gate ../../checkpoints/early_exit_gate.pt
   ```

3. **Sweep thresholds** → CSV proxy metrics:

   ```bash
   uv run python ../../scripts/sweep_early_exit.py \
     --checkpoint /path/to/policy.ckpt \
     --mode gate \
     --gate ../../checkpoints/early_exit_gate.pt \
     --thresholds 0.7 0.8 0.9 \
     --max-batches 50 \
     --out-csv ../../experiments/runs/sweep_gate.csv
   ```

   Max-probability baseline:

   ```bash
   uv run python ../../scripts/sweep_early_exit.py \
     --checkpoint /path/to/policy.ckpt \
     --mode maxprob \
     --thresholds 0.7 0.8 0.9 \
     --max-batches 50 \
     --out-csv ../../experiments/runs/sweep_maxprob.csv
   ```

4. **Simulator (optional)**:

   ```bash
   ./scripts/eval_libero.sh /path/to/oatpolicy.ckpt --num_exp 5
   ```

---

## Hydra · inference overrides

```bash
policy.early_exit_gate._target_=oat_ext.early_exit.EarlyExitGate
policy.early_exit_gate.n_emb=256
policy.early_exit_gate_checkpoint=/path/to/early_exit_gate.pt
policy.use_early_exit_inference=true
policy.early_exit_threshold=0.9
```

---

## Fork touchpoints

| File | Delta |
|------|--------|
| `third_party/oat/oat/model/autoregressive/transformer_cache.py` | `return_hidden`; early-exit inside `generate()` |
| `third_party/oat/oat/policy/oatpolicy.py` | Gate wiring, checkpoint load, inference flags |

---

## Tests

Lightweight checks for `oat_ext` only (full OAT stack not required):

```bash
pip install -r requirements.txt
pytest
```

---

## Remote GPU (Vast-style)

After syncing the repo on a rented GPU, run the bundled pipeline (writes `checkpoints/early_exit_gate.pt` and `experiments/runs/sweep_gate_trained.csv` by default):

```bash
./scripts/vast_run_early_exit.sh --checkpoint /path/to/policy.ckpt
```

Then add figures and benchmark tables to the README using [docs/results-and-visuals.md](docs/results-and-visuals.md) (CSV schema, plot script, suggested tables).

---

## Documentation

| Doc | Purpose |
|-----|---------|
| [docs/early-exit.md](docs/early-exit.md) | Pipeline, hypothesis, APIs, Hydra, limitations |
| [docs/experiments-section-template.md](docs/experiments-section-template.md) | Report-ready experiment skeleton |
| [docs/results-and-visuals.md](docs/results-and-visuals.md) | Post-run artifacts, plots, README benchmark strip |
