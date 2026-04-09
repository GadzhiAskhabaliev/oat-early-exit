# Experiments section (draft template for the test assignment report)

Use this as a skeleton; fill numbers after you run protocols in `EARLY_EXIT.md`.

## Setup

- **Hardware**: (e.g. Colab Pro A100, Vast.ai instance, local GPU model)
- **Software**: OAT commit / fork date; `uv sync`; `PYTHONPATH` for `oat_ext`
- **Checkpoints**: path to OAT tokenizer; path to OAT policy (frozen or fine-tuned)
- **Data**: LIBERO multitask zarr subset (N demos) if budget-limited

## Protocol A — Proxy metrics (no simulator)

Cheap signals when rollouts are too expensive:

1. **Action reconstruction MSE** on a held-out slice of the training zarr: run `tokenize → detokenize` on full sequences; report mean MSE (same as tokenizer quality smoke test).
2. **Prefix reconstruction curve**: for each prefix length \(k\), MSE between `detokenize(tokens[:, :k])` and ground-truth actions (see `oat_ext.early_exit_supervision.mse_per_prefix`). Plot mean MSE vs \(k\).
3. **Early-exit rate** during `predict_action` with fixed thresholds: fraction of rollouts (or batches) where generation stopped before `latent_horizon`; average number of generated tokens (see `batch_early_exit_stats`).

## Protocol B — Simulator (LIBERO)

1. **Baseline**: full-length generation (`use_early_exit_inference=false`).
2. **Ablations**: max-prob heuristic (`early_exit_max_prob` ∈ {0.90, 0.95, 0.99}); learned gate with thresholds ∈ {0.7, 0.8, 0.9}.
3. **Metrics**: task success rate (mean ± stderr over seeds); wall-clock ms per `predict_action` call (sync, same batch size).

## Results table (example layout)

| Method | Threshold | Success (%) | Avg. tokens ↓ | Latency (ms) ↓ |
|--------|-----------|-------------|---------------|----------------|
| No early exit | — | … | 8.0 | … |
| Max prob | 0.95 | … | … | … |
| Learned gate | 0.85 | … | … | … |

## Limitations (required honesty)

- Short training / few epochs / subset of demos
- Reconstruction labels are a proxy for task success
- Possible off-by-one alignment between gate timestep and “semantic” prefix (document if you observe it)

## Connection to BLT / H-Net (one paragraph for the grader)

Early exit allocates **less compute** when the model is already confident or when a short prefix reconstructs the action well — analogous in spirit to **adaptive depth / patch boundaries** in BLT. It can be described as a **two-level** decision: a coarse decision (“stop generating tokens”) before spending the full budget, related in narrative to **hierarchical** processing in H-Net, even though the implementation is a small gate rather than a full multi-scale backbone.
