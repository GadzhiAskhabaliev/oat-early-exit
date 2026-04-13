# LIBERO Training/Eval Debug Journal

This file tracks the real run history on Vast.ai for the OAT policy lab, including failures, root causes, fixes, and decisions.

## Scope

- Project: OAT early-exit lab (LIBERO benchmark)
- Goal: get a stable policy training pipeline and non-zero evaluation success
- Environment: Vast.ai remote GPU via SSH + tmux

## Current Status

### Historical (2026-04-11) — quick eval gate failed

- Quick evaluation finished successfully from infrastructure perspective.
- Result was bad from model quality perspective:
  - `mean_success_rate = 0.0`
  - all 10 LIBERO tasks in the quick-eval JSON are `0.0`.
- Decision at the time: that checkpoint was not accepted as final.
- Artifacts mentioned then:
  - `/root/oat-early-exit/experiments/runs/eval_libero_quick_fast/eval_log_quick.json`
  - `/root/oat-early-exit/experiments/runs/eval_libero_quick/eval_log_quick.json`

### Final pipeline outcome (2026-04-12) — for lab reviewers

- **Policy training:** stable 30-epoch FP32 run `train30_20260411_134306` under `third_party/oat/output/manual/…`, `logs.json` ~9 MB, `val_loss` tracked per epoch (see `README` / submission doc for metrics narrative).
- **Simulator eval:** `experiments/runs/eval_libero_7to8h_20260412_112444/eval_log.json` — `mean_success_rate_mean ≈ 0.223` with **`n_test=350`**, `n_parallel_envs=12`, `n_test_vis=0` (shorter than default 500 for wall-clock; metric is an estimate, not comparable to full-500 tables without a separate run).
- **Artifacts off-instance:** checkpoint + logs + eval pushed to **Hugging Face** (`hackhackhack66666/oat-libero-policy-early-exit`); local **`oat_lab_backup_*.tgz`** on laptop mirrors `latest.ckpt`, `logs.json`, `eval_log.json`.
- **Figures:** real `logs.json` / `eval_log.json` plotted to `docs/assets/figure_training_curves.png` and `figure_eval_summary.png`; sweep plot is omitted in the default README snapshot unless a real `sweep_early_exit.py` CSV is provided.

## Confirmed Observations

1. Training/eval infrastructure is mostly stable after SSH and reboot fixes.
2. Checkpoints are being saved (`latest.ckpt` exists and has expected size).
3. `loss=0` in progress view is not always sufficient as a health indicator; JSON logs must be checked.
4. One quick-eval output path mismatch happened:
   - checked `eval_libero_quick_fast`
   - actual file was saved under `eval_libero_quick`.
5. Latest quick-eval artifact read:
   - `/root/oat-early-exit/experiments/runs/eval_libero_quick_fast/eval_log_quick.json` (latest)
   - `/root/oat-early-exit/experiments/runs/eval_libero_quick/eval_log_quick.json` (earlier read)
   - aggregate `mean_success_rate: 0.0` in both observed runs.

## Failures and Fixes Log

### 1) SSH instability/timeouts

- Symptom: frequent disconnects and timeouts.
- Mitigation used:
  - `ServerAliveInterval=30`
  - `ServerAliveCountMax=4`
  - `IPQoS=none`
  - explicit `KexAlgorithms=curve25519-sha256` where needed.
  - prefer reconnect + tmux, not foreground long runs in plain SSH.

### 2) Multiple concurrent/stale runs (process overlap)

- Symptom: multiple train/eval processes at once, mixed logs, unclear run ownership.
- Root cause: repeated relaunches without cleanup after disconnects/timeouts.
- Fix:
  - `pkill -f "scripts/run_workspace.py --config-name=train_oatpolicy"` before fresh launch;
  - keep one authoritative run tag and one active tmux pane per job.

### 3) OOM/SIGKILL risk from dataset loading

- Symptom: training killed abruptly (`SIGKILL`) on large data handling.
- Root cause: memory-heavy defaults (`copy_to_memory=true`) and non-streaming normalization behavior.
- Fixes used:
  - set `task.policy.dataset.copy_to_memory=false`;
  - patch/adjust normalization path to avoid full-array memory spikes on zarr-backed data.

### 4) `wandb` interactive prompts in headless environment

- Symptom: run blocks waiting for interactive input.
- Root cause: tracking service prompt in non-interactive remote shell.
- Fix:
  - `OAT_DISABLE_WANDB=1` for stable unattended runs.

### 5) CUDA/NVML mismatch on instance

- Symptom: `cudaGetDeviceCount` error 804, NVML mismatch.
- Fix: instance reboot (`sudo reboot`), then verify with `nvidia-smi`.

### 6) Ambiguous `loss=0` interpretation

- Symptom: tqdm/log rows with many `train_loss: 0.0`.
- Mitigation:
  - force stable short retrain in FP32 (`training.allow_bf16=false`);
  - validate each epoch;
  - checkpoint each epoch;
  - inspect JSON logs (not only progress bar).
  - compare percent of zero rows + last non-zero train loss.

### 7) Resume/run-dir ambiguity across relaunches

- Symptom: new run accidentally continuing old state or writing into confusing paths.
- Root cause: repeated launches with implicit defaults.
- Fix:
  - enforce unique `hydra.run.dir` per launch;
  - use `training.resume=false` unless resume is explicitly intended.

### 8) Missing env/path variables in new SSH sessions

- Symptom:
  - `uv` not found;
  - checkpoint load failures because `CKPT` variable is empty.
- Root cause: clean shell on reconnect without exported vars.
- Fix:
  - re-export `PATH="$HOME/.local/bin:$PATH"`;
  - re-define `RUN`/`CKPT` before eval commands.

### 9) Eval runtime unexpectedly long

- Symptom: evaluation appears "stuck" for a long time.
- Root cause: default LIBERO eval settings are heavy (`n_test`, env parallelism).
- Fix:
  - use quick-eval overrides (`n_test` lower, fewer parallel envs);
  - run smaller `num_exp` only for smoke checks.

### 10) Eval output directory collision/overwrite prompt

- Symptom: prompt `Output path ... already exists! Overwrite? [y/N]:`
- Root cause: reusing same eval output directory.
- Fix:
  - write to timestamped fresh output dirs each run.

### 11) Eval result file confusion

- Symptom: `No such file or directory` for quick-eval JSON.
- Cause: wrong directory name used when reading output.
- Fix: locate latest `eval_libero_quick*` run and read actual JSON there.

### 12) Shell command formatting/paste mistakes under pressure

- Symptom: multi-command one-liners pasted as a single malformed command, parsing errors.
- Root cause: commands merged without separators/newlines.
- Fix:
  - provide short copy-paste-safe blocks;
  - execute step-by-step with explicit verification after each step.

## Next Controlled Cycle (must follow)

1. Kill stale train/eval processes.
2. Launch short retrain (3 epochs, FP32, no resume, checkpoint each epoch).
3. After epoch 1:
   - inspect `logs.json`,
   - compute zero-loss ratio and last non-zero train loss,
   - confirm `val_loss` row exists.
4. Run quick eval on the new checkpoint.
5. Record:
   - run path,
   - checkpoint path,
   - train-loss diagnostics,
   - quick-eval `mean_success_rate`,
   - go/no-go decision.

## Run Entry Template (copy for each run)

```md
### Run: <run_tag>
- Date/time:
- Train command:
- Checkpoint:
- Log path:
- Train diagnostics:
  - train_rows:
  - zero_train_loss (%):
  - last_nonzero_train:
  - last_val_loss:
- Eval command:
- Eval output path:
- mean_success_rate:
- Notes / anomalies:
- Decision: continue | stop | retrain
```

## Acceptance Gate for Lab Report

- Do not treat a run as final unless:
  - training logs show healthy non-degenerate behavior after warmup,
  - quick eval is reproducible and non-zero,
  - result and commands are documented in this journal.

## Decision (2026-04-11): Fix Strategy

Chosen strategy is a **controlled short retrain + quick eval gate**, not a long run:

1. Start clean (`pkill` stale jobs, unique run dir, `resume=false`).
2. Train in FP32 (`allow_bf16=false`) for 3 epochs with val/checkpoint every epoch.
3. Hard gate after epoch 1:
   - if logs are degenerate (all/near-all zero loss, no meaningful non-zero values), stop and debug config/data compatibility;
   - if logs are healthy, continue to epoch 3.
4. Run quick eval on the new checkpoint (small `n_test`, limited parallel envs).
5. Promote to long run only if quick eval is reproducible and non-zero.

Rationale:

- Existing 30-epoch run and repeated quick evals produced `mean_success_rate=0.0`, so continuing same recipe is wasteful.
- FP32 + strict logging gate minimizes cost while maximizing diagnostic signal.
- This protocol avoids repeating earlier failure modes (overlapping runs, hidden resume, path confusion).

## Implemented Code Fix (2026-04-11)

- Added built-in zero-loss fail-fast in `third_party/oat/oat/workspace/train_policy.py`:
  - training now raises `RuntimeError` if train loss is non-finite;
  - training now raises `RuntimeError` on `OAT_ZERO_LOSS_PATIENCE` consecutive steps where `train_loss <= OAT_ZERO_LOSS_EPS`.
- Added default guard env vars in `scripts/train_oatpolicy.sh`:
  - `OAT_ZERO_LOSS_PATIENCE=200`
  - `OAT_ZERO_LOSS_EPS=0.0`
- Goal: prevent silent multi-hour runs with collapsed zero loss.

### Root-cause fix candidate (2026-04-11)

- `OATPolicy.forward` previously tokenized actions under `torch.inference_mode()`.
- FSQ quantization uses straight-through tricks; running tokenization under `inference_mode` during training can break autograd expectations and contribute to unstable / degenerate loss behavior.
- Fix: use `inference_mode` only when gradients are disabled; during training forward, tokenize without `inference_mode`.

### Root-cause fix (confirmed by diagnostics)

- Symptom observed on a real batch: tokenized actions had `uniq=1` with all token ids equal to `500`, while raw actions had non-trivial variance.
- Initial hypothesis: missing `action_tokenizer.set_normalizer(...)`.
- Correction: `OATTok` loads its own `normalizer` from the tokenizer checkpoint; overwriting it with dataset-fitted action stats is only valid if those stats exactly match tokenizer training (often false in practice).
- Fix: `OATPolicy.set_normalizer` applies **only observation keys** from the dataset normalizer to `obs_encoder`, and leaves `action_tokenizer.normalizer` as loaded from `OAT_TOK_CKPT`.

### Follow-up fix (checkpoint load / eval parity)

- `BasePolicy.from_checkpoint` historically returned a policy **without** re-fitting/applying the dataset normalizer.
- That made ad-hoc diagnostics (and any code path relying on `from_checkpoint`) disagree with `TrainPolicyWorkspace.run()`.
- Fix: `from_checkpoint` now instantiates `cfg.task.policy.dataset`, computes `get_normalizer()`, and applies it to `workspace.model` (and EMA model if enabled).

## Token batch diagnostics (`scripts/diag_libero_tokens.py`)

This script answers: **on real zarr minibatches, is the path *data → tokenizer normalizer → encoder/FSQ → discrete tokens → policy loss* sane?** It is **not** training and **not** LIBERO sim evaluation; it is a cheap offline probe to avoid guessing from tqdm `train_loss` alone.

### What it does

1. Loads a **policy checkpoint** via `BasePolicy.from_checkpoint` (same path as training/eval checkpoints).
2. Builds `ZarrDataset` against the given `--zarr` (default `data/libero/libero10_N500.zarr` when run from `third_party/oat`).
3. Draws `--batches` random training minibatches.
4. For each batch, under `torch.no_grad()` for the probe branch:
   - logs raw `action` mean/std;
   - runs **tokenizer** `normalizer["action"].normalize`, encoder, quantizer, and prints **token** stats: shape, `zero_frac`, **`uniq`** (count of distinct ids in that batch), `minmax`;
   - also latent / quantized latent mean/std for saturation checks.
5. Calls `policy(batch)` once per batch and prints **`loss_f64`** and whether the loss is finite.

### How to read the printed line

- **`uniq` small (e.g. 1) with wide raw actions** → token collapse or severe saturation in the tokenizer path; investigate tokenizer checkpoint vs zarr, FSQ, or normalizer mismatch (see root-cause section above).
- **`uniq` healthy** but loss looks wrong only **without** `--reset-policy-weights` → likely **collapsed or overfit AR policy head** on that checkpoint; retrain or compare to a fresh init.
- Run **twice**: default, then with **`--reset-policy-weights`** (re-inits policy modules that expose `reset_parameters`). If collapse disappears only after reset, the tokenizer/data path is probably fine and the saved policy weights are the problem.

### Remote / SSH ergonomics

- On Vast, **`uv` is often missing from PATH** in a fresh tmux shell. Either prepend `export PATH="${HOME}/.local/bin:${PATH}"` or use the repo wrapper **`./scripts/run_diag_libero_tokens.sh`** from the repo root (it sets `PATH`, `PYTHONPATH`, picks `uv run` or `third_party/oat/.venv/bin/python`, and can auto-pick the latest `output/manual/**/*.ckpt` when `CKPT` is unset).
- Optional persistence: add the same `PATH` line to `~/.bashrc` on the instance so every login sees `uv`.

### Example commands (from `third_party/oat`)

```bash
export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${HOME}/oat-early-exit/src:${PYTHONPATH:-}"
export CKPT=output/manual/<run_tag>/checkpoints/latest.ckpt
uv run python ../../scripts/diag_libero_tokens.py --ckpt "$CKPT" --zarr data/libero/libero10_N500.zarr --batches 5
uv run python ../../scripts/diag_libero_tokens.py --ckpt "$CKPT" --zarr data/libero/libero10_N500.zarr --batches 5 --reset-policy-weights
```

From repo root (after `git pull`):

```bash
./scripts/run_diag_libero_tokens.sh
./scripts/run_diag_libero_tokens.sh --reset-policy-weights
```

### Fix: zero encoder latents + collapsed tokens after `from_checkpoint` (2026-04-11)

- Symptom: `lat_mean_std=(0,0)`, `uniq=1`, constant token id (e.g. 500), while normalized actions still vary.
- Cause class: `RegisterEncoder` uses `LinearHead` with default **zero** weight init; if the **policy checkpoint** carries a broken / never-updated copy of `action_tokenizer` weights, the head stays at zero and the encoder path collapses before FSQ.
- Fix:
  1. After `load_payload` in `BasePolicy.from_checkpoint`, **re-load `action_tokenizer` state** from `policy.action_tokenizer.checkpoint` on disk (same path Hydra used at build time), for both `workspace.model` and `workspace.ema_model` when present.
  2. `RegisterEncoder` attention mask: cache only a **CPU** mask pattern and `.to(device)` per forward (avoid caching CUDA tensors via `lru_cache`).
  3. `diag_libero_tokens.py`: build `ZarrDataset` with `n_obs_steps` / `n_action_steps` read from the **policy cfg** in the checkpoint (avoids silent horizon mismatch vs training).
  4. `diag_libero_tokens.py`: run **`policy.eval()`** and keep **`action_tokenizer.eval()`** — a whole-policy `.train()` enables dropout inside the frozen tokenizer transformer and can distort encoder probes; encoder probe runs under **autocast disabled** for stable floats.

- **Corrupt OATTok on disk:** if `/root/oattok_libero10.ckpt` (or any ref) still has `|encoder.head.proj|_1≈0` after reload, no policy-side fix can recover it — retrain `train_oattok`. `scripts/inspect_oattok_ckpt.py` checks this in one command; `BasePolicy.from_checkpoint` now **raises** unless `OAT_SKIP_TOK_HEAD_SANITY=1`.

---

## Additional issues & ops notes (for lab review)

Items below were hit in practice; some are not repeated in the numbered log above.

### A) LIBERO / dataset path warning during eval

- **Symptom:** log line that `…/datasets` does not exist (or similar).
- **Impact:** in our runs eval **continued** and completed; treat as environment warning unless eval aborts.
- **Action:** document path; optional fix is aligning LIBERO install / env vars per upstream README.

### B) Eval progress counter vs `n_test`

- **Symptom:** tqdm text like `Eval … 10/30` while `n_test=350`.
- **Explanation:** runner advances in **chunks** of parallel envs; number of steps is **`ceil(n_test / n_parallel_envs)`**, not `n_test`. Not a hang.

### C) Eval does not train the policy

- Full LIBERO sim eval **only measures** success; it does **not** update weights. Shorter `n_test` changes **statistical estimate** and wall-clock, not “partial training”.

### D) Reading overfitting / health — use `val_loss`

- Do not infer training health from the **last few `train_loss` lines** alone; compare **epoch-end** `val_loss` (and curves in `logs.json`).

### E) Git / GitHub vs large artifacts

- `output/`, `*.ckpt`, `experiments/runs/` are **`.gitignore`d** by design; `git pull` will not restore checkpoints.
- **Backup strategy used:** Hugging Face Hub + single **`tar`** of `latest.ckpt` + `logs.json` + `eval_log.json`, copied off the instance with **`rsync -avP`** (resume) when **`scp` stalled at 0 B/s** on ~400 MB.

### F) SSH / Vast.ai

- Minimal `ssh -p … root@IP` **without** `-i ~/.ssh/…`** can fail auth; add **`-4`**, **`ConnectTimeout`**, **`ServerAliveInterval`**, and correct **key**.
- After **instance stop/start**, **IP and port often change** — always copy the fresh `ssh` string from the Vast UI.

### G) Shell hygiene on reconnect

- **`uv` not in PATH** in a new tmux pane → `export PATH="$HOME/.local/bin:$PATH"` (or use repo scripts that set it).
- **`$LOG` / `$CKPT` empty** after opening a new shell → re-export before `grep "$LOG"` or eval commands.
- **`rg` missing** on minimal Ubuntu images → use **`grep -r`**.
- **Duplicate tmux session names** (`eval_libero` twice) → confusing attaches; use **unique session names** per job.

### H) Hugging Face CLI / tokens

- **`export HF_TOKEN= hf_…`** (space after `=`) sets an **empty** token and breaks auth — must be **`export HF_TOKEN=hf_…`** (no space) or use quotes.
- If a token was pasted into a **shared log or chat**, **revoke** it on https://huggingface.co/settings/tokens and create a new one.

### I) `tmux` / process hygiene (duplicate of §2, emphasis)

- Before long retrains: **`pkill`** stale `run_workspace.py` / eval processes; one train + one eval **session** naming convention.

### J) Optional tooling gaps

- No other obvious cruft beyond the above: **`notebooks/.gitkeep`** is only an empty placeholder; **`.pytest_cache/`** is gitignored.

---

## Reviewer quick index

| Topic | Where in this file |
|------|---------------------|
| SSH / timeouts | §1, §F |
| OOM / `copy_to_memory` | §3 |
| WandB headless | §4 |
| CUDA / reboot | §5 |
| Zero / degenerate loss | §6, §“Implemented Code Fix”, §“Root-cause fix” |
| Normalizer / tokenizer collapse | §“Root-cause fix”, §diag script |
| `from_checkpoint` vs training parity | §“Follow-up fix” |
| Eval long / chunks / not training | §9, §B, §C |
| HF / token / backup | §H, §E |
| Final numbers & paths | **“Final pipeline outcome (2026-04-12)”** |
