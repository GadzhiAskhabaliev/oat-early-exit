# Course test assignment: VLA tokenization and researcher assistants — submission pack

The official brief (Part 1: tool survey; Part 2: research + report on BLT / H-Net / OAT with code changes and experiments) comes from the instructors. Below is a **structured submission of all four deliverables** in one place. Reproduction details and the early-exit hypothesis are also in [`early-exit.md`](early-exit.md).

**Links from the assignment brief:**  
[1] Deep Research — https://openai.com/index/introducing-deep-research/  
[2] Claude Code — https://claude.com/product/claude-code  
[3] AutoResearch — https://github.com/karpathy/autoresearch  
[4] AutoResearchClaw — https://github.com/aiming-lab/AutoResearchClaw  
[5] BLT — https://arxiv.org/abs/2412.09871v1  
[6] H-Net — https://arxiv.org/abs/2507.07955  
[7] OAT (code) — https://github.com/Chaoqi-LIU/oat  

---

## Part 1. Tool survey (research assistants)

### 1.1 Tools used in practice

| Tool | Role in the project | Pros | Cons / risks |
|------|----------------------|------|----------------|
| **Cursor (IDE + Agent)** | Main loop: navigate `third_party/oat`, edit code, search the repo, run commands, iterate on build/import errors | File context, fast diffs, integrated terminal | Discipline: avoid huge diffs; validate hypotheses with logs, not by eye |
| **Chat assistant (browser LLM or similar)** | Paper skim, informal hypothesis shaping, draft report text | Faster onboarding for BLT/H-Net/OAT | Detail hallucinations; must cross-check arXiv/README |
| **Git + GitHub** | Version control, `git push`/`pull` between laptop and GPU instance | Reproducibility, code backup | Artifacts (`output/`, `*.ckpt`) are `.gitignore`d — transfer separately (`scp`, `tar`) |
| **Vast.ai + SSH + tmux** | Long train/eval without tying to a local laptop | Cheaper than owning a GPU | Unstable SSH, port changes; disk loss risk when the instance stops |
| **uv + Hydra** (OAT stack) | Env and train/eval configs | Repeatable `override`s | Learning curve: two sources of truth (yaml vs shell) — document explicitly |

### 1.2 Mapping to the brief’s tool list

- **Deep Research [1]** — good for **broad surveys** and collecting links; here that role was partly covered by **manual arXiv + OAT README** plus targeted chat-LLM prompts.
- **Claude Code [2]** — conceptually close to **Cursor agent mode**: edits in an existing repo, multi-step tasks. We did not rely on a separate “Claude Code” binary; Cursor Agent + commit discipline sufficed.
- **AutoResearch / AutoResearchClaw [3][4]** — closed loop “idea → code → experiment”; **not used end-to-end in autopilot** (risk of silent errors on expensive GPU time). Instead we used a **manual loop**: run journal, smoke tests, then long training.

**Part 1 takeaway:** for an engineering task “fork + patch + LIBERO”, **Cursor + git + remote GPU** is a good fit; keep “auto-research” for literature review, and run critical code/experiments **under control** with logs and checkpoints.

---

## Part 2. Research thread: paper motivation and hypothesis

### 2.1 What BLT, H-Net, and OAT do (idea level, not a verbatim recap)

- **BLT [5]** — **local / hierarchical** processing and a more economical stream representation (byte-like patterns, “not everything matters equally” at each step). For us the key pull is **adaptive compute**: do not always deploy “full power”.
- **H-Net [6]** — **hierarchy** and task decomposition; for VLA, manipulation unfolds in time (approach, grasp, release) — natural to think about **different “mental plan” lengths** in tokens.
- **OAT [7]** — **ordered action tokenization** and an autoregressive policy over discrete action tokens; strong **VLA** link: observation → token sequence → action.

### 2.2 How **our** hypothesis (early exit) emerged

The brief suggested something like “**integrating BLT and OAT**”. This repository implements a **meaningful pragmatic variant**, not a literal fused backbone:

> **Hypothesis:** during autoregressive **OAT action-token** generation, **stop decoding earlier** when the model is already “confident” (or reconstructs the action well from a prefix), and **only complete the full horizon** in hard cases — a **latency vs quality** trade-off.

This aligns with the **spirit** of BLT/H-Net (do not spend full compute / length where a short prefix suffices) and builds on **OAT** as the AR substrate. Implementation: `EarlyExitGate`, `max_prob` heuristic, `transformer_cache` — see [`early-exit.md`](early-exit.md).

### 2.3 What was implemented in code (short)

- Extension **`src/oat_ext/`** (gate, configs, tests).
- Patches under **`third_party/oat/`** (inference, policy, workspace as needed).
- Scripts: `scripts/train_oatpolicy.sh`, `scripts/eval_libero.sh`, offline gate / sweep — see root [`README.md`](../README.md).

### 2.4 Experiments (LIBERO cycle we actually ran)

1. **Policy training (OAT on top of a frozen OATTok):** stable FP32 recipe, 30 epochs, validation every epoch, zarr `libero10_N500`, `lazy_eval=true`. Run: `output/manual/train30_20260411_134306/`, log `logs.json`, checkpoint `checkpoints/latest.ckpt` (~423 MiB).
2. **Simulation eval:** `eval_policy_sim.py` with CLI **`--n-test` / `--n-parallel-envs` / `--n-test-vis`** (on `main`), shorter episode budget for manageable wall-clock.
3. **Eval outcome (example successful run):** `mean_success_rate_mean ≈ 0.223` with `n_test=350`, `n_parallel_envs=12`, `n_test_vis=0`; artifact: `experiments/runs/eval_libero_7to8h_20260412_112444/eval_log.json` (on the server; paths may differ after copy).

*Note:* full `n_test=500` and video visualization increase runtime; for paper-style comparison, run and document a separate “full eval”.

---

## Part 3. Assistant usage logs

**Brief requirement:** attach logs of conversations with assistants.

**What to attach physically:**

1. **Cursor export** (Chat / Composer / Agent) for the repo work period — PDF or `.md` / `.txt`, next to the report or as an archive link.
2. **GPU run journal:** [`libero-debug-journal.md`](libero-debug-journal.md) — commands, paths, incidents (SSH, OOM, zero loss, eval), fixes.

*For a final ZIP to graders:* e.g. `exports/cursor_chat_YYYYMMDD.md` + a copy of `docs/libero-debug-journal.md`.

---

## Part 4. Reproduction for graders (“Claude Code” style)

### 4.1 Clone and environment

```bash
git clone https://github.com/GadzhiAskhabaliev/oat-early-exit.git
cd oat-early-exit
./scripts/install_oat.sh
export PATH="$HOME/.local/bin:$PATH"
```

You need: **GPU**, **LIBERO-10 zarr** (`./scripts/download_libero10_zarr.sh` or an existing `third_party/oat/data/libero/libero10_N500.zarr`), **OATTok** checkpoint (`OAT_TOK_CKPT`).

### 4.2 Policy training (example)

```bash
export OAT_TOK_CKPT=/path/to/oattok_libero10.ckpt
./scripts/train_oatpolicy.sh
# optional Hydra overrides at end of line; see comments in train_oatpolicy.sh
```

Key files: `third_party/oat/scripts/run_workspace.py`, `third_party/oat/oat/workspace/train_policy.py`, config `third_party/oat/oat/config/train_oatpolicy.yaml` + shell overrides.

### 4.3 LIBERO eval (runtime tuning)

```bash
cd third_party/oat
export PATH="$HOME/.local/bin:$PATH"

CKPT=/path/to/output/manual/<run>/checkpoints/latest.ckpt
OUT=/path/to/experiments/runs/eval_$(date +%Y%m%d_%H%M%S)

uv run scripts/eval_policy_sim.py -c "$CKPT" -o "$OUT" -n 1 \
  --n-test 350 \
  --n-parallel-envs 12 \
  --n-test-vis 0
```

Closer to checkpoint defaults: `--n-test 500 --n-parallel-envs 20 --n-test-vis 0` (slower).

Output: **`$OUT/eval_log.json`** (includes `mean_success_rate_mean`, `env_runner_resolved`).

### 4.4 Early exit (optional)

See [`early-exit.md`](early-exit.md), `scripts/vast_run_early_exit.sh`, `pytest` under `tests/`.

---

## Part 5. Analytical report (draft) + author commentary

### 5.1 Draft (paper skeleton)

**Working title:** Adaptive early exit for autoregressive OAT action tokens in VLA-style manipulation policies.

**Abstract.** Visuomotor policies based on autoregression over discrete action tokens (OAT) pay the cost of full decode length at every control step. We propose **early exit** during token generation: a learnable gate and a confidence heuristic. The intended effect is lower average inference latency with controlled quality risk. The implementation lives in an OAT fork; experiments include offline proxy metrics and (with GPU) LIBERO simulation eval.

**1. Introduction.** VLA and action tokenization; adaptive compute motivation (ideas from hierarchy/locality [5][6] and ordered tokens [7]).

**2. Related work.** BLT, H-Net, OAT — briefly; explicit link: **early stopping of generation** as a practical compute lever.

**3. Method.** `EarlyExitGate`, KV cache, hooks in `generate()`, gate vs max-prob modes (see `early-exit.md`).

**4. Experiments.**  
- Train: LIBERO-10, zarr 500 demos, `train_loss` / `val_loss` per epoch.  
- Eval: `mean_success_rate` in sim; **state explicitly** `n_test`, parallelism, `n_test_vis`.  
- Proxy metrics without full sim (if used) — sweep CSV.  
- **Filled environment/results matrix (English):** see *Filled snapshot (this project, April 2026)* in [`experiments-section-template.md`](experiments-section-template.md) — copy into the final PDF report.

**5. Results.** Example: final-epoch `val_loss` ~2.22; sim success ~22% with shortened `n_test=350` (do not mix with full-500 without a separate run).

**6. Limitations.** Depends on OATTok quality; `lazy_eval` and no rollout metrics during train; checkpointing by `mean_success_rate` vs `val_loss` (see journal); LIBERO `datasets` path warnings on some installs.

**7. Conclusion.** Early exit as a cheap engineering acceleration lever; next — gate calibration, full eval, wall-clock measurement.

### 5.2 Author commentary (replace with your own paragraph)

> **Template (replace with your own text):**  
> “In this assignment I used OAT as the working base for VLA tokenization and took **adaptive budget** motivation from lines of work like BLT/H-Net not as a literal architecture merge, but as a **research thesis**: full AR decode length is not always necessary. In practice most time went into stabilizing the pipeline (data, normalization, checkpoints, remote GPU). The early-exit hypothesis is implemented in code, but the **main measurable win of this cycle** for me is a **non-zero LIBERO success rate** after long policy training; next I would separately measure **inference speedup** and gate vs heuristic ablations. What did not work out: ___ (e.g. full `n_test=500` within time budget / automatic top-k by `val_loss` without config work).”

---

## Pre-submission checklist

- [ ] PDF/archive for **Part 1** (tool survey) — can be this file plus extra tables.
- [ ] **Assistant logs** (Cursor export + `libero-debug-journal.md`).
- [ ] **Repository** (GitHub link) + this file under `docs/`.
- [ ] **Report:** Section 5 above + **one paragraph** of personal conclusions at the end of the report as its own block.
- [ ] **Artifact backup** (`latest.ckpt`, `eval_log.json`, `logs.json`) off the Vast instance. Steps: **“Before you delete the instance”** in root [`README.md`](../README.md) (`tar` → `scp`/`rsync` → laptop `artifacts/`; optional Hugging Face Hub or Zenodo).
