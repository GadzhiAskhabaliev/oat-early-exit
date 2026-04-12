#!/usr/bin/env bash
# Train OAT policy (train_oatpolicy) on top of a frozen OATTok checkpoint.
# For Vast/headless: defaults OAT_DISABLE_WANDB=1 and copy_to_memory=false (lower RAM).
#
# Needs: libero10 zarr (third_party/oat/data/libero/libero10_N500.zarr), GPU, uv/.venv from install_oat.sh.
#
# Example:
#   export OAT_TOK_CKPT=/root/oattok_libero10.ckpt
#   ./scripts/train_oatpolicy.sh
# Default "stable" recipe:
#   bf16 off, num_epochs=30, val/checkpoint every epoch, resume=false.
# Override any Hydra key at the end of the command line, e.g.:
#   ./scripts/train_oatpolicy.sh training.num_epochs=10 training.allow_bf16=true
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OAT="${ROOT}/third_party/oat"
export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

if [[ -z "${OAT_TOK_CKPT:-}" ]]; then
  echo "Error: set OAT_TOK_CKPT to the OATTok checkpoint path (e.g. .../train_oattok.../checkpoints/latest.ckpt)" >&2
  exit 1
fi

# Refuse to launch multi-hour policy training on a broken tokenizer (zero LinearHead).
# Bypass: OAT_SKIP_OATTOK_INSPECT=1
if [[ "${OAT_SKIP_OATTOK_INSPECT:-0}" != "1" ]]; then
  echo "==> OATTok sanity (inspect_oattok_ckpt.py) on ${OAT_TOK_CKPT}"
  ( cd "${OAT}" && uv run python "${ROOT}/scripts/inspect_oattok_ckpt.py" "${OAT_TOK_CKPT}" )
fi

export OAT_DISABLE_WANDB="${OAT_DISABLE_WANDB:-1}"
# Fail fast on silent zero-loss collapse (set 0 to disable).
export OAT_ZERO_LOSS_PATIENCE="${OAT_ZERO_LOSS_PATIENCE:-200}"
export OAT_ZERO_LOSS_EPS="${OAT_ZERO_LOSS_EPS:-0.0}"

RUN_TAG="${OAT_RUN_TAG:-train30_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="output/manual/${RUN_TAG}"

BASE_OVERRIDES=(
  "hydra.run.dir=${RUN_DIR}"
  training.resume=false
  training.allow_bf16=false
  training.num_epochs=30
  training.val_every=1
  training.checkpoint_every=1
)

cd "${OAT}"
HYDRA_FULL_ERROR=1 MUJOCO_GL=egl uv run accelerate launch \
  --num_processes "${OAT_NUM_PROCESSES:-1}" \
  scripts/run_workspace.py \
  --config-name=train_oatpolicy \
  task/policy=libero/libero10 \
  task.policy.lazy_eval=true \
  "policy.action_tokenizer.checkpoint=${OAT_TOK_CKPT}" \
  task.policy.dataset.zarr_path=data/libero/libero10_N500.zarr \
  task.policy.dataset.copy_to_memory=false \
  dataloader.batch_size=32 \
  val_dataloader.batch_size=32 \
  dataloader.num_workers=0 \
  val_dataloader.num_workers=0 \
  dataloader.persistent_workers=false \
  val_dataloader.persistent_workers=false \
  training.use_ema=false \
  training.gradient_accumulate_every=4 \
  "${BASE_OVERRIDES[@]}" \
  "$@"
