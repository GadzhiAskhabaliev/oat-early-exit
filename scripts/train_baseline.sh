#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OAT="${ROOT}/third_party/oat"
export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export OAT_DISABLE_WANDB="${OAT_DISABLE_WANDB:-1}"

cd "${OAT}"
# Нужен датасет libero10 zarr (third_party/oat/README.md).
# Обучение рассчитано на GPU (CUDA): --num_processes = число используемых GPU.
# Одна карта: оставьте --num_processes 1; при необходимости: CUDA_VISIBLE_DEVICES=0
HYDRA_FULL_ERROR=1 uv run accelerate launch \
  --num_processes 1 \
  scripts/run_workspace.py \
  --config-name=train_oattok \
  task/tokenizer=libero/libero10 \
  "$@"
