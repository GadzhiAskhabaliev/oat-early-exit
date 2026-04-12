#!/usr/bin/env bash
# Build a single .tgz with policy checkpoint + training logs + eval log (+ any sweep CSVs).
# Run on the GPU server from repo root after training/eval.
#
# Defaults match the LIBERO train30 + eval layout from this lab; override with env vars:
#   OAT_BACKUP_RUN_DIR   — e.g. third_party/oat/output/manual/train30_20260411_134306
#   OAT_BACKUP_EVAL_DIR  — e.g. experiments/runs/eval_libero_7to8h_20260412_112444
#   OAT_BACKUP_TAR       — output path (default: ~/oat_lab_backup_YYYYMMDD_HHMMSS.tgz)
#
# Example:
#   cd ~/oat-early-exit && ./scripts/tar_lab_backup.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

RUN_DIR="${OAT_BACKUP_RUN_DIR:-third_party/oat/output/manual/train30_20260411_134306}"
EVAL_DIR="${OAT_BACKUP_EVAL_DIR:-experiments/runs/eval_libero_7to8h_20260412_112444}"
OUT="${OAT_BACKUP_TAR:-${HOME}/oat_lab_backup_$(date +%Y%m%d_%H%M%S).tgz}"

CKPT="${RUN_DIR}/checkpoints/latest.ckpt"
LOGS="${RUN_DIR}/logs.json"
EVAL="${EVAL_DIR}/eval_log.json"

if [[ ! -f "${CKPT}" ]]; then
  echo "Error: checkpoint not found: ${CKPT}" >&2
  echo "Set OAT_BACKUP_RUN_DIR to your run directory." >&2
  exit 1
fi

shopt -s nullglob
RUN_CSVS=(experiments/runs/*.csv)

LIST=("${CKPT}")
[[ -f "${LOGS}" ]] && LIST+=("${LOGS}") || echo "WARN: missing ${LOGS} (archive will omit it)" >&2
[[ -f "${EVAL}" ]] && LIST+=("${EVAL}") || echo "WARN: missing ${EVAL} (archive will omit it)" >&2
for f in "${RUN_CSVS[@]}"; do
  LIST+=("$f")
done

echo "==> Archiving into ${OUT}"
echo "    Files: ${LIST[*]}"
tar -czvf "${OUT}" "${LIST[@]}"
echo "==> Done: ${OUT} ($(du -h "${OUT}" | cut -f1))"
echo "    Download from laptop, e.g.:"
echo "    scp -P PORT -i ~/.ssh/vast_ai root@HOST:${OUT} ."
