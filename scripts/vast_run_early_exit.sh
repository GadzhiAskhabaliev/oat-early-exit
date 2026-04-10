#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run offline EarlyExitGate training + threshold sweep on Vast/remote GPU.

Usage:
  ./scripts/vast_run_early_exit.sh --checkpoint /abs/path/to/policy.ckpt [options]

Required:
  --checkpoint PATH          Path to OAT policy checkpoint (.ckpt)

Optional:
  --mse-threshold FLOAT      Default: 0.015
  --epochs INT               Default: 3
  --train-max-batches INT    Default: 200
  --sweep-max-batches INT    Default: 100
  --thresholds "LIST"        Default: "0.7 0.8 0.85 0.9"
  --out-gate PATH            Default: <repo>/checkpoints/early_exit_gate.pt
  --out-csv PATH             Default: <repo>/experiments/runs/sweep_gate_trained.csv
  --device DEVICE            Default: cuda
  --python BIN               Default: "uv run python"
  --skip-train               Skip gate training and run sweep only
  -h, --help                 Show this help

Example:
  ./scripts/vast_run_early_exit.sh \
    --checkpoint /workspace/checkpoints/libero10_policy.ckpt \
    --epochs 3 \
    --train-max-batches 200 \
    --sweep-max-batches 100
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OAT_DIR="${REPO_ROOT}/third_party/oat"

CHECKPOINT=""
MSE_THRESHOLD="0.015"
EPOCHS="3"
TRAIN_MAX_BATCHES="200"
SWEEP_MAX_BATCHES="100"
THRESHOLDS="0.7 0.8 0.85 0.9"
OUT_GATE="${REPO_ROOT}/checkpoints/early_exit_gate.pt"
OUT_CSV="${REPO_ROOT}/experiments/runs/sweep_gate_trained.csv"
DEVICE="cuda"
PYTHON_CMD="uv run python"
SKIP_TRAIN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --mse-threshold) MSE_THRESHOLD="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --train-max-batches) TRAIN_MAX_BATCHES="$2"; shift 2 ;;
    --sweep-max-batches) SWEEP_MAX_BATCHES="$2"; shift 2 ;;
    --thresholds) THRESHOLDS="$2"; shift 2 ;;
    --out-gate) OUT_GATE="$2"; shift 2 ;;
    --out-csv) OUT_CSV="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --python) PYTHON_CMD="$2"; shift 2 ;;
    --skip-train) SKIP_TRAIN="1"; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${CHECKPOINT}" ]]; then
  echo "Error: --checkpoint is required."
  usage
  exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Error: checkpoint not found: ${CHECKPOINT}"
  exit 1
fi

if [[ ! -d "${OAT_DIR}" ]]; then
  echo "Error: OAT directory not found: ${OAT_DIR}"
  exit 1
fi

mkdir -p "$(dirname "${OUT_GATE}")" "$(dirname "${OUT_CSV}")"

export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/src:${OAT_DIR}:${PYTHONPATH:-}"

cd "${OAT_DIR}"

echo "=== Vast early-exit pipeline ==="
echo "Checkpoint: ${CHECKPOINT}"
echo "Device: ${DEVICE}"
echo "Out gate: ${OUT_GATE}"
echo "Out CSV: ${OUT_CSV}"

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  echo "=== Step 1/2: offline gate training ==="
  ${PYTHON_CMD} "../../scripts/train_early_exit_offline.py" \
    --checkpoint "${CHECKPOINT}" \
    --mse-threshold "${MSE_THRESHOLD}" \
    --epochs "${EPOCHS}" \
    --max-batches "${TRAIN_MAX_BATCHES}" \
    --device "${DEVICE}" \
    --out-gate "${OUT_GATE}"
else
  echo "=== Step 1/2: skipped training (--skip-train) ==="
fi

echo "=== Step 2/2: threshold sweep (gate mode) ==="
${PYTHON_CMD} "../../scripts/sweep_early_exit.py" \
  --checkpoint "${CHECKPOINT}" \
  --mode gate \
  --gate "${OUT_GATE}" \
  --thresholds ${THRESHOLDS} \
  --max-batches "${SWEEP_MAX_BATCHES}" \
  --device "${DEVICE}" \
  --out-csv "${OUT_CSV}"

echo "=== Done ==="
echo "Gate checkpoint: ${OUT_GATE}"
echo "Sweep CSV: ${OUT_CSV}"
