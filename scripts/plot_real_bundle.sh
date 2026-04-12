#!/usr/bin/env bash
# Plot training / eval / (optional) sweep PNGs into docs/assets/ using the same dark theme as the demo.
#
# Usage:
#   1) Copy logs.json + eval_log.json (+ optional sweep.csv) from the server into a local folder, e.g. ./artifacts/server_run/
#   2) From repo root:
#        ./scripts/plot_real_bundle.sh ./artifacts/server_run
#      or with sweep:
#        ./scripts/plot_real_bundle.sh ./artifacts/server_run ./artifacts/server_run/sweep_gate.csv
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="${1:?Pass local directory that contains logs.json (JSONL) and eval_log.json}"
SWEEP="${2:-}"

LOGS="${DIR}/logs.json"
EVAL="${DIR}/eval_log.json"

if [[ ! -f "${LOGS}" ]]; then
  echo "Missing: ${LOGS}" >&2
  exit 1
fi
if [[ ! -f "${EVAL}" ]]; then
  echo "Missing: ${EVAL}" >&2
  exit 1
fi

PY="${ROOT}/third_party/oat/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  PY="$(command -v python3 || true)"
fi
if [[ -z "${PY}" ]]; then
  echo "No python found. Run ./scripts/install_oat.sh first." >&2
  exit 1
fi

"${PY}" -c "import matplotlib" 2>/dev/null || { echo "Install matplotlib: pip install matplotlib" >&2; exit 1; }

OUT="${ROOT}/docs/assets"
mkdir -p "${OUT}"

echo "==> Training curves from ${LOGS}"
"${PY}" "${ROOT}/scripts/plot_training_logs.py" --logs "${LOGS}" --out "${OUT}/figure_training_curves.png"

echo "==> Eval summary from ${EVAL}"
"${PY}" "${ROOT}/scripts/plot_eval_log.py" --eval-log "${EVAL}" --out "${OUT}/figure_eval_summary.png"

if [[ -n "${SWEEP}" && -f "${SWEEP}" ]]; then
  echo "==> Early-exit sweep from ${SWEEP}"
  "${PY}" "${ROOT}/scripts/plot_sweep_csv.py" --csv "${SWEEP}" --out "${OUT}/figure_early_exit_sweep.png"
else
  echo "==> No sweep CSV passed; keeping existing figure_early_exit_sweep.png (re-run with 2nd arg to refresh)"
fi

echo "Done. Figures in ${OUT}/"
