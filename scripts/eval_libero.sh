#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OAT="${ROOT}/third_party/oat"
cd "${OAT}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <path/to/oatpolicy.ckpt> [extra args to eval_policy_sim.py]"
  echo "Tune eval wall time, e.g.: --n-test 350 --n-parallel-envs 12 --n-test-vis 0"
  exit 1
fi

CHECKPOINT="$1"
shift
uv run scripts/eval_policy_sim.py \
  --checkpoint "${CHECKPOINT}" \
  --output_dir "${ROOT}/experiments/runs/eval_libero" \
  "$@"
