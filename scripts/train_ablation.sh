#!/usr/bin/env bash
set -euo pipefail
# После интеграции BLT/H-Net: передайте hydra-overrides вторым аргументом или через "$@"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
exec "${ROOT}/scripts/train_baseline.sh" "$@"
