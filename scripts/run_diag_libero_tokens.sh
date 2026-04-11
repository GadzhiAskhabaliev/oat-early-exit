#!/usr/bin/env bash
# Один вход для диагностики LIBERO-токенов на Vast/SSH без ручного PATH и без uv в tmux.
#
# Из корня репо:
#   ./scripts/run_diag_libero_tokens.sh
#   CKPT=/path/to/latest.ckpt ./scripts/run_diag_libero_tokens.sh
#   ./scripts/run_diag_libero_tokens.sh --reset-policy-weights
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OAT="${ROOT}/third_party/oat"
export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

ZARR="${ZARR:-data/libero/libero10_N500.zarr}"
BATCHES="${BATCHES:-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"

cd "${OAT}"

pick_python() {
  if command -v uv >/dev/null 2>&1; then
    echo "uv run python"
    return
  fi
  if [[ -x "${OAT}/.venv/bin/python" ]]; then
    echo "${OAT}/.venv/bin/python"
    return
  fi
  echo "Error: ни uv (~/.local/bin), ни ${OAT}/.venv/bin/python не найдены." >&2
  echo "Запусти с сервера: cd ${ROOT} && ./scripts/install_oat.sh" >&2
  exit 1
}

PYRUN="$(pick_python)"
if [[ "${PYRUN}" == "uv run python" ]]; then
  runpy() { uv run python "$@"; }
else
  runpy() { "${PYRUN}" "$@"; }
fi

if [[ -z "${CKPT:-}" ]]; then
  CKPT="$(find output/manual -type f -name '*.ckpt' 2>/dev/null | xargs -r ls -t 2>/dev/null | head -1 || true)"
fi
if [[ -z "${CKPT}" ]]; then
  echo "Error: нет CKPT. Задай явно:" >&2
  echo "  CKPT=/path/to/latest.ckpt ${ROOT}/scripts/run_diag_libero_tokens.sh" >&2
  exit 1
fi

echo "==> CKPT=${CKPT}"
echo "==> zarr=${ZARR} batches=${BATCHES} python=${PYRUN}"

EXTRA=("$@")
runpy "${ROOT}/scripts/diag_libero_tokens.py" \
  --ckpt "${CKPT}" \
  --zarr "${ZARR}" \
  --batch-size "${BATCH_SIZE}" \
  --batches "${BATCHES}" \
  "${EXTRA[@]}"
