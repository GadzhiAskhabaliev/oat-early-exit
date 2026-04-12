#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OAT="${ROOT}/third_party/oat"

# uv: https://docs.astral.sh/uv/ (curl install script)
export PATH="${HOME}/.local/bin:${PATH}"

cd "${OAT}"

if command -v uv >/dev/null 2>&1; then
  echo "==> CMake for egl-probe (robomimic): pip cmake + CMAKE_POLICY_VERSION_MINIMUM"
  cd "${OAT}"
  if [ ! -d .venv ]; then
    uv venv
  fi
  uv pip install cmake --python .venv/bin/python
  CMAKE_BIN="$(.venv/bin/python -c "import pathlib, cmake; print(pathlib.Path(cmake.__file__).parent / 'data' / 'bin')")"
  export PATH="${CMAKE_BIN}:${PATH}"
  # CMake 4.x: legacy egl-probe CMakeLists needs policy >=3.5
  export CMAKE_POLICY_VERSION_MINIMUM=3.5

  echo "==> uv sync + editable install"
  uv sync
  uv pip install -e .
else
  echo "uv not found. Install: https://docs.astral.sh/uv/"
  echo "Or: cd \"${OAT}\" && pip install -e .  (you may need to resolve deps from pyproject.toml manually)"
  exit 1
fi

echo "==> Lab extras (pytest, omegaconf, …) into OAT venv"
cd "${ROOT}"
uv pip install -r requirements.txt --python "${OAT}/.venv/bin/python"

echo "Done. Baseline: scripts/train_baseline.sh"
