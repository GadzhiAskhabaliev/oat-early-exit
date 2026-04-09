#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OAT="${ROOT}/third_party/oat"

# uv из стандартной установки (curl https://astral.sh/uv/install.sh)
export PATH="${HOME}/.local/bin:${PATH}"

echo "==> Submodule LIBERO (если ещё не подтянут)"
cd "${OAT}"
git submodule update --init --recursive

if command -v uv >/dev/null 2>&1; then
  echo "==> CMake для сборки egl-probe (robomimic): pip-пакет cmake + CMAKE_POLICY_VERSION_MINIMUM"
  cd "${OAT}"
  if [ ! -d .venv ]; then
    uv venv
  fi
  uv pip install cmake --python .venv/bin/python
  CMAKE_BIN="$(.venv/bin/python -c "import pathlib, cmake; print(pathlib.Path(cmake.__file__).parent / 'data' / 'bin')")"
  export PATH="${CMAKE_BIN}:${PATH}"
  # CMake 4.x: старый CMakeLists.txt в egl-probe требует политику >=3.5
  export CMAKE_POLICY_VERSION_MINIMUM=3.5

  echo "==> uv sync + editable install"
  uv sync
  uv pip install -e .
else
  echo "uv не найден. Установите: https://docs.astral.sh/uv/"
  echo "Либо: cd \"${OAT}\" && pip install -e .  (может потребовать ручной установки зависимостей из pyproject.toml)"
  exit 1
fi

echo "==> Зависимости лаборатории (pytest, omegaconf)"
cd "${ROOT}"
pip install -r requirements.txt

echo "Готово. Baseline: scripts/train_baseline.sh"
