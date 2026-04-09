#!/usr/bin/env bash
# Готовый multitask zarr libero10 (см. README OAT). При 429 от Hugging Face — повторите позже или скачайте вручную.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OAT="${ROOT}/third_party/oat"
DEST="${OAT}/data/libero"
ZIP_NAME="libero10_N500.zarr.zip"
URL="https://huggingface.co/datasets/chaoqi-liu/libero10_N500.zarr/resolve/main/${ZIP_NAME}?download=true"
OUT_Z="${DEST}/${ZIP_NAME}"

mkdir -p "${DEST}"
if [ -d "${DEST}/libero10_N500.zarr" ]; then
  echo "Уже есть: ${DEST}/libero10_N500.zarr"
  exit 0
fi

echo "Скачивание ${ZIP_NAME} в ${DEST} (несколько попыток при 429)..."
for i in 1 2 3 4 5; do
  if curl -L --fail --retry 3 --retry-delay 10 -o "${OUT_Z}" "${URL}"; then
    break
  fi
  echo "Попытка $i не удалась, жду 60 с..."
  sleep 60
done

if [ ! -s "${OUT_Z}" ]; then
  echo "Не удалось скачать. Откройте в браузере или используйте git-lfs/huggingface-cli:"
  echo "  ${URL}"
  exit 1
fi

unzip -q -o "${OUT_Z}" -d "${DEST}"
rm -f "${OUT_Z}"
echo "Готово: ${DEST}/libero10_N500.zarr"
