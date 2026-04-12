#!/usr/bin/env bash
# Prebuilt multitask libero10 zarr (see OAT README). On HTTP 429 from Hugging Face, retry later or download manually.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OAT="${ROOT}/third_party/oat"
DEST="${OAT}/data/libero"
ZIP_NAME="libero10_N500.zarr.zip"
URL="https://huggingface.co/datasets/chaoqi-liu/libero10_N500.zarr/resolve/main/${ZIP_NAME}?download=true"
OUT_Z="${DEST}/${ZIP_NAME}"

mkdir -p "${DEST}"
if [ -d "${DEST}/libero10_N500.zarr" ]; then
  echo "Already present: ${DEST}/libero10_N500.zarr"
  exit 0
fi

echo "Downloading ${ZIP_NAME} into ${DEST} (retries on 429)..."
echo "Tip: curl -C - resumes a partial .zip after SSH drops."
for i in 1 2 3 4 5; do
  if curl -L -C - --fail --retry 3 --retry-delay 10 -o "${OUT_Z}" "${URL}"; then
    break
  fi
  echo "Attempt $i failed, sleeping 60s..."
  sleep 60
done

if [ ! -s "${OUT_Z}" ]; then
  echo "Download failed. Open in a browser or use git-lfs / huggingface-cli:"
  echo "  ${URL}"
  exit 1
fi

unzip -q -o "${OUT_Z}" -d "${DEST}"
rm -f "${OUT_Z}"
echo "Done: ${DEST}/libero10_N500.zarr"
