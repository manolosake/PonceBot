#!/usr/bin/env bash
set -euo pipefail

# Idempotent installer for Piper (local/free TTS) used by PonceBot voice replies.
#
# Installs the official Piper Linux x86_64 bundle into:  <repo>/bin/piper/
# Models are NOT downloaded here (they are large and voice-choice is subjective).
# Place your model at: <repo>/models/piper/<voice>.onnx (+ .onnx.json).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PIPER_VERSION="${PIPER_VERSION:-2023.11.14-2}"
PIPER_TARBALL_URL="https://github.com/rhasspy/piper/releases/download/${PIPER_VERSION}/piper_linux_x86_64.tar.gz"

DEST_DIR="${ROOT_DIR}/bin/piper"
BIN_PATH="${DEST_DIR}/piper"

if [[ -x "${BIN_PATH}" ]]; then
  echo "piper already installed: ${BIN_PATH}"
  exit 0
fi

tmpdir="$(mktemp -d)"
cleanup() { rm -rf "${tmpdir}"; }
trap cleanup EXIT

echo "Downloading Piper ${PIPER_VERSION}..."
curl -L --fail -o "${tmpdir}/piper_linux_x86_64.tar.gz" "${PIPER_TARBALL_URL}"
tar -xzf "${tmpdir}/piper_linux_x86_64.tar.gz" -C "${tmpdir}"

rm -rf "${DEST_DIR}"
mkdir -p "${DEST_DIR}"
cp -a "${tmpdir}/piper/." "${DEST_DIR}/"
chmod +x "${DEST_DIR}/piper" "${DEST_DIR}/piper_phonemize" 2>/dev/null || true

echo "Installed: ${BIN_PATH}"
echo
echo "Next: configure a model path in codexbot.env, e.g.:"
echo "  BOT_TTS_PIPER_MODEL_PATH=${ROOT_DIR}/models/piper/es_MX-ald-medium.onnx"

