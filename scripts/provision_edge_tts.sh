#!/usr/bin/env bash
set -euo pipefail

# Installs edge-tts into the repo venv for high-quality free online TTS.
# Note: edge-tts sends the TTS text to Microsoft (online).
#
# Usage:
#   ./scripts/provision_edge_tts.sh

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

PY="${HERE}/.venv/bin/python"
PIP="${HERE}/.venv/bin/pip"

if [[ ! -x "${PY}" || ! -x "${PIP}" ]]; then
  echo "Missing venv at ${HERE}/.venv (create it first)."
  exit 1
fi

echo "Installing edge-tts into venv..."
"${PIP}" install --upgrade pip >/dev/null
"${PIP}" install --upgrade edge-tts >/dev/null

echo "OK. Verify with:"
echo "  ${PY} -c 'import edge_tts; print(edge_tts.__version__)'"

