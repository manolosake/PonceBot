#!/usr/bin/env bash
set -euo pipefail

# Ensure a python3 executable with pytest is available in clean-shell contexts.
# The fallback is repo-local only (.venv), avoiding host-global preconfiguration.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 -m pytest [args...]" >&2
  exit 2
fi

if ! python3 -m pytest --version >/dev/null 2>&1; then
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    python3 -m venv "${VENV_DIR}"
  fi
  if ! "${VENV_DIR}/bin/python" -m pytest --version >/dev/null 2>&1; then
    "${VENV_DIR}/bin/python" -m pip install --quiet --upgrade pip pytest
  fi
  export PATH="${VENV_DIR}/bin:${PATH:-/usr/bin:/bin}"
fi

exec python3 "$@"
