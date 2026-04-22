#!/usr/bin/env bash
set -euo pipefail

# Canonical deterministic pytest bootstrap for clean-shell replay.
# Usage example:
#   env -i PATH=/usr/bin:/bin HOME="$HOME" ./scripts/bootstrap_pytest_python3.sh -m pytest --version

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
PY_BIN="${BOOTSTRAP_PYTEST_PYTHON_BIN:-python3}"
VENV_DIR="${BOOTSTRAP_PYTEST_VENV_DIR:-$ROOT_DIR/.codexbot_tmp/pytest_python3_venv}"
VENV_PY="$VENV_DIR/bin/python"

mkdir -p "$(dirname "$VENV_DIR")"

if [[ ! -x "$VENV_PY" ]]; then
  "$PY_BIN" -m venv "$VENV_DIR"
fi

"$VENV_PY" -m pip install --upgrade pip pytest >/dev/null

if [[ "$#" -eq 0 ]]; then
  set -- -m pytest --version
fi

exec "$VENV_PY" "$@"
