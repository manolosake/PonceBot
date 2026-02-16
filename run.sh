#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

ENV_FILE="${ENV_FILE:-$HERE/codexbot.env}"
LOCAL_ENV_FILE="${ENV_LOCAL_FILE:-$HERE/.env.local}"
CHECK_ONLY=0
if [[ "${1:-}" == "--check-env" ]]; then
  CHECK_ONLY=1
fi

load_env_file() {
  local file="$1"
  local line key value lineno=0
  while IFS= read -r line || [[ -n "$line" ]]; do
    lineno=$((lineno + 1))
    line="${line%$'\r'}"
    [[ -z "${line//[[:space:]]/}" ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue

    if [[ "$line" =~ ^[[:space:]]*export[[:space:]]+ ]]; then
      line="${line#export }"
      line="${line#"${line%%[![:space:]]*}"}"
    fi

    if [[ "$line" != *=* ]]; then
      echo "Invalid env line $lineno in $file: missing '='"
      return 1
    fi

    key="${line%%=*}"
    value="${line#*=}"

    key="${key#"${key%%[![:space:]]*}"}"
    key="${key%"${key##*[![:space:]]}"}"
    value="${value#"${value%%[![:space:]]*}"}"

    if [[ ! "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
      echo "Invalid env key '$key' on line $lineno in $file"
      return 1
    fi

    if [[ ${#value} -ge 2 ]]; then
      if [[ "${value:0:1}" == "\"" && "${value: -1}" == "\"" ]]; then
        value="${value:1:${#value}-2}"
      elif [[ "${value:0:1}" == "'" && "${value: -1}" == "'" ]]; then
        value="${value:1:${#value}-2}"
      fi
    fi

    export "$key=$value"
  done < "$file"
}

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: $ENV_FILE"
  echo "Create it from: $HERE/codexbot.env.example"
  exit 1
fi

load_env_file "$ENV_FILE"

if [[ -f "$LOCAL_ENV_FILE" ]]; then
  load_env_file "$LOCAL_ENV_FILE"
fi

: "${TELEGRAM_BOT_TOKEN:?Missing TELEGRAM_BOT_TOKEN in $ENV_FILE}"

if [[ "$CHECK_ONLY" == "1" ]]; then
  echo "Env OK: $ENV_FILE"
  exit 0
fi

export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"

PYTHON_BIN="${CODEXBOT_PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "$HERE/.venv/bin/python" ]]; then
    PYTHON_BIN="$HERE/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

exec "$PYTHON_BIN" "$HERE/bot.py"
