#!/usr/bin/env bash
set -euo pipefail

UNIT_NAME="${1:-codexbot.service}"
BOT_HOME="${BOT_HOME:-$HOME/codexbot}"
ENV_FILE="${ENV_FILE:-$BOT_HOME/codexbot.env}"
STATE_FILE="${STATE_FILE:-$BOT_HOME/state.json}"

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

read_env_value() {
  local key="$1"
  local file="$2"
  [[ -f "$file" ]] || return 1

  local line lhs rhs
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ -z "${line//[[:space:]]/}" ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ "$line" == *=* ]] || continue

    lhs="${line%%=*}"
    rhs="${line#*=}"
    lhs="$(trim "$lhs")"
    rhs="${rhs#"${rhs%%[![:space:]]*}"}"

    [[ "$lhs" == "$key" ]] || continue

    if [[ ${#rhs} -ge 2 ]]; then
      if [[ "${rhs:0:1}" == "\"" && "${rhs: -1}" == "\"" ]]; then
        rhs="${rhs:1:${#rhs}-2}"
      elif [[ "${rhs:0:1}" == "'" && "${rhs: -1}" == "'" ]]; then
        rhs="${rhs:1:${#rhs}-2}"
      fi
    fi
    printf '%s' "$rhs"
    return 0
  done < "$file"
  return 1
}

BOT_TOKEN="$(read_env_value TELEGRAM_BOT_TOKEN "$ENV_FILE" || true)"
NOTIFY_CHAT_ID="$(read_env_value TELEGRAM_NOTIFY_CHAT_ID "$ENV_FILE" || true)"

if [[ -z "$NOTIFY_CHAT_ID" && -f "$STATE_FILE" ]]; then
  NOTIFY_CHAT_ID="$(python3 - "$STATE_FILE" <<'PY'
import json
import sys
from pathlib import Path

try:
    p = Path(sys.argv[1])
    data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    value = data.get("notify_chat_id")
    if value is not None:
        print(value)
except Exception:
    pass
PY
)"
fi

if [[ -z "$BOT_TOKEN" || -z "$NOTIFY_CHAT_ID" ]]; then
  exit 0
fi

if [[ ! "$NOTIFY_CHAT_ID" =~ ^-?[0-9]+$ ]]; then
  exit 0
fi

HOST="$(hostname -s 2>/dev/null || hostname 2>/dev/null || printf 'unknown-host')"
WHEN_UTC="$(date -u +'%Y-%m-%d %H:%M:%S UTC')"
ACTIVE_STATE="$(systemctl --user show "$UNIT_NAME" -p ActiveState --value 2>/dev/null || printf 'unknown')"
SUB_STATE="$(systemctl --user show "$UNIT_NAME" -p SubState --value 2>/dev/null || printf 'unknown')"
RESULT_STATE="$(systemctl --user show "$UNIT_NAME" -p Result --value 2>/dev/null || printf 'unknown')"
RESTARTS="$(systemctl --user show "$UNIT_NAME" -p NRestarts --value 2>/dev/null || printf 'unknown')"

LAST_LOGS="$(journalctl --user -u "$UNIT_NAME" --no-pager -n 8 2>/dev/null | tail -n 8)"
if [[ ${#LAST_LOGS} -gt 1200 ]]; then
  LAST_LOGS="${LAST_LOGS:0:1200}..."
fi

MSG=$(
  cat <<EOF
codexbot alert: service failure
host: $HOST
time: $WHEN_UTC
unit: $UNIT_NAME
active: $ACTIVE_STATE/$SUB_STATE
result: $RESULT_STATE
restarts: $RESTARTS

recent logs:
$LAST_LOGS
EOF
)

curl -fsS -X POST "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
  --data-urlencode "chat_id=${NOTIFY_CHAT_ID}" \
  --data-urlencode "text=${MSG}" \
  --data-urlencode "disable_notification=true" \
  >/dev/null || true

