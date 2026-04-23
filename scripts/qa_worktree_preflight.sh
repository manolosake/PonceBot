#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  qa_worktree_preflight.sh --repo <repo_path> --branch <order_branch> --artifacts <artifacts_dir> [--worktree <path>] [--allow-dirty-main]

Behavior:
  - Validates origin/<branch> exists and resolves a commit SHA.
  - Detects dirty main repo state before checkout.
  - Fails fast on dirty state unless isolated temp worktree flow is used (default).
  - Executes c00_checkout against validated remote ref and records transcript + SHA evidence.
USAGE
}

REPO=""
BRANCH=""
ART=""
WORKTREE=""
ALLOW_DIRTY_MAIN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      REPO="$2"
      shift 2
      ;;
    --branch)
      BRANCH="$2"
      shift 2
      ;;
    --artifacts)
      ART="$2"
      shift 2
      ;;
    --worktree)
      WORKTREE="$2"
      shift 2
      ;;
    --allow-dirty-main)
      ALLOW_DIRTY_MAIN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$REPO" || -z "$BRANCH" || -z "$ART" ]]; then
  usage >&2
  exit 2
fi

mkdir -p "$ART"
TRANSCRIPT="$ART/command_transcript.log"
COMMAND_JSON="$ART/command_transcript.json"
RESULT_JSON="$ART/preflight_result.json"

: > "$TRANSCRIPT"
echo "[]" > "$COMMAND_JSON"

json_append() {
  local id="$1"
  local cwd="$2"
  local cmd="$3"
  local start="$4"
  local end="$5"
  local exit_code="$6"
  local out_file="$7"
  python3 - <<'PY' "$COMMAND_JSON" "$id" "$cwd" "$cmd" "$start" "$end" "$exit_code" "$out_file"
import json,sys
path,id_,cwd,cmd,start,end,exit_code,out_file = sys.argv[1:9]
rows = json.load(open(path, "r", encoding="utf-8"))
rows.append({
    "id": id_,
    "cwd": cwd,
    "cmd": cmd,
    "start": start,
    "end": end,
    "exit_code": int(exit_code),
    "output_file": out_file,
})
with open(path, "w", encoding="utf-8") as fh:
    json.dump(rows, fh, indent=2)
    fh.write("\n")
PY
}

run_cmd() {
  local id="$1"
  local cwd="$2"
  local cmd="$3"
  local out_file="$ART/${id}.out"
  local start
  local end
  local rc
  start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  {
    echo "===== ${id} ====="
    echo "start=${start}"
    echo "cwd=${cwd}"
    echo "cmd=${cmd}"
  } >> "$TRANSCRIPT"
  set +e
  bash -lc "cd '$cwd' && $cmd" > "$out_file" 2>&1
  rc=$?
  set -e
  end="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  {
    echo "exit_code=${rc}"
    echo "end=${end}"
    echo "output_file=${out_file}"
    echo
  } >> "$TRANSCRIPT"
  json_append "$id" "$cwd" "$cmd" "$start" "$end" "$rc" "$out_file"
  return "$rc"
}

if [[ -z "$WORKTREE" ]]; then
  WORKTREE="/tmp/qa_preflight_${BRANCH//\//_}"
fi

DIRTY_STATE="$(git -C "$REPO" status --porcelain || true)"
DIRTY_MAIN=false
if [[ -n "$DIRTY_STATE" ]]; then
  DIRTY_MAIN=true
fi

{
  echo "repo=$REPO"
  echo "branch=$BRANCH"
  echo "requested_worktree=$WORKTREE"
  echo "dirty_main=$DIRTY_MAIN"
} >> "$TRANSCRIPT"

if [[ "$DIRTY_MAIN" == "true" && "$ALLOW_DIRTY_MAIN" -ne 1 ]]; then
  echo "dirty_state_action=isolated_worktree_required" >> "$TRANSCRIPT"
fi

run_cmd "c00_fetch" "$REPO" "git fetch origin '$BRANCH'" || {
  echo "preflight_status=fetch_failed" >> "$TRANSCRIPT"
  exit 1
}

REMOTE_REF="origin/$BRANCH"
RESOLVED_SHA="$(git -C "$REPO" rev-parse --verify "${REMOTE_REF}^{commit}" 2>/dev/null || true)"
if [[ -z "$RESOLVED_SHA" ]]; then
  echo "preflight_status=missing_remote_ref" >> "$TRANSCRIPT"
  echo "missing_ref=$REMOTE_REF" >> "$TRANSCRIPT"
  exit 1
fi

echo "$RESOLVED_SHA" > "$ART/validated_sha.txt"
echo "$REMOTE_REF" > "$ART/validated_ref.txt"
echo "validated_ref=$REMOTE_REF" >> "$TRANSCRIPT"
echo "validated_sha=$RESOLVED_SHA" >> "$TRANSCRIPT"

if [[ "$DIRTY_MAIN" == "true" && "$ALLOW_DIRTY_MAIN" -ne 1 ]]; then
  # Fail fast for in-place checkout path by forcing isolated worktree mode only.
  echo "preflight_guard=dirty_main_detected_no_inplace_checkout" >> "$TRANSCRIPT"
fi

if [[ -d "$WORKTREE" ]]; then
  run_cmd "c00_worktree_remove" "$REPO" "git worktree remove --force '$WORKTREE'" || true
fi

if run_cmd "c00_checkout" "$REPO" "git worktree add --detach '$WORKTREE' '$REMOTE_REF'"; then
  CHECKOUT_SHA="$(git -C "$WORKTREE" rev-parse HEAD)"
  echo "$CHECKOUT_SHA" > "$ART/checkout_sha.txt"
  echo "checkout_sha=$CHECKOUT_SHA" >> "$TRANSCRIPT"
else
  CHECKOUT_SHA=""
fi

MATCHES_VALIDATED_SHA=false
if [[ -n "$CHECKOUT_SHA" && "$CHECKOUT_SHA" == "$RESOLVED_SHA" ]]; then
  MATCHES_VALIDATED_SHA=true
fi

python3 - <<'PY' "$RESULT_JSON" "$BRANCH" "$REMOTE_REF" "$RESOLVED_SHA" "$CHECKOUT_SHA" "$MATCHES_VALIDATED_SHA" "$DIRTY_MAIN"
import json,sys
path,branch,remote_ref,resolved_sha,checkout_sha,matches,dirty = sys.argv[1:8]
payload = {
    "branch": branch,
    "validated_ref": remote_ref,
    "validated_sha": resolved_sha,
    "checkout_sha": checkout_sha,
    "checkout_matches_validated_sha": matches.lower() == "true",
    "dirty_main_detected": dirty.lower() == "true",
    "c00_checkout": 0 if (checkout_sha and checkout_sha == resolved_sha) else 1,
}
with open(path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2)
    fh.write("\n")
PY

if [[ "$MATCHES_VALIDATED_SHA" != "true" ]]; then
  echo "preflight_status=checkout_sha_mismatch" >> "$TRANSCRIPT"
  exit 1
fi

echo "preflight_status=ok" >> "$TRANSCRIPT"
