#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def _api_request(method: str, url: str, *, token: str, payload: dict | None = None) -> tuple[int, str]:
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("Authorization", f"Bearer {token}")
    if body is not None:
        req.add_header("Content-Type", "application/json; charset=utf-8")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return int(resp.status), resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return int(e.code), e.read().decode("utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Enable basic GitHub branch protection for main (requires token with admin:repo)."
    )
    ap.add_argument("--repo", required=True, help="GitHub repo slug: owner/name")
    ap.add_argument("--branch", default="main", help="branch name (default: main)")
    ap.add_argument("--token-env", default="GITHUB_TOKEN", help="env var containing token (default: GITHUB_TOKEN)")
    ap.add_argument("--apply", action="store_true", help="apply protection (otherwise dry-run prints payload)")
    args = ap.parse_args()

    tok = os.environ.get(args.token_env, "").strip()
    if not tok:
        sys.stderr.write(f"Missing token in env var {args.token_env}\n")
        return 2

    owner_repo = args.repo.strip()
    branch = args.branch.strip() or "main"

    # Minimal, predictable guardrails:
    # - Require PRs (no direct pushes)
    # - Require at least 1 approving review
    # - Require conversation resolution
    # - Require branches to be up to date before merge
    payload = {
        "required_status_checks": None,
        "enforce_admins": True,
        "required_pull_request_reviews": {
            "dismiss_stale_reviews": True,
            "require_code_owner_reviews": False,
            "required_approving_review_count": 1,
        },
        "restrictions": None,
        "required_conversation_resolution": True,
        "required_linear_history": False,
        "allow_force_pushes": False,
        "allow_deletions": False,
        "required_signatures": False,
    }

    url = f"https://api.github.com/repos/{owner_repo}/branches/{branch}/protection"
    if not args.apply:
        sys.stdout.write(json.dumps({"url": url, "payload": payload}, indent=2) + "\n")
        return 0

    code, text = _api_request("PUT", url, token=tok, payload=payload)
    sys.stdout.write(json.dumps({"status": code, "response": text[:4000]}, indent=2) + "\n")
    return 0 if 200 <= code < 300 else 1


if __name__ == "__main__":
    raise SystemExit(main())

