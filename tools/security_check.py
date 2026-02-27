#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys


TRUE_SET = {"1", "true", "yes", "on"}


def _is_true(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in TRUE_SET


def _split_csv(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    return [x.strip() for x in raw.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Security posture checks for codexbot runtime env")
    ap.add_argument("--strict", action="store_true", help="Return non-zero on findings")
    args = ap.parse_args()

    findings: list[str] = []
    warnings: list[str] = []

    dangerous = _is_true("CODEX_DANGEROUS_BYPASS_SANDBOX")
    breakglass_reason = os.environ.get("BOT_BREAKGLASS_REASON", "").strip()
    if dangerous and not breakglass_reason:
        findings.append("CODEX_DANGEROUS_BYPASS_SANDBOX=1 requires BOT_BREAKGLASS_REASON")

    status_http_enabled = _is_true("BOT_STATUS_HTTP_ENABLED")
    status_http_token = os.environ.get("BOT_STATUS_HTTP_TOKEN", "").strip()
    if status_http_enabled and not status_http_token:
        findings.append("BOT_STATUS_HTTP_ENABLED=1 requires BOT_STATUS_HTTP_TOKEN")

    origins = _split_csv("BOT_STATUS_HTTP_ALLOWED_ORIGINS")
    if "*" in origins:
        findings.append("BOT_STATUS_HTTP_ALLOWED_ORIGINS must not contain '*'")
    if status_http_enabled and not origins:
        warnings.append("BOT_STATUS_HTTP_ALLOWED_ORIGINS is empty; browser cross-origin access will be blocked")

    if findings:
        print("[security-check] FAIL")
        for f in findings:
            print(f"- {f}")
    else:
        print("[security-check] PASS")

    for w in warnings:
        print(f"[security-check] WARN: {w}")

    if args.strict and findings:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
