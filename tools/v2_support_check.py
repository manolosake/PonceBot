#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate v2 support artifacts: verify, runtime contract and preview paths.")
    ap.add_argument("--workspace", required=True, help="Workspace root path")
    ap.add_argument("--verify-log", required=True, help="Path to verify.log")
    ap.add_argument("--out", required=True, help="Path to write JSON report")
    args = ap.parse_args()

    ws = Path(args.workspace).expanduser().resolve()
    verify_log = Path(args.verify_log).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    errors: list[str] = []
    checks: dict[str, Any] = {}

    checks["workspace_exists"] = ws.exists() and ws.is_dir()
    if not checks["workspace_exists"]:
        errors.append("workspace missing")

    preview_html = ws / ".codexbot_preview" / "preview.html"
    index_html = ws / ".codexbot_preview" / "index.html"
    checks["preview_html_exists"] = preview_html.exists()
    checks["index_html_exists"] = index_html.exists()
    checks["preview_contract_established"] = bool(checks["preview_html_exists"] and checks["index_html_exists"])
    if not checks["preview_contract_established"]:
        errors.append("preview contract paths missing (.codexbot_preview/preview.html and index.html)")

    checks["verify_log_exists"] = verify_log.exists() and verify_log.is_file()
    if not checks["verify_log_exists"]:
        errors.append("verify log missing")
    else:
        txt = _read_text(verify_log)
        checks["verify_contains_command"] = "CMD: make verify" in txt
        checks["verify_exit_code_0"] = "EXIT_CODE=0" in txt
        if not checks["verify_contains_command"]:
            errors.append("verify.log missing command marker")
        if not checks["verify_exit_code_0"]:
            errors.append("verify.log missing EXIT_CODE=0")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "checks": checks,
        "paths": {
            "workspace": str(ws),
            "preview_html": str(preview_html),
            "index_html": str(index_html),
            "verify_log": str(verify_log),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
