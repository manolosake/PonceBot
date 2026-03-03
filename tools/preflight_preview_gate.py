#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import urlopen


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _http_status(url: str) -> tuple[int, str]:
    try:
        with urlopen(url, timeout=5) as response:
            status = int(getattr(response, "status", 200))
            body = response.read(2048).decode("utf-8", errors="replace")
            return status, body
    except HTTPError as exc:
        return int(exc.code), str(exc)


def run_preflight(workspace: Path) -> dict[str, Any]:
    preview_root = workspace / ".codexbot_preview"
    preview_html = preview_root / "preview.html"
    index_html = preview_root / "index.html"
    errors: list[str] = []
    checks: dict[str, Any] = {}

    checks["preview_html_exists"] = preview_html.exists()
    checks["index_html_exists"] = index_html.exists()
    if not checks["preview_html_exists"]:
        errors.append("missing .codexbot_preview/preview.html")
    if not checks["index_html_exists"]:
        errors.append("missing .codexbot_preview/index.html")

    port = _free_port()
    handler = partial(SimpleHTTPRequestHandler, directory=str(preview_root))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        status_idx, body_idx = _http_status(f"http://127.0.0.1:{port}/")
        status_preview, body_preview = _http_status(f"http://127.0.0.1:{port}/preview.html")
    finally:
        server.shutdown()
        server.server_close()

    checks["http_index_status"] = status_idx
    checks["http_preview_status"] = status_preview
    checks["http_index_200"] = status_idx == 200
    checks["http_preview_200"] = status_preview == 200
    checks["http_index_has_preview_ref"] = "preview.html" in body_idx.lower()
    checks["http_preview_not_missing_banner"] = "preview missing" not in body_preview.lower()

    if not checks["http_index_200"]:
        errors.append(f"index endpoint returned {status_idx}, expected 200")
    if not checks["http_preview_200"]:
        errors.append(f"preview endpoint returned {status_preview}, expected 200")
    if not checks["http_index_has_preview_ref"]:
        errors.append("index response does not reference preview.html")
    if not checks["http_preview_not_missing_banner"]:
        errors.append("preview body contains 'preview missing'")

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "checks": checks,
        "paths": {
            "workspace": str(workspace),
            "preview_root": str(preview_root),
            "preview_html": str(preview_html),
            "index_html": str(index_html),
            "base_url": f"http://127.0.0.1:{port}/",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Hard preflight gate for QA preview contract.")
    ap.add_argument("--workspace", required=True, help="QA workspace root")
    ap.add_argument("--out", default="", help="Optional output JSON path")
    args = ap.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    result = run_preflight(workspace)
    payload = json.dumps(result, indent=2, ensure_ascii=False)

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")

    print(payload)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
