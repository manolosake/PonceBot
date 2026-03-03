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
from urllib.request import urlopen


def _pick_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _fetch(url: str) -> tuple[int, str]:
    with urlopen(url, timeout=5) as response:
        status = int(getattr(response, "status", 200))
        body = response.read(4096).decode("utf-8", errors="replace")
        return status, body


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate preview contract over local endpoint.")
    ap.add_argument("--workspace", required=True, help="Workspace root")
    ap.add_argument("--out", required=True, help="Output JSON report path")
    args = ap.parse_args()

    ws = Path(args.workspace).expanduser().resolve()
    preview_root = ws / ".codexbot_preview"
    preview_html = preview_root / "preview.html"
    index_html = preview_root / "index.html"
    out_path = Path(args.out).expanduser().resolve()

    errors: list[str] = []
    checks: dict[str, Any] = {}

    checks["preview_html_exists"] = preview_html.exists()
    checks["index_html_exists"] = index_html.exists()
    if not checks["preview_html_exists"]:
        errors.append("missing preview.html")
    if not checks["index_html_exists"]:
        errors.append("missing index.html")

    port = _pick_port()
    handler = partial(SimpleHTTPRequestHandler, directory=str(preview_root))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    idx_status, idx_body = 0, ""
    pv_status, pv_body = 0, ""
    try:
        idx_status, idx_body = _fetch(f"http://127.0.0.1:{port}/")
        pv_status, pv_body = _fetch(f"http://127.0.0.1:{port}/preview.html")
    except Exception as exc:
        errors.append(f"endpoint probe failed: {exc}")
    finally:
        server.shutdown()
        server.server_close()

    checks["index_http_200"] = idx_status == 200
    checks["preview_http_200"] = pv_status == 200
    checks["index_references_preview"] = "preview.html" in idx_body.lower()
    checks["preview_not_missing"] = "preview missing" not in pv_body.lower()
    if not checks["index_http_200"]:
        errors.append("index endpoint not 200")
    if not checks["preview_http_200"]:
        errors.append("preview endpoint not 200")
    if not checks["index_references_preview"]:
        errors.append("index does not reference preview")
    if not checks["preview_not_missing"]:
        errors.append("preview contains missing banner")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "checks": checks,
        "paths": {
            "preview_html": str(preview_html),
            "index_html": str(index_html),
            "base_url": f"http://127.0.0.1:{port}/",
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
