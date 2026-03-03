#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import urlopen


PLACEHOLDER_MARKERS = (
    "preview missing",
    "placeholder",
    "placeholder scene",
    "todo",
    "lorem ipsum",
)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _pick_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _http_get(url: str) -> tuple[int, bytes, str]:
    try:
        with urlopen(url, timeout=5) as response:
            status = int(getattr(response, "status", 200))
            body = response.read()
            return status, body, ""
    except HTTPError as exc:
        return int(exc.code), b"", str(exc)
    except Exception as exc:  # noqa: BLE001
        return 0, b"", str(exc)


def _load_manifest(preview_root: Path, manifest_path_arg: str) -> tuple[dict[str, Any], str]:
    if manifest_path_arg:
        manifest_path = Path(manifest_path_arg).expanduser().resolve()
    else:
        manifest_path = preview_root / "preview_manifest.json"
    if not manifest_path.exists():
        return {}, str(manifest_path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"_invalid_json": True}, str(manifest_path)
    return payload, str(manifest_path)


def run_gate(workspace: Path, expected_branch: str, expected_preview_sha: str, manifest_path_arg: str) -> dict[str, Any]:
    preview_root = workspace / ".codexbot_preview"
    preview_html = preview_root / "preview.html"
    index_html = preview_root / "index.html"
    manifest, manifest_path = _load_manifest(preview_root, manifest_path_arg)

    errors: list[str] = []
    checks: dict[str, Any] = {}

    checks["preview_exists"] = preview_html.exists()
    checks["index_exists"] = index_html.exists()
    if not checks["preview_exists"]:
        errors.append("missing preview.html")
    if not checks["index_exists"]:
        errors.append("missing index.html")

    preview_size = int(preview_html.stat().st_size) if preview_html.exists() else 0
    index_size = int(index_html.stat().st_size) if index_html.exists() else 0
    preview_sha = _sha256(preview_html) if preview_size > 0 else ""
    index_sha = _sha256(index_html) if index_size > 0 else ""
    checks["preview_size_gt_zero"] = preview_size > 0
    checks["index_size_gt_zero"] = index_size > 0
    if not checks["preview_size_gt_zero"]:
        errors.append("preview.html is empty")
    if not checks["index_size_gt_zero"]:
        errors.append("index.html is empty")

    manifest_expected_sha = str(manifest.get("preview_sha256", "")).strip()
    effective_expected_sha = expected_preview_sha.strip() or manifest_expected_sha
    checks["expected_preview_sha_available"] = bool(effective_expected_sha)
    if not checks["expected_preview_sha_available"]:
        errors.append("missing expected preview signature (arg --expected-preview-sha or manifest preview_sha256)")

    if checks["expected_preview_sha_available"]:
        checks["preview_sha_matches_expected"] = preview_sha == effective_expected_sha
        if not checks["preview_sha_matches_expected"]:
            errors.append("preview sha256 does not match expected signature")
    else:
        checks["preview_sha_matches_expected"] = False

    manifest_index_sha = str(manifest.get("index_sha256", "")).strip()
    checks["index_sha_matches_manifest"] = bool(manifest_index_sha) and (index_sha == manifest_index_sha)
    if manifest_index_sha and not checks["index_sha_matches_manifest"]:
        errors.append("index sha256 does not match manifest index_sha256")

    scene = manifest.get("scene_signature", {}) if isinstance(manifest, dict) else {}
    checks["manifest_exists"] = bool(manifest)
    checks["manifest_json_valid"] = not bool(manifest.get("_invalid_json"))
    if manifest and not checks["manifest_json_valid"]:
        errors.append("preview_manifest.json is invalid json")

    if expected_branch:
        checks["manifest_branch_matches_expected"] = str(manifest.get("order_branch", "")).strip() == expected_branch
        if manifest and not checks["manifest_branch_matches_expected"]:
            errors.append("manifest order_branch mismatch vs ORDER_BRANCH")
    else:
        checks["manifest_branch_matches_expected"] = True

    checks["scene_signature_present"] = isinstance(scene, dict) and bool(scene)
    checks["scene_not_placeholder"] = str(scene.get("wormhole_mode", "")).strip().lower() != "placeholder"
    checks["scene_hyperboloid_neck"] = bool(scene.get("hyperboloid_neck", False))
    checks["scene_glow_enabled"] = bool(scene.get("volumetric_glow_enabled", False))
    if manifest:
        if not checks["scene_signature_present"]:
            errors.append("scene_signature missing in manifest")
        if checks["scene_signature_present"] and not checks["scene_not_placeholder"]:
            errors.append("scene_signature indicates placeholder mode")
        if checks["scene_signature_present"] and not checks["scene_hyperboloid_neck"]:
            errors.append("scene_signature.hyperboloid_neck must be true")
        if checks["scene_signature_present"] and not checks["scene_glow_enabled"]:
            errors.append("scene_signature.volumetric_glow_enabled must be true")

    port = _pick_port()
    handler = partial(SimpleHTTPRequestHandler, directory=str(preview_root))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        root_status, root_body, root_err = _http_get(f"http://127.0.0.1:{port}/")
        prev_status, prev_body, prev_err = _http_get(f"http://127.0.0.1:{port}/preview.html")
    finally:
        server.shutdown()
        server.server_close()

    checks["http_root_200"] = root_status == 200
    checks["http_preview_200"] = prev_status == 200
    if not checks["http_root_200"]:
        errors.append(f"index endpoint returned {root_status}")
    if not checks["http_preview_200"]:
        errors.append(f"preview endpoint returned {prev_status}")

    served_preview_sha = hashlib.sha256(prev_body).hexdigest() if prev_body else ""
    checks["served_preview_sha_matches_file"] = bool(served_preview_sha) and (served_preview_sha == preview_sha)
    if not checks["served_preview_sha_matches_file"]:
        errors.append("served preview hash differs from preview.html hash")

    prev_text = prev_body.decode("utf-8", errors="replace").lower()
    marker_hits = [marker for marker in PLACEHOLDER_MARKERS if marker in prev_text]
    checks["placeholder_markers_absent"] = len(marker_hits) == 0
    if marker_hits:
        errors.append(f"placeholder markers found in served preview: {', '.join(marker_hits)}")

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "checks": checks,
        "expected": {
            "order_branch": expected_branch,
            "preview_sha256": effective_expected_sha,
        },
        "observed": {
            "preview_size_bytes": preview_size,
            "index_size_bytes": index_size,
            "preview_sha256": preview_sha,
            "index_sha256": index_sha,
            "served_preview_sha256": served_preview_sha,
            "placeholder_marker_hits": marker_hits,
        },
        "http": {
            "root": {"url": f"http://127.0.0.1:{port}/", "status_code": root_status, "error": root_err},
            "preview": {
                "url": f"http://127.0.0.1:{port}/preview.html",
                "status_code": prev_status,
                "error": prev_err,
            },
        },
        "paths": {
            "workspace": str(workspace),
            "preview_root": str(preview_root),
            "index_html": str(index_html),
            "preview_html": str(preview_html),
            "manifest_path": manifest_path,
        },
        "manifest_subset": {
            "build_id": manifest.get("build_id", ""),
            "order_branch": manifest.get("order_branch", ""),
            "head_sha": manifest.get("head_sha", ""),
            "preview_sha256": manifest.get("preview_sha256", ""),
            "index_sha256": manifest.get("index_sha256", ""),
            "scene_signature": scene if isinstance(scene, dict) else {},
        },
        "captured_at_utc": _utc_now(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview integrity gate (anti-placeholder + expected build signature).")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--expected-branch", default="")
    parser.add_argument("--expected-preview-sha", default="")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    result = run_gate(workspace, args.expected_branch.strip(), args.expected_preview_sha.strip(), args.manifest_path.strip())

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
