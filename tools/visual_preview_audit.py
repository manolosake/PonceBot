#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import tarfile
import time
import zlib
from pathlib import Path
from typing import Any, Callable

# Ensure repo-root imports work when invoked as: python3 tools/visual_preview_audit.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orchestrator.screenshot import Viewport, capture_html_file


VIEWPORT_SPECS: list[tuple[str, Viewport]] = [
    ("desktop", Viewport(width=1366, height=768)),
    ("tablet", Viewport(width=1024, height=1366)),
    ("mobile", Viewport(width=412, height=915)),
]
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _png_dimensions(path: Path) -> tuple[int, int]:
    raw = path.read_bytes()
    if len(raw) < 24 or not raw.startswith(PNG_SIGNATURE):
        raise ValueError("invalid png header")
    w, h = struct.unpack(">II", raw[16:24])
    if w <= 0 or h <= 0:
        raise ValueError("invalid png dimensions")
    return int(w), int(h)


def _validate_png(path: Path) -> tuple[bool, str, int, int]:
    if not path.exists() or not path.is_file():
        return False, "missing file", 0, 0
    try:
        w, h = _png_dimensions(path)
    except Exception as e:
        return False, str(e), 0, 0
    if path.stat().st_size < 100:
        return False, "png too small", w, h
    return True, "", w, h


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", crc)


def write_synthetic_png(path: Path, *, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = b"\x00" + (b"\x74\xb9\xff\xff" * width)  # RGBA with no filter
    raw = row * height
    ihdr = struct.pack(">IIBBBBB", int(width), int(height), 8, 6, 0, 0, 0)
    out = PNG_SIGNATURE
    out += _png_chunk(b"IHDR", ihdr)
    out += _png_chunk(b"IDAT", zlib.compress(raw, level=9))
    out += _png_chunk(b"IEND", b"")
    path.write_bytes(out)


def _capture_with_mode(
    *,
    mode: str,
    html_path: Path,
    out_path: Path,
    viewport: Viewport,
    timeout_ms: int,
    allowed_hosts: set[str],
    allow_private: bool,
    block_network: bool,
) -> None:
    if mode == "synthetic":
        write_synthetic_png(out_path, width=int(viewport.width), height=int(viewport.height))
        return
    capture_html_file(
        html_path,
        out_path,
        viewport=viewport,
        timeout_ms=timeout_ms,
        allowed_hosts=allowed_hosts,
        allow_private=allow_private,
        block_network=block_network,
    )


def _capture_or_validate(
    *,
    label: str,
    viewport: Viewport,
    html_path: Path,
    out_path: Path,
    force_capture: bool,
    max_attempts: int,
    backoff_initial: float,
    backoff_max: float,
    capture_once: Callable[[], None],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "viewport": label,
        "target": {"width": int(viewport.width), "height": int(viewport.height)},
        "path": str(out_path),
        "source_preview_html": str(html_path),
        "status": "pending",
        "attempts": 0,
        "captured": False,
        "validated_existing": False,
        "backoff_seconds": [],
        "last_error": "",
        "sha256": "",
        "bytes": 0,
        "actual_dimensions": {"width": 0, "height": 0},
    }

    if not force_capture:
        ok, reason, w, h = _validate_png(out_path)
        if ok:
            record["status"] = "validated"
            record["validated_existing"] = True
            record["sha256"] = _sha256_file(out_path)
            record["bytes"] = int(out_path.stat().st_size)
            record["actual_dimensions"] = {"width": int(w), "height": int(h)}
            return record
        if reason:
            record["last_error"] = f"existing file invalid: {reason}"

    attempts = max(1, int(max_attempts))
    sleep_s = max(0.0, float(backoff_initial))
    for i in range(1, attempts + 1):
        record["attempts"] = i
        try:
            capture_once()
            ok, reason, w, h = _validate_png(out_path)
            if not ok:
                raise RuntimeError(f"captured file invalid: {reason}")
            record["status"] = "captured"
            record["captured"] = True
            record["sha256"] = _sha256_file(out_path)
            record["bytes"] = int(out_path.stat().st_size)
            record["actual_dimensions"] = {"width": int(w), "height": int(h)}
            record["last_error"] = ""
            return record
        except Exception as e:
            record["last_error"] = str(e)
            if i >= attempts:
                break
            wait = min(float(backoff_max), float(sleep_s))
            record["backoff_seconds"].append(wait)
            time.sleep(wait)
            sleep_s = max(wait * 2.0, float(backoff_initial))

    record["status"] = "failed"
    return record


def run_audit(
    *,
    preview_html: Path,
    artifacts_dir: Path,
    ticket_id: str,
    order_branch: str,
    max_attempts: int,
    backoff_initial: float,
    backoff_max: float,
    timeout_ms: int,
    allowed_hosts: set[str],
    allow_private: bool,
    block_network: bool,
    force_capture: bool,
    capture_mode: str,
) -> dict[str, Any]:
    preview_html = preview_html.expanduser().resolve()
    artifacts_dir = artifacts_dir.expanduser().resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if not preview_html.exists() or not preview_html.is_file():
        raise FileNotFoundError(f"preview html not found: {preview_html}")

    preview_copy = artifacts_dir / "preview.html"
    preview_copy.write_text(preview_html.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")

    captures: list[dict[str, Any]] = []
    for label, viewport in VIEWPORT_SPECS:
        out_png = artifacts_dir / f"preview-{label}.png"
        rec = _capture_or_validate(
            label=label,
            viewport=viewport,
            html_path=preview_copy,
            out_path=out_png,
            force_capture=force_capture,
            max_attempts=max_attempts,
            backoff_initial=backoff_initial,
            backoff_max=backoff_max,
            capture_once=lambda p=preview_copy, o=out_png, v=viewport: _capture_with_mode(
                mode=capture_mode,
                html_path=p,
                out_path=o,
                viewport=v,
                timeout_ms=timeout_ms,
                allowed_hosts=allowed_hosts,
                allow_private=allow_private,
                block_network=block_network,
            ),
        )
        captures.append(rec)

    all_ok = all(c.get("status") in ("validated", "captured") for c in captures)
    manifest = {
        "schema_version": 1,
        "tool": "visual_preview_audit",
        "generated_at_utc": _utc_now(),
        "status": "PASS" if all_ok else "FAIL",
        "ticket_id": str(ticket_id or ""),
        "order_branch": str(order_branch or ""),
        "preview_html": {
            "path": str(preview_copy),
            "sha256": _sha256_file(preview_copy),
            "bytes": int(preview_copy.stat().st_size),
        },
        "requested_viewports": [
            {"name": n, "width": int(v.width), "height": int(v.height)} for n, v in VIEWPORT_SPECS
        ],
        "retry_policy": {
            "max_attempts": int(max_attempts),
            "backoff_initial_seconds": float(backoff_initial),
            "backoff_max_seconds": float(backoff_max),
        },
        "capture_policy": {
            "capture_mode": capture_mode,
            "force_capture": bool(force_capture),
            "timeout_ms": int(timeout_ms),
            "block_network": bool(block_network),
            "allow_private": bool(allow_private),
            "allowed_hosts": sorted(allowed_hosts),
        },
        "captures": captures,
        "summary": {
            "validated_count": int(sum(1 for c in captures if c.get("validated_existing"))),
            "captured_count": int(sum(1 for c in captures if c.get("captured"))),
            "failed_count": int(sum(1 for c in captures if c.get("status") == "failed")),
        },
    }

    manifest_path = artifacts_dir / "visual_preview_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    bundle_path = artifacts_dir / "visual_preview_audit_bundle.tar.gz"
    with tarfile.open(bundle_path, "w:gz") as tf:
        for p in [preview_copy, manifest_path, *(artifacts_dir / f"preview-{n}.png" for n, _ in VIEWPORT_SPECS)]:
            if p.exists() and p.is_file():
                tf.add(str(p), arcname=f"visual_preview_audit/{p.name}")

    report = {
        "status": manifest["status"],
        "artifacts_dir": str(artifacts_dir),
        "manifest_path": str(manifest_path),
        "bundle_path": str(bundle_path),
        "captures": [
            {
                "viewport": c["viewport"],
                "status": c["status"],
                "path": c["path"],
                "attempts": c["attempts"],
                "last_error": c["last_error"],
            }
            for c in captures
        ],
    }
    report_path = artifacts_dir / "visual_preview_audit_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Validate/capture preview screenshots and package audit artifacts")
    ap.add_argument("--preview-html", required=True, help="Path to preview.html")
    ap.add_argument("--artifacts-dir", required=True, help="Output artifacts directory")
    ap.add_argument("--ticket-id", required=True)
    ap.add_argument("--order-branch", default="")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--backoff-initial-seconds", type=float, default=1.0)
    ap.add_argument("--backoff-max-seconds", type=float, default=8.0)
    ap.add_argument("--timeout-ms", type=int, default=30_000)
    ap.add_argument("--allowed-host", action="append", default=[])
    ap.add_argument("--allow-private", action="store_true")
    ap.add_argument("--allow-network", action="store_true", help="Allow http(s) loads while rendering")
    ap.add_argument("--force-capture", action="store_true", help="Capture even when existing PNGs are valid")
    ap.add_argument(
        "--capture-mode",
        choices=["playwright", "synthetic"],
        default="playwright",
        help="Capture backend; synthetic is for smoke/tests without Playwright",
    )
    return ap


def main() -> int:
    args = build_parser().parse_args()
    allowed_hosts = {str(h).strip().lower() for h in (args.allowed_host or []) if str(h).strip()}

    try:
        report = run_audit(
            preview_html=Path(args.preview_html),
            artifacts_dir=Path(args.artifacts_dir),
            ticket_id=str(args.ticket_id),
            order_branch=str(args.order_branch),
            max_attempts=int(args.max_attempts),
            backoff_initial=float(args.backoff_initial_seconds),
            backoff_max=float(args.backoff_max_seconds),
            timeout_ms=int(args.timeout_ms),
            allowed_hosts=allowed_hosts,
            allow_private=bool(args.allow_private),
            block_network=(not bool(args.allow_network)),
            force_capture=bool(args.force_capture),
            capture_mode=str(args.capture_mode),
        )
    except Exception as e:
        payload = {"status": "FAIL", "error": str(e)}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if str(report.get("status")) == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
