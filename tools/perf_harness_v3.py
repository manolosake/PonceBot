#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


THRESHOLDS = {
    "desktop": {"fps_min": 45.0, "p95_max_ms": 28.0, "viewport": {"width": 1600, "height": 900}},
    "tablet": {"fps_min": 35.0, "p95_max_ms": 33.0, "viewport": {"width": 1024, "height": 768}},
    "mobile": {"fps_min": 30.0, "p95_max_ms": 42.0, "viewport": {"width": 390, "height": 844}},
}


def _pick_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    if len(arr) == 1:
        return float(arr[0])
    idx = (len(arr) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(arr) - 1)
    frac = idx - lo
    return float(arr[lo] * (1.0 - frac) + arr[hi] * frac)


def _collect_frame_times(url: str, *, width: int, height: int, duration_seconds: int) -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": width, "height": height})
        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(600)
        deltas = page.evaluate(
            """
            async (durationMs) => await new Promise((resolve) => {
              const arr = [];
              const t0 = performance.now();
              let last = t0;
              function tick(now) {
                arr.push(now - last);
                last = now;
                if (now - t0 >= durationMs) return resolve(arr);
                requestAnimationFrame(tick);
              }
              requestAnimationFrame(tick);
            })
            """,
            int(duration_seconds * 1000),
        )
        context.close()
        browser.close()

    samples = [float(v) for v in deltas if isinstance(v, (int, float)) and v > 0]
    avg_frame = (sum(samples) / len(samples)) if samples else 0.0
    avg_fps = (1000.0 / avg_frame) if avg_frame > 0 else 0.0
    p95 = _quantile(samples, 0.95)
    return {
        "sample_count": len(samples),
        "avg_fps": round(avg_fps, 3),
        "avg_frame_time_ms": round(avg_frame, 4),
        "p95_frame_time_ms": round(p95, 4),
    }


def _serve_preview(preview_root: Path):
    port = _pick_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(SimpleHTTPRequestHandler, directory=str(preview_root)))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port


def _run_workspace(workspace: Path, *, duration_seconds: int, tag: str) -> dict[str, Any]:
    preview_root = workspace / ".codexbot_preview"
    preview_html = preview_root / "preview.html"
    if not preview_html.exists():
        return {
            "status": "FAIL",
            "tag": tag,
            "errors": [f"missing preview file: {preview_html}"],
            "viewports": {},
            "workspace": str(workspace),
        }

    server, port = _serve_preview(preview_root)
    try:
        url = f"http://127.0.0.1:{port}/preview.html"
        per_viewport: dict[str, Any] = {}
        errors: list[str] = []
        for name, cfg in THRESHOLDS.items():
            m = _collect_frame_times(
                url,
                width=int(cfg["viewport"]["width"]),
                height=int(cfg["viewport"]["height"]),
                duration_seconds=duration_seconds,
            )
            gate = (m["avg_fps"] >= float(cfg["fps_min"])) and (m["p95_frame_time_ms"] <= float(cfg["p95_max_ms"]))
            per_viewport[name] = {
                "metrics": m,
                "thresholds": {"fps_min": cfg["fps_min"], "p95_max_ms": cfg["p95_max_ms"]},
                "pass": gate,
            }
            if not gate:
                errors.append(f"{name} failed threshold gate")
    finally:
        server.shutdown()
        server.server_close()

    return {
        "status": "PASS" if not errors else "FAIL",
        "tag": tag,
        "errors": errors,
        "workspace": str(workspace),
        "viewports": per_viewport,
        "duration_seconds": duration_seconds,
        "collected_at": time.time(),
    }


def _build_comparison(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in THRESHOLDS:
        b = baseline.get("viewports", {}).get(name, {}).get("metrics", {})
        c = candidate.get("viewports", {}).get(name, {}).get("metrics", {})
        out[name] = {
            "baseline": b,
            "candidate": c,
            "delta_avg_fps": round(float(c.get("avg_fps", 0.0)) - float(b.get("avg_fps", 0.0)), 3),
            "delta_p95_ms": round(float(c.get("p95_frame_time_ms", 0.0)) - float(b.get("p95_frame_time_ms", 0.0)), 4),
            "candidate_pass": bool(candidate.get("viewports", {}).get(name, {}).get("pass", False)),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Reproducible V3 perf harness: baseline vs candidate by viewport.")
    ap.add_argument("--baseline-workspace", required=True)
    ap.add_argument("--candidate-workspace", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--duration-seconds", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = _run_workspace(Path(args.baseline_workspace).expanduser().resolve(), duration_seconds=max(5, args.duration_seconds), tag="baseline")
    candidate = _run_workspace(Path(args.candidate_workspace).expanduser().resolve(), duration_seconds=max(5, args.duration_seconds), tag="candidate")
    comparison = _build_comparison(baseline, candidate)

    overall_pass = baseline.get("status") == "PASS" and candidate.get("status") == "PASS" and all(
        v.get("candidate_pass", False) for v in comparison.values()
    )
    report = {
        "status": "PASS" if overall_pass else "FAIL",
        "baseline_status": baseline.get("status"),
        "candidate_status": candidate.get("status"),
        "comparison": comparison,
        "thresholds": {k: {"fps_min": v["fps_min"], "p95_max_ms": v["p95_max_ms"]} for k, v in THRESHOLDS.items()},
    }

    (out_dir / "baseline_metrics.json").write_text(json.dumps(baseline, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "candidate_metrics.json").write_text(json.dumps(candidate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "baseline_vs_candidate_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
