#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Check:
    key: str
    ok: bool
    details: str


def _utc_iso(ts: float | None = None) -> str:
    t = float(time.time() if ts is None else ts)
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _gather_visuals(artifacts_dir: Path) -> tuple[dict[str, list[str]], list[Path]]:
    by_kind: dict[str, list[str]] = {"desktop": [], "tablet": [], "mobile": []}
    all_images: list[Path] = []
    for p in artifacts_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            continue
        all_images.append(p)
        name = p.name.lower()
        for kind in by_kind:
            if kind in name:
                by_kind[kind].append(str(p.relative_to(artifacts_dir)))
    return by_kind, all_images


def _find_preview_html(artifacts_dir: Path, workspace_dir: Path) -> Path | None:
    candidates = [
        artifacts_dir / "preview.html",
        artifacts_dir / ".codexbot_preview" / "preview.html",
        workspace_dir / ".codexbot_preview" / "preview.html",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


_DIFF_PATH_RE = re.compile(r"^diff --git a/(.+?) b/(.+)$")


def _extract_patch_paths(patch_text: str) -> set[str]:
    out: set[str] = set()
    for raw in patch_text.splitlines():
        line = raw.strip()
        m = _DIFF_PATH_RE.match(line)
        if m:
            out.add(m.group(1))
            out.add(m.group(2))
            continue
        if line.startswith("+++ b/"):
            out.add(line[6:])
            continue
        if line.startswith("--- a/"):
            out.add(line[6:])
            continue
    return {p for p in out if p and p != "/dev/null"}


def _extract_status_paths(status_text: str) -> set[str]:
    out: set[str] = set()
    for raw in status_text.splitlines():
        if not raw.strip():
            continue
        if len(raw) < 4:
            continue
        path_part = raw[3:]
        if " -> " in path_part:
            path_part = path_part.split(" -> ", 1)[1]
        path_part = path_part.strip().strip('"')
        if path_part:
            out.add(path_part)
    return out


def _looks_like_report_file(path: Path) -> bool:
    name = path.name.lower()
    if not name.endswith(".json"):
        return False
    keys = ("report", "summary", "manifest", "post_publish", "trace")
    return any(k in name for k in keys)


def _collect_report_files(artifacts_dir: Path) -> list[Path]:
    return sorted([p for p in artifacts_dir.rglob("*.json") if _looks_like_report_file(p)])


def _collect_telegram_evidence(artifacts_dir: Path) -> list[Path]:
    out: list[Path] = []
    for p in artifacts_dir.rglob("*"):
        if not p.is_file():
            continue
        if "telegram" in p.name.lower():
            out.append(p)
    return sorted(out)


def _check_non_empty(path: Path, key: str) -> Check:
    if not path.exists():
        return Check(key, False, f"missing: {path.name}")
    sz = path.stat().st_size
    return Check(key, sz > 0, f"{path.name} bytes={sz}")


def _normalize_path_like(value: str) -> str:
    v = (value or "").strip().replace("\\", "/")
    while v.startswith("./"):
        v = v[2:]
    return v


def _iter_key_values(obj: object, *, target_key: str) -> list[object]:
    out: list[object] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k) == target_key:
                out.append(v)
            out.extend(_iter_key_values(v, target_key=target_key))
    elif isinstance(obj, list):
        for it in obj:
            out.extend(_iter_key_values(it, target_key=target_key))
    return out


def _extract_files_in_patch_claims(report_json: object) -> list[list[str]]:
    claims: list[list[str]] = []
    for value in _iter_key_values(report_json, target_key="files_in_patch"):
        if isinstance(value, list):
            cleaned = [_normalize_path_like(str(x)) for x in value if str(x).strip()]
            claims.append(cleaned)
    return claims


def _extract_visual_file_claims(report_json: object) -> list[str]:
    claims: list[str] = []
    visual_keys = ("screenshot", "visual", "preview", "evidence")
    valid_exts = (".png", ".jpg", ".jpeg", ".webp", ".html")

    def walk(node: object, parent_key: str = "") -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, parent_key=str(k))
            return
        if isinstance(node, list):
            for v in node:
                walk(v, parent_key=parent_key)
            return
        if isinstance(node, str):
            key = parent_key.lower()
            if not any(token in key for token in visual_keys):
                return
            s = _normalize_path_like(node)
            if "://" in s:
                return
            if not s.lower().endswith(valid_exts):
                return
            claims.append(s)

    walk(report_json)
    # de-dup preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for c in claims:
        if c in seen:
            continue
        seen.add(c)
        deduped.append(c)
    return deduped


def _validate(
    *,
    artifacts_dir: Path,
    workspace_dir: Path,
    require_telegram: bool,
) -> tuple[bool, list[Check], dict[str, object]]:
    checks: list[Check] = []

    patch_p = artifacts_dir / "changes.patch"
    status_p = artifacts_dir / "git_status.txt"
    checks.append(_check_non_empty(patch_p, "required_changes_patch_non_empty"))
    checks.append(_check_non_empty(status_p, "required_git_status_non_empty"))

    patch_text = _read_text(patch_p) if patch_p.exists() else ""
    status_text = _read_text(status_p) if status_p.exists() else ""
    patch_paths = _extract_patch_paths(patch_text)
    status_paths = _extract_status_paths(status_text)
    missing_in_patch = sorted(p for p in status_paths if p not in patch_paths)
    orphan_in_patch = sorted(p for p in patch_paths if p not in status_paths)

    patch_vs_status_ok = not missing_in_patch
    checks.append(
        Check(
            "patch_vs_status_consistency",
            patch_vs_status_ok,
            (
                "ok"
                if patch_vs_status_ok
                else "missing_in_patch="
                + ", ".join(missing_in_patch[:25])
            ),
        )
    )

    report_files = _collect_report_files(artifacts_dir)
    checks.append(
        Check(
            "required_reports_present",
            len(report_files) > 0,
            f"report_files={len(report_files)}",
        )
    )

    report_patch_mismatches: list[dict[str, object]] = []
    report_visual_mismatches: list[dict[str, str]] = []
    all_rel_files = {str(p.relative_to(artifacts_dir)) for p in artifacts_dir.rglob("*") if p.is_file()}
    all_basenames = {Path(p).name for p in all_rel_files}
    for rp in report_files:
        try:
            parsed = json.loads(_read_text(rp))
        except Exception as e:
            report_patch_mismatches.append(
                {
                    "report": str(rp.relative_to(artifacts_dir)),
                    "error": f"invalid_json: {e}",
                }
            )
            continue

        for claim in _extract_files_in_patch_claims(parsed):
            claimed = {_normalize_path_like(x) for x in claim if x}
            missing = sorted(p for p in claimed if p not in patch_paths)
            unexpected = sorted(p for p in patch_paths if p not in claimed)
            if missing or unexpected:
                report_patch_mismatches.append(
                    {
                        "report": str(rp.relative_to(artifacts_dir)),
                        "missing_in_patch": missing[:20],
                        "extra_in_patch": unexpected[:20],
                    }
                )

        for visual_claim in _extract_visual_file_claims(parsed):
            normalized = _normalize_path_like(visual_claim)
            exists_rel = normalized in all_rel_files
            exists_basename = Path(normalized).name in all_basenames
            if not (exists_rel or exists_basename):
                report_visual_mismatches.append(
                    {
                        "report": str(rp.relative_to(artifacts_dir)),
                        "declared_visual_file": normalized,
                    }
                )

    checks.append(
        Check(
            "integrity_report_patch_alignment",
            len(report_patch_mismatches) == 0,
            (
                "ok"
                if len(report_patch_mismatches) == 0
                else "mismatch_in_report="
                + ", ".join(str(m.get("report")) for m in report_patch_mismatches[:5])
            ),
        )
    )
    checks.append(
        Check(
            "integrity_report_visual_alignment",
            len(report_visual_mismatches) == 0,
            (
                "ok"
                if len(report_visual_mismatches) == 0
                else "missing_visual_declared_in_report="
                + ", ".join(str(m.get("report")) for m in report_visual_mismatches[:5])
            ),
        )
    )

    preview_path = _find_preview_html(artifacts_dir, workspace_dir)
    if preview_path is None:
        checks.append(Check("preview_html_valid", False, "preview.html not found"))
    else:
        preview_text = _read_text(preview_path).strip()
        looks_html = ("<html" in preview_text.lower()) or ("<!doctype html" in preview_text.lower())
        checks.append(
            Check(
                "preview_html_valid",
                bool(preview_text) and looks_html,
                f"path={preview_path} bytes={len(preview_text.encode('utf-8'))}",
            )
        )

    visual_by_kind, all_images = _gather_visuals(artifacts_dir)
    for kind in ("desktop", "tablet", "mobile"):
        files = visual_by_kind.get(kind, [])
        checks.append(
            Check(
                f"required_{kind}_screenshot",
                len(files) > 0,
                f"matches={len(files)}",
            )
        )

    telegram_files = _collect_telegram_evidence(artifacts_dir)
    if require_telegram:
        checks.append(
            Check(
                "telegram_traceability_present",
                len(telegram_files) > 0,
                f"telegram_files={len(telegram_files)}",
            )
        )

    ok = all(c.ok for c in checks)
    details: dict[str, object] = {
        "artifacts_dir": str(artifacts_dir),
        "workspace_dir": str(workspace_dir),
        "status_paths_count": len(status_paths),
        "patch_paths_count": len(patch_paths),
        "missing_in_patch": missing_in_patch,
        "orphan_in_patch": orphan_in_patch[:50],
        "report_files": [str(p.relative_to(artifacts_dir)) for p in report_files],
        "report_patch_mismatches": report_patch_mismatches[:20],
        "report_visual_mismatches": report_visual_mismatches[:20],
        "telegram_files": [str(p.relative_to(artifacts_dir)) for p in telegram_files],
        "visual_images_total": len(all_images),
        "visual_matches": visual_by_kind,
    }
    return ok, checks, details


def _write_text(path: Path, lines: Iterable[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Delivery evidence guard: validates mandatory artifacts and consistency."
    )
    ap.add_argument("--artifacts-dir", required=True, help="artifact directory to validate")
    ap.add_argument("--workspace-dir", default=".", help="workspace root (for .codexbot_preview fallback)")
    ap.add_argument("--report-name", default="sre_evidence_gate_report.json", help="json report filename")
    ap.add_argument("--log-name", default="sre_evidence_gate.log", help="log filename")
    ap.add_argument("--no-require-telegram", action="store_true", help="disable telegram evidence requirement")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve()
    workspace_dir = Path(args.workspace_dir).expanduser().resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    ok, checks, details = _validate(
        artifacts_dir=artifacts_dir,
        workspace_dir=workspace_dir,
        require_telegram=not args.no_require_telegram,
    )
    status = "PASS" if ok else "FAIL"
    failed_keys = [c.key for c in checks if not c.ok]
    reason = "all checks passed" if ok else ("failed checks: " + ", ".join(failed_keys))

    payload = {
        "schema_version": 1,
        "generated_at": _utc_iso(),
        "status": status,
        "reason": reason,
        "checks": [{"key": c.key, "ok": c.ok, "details": c.details} for c in checks],
        "details": details,
    }

    report_path = artifacts_dir / args.report_name
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log_lines = [
        f"timestamp={payload['generated_at']}",
        f"status={status}",
        f"reason={reason}",
    ]
    for c in checks:
        log_lines.append(f"[{'PASS' if c.ok else 'FAIL'}] {c.key}: {c.details}")
    _write_text(artifacts_dir / args.log_name, log_lines)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
