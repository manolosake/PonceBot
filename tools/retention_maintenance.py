#!/usr/bin/env python3
"""
Retention + lightweight backup for codexbot logs and artifacts.

Design goals:
- Keep artifacts directory fast for dashboard usage (prune old job dirs).
- Preserve build deliverables (APKs, INSTALL.md, BUILD_INFO) by archiving before pruning.
- Emit an index file so dashboards can query history without scanning the whole tree.
- Vacuum user journald to cap log growth.

This is intended to run as a user-level systemd timer (safe; no restarts of codexbot.service).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RetentionConfig:
    artifacts_dir: Path
    archive_dir: Path
    backups_dir: Path
    index_json: Path
    index_jsonl: Path
    keep_days_artifacts: int
    keep_days_archives: int
    keep_days_backups: int
    keep_latest_apks: int
    journald_vacuum_time: str
    journald_vacuum_size: str
    max_index_entries: int


def _env_int(name: str, default: int) -> int:
    try:
        v = os.environ.get(name, "").strip()
        return default if not v else int(v)
    except Exception:
        return default


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name, "")
    v = v.strip()
    return v or default


def _now() -> float:
    return time.time()


def _dir_size_bytes(root: Path) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            p = Path(dirpath) / fn
            try:
                total += p.stat().st_size
            except FileNotFoundError:
                continue
    return total


def _sha256(path: Path, *, limit_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    read = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            if limit_bytes is not None and read + len(chunk) > limit_bytes:
                chunk = chunk[: max(0, limit_bytes - read)]
            h.update(chunk)
            read += len(chunk)
            if limit_bytes is not None and read >= limit_bytes:
                break
    return h.hexdigest()


def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _atomic_write(path: Path, data: str) -> None:
    _mkdir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


def _is_job_dir(p: Path) -> bool:
    # artifacts are UUID-named directories
    name = p.name
    if len(name) != 36:
        return False
    # cheap UUID-ish check
    return name.count("-") == 4


def load_config() -> RetentionConfig:
    base = Path("/home/aponce/codexbot/data")
    artifacts_dir = Path(_env_str("CODEXBOT_ARTIFACTS_DIR", str(base / "artifacts")))
    archive_dir = Path(_env_str("CODEXBOT_ARCHIVE_DIR", str(base / "archives")))
    backups_dir = Path(_env_str("CODEXBOT_BACKUPS_DIR", str(base / "backups")))
    return RetentionConfig(
        artifacts_dir=artifacts_dir,
        archive_dir=archive_dir,
        backups_dir=backups_dir,
        index_json=Path(_env_str("CODEXBOT_ARTIFACTS_INDEX_JSON", str(base / "artifacts_index.json"))),
        index_jsonl=Path(_env_str("CODEXBOT_ARTIFACTS_INDEX_JSONL", str(base / "artifacts_index.jsonl"))),
        keep_days_artifacts=_env_int("CODEXBOT_KEEP_DAYS_ARTIFACTS", 30),
        keep_days_archives=_env_int("CODEXBOT_KEEP_DAYS_ARCHIVES", 180),
        keep_days_backups=_env_int("CODEXBOT_KEEP_DAYS_BACKUPS", 30),
        keep_latest_apks=_env_int("CODEXBOT_KEEP_LATEST_APKS", 20),
        journald_vacuum_time=_env_str("CODEXBOT_JOURNALD_VACUUM_TIME", "14d"),
        journald_vacuum_size=_env_str("CODEXBOT_JOURNALD_VACUUM_SIZE", "200M"),
        max_index_entries=_env_int("CODEXBOT_MAX_INDEX_ENTRIES", 5000),
    )


def build_index(cfg: RetentionConfig) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not cfg.artifacts_dir.exists():
        return entries

    for p in sorted(cfg.artifacts_dir.iterdir(), key=lambda x: x.stat().st_mtime if x.exists() else 0, reverse=True):
        if len(entries) >= cfg.max_index_entries:
            break
        if not p.is_dir() or not _is_job_dir(p):
            continue

        try:
            st = p.stat()
        except FileNotFoundError:
            continue
        files: dict[str, dict[str, Any]] = {}
        for child in p.iterdir():
            if not child.is_file():
                continue
            name = child.name
            if name.endswith(".apk") or name in ("INSTALL.md", "BUILD_INFO.txt") or name.startswith("BUILD_INFO"):
                try:
                    cst = child.stat()
                except FileNotFoundError:
                    continue
                files[name] = {"size": cst.st_size, "mtime": cst.st_mtime}

        size_bytes = _dir_size_bytes(p)
        entries.append(
            {
                "job_id": p.name,
                "mtime": st.st_mtime,
                "size_bytes": size_bytes,
                "files": files,
                "kind": ("build" if any(k.endswith(".apk") for k in files) else "artifact"),
            }
        )
    return entries


def _archive_job_dir(cfg: RetentionConfig, job_dir: Path, *, ts: float) -> None:
    # Archive only important build outputs to keep storage bounded.
    # Keep under archive_dir/jobs/<job_id>/...
    dst = cfg.archive_dir / "jobs" / job_dir.name
    _mkdir(dst)

    keep_names = {"INSTALL.md", "BUILD_INFO.txt"}
    for child in job_dir.iterdir():
        if not child.is_file():
            continue
        name = child.name
        if name.endswith(".apk") or name.endswith(".aab") or name in keep_names or name.startswith("BUILD_INFO") or name.endswith(".zip"):
            try:
                shutil.copy2(child, dst / name)
            except FileNotFoundError:
                continue

    # marker for archive time
    (dst / "ARCHIVED_AT.txt").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)) + "\n", encoding="utf-8")


def prune_artifacts(cfg: RetentionConfig) -> dict[str, int]:
    stats = {"archived": 0, "deleted": 0, "skipped": 0}
    cutoff = _now() - cfg.keep_days_artifacts * 86400
    if not cfg.artifacts_dir.exists():
        return stats

    for p in cfg.artifacts_dir.iterdir():
        if not p.is_dir() or not _is_job_dir(p):
            continue
        try:
            m = p.stat().st_mtime
        except FileNotFoundError:
            continue
        if m >= cutoff:
            stats["skipped"] += 1
            continue
        try:
            _archive_job_dir(cfg, p, ts=_now())
            stats["archived"] += 1
        except Exception:
            # Archive best-effort; do not delete if archive failed.
            continue
        try:
            shutil.rmtree(p)
            stats["deleted"] += 1
        except Exception:
            continue
    return stats


def prune_archives(cfg: RetentionConfig) -> dict[str, int]:
    stats = {"deleted_jobs": 0, "kept_apks": 0}
    root = cfg.archive_dir / "jobs"
    if not root.exists():
        return stats

    # Keep latest N apks even if older, as a safety net.
    apk_paths: list[Path] = []
    for job_dir in root.iterdir():
        if not job_dir.is_dir():
            continue
        for child in job_dir.iterdir():
            if child.is_file() and child.name.endswith(".apk"):
                apk_paths.append(child)
    apk_paths.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    keep_apk_set = set(p.parent for p in apk_paths[: cfg.keep_latest_apks])
    stats["kept_apks"] = len(keep_apk_set)

    cutoff = _now() - cfg.keep_days_archives * 86400
    for job_dir in root.iterdir():
        if not job_dir.is_dir():
            continue
        try:
            m = job_dir.stat().st_mtime
        except FileNotFoundError:
            continue
        if job_dir in keep_apk_set:
            continue
        if m < cutoff:
            try:
                shutil.rmtree(job_dir)
                stats["deleted_jobs"] += 1
            except Exception:
                continue
    return stats


def run_journald_vacuum(cfg: RetentionConfig) -> dict[str, Any]:
    # Only user journal (safe without sudo).
    out: dict[str, Any] = {"vacuumed": False}
    try:
        subprocess.run(["journalctl", "--user", f"--vacuum-time={cfg.journald_vacuum_time}"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["journalctl", "--user", f"--vacuum-size={cfg.journald_vacuum_size}"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out["vacuumed"] = True
    except Exception as e:
        out["error"] = repr(e)
    return out


def write_index(cfg: RetentionConfig, entries: list[dict[str, Any]]) -> None:
    # jsonl for append-friendly ingestion; json for easy UI consumption.
    _atomic_write(cfg.index_json, json.dumps({"generated_at": _now(), "entries": entries}, ensure_ascii=True) + "\n")
    lines = []
    for e in entries:
        lines.append(json.dumps(e, ensure_ascii=True))
    _atomic_write(cfg.index_jsonl, "\n".join(lines) + ("\n" if lines else ""))


def write_backup(cfg: RetentionConfig) -> dict[str, Any]:
    # Create a daily tar.gz of the archive + index files.
    ts = time.strftime("%Y%m%d", time.gmtime(_now()))
    out_dir = cfg.backups_dir / "daily"
    _mkdir(out_dir)
    tar_path = out_dir / f"codexbot_archives_{ts}.tar.gz"
    if tar_path.exists():
        return {"created": False, "path": str(tar_path)}

    try:
        with tarfile.open(tar_path, "w:gz") as tf:
            if (cfg.archive_dir).exists():
                tf.add(cfg.archive_dir, arcname="archives")
            if cfg.index_json.exists():
                tf.add(cfg.index_json, arcname=cfg.index_json.name)
            if cfg.index_jsonl.exists():
                tf.add(cfg.index_jsonl, arcname=cfg.index_jsonl.name)
        return {"created": True, "path": str(tar_path), "size_bytes": tar_path.stat().st_size}
    except Exception as e:
        try:
            tar_path.unlink(missing_ok=True)
        except Exception:
            pass
        return {"created": False, "error": repr(e)}


def prune_backups(cfg: RetentionConfig) -> dict[str, int]:
    stats = {"deleted": 0}
    root = cfg.backups_dir / "daily"
    if not root.exists():
        return stats
    cutoff = _now() - cfg.keep_days_backups * 86400
    for p in root.iterdir():
        if not p.is_file() or not p.name.endswith(".tar.gz"):
            continue
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
                stats["deleted"] += 1
        except Exception:
            continue
    return stats


def main() -> int:
    cfg = load_config()
    _mkdir(cfg.archive_dir)
    _mkdir(cfg.backups_dir)

    entries = build_index(cfg)
    write_index(cfg, entries)

    prune_stats = prune_artifacts(cfg)
    archive_prune_stats = prune_archives(cfg)

    vacuum_stats = run_journald_vacuum(cfg)
    backup_stats = write_backup(cfg)
    backup_prune_stats = prune_backups(cfg)

    summary = {
        "ts": _now(),
        "artifacts_dir": str(cfg.artifacts_dir),
        "archives_dir": str(cfg.archive_dir),
        "backups_dir": str(cfg.backups_dir),
        "index_json": str(cfg.index_json),
        "index_jsonl": str(cfg.index_jsonl),
        "index_entries": len(entries),
        "prune_artifacts": prune_stats,
        "prune_archives": archive_prune_stats,
        "journald_vacuum": vacuum_stats,
        "backup": backup_stats,
        "prune_backups": backup_prune_stats,
    }

    print(json.dumps(summary, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

