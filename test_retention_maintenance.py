from __future__ import annotations

import time
from pathlib import Path

from tools.retention_maintenance import RetentionConfig, prune_proactive_health_reports


def _cfg(root: Path, *, keep_days: int = 7, keep_latest: int = 2) -> RetentionConfig:
    return RetentionConfig(
        artifacts_dir=root / "artifacts",
        proactive_health_dir=root / "artifacts" / "proactive_health",
        archive_dir=root / "archives",
        backups_dir=root / "backups",
        index_json=root / "artifacts_index.json",
        index_jsonl=root / "artifacts_index.jsonl",
        keep_days_artifacts=30,
        keep_days_proactive_health_reports=keep_days,
        keep_days_archives=180,
        keep_days_backups=30,
        keep_latest_apks=20,
        keep_latest_proactive_health_reports=keep_latest,
        journald_vacuum_time="14d",
        journald_vacuum_size="200M",
        max_index_entries=5000,
    )


def _touch(path: Path, *, age_days: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(path.name, encoding="utf-8")
    ts = time.time() - age_days * 86400
    path.touch()
    import os

    os.utime(path, (ts, ts))


def test_prune_proactive_health_reports_keeps_latest_pair_and_latest_files(tmp_path: Path) -> None:
    root = tmp_path
    health = root / "artifacts" / "proactive_health"
    _touch(health / "latest.json", age_days=90)
    _touch(health / "latest.md", age_days=90)
    _touch(health / "report_20260101_000000.json", age_days=90)
    _touch(health / "report_20260101_000000.md", age_days=90)
    _touch(health / "report_20260102_000000.json", age_days=80)
    _touch(health / "report_20260102_000000.md", age_days=80)
    _touch(health / "report_20260525_000000.json", age_days=1)
    _touch(health / "report_20260525_000000.md", age_days=1)

    stats = prune_proactive_health_reports(_cfg(root, keep_days=7, keep_latest=1))

    assert stats["deleted_files"] == 4
    assert (health / "latest.json").exists()
    assert (health / "latest.md").exists()
    assert not (health / "report_20260101_000000.json").exists()
    assert not (health / "report_20260102_000000.md").exists()
    assert (health / "report_20260525_000000.json").exists()
    assert (health / "report_20260525_000000.md").exists()


def test_prune_proactive_health_reports_keeps_latest_n_even_when_old(tmp_path: Path) -> None:
    health = tmp_path / "artifacts" / "proactive_health"
    _touch(health / "report_20260101_000000.json", age_days=90)
    _touch(health / "report_20260101_000000.md", age_days=90)
    _touch(health / "report_20260102_000000.json", age_days=80)
    _touch(health / "report_20260102_000000.md", age_days=80)
    _touch(health / "report_20260103_000000.json", age_days=70)
    _touch(health / "report_20260103_000000.md", age_days=70)

    stats = prune_proactive_health_reports(_cfg(tmp_path, keep_days=0, keep_latest=2))

    assert stats["deleted_files"] == 2
    assert not (health / "report_20260101_000000.json").exists()
    assert (health / "report_20260102_000000.json").exists()
    assert (health / "report_20260103_000000.md").exists()
