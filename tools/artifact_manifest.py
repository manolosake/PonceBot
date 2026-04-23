#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import ntpath
import time
from pathlib import Path
from typing import Any


def _utc_iso(ts: float | None = None) -> str:
    t = float(time.time() if ts is None else ts)
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _is_absolute_entry_name(name: str) -> bool:
    # Use both local-path semantics and Windows semantics so checks are platform-agnostic.
    return Path(name).is_absolute() or ntpath.isabs(name)


def _entry_candidate_path(name: str) -> Path:
    # Interpret Windows-style separators consistently for platform-agnostic safety checks.
    return Path(name.replace('\\', '/'))


def _collect_files(
    artifacts_dir: Path,
    *,
    manifest_name: str,
    validation_name: str,
    excluded_names: set[str] | None = None,
) -> list[Path]:
    excluded = set(excluded_names or set())
    files: list[Path] = []
    for p in sorted(artifacts_dir.iterdir()):
        if not p.is_file():
            continue
        if p.name in {manifest_name, validation_name}:
            continue
        if p.name in excluded:
            continue
        files.append(p)
    return files


def _build_entries(
    artifacts_dir: Path,
    *,
    manifest_name: str,
    validation_name: str,
    excluded_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for p in _collect_files(
        artifacts_dir,
        manifest_name=manifest_name,
        validation_name=validation_name,
        excluded_names=excluded_names,
    ):
        st = p.stat()
        entries.append(
            {
                'name': p.name,
                'path': str(p),
                'size_bytes': int(st.st_size),
                'sha256': _sha256(p),
                'mtime_utc': _utc_iso(st.st_mtime),
            }
        )
    return entries


def cmd_write(args: argparse.Namespace) -> int:
    art = Path(args.artifacts_dir).expanduser().resolve()
    art.mkdir(parents=True, exist_ok=True)
    manifest_path = art / args.output
    validation_name = args.validation
    entries = _build_entries(
        art,
        manifest_name=manifest_path.name,
        validation_name=validation_name,
        excluded_names=set(args.exclude or []),
    )
    payload = {
        'schema_version': 1,
        'generated_at': _utc_iso(),
        'artifacts_dir': str(art),
        'entry_count': len(entries),
        'entries': entries,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(str(manifest_path))
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    art = Path(args.artifacts_dir).expanduser().resolve()
    manifest_path = art / args.manifest
    if not manifest_path.exists():
        print(json.dumps({'ok': False, 'error': f'missing manifest: {manifest_path}'}, indent=2))
        return 2

    obj = json.loads(manifest_path.read_text(encoding='utf-8'))
    entries = obj.get('entries') if isinstance(obj, dict) else None
    if not isinstance(entries, list):
        print(json.dumps({'ok': False, 'error': 'invalid manifest entries'}, indent=2))
        return 2

    mismatches: list[dict[str, Any]] = []
    names_in_manifest = set()
    for entry in entries:
        if not isinstance(entry, dict):
            mismatches.append({'name': '<invalid>', 'reason': 'entry is not object'})
            continue
        name = str(entry.get('name') or '').strip()
        if not name:
            mismatches.append({'name': '<missing>', 'reason': 'entry missing name'})
            continue
        names_in_manifest.add(name)
        candidate = _entry_candidate_path(name)
        if _is_absolute_entry_name(name):
            mismatches.append({'name': name, 'reason': 'entry name is absolute path'})
            continue
        p = (art / candidate).resolve()
        try:
            p.relative_to(art)
        except ValueError:
            mismatches.append({'name': name, 'reason': 'entry path escapes artifacts dir'})
            continue
        if not p.exists():
            mismatches.append({'name': name, 'reason': 'file missing on disk'})
            continue
        if not p.is_file():
            mismatches.append({'name': name, 'reason': 'path is not a file'})
            continue
        size = p.stat().st_size
        sha = _sha256(p)
        if int(entry.get('size_bytes', -1)) != int(size):
            mismatches.append({'name': name, 'reason': 'size mismatch', 'manifest': entry.get('size_bytes'), 'actual': size})
        if str(entry.get('sha256') or '') != sha:
            mismatches.append({'name': name, 'reason': 'sha256 mismatch', 'manifest': entry.get('sha256'), 'actual': sha})

    # Optional strict mode: fail on on-disk files omitted from manifest.
    if bool(args.strict_extra):
        validation_name = args.validation
        for p in _collect_files(
            art,
            manifest_name=manifest_path.name,
            validation_name=validation_name,
            excluded_names=set(args.exclude or []),
        ):
            if p.name not in names_in_manifest:
                mismatches.append({'name': p.name, 'reason': 'file missing from manifest'})

    ok = not mismatches
    result = {
        'schema_version': 1,
        'checked_at': _utc_iso(),
        'artifacts_dir': str(art),
        'manifest_path': str(manifest_path),
        'ok': ok,
        'mismatch_count': len(mismatches),
        'mismatches': mismatches,
    }

    if args.output:
        (art / args.output).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0 if ok else 2


def main() -> int:
    ap = argparse.ArgumentParser(description='Write and validate artifact manifest using size+sha256.')
    sub = ap.add_subparsers(dest='cmd', required=True)

    ap_write = sub.add_parser('write', help='write FINAL_MANIFEST-style file from current on-disk artifacts')
    ap_write.add_argument('--artifacts-dir', required=True)
    ap_write.add_argument('--output', default='FINAL_MANIFEST.json')
    ap_write.add_argument('--validation', default='FINAL_VALIDATION.json')
    ap_write.add_argument('--exclude', action='append', default=[], help='file name to exclude from manifest listing')
    ap_write.set_defaults(func=cmd_write)

    ap_check = sub.add_parser('check', help='verify manifest vs on-disk files')
    ap_check.add_argument('--artifacts-dir', required=True)
    ap_check.add_argument('--manifest', default='FINAL_MANIFEST.json')
    ap_check.add_argument('--output', default='FINAL_VALIDATION.json')
    ap_check.add_argument('--validation', default='FINAL_VALIDATION.json')
    ap_check.add_argument('--exclude', action='append', default=[], help='file name excluded from strict-extra scan')
    ap_check.add_argument('--strict-extra', action='store_true', help='also fail when extra on-disk files are not listed')
    ap_check.set_defaults(func=cmd_check)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == '__main__':
    raise SystemExit(main())
