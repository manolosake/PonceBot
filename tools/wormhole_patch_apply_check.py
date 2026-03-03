#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def _parse_status_line(line: str) -> str:
    if len(line) >= 4 and line[2] == ' ':
        payload = line[3:]
    elif len(line) >= 3 and line[1] == ' ':
        payload = line[2:]
    else:
        payload = line
    if ' -> ' in payload:
        payload = payload.split(' -> ', 1)[1]
    return payload.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description='Generate patch_apply_check.json from git_status and changes.patch')
    ap.add_argument('--artifacts-dir', required=True)
    ap.add_argument('--repo-root', default='.')
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    art = Path(args.artifacts_dir).resolve()
    repo = Path(args.repo_root).resolve()
    status_path = art / 'git_status.txt'
    patch_path = art / 'changes.patch'
    out_path = Path(args.out).resolve() if args.out else art / 'patch_apply_check.json'

    declared: list[str] = []
    for raw in status_path.read_text(encoding='utf-8', errors='replace').splitlines():
        line = raw.rstrip('\n')
        if not line:
            continue
        p = _parse_status_line(line)
        if p:
            declared.append(p)

    proc = subprocess.run(
        ['git', 'apply', '--check', '--reverse', str(patch_path)],
        cwd=str(repo),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    payload = {
        'status': 'PASS' if proc.returncode == 0 else 'FAIL',
        'exit_code': int(proc.returncode),
        'patch_path': str(patch_path),
        'declared_files': declared,
        'stdout': (proc.stdout or '').strip(),
        'stderr': (proc.stderr or '').strip(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if proc.returncode == 0 else 2


if __name__ == '__main__':
    raise SystemExit(main())
