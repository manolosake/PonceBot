#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace('+00:00','Z')

def snapshot_files(base_dir: Path, rel_paths: list[str]) -> dict[str, Any]:
    out={}
    ts=utc_now()
    for rel in rel_paths:
        p=(base_dir/rel).resolve()
        if not p.exists() or not p.is_file():
            out[rel]={"exists":False,"size_bytes":0,"sha256":"","mtime_epoch":0.0,"captured_at_utc":ts}
            continue
        b=p.read_bytes(); st=p.stat()
        out[rel]={"exists":True,"size_bytes":len(b),"sha256":hashlib.sha256(b).hexdigest(),"mtime_epoch":float(st.st_mtime),"captured_at_utc":ts}
    return out

def compare_snapshots(t0: dict[str, Any], tn: dict[str, Any]) -> list[dict[str, Any]]:
    drift=[]
    for rel in sorted(set(t0)|set(tn)):
        a=t0.get(rel,{}); b=tn.get(rel,{})
        for f in ("exists","size_bytes","sha256","mtime_epoch"):
            if a.get(f)!=b.get(f):
                drift.append({"file":rel,"field":f,"t0":a.get(f),"tN":b.get(f)})
    return drift

def main() -> int:
    ap=argparse.ArgumentParser(description='Manifest drift checker')
    ap.add_argument('--artifacts-dir', required=True)
    ap.add_argument('--critical-files', required=True)
    ap.add_argument('--sleep-seconds', type=float, default=2.0)
    ap.add_argument('--manifest-t0', required=True)
    ap.add_argument('--manifest-tn', required=True)
    ap.add_argument('--report', required=True)
    args=ap.parse_args()

    base=Path(args.artifacts_dir).resolve()
    files=[x.strip() for x in args.critical_files.split(',') if x.strip()]
    t0=snapshot_files(base, files)
    time.sleep(max(args.sleep_seconds,0.0))
    tn=snapshot_files(base, files)
    drift=compare_snapshots(t0, tn)
    report={"check":"manifest_drift_checker","artifacts_dir":str(base),"critical_files":files,"captured_at_utc":utc_now(),"drift_detected":bool(drift),"drift":drift,"exit_code":1 if drift else 0}

    Path(args.manifest_t0).write_text(json.dumps(t0,indent=2,ensure_ascii=True)+'\n',encoding='utf-8')
    Path(args.manifest_tn).write_text(json.dumps(tn,indent=2,ensure_ascii=True)+'\n',encoding='utf-8')
    Path(args.report).write_text(json.dumps(report,indent=2,ensure_ascii=True)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=True))
    return report['exit_code']

if __name__=='__main__':
    raise SystemExit(main())
