#!/usr/bin/env python3
"""Poll origin/main and deploy registered repos on r530.

This intentionally complements PonceBot's post-merge deploy path. The bot deploys
work it merges itself; this monitor catches direct commits and GitHub PR merges
that land on a repo's default branch outside the bot process.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DB = Path("/home/aponce/codexbot/data/jobs.sqlite")
DEFAULT_STATE = Path("/home/aponce/codexbot/data/main_deploy_state.json")
DEFAULT_EVENTS = Path("/home/aponce/codexbot/data/main_deploy_events.jsonl")
DEFAULT_STATIC_ROOT = Path("/home/aponce/production-sites")
DEFAULT_STATIC_PORT = 8890
LOCK_PATH = Path("/tmp/poncebot-main-deploy-monitor.lock")


@dataclass(frozen=True)
class RepoTarget:
    repo_id: str
    path: Path
    default_branch: str
    metadata: dict[str, Any]


def _now() -> float:
    return time.time()


def slugify(value: str) -> str:
    out = []
    for ch in str(value or "").strip().lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "-":
            out.append("-")
    slug = "".join(out).strip("-")
    return slug or "repo"


def run(
    argv: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def tail(text: str, limit: int = 2000) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return "..." + text[-limit:]


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = dict(event)
    event.setdefault("ts", _now())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def acquire_lock(path: Path) -> Any:
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise SystemExit("another main deploy monitor run is active")
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def _parse_metadata(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def load_targets(db_path: Path) -> list[RepoTarget]:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT repo_id, path, default_branch, status, metadata
        FROM repo_registry
        WHERE status = 'active'
        ORDER BY priority ASC, repo_id ASC
        """
    ).fetchall()
    targets: list[RepoTarget] = []
    seen_paths: set[str] = set()
    for row in rows:
        repo_path = Path(str(row["path"] or "")).expanduser()
        try:
            repo_path = repo_path.resolve()
        except Exception:
            pass
        key = str(repo_path)
        if key in seen_paths or not (repo_path / ".git").exists():
            continue
        seen_paths.add(key)
        targets.append(
            RepoTarget(
                repo_id=str(row["repo_id"] or key),
                path=repo_path,
                default_branch=str(row["default_branch"] or "main").strip() or "main",
                metadata=_parse_metadata(row["metadata"]),
            )
        )
    return targets


def git(repo: Path, args: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return run(["git", *args], cwd=repo, timeout=timeout)


def remote_head(repo: Path, branch: str) -> tuple[bool, str, str]:
    origin = git(repo, ["remote", "get-url", "origin"], timeout=30)
    if origin.returncode != 0:
        return False, "", "no_origin_remote"
    fetch = git(repo, ["fetch", "origin", "--prune"], timeout=180)
    if fetch.returncode != 0:
        return False, "", tail(fetch.stderr or fetch.stdout)
    rev = git(repo, ["rev-parse", f"origin/{branch}"], timeout=60)
    if rev.returncode != 0:
        return False, "", tail(rev.stderr or rev.stdout)
    return True, str(rev.stdout or "").strip(), ""


def ensure_clean_fast_forward(repo: Path, branch: str) -> tuple[bool, str, str]:
    status = git(repo, ["status", "--porcelain", "--untracked-files=no"], timeout=60)
    if status.returncode != 0:
        return False, "", tail(status.stderr or status.stdout)
    if str(status.stdout or "").strip():
        return False, "", "tracked_changes_present"
    current = git(repo, ["rev-parse", "--abbrev-ref", "HEAD"], timeout=60)
    if current.returncode != 0:
        return False, "", tail(current.stderr or current.stdout)
    current_branch = str(current.stdout or "").strip()
    if current_branch != branch:
        checkout = git(repo, ["checkout", branch], timeout=120)
        if checkout.returncode != 0:
            return False, "", tail(checkout.stderr or checkout.stdout)
    merge = git(repo, ["merge", "--ff-only", f"origin/{branch}"], timeout=180)
    if merge.returncode != 0:
        return False, "", tail(merge.stderr or merge.stdout)
    rev = git(repo, ["rev-parse", "HEAD"], timeout=60)
    if rev.returncode != 0:
        return False, "", tail(rev.stderr or rev.stdout)
    return True, str(rev.stdout or "").strip(), ""


def command_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(part) for part in value if str(part).strip()]
    if isinstance(value, str) and value.strip():
        import shlex

        return [str(part) for part in shlex.split(value) if str(part).strip()]
    return []


def discover_script_policy(target: RepoTarget) -> dict[str, Any]:
    for candidate in (target.path / "scripts" / "deploy.sh", target.path / "deploy.sh"):
        if candidate.exists() and candidate.is_file():
            rel = os.path.relpath(str(candidate), str(target.path))
            return {
                "type": "script",
                "source": "repo_script",
                "cwd": str(target.path),
                "command": ["bash", rel],
                "timeout_seconds": 1800,
            }
    return {}


def static_candidate(target: RepoTarget) -> bool:
    return (target.path / "index.html").is_file()


def deploy_policy(target: RepoTarget) -> dict[str, Any]:
    metadata_policy = target.metadata.get("deploy")
    if isinstance(metadata_policy, dict) and metadata_policy.get("enabled", True):
        policy = dict(metadata_policy)
        policy.setdefault("type", "script")
        policy.setdefault("source", "repo_registry")
        return policy
    script_policy = discover_script_policy(target)
    if script_policy:
        return script_policy
    if static_candidate(target):
        return {"type": "static", "source": "static_index"}
    return {"type": "validated_checkout", "source": "no_runtime_policy"}


def _static_ignore(dir_name: str, names: list[str]) -> set[str]:
    ignored = {
        ".git",
        ".gradle",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".venv",
        "venv",
        "build",
        "dist",
        "coverage",
    }
    return {name for name in names if name in ignored or name.endswith(".pyc")}


def _write_static_index(static_root: Path, state: dict[str, Any]) -> None:
    current_root = static_root / "current"
    current_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for repo_id, item in sorted((state.get("repos") or {}).items()):
        if item.get("deploy_type") != "static":
            continue
        slug = str(item.get("static_slug") or slugify(repo_id))
        status = str(item.get("status") or "unknown")
        head = str(item.get("deployed_head") or "")[:12]
        rows.append(
            f'<li><a href="./{slug}/">{slug}</a> '
            f'<span>{status}</span> <code>{head}</code></li>'
        )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>PonceBot Production Sites</title>"
        "<style>body{font-family:system-ui,sans-serif;background:#071014;color:#dff;padding:32px}"
        "a{color:#68e6ff}li{margin:10px 0}code{color:#9ef}</style></head>"
        "<body><h1>PonceBot Production Sites</h1><ul>"
        + "\n".join(rows)
        + "</ul></body></html>"
    )
    (current_root / "index.html").write_text(html, encoding="utf-8")


def deploy_static(target: RepoTarget, head: str, static_root: Path, state: dict[str, Any]) -> dict[str, Any]:
    slug = slugify(target.metadata.get("repo_name") or target.repo_id)
    releases = static_root / "releases" / slug
    release_dir = releases / head[:12]
    current_root = static_root / "current"
    current_link = current_root / slug
    releases.mkdir(parents=True, exist_ok=True)
    current_root.mkdir(parents=True, exist_ok=True)
    if release_dir.exists():
        shutil.rmtree(release_dir)
    shutil.copytree(target.path, release_dir, ignore=_static_ignore)
    tmp_link = current_link.with_name(current_link.name + ".tmp")
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    tmp_link.symlink_to(release_dir, target_is_directory=True)
    tmp_link.replace(current_link)
    _write_static_index(static_root, state)
    verify_url = f"http://127.0.0.1:{DEFAULT_STATIC_PORT}/{slug}/"
    verify = run(["curl", "-fsS", "-I", verify_url], timeout=20)
    if verify.returncode != 0:
        return {
            "ok": False,
            "status": "failed",
            "reason": "static_verify_failed",
            "detail": tail(verify.stderr or verify.stdout),
            "url": verify_url,
            "static_slug": slug,
        }
    return {
        "ok": True,
        "status": "ok",
        "reason": "static_deploy_ok",
        "url": verify_url,
        "static_slug": slug,
        "release_dir": str(release_dir),
    }


def deploy_command_policy(target: RepoTarget, policy: dict[str, Any], head: str) -> dict[str, Any]:
    command = command_list(policy.get("command"))
    if not command:
        return {"ok": False, "status": "failed", "reason": "missing_command"}
    cwd = Path(str(policy.get("cwd") or target.path)).expanduser()
    if str(cwd) == str(target.path):
        cwd = target.path
    timeout = int(policy.get("timeout_seconds") or 900)
    timeout = max(5, min(3600, timeout))
    env = dict(os.environ)
    env.update(
        {
            "PONCEBOT_REPO_ID": target.repo_id,
            "PONCEBOT_REPO_PATH": str(target.path),
            "PONCEBOT_DEFAULT_BRANCH": target.default_branch,
            "PONCEBOT_DEPLOY_COMMIT": head,
            "PONCEBOT_DEPLOY_REASON": "main_monitor",
        }
    )
    if policy.get("background"):
        subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return {
            "ok": True,
            "status": "scheduled",
            "reason": "deploy_scheduled",
            "command": " ".join(command),
        }
    result = run(command, cwd=cwd, env=env, timeout=timeout)
    if result.returncode != 0:
        return {
            "ok": False,
            "status": "failed",
            "reason": "command_failed",
            "command": " ".join(command),
            "stdout": tail(result.stdout),
            "stderr": tail(result.stderr),
        }
    verify_command = command_list(policy.get("verify_command"))
    if verify_command:
        verify = run_verify_with_retries(verify_command, cwd=cwd, env=env, window_seconds=min(timeout, 180))
        if not verify["ok"]:
            return {
                "ok": False,
                "status": "failed",
                "reason": "verify_failed",
                "command": " ".join(command),
                "verify_command": " ".join(verify_command),
                "stdout": verify.get("stdout", ""),
                "stderr": verify.get("stderr", ""),
                "detail": verify.get("detail", ""),
                "verify_attempts": verify.get("attempts", 0),
            }
    return {
        "ok": True,
        "status": "ok",
        "reason": "deploy_ok",
        "command": " ".join(command),
        "stdout": tail(result.stdout),
    }


def run_verify_with_retries(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    window_seconds: int = 120,
    interval_seconds: float = 3.0,
) -> dict[str, Any]:
    deadline = time.time() + max(5, int(window_seconds))
    attempts = 0
    last_stdout = ""
    last_stderr = ""
    while True:
        attempts += 1
        result = run(command, cwd=cwd, env=env, timeout=min(30, max(5, int(window_seconds))))
        last_stdout = tail(result.stdout)
        last_stderr = tail(result.stderr)
        if result.returncode == 0:
            return {"ok": True, "attempts": attempts, "stdout": last_stdout, "stderr": last_stderr}
        if time.time() >= deadline:
            return {
                "ok": False,
                "attempts": attempts,
                "stdout": last_stdout,
                "stderr": last_stderr,
                "detail": last_stderr or last_stdout or f"verify failed after {attempts} attempts",
            }
        time.sleep(interval_seconds)


def validate_checkout(target: RepoTarget) -> dict[str, Any]:
    checks: list[tuple[str, list[str]]] = []
    if (target.path / "package.json").exists():
        checks.append(("node_syntax", ["bash", "-lc", "find . -maxdepth 4 -type f \\( -name '*.js' -o -name '*.mjs' -o -name '*.cjs' \\) -not -path './node_modules/*' -not -path './.git/*' -print0 | xargs -0 -r -n1 node --check"]))
    py_files = list(target.path.glob("*.py"))
    if py_files:
        checks.append(("python_compile", ["bash", "-lc", "python3 -m py_compile *.py"]))
    if (target.path / "gradlew").exists():
        jdk_home = ""
        find_java = run(
            [
                "bash",
                "-lc",
                "find /home/aponce/.local/jdks -maxdepth 4 -type f -path '*/bin/java' | head -1 | sed 's#/bin/java##'",
            ],
            timeout=30,
        )
        if find_java.returncode == 0:
            jdk_home = str(find_java.stdout or "").strip()
        java_exports = ""
        if jdk_home:
            java_exports = f"export JAVA_HOME={jdk_home!r}; export PATH={str(Path(jdk_home) / 'bin')!r}:$PATH; "
        gradle_env = (
            java_exports +
            "mkdir -p /home/aponce/.gradle-codex/main-deploy-project-cache "
            "/home/aponce/.gradle-codex/main-deploy-user-home "
            "/home/aponce/.gradle-codex/main-deploy-build; "
            "cat > /tmp/poncebot-main-deploy-gradle-init.gradle <<'GRADLE'\n"
            "allprojects {\n"
            "    buildDir = new File('/home/aponce/.gradle-codex/main-deploy-build/' + "
            "(project.path == ':' ? 'root' : project.path.substring(1).replace(':', '/')))\n"
            "}\n"
            "GRADLE\n"
            "GRADLE_USER_HOME=/home/aponce/.gradle-codex/main-deploy-user-home "
            "./gradlew -I /tmp/poncebot-main-deploy-gradle-init.gradle "
            "--project-cache-dir /home/aponce/.gradle-codex/main-deploy-project-cache "
            "test --no-daemon"
        )
        checks.append(("gradle_test", ["bash", "-lc", gradle_env]))
    for name, command in checks:
        check_timeout = 900 if name == "gradle_test" else 120
        check = run(command, cwd=target.path, timeout=check_timeout)
        if check.returncode != 0:
            return {
                "ok": False,
                "status": "failed",
                "reason": f"{name}_failed",
                "detail": tail(check.stderr or check.stdout),
            }
    return {"ok": True, "status": "ok", "reason": "checkout_validated"}


def deploy_target(target: RepoTarget, head: str, static_root: Path, state: dict[str, Any]) -> dict[str, Any]:
    policy = deploy_policy(target)
    policy_type = str(policy.get("type") or "script")
    if policy_type == "static":
        result = deploy_static(target, head, static_root, state)
    elif policy_type == "validated_checkout":
        result = validate_checkout(target)
    else:
        result = deploy_command_policy(target, policy, head)
    result["deploy_type"] = policy_type
    result["policy_source"] = policy.get("source")
    return result


def monitor_once(
    *,
    db_path: Path,
    state_path: Path,
    events_path: Path,
    static_root: Path,
    deploy_current: bool,
    dry_run: bool,
) -> int:
    state = read_json(state_path, {"repos": {}, "updated_at": None})
    repos_state = state.setdefault("repos", {})
    targets = load_targets(db_path)
    failures = 0
    changed = 0
    for target in targets:
        repo_state = dict(repos_state.get(target.repo_id) or {})
        ok, head, detail = remote_head(target.path, target.default_branch)
        if not ok:
            reason = "no_origin_remote" if detail == "no_origin_remote" else "fetch_failed"
            status = "skipped" if reason == "no_origin_remote" else "failed"
            if status == "failed":
                failures += 1
            repo_state.update({"status": status, "reason": reason, "detail": detail, "last_checked_at": _now()})
            repos_state[target.repo_id] = repo_state
            append_event(events_path, {"event": reason, "repo_id": target.repo_id, "path": str(target.path), "detail": detail})
            continue
        previous = str(repo_state.get("deployed_head") or "")
        should_deploy = deploy_current or previous != head
        if not should_deploy:
            repo_state.update({"status": "ok", "remote_head": head, "last_checked_at": _now()})
            repos_state[target.repo_id] = repo_state
            continue
        changed += 1
        if dry_run:
            repo_state.update({"status": "dry_run", "remote_head": head, "pending_head": head, "last_checked_at": _now()})
            repos_state[target.repo_id] = repo_state
            append_event(events_path, {"event": "dry_run_pending_deploy", "repo_id": target.repo_id, "head": head})
            continue
        sync_ok, local_head, sync_detail = ensure_clean_fast_forward(target.path, target.default_branch)
        if not sync_ok:
            failures += 1
            repo_state.update(
                {
                    "status": "failed",
                    "reason": "sync_failed",
                    "remote_head": head,
                    "detail": sync_detail,
                    "last_checked_at": _now(),
                }
            )
            repos_state[target.repo_id] = repo_state
            append_event(events_path, {"event": "sync_failed", "repo_id": target.repo_id, "head": head, "detail": sync_detail})
            continue
        result = deploy_target(target, local_head or head, static_root, state)
        if not result.get("ok"):
            failures += 1
        repo_state.update(
            {
                "status": result.get("status") or ("ok" if result.get("ok") else "failed"),
                "reason": result.get("reason"),
                "remote_head": head,
                "deployed_head": (local_head or head) if result.get("ok") else previous,
                "attempted_head": local_head or head,
                "deploy_type": result.get("deploy_type"),
                "policy_source": result.get("policy_source"),
                "static_slug": result.get("static_slug"),
                "url": result.get("url"),
                "detail": result.get("detail") or result.get("stderr") or result.get("stdout") or "",
                "last_checked_at": _now(),
                "last_deploy_at": _now(),
            }
        )
        repos_state[target.repo_id] = repo_state
        append_event(
            events_path,
            {
                "event": "deploy_result",
                "repo_id": target.repo_id,
                "path": str(target.path),
                "head": local_head or head,
                "status": repo_state["status"],
                "reason": repo_state.get("reason"),
                "deploy_type": repo_state.get("deploy_type"),
                "url": repo_state.get("url"),
            },
        )
    state["updated_at"] = _now()
    state["target_count"] = len(targets)
    state["changed_count"] = changed
    state["failure_count"] = failures
    _write_static_index(static_root, state)
    write_json_atomic(state_path, state)
    print(json.dumps({"targets": len(targets), "changed": changed, "failures": failures, "state": str(state_path)}, sort_keys=True))
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--static-root", type=Path, default=DEFAULT_STATIC_ROOT)
    parser.add_argument("--deploy-current", action="store_true", help="Deploy current default-branch heads even if already recorded")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    lock = acquire_lock(LOCK_PATH)
    try:
        return monitor_once(
            db_path=args.db,
            state_path=args.state,
            events_path=args.events,
            static_root=args.static_root,
            deploy_current=bool(args.deploy_current),
            dry_run=bool(args.dry_run),
        )
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
