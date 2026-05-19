import json
import sqlite3
import subprocess
from pathlib import Path

from tools import main_deploy_monitor as mdm


def test_slugify_is_stable():
    assert mdm.slugify("ExecutiveDashboard") == "executivedashboard"
    assert mdm.slugify("QuoteKit Studio!") == "quotekit-studio"
    assert mdm.slugify("") == "repo"


def test_load_targets_reads_active_repos(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, stdout=subprocess.PIPE)
    db = tmp_path / "jobs.sqlite"
    con = sqlite3.connect(db)
    con.execute(
        """
        CREATE TABLE repo_registry (
          repo_id TEXT PRIMARY KEY,
          path TEXT NOT NULL UNIQUE,
          default_branch TEXT NOT NULL DEFAULT 'main',
          autonomy_enabled INTEGER NOT NULL DEFAULT 1,
          priority INTEGER NOT NULL DEFAULT 2,
          runtime_mode TEXT NOT NULL DEFAULT 'ceo-bounded',
          daily_budget REAL NOT NULL DEFAULT 0.0,
          status TEXT NOT NULL DEFAULT 'active',
          metadata TEXT NOT NULL DEFAULT '{}',
          created_at REAL NOT NULL,
          updated_at REAL NOT NULL
        )
        """
    )
    con.execute(
        "INSERT INTO repo_registry(repo_id,path,default_branch,status,metadata,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
        ("repo-1", str(repo), "main", "active", json.dumps({"repo_name": "Demo"}), 1.0, 1.0),
    )
    con.commit()

    targets = mdm.load_targets(db)

    assert len(targets) == 1
    assert targets[0].repo_id == "repo-1"
    assert targets[0].metadata["repo_name"] == "Demo"


def test_static_policy_for_index_repo(tmp_path):
    repo = tmp_path / "site"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "index.html").write_text("<h1>ok</h1>", encoding="utf-8")
    target = mdm.RepoTarget("site-1", repo, "main", {"repo_name": "Site"})

    policy = mdm.deploy_policy(target)

    assert policy["type"] == "static"
    assert policy["source"] == "static_index"


def test_static_policy_for_codexbot_preview_repo(tmp_path):
    repo = tmp_path / "preview"
    repo.mkdir()
    (repo / ".git").mkdir()
    preview = repo / ".codexbot_preview"
    preview.mkdir()
    (preview / "preview.html").write_text("<h1>preview</h1>", encoding="utf-8")
    target = mdm.RepoTarget("preview-1", repo, "main", {"repo_name": "Preview"})

    policy = mdm.deploy_policy(target)

    assert policy["type"] == "static"
    assert policy["source"] == "codexbot_preview"
    assert policy["source_dir"] == ".codexbot_preview"
    assert policy["entrypoint"] == "preview.html"


def test_landing_policy_for_readme_only_repo(tmp_path):
    repo = tmp_path / "cli"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "README.md").write_text("# CLI Product\n\nA useful product.", encoding="utf-8")
    target = mdm.RepoTarget("cli-1", repo, "main", {"repo_name": "CLI Product"})

    policy = mdm.deploy_policy(target)

    assert policy["type"] == "static_landing"
    assert policy["source"] == "readme_landing"


def test_static_preview_deploy_copies_preview_to_index(tmp_path, monkeypatch):
    repo = tmp_path / "preview"
    repo.mkdir()
    preview = repo / ".codexbot_preview"
    preview.mkdir()
    (preview / "preview.html").write_text("<h1>Preview</h1>", encoding="utf-8")
    target = mdm.RepoTarget("preview-1", repo, "main", {"repo_name": "Preview"})

    monkeypatch.setattr(
        mdm,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="HTTP/1.1 200 OK", stderr=""),
    )

    result = mdm.deploy_static(
        target,
        "abcdef1234567890",
        tmp_path / "static",
        {"repos": {}},
        policy={"source_dir": ".codexbot_preview", "entrypoint": "preview.html"},
    )

    assert result["ok"] is True
    deployed_index = tmp_path / "static" / "current" / "preview" / "index.html"
    assert deployed_index.read_text(encoding="utf-8") == "<h1>Preview</h1>"


def test_readme_landing_deploy_writes_product_page_and_index(tmp_path, monkeypatch):
    repo = tmp_path / "cli"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "README.md").write_text("# CLI Product\n\n## Offer\n\n- Useful output", encoding="utf-8")
    (repo / "VALIDATION.md").write_text("# Validation\n\nPASS demo command.", encoding="utf-8")
    target = mdm.RepoTarget("cli-1", repo, "main", {"repo_name": "CLI Product"})

    monkeypatch.setattr(
        mdm,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="HTTP/1.1 200 OK", stderr=""),
    )

    result = mdm.deploy_readme_landing(target, "abcdef1234567890", tmp_path / "static", {"repos": {}})

    assert result["ok"] is True
    assert result["reason"] == "static_landing_deploy_ok"
    page = (tmp_path / "static" / "current" / "cli-product" / "index.html").read_text(encoding="utf-8")
    assert "CLI Product" in page
    assert "Validation Evidence" in page


def test_static_index_renders_existing_static_landing_rows(tmp_path):
    mdm._write_static_index(
        tmp_path / "static",
        {
            "repos": {
                "cli-1": {
                    "deploy_type": "static_landing",
                    "static_slug": "cli-product",
                    "title": "CLI Product",
                    "status": "ok",
                    "deployed_head": "abcdef1234567890",
                }
            }
        },
    )

    index = (tmp_path / "static" / "current" / "index.html").read_text(encoding="utf-8")
    assert "CLI Product" in index
    assert "./cli-product/" in index
    assert "static_landing" in index


def test_remote_head_retries_transient_fetch(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    calls = []

    def fake_git(_repo, args, timeout=120):
        calls.append(list(args))
        if args[:3] == ["remote", "get-url", "origin"]:
            return subprocess.CompletedProcess(args, 0, stdout="git@github.com:x/y.git\n", stderr="")
        if args[:3] == ["fetch", "origin", "--prune"]:
            if len([call for call in calls if call[:3] == ["fetch", "origin", "--prune"]]) == 1:
                return subprocess.CompletedProcess(args, 128, stdout="", stderr="ssh: Could not resolve hostname github.com")
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if args[:2] == ["rev-parse", "origin/main"]:
            return subprocess.CompletedProcess(args, 0, stdout="abc123\n", stderr="")
        raise AssertionError(args)

    monkeypatch.setattr(mdm, "git", fake_git)
    monkeypatch.setattr(mdm.time, "sleep", lambda _seconds: None)

    ok, head, detail = mdm.remote_head(repo, "main")

    assert ok is True
    assert head == "abc123"
    assert detail == ""
    assert calls.count(["fetch", "origin", "--prune"]) == 2


def test_monitor_once_preserves_previous_ok_state_on_transient_fetch(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    db = tmp_path / "jobs.sqlite"
    con = sqlite3.connect(db)
    con.execute(
        """
        CREATE TABLE repo_registry (
          repo_id TEXT PRIMARY KEY,
          path TEXT NOT NULL UNIQUE,
          default_branch TEXT NOT NULL DEFAULT 'main',
          autonomy_enabled INTEGER NOT NULL DEFAULT 1,
          priority INTEGER NOT NULL DEFAULT 2,
          runtime_mode TEXT NOT NULL DEFAULT 'ceo-bounded',
          daily_budget REAL NOT NULL DEFAULT 0.0,
          status TEXT NOT NULL DEFAULT 'active',
          metadata TEXT NOT NULL DEFAULT '{}',
          created_at REAL NOT NULL,
          updated_at REAL NOT NULL
        )
        """
    )
    con.execute(
        "INSERT INTO repo_registry(repo_id,path,default_branch,status,metadata,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
        ("repo-1", str(repo), "main", "active", "{}", 1.0, 1.0),
    )
    con.commit()
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps({"repos": {"repo-1": {"status": "ok", "remote_head": "abc123", "deployed_head": "abc123"}}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        mdm,
        "remote_head",
        lambda _repo, _branch: (False, "", "ssh: Could not resolve hostname github.com: Temporary failure in name resolution"),
    )

    rc = mdm.monitor_once(
        db_path=db,
        state_path=state_path,
        events_path=tmp_path / "events.jsonl",
        static_root=tmp_path / "static",
        deploy_current=False,
        dry_run=False,
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    repo_state = state["repos"]["repo-1"]
    assert rc == 0
    assert state["failure_count"] == 0
    assert repo_state["status"] == "ok"
    assert repo_state["reason"] == "fetch_transient_preserved"
    assert repo_state["remote_head"] == "abc123"
