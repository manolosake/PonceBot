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
