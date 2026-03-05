from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.bundle_postwrite_verify import run_verify


class TestBundlePostwriteVerify(unittest.TestCase):
    def test_pass_when_files_match_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            f1 = d / "changes.patch"
            f1.write_text("abc\n", encoding="utf-8")
            f2 = d / "git_status.txt"
            f2.write_text(" M x\n", encoding="utf-8")
            manifest = {
                "files": [
                    {
                        "path": str(f1),
                        "exists": True,
                        "size_bytes": f1.stat().st_size,
                        "sha256": __import__("hashlib").sha256(f1.read_bytes()).hexdigest(),
                    },
                    {
                        "path": str(f2),
                        "exists": True,
                        "size_bytes": f2.stat().st_size,
                        "sha256": __import__("hashlib").sha256(f2.read_bytes()).hexdigest(),
                    },
                ]
            }
            mp = d / "bundle_immutability_manifest.json"
            mp.write_text(json.dumps(manifest), encoding="utf-8")
            out = run_verify(d, mp)
            self.assertEqual(out["status"], "PASS")
            self.assertEqual(out["exit_code"], 0)

    def test_fail_when_file_changes_after_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            f1 = d / "changes.patch"
            f1.write_text("abc\n", encoding="utf-8")
            manifest = {
                "files": [
                    {
                        "path": str(f1),
                        "exists": True,
                        "size_bytes": f1.stat().st_size,
                        "sha256": __import__("hashlib").sha256(f1.read_bytes()).hexdigest(),
                    }
                ]
            }
            mp = d / "bundle_immutability_manifest.json"
            mp.write_text(json.dumps(manifest), encoding="utf-8")
            f1.write_text("abc\nmutated\n", encoding="utf-8")
            out = run_verify(d, mp)
            self.assertEqual(out["status"], "FAIL")
            self.assertEqual(out["exit_code"], 2)


if __name__ == "__main__":
    unittest.main()
