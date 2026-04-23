from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from tools.artifact_manifest import cmd_check, cmd_write


class TestArtifactManifest(unittest.TestCase):
    def _run_check(self, artifacts_dir: Path, *, manifest: str = "FINAL_MANIFEST.json") -> tuple[int, dict]:
        stdout = StringIO()
        args = argparse.Namespace(
            artifacts_dir=str(artifacts_dir),
            manifest=manifest,
            output="",
            validation="FINAL_VALIDATION.json",
            exclude=[],
            strict_extra=False,
        )
        with redirect_stdout(stdout):
            rc = cmd_check(args)
        return rc, json.loads(stdout.getvalue())

    def test_check_passes_for_valid_in_tree_entry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            (artifacts_dir / "proof.txt").write_text("ok\n", encoding="utf-8")
            write_args = argparse.Namespace(
                artifacts_dir=str(artifacts_dir),
                output="FINAL_MANIFEST.json",
                validation="FINAL_VALIDATION.json",
                exclude=[],
            )
            cmd_write(write_args)

            rc, payload = self._run_check(artifacts_dir)

            self.assertEqual(rc, 0)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["mismatch_count"], 0)

    def test_check_rejects_parent_escape_entry_name(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            artifacts_dir = root / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            outside = root / "outside.txt"
            outside.write_text("outside\n", encoding="utf-8")

            manifest = {
                "schema_version": 1,
                "entries": [
                    {
                        "name": "../outside.txt",
                        "size_bytes": outside.stat().st_size,
                        "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
                    }
                ],
            }
            (artifacts_dir / "FINAL_MANIFEST.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            rc, payload = self._run_check(artifacts_dir)

            self.assertEqual(rc, 2)
            self.assertFalse(payload["ok"])
            self.assertIn(
                {"name": "../outside.txt", "reason": "entry path escapes artifacts dir"},
                payload["mismatches"],
            )

    def test_check_rejects_windows_parent_escape_entry_name(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            artifacts_dir = root / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            outside = root / "outside.txt"
            outside.write_text("outside\n", encoding="utf-8")

            manifest = {
                "schema_version": 1,
                "entries": [
                    {
                        "name": r"..\outside.txt",
                        "size_bytes": outside.stat().st_size,
                        "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
                    }
                ],
            }
            (artifacts_dir / "FINAL_MANIFEST.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            rc, payload = self._run_check(artifacts_dir)

            self.assertEqual(rc, 2)
            self.assertFalse(payload["ok"])
            self.assertIn(
                {"name": r"..\outside.txt", "reason": "entry path escapes artifacts dir"},
                payload["mismatches"],
            )

    def test_check_rejects_absolute_entry_name(self) -> None:
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as outside_td:
            artifacts_dir = Path(td)
            outside = Path(outside_td) / "outside.txt"
            outside.write_text("outside\n", encoding="utf-8")

            manifest = {
                "schema_version": 1,
                "entries": [
                    {
                        "name": str(outside),
                        "size_bytes": outside.stat().st_size,
                        "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
                    }
                ],
            }
            (artifacts_dir / "FINAL_MANIFEST.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            rc, payload = self._run_check(artifacts_dir)

            self.assertEqual(rc, 2)
            self.assertFalse(payload["ok"])
            self.assertIn(
                {"name": str(outside), "reason": "entry name is absolute path"},
                payload["mismatches"],
            )

    def test_check_rejects_windows_drive_absolute_entry_name(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            windows_abs_names = ["C:/temp/outside.txt", r"C:\temp\outside.txt"]

            for name in windows_abs_names:
                manifest = {
                    "schema_version": 1,
                    "entries": [
                        {
                            "name": name,
                            "size_bytes": 1,
                            "sha256": "deadbeef",
                        }
                    ],
                }
                (artifacts_dir / "FINAL_MANIFEST.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )

                rc, payload = self._run_check(artifacts_dir)

                self.assertEqual(rc, 2)
                self.assertFalse(payload["ok"])
                self.assertIn(
                    {"name": name, "reason": "entry name is absolute path"},
                    payload["mismatches"],
                )

    def test_check_rejects_windows_unc_absolute_entry_name(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifacts_dir = Path(td)
            name = r"\\server\share\outside.txt"

            manifest = {
                "schema_version": 1,
                "entries": [
                    {
                        "name": name,
                        "size_bytes": 1,
                        "sha256": "deadbeef",
                    }
                ],
            }
            (artifacts_dir / "FINAL_MANIFEST.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            rc, payload = self._run_check(artifacts_dir)

            self.assertEqual(rc, 2)
            self.assertFalse(payload["ok"])
            self.assertIn(
                {"name": name, "reason": "entry name is absolute path"},
                payload["mismatches"],
            )


if __name__ == "__main__":
    unittest.main()
