from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "tools" / "delivery_evidence_gate.py"


class DeliveryEvidenceGateTests(unittest.TestCase):
    def test_pass_when_required_evidence_exists(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            # Required consistency files.
            (art / "git_status.txt").write_text(" M README.md\n M tools/gate.py\n", encoding="utf-8")
            (art / "changes.patch").write_text(
                (
                    "diff --git a/README.md b/README.md\n"
                    "--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-a\n+b\n"
                    "diff --git a/tools/gate.py b/tools/gate.py\n"
                    "--- a/tools/gate.py\n+++ b/tools/gate.py\n@@ -1 +1 @@\n-a\n+b\n"
                ),
                encoding="utf-8",
            )
            # Required report.
            (art / "sample_summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
            # Integrity report consistency claims.
            (art / "integrity_report.json").write_text(
                json.dumps(
                    {
                        "files_in_patch": ["README.md", "tools/gate.py"],
                        "visual_evidence": [
                            "desktop_capture.png",
                            "tablet_capture.png",
                            "mobile_capture.png",
                            "preview.html",
                        ],
                    }
                ),
                encoding="utf-8",
            )
            # Required visual evidence.
            (art / "desktop_capture.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            (art / "tablet_capture.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            (art / "mobile_capture.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            # Preview validity.
            (art / "preview.html").write_text("<!doctype html><html><body>x</body></html>\n", encoding="utf-8")
            # Telegram traceability.
            (art / "telegram_trace.jsonl").write_text('{"event":"send"}\n', encoding="utf-8")

            p = subprocess.run(
                [sys.executable, str(SCRIPT), "--artifacts-dir", str(art), "--workspace-dir", str(ROOT)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(p.returncode, 0, msg=p.stdout + "\n" + p.stderr)
            report = json.loads((art / "sre_evidence_gate_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report.get("status"), "PASS")

    def test_fail_when_patch_empty_and_visual_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            (art / "git_status.txt").write_text(" M README.md\n", encoding="utf-8")
            (art / "changes.patch").write_text("", encoding="utf-8")
            (art / "sample_report.json").write_text("{}", encoding="utf-8")
            (art / "preview.html").write_text("<html></html>\n", encoding="utf-8")
            (art / "telegram_trace.json").write_text('{"source":"telegram"}\n', encoding="utf-8")
            # Only desktop present -> tablet/mobile missing must fail.
            (art / "desktop_capture.png").write_bytes(b"\x89PNG\r\n\x1a\n")

            p = subprocess.run(
                [sys.executable, str(SCRIPT), "--artifacts-dir", str(art), "--workspace-dir", str(ROOT)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(p.returncode, 2, msg=p.stdout + "\n" + p.stderr)
            report = json.loads((art / "sre_evidence_gate_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report.get("status"), "FAIL")
            failed = [c["key"] for c in report.get("checks", []) if not c.get("ok")]
            self.assertIn("required_changes_patch_non_empty", failed)
            self.assertIn("required_tablet_screenshot", failed)
            self.assertIn("required_mobile_screenshot", failed)

    def test_fail_when_files_in_patch_claim_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            (art / "git_status.txt").write_text(" M README.md\n", encoding="utf-8")
            (art / "changes.patch").write_text(
                "diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-a\n+b\n",
                encoding="utf-8",
            )
            (art / "sample_summary.json").write_text("{}", encoding="utf-8")
            (art / "integrity_report.json").write_text(
                json.dumps(
                    {
                        "files_in_patch": ["README.md", "missing.py"],
                        "visual_evidence": ["desktop_capture.png", "tablet_capture.png", "mobile_capture.png", "preview.html"],
                    }
                ),
                encoding="utf-8",
            )
            (art / "preview.html").write_text("<html></html>\n", encoding="utf-8")
            (art / "telegram_trace.json").write_text('{"source":"telegram"}\n', encoding="utf-8")
            (art / "desktop_capture.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            (art / "tablet_capture.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            (art / "mobile_capture.png").write_bytes(b"\x89PNG\r\n\x1a\n")

            p = subprocess.run(
                [sys.executable, str(SCRIPT), "--artifacts-dir", str(art), "--workspace-dir", str(ROOT)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(p.returncode, 2, msg=p.stdout + "\n" + p.stderr)
            report = json.loads((art / "sre_evidence_gate_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report.get("status"), "FAIL")
            failed = [c["key"] for c in report.get("checks", []) if not c.get("ok")]
            self.assertIn("integrity_report_patch_alignment", failed)


if __name__ == "__main__":
    unittest.main()
