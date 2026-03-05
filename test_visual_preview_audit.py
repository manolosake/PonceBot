from __future__ import annotations

import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from orchestrator.screenshot import Viewport
from tools import visual_preview_audit as vpa


class TestVisualPreviewAudit(unittest.TestCase):
    def test_run_audit_synthetic_capture_writes_manifest_and_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            html = root / "preview.html"
            html.write_text("<html><body><h1>fixture</h1></body></html>\n", encoding="utf-8")
            out = root / "artifacts"

            report = vpa.run_audit(
                preview_html=html,
                artifacts_dir=out,
                ticket_id="t-123",
                order_branch="order/test",
                max_attempts=2,
                backoff_initial=0.0,
                backoff_max=0.0,
                timeout_ms=500,
                allowed_hosts=set(),
                allow_private=False,
                block_network=True,
                force_capture=False,
                capture_mode="synthetic",
            )

            self.assertEqual(report["status"], "PASS")
            manifest_path = out / "visual_preview_manifest.json"
            bundle_path = out / "visual_preview_audit_bundle.tar.gz"
            self.assertTrue(manifest_path.exists())
            self.assertTrue(bundle_path.exists())

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "PASS")
            self.assertEqual(manifest["ticket_id"], "t-123")
            self.assertEqual(len(manifest["captures"]), 3)
            self.assertEqual(manifest["summary"]["captured_count"], 3)
            self.assertEqual(manifest["summary"]["failed_count"], 0)

            with tarfile.open(bundle_path, "r:gz") as tf:
                names = set(tf.getnames())
            self.assertIn("visual_preview_audit/preview.html", names)
            self.assertIn("visual_preview_audit/visual_preview_manifest.json", names)
            self.assertIn("visual_preview_audit/preview-desktop.png", names)
            self.assertIn("visual_preview_audit/preview-tablet.png", names)
            self.assertIn("visual_preview_audit/preview-mobile.png", names)

    def test_capture_or_validate_accepts_existing_png(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            html = root / "preview.html"
            html.write_text("<html></html>", encoding="utf-8")
            png = root / "preview-desktop.png"
            vpa.write_synthetic_png(png, width=1366, height=768)

            rec = vpa._capture_or_validate(
                label="desktop",
                viewport=Viewport(width=1366, height=768),
                html_path=html,
                out_path=png,
                force_capture=False,
                max_attempts=3,
                backoff_initial=0.0,
                backoff_max=0.0,
                capture_once=lambda: self.fail("capture should not run"),
            )

            self.assertEqual(rec["status"], "validated")
            self.assertTrue(rec["validated_existing"])
            self.assertEqual(rec["attempts"], 0)
            self.assertTrue(rec["sha256"])

    def test_capture_or_validate_retries_then_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            html = root / "preview.html"
            html.write_text("<html></html>", encoding="utf-8")
            out = root / "preview-mobile.png"

            attempts = {"n": 0}

            def _boom() -> None:
                attempts["n"] += 1
                raise RuntimeError("boom")

            rec = vpa._capture_or_validate(
                label="mobile",
                viewport=Viewport(width=412, height=915),
                html_path=html,
                out_path=out,
                force_capture=True,
                max_attempts=3,
                backoff_initial=0.0,
                backoff_max=0.0,
                capture_once=_boom,
            )

            self.assertEqual(attempts["n"], 3)
            self.assertEqual(rec["status"], "failed")
            self.assertEqual(rec["attempts"], 3)
            self.assertEqual(len(rec["backoff_seconds"]), 2)
            self.assertIn("boom", rec["last_error"])


if __name__ == "__main__":
    unittest.main()
