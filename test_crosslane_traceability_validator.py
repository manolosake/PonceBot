from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.crosslane_traceability_validator import run_validation


class TestCrosslaneTraceabilityValidator(unittest.TestCase):
    def _write_common_files(self, d: Path, *, expected_branch: str, reported_branch: str, branch_matches_expected: bool) -> None:
        (d / "wormhole_scene_trace.json").write_text(
            json.dumps(
                {
                    "expected_order_branch": expected_branch,
                    "reported_branch": reported_branch,
                    "branch_matches_expected": branch_matches_expected,
                }
            ),
            encoding="utf-8",
        )
        (d / "git_status.txt").write_text("M\ttools/a.py\n?? tools/new.py\n", encoding="utf-8")
        (d / "changes.patch").write_text(
            (
                "diff --git a/tools/a.py b/tools/a.py\n"
                "diff --git a/tools/new.py b/tools/new.py\n"
            ),
            encoding="utf-8",
        )

    def test_pass(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            expected = "feature/order-4084914d-nuevo-proyecto-ceo-wormhole-vnext-hipe"
            self._write_common_files(d, expected_branch=expected, reported_branch=expected, branch_matches_expected=True)
            (d / "runtime_telemetry_report.json").write_text(
                json.dumps(
                    {
                        "viewports": {
                            "desktop": {
                                "sample_count_frames": 500,
                                "avg_fps": 58.2,
                                "p95_frame_ms": 18.7,
                                "degrade_level": 0,
                                "preset": "cinematic",
                                "quality_tier": "ultra",
                            },
                            "tablet": {
                                "sample_count_frames": 500,
                                "avg_fps": 51.0,
                                "p95_frame_ms": 20.2,
                                "degrade_level": 0,
                                "preset": "balanced",
                                "quality_tier": "high",
                            },
                            "mobile": {
                                "sample_count_frames": 500,
                                "avg_fps": 45.5,
                                "p95_frame_ms": 21.5,
                                "degrade_level": 1,
                                "preset": "performance",
                                "quality_tier": "medium",
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            report = run_validation(
                artifacts_dir=d,
                expected_branch=expected,
                backend_trace_path=d / "wormhole_scene_trace.json",
                git_status_path=d / "git_status.txt",
                changes_patch_path=d / "changes.patch",
                runtime_telemetry_path=d / "runtime_telemetry_report.json",
            )
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["mismatches"], [])

    def test_fail_on_branch_patch_and_telemetry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            expected = "feature/order-4084914d-nuevo-proyecto-ceo-wormhole-vnext-hipe"
            self._write_common_files(
                d,
                expected_branch=expected,
                reported_branch="feature/other-branch",
                branch_matches_expected=False,
            )
            # Introduce patch mismatch: remove untracked path from patch.
            (d / "changes.patch").write_text("diff --git a/tools/a.py b/tools/a.py\n", encoding="utf-8")
            # Missing mobile viewport.
            (d / "runtime_telemetry_report.json").write_text(
                json.dumps(
                    {
                        "viewports": {
                            "desktop": {
                                "sample_count_frames": 500,
                                "avg_fps": 58.2,
                                "p95_frame_ms": 18.7,
                                "degrade_level": 0,
                                "preset": "cinematic",
                                "quality_tier": "ultra",
                            },
                            "tablet": {
                                "sample_count_frames": 500,
                                "avg_fps": 51.0,
                                "p95_frame_ms": 20.2,
                                "degrade_level": 0,
                                "preset": "balanced",
                                "quality_tier": "high",
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            report = run_validation(
                artifacts_dir=d,
                expected_branch=expected,
                backend_trace_path=d / "wormhole_scene_trace.json",
                git_status_path=d / "git_status.txt",
                changes_patch_path=d / "changes.patch",
                runtime_telemetry_path=d / "runtime_telemetry_report.json",
            )
            self.assertEqual(report["status"], "FAIL")
            codes = {m["code"] for m in report["mismatches"]}
            self.assertIn("BRANCH_MISMATCH", codes)
            self.assertIn("FRONTEND_BUNDLE_INCOHERENT", codes)
            self.assertIn("MISSING_VIEWPORT_TELEMETRY", codes)


if __name__ == "__main__":
    unittest.main()

