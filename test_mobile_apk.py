import json
import tempfile
import time
import unittest
from pathlib import Path

import bot


class _DummyCfg:
    def __init__(self, *, artifacts_dir: Path, build_script: Path, project_dir: Path) -> None:
        self.auth_enabled = False
        self.mobile_app_artifacts_dir = artifacts_dir
        self.mobile_app_build_script = build_script
        self.mobile_app_project_dir = project_dir
        self.mobile_app_keystore_path = project_dir / "keystore.jks"
        self.mobile_app_key_alias = "omnicrew_release"
        self.mobile_app_keystore_pass = "x"
        self.mobile_app_key_pass = "x"


class _FakeAPI:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.docs: list[Path] = []

    def send_message(self, chat_id: int, text: str, reply_to_message_id: int | None = None):
        self.messages.append(str(text))

    def send_document(self, chat_id: int, file_path: Path, filename=None, caption=None, reply_to_message_id=None):
        self.docs.append(Path(file_path))


class TestMobileApkFlow(unittest.TestCase):
    def setUp(self) -> None:
        bot._APK_BUILD_ACTIVE = False

    def tearDown(self) -> None:
        bot._APK_BUILD_ACTIVE = False

    def test_mobile_latest_apk(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old_apk = root / "omnicrewapp-1.0.0-a-universal-release.apk"
            new_apk = root / "omnicrewapp-1.0.0-b-universal-release.apk"
            old_apk.write_bytes(b"old")
            time.sleep(0.05)
            new_apk.write_bytes(b"new")
            got = bot._mobile_latest_apk(root)
            self.assertIsNotNone(got)
            self.assertEqual(got, new_apk)

    def test_marker_apk_latest_sends_document(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            project = root / "project"
            project.mkdir(parents=True)
            apk = root / "omnicrewapp-1.0.0-test-universal-release.apk"
            apk.write_bytes(b"apk")
            report = {
                "version": "1.0.0",
                "git_sha": "abc123",
                "built_at": "2026-02-21T06:00:00Z",
                "sha256": "deadbeef",
                "artifact_path": str(apk),
            }
            (root / "build_report.json").write_text(json.dumps(report), encoding="utf-8")
            cfg = _DummyCfg(artifacts_dir=root, build_script=project / "noop.sh", project_dir=project)
            api = _FakeAPI()

            handled = bot._send_orchestrator_marker_response(
                kind="apk_latest",
                payload="",
                cfg=cfg,
                api=api,
                chat_id=1,
                user_id=1,
                reply_to_message_id=None,
                orch_q=None,
                profiles=None,
            )
            self.assertTrue(handled)
            self.assertEqual(len(api.docs), 1)
            self.assertEqual(api.docs[0], apk)

    def test_marker_apk_build_worker(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            project = root / "project"
            artifacts = root / "artifacts"
            project.mkdir(parents=True)
            artifacts.mkdir(parents=True)

            build_script = root / "build.sh"
            build_script.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "mkdir -p \"$MOBILE_APP_ARTIFACTS_DIR\"\n"
                "APK=\"$MOBILE_APP_ARTIFACTS_DIR/omnicrewapp-1.0.0-test-universal-release.apk\"\n"
                "echo test > \"$APK\"\n"
                "python3 - <<PY2\n"
                "import json, os, pathlib\n"
                "rep={\"version\":\"1.0.0\",\"git_sha\":\"smoke\",\"built_at\":\"2026-02-21T06:00:00Z\",\"sha256\":\"na\",\"artifact_path\":os.environ[\"MOBILE_APP_ARTIFACTS_DIR\"]+\"/omnicrewapp-1.0.0-test-universal-release.apk\"}\n"
                "pathlib.Path(os.environ[\"MOBILE_APP_ARTIFACTS_DIR\"]+\"/build_report.json\").write_text(json.dumps(rep))\n"
                "PY2\n",
                encoding="utf-8",
            )
            build_script.chmod(0o755)

            cfg = _DummyCfg(artifacts_dir=artifacts, build_script=build_script, project_dir=project)
            api = _FakeAPI()

            handled = bot._send_orchestrator_marker_response(
                kind="apk_build",
                payload="",
                cfg=cfg,
                api=api,
                chat_id=1,
                user_id=1,
                reply_to_message_id=None,
                orch_q=None,
                profiles=None,
            )
            self.assertTrue(handled)

            deadline = time.time() + 8
            while time.time() < deadline and not api.docs:
                time.sleep(0.2)

            self.assertEqual(len(api.docs), 1)
            self.assertTrue(api.docs[0].exists())
            self.assertIn("apk_build: done", api.messages)


if __name__ == "__main__":
    unittest.main()
