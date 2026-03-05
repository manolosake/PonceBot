import json
import tempfile
import unittest
from pathlib import Path

from tools.wormhole_scene_contract import validate_contract


class TestWormholeSceneContract(unittest.TestCase):
    def test_contract_file_is_valid(self) -> None:
        p = Path("docs/contracts/wormhole_scene_contract.v1.json")
        data = json.loads(p.read_text(encoding="utf-8"))
        errs = validate_contract(data)
        self.assertEqual(errs, [])

    def test_validate_detects_missing_fields(self) -> None:
        payload = {
            "contract_name": "wormhole_scene_contract",
            "contract_version": "1.0.0",
            "scene_version": "wormhole-v3",
            "seed": {"value": 1},
            "quality_presets": {
                "cinematic": {},
                "balanced": {},
                "performance": {},
            },
            "viewport_defaults": {"desktop": "cinematic", "tablet": "balanced", "mobile": "performance"},
            "parameter_ranges": {},
            "traceability": {},
        }
        errs = validate_contract(payload)
        self.assertTrue(any("missing keys" in e for e in errs))


if __name__ == "__main__":
    unittest.main()
