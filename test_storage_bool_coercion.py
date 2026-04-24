from __future__ import annotations

import unittest

from orchestrator.storage import _coerce_bool


class CoerceBoolTests(unittest.TestCase):
    def test_accepts_text_true_values(self) -> None:
        for raw in ("true", "TRUE", " yes ", "On", "t"):
            with self.subTest(raw=raw):
                self.assertTrue(_coerce_bool(raw, False))

    def test_accepts_text_false_values(self) -> None:
        for raw in ("false", "FALSE", " no ", "Off", "f"):
            with self.subTest(raw=raw):
                self.assertFalse(_coerce_bool(raw, True))

    def test_invalid_value_uses_default(self) -> None:
        self.assertTrue(_coerce_bool("not-a-bool", True))
        self.assertFalse(_coerce_bool("not-a-bool", False))


if __name__ == "__main__":
    unittest.main()
