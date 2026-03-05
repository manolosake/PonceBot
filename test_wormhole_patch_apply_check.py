import unittest

from tools.wormhole_patch_apply_check import _parse_status_line


class TestWormholePatchApplyCheck(unittest.TestCase):
    def test_parse_porcelain(self) -> None:
        self.assertEqual(_parse_status_line(" M tools/file.py"), "tools/file.py")

    def test_parse_name_status_with_tab(self) -> None:
        self.assertEqual(_parse_status_line("M\ttools/file.py"), "tools/file.py")


if __name__ == "__main__":
    unittest.main()
