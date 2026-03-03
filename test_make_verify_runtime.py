import shutil
import subprocess
import unittest


class TestMakeVerifyRuntime(unittest.TestCase):
    def test_makefile_prefers_python3_with_python_fallback(self) -> None:
        expected = shutil.which("python3") or shutil.which("python")
        self.assertIsNotNone(expected, "A python runtime is required for this test environment")

        out = subprocess.check_output(["make", "-n", "lint"], text=True)
        compile_lines = [line for line in out.splitlines() if " -m py_compile " in line]
        self.assertTrue(compile_lines, msg=f"Could not find py_compile command in output: {out}")
        first_line = compile_lines[0].strip()
        self.assertTrue(
            first_line.startswith(expected),
            msg=f"Expected make lint to use runtime {expected}, got: {first_line}",
        )


if __name__ == "__main__":
    unittest.main()
