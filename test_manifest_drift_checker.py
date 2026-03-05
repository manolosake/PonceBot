import tempfile
import unittest
from pathlib import Path
from tools.manifest_drift_checker import snapshot_files, compare_snapshots

class TestManifestDriftChecker(unittest.TestCase):
    def test_snapshot_fields(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a.txt'; p.write_text('hello', encoding='utf-8')
            s=snapshot_files(Path(td), ['a.txt'])['a.txt']
            self.assertTrue(s['exists'])
            self.assertEqual(s['size_bytes'], 5)
            self.assertEqual(len(s['sha256']), 64)
    def test_detects_drift(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a.txt'; p.write_text('v1', encoding='utf-8')
            t0=snapshot_files(Path(td), ['a.txt'])
            p.write_text('v2x', encoding='utf-8')
            tn=snapshot_files(Path(td), ['a.txt'])
            d=compare_snapshots(t0, tn)
            self.assertTrue(d)
            self.assertIn('sha256', {x['field'] for x in d})
    def test_no_drift(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a.txt'; p.write_text('stable', encoding='utf-8')
            t0=snapshot_files(Path(td), ['a.txt']); tn=snapshot_files(Path(td), ['a.txt'])
            self.assertEqual(compare_snapshots(t0, tn), [])

if __name__=='__main__':
    unittest.main()
