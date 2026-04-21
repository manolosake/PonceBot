import unittest

from tools import release_governance as rg


class TestReleaseGovernanceTraceability(unittest.TestCase):
    def test_traceability_mismatch_seed_vs_reseed_not_counted(self) -> None:
        log_text = (
            "abc1234 order:019abfcf key:proactive_cli_seed_r1\n"
            "def5678 order:019abfcf key:proactive_cli_seed_r1_2\n"
        )
        count = rg._traceability_count_from_log(
            log_text,
            order_token="order:019abfcf",
            key_token="key:proactive_cli_reseed_r1",
        )
        self.assertEqual(count, 0)

    def test_traceability_cross_order_false_positive_not_counted(self) -> None:
        log_text = (
            "1111111 order:deadbeef key:proactive_cli_reseed_r1\n"
            "2222222 order:019abfcf key:proactive_cli_reseed_r1\n"
        )
        count = rg._traceability_count_from_log(
            log_text,
            order_token="order:deadbeef",
            key_token="key:proactive_cli_reseed_r1",
        )
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
