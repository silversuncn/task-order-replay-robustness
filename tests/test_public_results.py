import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from verify_public_results import build_report


class PublicResultsTest(unittest.TestCase):
    def test_public_results_pass(self):
        report = build_report()
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["formal_metrics_rows"], 200)
        self.assertEqual(len(report["seeds"]), 10)
        self.assertEqual(len(report["orders"]), 5)
        self.assertEqual(report["best_method"], "experience_replay")
        self.assertEqual(report["best_memory_budget"], 500)


if __name__ == "__main__":
    unittest.main()
