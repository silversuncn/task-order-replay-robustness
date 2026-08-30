import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from verify_public_results import build_report


class PublicResultsTest(unittest.TestCase):
    def test_public_results_pass(self):
        report = build_report()
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["formal_results_v2_rows"], 200)
        self.assertEqual(report["bwt_task_level_v2_rows"], 1000)
        self.assertEqual(report["extended_budget_v2_rows"], 300)
        self.assertEqual(report["ewc_lambda_search_rows"], 35)
        self.assertEqual(report["selected_epochs"], 10)
        self.assertEqual(report["selected_batch_size"], 128)
        self.assertEqual(report["selected_ewc_lambda"], 100.0)
        self.assertEqual(len(report["seeds"]), 10)
        self.assertEqual(len(report["orders"]), 5)
        self.assertEqual(report["method_means"]["er500"], 0.591408)
        self.assertEqual(report["method_means"]["er100"], 0.318216)
        self.assertEqual(report["method_means"]["ewc"], 0.196428)
        self.assertEqual(report["method_means"]["no_replay"], 0.195824)
        self.assertEqual(
            report["figures"],
            [
                "fig1_cell_faa_v2.png",
                "fig2_faa_boxplot_v2.png",
                "fig3_order_heatmap_v2.png",
            ],
        )


if __name__ == "__main__":
    unittest.main()
