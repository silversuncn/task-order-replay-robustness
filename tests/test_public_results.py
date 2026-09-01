import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from verify_public_results import build_report


class PublicResultsTest(unittest.TestCase):
    def test_public_results_pass(self):
        report = build_report()
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["formal_results_v2_reference_rows"], 200)
        self.assertEqual(report["results_120perm_4methods_rows"], 4800)
        self.assertEqual(report["results_compute_matched_rows"], 1200)
        self.assertEqual(report["buffer_composition_rows"], 18000)
        self.assertEqual(report["legacy_comparison_rows"], 200)
        self.assertEqual(report["method_summary"]["er500"]["n_rows"], 1200)
        self.assertEqual(report["method_summary"]["er100"]["n_rows"], 1200)
        self.assertEqual(report["method_summary"]["ewc"]["n_rows"], 1200)
        self.assertEqual(report["method_summary"]["no_replay"]["n_rows"], 1200)
        self.assertEqual(report["method_summary"]["er500"]["faa_mean"], 0.610559)
        self.assertEqual(report["method_summary"]["er100"]["faa_mean"], 0.338505)
        self.assertEqual(report["method_summary"]["ewc"]["faa_mean"], 0.19767)
        self.assertEqual(report["method_summary"]["no_replay"]["faa_mean"], 0.197288)
        self.assertEqual(report["compute_matched"]["FAA_mean"], 0.582897)
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
