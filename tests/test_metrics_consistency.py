import csv
import json
import math
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = REPO_ROOT / "evaluation" / "latest" / "metrics.json"
PER_CLASS_PATH = REPO_ROOT / "evaluation" / "latest" / "per_class_metrics.csv"
CONFIG_PATH = REPO_ROOT / "raspberry_pi_twostage_deploy" / "config.json"


class MetricsConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_precision_recall_and_f1_match_confusion_counts(self):
        tp = self.metrics["tp"]
        fp = self.metrics["fp"]
        fn = self.metrics["fn"]

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

        self.assertTrue(math.isclose(self.metrics["end_to_end_precision"], precision, rel_tol=1e-12))
        self.assertTrue(math.isclose(self.metrics["end_to_end_recall"], recall, rel_tol=1e-12))
        self.assertTrue(math.isclose(self.metrics["end_to_end_f1"], f1, rel_tol=1e-12))

    def test_macro_f1_is_average_of_committed_per_class_f1_values(self):
        per_class = self.metrics["per_class"]
        macro_f1 = sum(row["f1"] for row in per_class.values()) / len(per_class)

        self.assertTrue(math.isclose(self.metrics["macro_f1"], macro_f1, rel_tol=1e-12))

    def test_other_sign_reject_rate_matches_false_accept_rate(self):
        false_accept_rate = self.metrics["other_sign_false_accept_rate"]

        self.assertTrue(
            math.isclose(
                self.metrics["other_sign_reject_rate"],
                1.0 - false_accept_rate,
                rel_tol=1e-12,
            )
        )

    def test_csv_per_class_file_matches_metrics_json_classes(self):
        with PER_CLASS_PATH.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        csv_classes = {row["class"] for row in rows}
        json_classes = set(self.metrics["per_class"])

        self.assertEqual(csv_classes, json_classes)

    def test_deploy_thresholds_match_latest_evaluation_thresholds(self):
        self.assertEqual(self.config["det_conf"], 0.15)
        self.assertEqual(self.config["det_iou"], 0.70)
        self.assertEqual(self.config["classifier_threshold"], 0.88)
        self.assertEqual(self.config["crop_margin"], 0.30)


if __name__ == "__main__":
    unittest.main()
