import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


class ExperimentTrackingDocsTests(unittest.TestCase):
    def test_experiment_tracking_files_exist(self):
        expected_files = [
            "README.md",
            "training_config.json",
            "latest_evaluation_manifest.json",
            "metrics_summary.csv",
        ]

        for name in expected_files:
            with self.subTest(name=name):
                self.assertTrue((EXPERIMENTS_DIR / name).exists())

    def test_training_config_documents_seed_and_split_policy(self):
        config = json.loads((EXPERIMENTS_DIR / "training_config.json").read_text(encoding="utf-8"))

        self.assertEqual(config["defaults"]["seed"], 42)
        self.assertEqual(config["dataset_split"]["strategy"], "grouped_stratified")
        self.assertFalse(config["dataset_split"]["split_manifest_committed"])
        self.assertIn("training/road_sign_twostage_dataset/reports/splits.json", config["dataset_split"]["generated_manifest"])

    def test_latest_evaluation_manifest_matches_committed_metrics(self):
        manifest = json.loads((EXPERIMENTS_DIR / "latest_evaluation_manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads((REPO_ROOT / "evaluation" / "latest" / "metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["source_metrics_file"], "evaluation/latest/metrics.json")
        self.assertEqual(manifest["metrics"]["test_images"], metrics["test_images"])
        self.assertEqual(manifest["metrics"]["tp"], metrics["tp"])
        self.assertEqual(manifest["metrics"]["fp"], metrics["fp"])
        self.assertEqual(manifest["metrics"]["fn"], metrics["fn"])
        self.assertEqual(manifest["metrics"]["end_to_end_f1"], metrics["end_to_end_f1"])


if __name__ == "__main__":
    unittest.main()
