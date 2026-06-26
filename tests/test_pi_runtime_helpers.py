import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = REPO_ROOT / "raspberry_pi_twostage_deploy" / "run_pi_inference.py"
CONFIG_PATH = REPO_ROOT / "raspberry_pi_twostage_deploy" / "config.json"


def load_runtime_module():
    spec = importlib.util.spec_from_file_location("run_pi_inference", RUNTIME_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PiRuntimeHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = load_runtime_module()
        cls.config = cls.runtime.load_config(CONFIG_PATH)

    def test_deploy_config_has_expected_classes_and_thresholds(self):
        classes = self.config["classes"]

        self.assertEqual(len(classes), 14)
        self.assertEqual(len(classes), len(set(classes)))
        self.assertIn(self.config["reject_class"], classes)
        self.assertEqual(self.config["classifier_input_size"], 224)
        self.assertGreaterEqual(self.config["det_conf"], 0.0)
        self.assertLessEqual(self.config["det_conf"], 1.0)
        self.assertGreaterEqual(self.config["det_iou"], 0.0)
        self.assertLessEqual(self.config["det_iou"], 1.0)
        self.assertGreaterEqual(self.config["classifier_threshold"], 0.0)
        self.assertLessEqual(self.config["classifier_threshold"], 1.0)
        self.assertGreaterEqual(self.config["crop_margin"], 0.0)

    def test_choose_detector_prefers_committed_ncnn_export(self):
        detector_path = self.runtime.choose_detector(self.config)

        self.assertTrue(detector_path.exists())
        self.assertTrue(detector_path.is_dir())
        self.assertEqual(detector_path.name, "best_ncnn_model")

    def test_choose_detector_reports_missing_override(self):
        with self.assertRaises(FileNotFoundError):
            self.runtime.choose_detector(self.config, "models/missing-detector.pt")

    def test_expanded_square_crop_keeps_box_inside_image(self):
        crop = self.runtime.expanded_square_xyxy([10, 20, 30, 40], 100, 100, margin=0.5)

        self.assertEqual(crop, (0, 10, 40, 50))

    def test_expanded_square_crop_handles_image_boundary(self):
        left, top, right, bottom = self.runtime.expanded_square_xyxy([0, 0, 10, 10], 20, 20, margin=0.3)

        self.assertGreaterEqual(left, 0)
        self.assertGreaterEqual(top, 0)
        self.assertLessEqual(right, 20)
        self.assertLessEqual(bottom, 20)
        self.assertGreater(right, left)
        self.assertGreater(bottom, top)

    def test_softmax_is_stable_and_normalized(self):
        probabilities = self.runtime.softmax(np.array([1000.0, 1001.0, 999.0], dtype=np.float32))

        self.assertTrue(np.all(np.isfinite(probabilities)))
        self.assertTrue(math.isclose(float(probabilities.sum()), 1.0, rel_tol=1e-6))
        self.assertEqual(int(np.argmax(probabilities)), 1)


if __name__ == "__main__":
    unittest.main()
