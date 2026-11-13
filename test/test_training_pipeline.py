"""Tests for the patient-level training split and metrics."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.breakhis import split_train_val_test  # noqa: E402
from src.engine.metrics import binary_metrics  # noqa: E402


def make_row(patient: str, label: str, fold: int, magnification: int = 100) -> dict:
    return {
        "path": "fake/%s.png" % patient,
        "label": label,
        "subtype": "A" if label == "benign" else "DC",
        "patient_id": patient,
        "magnification": str(magnification),
        "fold": str(fold),
    }


class TestPatientSplit(unittest.TestCase):
    def test_no_patient_overlap_and_full_coverage(self):
        rows = [
            make_row("P%02d" % index, "benign" if index < 10 else "malignant", index % 5)
            for index in range(20)
        ]
        train, val, test = split_train_val_test(rows, test_fold=0, val_fraction=0.2, seed=42)
        train_patients = {row["patient_id"] for row in train}
        val_patients = {row["patient_id"] for row in val}
        test_patients = {row["patient_id"] for row in test}

        self.assertFalse(train_patients & val_patients)
        self.assertFalse(train_patients & test_patients)
        self.assertFalse(val_patients & test_patients)
        self.assertEqual(train_patients | val_patients | test_patients, {row["patient_id"] for row in rows})
        self.assertTrue(all(row["fold"] == "0" for row in test))


class TestBinaryMetrics(unittest.TestCase):
    def test_perfect_predictions(self):
        metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["sensitivity"], 1.0)
        self.assertEqual(metrics["specificity"], 1.0)
        self.assertEqual(metrics["auc"], 1.0)

    def test_single_class_auc_is_nan(self):
        metrics = binary_metrics([0, 0], [0.1, 0.2])
        self.assertTrue(math.isnan(metrics["auc"]))
        self.assertEqual(metrics["balanced_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
