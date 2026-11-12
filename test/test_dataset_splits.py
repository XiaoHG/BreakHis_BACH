"""Tests for dataset parsing and split construction (stdlib only)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import bach, breakhis  # noqa: E402


def make_record(patient, label, subtype, magnification=100, path=None):
    return breakhis.BreakHisRecord(
        path=path or "fake/%s-%d.png" % (patient, magnification),
        label=label,
        subtype=subtype,
        patient_id=patient,
        magnification=magnification,
    )


class TestBreakHisParsing(unittest.TestCase):
    def test_parse_filename(self):
        info = breakhis.parse_filename("SOB_B_A-14-22549AB-100-001.png")
        self.assertEqual(info["label"], "benign")
        self.assertEqual(info["subtype"], "A")
        self.assertEqual(info["patient_id"], "14-22549AB")
        self.assertEqual(info["magnification"], 100)

    def test_parse_malignant(self):
        info = breakhis.parse_filename("SOB_M_DC-14-13412-400-123.png")
        self.assertEqual(info["label"], "malignant")
        self.assertEqual(info["subtype"], "DC")
        self.assertEqual(info["patient_id"], "14-13412")
        self.assertEqual(info["magnification"], 400)

    def test_rejects_unknown_name(self):
        with self.assertRaises(ValueError):
            breakhis.parse_filename("not-a-breakhis-image.png")


class TestPatientFolds(unittest.TestCase):
    def test_no_patient_leakage(self):
        records = []
        for index in range(12):
            label = "benign" if index < 4 else "malignant"
            subtype = "A" if index < 4 else "DC"
            patient = "P%02d" % index
            for magnification in (40, 100):
                records.append(make_record(patient, label, subtype, magnification))
        rows = breakhis.assign_patient_folds(records, n_folds=5, seed=42)
        report = breakhis.validate_rows(rows)
        self.assertEqual(report["n_patients"], 12)
        self.assertEqual(report["n_images"], 24)
        self.assertEqual(report["patient_overlap"], {})

    def test_subtype_folders_of_one_slide_stay_together(self):
        records = [
            make_record("14-13412", "malignant", "DC", 100),
            make_record("14-13412", "malignant", "LC", 200),
            make_record("other", "malignant", "DC", 100),
        ]
        rows = breakhis.assign_patient_folds(records, n_folds=5, seed=42)
        folds = [row["fold"] for row in rows if row["patient_id"] == "14-13412"]
        self.assertEqual(len(folds), 2)
        self.assertEqual(len(set(folds)), 1)

    def test_scan_ignores_appledouble(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a").mkdir()
            (root / "a" / "SOB_B_A-14-22549AB-100-001.png").write_bytes(b"")
            (root / "a" / "._SOB_B_A-14-22549AB-100-001.png").write_bytes(b"")
            records = breakhis.scan(root)
            self.assertEqual(len(records), 1)


class TestBachSplit(unittest.TestCase):
    def test_eva_split_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Photos"
            for class_name in bach.CLASSES:
                class_dir = root / class_name
                class_dir.mkdir(parents=True)
                prefix = {
                    "Benign": "b",
                    "InSitu": "is",
                    "Invasive": "iv",
                    "Normal": "n",
                }[class_name]
                for index in range(1, 101):
                    (class_dir / ("%s%03d.tif" % (prefix, index))).write_bytes(b"")
            rows = bach.build_split(root)
            self.assertEqual(len(rows), 400)
            self.assertEqual(
                Counter(row["split"] for row in rows),
                {"train": 268, "val": 132},
            )
            self.assertEqual(
                Counter(row["class_name"] for row in rows),
                {name: 100 for name in bach.CLASSES},
            )


if __name__ == "__main__":
    unittest.main()
