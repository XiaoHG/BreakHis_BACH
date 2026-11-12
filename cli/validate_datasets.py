#!/usr/bin/env python3
"""Validate downloaded datasets and generated splits.

Usage (from the project root):

    python3 cli/validate_datasets.py

Checks raw image counts, class distribution, dimensions/bit depth, patient
grouping and split coverage. Exits non-zero if any hard requirement fails.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import bach, breakhis  # noqa: E402


class Checker:
    def __init__(self) -> None:
        self.errors: list = []
        self.warnings: list = []

    def check(self, condition: bool, message: str) -> None:
        if condition:
            print("  [OK]   %s" % message)
        else:
            print("  [FAIL] %s" % message)
            self.errors.append(message)

    def warn(self, condition: bool, message: str) -> None:
        if not condition:
            print("  [WARN] %s" % message)
            self.warnings.append(message)


def read_csv(path: Path) -> list:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_breakhis(dataset_root: Path, checker: Checker) -> list:
    print("\nBreakHis raw data")
    image_root = dataset_root / "BreakHis" / "BreaKHis_v1"
    records = breakhis.scan(image_root, path_base=PROJECT_ROOT)
    checker.check(len(records) == 7909, "image count = 7,909 (found %d)" % len(records))

    class_counts = Counter(record.label for record in records)
    checker.check(
        class_counts == {"benign": 2480, "malignant": 5429},
        "class counts 2,480 benign / 5,429 malignant (found %s)"
        % dict(class_counts),
    )

    patients = len({record.patient_id for record in records})
    checker.check(patients == 81, "unique slide IDs = 81 (found %d)" % patients)
    checker.check(
        len({record.subtype for record in records}) == 8,
        "8 histological subtypes present",
    )
    checker.check(
        {record.magnification for record in records} == {40, 100, 200, 400},
        "4 magnifications present",
    )

    sizes = Counter()
    bit_depths = Counter()
    color_types = Counter()
    for record in records:
        header = breakhis.png_header(PROJECT_ROOT / record.path)
        sizes[(header["width"], header["height"])] += 1
        bit_depths[header["bit_depth"]] += 1
        color_types[header["color_type"]] += 1

    checker.check(set(sizes) <= {(700, 460), (700, 456)}, "PNG dimensions 700x460/456")
    checker.warn(
        sizes.get((700, 456), 0) == 0,
        "%d images are 700x456 (all papillary carcinoma, slide 14-12465); "
        "resize consistently" % sizes.get((700, 456), 0),
    )
    checker.check(bit_depths == {8: len(records)}, "all PNG images are 8-bit")
    checker.check(color_types == {2: len(records)}, "all PNG images are RGB")
    return records


def validate_bach(dataset_root: Path, checker: Checker) -> list:
    print("\nBACH raw data")
    photo_root = (
        dataset_root
        / "BACH"
        / "ICIAR2018_BACH_Challenge"
        / "ICIAR2018_BACH_Challenge"
        / "Photos"
    )
    rows = bach.build_split(photo_root, path_base=PROJECT_ROOT)
    checker.check(len(rows) == 400, "labelled image count = 400 (found %d)" % len(rows))

    class_counts = Counter(row["class_name"] for row in rows)
    checker.check(
        class_counts == {name: 100 for name in bach.CLASSES},
        "4 classes x 100 images (found %s)" % dict(class_counts),
    )
    checker.check(
        {row["split"] for row in rows} == {"train", "val"},
        "EVA split provides train/val",
    )
    split_counts = Counter(row["split"] for row in rows)
    checker.check(split_counts == {"train": 268, "val": 132}, "split 268/132")

    sizes = Counter()
    bits = Counter()
    samples = Counter()
    for row in rows:
        header = bach.tiff_header(PROJECT_ROOT / row["path"])
        sizes[(header["width"], header["height"])] += 1
        bits[header["bits_per_sample"]] += 1
        samples[header["samples_per_pixel"]] += 1

    checker.check(sizes == {(2048, 1536): 400}, "all images 2048x1536")
    checker.check(
        set(bits) <= {8, 16},
        "TIFF bit depth is 8 or 16 (found %s)" % dict(bits),
    )
    checker.warn(
        len(bits) == 1,
        "mixed TIFF bit depth %s; normalise per image before training"
        % dict(bits),
    )
    checker.check(samples == {3: 400}, "all TIFF images have 3 channels")
    return rows


def validate_breakhis_split(dataset_root: Path, records: list, checker: Checker) -> None:
    print("\nBreakHis generated split")
    split_path = dataset_root / "BreakHis" / "splits" / "patient_5fold.csv"
    if not split_path.exists():
        checker.check(False, "missing %s (run cli/build_dataset_splits.py)" % split_path)
        return

    rows = read_csv(split_path)
    checker.check(len(rows) == len(records), "split covers every image")
    checker.check(
        {row["path"] for row in rows} == {record.path for record in records},
        "split paths match scanned images",
    )
    try:
        report = breakhis.validate_rows(rows)
        checker.check(True, "no patient leakage across folds")
        print(
            "         %d images / %d patients, fold sizes %s"
            % (report["n_images"], report["n_patients"], report["fold_sizes"])
        )
    except ValueError as exc:
        checker.check(False, str(exc))


def validate_bach_split(dataset_root: Path, checker: Checker) -> None:
    print("\nBACH generated split")
    split_path = dataset_root / "BACH" / "splits" / "eva_patient_split.csv"
    if not split_path.exists():
        checker.check(False, "missing %s (run cli/build_dataset_splits.py)" % split_path)
        return
    rows = read_csv(split_path)
    checker.check(len(rows) == 400, "split covers 400 images")
    checker.check(
        Counter(row["split"] for row in rows) == {"train": 268, "val": 132},
        "split counts 268/132",
    )
    checker.check(
        all((PROJECT_ROOT / row["path"]).is_file() for row in rows),
        "all split paths exist on disk",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "Dataset",
        help="dataset root directory (default: <project>/Dataset)",
    )
    args = parser.parse_args()

    checker = Checker()
    records = validate_breakhis(args.dataset_root, checker)
    validate_bach(args.dataset_root, checker)
    validate_breakhis_split(args.dataset_root, records, checker)
    validate_bach_split(args.dataset_root, checker)

    print("\nSummary: %d error(s), %d warning(s)" % (
        len(checker.errors), len(checker.warnings)
    ))
    if checker.errors:
        for error in checker.errors:
            print("  ERROR: %s" % error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
