#!/usr/bin/env python3
"""Build leakage-free split files for the downloaded BreakHis and BACH data.

Usage (from the project root):

    python3 cli/build_dataset_splits.py

Outputs:
    Dataset/BreakHis/splits/metadata.csv
    Dataset/BreakHis/splits/patient_5fold.csv
    Dataset/BreakHis/splits/split_report.json
    Dataset/BACH/splits/eva_patient_split.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import bach, breakhis  # noqa: E402


def build_breakhis(dataset_root: Path, n_folds: int, seed: int) -> dict:
    data_root = dataset_root / "BreakHis"
    image_root = data_root / "BreaKHis_v1"
    out_dir = data_root / "splits"

    records = breakhis.scan(image_root, path_base=PROJECT_ROOT)
    rows = breakhis.assign_patient_folds(records, n_folds=n_folds, seed=seed)
    report = breakhis.write_outputs(rows, out_dir, n_folds=n_folds, seed=seed)

    print("  BreakHis: patient-level %d-fold split (slide ID grouping)" % n_folds)
    print(
        "  BreakHis: %d images, %d patients -> %s"
        % (report["n_images"], report["n_patients"], out_dir)
    )
    for fold, info in report["folds"].items():
        print(
            "    fold %s: %d images / %d patients"
            % (fold, info["n_images"], info["n_patients"])
        )
    return report


def build_bach(dataset_root: Path) -> dict:
    data_root = dataset_root / "BACH"
    photo_root = (
        data_root / "ICIAR2018_BACH_Challenge" / "ICIAR2018_BACH_Challenge" / "Photos"
    )
    out_dir = data_root / "splits"

    rows = bach.build_split(photo_root, path_base=PROJECT_ROOT)
    report = bach.write_split(rows, out_dir)

    print("  BACH: %d images -> %s" % (report["n_images"], out_dir))
    for split, info in report["splits"].items():
        print(
            "    %s: %d images %s"
            % (split, info["n_images"], info["class_counts"])
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "Dataset",
        help="dataset root directory (default: <project>/Dataset)",
    )
    parser.add_argument("--folds", type=int, default=5, help="BreakHis folds")
    parser.add_argument("--seed", type=int, default=42, help="BreakHis split seed")
    args = parser.parse_args()

    print("Building dataset splits under %s" % args.dataset_root)
    build_breakhis(args.dataset_root, args.folds, args.seed)
    build_bach(args.dataset_root)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
