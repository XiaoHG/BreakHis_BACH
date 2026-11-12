"""BreakHis metadata parsing and patient-level split construction.

The project uses a deterministic, leakage-free patient-level 5-fold protocol.
The patient identifier is the slide ID parsed from the filename, so the two
subtype folders of slide ``14-13412`` (DC and LC) are kept in the same fold.
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

FILENAME_RE = re.compile(
    r"^[A-Z]+_(?P<bm>[BM])_(?P<subtype>[A-Z]+)-"
    r"(?P<patient>\d+-\d+[A-Z]*)-(?P<mag>40|100|200|400)-(?P<seq>\d+)\.png$"
)

LABELS = {"B": "benign", "M": "malignant"}
SPLIT_FIELDS = ["path", "label", "subtype", "patient_id", "magnification", "fold"]
METADATA_FIELDS = ["path", "label", "subtype", "patient_id", "magnification"]


@dataclass(frozen=True)
class BreakHisRecord:
    path: str
    label: str
    subtype: str
    patient_id: str
    magnification: int


def parse_filename(name: str) -> Dict[str, object]:
    """Parse a BreakHis filename into label, subtype, patient and magnification."""
    match = FILENAME_RE.match(name)
    if match is None:
        raise ValueError("unrecognized BreakHis filename: %s" % name)
    return {
        "label": LABELS[match.group("bm")],
        "subtype": match.group("subtype"),
        "patient_id": match.group("patient"),
        "magnification": int(match.group("mag")),
    }


def scan(root: str | Path, path_base: str | Path | None = None) -> List[BreakHisRecord]:
    """Scan a BreaKHis_v1 tree, skipping macOS ``._*`` metadata files."""
    root = Path(root)
    path_base = Path(path_base) if path_base is not None else root
    records: List[BreakHisRecord] = []
    for path in sorted(root.rglob("*.png")):
        if path.name.startswith("._"):
            continue
        info = parse_filename(path.name)
        try:
            relative = path.relative_to(path_base)
        except ValueError:
            relative = path
        records.append(BreakHisRecord(path=relative.as_posix(), **info))
    if not records:
        raise ValueError("no BreakHis PNG images found under %s" % root)
    return records


def assign_patient_folds(
    records: Sequence[BreakHisRecord], n_folds: int = 5, seed: int = 42
) -> List[Dict[str, object]]:
    """Assign whole patients to folds, stratified by class and subtype.

    Patients are processed class by class, largest first, and each is placed
    in the fold that minimises a weighted cost over (subtype load, class load,
    total load). The subtype term keeps rare subtypes spread across folds;
    the class term keeps the benign/malignant ratio balanced. The result is
    deterministic for a given seed and guarantees that every patient appears
    in exactly one fold.

    With 81 slide-level patients and four benign subtypes, a small number of
    subtype x fold cells cannot be covered (e.g. adenosis has only 4 patients
    for 5 folds). This is inherent to patient-level 5-fold CV and is reported
    rather than hidden.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")

    by_patient: Dict[str, List[BreakHisRecord]] = defaultdict(list)
    for record in records:
        by_patient[record.patient_id].append(record)

    labels_per_patient = {
        patient_id: {record.label for record in group}
        for patient_id, group in by_patient.items()
    }
    mixed = sorted(pid for pid, labels in labels_per_patient.items() if len(labels) != 1)
    if mixed:
        raise ValueError("patients with mixed labels: %s" % ", ".join(mixed))

    subtypes_per_patient = {
        patient_id: {record.subtype for record in group}
        for patient_id, group in by_patient.items()
    }

    rng = random.Random(seed)
    subtype_load = [Counter() for _ in range(n_folds)]
    class_load = [Counter() for _ in range(n_folds)]
    totals = [0] * n_folds
    patient_fold: Dict[str, int] = {}

    for label in sorted({record.label for record in records}):
        patient_ids = [
            pid for pid, group in by_patient.items() if group[0].label == label
        ]
        rng.shuffle(patient_ids)
        patient_ids.sort(key=lambda pid: -len(by_patient[pid]))
        for pid in patient_ids:
            size = len(by_patient[pid])
            subtypes = subtypes_per_patient[pid]

            def cost(fold: int) -> tuple:
                subtype_cost = sum(subtype_load[fold][s] for s in subtypes)
                return (
                    3 * subtype_cost + 10 * class_load[fold][label] + totals[fold],
                    fold,
                )

            fold = min(range(n_folds), key=cost)
            patient_fold[pid] = fold
            class_load[fold][label] += size
            totals[fold] += size
            for subtype in subtypes:
                subtype_load[fold][subtype] += size

    return [
        dict(asdict(record), fold=patient_fold[record.patient_id])
        for record in records
    ]


def validate_rows(rows: Iterable[Dict[str, object]]) -> Dict[str, object]:
    """Validate coverage and patient isolation, returning a summary report."""
    rows = list(rows)
    if not rows:
        raise ValueError("no split rows to validate")

    fold_patients: Dict[int, set] = defaultdict(set)
    fold_counts: Counter = Counter()
    for row in rows:
        fold = int(row["fold"])
        fold_patients[fold].add(str(row["patient_id"]))
        fold_counts[fold] += 1

    overlaps = {}
    folds = sorted(fold_patients)
    for i, left in enumerate(folds):
        for right in folds[i + 1:]:
            shared = fold_patients[left] & fold_patients[right]
            if shared:
                overlaps["%s-%s" % (left, right)] = sorted(shared)

    patient_folds: Dict[str, set] = defaultdict(set)
    for row in rows:
        patient_folds[str(row["patient_id"])].add(int(row["fold"]))
    multi_fold = sorted(pid for pid, fs in patient_folds.items() if len(fs) > 1)

    if overlaps or multi_fold:
        raise ValueError(
            "patient leakage detected: overlaps=%s multi_fold=%s"
            % (overlaps, multi_fold)
        )

    return {
        "n_images": len(rows),
        "n_patients": len(patient_folds),
        "fold_sizes": {str(k): fold_counts[k] for k in folds},
        "fold_patients": {str(k): len(fold_patients[k]) for k in folds},
        "patient_overlap": overlaps,
    }


def split_report(
    rows: Sequence[Dict[str, object]], n_folds: int = 5, seed: int = 42
) -> Dict[str, object]:
    """Build a JSON-serialisable report with per-fold class/magnification counts."""
    report: Dict[str, object] = {
        "protocol": (
            "patient-level %d-fold split; patient identifier = slide ID; "
            "folds are disjoint" % n_folds
        ),
        "seed": seed,
        "n_images": len(rows),
        "n_patients": len({row["patient_id"] for row in rows}),
        "folds": {},
    }
    for fold in sorted({int(row["fold"]) for row in rows}):
        subset = [row for row in rows if int(row["fold"]) == fold]
        report["folds"][str(fold)] = {
            "n_images": len(subset),
            "n_patients": len({row["patient_id"] for row in subset}),
            "class_counts": dict(Counter(row["label"] for row in subset)),
            "magnification_counts": {
                str(k): v
                for k, v in sorted(
                    Counter(int(row["magnification"]) for row in subset).items()
                )
            },
            "subtype_counts": dict(
                sorted(Counter(str(row["subtype"]) for row in subset).items())
            ),
        }
    return report


def write_outputs(
    rows: Sequence[Dict[str, object]],
    out_dir: str | Path,
    n_folds: int = 5,
    seed: int = 42,
) -> Dict[str, object]:
    """Write metadata, split CSV and report; return the report."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(rows)
    validate_rows(rows)
    report = split_report(rows, n_folds=n_folds, seed=seed)

    with (out_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in METADATA_FIELDS} for row in rows
        )

    with (out_dir / "patient_5fold.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=SPLIT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with (out_dir / "split_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    return report


def png_header(path: str | Path) -> Dict[str, int]:
    """Return width, height, bit depth and colour type from a PNG IHDR chunk."""
    import struct

    with Path(path).open("rb") as handle:
        header = handle.read(26)
    if len(header) < 26 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG file: %s" % path)
    width, height = struct.unpack(">II", header[16:24])
    return {
        "width": width,
        "height": height,
        "bit_depth": header[24],
        "color_type": header[25],
    }
