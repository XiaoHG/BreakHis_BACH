"""BACH (ICIAR 2018) metadata and patient-disjoint split construction.

The official patient sheet shipped with BACH is only partially filled in
(145 of 400 images), so patient identities cannot be reconstructed reliably
from it. Instead we use the index ranges hard-coded by the EVA project, which
documents them as patient-disjoint:

https://github.com/kaiko-ai/eva/blob/main/src/eva/vision/data/datasets/classification/bach.py

The split is 268 train / 132 validation over the 400 labelled microscopy
images. Cross-checking against the official patient sheet shows no patient
appearing in both sides.

Note: the TIFF files have mixed bit depth (381 are 16-bit, 19 are 8-bit, all
in the training side). Normalise each image according to its own bit depth
rather than assuming a single dtype.
"""

from __future__ import annotations

import csv
import struct
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

CLASSES = ("Benign", "InSitu", "Invasive", "Normal")
BINARY_LABEL = {
    "Normal": "benign",
    "Benign": "benign",
    "InSitu": "malignant",
    "Invasive": "malignant",
}

# EVA patient-disjoint ranges over the flattened, class-sorted 0..399 index.
EVA_TRAIN_RANGES = ((0, 41), (59, 60), (90, 139), (169, 240), (258, 260),
                    (273, 345), (368, 400))
EVA_VAL_RANGES = ((41, 59), (60, 90), (139, 169), (240, 258), (260, 273),
                  (345, 368))

SPLIT_FIELDS = ["path", "filename", "class_name", "label", "split"]


def _indices(ranges: Sequence[Tuple[int, int]]) -> set:
    return {index for start, stop in ranges for index in range(start, stop)}


def scan_photos(root: str | Path, path_base: str | Path | None = None) -> List[Tuple[Path, str]]:
    """Return (path, class) in torchvision folder order: classes then filenames."""
    root = Path(root)
    path_base = Path(path_base) if path_base is not None else root
    samples: List[Tuple[Path, str]] = []
    for class_name in CLASSES:
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise ValueError("missing BACH class directory: %s" % class_dir)
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() not in (".tif", ".tiff"):
                continue
            if path.name.startswith("._"):
                continue
            try:
                relative = path.relative_to(path_base)
            except ValueError:
                relative = path
            samples.append((relative, class_name))
    if len(samples) != 400:
        raise ValueError("expected 400 labelled BACH images, found %d" % len(samples))
    return samples


def build_split(root: str | Path, path_base: str | Path | None = None) -> List[Dict[str, str]]:
    """Build the 268/132 EVA patient-disjoint split."""
    samples = scan_photos(root, path_base=path_base)
    train_indices = _indices(EVA_TRAIN_RANGES)
    val_indices = _indices(EVA_VAL_RANGES)
    if train_indices & val_indices:
        raise ValueError("EVA train/validation ranges overlap")

    rows: List[Dict[str, str]] = []
    for index, (path, class_name) in enumerate(samples):
        if index in train_indices:
            split = "train"
        elif index in val_indices:
            split = "val"
        else:
            raise ValueError("image index %d is not covered by EVA ranges" % index)
        rows.append(
            {
                "path": path.as_posix(),
                "filename": path.name,
                "class_name": class_name,
                "label": BINARY_LABEL[class_name],
                "split": split,
            }
        )

    counts = Counter(row["split"] for row in rows)
    if counts != {"train": 268, "val": 132}:
        raise ValueError("unexpected EVA split counts: %s" % dict(counts))
    return rows


def write_split(rows: Sequence[Dict[str, str]], out_dir: str | Path) -> Dict[str, object]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "eva_patient_split.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SPLIT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    per_split = {}
    for split in ("train", "val"):
        subset = [row for row in rows if row["split"] == split]
        per_split[split] = {
            "n_images": len(subset),
            "class_counts": dict(Counter(row["class_name"] for row in subset)),
            "label_counts": dict(Counter(row["label"] for row in subset)),
        }
    return {
        "protocol": "EVA patient-disjoint index ranges (268/132)",
        "n_images": len(rows),
        "splits": per_split,
    }


def tiff_header(path: str | Path) -> Dict[str, int]:
    """Read width, height, samples/pixel and bits/sample from a TIFF IFD.

    Supports the uncompressed, little/big-endian baseline TIFFs used by BACH.
    """
    path = Path(path)
    with path.open("rb") as handle:
        header = handle.read(8)
        if len(header) < 8:
            raise ValueError("truncated TIFF: %s" % path)
        if header[:2] == b"II":
            endian = "<"
        elif header[:2] == b"MM":
            endian = ">"
        else:
            raise ValueError("not a TIFF file: %s" % path)
        if struct.unpack(endian + "H", header[2:4])[0] != 42:
            raise ValueError("unsupported TIFF magic: %s" % path)

        ifd_offset = struct.unpack(endian + "I", header[4:8])[0]
        handle.seek(ifd_offset)
        entry_count = struct.unpack(endian + "H", handle.read(2))[0]

        tags = {}
        for _ in range(entry_count):
            entry = handle.read(12)
            tag, value_type, count = struct.unpack(endian + "HHI", entry[:8])
            size = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8}.get(value_type, 1) * count
            if size <= 4:
                data = entry[8:8 + size]
            else:
                offset = struct.unpack(endian + "I", entry[8:12])[0]
                position = handle.tell()
                handle.seek(offset)
                data = handle.read(size)
                handle.seek(position)
            if value_type == 3:
                values = struct.unpack(endian + "H" * count, data)
            elif value_type == 4:
                values = struct.unpack(endian + "I" * count, data)
            else:
                values = tuple(data)
            tags[tag] = values

    def first(tag: int) -> int:
        values = tags.get(tag)
        if not values:
            raise ValueError("TIFF tag %d missing: %s" % (tag, path))
        return int(values[0])

    return {
        "width": first(256),
        "height": first(257),
        "bits_per_sample": first(258),
        "samples_per_pixel": first(277),
    }
