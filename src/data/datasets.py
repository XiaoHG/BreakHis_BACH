"""PyTorch datasets and transforms for BreakHis binary classification."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABEL_TO_INDEX = {"benign": 0, "malignant": 1}
INDEX_TO_LABEL = {value: key for key, value in LABEL_TO_INDEX.items()}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class BreakHisDataset(Dataset):
    """BreakHis images described by rows of the patient-level split CSV."""

    def __init__(
        self,
        rows: Sequence[Dict[str, str]],
        transform: Optional[Callable] = None,
        project_root: str | Path = PROJECT_ROOT,
    ) -> None:
        self.rows = list(rows)
        self.transform = transform
        self.project_root = Path(project_root)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = Path(row["path"])
        if not path.is_absolute():
            path = self.project_root / path
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, LABEL_TO_INDEX[row["label"]]


def build_transforms(image_size: int = 224, train: bool = True):
    """ImageNet-style transforms; light augmentation for histopathology."""
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
