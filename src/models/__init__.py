"""Baseline CNN models for BreakHis binary classification."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_model(name: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Build a torchvision baseline with a task-specific classifier head."""
    name = name.lower()
    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name in {"efficientnet_b0", "efficientnet-b0"}:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError("unknown model: %s" % name)
