#!/usr/bin/env python3
"""Train BreakHis baselines under patient-level folds.

Examples:
    # smoke test on one fold (no pretrained download)
    python3 cli/train.py --model resnet50 --folds 0 --epochs 1 \
        --max-batches 20 --batch-size 16 --workers 0 --no-pretrained

    # full 5-fold baseline
    python3 cli/train.py --model resnet50 --epochs 15 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.breakhis import load_split_rows, split_train_val_test  # noqa: E402
from src.data.datasets import BreakHisDataset, build_transforms  # noqa: E402
from src.engine.metrics import binary_metrics  # noqa: E402
from src.engine.train import (  # noqa: E402
    compute_class_weights,
    evaluate,
    fit,
    get_device,
    load_checkpoint,
    make_loader,
    set_seed,
)
from src.models import build_model  # noqa: E402

METRIC_KEYS = [
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "auc",
    "sensitivity",
    "specificity",
]


def parse_folds(value: str, rows) -> list:
    if value == "all":
        return sorted({int(row["fold"]) for row in rows})
    return [int(item) for item in value.split(",") if item.strip()]


def per_magnification_metrics(test_rows, y_true, y_prob) -> dict:
    result = {}
    for magnitude in sorted({int(row["magnification"]) for row in test_rows}):
        indices = [
            index
            for index, row in enumerate(test_rows)
            if int(row["magnification"]) == magnitude
        ]
        result[str(magnitude)] = binary_metrics(y_true[indices], y_prob[indices])
    return result


def summary_stats(values) -> dict:
    values = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if values.size == 0:
        return {"mean": None, "std": None}
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if values.size > 1 else 0.0}


def aggregate(fold_results: list) -> dict:
    summary = {"overall": {}, "per_magnification": {}}
    for key in METRIC_KEYS:
        summary["overall"][key] = summary_stats(
            [result["test"][key] for result in fold_results]
        )

    magnitudes = sorted(
        {
            magnitude
            for result in fold_results
            for magnitude in result["test"]["per_magnification"]
        },
        key=int,
    )
    for magnitude in magnitudes:
        summary["per_magnification"][magnitude] = {}
        for key in METRIC_KEYS:
            values = [
                result["test"]["per_magnification"][magnitude][key]
                for result in fold_results
                if magnitude in result["test"]["per_magnification"]
            ]
            summary["per_magnification"][magnitude][key] = summary_stats(values)
    return summary


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="resnet50", choices=["resnet50", "efficientnet_b0"])
    parser.add_argument("--folds", default="all", help="'all' or comma-separated fold ids")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split-path", default="Dataset/BreakHis/splits/patient_5fold.csv")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use ImageNet weights (fall back to random init if download fails)",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="limit training batches per epoch (smoke test)",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help="limit validation/test batches (smoke test)",
    )
    args = parser.parse_args()

    split_path = Path(args.split_path)
    if not split_path.is_absolute():
        split_path = PROJECT_ROOT / split_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir = out_dir / (args.run_name or args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_split_rows(split_path)
    folds = parse_folds(args.folds, rows)
    device = get_device(args.device)
    print("model=%s device=%s folds=%s" % (args.model, device, folds))
    print("split=%s (%d images)" % (split_path, len(rows)))

    fold_results = []
    pretrained_used = args.pretrained

    for fold in folds:
        seed = args.seed + fold
        set_seed(seed)
        train_rows, val_rows, test_rows = split_train_val_test(
            rows, fold, val_fraction=args.val_fraction, seed=args.seed
        )
        print(
            "fold %d: train=%d val=%d test=%d patients(train/val/test)=%d/%d/%d"
            % (
                fold,
                len(train_rows),
                len(val_rows),
                len(test_rows),
                len({row["patient_id"] for row in train_rows}),
                len({row["patient_id"] for row in val_rows}),
                len({row["patient_id"] for row in test_rows}),
            )
        )

        train_loader = make_loader(
            BreakHisDataset(train_rows, build_transforms(args.image_size, train=True)),
            args.batch_size,
            shuffle=True,
            workers=args.workers,
        )
        val_loader = make_loader(
            BreakHisDataset(val_rows, build_transforms(args.image_size, train=False)),
            args.batch_size,
            shuffle=False,
            workers=args.workers,
        )
        test_loader = make_loader(
            BreakHisDataset(test_rows, build_transforms(args.image_size, train=False)),
            args.batch_size,
            shuffle=False,
            workers=args.workers,
        )

        try:
            model = build_model(args.model, pretrained=args.pretrained)
        except Exception as exc:
            if not args.pretrained:
                raise
            print("[warn] pretrained weights unavailable (%s); using random init" % exc)
            model = build_model(args.model, pretrained=False)
            pretrained_used = False
        model = model.to(device)

        fold_dir = out_dir / ("fold%d" % fold)
        history, best_score = fit(
            model,
            train_loader,
            val_loader,
            device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            out_dir=fold_dir,
            class_weights=compute_class_weights(train_rows),
            max_batches=args.max_batches,
            max_eval_batches=args.max_eval_batches,
        )
        load_checkpoint(model, fold_dir / "best.pt", device)

        test_metrics, y_true, y_prob = evaluate(
            model,
            test_loader,
            device,
            criterion=nn.CrossEntropyLoss(),
            max_batches=args.max_eval_batches,
        )
        test_metrics["per_magnification"] = per_magnification_metrics(
            test_rows[: len(y_true)], y_true, y_prob
        )

        result = {
            "fold": fold,
            "seed": seed,
            "best_val_balanced_accuracy": best_score,
            "test": test_metrics,
            "counts": {
                "train": len(train_rows),
                "val": len(val_rows),
                "test": len(test_rows),
            },
        }
        fold_results.append(result)
        with (fold_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        print(
            "fold %d test: acc=%.4f macro_f1=%.4f bal_acc=%.4f auc=%.4f"
            % (
                fold,
                test_metrics.get("accuracy", float("nan")),
                test_metrics.get("macro_f1", float("nan")),
                test_metrics.get("balanced_accuracy", float("nan")),
                test_metrics.get("auc", float("nan")),
            )
        )

    summary = {
        "model": args.model,
        "pretrained": pretrained_used,
        "git_commit": git_commit(),
        "config": vars(args),
        "folds": fold_results,
        "metrics": aggregate(fold_results),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("\nSummary (%d folds)" % len(fold_results))
    for key in METRIC_KEYS:
        stats = summary["metrics"]["overall"][key]
        if stats["mean"] is not None:
            print("  %-20s %.4f ± %.4f" % (key, stats["mean"], stats["std"]))
    print("  saved: %s" % (out_dir / "summary.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
