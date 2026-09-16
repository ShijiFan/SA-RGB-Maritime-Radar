#!/usr/bin/env python
"""Train SA-RGB, RadarGrid-Lite, SAFR, or baseline detectors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sargb import register_with_ultralytics
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to dataset data.yaml")
    parser.add_argument(
        "--cfg",
        type=str,
        default=str(ROOT / "configs" / "radargrid_lite.yaml"),
        help="Path to model architecture YAML or weights .pt",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", type=Path, default=ROOT / "runs")
    parser.add_argument("--name", type=str, default="exp")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr0", type=float, default=0.0002)
    parser.add_argument("--lrf", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--patience", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_with_ultralytics()

    print("=" * 60)
    print(f"Initializing model from: {args.cfg}")
    print(f"Dataset config:           {args.data}")
    print(f"Training parameters:      {args.epochs} epochs, batch {args.batch}, seed {args.seed}")
    print("=" * 60)

    model = YOLO(args.cfg)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        seed=args.seed,
        workers=args.workers,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=True,
        plots=True,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=True,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        # Minimal spatial augmentations to preserve radar cell grid alignment
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.02,
        scale=0.05,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        erasing=0.0,
        close_mosaic=0,
    )


if __name__ == "__main__":
    main()
