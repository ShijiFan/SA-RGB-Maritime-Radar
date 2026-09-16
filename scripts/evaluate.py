#!/usr/bin/env python
"""Evaluate trained detector checkpoints on test splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sargb import register_with_ultralytics
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--data", type=Path, required=True, help="Path to dataset data.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_with_ultralytics()

    model = YOLO(str(args.weights))
    results = model.val(
        data=str(args.data.resolve()),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots=False,
        verbose=False,
    )

    metrics = {
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
    }

    print("\n" + "=" * 45)
    print(f"Checkpoint: {args.weights.name}")
    print(f"Split:      {args.split}")
    print(f"Precision:  {metrics['precision'] * 100:.2f}%")
    print(f"Recall:     {metrics['recall'] * 100:.2f}%")
    print(f"mAP50:      {metrics['mAP50'] * 100:.2f}%")
    print(f"mAP50-95:   {metrics['mAP50_95'] * 100:.2f}%")
    print("=" * 45)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Saved metrics to: {args.output}")


if __name__ == "__main__":
    main()
