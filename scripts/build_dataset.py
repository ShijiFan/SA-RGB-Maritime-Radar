#!/usr/bin/env python
"""Build SA-RGB radar-grid dataset from raw IPIX .mat files.

Features extracted:
- G3 (Tri-feature): RAA, RDPH, RVE
- G7 (Full set):    Hurst, RAA, RDPH, RVE, RI, NR, MS

Rendering:
- 100x100 pixels per macro-cell (4 sub-slots of 50x50 px)
- 1st-99th percentile normalization fitted strictly on training windows
- Standard YOLO detection format (labels with normalized bounding boxes)
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
from scipy.io import loadmat
from sargb import (
    extract_radar_features,
    fit_training_percentiles,
    quantize_features,
    render_sargb_image,
)
from sargb.slot_encoding import yolo_box_label

POLARIZATIONS = ("HH", "HV", "VH", "VV")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Directory containing IPIX .mat files")
    parser.add_argument("--output", type=Path, required=True, help="Destination directory for YOLO dataset")
    parser.add_argument("--duration", type=float, default=0.512, help="Observation duration in seconds")
    parser.add_argument("--sample-rate", type=int, default=1000, help="Radar PRF (Hz)")
    parser.add_argument("--cell-pixels", type=int, default=100, help="Pixel size of each radar cell")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)

    window = int(round(args.duration * args.sample_rate))
    mat_files = sorted(args.source.glob("*.mat"))
    if not mat_files:
        print(f"No .mat files found in {args.source}")
        return

    print(f"Found {len(mat_files)} .mat files. Processing with window={window} ({args.duration}s)...")

    # Generate data.yaml
    data_yaml = args.output / "data.yaml"
    data_yaml.write_text(
        f"path: {args.output.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: 1\n"
        f"names: ['target']\n",
        encoding="utf-8",
    )
    print(f"Created dataset configuration: {data_yaml}")


if __name__ == "__main__":
    main()
