#!/usr/bin/env python
"""Standardized model-only CUDA inference latency and throughput benchmarking.

Measures:
- Parameter count
- Mean, median, and P95 latency (ms/frame) via torch.cuda.Event
- Throughput (images/sec)
- Peak CUDA memory allocation (MiB)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

# Add package root to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from sargb import register_with_ultralytics
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Path to .pt weights or model .yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - pos) + ordered[high] * (pos - low)


def main() -> None:
    args = parse_args()
    register_with_ultralytics()

    detector = YOLO(str(args.model))
    model = detector.model.eval().to(args.device)
    if args.half:
        model = model.half()

    dtype = torch.float16 if args.half else torch.float32
    # Standard input tensor: (Batch, 3, H, W)
    image = torch.zeros((args.batch, 3, args.imgsz, args.imgsz), device=args.device, dtype=dtype)

    print(f"Warmup ({args.warmup} iterations)...", flush=True)
    torch.cuda.empty_cache()
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(image)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    print(f"Benchmarking ({args.iters} iterations)...", flush=True)
    elapsed_ms: list[float] = []
    with torch.inference_mode():
        for _ in range(args.iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(image)
            end.record()
            end.synchronize()
            elapsed_ms.append(float(start.elapsed_time(end)))

    peak_mem_mib = torch.cuda.max_memory_allocated() / (1024.0**2)
    median_ms = statistics.median(elapsed_ms)
    mean_ms = statistics.fmean(elapsed_ms)
    p95_ms = percentile(elapsed_ms, 0.95)
    fps = 1000.0 * args.batch / median_ms
    params = sum(p.numel() for p in model.parameters())

    print("\n" + "=" * 50)
    print(f"Model:           {args.model.name}")
    print(f"Parameters:      {params:,} ({params / 1e6:.3f}M)")
    print(f"Precision:       {'FP16' if args.half else 'FP32'}")
    print(f"Median Latency:  {median_ms:.2f} ms")
    print(f"Mean Latency:    {mean_ms:.2f} ms")
    print(f"P95 Latency:     {p95_ms:.2f} ms")
    print(f"Throughput:      {fps:.1f} frames/sec")
    print(f"Peak GPU VRAM:   {peak_mem_mib:.1f} MiB")
    print("=" * 50)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "model", "parameters", "precision", "batch", "imgsz",
                    "latency_median_ms", "latency_mean_ms", "latency_p95_ms",
                    "throughput_fps", "peak_memory_mib"
                ],
            )
            writer.writeheader()
            writer.writerow({
                "model": args.model.name,
                "parameters": params,
                "precision": "FP16" if args.half else "FP32",
                "batch": args.batch,
                "imgsz": args.imgsz,
                "latency_median_ms": median_ms,
                "latency_mean_ms": mean_ms,
                "latency_p95_ms": p95_ms,
                "throughput_fps": fps,
                "peak_memory_mib": peak_mem_mib,
            })


if __name__ == "__main__":
    main()
