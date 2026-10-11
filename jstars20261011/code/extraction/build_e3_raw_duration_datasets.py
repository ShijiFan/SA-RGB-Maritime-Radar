#!/usr/bin/env python
"""Build controlled raw-feature E3 observation-duration data sets.

This builder intentionally forms a separate, all-raw duration-sweep protocol.
The historic hybrid main protocol uses precomputed 0.512-s RAA/RDPH/RVE maps;
those maps do not exist at the other durations.  Here all seven descriptors are
re-extracted from the same raw IPIX1993 pulses at every duration.  Comparisons
are therefore valid *within this sweep* and are never substituted for the
historic hybrid main table.

For ``matched`` mode, every record-polarization sequence is split on identical
physical time boundaries and then subsampled inside each chronological split to
90/23/15 images.  Rendering percentiles are fitted only on the selected train
segments of that duration and sequence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score


POLARIZATIONS = ("HH", "HV", "VH", "VV")
FEATURES = ("Hurst", "RAA", "RDPH", "RVE", "RI", "NR", "MS")
CONFIG_FEATURES = {
    "G3": ("RAA", "RDPH", "RVE"),
    "G7": FEATURES,
    # Slot B is neutral. Slot C keeps MS at the same address used by G7.
    "G4robust": ("RAA", "RDPH", "RVE", "MS"),
}
FULL_SPLITS = {
    1.024: (90, 23, 15),
    0.512: (180, 46, 30),
    0.256: (360, 92, 60),
    0.128: (720, 184, 120),
}
MATCHED_SPLIT = (90, 23, 15)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools-dir", type=Path, required=True)
    parser.add_argument("--raw-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--durations", nargs="+", type=float, choices=tuple(FULL_SPLITS), default=tuple(FULL_SPLITS))
    parser.add_argument("--configs", nargs="+", choices=tuple(CONFIG_FEATURES), default=tuple(CONFIG_FEATURES))
    parser.add_argument("--regime", choices=("matched", "full"), default="matched")
    parser.add_argument("--sample-rate", type=int, default=1000)
    parser.add_argument("--cell-pixels", type=int, default=100)
    parser.add_argument("--anchor-value", type=int, default=128)
    parser.add_argument("--clip-low", type=float, default=1.0)
    parser.add_argument("--clip-high", type=float, default=99.0)
    parser.add_argument("--tf-nperseg", type=int, default=64)
    parser.add_argument("--tf-noverlap", type=int, default=48)
    parser.add_argument("--tf-nfft", type=int, default=128)
    parser.add_argument("--tf-threshold", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true", help="Reuse an existing feature cache and complete output data sets.")
    return parser.parse_args()


def duration_tag(duration: float) -> str:
    return f"{duration:.3f}".replace(".", "p")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evenly_spaced_indices(total: int, count: int) -> np.ndarray:
    if count < 1 or count > total:
        raise ValueError(f"Cannot select {count} samples from {total}")
    if count == total:
        return np.arange(total, dtype=np.int64)
    if count == 1:
        return np.array([0], dtype=np.int64)
    indices = np.rint(np.linspace(0, total - 1, count)).astype(np.int64)
    if len(np.unique(indices)) != count:
        raise RuntimeError(f"Equal-interval subsample has duplicate indices: n={total}, k={count}")
    return indices


def selection_indices(duration: float, regime: str) -> dict[str, np.ndarray]:
    train, val, test = FULL_SPLITS[duration]
    ranges = {
        "train": np.arange(0, train, dtype=np.int64),
        "val": np.arange(train, train + val, dtype=np.int64),
        "test": np.arange(train + val, train + val + test, dtype=np.int64),
    }
    if regime == "full":
        return ranges
    return {
        split: base[evenly_spaced_indices(len(base), wanted)]
        for split, base, wanted in zip(("train", "val", "test"), (ranges["train"], ranges["val"], ranges["test"]), MATCHED_SPLIT)
    }


def load_helpers(tools_dir: Path) -> dict[str, Any]:
    resolved = str(tools_dir.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    from paper3_slotrgb_dataset import (  # pylint: disable=import-outside-toplevel
        FEATURE_ORDER,
        META_1993,
        extract_features,
        fit_range,
        grid_shape,
        load_complex_pol,
        ordered_files,
        quantize_selected,
        render_slotrgb,
        yolo_label,
    )
    return {
        "FEATURE_ORDER": FEATURE_ORDER,
        "META_1993": META_1993,
        "extract_features": extract_features,
        "fit_range": fit_range,
        "grid_shape": grid_shape,
        "load_complex_pol": load_complex_pol,
        "ordered_files": ordered_files,
        "quantize_selected": quantize_selected,
        "render_slotrgb": render_slotrgb,
        "yolo_label": yolo_label,
    }


def write_yaml(dataset: Path) -> None:
    (dataset / "data.yaml").write_text(
        "\n".join(
            (
                f"path: {dataset.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: target",
                "",
            )
        ),
        encoding="utf-8",
    )


def ensure_dataset_dirs(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output}")
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)


def cache_paths(cache_dir: Path, duration: float) -> tuple[Path, Path]:
    tag = duration_tag(duration)
    return cache_dir / f"raw_features_{tag}.npz", cache_dir / f"raw_features_{tag}.json"


def extract_or_load_cache(args: argparse.Namespace, helpers: dict[str, Any], duration: float) -> tuple[dict[str, np.ndarray], dict]:
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    array_path, manifest_path = cache_paths(cache_dir, duration)
    if args.resume and array_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if float(manifest.get("duration_s", -1)) != duration:
            raise ValueError(f"Duration mismatch in cache: {array_path}")
        arrays = {key: value for key, value in np.load(array_path).items()}
        return arrays, manifest
    if array_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Partial or existing cache requires --resume: {array_path}")

    ordered = helpers["ordered_files"](args.raw_source.resolve(), helpers["META_1993"], 0)
    if len(ordered) != 10:
        raise ValueError(f"Expected ten IPIX1993 records; found {len(ordered)}")
    expected_segments = sum(FULL_SPLITS[duration])
    window = int(round(duration * args.sample_rate))
    arrays: dict[str, np.ndarray] = {}
    records = []
    for group, (file_path, meta) in enumerate(ordered, start=1):
        mat = loadmat(str(file_path), squeeze_me=True, struct_as_record=False)
        for pol in POLARIZATIONS:
            data = helpers["load_complex_pol"](mat, pol)
            if data.shape[0] != 14:
                raise ValueError(f"Expected 14 cells in {file_path.name} {pol}; found {data.shape[0]}")
            if data.shape[1] != expected_segments * window:
                raise ValueError(
                    f"Pulse/segment mismatch in {file_path.name} {pol}: {data.shape[1]} != {expected_segments}*{window}"
                )
            features = helpers["extract_features"](
                data=data,
                window=window,
                sample_rate=args.sample_rate,
                tf_nperseg=args.tf_nperseg,
                tf_noverlap=args.tf_noverlap,
                tf_nfft=args.tf_nfft,
                tf_threshold=args.tf_threshold,
                required_names=FEATURES,
            )
            if features.shape != (expected_segments, 14, len(FEATURES)):
                raise ValueError(f"Unexpected feature tensor {features.shape} for {file_path.name} {pol}")
            arrays[f"g{group:02d}_{pol}"] = features.astype(np.float32)
            records.append(
                {
                    "group": group,
                    "polarization": pol,
                    "raw_file": str(file_path.resolve()),
                    "raw_sha256": sha256(file_path),
                    "target_cell": int(meta.target_cell),
                    "cells": int(data.shape[0]),
                    "pulses": int(data.shape[1]),
                    "segments": int(features.shape[0]),
                }
            )
            print(f"EXTRACT duration={duration:.3f}s group={group:02d} pol={pol}", flush=True)

    np.savez_compressed(array_path, **arrays)
    manifest = {
        "dataset": "IPIX1993",
        "protocol": "E3_raw_reextraction_duration_sweep",
        "duration_s": duration,
        "sample_rate_hz": args.sample_rate,
        "window_pulses": window,
        "feature_order": list(FEATURES),
        "feature_extraction": {
            "tf_nperseg": args.tf_nperseg,
            "tf_noverlap": args.tf_noverlap,
            "tf_nfft": args.tf_nfft,
            "tf_threshold": args.tf_threshold,
        },
        "cache": str(array_path),
        "records": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return arrays, manifest


def feature_degradation_rows(
    arrays: dict[str, np.ndarray],
    duration: float,
    helpers: dict[str, Any],
    regime: str,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    selected = selection_indices(duration, regime)
    feature_order = helpers["FEATURE_ORDER"]
    target_cells = {index + 1: int(meta.target_cell) for index, meta in enumerate(helpers["META_1993"])}
    rows: list[dict[str, float | int | str]] = []
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for key, values in arrays.items():
        group = int(key[1:3])
        pol = key.split("_", maxsplit=1)[1]
        test_values = values[selected["test"]]
        target_index = target_cells[group] - 1
        for feature_index, feature in enumerate(feature_order):
            target = test_values[:, target_index, feature_index]
            clutter = np.delete(test_values[:, :, feature_index], target_index, axis=1).reshape(-1)
            labels = np.concatenate((np.ones(target.size, dtype=np.int8), np.zeros(clutter.size, dtype=np.int8)))
            scores = np.concatenate((target, clutter))
            raw_auc = float(roc_auc_score(labels, scores))
            separation_auc = max(raw_auc, 1.0 - raw_auc)
            pooled = float(np.sqrt((np.var(target, ddof=1) + np.var(clutter, ddof=1)) / 2.0))
            effect = abs(float(np.mean(target) - np.mean(clutter))) / max(pooled, 1e-12)
            row = {
                "duration_s": duration,
                "regime": regime,
                "group": group,
                "polarization": pol,
                "feature": feature,
                "test_target_values": int(target.size),
                "test_clutter_values": int(clutter.size),
                "raw_auc_target_higher": raw_auc,
                "separation_auc_direction_free": separation_auc,
                "abs_standardized_mean_difference": effect,
            }
            rows.append(row)
            grouped[feature].append({"auc": separation_auc, "effect": effect})

    aggregate = []
    for feature in feature_order:
        vals = grouped[feature]
        aggregate.append(
            {
                "duration_s": duration,
                "regime": regime,
                "feature": feature,
                "units": len(vals),
                "separation_auc_mean": float(np.mean([item["auc"] for item in vals])),
                "separation_auc_std": float(np.std([item["auc"] for item in vals], ddof=1)),
                "abs_standardized_mean_difference_mean": float(np.mean([item["effect"] for item in vals])),
                "abs_standardized_mean_difference_std": float(np.std([item["effect"] for item in vals], ddof=1)),
            }
        )
    return rows, aggregate


def build_config_dataset(
    args: argparse.Namespace,
    helpers: dict[str, Any],
    arrays: dict[str, np.ndarray],
    cache_manifest: dict,
    duration: float,
    regime: str,
    config: str,
) -> dict:
    tag = duration_tag(duration)
    output = args.output_root.resolve() / f"e3_raw_{tag}_{regime}_{config.lower()}"
    if output.exists() and any(output.iterdir()):
        report_path = output / "build_report.json"
        if args.resume and report_path.is_file():
            return json.loads(report_path.read_text(encoding="utf-8"))
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output}")
    ensure_dataset_dirs(output)
    selected_indices = selection_indices(duration, regime)
    feature_order = helpers["FEATURE_ORDER"]
    selected_feature_names = CONFIG_FEATURES[config]
    selected_feature_indices = [feature_order.index(name) for name in selected_feature_names]
    target_cells = {index + 1: int(meta.target_cell) for index, meta in enumerate(helpers["META_1993"])}
    split_counts = {"train": 0, "val": 0, "test": 0}
    rows: list[dict[str, Any]] = []
    normalization = []
    for key in sorted(arrays):
        values = arrays[key].astype(np.float64)
        group = int(key[1:3])
        pol = key.split("_", maxsplit=1)[1]
        train_indices = selected_indices["train"]
        lows, highs, fit_description = helpers["fit_range"](
            values[train_indices],
            selected_feature_indices,
            len(train_indices),
            "percentile",
            args.clip_low,
            args.clip_high,
        )
        q = helpers["quantize_selected"](values, selected_feature_indices, lows, highs)
        target_cell = target_cells[group]
        label = helpers["yolo_label"](target_cell, 14, "odd-even", 1.0)
        for split in ("train", "val", "test"):
            for index in selected_indices[split]:
                name = f"1993_{pol.lower()}_group{group:02d}_raw_{tag}_{regime}_{config}_{int(index) + 1:04d}"
                image = helpers["render_slotrgb"](q, int(index), 14, args.cell_pixels, "odd-even", args.anchor_value)
                Image.fromarray(image).save(output / "images" / split / f"{name}.png")
                (output / "labels" / split / f"{name}.txt").write_text(label, encoding="utf-8")
                split_counts[split] += 1
        rows.append(
            {
                "group": group,
                "polarization": pol,
                "target_cell": target_cell,
                "train": len(selected_indices["train"]),
                "val": len(selected_indices["val"]),
                "test": len(selected_indices["test"]),
            }
        )
        normalization.append(
            {
                "group": group,
                "polarization": pol,
                "feature_names": list(selected_feature_names),
                "low": lows.tolist(),
                "high": highs.tolist(),
                "fit": fit_description,
                "fit_train_indices_zero_based": train_indices.tolist(),
            }
        )
    write_yaml(output)
    with (output / "build_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "experiment": "E3_duration_sweep",
        "protocol": "raw_reextraction; matched comparisons only within this sweep",
        "duration_s": duration,
        "regime": regime,
        "feature_group": config,
        "features": list(selected_feature_names),
        "anchor_value": args.anchor_value,
        "active_features": list(selected_feature_names),
        "slot_assignment": {
            "A": ["RAA", "RDPH", "RVE"],
            "B": ["Hurst", "RI", "NR"] if config == "G7" else ["anchor", "anchor", "anchor"],
            "C": ["MS", "anchor", "anchor"] if config in {"G7", "G4robust"} else ["anchor", "anchor", "anchor"],
            "anchor_value": args.anchor_value,
        },
        "splits_per_record_polarization": {split: len(indices) for split, indices in selected_indices.items()},
        "total_splits": split_counts,
        "selection_indices_zero_based": {split: indices.tolist() for split, indices in selected_indices.items()},
        "normalization": "per record-polarization train split, 1st-99th percentile",
        "normalization_records": normalization,
        "raw_feature_cache": cache_manifest["cache"],
        "raw_feature_cache_protocol": cache_manifest["protocol"],
        "raw_feature_order": cache_manifest["feature_order"],
        "output": str(output),
    }
    (output / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    args.raw_source = args.raw_source.resolve()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    helpers = load_helpers(args.tools_dir)
    generation: list[dict[str, Any]] = []
    feature_long: list[dict[str, float | int | str]] = []
    feature_summary: list[dict[str, float | int | str]] = []
    for duration in sorted(args.durations, reverse=True):
        arrays, cache_manifest = extract_or_load_cache(args, helpers, duration)
        unit_rows, summary_rows = feature_degradation_rows(arrays, duration, helpers, args.regime)
        feature_long.extend(unit_rows)
        feature_summary.extend(summary_rows)
        for config in args.configs:
            report = build_config_dataset(args, helpers, arrays, cache_manifest, duration, args.regime, config)
            generation.append(
                {
                    "duration_s": duration,
                    "regime": args.regime,
                    "config": config,
                    "dataset": report["output"],
                    **report["total_splits"],
                }
            )
            print(f"BUILT duration={duration:.3f}s regime={args.regime} config={config}", flush=True)
    artifact_root = args.output_root.parent
    with (artifact_root / f"E3_feature_degradation_long_{args.regime}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(feature_long[0]))
        writer.writeheader()
        writer.writerows(feature_long)
    with (artifact_root / f"E3_feature_degradation_{args.regime}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(feature_summary[0]))
        writer.writeheader()
        writer.writerows(feature_summary)
    result = {
        "experiment": "E3_duration_sweep",
        "regime": args.regime,
        "durations_s": sorted(args.durations),
        "configs": args.configs,
        "matched_split": list(MATCHED_SPLIT) if args.regime == "matched" else None,
        "datasets": generation,
        "feature_degradation_summary": str(artifact_root / f"E3_feature_degradation_{args.regime}.csv"),
        "boundary": "All durations use raw re-extraction. Their absolute metrics are not substitutes for the hybrid 0.512-s main table.",
    }
    (artifact_root / f"E3_duration_dataset_build_{args.regime}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
