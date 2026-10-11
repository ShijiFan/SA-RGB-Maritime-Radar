"""Build Paper3 slot-addressable RGB radar-grid datasets.

The script is intentionally separate from ``triradar_dataset.py`` because
Paper3 tests a different representation question: how to encode more than one
RGB triplet while keeping range-cell topology and secondary-cell context.

Supported image protocols:

- ``rgb``: classic one-slot RAA/RDPH/RVE RGB rendering.
- ``slotrgb``: macro-cell with fixed semantic RGB slots.
- ``geometry``: feature value encoded by square side length, used as a fair
  readability baseline.

The seven-feature implementation follows the local Paper3 evidence map:
Hurst, RAA, RDPH, RVE, RI, NR, and MS. RI/NR/MS are extracted from a
reference-normalized short-time time-frequency distribution. This is an
operational STFT-based equivalent of the NTFD idea used for reproducible
experiments; the exact settings are written to build_report.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage, signal
from scipy.io import loadmat


EPS = 1e-12
POLARIZATIONS = ("HH", "HV", "VH", "VV")

FEATURE_ORDER = ("Hurst", "RAA", "RDPH", "RVE", "RI", "NR", "MS")
FEATURE_GROUPS = {
    "G1": ("RAA",),
    "G3": ("RAA", "RDPH", "RVE"),
    "G4": ("RAA", "RDPH", "RVE", "Hurst"),
    "G5": ("RAA", "RDPH", "RVE", "Hurst", "RI"),
    "G7": ("Hurst", "RAA", "RDPH", "RVE", "RI", "NR", "MS"),
}


@dataclass(frozen=True)
class FileMeta:
    stem: str
    target_cell: int


META_1993 = (
    FileMeta("19931107_135603", 9),
    FileMeta("19931108_220902", 7),
    FileMeta("19931109_191449", 7),
    FileMeta("19931109_202217", 7),
    FileMeta("19931110_001635", 7),
    FileMeta("19931111_163625", 8),
    FileMeta("19931118_023604", 8),
    FileMeta("19931118_162155", 7),
    FileMeta("19931118_162658", 7),
    FileMeta("19931118_174259", 7),
)

META_1998 = (
    FileMeta("19980204_202225", 24),
    FileMeta("19980204_202525", 7),
    FileMeta("19980204_163113", 24),
    FileMeta("19980205_171437", 7),
    FileMeta("19980205_180558", 7),
    FileMeta("19980212_195704", 7),
    FileMeta("19980223_164055", 31),
    FileMeta("19980223_173317", 32),
    FileMeta("19980223_173950", 29),
)

SPLITS_1993 = {
    0.256: (300, 150, 62),
    0.512: (181, 45, 30),
    1.024: (75, 25, 28),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Folder with IPIX .mat files.")
    parser.add_argument("--output", type=Path, required=True, help="Output YOLO dataset folder.")
    parser.add_argument("--dataset", choices=("1993", "1998"), default="1993")
    parser.add_argument("--protocol", choices=("rgb", "slotrgb", "geometry"), default="slotrgb")
    parser.add_argument("--feature-group", choices=tuple(FEATURE_GROUPS), default="G7")
    parser.add_argument("--duration", type=float, choices=(0.256, 0.512, 1.024), default=0.512)
    parser.add_argument("--sample-rate", type=int, default=1000)
    parser.add_argument("--polarizations", nargs="+", default=list(POLARIZATIONS), choices=POLARIZATIONS)
    parser.add_argument("--norm", choices=("percentile", "minmax-train", "minmax-full"), default="percentile")
    parser.add_argument(
        "--cell-mode",
        choices=("full", "target-window14"),
        default="full",
        help="Render all range cells, or crop/pad a target-centered 14-cell local grid for protocol-aligned IPIX1998 diagnostics.",
    )
    parser.add_argument("--clip-low", type=float, default=1.0)
    parser.add_argument("--clip-high", type=float, default=99.0)
    parser.add_argument("--cell-pixels", type=int, default=100)
    parser.add_argument("--box-scale", type=float, default=1.0)
    parser.add_argument("--order", choices=("odd-even", "row-major", "column-major"), default="odd-even")
    parser.add_argument("--limit-files", type=int, default=0, help="Process only first N matched records.")
    parser.add_argument(
        "--anchor-value",
        type=int,
        default=128,
        help="Neutral uint8 value for unused slot/channel positions. Use 128 so absence is not encoded as a low feature value.",
    )
    parser.add_argument("--tf-nperseg", type=int, default=64)
    parser.add_argument("--tf-noverlap", type=int, default=48)
    parser.add_argument("--tf-nfft", type=int, default=128)
    parser.add_argument("--tf-threshold", type=float, default=2.0)
    parser.add_argument("--geometry-min-side", type=int, default=4)
    parser.add_argument("--geometry-max-side", type=int, default=22)
    return parser.parse_args()


def ordered_files(source: Path, metas: tuple[FileMeta, ...], limit: int = 0) -> list[tuple[Path, FileMeta]]:
    pairs: list[tuple[Path, FileMeta]] = []
    for meta in metas:
        matches = sorted(source.glob(f"{meta.stem}*.mat"))
        if matches:
            pairs.append((matches[0], meta))
    if not pairs:
        raise FileNotFoundError(f"No .mat files matched metadata in {source}")
    return pairs[:limit] if limit > 0 else pairs


def load_complex_pol(mat: dict, pol: str) -> np.ndarray:
    obj = mat[pol]
    data = np.asarray(obj.I, dtype=np.float64) + 1j * np.asarray(obj.Q, dtype=np.float64)
    if data.shape[0] < data.shape[1]:
        return data
    return data.T


def windowed(data: np.ndarray, window: int) -> np.ndarray:
    n_cells, n_samples = data.shape
    n_images = n_samples // window
    trimmed = data[:, : n_images * window]
    return trimmed.reshape(n_cells, n_images, window)


def split_counts(dataset: str, duration: float, n_images: int) -> tuple[int, int, int]:
    if dataset == "1993" and duration in SPLITS_1993 and sum(SPLITS_1993[duration]) == n_images:
        return SPLITS_1993[duration]
    train = int(math.floor(n_images * 0.7))
    val = int(math.floor(n_images * 0.15))
    test = n_images - train - val
    return train, val, test


def split_name(index: int, counts: tuple[int, int, int]) -> str:
    train, val, _ = counts
    if index < train:
        return "train"
    if index < train + val:
        return "val"
    return "test"


def relative_to_reference(values: np.ndarray) -> np.ndarray:
    n_cells = values.shape[0]
    ref = (values.sum(axis=0, keepdims=True) - values) / max(n_cells - 1, 1)
    return values / (ref + EPS)


def hurst_structure(x: np.ndarray) -> float:
    n = x.size
    lags = np.array([4, 8, 16, 32, 64, 128, 256], dtype=int)
    lags = lags[lags <= max(4, n // 2)]
    vals = []
    used = []
    for lag in lags:
        diff = x[lag:] - x[:-lag]
        value = float(np.sqrt(np.mean(np.abs(diff) ** 2)))
        if value > EPS:
            vals.append(value)
            used.append(lag)
    if len(vals) < 2:
        return 0.0
    slope, _ = np.polyfit(np.log2(used), np.log2(vals), 1)
    return float(slope)


def extract_time_frequency_features(
    seg_img: np.ndarray,
    sample_rate: int,
    nperseg: int,
    noverlap: int,
    nfft: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, _, z = signal.stft(
        seg_img,
        fs=sample_rate,
        nperseg=min(nperseg, seg_img.shape[-1]),
        noverlap=min(noverlap, max(0, min(nperseg, seg_img.shape[-1]) - 1)),
        nfft=max(nfft, min(nperseg, seg_img.shape[-1])),
        axis=-1,
        return_onesided=False,
        boundary=None,
        padded=False,
    )
    power = np.abs(z) ** 2
    n_cells = power.shape[0]
    sums = power.sum(axis=0, keepdims=True)
    sqs = (power**2).sum(axis=0, keepdims=True)
    mean_ref = (sums - power) / max(n_cells - 1, 1)
    var_ref = (sqs - power**2) / max(n_cells - 1, 1) - mean_ref**2
    std_ref = np.sqrt(np.maximum(var_ref, 0.0))
    ntfd = (power - mean_ref) / (std_ref + EPS)
    positive = np.maximum(ntfd, 0.0)
    ri = positive.max(axis=1).sum(axis=1)

    nr = np.zeros(n_cells, dtype=np.float64)
    ms = np.zeros(n_cells, dtype=np.float64)
    structure = np.ones((3, 3), dtype=np.uint8)
    binary = positive > threshold
    for c in range(n_cells):
        labels, count = ndimage.label(binary[c], structure=structure)
        nr[c] = count
        if count:
            sizes = ndimage.sum(binary[c], labels, index=np.arange(1, count + 1))
            ms[c] = float(np.max(sizes))
    return ri, nr, ms


def extract_features(
    data: np.ndarray,
    window: int,
    sample_rate: int,
    tf_nperseg: int,
    tf_noverlap: int,
    tf_nfft: int,
    tf_threshold: float,
    required_names: tuple[str, ...],
) -> np.ndarray:
    seg = windowed(data, window)
    n_cells, n_images, _ = seg.shape

    amp_mean = np.mean(np.abs(seg), axis=2)
    spectrum = np.fft.fftshift(np.fft.fft(seg, axis=2), axes=2)
    mag = np.abs(spectrum)
    rdph_abs = mag.max(axis=2) / (mag.mean(axis=2) + EPS)
    prob = mag / (mag.sum(axis=2, keepdims=True) + EPS)
    rve_abs = -np.sum(prob * np.log(prob + EPS), axis=2)

    raa = relative_to_reference(amp_mean)
    rdph = relative_to_reference(rdph_abs)
    rve = relative_to_reference(rve_abs)

    out = np.zeros((n_images, n_cells, len(FEATURE_ORDER)), dtype=np.float64)
    out[:, :, FEATURE_ORDER.index("RAA")] = raa.T
    out[:, :, FEATURE_ORDER.index("RDPH")] = rdph.T
    out[:, :, FEATURE_ORDER.index("RVE")] = rve.T

    need_hurst = "Hurst" in required_names
    need_tf = any(name in required_names for name in ("RI", "NR", "MS"))
    if need_hurst or need_tf:
        for i in range(n_images):
            seg_img = seg[:, i, :]
            if need_hurst:
                out[i, :, FEATURE_ORDER.index("Hurst")] = [hurst_structure(seg_img[c]) for c in range(n_cells)]
            if need_tf:
                ri, nr, ms = extract_time_frequency_features(
                    seg_img,
                    sample_rate=sample_rate,
                    nperseg=tf_nperseg,
                    noverlap=tf_noverlap,
                    nfft=tf_nfft,
                    threshold=tf_threshold,
                )
                out[i, :, FEATURE_ORDER.index("RI")] = ri
                out[i, :, FEATURE_ORDER.index("NR")] = nr
                out[i, :, FEATURE_ORDER.index("MS")] = ms
    return out


def feature_indices(feature_names: tuple[str, ...]) -> list[int]:
    return [FEATURE_ORDER.index(name) for name in feature_names]


def fit_range(
    features: np.ndarray,
    selected: list[int],
    train_count: int,
    norm: str,
    low: float,
    high: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    selected_features = features[:, :, selected]
    if norm == "percentile":
        fit_data = selected_features[:train_count]
        lows = np.nanpercentile(fit_data.reshape(-1, len(selected)), low, axis=0)
        highs = np.nanpercentile(fit_data.reshape(-1, len(selected)), high, axis=0)
        fit_desc = f"train split only percentiles {low:g}-{high:g}"
    else:
        fit_data = selected_features if norm == "minmax-full" else selected_features[:train_count]
        flat = fit_data.reshape(-1, len(selected))
        lows = np.nanmin(flat, axis=0)
        highs = np.nanmax(flat, axis=0)
        fit_desc = "full sequence min/max" if norm == "minmax-full" else "train split only min/max"
    same = np.isclose(highs, lows)
    highs[same] = lows[same] + 1.0
    return lows, highs, fit_desc


def quantize_selected(features: np.ndarray, selected: list[int], lows: np.ndarray, highs: np.ndarray) -> dict[str, np.ndarray]:
    selected_features = features[:, :, selected]
    clipped = np.clip(selected_features, lows.reshape(1, 1, -1), highs.reshape(1, 1, -1))
    normalized = (clipped - lows.reshape(1, 1, -1)) / (highs - lows).reshape(1, 1, -1)
    values = np.rint(normalized * 255.0).clip(0, 255).astype(np.uint8)
    return {FEATURE_ORDER[idx]: values[:, :, pos] for pos, idx in enumerate(selected)}


def select_target_window14(features: np.ndarray, target_cell: int) -> tuple[np.ndarray, int, list[int | None]]:
    """Crop/pad a 14-cell local window with the target mapped to cell 8.

    This mirrors the Paper2 IPIX1998 protocol-alignment diagnostic. It is used
    only to test whether weak 1998 scores are caused by full-grid topology and
    target-scale mismatch rather than by the seven-feature representation itself.
    """
    n_images, n_cells, n_features = features.shape
    target_zero = target_cell - 1
    start = target_zero - 7
    selected_indices: list[int | None] = []
    window = np.zeros((n_images, 14, n_features), dtype=features.dtype)
    for offset in range(14):
        idx = start + offset
        if 0 <= idx < n_cells:
            window[:, offset, :] = features[:, idx, :]
            selected_indices.append(idx)
        else:
            selected_indices.append(None)
    return window, 8, selected_indices


def grid_shape(n_cells: int) -> tuple[int, int]:
    if n_cells in (31, 32):
        return 2, 16
    if n_cells % 2 == 0:
        return 2, n_cells // 2
    cols = int(math.ceil(math.sqrt(n_cells)))
    rows = int(math.ceil(n_cells / cols))
    return rows, cols


def cell_to_row_col(cell_index_zero: int, rows: int, cols: int, order: str) -> tuple[int, int]:
    if order == "odd-even":
        return cell_index_zero % 2, cell_index_zero // 2
    if order == "column-major":
        return cell_index_zero % rows, cell_index_zero // rows
    return cell_index_zero // cols, cell_index_zero % cols


def put_cell(image: np.ndarray, idx: int, n_cells: int, cell_pixels: int, order: str, block: np.ndarray) -> None:
    rows, cols = grid_shape(n_cells)
    r, c = cell_to_row_col(idx, rows, cols, order)
    if r >= rows or c >= cols:
        return
    y0, y1 = r * cell_pixels, (r + 1) * cell_pixels
    x0, x1 = c * cell_pixels, (c + 1) * cell_pixels
    image[y0:y1, x0:x1, :] = block


def value(q: dict[str, np.ndarray], name: str, image_idx: int, cell_idx: int, anchor: int) -> int:
    if name not in q:
        return anchor
    return int(q[name][image_idx, cell_idx])


def render_rgb(q: dict[str, np.ndarray], image_idx: int, n_cells: int, cell_pixels: int, order: str, anchor: int) -> np.ndarray:
    rows, cols = grid_shape(n_cells)
    image = np.full((rows * cell_pixels, cols * cell_pixels, 3), anchor, dtype=np.uint8)
    for idx in range(n_cells):
        color = np.array(
            [
                value(q, "RAA", image_idx, idx, anchor),
                value(q, "RDPH", image_idx, idx, anchor),
                value(q, "RVE", image_idx, idx, anchor),
            ],
            dtype=np.uint8,
        )
        block = np.zeros((cell_pixels, cell_pixels, 3), dtype=np.uint8)
        block[:, :, :] = color
        put_cell(image, idx, n_cells, cell_pixels, order, block)
    return image


def render_slotrgb(q: dict[str, np.ndarray], image_idx: int, n_cells: int, cell_pixels: int, order: str, anchor: int) -> np.ndarray:
    rows, cols = grid_shape(n_cells)
    image = np.full((rows * cell_pixels, cols * cell_pixels, 3), anchor, dtype=np.uint8)
    half = cell_pixels // 2
    for idx in range(n_cells):
        block = np.full((cell_pixels, cell_pixels, 3), anchor, dtype=np.uint8)
        block[:half, :half, :] = (
            value(q, "RAA", image_idx, idx, anchor),
            value(q, "RDPH", image_idx, idx, anchor),
            value(q, "RVE", image_idx, idx, anchor),
        )
        block[:half, half:, :] = (
            value(q, "Hurst", image_idx, idx, anchor),
            value(q, "RI", image_idx, idx, anchor),
            value(q, "NR", image_idx, idx, anchor),
        )
        block[half:, :half, :] = (
            value(q, "MS", image_idx, idx, anchor),
            anchor,
            anchor,
        )
        put_cell(image, idx, n_cells, cell_pixels, order, block)
    return image


def draw_square(slot: np.ndarray, side: int) -> None:
    h, w, _ = slot.shape
    side = int(max(1, min(side, h, w)))
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    slot[y0 : y0 + side, x0 : x0 + side, :] = 255


def render_geometry(
    q: dict[str, np.ndarray],
    feature_names: tuple[str, ...],
    image_idx: int,
    n_cells: int,
    cell_pixels: int,
    order: str,
    anchor: int,
    min_side: int,
    max_side: int,
) -> np.ndarray:
    rows, cols = grid_shape(n_cells)
    image = np.full((rows * cell_pixels, cols * cell_pixels, 3), anchor, dtype=np.uint8)
    slot_rows, slot_cols = 2, 4
    slot_h = cell_pixels // slot_rows
    slot_w = cell_pixels // slot_cols
    for idx in range(n_cells):
        block = np.full((cell_pixels, cell_pixels, 3), anchor, dtype=np.uint8)
        for fpos, fname in enumerate(feature_names):
            rr = fpos // slot_cols
            cc = fpos % slot_cols
            y0 = rr * slot_h
            x0 = cc * slot_w
            v = value(q, fname, image_idx, idx, anchor)
            side = min_side + round((max_side - min_side) * (v / 255.0))
            draw_square(block[y0 : y0 + slot_h, x0 : x0 + slot_w, :], side)
        put_cell(image, idx, n_cells, cell_pixels, order, block)
    return image


def yolo_label(target_cell: int, n_cells: int, order: str, box_scale: float) -> str:
    rows, cols = grid_shape(n_cells)
    target_zero = target_cell - 1
    if target_zero < 0 or target_zero >= n_cells:
        raise ValueError(f"Target cell {target_cell} is outside {n_cells} cells.")
    r, c = cell_to_row_col(target_zero, rows, cols, order)
    x = (c + 0.5) / cols
    y = (r + 0.5) / rows
    w = min(1.0 / cols * box_scale, 1.0)
    h = min(1.0 / rows * box_scale, 1.0)
    return f"0 {x:.8f} {y:.8f} {w:.8f} {h:.8f}\n"


def ensure_dirs(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output folder already exists and is not empty: {output}")
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_yaml(output: Path) -> None:
    text = "\n".join(
        (
            f"path: {output.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: target",
            "",
        )
    )
    (output / "data.yaml").write_text(text, encoding="utf-8")


def save_sample(output: Path, split: str, name: str, image: np.ndarray, label: str) -> None:
    Image.fromarray(image).save(output / "images" / split / f"{name}.png")
    (output / "labels" / split / f"{name}.txt").write_text(label, encoding="utf-8")


def build_dataset(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    ensure_dirs(output)
    metas = META_1993 if args.dataset == "1993" else META_1998
    files = ordered_files(args.source, metas, args.limit_files)
    window = int(round(args.duration * args.sample_rate))
    feature_names = FEATURE_GROUPS[args.feature_group]
    selected = feature_indices(feature_names)
    counts_by_split = {"train": 0, "val": 0, "test": 0}
    rows = []
    clip_records = []

    for file_path, meta in files:
        mat = loadmat(str(file_path), squeeze_me=True, struct_as_record=False)
        for pol in args.polarizations:
            data = load_complex_pol(mat, pol)
            features = extract_features(
                data=data,
                window=window,
                sample_rate=args.sample_rate,
                tf_nperseg=args.tf_nperseg,
                tf_noverlap=args.tf_noverlap,
                tf_nfft=args.tf_nfft,
                tf_threshold=args.tf_threshold,
                required_names=feature_names,
            )
            original_cells = features.shape[1]
            target_cell = meta.target_cell
            selected_indices: list[int | None] | None = None
            if args.cell_mode == "target-window14":
                features, target_cell, selected_indices = select_target_window14(features, meta.target_cell)
            n_images, n_cells, _ = features.shape
            counts = split_counts(args.dataset, args.duration, n_images)
            lows, highs, fit_desc = fit_range(features, selected, counts[0], args.norm, args.clip_low, args.clip_high)
            q = quantize_selected(features, selected, lows, highs)
            label = yolo_label(target_cell, n_cells, args.order, args.box_scale)

            for idx in range(n_images):
                split = split_name(idx, counts)
                name = f"{args.dataset}_{file_path.stem}_{pol}_{args.protocol}_{args.feature_group}_{idx + 1:04d}"
                if args.protocol == "rgb":
                    image = render_rgb(q, idx, n_cells, args.cell_pixels, args.order, args.anchor_value)
                elif args.protocol == "slotrgb":
                    image = render_slotrgb(q, idx, n_cells, args.cell_pixels, args.order, args.anchor_value)
                else:
                    image = render_geometry(
                        q,
                        feature_names,
                        idx,
                        n_cells,
                        args.cell_pixels,
                        args.order,
                        args.anchor_value,
                        args.geometry_min_side,
                        args.geometry_max_side,
                    )
                save_sample(output, split, name, image, label)
                counts_by_split[split] += 1

            rows.append(
                [
                    file_path.name,
                    pol,
                    n_cells,
                    n_images,
                    *counts,
                    original_cells,
                    meta.target_cell,
                    target_cell,
                ]
            )
            clip_records.append(
                {
                    "file": file_path.name,
                    "polarization": pol,
                    "features": list(feature_names),
                    "low": lows.tolist(),
                    "high": highs.tolist(),
                    "fit": fit_desc,
                    "original_cells": original_cells,
                    "rendered_cells": n_cells,
                    "original_target_cell": meta.target_cell,
                    "mapped_target_cell": target_cell,
                    "selected_original_cell_indices_zero_based": selected_indices,
                }
            )

    write_yaml(output)
    with (output / "build_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "file",
                "polarization",
                "cells",
                "images",
                "train",
                "val",
                "test",
                "original_cells",
                "original_target_cell",
                "mapped_target_cell",
            ]
        )
        writer.writerows(rows)

    report = {
        "dataset": args.dataset,
        "protocol": args.protocol,
        "feature_group": args.feature_group,
        "features": list(feature_names),
        "feature_order_all": list(FEATURE_ORDER),
        "duration": args.duration,
        "window_samples": window,
        "source": str(args.source.resolve()),
        "output": str(output),
        "splits": counts_by_split,
        "normalization": args.norm,
        "cell_mode": args.cell_mode,
        "clip_low": args.clip_low,
        "clip_high": args.clip_high,
        "order": args.order,
        "cell_pixels": args.cell_pixels,
        "box_scale": args.box_scale,
        "anchor_value": args.anchor_value,
        "time_frequency": {
            "method": "STFT-based reference-normalized time-frequency distribution",
            "nperseg": args.tf_nperseg,
            "noverlap": args.tf_noverlap,
            "nfft": args.tf_nfft,
            "threshold": args.tf_threshold,
            "ri": "sum over time of positive NTFD ridge maxima",
            "nr": "8-connected component count of positive NTFD above threshold",
            "ms": "largest 8-connected component size above threshold",
        },
        "geometry": {
            "min_side": args.geometry_min_side,
            "max_side": args.geometry_max_side,
            "slot_rows": 2,
            "slot_cols": 4,
        },
        "clip_records": clip_records,
    }
    (output / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    report = build_dataset(args)
    print(json.dumps({k: report[k] for k in ("output", "splits", "protocol", "feature_group")}, indent=2))


if __name__ == "__main__":
    main()
