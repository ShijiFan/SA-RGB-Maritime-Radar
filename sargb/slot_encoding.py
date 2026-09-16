"""Slot-Addressable RGB (SA-RGB) radar-grid encoding and image synthesis.

Maps multi-feature radar descriptors (K > 3) into standardized 3-channel RGB
tensors via fixed spatial-channel slot assignments, preserving topological
neighborhoods while maintaining compatibility with standard vision backbones.
"""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

FEATURE_ORDER = ("Hurst", "RAA", "RDPH", "RVE", "RI", "NR", "MS")
FEATURE_GROUPS = {
    "G1": ("RAA",),
    "G3": ("RAA", "RDPH", "RVE"),
    "G4": ("RAA", "RDPH", "RVE", "Hurst"),
    "G5": ("RAA", "RDPH", "RVE", "Hurst", "RI"),
    "G7": ("Hurst", "RAA", "RDPH", "RVE", "RI", "NR", "MS"),
}


def fit_training_percentiles(
    features: np.ndarray,
    train_count: int,
    low: float = 1.0,
    high: float = 99.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit 1st-99th percentile clipping ranges strictly on training windows.

    Guarantees zero information leakage from validation or held-out test splits.
    """
    train_data = features[:train_count].reshape(-1, features.shape[-1])
    lows = np.nanpercentile(train_data, low, axis=0)
    highs = np.nanpercentile(train_data, high, axis=0)
    same = np.isclose(highs, lows)
    highs[same] = lows[same] + 1.0
    return lows, highs


def quantize_features(
    features: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
) -> dict[str, np.ndarray]:
    """Clip and quantize continuous features to 8-bit dynamic range [0, 255]."""
    n_features = features.shape[-1]
    clipped = np.clip(features, lows.reshape(1, 1, -1), highs.reshape(1, 1, -1))
    normalized = (clipped - lows.reshape(1, 1, -1)) / (highs - lows).reshape(1, 1, -1)
    values = np.rint(normalized * 255.0).clip(0, 255).astype(np.uint8)
    return {FEATURE_ORDER[i]: values[:, :, i] for i in range(n_features)}


def grid_shape(n_cells: int) -> tuple[int, int]:
    """Compute 2-row radar grid geometry."""
    if n_cells in (31, 32):
        return 2, 16
    if n_cells % 2 == 0:
        return 2, n_cells // 2
    cols = int(math.ceil(math.sqrt(n_cells)))
    rows = int(math.ceil(n_cells / cols))
    return rows, cols


def cell_to_row_col(cell_index_zero: int, rows: int, cols: int, order: str = "odd-even") -> tuple[int, int]:
    """Map sequential 1D range cell index to 2D grid position."""
    if order == "odd-even":
        return cell_index_zero % 2, cell_index_zero // 2
    if order == "column-major":
        return cell_index_zero % rows, cell_index_zero // rows
    return cell_index_zero // cols, cell_index_zero % cols


def encode_sargb_cell(
    q_dict: dict[str, int],
    cell_pixels: int = 100,
    anchor: int = 128,
) -> np.ndarray:
    """Encode one radar macro-cell (100x100 px) with 4 sub-quadrant slots.

    Slot layout:
    - Slot 0 (top-left, 50x50):     R=RAA,   G=RDPH,  B=RVE
    - Slot 1 (top-right, 50x50):    R=Hurst, G=RI,    B=NR
    - Slot 2 (bottom-left, 50x50):  R=MS,    G=anchor, B=anchor
    - Slot 3 (bottom-right, 50x50): R=anchor, G=anchor, B=anchor
    """
    block = np.full((cell_pixels, cell_pixels, 3), anchor, dtype=np.uint8)
    half = cell_pixels // 2

    def get_val(name: str) -> int:
        return q_dict.get(name, anchor)

    # Slot 0: classic tri-features (Doppler/amplitude)
    block[:half, :half, :] = (get_val("RAA"), get_val("RDPH"), get_val("RVE"))
    # Slot 1: fractal & time-frequency structure
    block[:half, half:, :] = (get_val("Hurst"), get_val("RI"), get_val("NR"))
    # Slot 2: time-frequency cluster extent
    block[half:, :half, :] = (get_val("MS"), anchor, anchor)
    # Slot 3: reserved neutral anchor
    block[half:, half:, :] = (anchor, anchor, anchor)

    return block


def render_sargb_image(
    quantized_burst: dict[str, np.ndarray],
    image_idx: int,
    n_cells: int = 14,
    cell_pixels: int = 100,
    order: str = "odd-even",
    anchor: int = 128,
) -> np.ndarray:
    """Synthesize full radar-grid image for one observation window."""
    rows, cols = grid_shape(n_cells)
    image = np.full((rows * cell_pixels, cols * cell_pixels, 3), anchor, dtype=np.uint8)

    for cell_idx in range(n_cells):
        cell_features = {
            name: int(quantized_burst[name][image_idx, cell_idx])
            for name in quantized_burst
        }
        block = encode_sargb_cell(cell_features, cell_pixels=cell_pixels, anchor=anchor)
        r, c = cell_to_row_col(cell_idx, rows, cols, order)
        y0, x0 = r * cell_pixels, c * cell_pixels
        image[y0 : y0 + cell_pixels, x0 : x0 + cell_pixels, :] = block

    return image


def yolo_box_label(
    target_cell_1indexed: int,
    n_cells: int = 14,
    order: str = "odd-even",
    box_scale: float = 1.0,
) -> str:
    """Generate normalized YOLO bounding box label string for the target cell."""
    rows, cols = grid_shape(n_cells)
    target_zero = target_cell_1indexed - 1
    if target_zero < 0 or target_zero >= n_cells:
        raise ValueError(f"Target cell {target_cell_1indexed} out of bounds for {n_cells} cells.")
    r, c = cell_to_row_col(target_zero, rows, cols, order)
    x_center = (c + 0.5) / cols
    y_center = (r + 0.5) / rows
    width = min(1.0 / cols * box_scale, 1.0)
    height = min(1.0 / rows * box_scale, 1.0)
    return f"0 {x_center:.8f} {y_center:.8f} {width:.8f} {height:.8f}\n"
