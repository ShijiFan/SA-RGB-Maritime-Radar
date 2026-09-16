"""Physical radar feature extraction for maritime small-target detection.

Implements the seven-feature feature set:
1. RAA: Relative Average Amplitude
2. RDPH: Relative Doppler Peak Height
3. RVE: Relative Vector Entropy
4. Hurst: Fractional Brownian motion Hurst exponent via second-order structure function
5. RI: Relative Inversion from Normalized Time-Frequency Distribution (NTFD)
6. NR: Number of Regions exceeding threshold in NTFD
7. MS: Maximum Region Size in NTFD
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

EPS = 1e-12
FEATURE_ORDER = ("Hurst", "RAA", "RDPH", "RVE", "RI", "NR", "MS")


def relative_to_reference(values: np.ndarray) -> np.ndarray:
    """Compute cell value relative to the mean of other range cells in the burst."""
    n_cells = values.shape[0]
    ref = (values.sum(axis=0, keepdims=True) - values) / max(n_cells - 1, 1)
    return values / (ref + EPS)


def hurst_structure(x: np.ndarray) -> float:
    """Estimate the Hurst exponent via second-order structure function."""
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
    sample_rate: int = 1000,
    nperseg: int = 64,
    noverlap: int = 48,
    nfft: int = 128,
    threshold: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract RI, NR, and MS from normalized STFT time-frequency distribution."""
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


def extract_radar_features(
    complex_data: np.ndarray,
    window: int = 512,
    sample_rate: int = 1000,
    tf_nperseg: int = 64,
    tf_noverlap: int = 48,
    tf_nfft: int = 128,
    tf_threshold: float = 2.0,
    feature_names: tuple[str, ...] = FEATURE_ORDER,
) -> np.ndarray:
    """Extract physical features for all range cells across chronological windows.

    Args:
        complex_data: (n_cells, n_pulses) complex I/Q matrix.
        window: observation duration in pulses (e.g., 512 pulses for 0.512s @ 1kHz).
        sample_rate: pulse repetition frequency (PRF).
        feature_names: tuple of feature names to compute.

    Returns:
        (n_windows, n_cells, len(FEATURE_ORDER)) feature matrix.
    """
    n_cells, n_samples = complex_data.shape
    n_images = n_samples // window
    seg = complex_data[:, : n_images * window].reshape(n_cells, n_images, window)

    # 1. Tri-features (Doppler and amplitude)
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

    need_hurst = "Hurst" in feature_names
    need_tf = any(name in feature_names for name in ("RI", "NR", "MS"))

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
