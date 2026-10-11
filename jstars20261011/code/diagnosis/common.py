"""Shared loaders and radar-metric helpers for Phase A diagnostics (no training).

Cell convention: metadata target_cell is 1-based; arrays use 0-based index t = target_cell - 1.
H0 cells follow the TAES record-disjoint protocol: all 13 non-target cells of a target frame.
Pure-clutter cells: |c - t| > GUARD (secondary/leakage cells excluded).
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "rev3" / "cache_e3"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

FEATURES = ["Hurst", "RAA", "RDPH", "RVE", "RI", "NR", "MCS"]  # cache order (MS == MCS)
POLS = ["HH", "HV", "VH", "VV"]
GUARD = 2
RECORDS = list(range(1, 11))


def load(duration="0p512"):
    """Return dict: rec -> dict(X=(P,F,14,7) array, t=target index)."""
    d = np.load(CACHE / f"raw_features_{duration}.npz")
    meta = json.load(open(CACHE / f"raw_features_{duration}.json", encoding="utf8"))
    tgt = {r["group"]: r["target_cell"] - 1 for r in meta["records"]}
    data = {}
    for g in RECORDS:
        X = np.stack([d[f"g{g:02d}_{p}"] for p in POLS]).astype(np.float64)  # (4, F, 14, 7)
        data[g] = dict(X=X, t=tgt[g])
    return data


def frame_normalize(X):
    """Per-frame robust normalization over all 14 cells (deployable: no target knowledge).

    X: (..., 14, K). Returns (x - median) / IQR computed across the cell axis per frame and feature.
    """
    med = np.median(X, axis=-2, keepdims=True)
    q75, q25 = np.percentile(X, [75, 25], axis=-2, keepdims=True)
    return (X - med) / (q75 - q25 + 1e-9)


def masks(t, ncell=14):
    c = np.arange(ncell)
    h0 = c != t
    pure = np.abs(c - t) > GUARD
    sec = (np.abs(c - t) <= GUARD) & (c != t)
    return h0, pure, sec


def loro_split(k):
    """TAES fold convention: test record k, validation k+1 (record 9 for k=10), train the rest."""
    val = k + 1 if k < 10 else 9
    train = [r for r in RECORDS if r not in (k, val)]
    return train, val, k


def calibrate(h0_scores, alpha):
    """Smallest threshold whose H0 activation rate (score >= tau) does not exceed alpha."""
    s = np.sort(np.asarray(h0_scores).ravel())[::-1]
    n = s.size
    allowed = int(np.floor(alpha * n))
    if allowed >= n:
        return -np.inf
    # tau must exceed the (allowed+1)-th largest score; keep ties together
    tau = np.nextafter(s[allowed], np.inf)
    return tau


def pd_at_pfa(h1, h0, pfa):
    """Oracle Pd at a realized false-alarm rate on the same (test) data."""
    tau = calibrate(h0, pfa)
    return float(np.mean(h1 >= tau))


def auc(pos, neg):
    pos = np.asarray(pos).ravel(); neg = np.asarray(neg).ravel()
    from scipy.stats import rankdata
    ranks = rankdata(np.concatenate([pos, neg]))
    rp = ranks[: pos.size].sum()
    return float((rp - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))
