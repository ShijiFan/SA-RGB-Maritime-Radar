"""D1: temporal-window and observation-duration diagnosis with a position-free HGB scorer (see D_PLAN.md)."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phaseA_20261008"))
from common import RECORDS, calibrate, frame_normalize, load, loro_split, pd_at_pfa  # noqa: E402

DURS = {"0p128": 0.128, "0p256": 0.256, "0p512": 0.512, "1p024": 1.024}
KS = (1, 2, 4, 8)
T0 = 8.192  # evaluated frames end at or after this time for every configuration
C = np.arange(14)


def base_features(X):
    """X (P,F,14,7) raw -> (P,F,14,14): frame-normalized descriptors and +-1 range peak margin."""
    Z = frame_normalize(X)
    pad = np.full(Z.shape[:2] + (1, 7), -np.inf)
    Zp = np.concatenate([pad, Z, pad], axis=2)
    nb = np.maximum(Zp[:, :, :-2], Zp[:, :, 2:])
    return np.concatenate([Z, Z - nb], axis=-1)


def temporal(B, K):
    """Causal window stats over K frames; output aligned to frames K-1..F-1 (P,F-K+1,14,D)."""
    if K == 1:
        return B
    W = sliding_window_view(B, K, axis=1)  # (P, F-K+1, 14, 14, K)
    return np.concatenate([W.mean(-1), W.std(-1), W.min(-1)], axis=-1)


def run():
    rows = []
    for tag, d in DURS.items():
        data = load(tag)
        base = {g: base_features(v["X"]) for g, v in data.items()}
        n0 = int(np.ceil(T0 / d)) - 1  # first evaluated frame index
        for K in KS:
            feats = {g: temporal(B, K)[:, n0 - (K - 1):] for g, B in base.items()}
            for k in RECORDS:
                tr, va, te = loro_split(k)
                Xtr = np.concatenate([feats[g].reshape(-1, feats[g].shape[-1]) for g in tr])
                ytr = np.concatenate([np.broadcast_to(C == data[g]["t"], feats[g].shape[:3]).ravel() for g in tr])
                t0 = time.time()
                m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                                   class_weight="balanced", random_state=0).fit(Xtr, ytr)
                sc = lambda g: np.log(np.clip(m.predict_proba(feats[g].reshape(-1, feats[g].shape[-1]))[:, 1], 1e-12, 1)).reshape(feats[g].shape[:3])
                Sv, St = sc(va), sc(te)
                tv, tt = data[va]["t"], data[te]["t"]
                h1, h0 = St[..., tt].ravel(), St[..., C != tt].ravel()
                row = dict(dur=d, K=K, dwell=round(d * K, 3), fold=k, n_frames=St.shape[0] * St.shape[1], dim=feats[te].shape[-1])
                for a in (1e-2, 1e-3):
                    row[f"pdor_{a}"] = pd_at_pfa(h1, h0, a)
                    tau = calibrate(Sv[..., C != tv], a)
                    row[f"pd_fix_{a}"] = float((h1 >= tau).mean())
                    row[f"pfa_fix_{a}"] = float((h0 >= tau).mean())
                row["exact"] = float((St.argmax(-1) == tt).mean())
                rows.append(row)
                print(f"d={d} K={K} fold={k} pdor={row['pdor_0.01']:.3f} pfa_fix={row['pfa_fix_0.01']:.4f} ({time.time() - t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(HERE / "D1_temporal.csv", index=False)


if __name__ == "__main__":
    run()
