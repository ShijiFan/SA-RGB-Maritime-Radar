"""D2: temporal features at 0.512 s under LORO and IPIX1993 -> IPIX1998, fixed and tail-CFAR rules (see D_PLAN.md)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phaseA_20261008"))
sys.path.insert(0, str(HERE.parent / "phaseC"))
from baselines_ref import evaluate  # noqa: E402
from common import POLS, RECORDS, calibrate, load, loro_split, pd_at_pfa  # noqa: E402
from d1_temporal import base_features, temporal  # noqa: E402

KS = (1, 2, 4, 8)
C = np.arange(14)
N0 = 15          # LORO: first evaluated 0.512-s frame (8.192 s), as in D1
N0_98 = 7        # IPIX1998: first evaluated frame for every K
TR, VA = 181, 45


def fit(slices):
    Xtr = np.concatenate([z.reshape(-1, z.shape[-1]) for z, _ in slices])
    ytr = np.concatenate([np.broadcast_to(C == t, z.shape[:3]).ravel() for z, t in slices])
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                          class_weight="balanced", random_state=0).fit(Xtr, ytr)


def logit(m, z):
    p = np.clip(m.predict_proba(z.reshape(-1, z.shape[-1]))[:, 1], 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p)).reshape(z.shape[:3])


def windows_1998_base():
    d = np.load(HERE.parent / "phaseA_20261008" / "out" / "ipix1998_7feat.npz")
    meta = json.load(open(HERE.parent / "phaseA_20261008" / "out" / "ipix1998_7feat.json", encoding="utf8"))
    X, T, R = [], [], []
    for r in meta["records"]:
        for p in POLS:
            k = f"r{r['rec']:02d}_{p}"
            for j, s in enumerate(d[k + "_starts"]):
                X.append(d[k + "_win"][:, j].astype(np.float64)); T.append(r["target_cell"] - 1 - int(s)); R.append(r["rec"])
    return base_features(np.stack(X)), np.array(T), np.array(R)


def main():
    data = load("0p512")
    base = {g: base_features(v["X"]) for g, v in data.items()}
    B98, T98, R98 = windows_1998_base()
    rows = []
    for K in KS:
        feats = {g: temporal(B, K) for g, B in base.items()}       # aligned to frames K-1..
        sl = lambda g, a, b: feats[g][:, a - (K - 1):b - (K - 1)]   # frames a..b-1
        F = base[1].shape[1]
        # ---- LORO
        for k in RECORDS:
            tr, va, te = loro_split(k)
            m = fit([(sl(g, N0, F), data[g]["t"]) for g in tr])
            Sv, St = logit(m, sl(va, N0, F)), logit(m, sl(te, N0, F))
            for (rule, a), v in evaluate(Sv, np.full(4, data[va]["t"]), St, np.full(4, data[te]["t"])).items():
                rows.append(dict(protocol="loro", K=K, fold=k, rule=rule, alpha=a, **v))
        # ---- IPIX1993 -> IPIX1998
        m = fit([(sl(g, max(K - 1, 0), TR), data[g]["t"]) for g in RECORDS])
        Sv = np.concatenate([logit(m, sl(g, TR, TR + VA)) for g in RECORDS])
        Tv = np.repeat([data[g]["t"] for g in RECORDS], 4)
        F98 = temporal(B98, K)[:, N0_98 - (K - 1):]
        St = logit(m, F98)
        for (rule, a), v in evaluate(Sv, Tv, St, T98).items():
            rows.append(dict(protocol="1998", K=K, fold=0, rule=rule, alpha=a, **v))
        h0m = C[None, None] != T98[:, None, None]
        h1 = np.take_along_axis(St, np.broadcast_to(T98[:, None, None], St.shape[:2] + (1,)), 2)[..., 0]
        for r in np.unique(R98):
            s = R98 == r
            rows.append(dict(protocol="1998rec", K=K, fold=int(r), rule="oracle", alpha=1e-2,
                             pd=pd_at_pfa(h1[s].ravel(), St[s][np.broadcast_to(h0m[s], St[s].shape)], 1e-2)))
        print(f"K={K} done", flush=True)
        pd.DataFrame(rows).to_csv(HERE / "D2_crosssite.csv", index=False)


if __name__ == "__main__":
    main()
