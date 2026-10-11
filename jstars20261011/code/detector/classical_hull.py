"""Reference (not pre-registered): tri-feature convex-hull detector in the spirit of Shui et al. (2014) under the
same record-disjoint protocol.

RAA/RDPH/RVE of each cell, standardized with training-record H0 statistics. Hull layers are peeled from a 20,000-point
subsample of training H0 cells; the deepest layer index L whose hull keeps the validation H0 activation (fraction
outside the hull) at or below alpha is frozen and applied to the test record. A cell is declared a target when its
feature vector lies outside the hull. IPIX1993 LORO and the IPIX1993 -> IPIX1998 cross-site protocol.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phaseA_20261008"))
from common import POLS, RECORDS, load, loro_split  # noqa: E402

COLS = [1, 2, 3]  # RAA, RDPH, RVE in cache order
C = np.arange(14)


def peel_depth(P):
    depth = np.full(len(P), -1); idx = np.arange(len(P)); d = 0
    while len(idx) > 10:
        h = ConvexHull(P[idx]); v = idx[h.vertices]; depth[v] = d; idx = np.setdiff1d(idx, v); d += 1
    depth[idx] = d
    return depth


def outside(P, depth, L, X):
    return Delaunay(P[depth >= L]).find_simplex(X) < 0


def fit_rule(Htr, Hval, alpha, rng):
    mu, sd = Htr.mean(0), Htr.std(0)
    P = (Htr[rng.choice(len(Htr), min(20000, len(Htr)), replace=False)] - mu) / sd
    depth = peel_depth(P); V = (Hval - mu) / sd
    lo, hi = 0, depth.max()  # activation increases with L; keep the largest L with activation <= alpha
    v0 = outside(P, depth, 0, V).mean()
    if v0 > alpha:  # no hull layer meets the budget on validation: report as infeasible, keep the outermost layer
        return dict(rule=(mu, sd, P, depth, 0), feasible=False, val_pfa=float(v0), layer=0, alpha=alpha)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if outside(P, depth, mid, V).mean() <= alpha else (lo, mid - 1)
    return dict(rule=(mu, sd, P, depth, lo), feasible=True, val_pfa=float(outside(P, depth, lo, V).mean()), layer=int(lo),
                alpha=alpha)


def evaluate(fit, X, t):
    """X (n_frames, 14, 3) test features, t (n_frames,) target index."""
    mu, sd, P, depth, L = fit["rule"]
    out = outside(P, depth, L, ((X - mu) / sd).reshape(-1, 3)).reshape(X.shape[:2])
    h1 = out[np.arange(len(t)), t]; h0 = out[C[None] != t[:, None]]
    adj = out[np.abs(C[None] - t[:, None]) == 1]
    return dict(pd=float(h1.mean()), pfa=float(h0.mean()), pfa_adj=float(adj.mean()), layer=int(L),
                feasible=fit["feasible"], val_pfa=fit["val_pfa"])


def h0(X, t):
    return X[:, C != t].reshape(-1, 3)


def main():
    data = load(); rng = np.random.default_rng(0); rows = []
    feat = {g: data[g]["X"][..., COLS].reshape(-1, 14, 3) for g in RECORDS}
    for a in (1e-2, 1e-3):
        for k in RECORDS:
            train, val, test = loro_split(k)
            rule = fit_rule(np.concatenate([h0(feat[g], data[g]["t"]) for g in train]), h0(feat[val], data[val]["t"]), a, rng)
            r = evaluate(rule, feat[test], np.full(len(feat[test]), data[test]["t"]))
            rows.append(dict(protocol="loro", alpha=a, fold=k, **r)); print("loro", a, k, r, flush=True)
        # cross-site
        TR, VA = 181, 45
        tr = np.concatenate([h0(data[g]["X"][:, :TR, :, COLS].reshape(-1, 14, 3), data[g]["t"]) for g in RECORDS])
        va = np.concatenate([h0(data[g]["X"][:, TR:TR + VA, :, COLS].reshape(-1, 14, 3), data[g]["t"]) for g in RECORDS])
        rule = fit_rule(tr, va, a, rng)
        d98 = np.load(HERE.parent / "phaseA_20261008" / "out" / "ipix1998_7feat.npz")
        meta = json.load(open(HERE.parent / "phaseA_20261008" / "out" / "ipix1998_7feat.json", encoding="utf8"))
        Xs, Ts = [], []
        for rr in meta["records"]:
            for p in POLS:
                key = f"r{rr['rec']:02d}_{p}"
                for j, s in enumerate(d98[key + "_starts"]):
                    Xs.append(d98[key + "_win"][:, j][..., COLS].astype(np.float64)); Ts.append(np.full(117, rr["target_cell"] - 1 - int(s)))
        r = evaluate(rule, np.concatenate(Xs), np.concatenate(Ts))
        rows.append(dict(protocol="1998", alpha=a, fold=0, **r)); print("1998", a, r, flush=True)
    df = pd.DataFrame(rows); df.to_csv(HERE / "C_classical_hull.csv", index=False)
    lo = df[df.protocol == "loro"]
    print(lo.groupby("alpha").agg(pd=("pd", "mean"), pfa=("pfa", "mean"), adj=("pfa_adj", "mean"),
                                  over2=("pfa", lambda x: None)).to_string())
    for a in (1e-2, 1e-3):
        d = lo[lo.alpha == a]
        print(a, "infeasible folds", list(d[~d.feasible].fold), "feasible-only Pd", round(d[d.feasible].pd.mean(), 3),
              "Pfa/a", round(d[d.feasible].pfa.mean() / a, 2))
        print(a, "over2", int((d.pfa > 2 * a).sum()), "/10  logerr", round(np.mean(np.abs(np.log10(np.clip(d.pfa, 1e-5, 1) / a))), 3))


if __name__ == "__main__":
    main()
