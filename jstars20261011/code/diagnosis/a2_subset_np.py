"""A2: information vs false-alarm controllability as a function of feature subset (LORO, no deep nets).

Per-cell shared (position-free) scorers trained on the 8 training records, threshold calibrated on the
validation record's H0 cells, frozen for the test record. Run for all 127 subsets of the 7 features with
logistic regression, and for the named groups G3/G4/G5/G7 with a nonlinear gradient-boosting scorer.
Context variant: each cell additionally sees its two range neighbours on each side (edge-replicated).
"""
import itertools
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import FEATURES, OUT, RECORDS, calibrate, frame_normalize, load, loro_split, masks, pd_at_pfa

ALPHAS = (1e-2, 1e-3)
GROUPS = {"G3": ["RAA", "RDPH", "RVE"], "G4": ["RAA", "RDPH", "RVE", "MCS"],
          "G5": ["RAA", "RDPH", "RVE", "Hurst", "RI"], "G7": FEATURES}


def with_context(Z, radius=2):
    """(P,F,14,K) -> (P,F,14,K*(2r+1)) by concatenating range neighbours (edge-replicated)."""
    idx = np.arange(14)
    parts = [Z[:, :, np.clip(idx + o, 0, 13), :] for o in range(-radius, radius + 1)]
    return np.concatenate(parts, axis=-1)


def record_arrays(rec, Z):
    t = rec["t"]
    h0, _, _ = masks(t)
    P, F = Z.shape[:2]
    y = np.zeros((P, F, 14), bool); y[:, :, t] = True
    return dict(Z=Z, y=y, h0=h0, t=t)


def evaluate(scores, R, alpha_tau):
    """scores: (P,F,14). Returns metric dict for one test record."""
    t, h0 = R["t"], R["h0"]
    h1s, h0s = scores[:, :, t].ravel(), scores[:, :, h0].ravel()
    out = dict(exact_cell=float(np.mean(scores.argmax(-1) == t)))
    for a, tau in alpha_tau.items():
        out[f"pfa_{a:g}"] = float(np.mean(h0s >= tau))
        out[f"pd_{a:g}"] = float(np.mean(h1s >= tau))
        out[f"pdor_{a:g}"] = pd_at_pfa(h1s, h0s, a)
    return out


def run(model_name, make_model, feat_cols, Zs, data, ctx):
    rows = []
    for k in RECORDS:
        train, val, test = loro_split(k)
        Rs = {g: record_arrays(data[g], Zs[g][..., feat_cols] if not ctx else with_context(Zs[g][..., feat_cols]))
              for g in (*train, val, test)}
        Xtr = np.concatenate([Rs[g]["Z"].reshape(-1, Rs[g]["Z"].shape[-1]) for g in train])
        ytr = np.concatenate([Rs[g]["y"].ravel() for g in train])
        sc = StandardScaler().fit(Xtr)
        m = make_model().fit(sc.transform(Xtr), ytr)

        def score(g):
            Z = Rs[g]["Z"]
            return m.decision_function(sc.transform(Z.reshape(-1, Z.shape[-1]))).reshape(Z.shape[:3]) \
                if hasattr(m, "decision_function") else \
                m.predict_proba(sc.transform(Z.reshape(-1, Z.shape[-1])))[:, 1].reshape(Z.shape[:3])

        sv = score(val)
        taus = {a: calibrate(sv[:, :, Rs[val]["h0"]], a) for a in ALPHAS}
        rows.append(dict(fold=k, **evaluate(score(test), Rs[test], taus)))
    return rows


def main(quick=False):
    data = load()
    reps = {"raw": {g: data[g]["X"] for g in RECORDS},
            "framenorm": {g: frame_normalize(data[g]["X"]) for g in RECORDS}}
    lr = lambda: LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    hgb = lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                                 class_weight="balanced", random_state=0)
    rows = []
    subsets = [s for n in range(1, 8) for s in itertools.combinations(range(7), n)]
    for rep, Zs in reps.items():
        for ctx in (False, True):
            for s in (subsets if not quick else [tuple(range(7))]):
                name = "+".join(FEATURES[i] for i in s)
                for r in run("LR", lr, list(s), Zs, data, ctx):
                    rows.append(dict(rep=rep, ctx=ctx, model="LR", subset=name, k=len(s), **r))
            for gname, feats in GROUPS.items():
                cols = [FEATURES.index(f) for f in feats]
                for r in run("HGB", hgb, cols, Zs, data, ctx):
                    rows.append(dict(rep=rep, ctx=ctx, model="HGB", subset=gname, k=len(cols), **r))
            print(rep, ctx, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "A2_subset_loro_folds.csv", index=False)


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
