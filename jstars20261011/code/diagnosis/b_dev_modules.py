"""Phase B development screen: candidate modules as transformations for position-free per-cell scorers.

Development data: IPIX1993 LORO only (tau calibrated on the validation record, frozen for the test record).
IPIX1998 is NOT used here; it is reserved for the pre-registered confirmation.

Configurations (cumulative where noted):
  B0 raw          raw features of the cell under test (CUT)
  B1 cn_frame     per-frame clutter normalization over all 14 cells (median / IQR)
  B2 cn_cut       CFAR-style normalization: reference = cells with |j-c| > GUARD (guard excludes leakage cells)
  B3 cn_cut+abs   B2 + normalized values of the +-2 neighbours (absolute context, NaN outside grid)
  B4 cn_cut+rel   B2 + relative context: CUT minus each neighbour, and CUT minus max(neighbours) (peak margin)
  B5 B4+state     B4 + frame clutter state (raw per-feature median and IQR over the frame) -> reliability gating proxy
  B6 B4+ocfar     B4 with output CFAR: logit score minus median logit of the CUT's reference cells
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from a2_subset_np import GROUPS
from common import FEATURES, GUARD, OUT, RECORDS, calibrate, load, loro_split, masks, pd_at_pfa

ALPHAS = (1e-2, 1e-3)
C = np.arange(14)
REF = np.abs(C[:, None] - C[None, :]) > GUARD  # REF[c, j]: j is a reference cell for CUT c


def cn_frame(X):
    med = np.median(X, axis=-2, keepdims=True)
    iqr = np.subtract(*np.percentile(X, [75, 25], axis=-2, keepdims=True))
    return (X - med) / (iqr + 1e-9)


def cn_cut(X):
    """X (P,F,14,K) -> CFAR-normalized (P,F,14,K) with a per-CUT guarded reference set."""
    out = np.empty_like(X)
    for c in range(14):
        R = X[:, :, REF[c], :]
        med = np.median(R, axis=-2); iqr = np.subtract(*np.percentile(R, [75, 25], axis=-2))
        out[:, :, c, :] = (X[:, :, c, :] - med) / (iqr + 1e-9)
    return out


def neighbours(Z, offsets=(-2, -1, 1, 2)):
    """Stack neighbour values (NaN outside the grid): (P,F,14,K,len(offsets))."""
    P, F, N, K = Z.shape
    out = np.full((P, F, N, K, len(offsets)), np.nan)
    for i, o in enumerate(offsets):
        src = C + o; ok = (src >= 0) & (src < N)
        view = out[..., i]  # basic index -> view, keeps axis order
        view[:, :, ok, :] = Z[:, :, src[ok], :]
    return out


def build(cfg, X):
    if cfg == "B0":
        return X
    if cfg == "B1":
        return cn_frame(X)
    Z = cn_cut(X)
    if cfg == "B2":
        return Z
    Nb = neighbours(Z)
    if cfg == "B3":
        return np.concatenate([Z, Nb.reshape(*Z.shape[:3], -1)], axis=-1)
    rel = Z[..., None] - Nb
    margin = Z - np.nanmax(Nb, axis=-1)
    R = np.concatenate([Z, rel.reshape(*Z.shape[:3], -1), margin], axis=-1)
    if cfg in ("B4", "B6"):
        return R
    if cfg == "B5":
        med = np.median(X, axis=-2, keepdims=True); iqr = np.subtract(*np.percentile(X, [75, 25], axis=-2, keepdims=True))
        state = np.broadcast_to(np.concatenate([med, iqr], axis=-1), (*X.shape[:3], 2 * X.shape[-1]))
        return np.concatenate([R, state], axis=-1)
    raise ValueError(cfg)


def output_cfar(logit):
    """logit (P,F,14) -> logit minus median of the CUT's guarded reference cells."""
    out = np.empty_like(logit)
    for c in range(14):
        out[..., c] = logit[..., c] - np.median(logit[..., REF[c]], axis=-1)
    return out


def models():
    return {"HGB": lambda: HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=0),
            "LR": lambda: LogisticRegression(class_weight="balanced", max_iter=3000)}


def run(cfg, group, mname, data):
    cols = [FEATURES.index(f) for f in GROUPS[group]]
    Z = {g: build(cfg, data[g]["X"][..., cols]) for g in RECORDS}
    rows = []
    for k in RECORDS:
        train, val, test = loro_split(k)
        Xtr = np.concatenate([Z[g].reshape(-1, Z[g].shape[-1]) for g in train])
        ytr = np.concatenate([np.broadcast_to(C == data[g]["t"], Z[g].shape[:3]).ravel() for g in train])
        if mname == "LR":
            mu, sd = np.nanmean(Xtr, 0), np.nanstd(Xtr, 0) + 1e-9
            prep = lambda A: np.nan_to_num((A - mu) / sd)
        else:
            prep = lambda A: A
        m = models()[mname]().fit(prep(Xtr), ytr)

        def score(g):
            A = Z[g].reshape(-1, Z[g].shape[-1])
            p = np.clip(m.predict_proba(prep(A))[:, 1], 1e-12, 1 - 1e-12)
            s = np.log(p / (1 - p)).reshape(Z[g].shape[:3])
            return output_cfar(s) if cfg == "B6" else s

        sv, st = score(val), score(test)
        t = data[test]["t"]; h0m = masks(t)[0]; adjm = np.abs(C - t) == 1
        h1, h0, adj = st[:, :, t].ravel(), st[:, :, h0m].ravel(), st[:, :, adjm].ravel()
        row = dict(cfg=cfg, group=group, model=mname, fold=k, exact_cell=float(np.mean(st.argmax(-1) == t)))
        for a in ALPHAS:
            tau = calibrate(sv[:, :, masks(data[val]["t"])[0]], a)
            row.update({f"pfa_{a:g}": float(np.mean(h0 >= tau)), f"pfa_adj_{a:g}": float(np.mean(adj >= tau)),
                        f"pd_{a:g}": float(np.mean(h1 >= tau)), f"pdor_{a:g}": pd_at_pfa(h1, h0, a)})
        rows.append(row)
    return rows


def main():
    data = load()
    cfgs = sys.argv[1:] or ["B0", "B1", "B2", "B3", "B4", "B5", "B6"]
    rows = []
    for cfg in cfgs:
        for group in ("G3", "G7"):
            for mname in ("HGB", "LR"):
                rows += run(cfg, group, mname, data)
                print(cfg, group, mname, "done", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "B_dev_folds.csv", index=False)
    summarize(pd.DataFrame(rows))


def summarize(df):
    g = df.groupby(["cfg", "group", "model"])
    s = g.agg(pdor_1e2=("pdor_0.01", "mean"), pd_1e2=("pd_0.01", "mean"), pfa_1e2=("pfa_0.01", "mean"),
              over2_1e2=("pfa_0.01", lambda x: int((x > 0.02).sum())), adj_1e2=("pfa_adj_0.01", "mean"),
              pdor_1e3=("pdor_0.001", "mean"), pd_1e3=("pd_0.001", "mean"), pfa_1e3=("pfa_0.001", "mean"),
              over2_1e3=("pfa_0.001", lambda x: int((x > 0.002).sum())), exact=("exact_cell", "mean"),
              pd_f1=("pd_0.01", lambda x: x.iloc[0]), pd_f6=("pd_0.01", lambda x: x.iloc[5])).reset_index()
    s.to_csv(OUT / "B_dev_summary.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.precision", 3)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
