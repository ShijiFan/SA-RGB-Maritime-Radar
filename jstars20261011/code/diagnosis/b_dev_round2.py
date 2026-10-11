"""Phase B development screen, round 2 (IPIX1993 LORO only; IPIX1998 untouched).

Rebases the context variants on per-frame normalization (B1, the best round-1 base) and compares two threshold
rules on identical scores:
  fixed   tau calibrated on the validation record's H0 cells, frozen (as before)
  cmap    causal clutter-map CFAR on the score output: for frame n of a test sequence, the reference set is the
          scores of all cells except the frame's arg-max cell over the previous W frames of the same sequence; the
          threshold is that set's q-quantile. q is calibrated on the validation record (same causal rule) so that the
          validation H0 activation rate equals alpha. No test labels are used.
Configs: C1 = B1 + absolute normalized neighbours, C2 = B1 + relative context, C3 = C2 + frame clutter state.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from a2_subset_np import GROUPS
from b_dev_modules import cn_frame, neighbours
from common import FEATURES, OUT, RECORDS, calibrate, load, loro_split, masks, pd_at_pfa

ALPHAS = (1e-2, 1e-3)
C = np.arange(14)
W = 64  # clutter-map memory in frames (~33 s at 0.512 s per frame)


def build(cfg, X):
    Z = cn_frame(X)
    if cfg == "B1":
        return Z
    Nb = neighbours(Z)
    if cfg == "C1":
        return np.concatenate([Z, Nb.reshape(*Z.shape[:3], -1)], axis=-1)
    R = np.concatenate([Z, (Z[..., None] - Nb).reshape(*Z.shape[:3], -1), Z - np.nanmax(Nb, axis=-1)], axis=-1)
    if cfg == "C2":
        return R
    med = np.median(X, axis=-2, keepdims=True); iqr = np.subtract(*np.percentile(X, [75, 25], axis=-2, keepdims=True))
    return np.concatenate([R, np.broadcast_to(np.concatenate([med, iqr], -1), (*X.shape[:3], 2 * X.shape[-1]))], -1)


def cmap_thresholds(S, q):
    """S (P,F,14) scores -> per-frame thresholds (P,F) from the previous W frames' non-argmax cells."""
    P, F, _ = S.shape
    ref = np.where(C[None, None] == S.argmax(-1)[..., None], np.nan, S)  # drop arg-max cell
    tau = np.full((P, F), np.inf)
    for f in range(1, F):
        R = ref[:, max(0, f - W):f].reshape(P, -1)
        tau[:, f] = np.nanquantile(R, q, axis=1)
    return tau


def calibrate_q(S, t, alpha):
    """Pick q on the validation record so that the H0 activation rate under the causal rule is <= alpha."""
    h0m = masks(t)[0]
    lo, hi = 0.5, 1.0
    for _ in range(30):
        q = (lo + hi) / 2
        tau = cmap_thresholds(S, q)
        rate = np.mean(S[:, :, h0m] >= tau[..., None])
        lo, hi = (lo, q) if rate <= alpha else (q, hi)
    return hi


def main():
    data = load()
    rows = []
    for cfg in ("B1", "C1", "C2", "C3"):
        for group in ("G3", "G7"):
            cols = [FEATURES.index(f) for f in GROUPS[group]]
            Z = {g: build(cfg, data[g]["X"][..., cols]) for g in RECORDS}
            for k in RECORDS:
                train, val, test = loro_split(k)
                Xtr = np.concatenate([Z[g].reshape(-1, Z[g].shape[-1]) for g in train])
                ytr = np.concatenate([np.broadcast_to(C == data[g]["t"], Z[g].shape[:3]).ravel() for g in train])
                m = HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=0).fit(Xtr, ytr)
                sc = lambda g: np.log(np.clip(m.predict_proba(Z[g].reshape(-1, Z[g].shape[-1]))[:, 1], 1e-12, 1)).reshape(Z[g].shape[:3])
                sv, st = sc(val), sc(test)
                tv, t = data[val]["t"], data[test]["t"]
                h0m, adjm = masks(t)[0], np.abs(C - t) == 1
                for a in ALPHAS:
                    rules = {"fixed": np.full(st.shape[:2], calibrate(sv[:, :, masks(tv)[0]], a))}
                    rules["cmap"] = cmap_thresholds(st, calibrate_q(sv, tv, a))
                    for rule, tau in rules.items():
                        ok = np.isfinite(tau)  # first frame of each sequence has no history
                        T = tau[..., None]
                        h1 = (st[:, :, t] >= tau)[ok]; h0 = (st[:, :, h0m] >= T)[ok]; adj = (st[:, :, adjm] >= T)[ok]
                        rows.append(dict(cfg=cfg, group=group, fold=k, alpha=a, rule=rule, pd=h1.mean(), pfa=h0.mean(),
                                         pfa_adj=adj.mean(),
                                         pdor=pd_at_pfa(st[:, :, t].ravel(), st[:, :, h0m].ravel(), a)))
                print(cfg, group, k, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "B_round2_folds.csv", index=False)
    s = df.groupby(["alpha", "rule", "cfg", "group"]).agg(
        pd=("pd", "mean"), pfa=("pfa", "mean"), over2=("pfa", lambda x: None), adj=("pfa_adj", "mean"),
        pdor=("pdor", "mean")).reset_index()
    s["over2"] = df.groupby(["alpha", "rule", "cfg", "group"]).apply(
        lambda d: int((d.pfa > 2 * d.alpha).sum())).values
    s["logerr"] = df.groupby(["alpha", "rule", "cfg", "group"]).apply(
        lambda d: float(np.mean(np.abs(np.log10(np.clip(d.pfa, 1e-5, 1) / d.alpha))))).values
    s.to_csv(OUT / "B_round2_summary.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.precision", 4)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
