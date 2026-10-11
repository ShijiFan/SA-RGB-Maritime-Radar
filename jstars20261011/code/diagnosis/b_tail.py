"""Phase B, tail extrapolation for alpha = 1e-3 (IPIX1993 LORO only; IPIX1998 untouched).

Scores: HGB logit on C2 (per-frame normalization + relative context), G3 and G7. Threshold rules, all causal and
label-free on the test record:
  fixed        validation-record quantile at alpha (reference)
  cmap{W}      empirical clutter-map quantile over the previous W frames (non-argmax cells), q calibrated on val at alpha
  texp{W}      clutter-map exponential tail: u = 90th percentile of the reference scores, beta = mean excess above u,
               tau(a) = u + beta * ln(0.1 / (c*a)). The scale c is calibrated on the validation record at 1e-2 ONLY and
               reused at 1e-3 (pure extrapolation), so the 1e-3 operating point never sees a 1e-3 calibration.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from a2_subset_np import GROUPS
from b_dev_round2 import build
from common import FEATURES, OUT, RECORDS, calibrate, load, loro_split, masks

C = np.arange(14)


def reference(S, W):
    """Yield (f, ref (P, n)) with the previous W frames' non-argmax scores."""
    ref = np.where(C[None, None] == S.argmax(-1)[..., None], np.nan, S)
    for f in range(1, S.shape[1]):
        yield f, ref[:, max(0, f - W):f].reshape(S.shape[0], -1)


def cmap_tau(S, W, q):
    tau = np.full(S.shape[:2], np.inf)
    for f, R in reference(S, W):
        tau[:, f] = np.nanquantile(R, q, axis=1)
    return tau


def tail_params(S, W):
    """Per-frame (u, beta) of an exponential tail fitted to the top 10% of the reference scores."""
    u = np.full(S.shape[:2], np.nan); b = np.full(S.shape[:2], np.nan)
    for f, R in reference(S, W):
        uu = np.nanquantile(R, 0.9, axis=1)
        ex = np.where(R > uu[:, None], R - uu[:, None], np.nan)
        u[:, f], b[:, f] = uu, np.nanmean(ex, axis=1)
    return u, b


def texp_tau(u, b, a, c):
    return u + b * np.log(0.1 / (c * a))


def rate(S, tau, t):
    ok = np.isfinite(tau)
    return float(np.mean((S[:, :, masks(t)[0]] >= tau[..., None])[ok]))


def bisect(fun, lo, hi, target, n=40, log=False):
    """fun is increasing in x: return the largest x in [lo, hi] with fun(x) <= target (conservative side)."""
    for _ in range(n):
        mid = np.sqrt(lo * hi) if log else (lo + hi) / 2
        lo, hi = (mid, hi) if fun(mid) <= target else (lo, mid)
    return lo


def main():
    data = load()
    rows = []
    for group in ("G3", "G7"):
        cols = [FEATURES.index(f) for f in GROUPS[group]]
        Z = {g: build("C2", data[g]["X"][..., cols]) for g in RECORDS}
        for k in RECORDS:
            train, val, test = loro_split(k)
            Xtr = np.concatenate([Z[g].reshape(-1, Z[g].shape[-1]) for g in train])
            ytr = np.concatenate([np.broadcast_to(C == data[g]["t"], Z[g].shape[:3]).ravel() for g in train])
            m = HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=0).fit(Xtr, ytr)

            def sc(g):
                p = np.clip(m.predict_proba(Z[g].reshape(-1, Z[g].shape[-1]))[:, 1], 1e-12, 1 - 1e-12)
                return np.log(p / (1 - p)).reshape(Z[g].shape[:3])

            sv, st = sc(val), sc(test)
            tv, t = data[val]["t"], data[test]["t"]
            taus = {}
            for a in (1e-2, 1e-3):
                taus[("fixed", a)] = np.full(st.shape[:2], calibrate(sv[:, :, masks(tv)[0]], a))
                for W in (64, 256):
                    q = 1 - bisect(lambda x: rate(sv, cmap_tau(sv, W, 1 - x), tv), 1e-6, 0.5, a, n=25, log=True)
                    taus[(f"cmap{W}", a)] = cmap_tau(st, W, q)
            for W in (64, 256):
                uv, bv = tail_params(sv, W); ut, bt = tail_params(st, W)
                c = bisect(lambda x: rate(sv, texp_tau(uv, bv, 1e-2, x), tv), 1e-3, 1e3, 1e-2, log=True)
                for a in (1e-2, 1e-3):
                    taus[(f"texp{W}", a)] = texp_tau(ut, bt, a, c)
            h0m = masks(t)[0]
            for (rule, a), tau in taus.items():
                ok = np.isfinite(tau)
                rows.append(dict(group=group, fold=k, rule=rule, alpha=a,
                                 pd=float(np.mean((st[:, :, t] >= tau)[ok])),
                                 pfa=float(np.mean((st[:, :, h0m] >= tau[..., None])[ok]))))
            print(group, k, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "B_tail_folds.csv", index=False)
    df["over2"] = (df.pfa > 2 * df.alpha).astype(int)
    df["logerr"] = np.abs(np.log10(np.clip(df.pfa, 1e-5, 1) / df.alpha))
    s = df.groupby(["alpha", "rule", "group"]).agg(pd=("pd", "mean"), pfa=("pfa", "mean"), pfa_med=("pfa", "median"),
                                                   over2=("over2", "sum"), logerr=("logerr", "mean")).reset_index()
    s.to_csv(OUT / "B_tail_summary.csv", index=False)
    pd.set_option("display.width", 200); pd.set_option("display.precision", 4)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
