"""A1 + A5: per-feature cross-record drift and single-feature threshold transfer.

For every feature, representation (raw / per-frame normalized) and record:
  - clutter median, IQR (pure-clutter cells), target-vs-clutter AUC, target-vs-secondary AUC
For every LORO fold (val record calibrates, test record evaluates), single-feature detector with the
pooled training-record sign: realized test Pfa at alpha=1e-2, Pd at tau, oracle Pd at test Pfa=1e-2.
"""
import numpy as np
import pandas as pd

from common import FEATURES, OUT, RECORDS, auc, calibrate, frame_normalize, load, loro_split, masks, pd_at_pfa

ALPHA = 1e-2


def cells(rec, Z):
    """Split a record's (P,F,14,K) array into H1, H0, pure, secondary samples of shape (n, K)."""
    t = rec["t"]
    h0, pure, sec = masks(t)
    return dict(h1=Z[:, :, t, :].reshape(-1, Z.shape[-1]),
                h0=Z[:, :, h0, :].reshape(-1, Z.shape[-1]),
                pure=Z[:, :, pure, :].reshape(-1, Z.shape[-1]),
                sec=Z[:, :, sec, :].reshape(-1, Z.shape[-1]))


def main():
    data = load()
    reps = {"raw": {g: data[g]["X"] for g in RECORDS},
            "framenorm": {g: frame_normalize(data[g]["X"]) for g in RECORDS}}
    rows, frows = [], []
    for rep, Zs in reps.items():
        S = {g: cells(data[g], Zs[g]) for g in RECORDS}
        for j, f in enumerate(FEATURES):
            for g in RECORDS:
                c = S[g]["pure"][:, j]
                q75, q25 = np.percentile(c, [75, 25])
                rows.append(dict(rep=rep, feature=f, record=g, clutter_median=np.median(c), clutter_iqr=q75 - q25,
                                 clutter_p99=np.percentile(c, 99),
                                 auc_tgt_vs_clutter=auc(S[g]["h1"][:, j], c),
                                 auc_tgt_vs_secondary=auc(S[g]["h1"][:, j], S[g]["sec"][:, j])))
            for k in RECORDS:
                train, val, test = loro_split(k)
                h1tr = np.concatenate([S[g]["h1"][:, j] for g in train])
                h0tr = np.concatenate([S[g]["h0"][:, j] for g in train])
                sign = 1.0 if np.median(h1tr) >= np.median(h0tr) else -1.0
                tau = calibrate(sign * S[val]["h0"][:, j], ALPHA)
                h1, h0, pure = (sign * S[test][x][:, j] for x in ("h1", "h0", "pure"))
                frows.append(dict(rep=rep, feature=f, fold=k, sign=sign,
                                  test_pfa=float(np.mean(h0 >= tau)), test_pfa_pure=float(np.mean(pure >= tau)),
                                  pd_at_tau=float(np.mean(h1 >= tau)), pd_oracle=pd_at_pfa(h1, h0, ALPHA),
                                  test_auc=auc(h1, h0)))
    rec = pd.DataFrame(rows); fold = pd.DataFrame(frows)
    rec.to_csv(OUT / "A1_feature_record_stats.csv", index=False)
    fold.to_csv(OUT / "A1_single_feature_loro.csv", index=False)

    # Drift index: between-record SD of clutter median / mean within-record IQR
    summ = rec.groupby(["rep", "feature"]).apply(lambda d: pd.Series(dict(
        drift_index=d.clutter_median.std(ddof=1) / d.clutter_iqr.mean(),
        auc_tc_mean=d.auc_tgt_vs_clutter.mean(), auc_tc_min=d.auc_tgt_vs_clutter.min(),
        auc_tc_max=d.auc_tgt_vs_clutter.max(),
        auc_ts_mean=d.auc_tgt_vs_secondary.mean()))).reset_index()
    fs = fold.groupby(["rep", "feature"]).agg(
        test_pfa_mean=("test_pfa", "mean"), pfa_gt_2a=("test_pfa", lambda x: int((x > 2 * ALPHA).sum())),
        log_pfa_err=("test_pfa", lambda x: float(np.mean(np.abs(np.log10(np.clip(x, 1e-5, 1) / ALPHA))))),
        pd_at_tau=("pd_at_tau", "mean"), pd_oracle=("pd_oracle", "mean"), test_auc=("test_auc", "mean")).reset_index()
    out = summ.merge(fs, on=["rep", "feature"])
    out.to_csv(OUT / "A1_summary.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.precision", 3)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
