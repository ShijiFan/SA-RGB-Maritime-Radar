"""Pre-registered analysis of Phase C IPIX1993 LORO (E1, E2 and secondary endpoints)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs_loro"


def load():
    rows = []
    for p in RUNS.glob("*.json"):
        r = json.loads(p.read_text())
        base = dict(arm=r["arm"], fold=r["fold"], seed=r["seed"], exact=r["exact_cell"],
                    pdor_1e2=r["pdor_0.01"], pdor_1e3=r["pdor_0.001"])
        for rule in ("texp64", "fixed"):
            for a, tag in ((0.01, "1e2"), (0.001, "1e3")):
                m = r[f"{rule}_{a:g}"]
                rows.append(dict(base, rule=rule, alpha=a, pd=m["pd"], pfa=m["pfa"], pfa_adj=m["pfa_adj"]))
    return pd.DataFrame(rows)


def compare(df, a1, a2, rule, alpha, metric="pd"):
    d = df[(df.rule == rule) & (df.alpha == alpha)]
    f = d.groupby(["arm", "fold"])[metric].mean().unstack(0)
    x, y = f[a1].values, f[a2].values
    p = wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
    return dict(a1=a1, a2=a2, rule=rule, alpha=alpha, metric=metric, mean1=x.mean(), mean2=y.mean(),
                diff=(x - y).mean(), wins=int((x > y).sum()), losses=int((x < y).sum()), p=p)


def main():
    df = load()
    n = df.groupby("arm").seed.size() // 4
    print("runs per arm:", n.to_dict())
    df["over2"] = (df.pfa > 2 * df.alpha).astype(int)
    df["logerr"] = np.abs(np.log10(np.clip(df.pfa, 1e-5, 1) / df.alpha))
    summ = df.groupby(["alpha", "rule", "arm"]).agg(
        pd=("pd", "mean"), pfa=("pfa", "mean"), over2=("over2", "sum"), logerr=("logerr", "mean"),
        adj=("pfa_adj", "mean"), pdor1e2=("pdor_1e2", "mean"), pdor1e3=("pdor_1e3", "mean"),
        exact=("exact", "mean")).reset_index()
    pd.set_option("display.width", 250); pd.set_option("display.precision", 4)
    print(summ.to_string(index=False))

    out = {}
    e1 = compare(df, "G7", "G3", "texp64", 0.01)
    e1["verdict"] = ("SUPPORTED" if e1["wins"] >= 8 and e1["p"] < 0.05 else
                     "PARTIAL" if e1["diff"] > 0 and e1["p"] < 0.10 else "NOT SUPPORTED")
    out["E1"] = e1
    s = summ[(summ.alpha == 0.01) & (summ.rule == "texp64")].set_index("arm")
    e2 = dict(G7_over2=int(s.loc["G7", "over2"]), G3_over2=int(s.loc["G3", "over2"]),
              G7_logerr=float(s.loc["G7", "logerr"]), G3_logerr=float(s.loc["G3", "logerr"]))
    e2["verdict"] = "SUPPORTED" if e2["G7_over2"] <= 3 and e2["G7_over2"] - e2["G3_over2"] <= 2 else "NOT SUPPORTED"
    out["E2"] = e2
    sec = [compare(df, "G7", "G3", "texp64", 0.001)]
    for b in ("D7", "P7", "G7-noSCN", "G7-absC"):
        sec.append(compare(df, "G7", b, "texp64", 0.01))
    sec.append(compare(df, "G7", "G3", "texp64", 0.01, "pfa_adj"))
    # rule ablation on identical G7 scores: Pfa transfer error
    d = df[(df.arm == "G7") & (df.alpha == 0.01)]
    f = d.groupby(["rule", "fold"]).logerr.mean().unstack(0)
    sec.append(dict(a1="texp64", a2="fixed", rule="G7", alpha=0.01, metric="logerr", mean1=f.texp64.mean(),
                    mean2=f.fixed.mean(), diff=(f.texp64 - f.fixed).mean(), wins=int((f.texp64 < f.fixed).sum()),
                    losses=int((f.texp64 > f.fixed).sum()), p=wilcoxon(f.texp64, f.fixed).pvalue))
    out["secondary"] = sec
    fold16 = df[(df.rule == "texp64") & (df.alpha == 0.01) & df.fold.isin([1, 6])].groupby(["arm", "fold"]).pd.mean()
    out["folds_1_6_pd"] = {f"{a}_f{f}": v for (a, f), v in fold16.items()}
    print(json.dumps(out, indent=1, default=float))
    summ.to_csv(HERE / "C_loro_summary.csv", index=False)
    (HERE / "C_loro_endpoints.json").write_text(json.dumps(out, indent=1, default=float))
    df.to_csv(HERE / "C_loro_long.csv", index=False)


if __name__ == "__main__":
    main()
