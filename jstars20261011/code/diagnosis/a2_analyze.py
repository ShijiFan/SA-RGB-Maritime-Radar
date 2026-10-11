"""Summarize A2: information (oracle Pd) vs controllability (calibrated Pfa error) by subset size."""
import numpy as np
import pandas as pd

from common import OUT

df = pd.read_csv(OUT / "A2_subset_loro_folds.csv")
for a in ("0.01", "0.001"):
    al = float(a)
    df[f"logerr_{a}"] = np.abs(np.log10(np.clip(df[f"pfa_{a}"], 1e-5, 1) / al))
    df[f"over2_{a}"] = (df[f"pfa_{a}"] > 2 * al).astype(int)

agg = dict(pdor_001=("pdor_0.01", "mean"), pd_001=("pd_0.01", "mean"), pfa_001=("pfa_0.01", "mean"),
           over2_001=("over2_0.01", "sum"), logerr_001=("logerr_0.01", "mean"),
           pdor_0001=("pdor_0.001", "mean"), pfa_0001=("pfa_0.001", "mean"), over2_0001=("over2_0.001", "sum"),
           exact=("exact_cell", "mean"))
per_subset = df.groupby(["rep", "ctx", "model", "subset", "k"]).agg(**agg).reset_index()
per_subset.to_csv(OUT / "A2_per_subset.csv", index=False)

pd.set_option("display.width", 250); pd.set_option("display.precision", 3); pd.set_option("display.max_rows", 200)
lr = per_subset[per_subset.model == "LR"]
print("=== LR: mean over subsets of each size ===")
print(lr.groupby(["rep", "ctx", "k"])[["pdor_001", "pd_001", "pfa_001", "over2_001", "logerr_001", "pdor_0001",
                                        "over2_0001", "exact"]].mean().to_string())
print("\n=== LR: best subset (oracle Pd@1e-2) per size ===")
idx = lr.groupby(["rep", "ctx", "k"])["pdor_001"].idxmax()
print(lr.loc[idx, ["rep", "ctx", "k", "subset", "pdor_001", "pfa_001", "over2_001", "exact"]].to_string(index=False))
print("\n=== Named groups (LR and HGB) ===")
named = {"RAA+RDPH+RVE": "G3", "RAA+RDPH+RVE+MCS": "G4", "Hurst+RAA+RDPH+RVE+RI": "G5",
         "Hurst+RAA+RDPH+RVE+RI+NR+MCS": "G7"}
g = per_subset.copy(); g["group"] = g.subset.map(lambda s: named.get(s, s))
g = g[g.group.isin(["G3", "G4", "G5", "G7"])]
print(g.sort_values(["rep", "ctx", "model", "group"])[["rep", "ctx", "model", "group", "pdor_001", "pd_001", "pfa_001",
                                                       "over2_001", "pdor_0001", "over2_0001", "exact"]].to_string(index=False))
