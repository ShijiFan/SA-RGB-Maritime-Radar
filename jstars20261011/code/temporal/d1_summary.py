"""Summarize D1 against the decision rules of D_PLAN.md."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
D = pd.read_csv(HERE / "D1_temporal.csv")
D["over"] = D["pfa_fix_0.01"] > 0.02
g = D.groupby(["dur", "K"]).agg(dwell=("dwell", "first"), pdor2=("pdor_0.01", "mean"), pdor3=("pdor_0.001", "mean"),
                                 pdfix=("pd_fix_0.01", "mean"), pfafix=("pfa_fix_0.01", "mean"), over=("over", "sum"),
                                 exact=("exact", "mean"))
g["pfa/a"] = g.pfafix / 0.01
print(g.drop(columns="pfafix").round(3).to_string())

print("\nPd (oracle, 1e-2) grid: rows d, columns K")
print(D.pivot_table(index="dur", columns="K", values="pdor_0.01").round(3).to_string())
print("\nEqual dwell T (oracle 1e-2):")
for T, h in D.groupby("dwell"):
    if h.dur.nunique() > 1:
        print(T, h.groupby(["dur", "K"])["pdor_0.01"].mean().round(3).to_dict())

print("\nDecision rules at d = 0.512:")
x = D[D.dur == 0.512].pivot(index="fold", columns="K", values="pdor_0.01")
for K in (2, 4, 8):
    diff = x[K] - x[1]
    print(f" K={K}: mean gain {diff.mean():+.3f}, folds gain>=0.05: {(diff >= 0.05).sum()}/10, Wilcoxon p={wilcoxon(x[K], x[1]).pvalue:.3f}")
o = D[D.dur == 0.512].groupby("K").over.sum()
print(" folds >2a (fixed, 1e-2):", o.to_dict())
