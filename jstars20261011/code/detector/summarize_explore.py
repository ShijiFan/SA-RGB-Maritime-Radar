"""Summarize the exploratory context study; first checks that grid3x3 reproduces the frozen Phase C runs."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
files = sys.argv[1:] or [str(p) for p in (HERE / "explore_context").glob("rows_*.csv")]
df = pd.concat([pd.read_csv(f) for f in files])

# reproduction check against frozen runs
L = df[(df.protocol == "loro") & (df.variant == "grid3x3")]
diff = []
for _, r in L.iterrows():
    j = HERE / "runs_loro" / f"{r.arm}_f{int(r.fold):02d}_s{int(r.seed)}.json"
    if j.exists():
        ref = json.loads(j.read_text())["texp64_0.01"]
        diff.append(abs(ref["pd"] - r["pd_0.01"]) + abs(ref["pfa"] - r["pfa_0.01"]))
if diff:
    print(f"grid3x3 vs frozen Phase C: {len(diff)} runs, max |dPd|+|dPfa| = {max(diff):.2e}")

df["over2_1e2"] = (df["pfa_0.01"] > 0.02).astype(int)
df["over2_1e3"] = (df["pfa_0.001"] > 0.002).astype(int)
pd.set_option("display.width", 250); pd.set_option("display.precision", 4)
s = df.groupby(["protocol", "arm", "variant"]).agg(
    pd1e2=("pd_0.01", "mean"), pfa1e2=("pfa_0.01", "mean"), over2_1e2=("over2_1e2", "sum"), adj1e2=("pfa_adj_0.01", "mean"),
    pd1e3=("pd_0.001", "mean"), pfa1e3=("pfa_0.001", "mean"), over2_1e3=("over2_1e3", "sum"), n=("seed", "size"))
print(s.to_string())
s.to_csv(HERE / "explore_context" / "summary.csv")
