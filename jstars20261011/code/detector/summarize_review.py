"""Summary numbers for the review revision (Results text and supplementary tables)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
out = {}

r1 = pd.read_csv(HERE / "C_review_R1_common_mask.csv")
g = r1.groupby(["arm", "rule"])[["pfa", "pfa_bg", "pfa_adj", "pd"]].mean()
print("R1 common mask\n", g.round(4))
out["R1"] = {f"{a}|{r}": v for (a, r), v in g.round(4).to_dict("index").items()}

r2 = pd.read_csv(HERE / "C_review_R2_W_trim.csv")
r2["over"] = r2.pfa > 2 * r2.alpha
g = r2.groupby(["arm", "W", "trim", "alpha"]).agg(pd=("pd", "mean"), pfa=("pfa", "mean"), over=("over", "sum"), n=("over", "size"))
g["pfa_a"] = g.pfa / g.index.get_level_values("alpha")
print("R2 W/trim\n", g.round(3))
out["R2"] = {"|".join(map(str, k)): v for k, v in g.round(4).to_dict("index").items()}

r3 = pd.read_csv(HERE / "C_review_R3_warmup.csv")
g = r3.groupby("arm")[["pfa_warm", "pfa_steady"]].mean()
print("R3 warmup\n", g.round(4))
out["R3"] = g.round(4).to_dict("index")

r5 = pd.read_csv(HERE / "C_review_R5_tail_calibration.csv")
g = r5.groupby(["arm", "alpha"]).pfa.mean()
g = (g / g.index.get_level_values("alpha")).round(3)
print("R5 ratio\n", g)
out["R5"] = {f"{a}|{al}": v for (a, al), v in g.to_dict().items()}

bg = pd.read_csv(HERE / "C_bg_pfa.csv")
g = (bg.groupby(["arm", "rule"]).pfa_bg.mean() / 1e-2).round(3)
print("bg/alpha\n", g)
out["bg"] = {f"{a}|{r}": v for (a, r), v in g.to_dict().items()}

print((HERE / "C_review_checks.json").read_text())
hull = pd.read_csv(HERE / "C_classical_hull.csv")
print(hull.to_string())
mi = pd.read_csv(HERE / "C_matched_inputs.csv")
print(mi.columns.tolist())
print(mi.head(3).to_string())
(HERE / "C_review_summary.json").write_text(json.dumps(out, indent=1))
