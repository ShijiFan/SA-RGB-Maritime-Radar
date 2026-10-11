"""Pre-registered E3 evaluation for the IPIX1998 one-shot run."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "runs_1998" / "summary.json").read_text())
meta = json.load(open(HERE.parent / "phaseA_20261008" / "out" / "ipix1998_7feat.json", encoding="utf8"))
names = {str(r["rec"]): f"{r['stem']}(t{r['target_cell']},{r['cells']}c)" for r in meta["records"]}
out = {}
for a in ("0.01", "0.001"):
    print("alpha", a)
    for arm in ("G7", "G3", "D7"):
        rr = [r for r in R if r["arm"] == arm]
        pdv = [r[a]["pd"] for r in rr]; pf = [r[a]["pfa"] for r in rr]; ad = [r[a]["pfa_adj"] for r in rr]
        out[f"{arm}_{a}"] = dict(pd=pdv, pfa=pf, adj=ad)
        print(f"  {arm}: Pd {np.mean(pdv):.3f} {np.round(pdv, 3)}  Pfa {np.mean(pf):.4f} {np.round(pf, 4)}  "
              f"adj {np.mean(ad):.3f}  exact {np.mean([r['exact_cell'] for r in rr]):.3f}")
    print("  record                         G7 Pd  G3 Pd  G7 Pfa  G3 Pfa")
    for k in sorted(R[0][a]["pd_by_record"], key=int):
        v = {arm: (np.mean([r[a]["pd_by_record"][k] for r in R if r["arm"] == arm]),
                   np.mean([r[a]["pfa_by_record"][k] for r in R if r["arm"] == arm])) for arm in ("G7", "G3")}
        print(f"  {names[k]:30s} {v['G7'][0]:.3f}  {v['G3'][0]:.3f}  {v['G7'][1]:.4f}  {v['G3'][1]:.4f}")
g7, g3 = out["G7_0.01"], out["G3_0.01"]
crit = dict(pd_g7_gt_g3=bool(np.mean(g7["pd"]) > np.mean(g3["pd"])),
            g7_pfa_le_2a_all_seeds=bool(all(p <= 0.02 for p in g7["pfa"])))
crit["verdict"] = "SUPPORTED" if all(crit.values()) else "NOT SUPPORTED"
print("E3:", crit)
(HERE / "C_1998_endpoints.json").write_text(json.dumps(dict(E3=crit, values=out), indent=1))
