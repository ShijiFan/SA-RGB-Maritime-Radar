"""Control S on IPIX1998 per record: oracle Pd at equal realized Pfa (1e-2) and tail-rule Pfa, mean4 vs trained K = 4."""
from pathlib import Path

import numpy as np
import pandas as pd

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from n5_rescore import evaluate_all, mean4  # noqa: E402
from common import pd_at_pfa  # noqa: E402

FIN = HERE / "final"; C = np.arange(14); rows = []
for s in (0, 1, 2):
    z1 = np.load(FIN / f"G7_K1_1998_s{s}_scores.npz"); z4 = np.load(FIN / f"G7_K4_1998_s{s}_scores.npz")
    rec, tt = z1["rec"], z1["test_t"]
    for name, Sv, St in (("mean4", mean4(z1["val"]), mean4(z1["test"])), ("net-K4", z4["val"][:, 3:], z4["test"][:, 3:])):
        _, pr = evaluate_all(Sv, z1["val_t"], St, tt, rec=rec)
        st = St[:, 1:]
        for p in pr:
            q = rec == p["record"]; h0 = C[None, None] != tt[q][:, None, None]
            h1 = np.take_along_axis(st[q], np.broadcast_to(tt[q][:, None, None], st[q].shape[:2] + (1,)), 2)[..., 0]
            rows.append(dict(p, method=name, seed=s, oracle=pd_at_pfa(h1.ravel(), st[q][np.broadcast_to(h0, st[q].shape)], 1e-2)))
d = pd.DataFrame(rows); d.to_csv(HERE / "S_1998_records.csv", index=False)
g = d.groupby(["method", "record"])[["pd", "pfa", "oracle"]].mean().unstack(0)
print(g.round(3).to_string())
print("oracle net>mean4 records:", int((g["oracle"]["net-K4"] > g["oracle"]["mean4"]).sum()),
      " pfa net<mean4 records:", int((g["pfa"]["net-K4"] < g["pfa"]["mean4"]).sum()))
