"""Background-cell activation rate (range distance > 1 from the target) for every arm and rule, from stored scores."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phaseA_20261008"))
from common import RECORDS, calibrate  # noqa: E402
from rgcs import tail_params, texp_tau  # noqa: E402

C = np.arange(14)
rows = []
for arm in ("G7", "G3", "D7", "P7", "G7-noSCN", "G7-absC"):
    for k in RECORDS:
        for s in (0, 1, 2):
            z = np.load(HERE / "runs_loro" / f"{arm}_f{k:02d}_s{s}_scores.npz")
            jr = json.loads((HERE / "runs_loro" / f"{arm}_f{k:02d}_s{s}.json").read_text())
            Sv, St, tv, tt = z["val"], z["test"], int(z["val_t"]), int(z["test_t"])
            ut, bt = tail_params(St)
            bg = np.abs(C - tt) > 1
            for rule, tau in (("texp64", texp_tau(ut, bt, 1e-2, jr["texp_c"])),
                              ("fixed", np.full(St.shape[:2], calibrate(Sv[:, :, C != tv], 1e-2)))):
                ok = np.isfinite(tau)
                rows.append(dict(arm=arm, fold=k, seed=s, rule=rule,
                                 pfa_bg=float((St[:, :, bg] >= tau[..., None])[ok].mean())))
df = pd.DataFrame(rows); df.to_csv(HERE / "C_bg_pfa.csv", index=False)
print((df.groupby(["arm", "rule"]).pfa_bg.mean() / 1e-2).round(2).to_string())
