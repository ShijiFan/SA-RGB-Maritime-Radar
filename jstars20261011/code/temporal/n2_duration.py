"""N2 (descriptive): temporal RadarGrid-CS across observation durations on IPIX1993 LORO; same design as N1.
Evaluated frames end at or after 8.192 s; training frames start at the first frame with 8 frames of history."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from n1_temporal_net import DEV, evaluate, feat_front, fit_net, lab, load, loro_split, net_scores, RECORDS  # noqa: E402,F401
from rgcs import fit_percentiles, quantize  # noqa: E402
from matched_inputs import tensor  # noqa: E402

DURS = {"0p256": 0.256, "1p024": 1.024, "0p512": 0.512}  # 0.128 s dropped: below the shortest duration used in the literature
KS = (1, 2, 4, 8)


def main():
    front = feat_front(); rows = []
    feat = lambda X, lo, hi: torch.as_tensor(tensor(front, quantize(X.reshape(-1, 14, 7), lo, hi)).reshape(*X.shape[:3], 72), device=DEV)
    for tag, d in DURS.items():
        data = load(tag); Fr = data[1]["X"].shape[1]
        n_tr, n_ev = np.arange(7, Fr), np.arange(int(np.ceil(8.192 / d)) - 1, Fr)
        for k in RECORDS:
            t0 = time.time()
            tr, va, te = loro_split(k)
            lo, hi = fit_percentiles(np.concatenate([data[g]["X"].reshape(-1, 7) for g in tr]))
            F = {g: feat(data[g]["X"], lo, hi) for g in (*tr, va, te)}
            Ftr = torch.cat([F[g] for g in tr]); ytr = lab(np.repeat([data[g]["t"] for g in tr], 4), len(tr) * 4)
            yv = lab(np.full(4, data[va]["t"]), 4)
            for K in KS:
                for s in (0, 1, 2):
                    m = fit_net(s, K, Ftr, n_tr, ytr, F[va], n_ev, yv)
                    res = evaluate(net_scores(m, F[va], n_ev, K), np.full(4, data[va]["t"]),
                                   net_scores(m, F[te], n_ev, K), np.full(4, data[te]["t"]))
                    for (rule, a), v in res.items():
                        rows.append(dict(dur=d, K=K, dwell=round(d * K, 3), fold=k, seed=s, rule=rule, alpha=a, **v))
            print(f"d={d} fold {k} ({time.time() - t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(HERE / "N2_duration.csv", index=False)


if __name__ == "__main__":
    main()
