"""Summarize N1 against rules N-a / N-b of D_PLAN.md."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
D = pd.read_csv(HERE / "N1_temporal_net.csv")
D["over"] = D.pfa > 2 * D.alpha
D["terr"] = np.abs(np.log10(np.clip(D.pfa, 1e-5, 1) / D.alpha))
agg = dict(pd=("pd", "mean"), pfa=("pfa", "mean"), adj=("pfa_adj", "mean"), over=("over", "sum"), n=("over", "size"), terr=("terr", "mean"))
for prot in ("loro", "1998"):
    g = D[(D.protocol == prot) & (D.rule != "oracle")].groupby(["scorer", "K", "rule", "alpha"]).agg(**agg)
    g["pfa/a"] = g.pfa / g.index.get_level_values("alpha")
    print(f"\n== {prot}\n", g.drop(columns="pfa").round(3).to_string())
    o = D[(D.protocol == prot) & (D.rule == "oracle")].groupby(["scorer", "K", "alpha"]).pd.mean().unstack()
    print("oracle Pd\n", o.round(3).to_string())

t = D[(D.protocol == "loro") & (D.scorer == "network") & (D.rule == "texp") & (D.alpha == 0.01)]
f = t.groupby(["K", "fold"]).pd.mean().unstack(0)
print("\nN-a: fold means\n", f.round(3).to_string())
for K in (4, 8):
    print(f" K={K} vs K=1: higher in {(f[K] > f[1]).sum()}/10, p={wilcoxon(f[K], f[1]).pvalue:.4f}, runs>2a {int(t[t.K == K].over.sum())}/30")
x = D[(D.protocol == "1998") & (D.scorer == "network") & (D.rule == "texp") & (D.alpha == 0.01)]
print("\nN-b: 1998 per seed\n", x[["K", "seed", "pd", "pfa"]].assign(pa=x.pfa / 0.01).round(3).to_string(index=False))
h = D[(D.protocol == "loro") & (D.rule == "texp") & (D.alpha == 0.01)].groupby(["scorer", "K", "fold"]).pd.mean().unstack(0)
for K in (1, 4, 8):
    hk = h.xs(K, level="K")
    print(f" net vs HGB K={K}: net higher {(hk.network > hk.HGB).sum()}/10, p={wilcoxon(hk.network, hk.HGB).pvalue:.3f}")
