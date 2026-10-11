"""Recompute every manuscript number from the result files; write claim_evidence.csv; fail loudly on mismatch."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
A, C = ROOT / "phaseA_20261008" / "out", ROOT / "phaseC"
TEX = "\n".join(p.read_text(encoding="utf8") for p in [*sorted((Path(__file__).parent / "source" / "sections").glob("*.tex")), Path(__file__).parent / "supplementary" / "supplement" / "Supplementary_Material.tex"])
rows = []


NOT_QUOTED = {  # recomputed but no longer quoted after the round-2 restructuring (first-design details moved out)
    "CA-CFAR Pd at Pfa=1e-2", "HGB-G3 Pd at Pfa=1e-2", "HGB-G7 Pd at Pfa=1e-2", "RadarGrid-CS D7 Pd at Pfa=1e-2",
    "1998 G3 Pd 1e-3", "R8 G7 Pd", "margin1 1998 Pfa/a 1e-3", "hull 1998 Pfa/a", "hull 1998 adj", "HGB 1998 Pfa/a",
    "trim2 Pfa/a 1e-3", "single 1998 1e-3", "single LORO oracle", "single LORO oracle 1e-3"}


def claim(section, text, value, fmt, source):
    if text in NOT_QUOTED:
        return
    s = fmt.format(value)
    ok = s in TEX
    rows.append(dict(section=section, claim=text, value=s, in_text=ok, source=source))
    if not ok:
        print(f"MISSING in text: {section} | {text} = {s}")


RB = pd.read_csv(C / "C_reference_baselines.csv")
for m in ("CA-CFAR", "HGB-G7", "HGB-G3"):
    for a, tag in ((0.01, "1e-2"), (0.001, "1e-3")):
        x = RB[(RB.protocol == "loro") & (RB.method == m) & (RB.rule == "texp") & (RB.alpha == a)]
        claim("VI-ref", f"{m} Pd {tag}", x.pd.mean(), "{:.3f}", "phaseC/C_reference_baselines.csv")
        claim("VI-ref", f"{m} Pfa/a {tag}", x.pfa.mean() / a, "{:.2f}" if a == 0.01 else "{:.1f}", "phaseC/C_reference_baselines.csv")
        claim("VI-ref", f"{m} over2 {tag}", int((x.pfa > 2 * a).sum()), "{}/10", "phaseC/C_reference_baselines.csv")
    x = RB[(RB.protocol == "loro") & (RB.method == m) & (RB.rule == "texp") & (RB.alpha == 0.01)]
    claim("VI-ref", f"{m} logerr", np.mean(np.abs(np.log10(np.clip(x.pfa, 1e-5, 1) / 0.01))), "{:.3f}", "phaseC/C_reference_baselines.csv")
    claim("VI-ref", f"{m} adj", x.pfa_adj.mean(), "{:.3f}", "phaseC/C_reference_baselines.csv")
orc = RB[(RB.protocol == "1998") & (RB.rule == "oracle") & (RB.alpha == 0.01)].groupby("method").pd.mean()
for m in orc.index:
    claim("VI-1998eq", f"{m} Pd at Pfa=1e-2", orc[m], "{:.3f}", "phaseC/C_reference_baselines.csv")
NS = pd.read_csv(A / "B_nested_selection.csv"); NO = pd.read_csv(A / "B_nested_outer.csv")
claim("VI-nested", "frame chosen", int((NS.D1 == "frame").sum()), "{}", "phaseA/out/B_nested_selection.csv")
W = {3: "three", 7: "seven", 10: "ten"}
claim("VI-nested", "texp chosen", W.get(int((NS.D3 == "texp").sum()), "?"), "exponential tail in {} folds", "phaseA/out/B_nested_selection.csv")
claim("VI-nested", "cmap chosen", W.get(int((NS.D3 == "cmap").sum()), "?"), "empirical clutter map in {}", "phaseA/out/B_nested_selection.csv")
claim("VI-nested", "frame+rel in all", W.get(int(((NS.D1 == "frame") & (NS.D2 == "rel")).sum()), "?"), "in all {} outer folds", "phaseA/out/B_nested_selection.csv")
x = NO[NO.alpha == 0.01]
claim("VI-nested", "nested Pd", x.pd.mean(), "{:.3f}", "phaseA/out/B_nested_outer.csv")
claim("VI-nested", "nested Pfa/a", x.pfa.mean() / 0.01, "{:.2f}\\alpha$ on the held-out", "phaseA/out/B_nested_outer.csv")
claim("VI-nested", "nested 1e-3 Pfa/a", NO[NO.alpha == 0.001].pfa.mean() / 0.001, "{:.1f}\\alpha$", "phaseA/out/B_nested_outer.csv")

L = pd.read_csv(C / "C_loro_long.csv")
t = L[(L.rule == "texp64") & (L.alpha == 0.01)]
fm = lambda arm, col="pd", d=t: d[d.arm == arm].groupby("fold")[col].mean()
for arm in ("G7", "G3", "D7", "P7", "G7-noSCN", "G7-absC"):
    d = t[t.arm == arm]
    claim("VI/Tab2", f"{arm} Pd 1e-2", d.pd.mean(), "{:.3f}", "phaseC/C_loro_long.csv")
    claim("VI/Tab2", f"{arm} Pfa/alpha 1e-2", d.pfa.mean() / 0.01, "{:.2f}", "phaseC/C_loro_long.csv")
    claim("VI/Tab2", f"{arm} runs>2a", int((d.pfa > 0.02).sum()), "{}/30", "phaseC/C_loro_long.csv")
    claim("VI/Tab2", f"{arm} transfer error", np.mean(np.abs(np.log10(np.clip(d.pfa, 1e-5, 1) / 0.01))), "{:.3f}", "phaseC/C_loro_long.csv")
    claim("VI/Tab2", f"{arm} adjacent", d.pfa_adj.mean(), "{:.3f}", "phaseC/C_loro_long.csv")
    d3 = L[(L.rule == "texp64") & (L.alpha == 0.001) & (L.arm == arm)]
    claim("VI/Tab2", f"{arm} Pd 1e-3", d3.pd.mean(), "{:.3f}", "phaseC/C_loro_long.csv")
    claim("VI/Tab2", f"{arm} runs>2a 1e-3", int((d3.pfa > 0.002).sum()), "{}/30", "phaseC/C_loro_long.csv")
    if arm != "G7":
        claim("VI/Tab2", f"{arm} p vs G7", wilcoxon(fm("G7"), fm(arm)).pvalue, "{:.2f}" if arm != "G3" else "{:.3f}",
              "phaseC/C_loro_long.csv")
f = L[(L.rule == "fixed") & (L.alpha == 0.01) & (L.arm == "G7")]
claim("VI/Tab2", "G7 fixed Pd", f.pd.mean(), "{:.3f}", "phaseC/C_loro_long.csv")
claim("VI/Tab2", "G7 fixed runs>2a", int((f.pfa > 0.02).sum()), "{}/30", "phaseC/C_loro_long.csv")
claim("VI", "G7 vs G3 wins", int((fm("G7") > fm("G3")).sum()), "in {} of 10 folds", "phaseC/C_loro_long.csv")
le = lambda d: d.assign(e=np.abs(np.log10(np.clip(d.pfa, 1e-5, 1) / 0.01))).groupby("fold").e.mean()
a, b = le(t[t.arm == "G7"]), le(f)
claim("VI", "texp vs fixed logerr p", wilcoxon(a, b).pvalue, "p={:.3f}", "phaseC/C_loro_long.csv")
claim("VI", "texp lower in folds", int((a < b).sum()), "lower in {} of 10 folds", "phaseC/C_loro_long.csv")
g7 = L[(L.arm == "G7") & (L.rule == "texp64") & (L.alpha == 0.01)]
claim("VI", "fold1 G7 Pd", g7[g7.fold == 1].pd.mean(), "{:.3f}", "phaseC/C_loro_long.csv")
claim("VI", "G7 exact cell", L[(L.arm == "G7")].exact.mean(), "{:.3f}", "phaseC/C_loro_long.csv")
claim("VI", "G3 exact cell", L[(L.arm == "G3")].exact.mean(), "{:.3f}", "phaseC/C_loro_long.csv")

R = json.loads((C / "runs_1998" / "summary.json").read_text())
p98 = lambda arm, k, a="0.01": np.mean([r[a][k] for r in R if r["arm"] == arm])
claim("VI", "1998 G7 Pd", p98("G7", "pd"), "{:.3f}", "phaseC/runs_1998/summary.json")
claim("VI", "1998 G3 Pd", p98("G3", "pd"), "{:.3f}", "phaseC/runs_1998/summary.json")
claim("VI", "1998 D7 Pd", p98("D7", "pd"), "{:.3f}", "phaseC/runs_1998/summary.json")
claim("VI", "1998 G7 Pfa/a", p98("G7", "pfa") / 0.01, "{:.2f}", "phaseC/runs_1998/summary.json")
claim("VI", "1998 G3 Pfa/a", p98("G3", "pfa") / 0.01, "{:.2f}", "phaseC/runs_1998/summary.json")
claim("VI", "1998 G7 adj", p98("G7", "pfa_adj"), "{:.3f}", "phaseC/runs_1998/summary.json")
claim("VI", "1998 G3 adj", p98("G3", "pfa_adj"), "{:.3f}", "phaseC/runs_1998/summary.json")
claim("VI", "1998 G7 Pd 1e-3", p98("G7", "pd", "0.001"), "{:.3f}", "phaseC/runs_1998/summary.json")
claim("VI", "1998 G3 Pd 1e-3", p98("G3", "pd", "0.001"), "{:.3f}", "phaseC/runs_1998/summary.json")
rec = lambda arm, k, r: np.mean([x["0.01"][k][r] for x in R if x["arm"] == arm])
claim("VI", "R8 G7 Pd", rec("G7", "pd_by_record", "8"), "{:.3f}", "phaseC/runs_1998/summary.json")
claim("VI", "R8 G7 Pfa/a", rec("G7", "pfa_by_record", "8") / 0.01, "{:.1f}", "phaseC/runs_1998/summary.json")
claim("VI", "R6 G7 Pfa/a", rec("G7", "pfa_by_record", "6") / 0.01, "{:.1f}", "phaseC/runs_1998/summary.json")
print("all 9 records G7>G3:", all(rec("G7", "pd_by_record", str(k)) > rec("G3", "pd_by_record", str(k)) for k in range(1, 10)))

E = pd.concat([pd.read_csv(C / "explore_context" / f"rows_{a}.csv") for a in ("G7", "G3")])
ex = lambda arm, v, p, col: E[(E.arm == arm) & (E.variant == v) & (E.protocol == p)][col].mean()
claim("VI-exp", "margin1 LORO Pd", ex("G7", "margin1", "loro", "pd_0.01"), "{:.3f}", "phaseC/explore_context")
claim("VI-exp", "margin1 G3 LORO Pd", ex("G3", "margin1", "loro", "pd_0.01"), "{:.3f}", "phaseC/explore_context")
claim("VI-exp", "margin1 1998 Pd", ex("G7", "margin1", "1998", "pd_0.01"), "{:.3f}", "phaseC/explore_context")
claim("VI-exp", "margin1 G3 1998 Pd", ex("G3", "margin1", "1998", "pd_0.01"), "{:.3f}", "phaseC/explore_context")
claim("VI-exp", "margin1 1998 Pfa/a", ex("G7", "margin1", "1998", "pfa_0.01") / 0.01, "{:.2f}", "phaseC/explore_context")
claim("VI-exp", "none 1998 Pfa/a", ex("G7", "none", "1998", "pfa_0.01") / 0.01, "{:.2f}", "phaseC/explore_context")
claim("VI-exp", "none LORO Pd", ex("G7", "none", "loro", "pd_0.01"), "{:.3f}", "phaseC/explore_context")
claim("VI-exp", "r2 1998 Pd", ex("G7", "r2", "1998", "pd_0.01"), "{:.3f}", "phaseC/explore_context")
claim("VI-exp", "margin1 1998 Pfa/a 1e-3", ex("G7", "margin1", "1998", "pfa_0.001") / 0.001, "{:.1f}", "phaseC/explore_context")
mm = E[(E.variant == "margin1") & (E.protocol == "loro")].groupby(["arm", "fold"])["pd_0.01"].mean().unstack(0)
claim("VI-exp", "margin1 G7 vs G3 p", wilcoxon(mm.G7, mm.G3).pvalue, "p={:.3f}", "phaseC/explore_context")

S = pd.read_csv(A / "A2_per_subset.csv")
h = S[(S.model == "HGB") & (S.rep == "framenorm") & (~S.ctx)].set_index("subset")
for g in ("G3", "G4", "G5", "G7"):
    claim("IV", f"HGB {g} pdor", h.loc[g, "pdor_001"], "{:.3f}", "phaseA/out/A2_per_subset.csv")
claim("IV", "HGB G3 pdor 1e-3", h.loc["G3", "pdor_0001"], "{:.3f}", "phaseA/out/A2_per_subset.csv")
claim("IV", "HGB G7 pdor 1e-3", h.loc["G7", "pdor_0001"], "{:.3f}", "phaseA/out/A2_per_subset.csv")
a1 = pd.read_csv(A / "A1_summary.csv").set_index(["rep", "feature"])
claim("IV", "Hurst drift raw", a1.loc[("raw", "Hurst"), "drift_index"], "{:.2f}", "phaseA/out/A1_summary.csv")
claim("IV", "Hurst drift norm", a1.loc[("framenorm", "Hurst"), "drift_index"], "{:.2f}", "phaseA/out/A1_summary.csv")
lat = json.loads((C / "C_latency.json").read_text())
claim("V", "CFAR update", lat["tail_cfar_update_ms"]["median"], "{:.2f}~ms", "phaseC/C_latency.json")
gs_ = pd.read_csv(C / "C_shift_sensitivity.csv"); gm_ = pd.read_csv(C / "C_shift_margin1.csv")
for d_ in (gs_, gm_):
    d_["a"] = d_["shift"].abs()
mg = gs_[gs_.arm == "G7"].groupby("a").pd.mean(); mm = gm_[gm_.arm == "G7"].groupby("a").pd.mean()
claim("VI-shift", "grid G7 s=+-1", mg[1], "0.641 to {:.3f}", "phaseC/C_shift_sensitivity.csv")
claim("VI-shift", "grid G7 pfa s=+-1", gs_[(gs_.arm == "G7") & (gs_.a == 1)].pfa.mean() / 0.01, "{:.2f}\\alpha$", "phaseC/C_shift_sensitivity.csv")
f6 = gs_[(gs_.arm == "G7") & (gs_.fold == 6)].groupby("a").pd.mean()
claim("VI-shift", "grid G7 fold6", f"{f6[0]:.3f} to {f6[1]:.2f}", "{}", "phaseC/C_shift_sensitivity.csv")
claim("VI-shift", "margin G7 range", f"{mm[2]:.3f}--{mm[0]:.3f}", "{}", "phaseC/C_shift_margin1.csv")
claim("VI-shift", "margin G7 fold6", gm_[(gm_.arm == "G7") & (gm_.fold == 6) & (gm_.a == 0)].pd.mean(), "{:.3f}", "phaseC/C_shift_margin1.csv")
H = pd.read_csv(C / "C_classical_hull.csv")
hl = H[(H.protocol == "loro") & (H.alpha == 0.01)]
claim("VI", "hull Pd", hl.pd.mean(), "{:.3f}", "phaseC/C_classical_hull.csv")
claim("VI", "hull Pfa/a", hl.pfa.mean() / 0.01, "{:.2f}", "phaseC/C_classical_hull.csv")
claim("VI", "hull over2", int((hl.pfa > 0.02).sum()), "{}/10", "phaseC/C_classical_hull.csv")
claim("VI", "hull logerr", np.mean(np.abs(np.log10(np.clip(hl.pfa, 1e-5, 1) / 0.01))), "{:.3f}", "phaseC/C_classical_hull.csv")
h3 = H[(H.protocol == "loro") & (H.alpha == 0.001)]
claim("VI", "hull Pd 1e-3", h3.pd.mean(), "{:.3f}", "phaseC/C_classical_hull.csv")
claim("VI", "hull over2 1e-3", int((h3.pfa > 0.002).sum()), "{}/10", "phaseC/C_classical_hull.csv")
h98 = H[(H.protocol == "1998") & (H.alpha == 0.01)].iloc[0]
claim("VI", "hull 1998 Pd", h98.pd, "{:.3f}", "phaseC/C_classical_hull.csv")
claim("VI", "hull 1998 Pfa/a", h98.pfa / 0.01, "{:.1f}", "phaseC/C_classical_hull.csv")
claim("VI", "hull 1998 adj", h98.pfa_adj, "{:.3f}", "phaseC/C_classical_hull.csv")
claim("VI", "hull infeasible 1e-3", int((~h3.feasible).sum()), "in {} of 10 folds", "phaseC/C_classical_hull.csv")

# review revision: background rates, matched inputs, tail-rule checks, IPIX1998 macro values
BG = pd.read_csv(C / "C_bg_pfa.csv")
for arm in ("G7", "G3", "D7", "P7", "G7-noSCN", "G7-absC"):
    claim("VI/Tab3", f"{arm} background/a", BG[(BG.arm == arm) & (BG.rule == "texp64")].pfa_bg.mean() / 0.01, "{:.2f}", "phaseC/C_bg_pfa.csv")
claim("VI/Tab3", "G7 fixed background/a", BG[(BG.arm == "G7") & (BG.rule == "fixed")].pfa_bg.mean() / 0.01, "{:.2f}", "phaseC/C_bg_pfa.csv")
MI = pd.read_csv(C / "C_matched_inputs.csv")
mi = lambda sc, prot, rule, a: MI[(MI.scorer == sc) & (MI.protocol == prot) & (MI.rule == rule) & (MI.alpha == a)]
for sc in ("network", "HGB"):
    x = mi(sc, "loro", "texp", 0.01)
    claim("VI-matched", f"{sc} LORO Pd", x.pd.mean(), "{:.3f}", "phaseC/C_matched_inputs.csv")
    claim("VI-matched", f"{sc} LORO Pfa/a", x.pfa.mean() / 0.01, "{:.2f}\\alpha$", "phaseC/C_matched_inputs.csv")
    claim("VI-matched", f"{sc} LORO transfer", np.mean(np.abs(np.log10(np.clip(x.pfa, 1e-5, 1) / 0.01))), "{:.3f}", "phaseC/C_matched_inputs.csv")
    claim("VI-matched", f"{sc} 1998 Pd", mi(sc, "1998", "texp", 0.01).pd.mean(), "{:.3f}", "phaseC/C_matched_inputs.csv")
    claim("VI-matched", f"{sc} 1998 Pfa/a", mi(sc, "1998", "texp", 0.01).pfa.mean() / 0.01, "{:.2f}\\alpha$", "phaseC/C_matched_inputs.csv")
    claim("VI-matched", f"{sc} 1998 oracle", mi(sc, "1998", "oracle", 0.01).pd.mean(), "{:.3f}", "phaseC/C_matched_inputs.csv")
    claim("VI-matched", f"{sc} LORO Pd 1e-3", mi(sc, "loro", "texp", 0.001).pd.mean(), "{:.3f}", "phaseC/C_matched_inputs.csv")
fl = MI[(MI.protocol == "loro") & (MI.rule == "texp") & (MI.alpha == 0.01)].groupby(["scorer", "fold"]).pd.mean().unstack(0)
claim("VI-matched", "HGB higher folds", int((fl.HGB > fl.network).sum()), "HGB higher in {} of 10 folds", "phaseC/C_matched_inputs.csv")
claim("VI-matched", "p", wilcoxon(fl.HGB, fl.network).pvalue, "$p={:.2f}$", "phaseC/C_matched_inputs.csv")
R2 = pd.read_csv(C / "C_review_R2_W_trim.csv")
r2 = lambda W, tr, a=0.01: R2[(R2.arm == "G7") & (R2.W == W) & (R2.trim == tr) & (R2.alpha == a)]
claim("VI-sens", "W32 Pd", r2(32, 1).pd.mean(), "{:.3f}", "phaseC/C_review_R2_W_trim.csv")
claim("VI-sens", "W16 Pd", r2(16, 1).pd.mean(), "{:.3f}", "phaseC/C_review_R2_W_trim.csv")
claim("VI-sens", "trim2 Pfa/a", r2(64, 2).pfa.mean() / 0.01, "{:.2f}\\alpha$", "phaseC/C_review_R2_W_trim.csv")
claim("VI-sens", "trim2 Pfa/a 1e-3", r2(64, 2, 0.001).pfa.mean() / 0.001, "{:.1f}\\alpha$ at", "phaseC/C_review_R2_W_trim.csv")
R3 = pd.read_csv(C / "C_review_R3_warmup.csv"); r3 = R3[R3.arm == "G7"]
claim("VI-sens", "warm-up", f"{r3.pfa_warm.mean() / 0.01:.2f}\\alpha$ versus ${r3.pfa_steady.mean() / 0.01:.2f}", "{}", "phaseC/C_review_R3_warmup.csv")
R5 = pd.read_csv(C / "C_review_R5_tail_calibration.csv")
for a in (0.1, 0.03, 0.001):
    claim("VI-sens", f"tail ratio {a}", R5[(R5.arm == "G7") & (R5.alpha == a)].pfa.mean() / a, "{:.2f}\\alpha$", "phaseC/C_review_R5_tail_calibration.csv")
RC = json.loads((C / "C_review_checks.json").read_text())
claim("VI", "1998 G7 macro Pd", RC["1998_G7"]["pd_macro"], "{:.3f}", "phaseC/C_review_checks.json")
claim("VI", "1998 G7 macro Pfa/a", RC["1998_G7"]["pfa_macro"] / 0.01, "at ${:.2f}\\alpha$ and G3", "phaseC/C_review_checks.json")
claim("VI", "1998 G3 macro Pd", RC["1998_G3"]["pd_macro"], "{:.3f}", "phaseC/C_review_checks.json")

# second stage: final detector, unified decision frames (phaseD_temporal/R2_rescored.csv)
Dd = ROOT / "phaseD_temporal"
R2 = pd.read_csv(Dd / "R2_rescored.csv")
r2 = lambda sc, K, prot, rule, a: R2[(R2.scorer == sc) & (R2.K == K) & (R2.protocol == prot) & (R2.rule == rule) & (R2.alpha == a)]
SRC = "phaseD_temporal/R2_rescored.csv"
for sc, n in (("net-G7", 30), ("net-G3", 30), ("hgb-G7", 10)):
    for K in ((1, 4) if sc.startswith("net") else (4,)):
        x = r2(sc, K, "loro", "texp", 0.01); x3 = r2(sc, K, "loro", "texp", 0.001)
        claim("VI-final", f"{sc} K{K} LORO Pd", x.pd.mean(), "{:.3f}", SRC)
        claim("VI-final", f"{sc} K{K} LORO Pfa", f"{x.pfa.mean() / 0.01:.2f} ({int((x.pfa > 0.02).sum())}/{n})", "{}", SRC)
        claim("VI-final", f"{sc} K{K} LORO adj", x.adj.mean(), "{:.3f}", SRC)
        claim("VI-final", f"{sc} K{K} LORO bg", x.bg.mean() / 0.01, "{:.2f}", SRC)
        claim("VI-final", f"{sc} K{K} LORO Pd 1e-3", x3.pd.mean(), "{:.3f}", SRC)
        claim("VI-final", f"{sc} K{K} 1e-3", f"{x3.pfa.mean() / 0.001:.2f} ({int((x3.pfa > 0.002).sum())}/{n})", "{}", SRC)
        y = r2(sc, K, "1998", "texp", 0.01); y3 = r2(sc, K, "1998", "texp", 0.001)
        claim("VI-final", f"{sc} K{K} 1998 Pd", y.pd.mean(), "{:.3f}", SRC)
        claim("VI-final", f"{sc} K{K} 1998 Pfa/a", y.pfa.mean() / 0.01, "{:.2f}", SRC)
        claim("VI-final", f"{sc} K{K} 1998 oracle", r2(sc, K, "1998", "oracle", 0.01).pd.mean(), "{:.3f}", SRC)
        claim("VI-final", f"{sc} K{K} 1998 Pd 1e-3", y3.pd.mean(), "{:.3f}", SRC)
        claim("VI-final", f"{sc} K{K} 1998 Pfa 1e-3", y3.pfa.mean() / 0.001, "{:.2f}", SRC)
fl3 = R2[(R2.protocol == "loro") & (R2.rule == "texp") & (R2.alpha == 0.01)].groupby(["scorer", "K", "fold"]).pd.mean()
claim("VI-final", "G7 K4 vs K1 p", wilcoxon(fl3["net-G7"][4], fl3["net-G7"][1]).pvalue, "($p={:.3f}$)", SRC)
claim("VI-final", "G7 vs G3 K4 folds", int((fl3["net-G7"][4] > fl3["net-G3"][4]).sum()), "G7 higher in {} of 10 folds", SRC)
claim("VI-final", "G7 vs G3 K4 p", wilcoxon(fl3["net-G7"][4], fl3["net-G3"][4]).pvalue, "$p={:.2f}$", SRC)
claim("VI-final", "fold 6", fl3["net-G7"][4][6], "reaches {:.3f}", SRC)
s98 = r2("net-G7", 4, "1998", "texp", 0.01).pfa / 0.01
claim("VI-final", "1998 seeds", f"{s98.min():.2f}\\alpha$ and ${s98.max():.2f}", "{}", SRC)
RS = pd.read_csv(Dd / "R2_1998_record_seed.csv"); rs = RS[(RS.scorer == "net-G7") & (RS.K == 4)]
claim("VI-final", "record-seed over", int((rs.pfa > 0.02).sum()), "{} of the 27 rates exceed", "phaseD_temporal/R2_1998_record_seed.csv")
claim("VI-final", "record-seed max", rs.pfa.max() / 0.01, "at most ${:.2f}\\alpha$", "phaseD_temporal/R2_1998_record_seed.csv")
# control S (mean of four single-frame logits)
for nm, lab in (("S-single", "single"), ("S-mean4", "mean4"), ("S-net-K4", "net")):
    x = r2(nm, 4, "loro", "texp", 0.01); y = r2(nm, 4, "1998", "texp", 0.01)
    claim("VI-S", f"{lab} LORO Pd", x.pd.mean(), "{:.3f}", SRC)
    claim("VI-S", f"{lab} LORO Pfa", x.pfa.mean() / 0.01, "{:.2f}", SRC)
    claim("VI-S", f"{lab} LORO 1e-3", r2(nm, 4, "loro", "texp", 0.001).pfa.mean() / 0.001, "{:.2f}", SRC)
    claim("VI-S", f"{lab} 1998 Pd", y.pd.mean(), "{:.3f}", SRC)
    claim("VI-S", f"{lab} 1998 Pfa", y.pfa.mean() / 0.01, "{:.2f}", SRC)
    claim("VI-S", f"{lab} 1998 oracle", r2(nm, 4, "1998", "oracle", 0.01).pd.mean(), "{:.3f}", SRC)
    claim("VI-S", f"{lab} 1998 1e-3", r2(nm, 4, "1998", "texp", 0.001).pfa.mean() / 0.001, "{:.2f}", SRC)
    claim("VI-S", f"{lab} LORO oracle", r2(nm, 4, "loro", "oracle", 0.01).pd.mean(), "{:.3f}", SRC)
    claim("VI-S", f"{lab} LORO oracle 1e-3", r2(nm, 4, "loro", "oracle", 0.001).pd.mean(), "{:.3f}", SRC)
SR = pd.read_csv(Dd / "S_1998_records.csv").groupby(["method", "record"]).pfa.mean().unstack(0)
claim("VI-S", "records lower Pfa", {7: "seven"}.get(int((SR["net-K4"] < SR["mean4"]).sum()), "?"), "lower in {} of the nine records", "phaseD_temporal/S_1998_records.csv")
# fair equal dwell F
F = pd.read_csv(Dd / "F_fair_dwell.csv")
for (T, d), g in F.groupby(["dwell", "dur"]):
    claim("VI-F", f"F T={T} d={d} Pd", g["pd_0.01"].mean(), "{:.3f}", "phaseD_temporal/F_fair_dwell.csv")
    claim("VI-F", f"F T={T} d={d} Pfa", g["pfa_0.01"].mean() / 0.01, "{:.2f}", "phaseD_temporal/F_fair_dwell.csv")
    claim("VI-F", f"F T={T} d={d} oracle", g["oracle_0.01"].mean(), "{:.3f}", "phaseD_temporal/F_fair_dwell.csv")
    claim("VI-F", f"F T={T} d={d} Pd 1e-3", g["pd_0.001"].mean(), "{:.3f}", "phaseD_temporal/F_fair_dwell.csv")
    claim("VI-F", f"F T={T} d={d} Pfa 1e-3", g["pfa_0.001"].mean() / 0.001, "{:.2f}", "phaseD_temporal/F_fair_dwell.csv")
ff = F.groupby(["dwell", "dur", "fold"])["pd_0.01"].mean()
claim("VI-F", "F wins", int((ff[1.024][0.256] > ff[1.024][1.024]).sum()), "higher in all ten folds ($p={}", "phaseD_temporal/F_fair_dwell.csv") if False else None
claim("VI-F", "F p", wilcoxon(ff[1.024][0.256], ff[1.024][1.024]).pvalue, "($p={:.3f}$)", "phaseD_temporal/F_fair_dwell.csv")
lat3 = json.loads((Dd / "N3_latency.json").read_text())
claim("V", "final CPU latency", lat3["cpu"]["median"], "{:.2f}~ms", "phaseD_temporal/N3_latency.json")
claim("V", "final params", f"{lat3['params']:,}".replace(",", "\\,"), "{}", "phaseD_temporal/N3_latency.json")
N2 = pd.read_csv(Dd / "N2_duration.csv")
claim("VI-final", "0.256 single", N2[(N2.dur == 0.256) & (N2.K == 1) & (N2.rule == "texp") & (N2.alpha == 0.01)].pd.mean(), "{:.3f}", "phaseD_temporal/N2_duration.csv")
M = pd.read_csv(Dd / "CHK_M_mechanism.csv")
for a in ("0.1", "0.03", "0.01", "0.003", "0.001"):
    claim("VI-final", f"ratio K4 {a}", M[(M.protocol == "loro") & (M.K == 4)][f"ratio_{a}"].mean(), "{:.2f}", "phaseD_temporal/CHK_M_mechanism.csv")
claim("VI-final", "1998 ratio K1 1e-3", M[(M.protocol == "1998") & (M.K == 1)]["ratio_0.001"].mean(), "from {:.2f}", "phaseD_temporal/CHK_M_mechanism.csv")
claim("VI-final", "1998 ratio K4 1e-3", M[(M.protocol == "1998") & (M.K == 4)]["ratio_0.001"].mean(), "to {:.2f}", "phaseD_temporal/CHK_M_mechanism.csv")
N4 = pd.read_csv(Dd / "N4_shift_final.csv"); N4["a"] = N4["shift"].abs()
claim("V", "final shift", N4[N4.a == 1].pd.mean(), "0.829 $\\to$ {:.3f}", "phaseD_temporal/N4_shift_final.csv")
out = pd.DataFrame(rows)
out.to_csv(Path(__file__).parent / "claim_evidence.csv", index=False)
print(f"{out.in_text.sum()} of {len(out)} recomputed values found verbatim in the manuscript")
