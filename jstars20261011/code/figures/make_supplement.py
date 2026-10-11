"""Build the supplementary PDF source (tables from stored CSVs, ablation figure) for the JSTARS manuscript."""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / "phaseC"
A = ROOT / "phaseA_20261008" / "out"
OUT = Path(__file__).resolve().parent / "supplementary" / "supplement"
OUT.mkdir(parents=True, exist_ok=True)
shutil.copy(Path(__file__).resolve().parent / "source" / "figures" / "fig_ablation.pdf", OUT / "fig_ablation.pdf")


def table(caption, label, cols, rows, note=None, align=None):
    align = align or "l" + "c" * (len(cols) - 1)
    s = [r"\begin{table}[!ht]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\footnotesize",
         rf"\begin{{tabular}}{{@{{}}{align}@{{}}}}", r"\toprule", " & ".join(cols) + r"\\", r"\midrule"]
    s += [" & ".join(map(str, r)) + r"\\" for r in rows]
    s += [r"\bottomrule", r"\end{tabular}"]
    if note:
        s += [r"\\[3pt]", rf"\parbox{{0.9\columnwidth}}{{\scriptsize {note}}}"]
    s += [r"\end{table}", ""]
    return "\n".join(s)


parts = []

# S1 parameters
par = [("G7", "\\sargb, 7 descriptors", 368, 27841), ("G3", "\\sargb, 3 descriptors", 368, 27841),
       ("D7", "7-vector per cell", 218, 18241), ("P7", "learned $7\\rightarrow3$", 98, 10582),
       ("G7 w/o SCN", "\\sargb, 7 descriptors", 368, 27841), ("G7 abs.\\ context", "\\sargb, 7 descriptors", 332, 25537),
       ("G7 range-ordered", "\\sargb, 7 descriptors", 72, 8897)]
parts.append(table("Parameters and Per-Cell Head Input Width of Every Network Arm.", "tab:s_params",
                   ["Arm", "Input", "Head input", "Parameters"],
                   [(a, b, c, f"{d:,}".replace(",", "\\,")) for a, b, c, d in par]))

# S2 W / trim sensitivity
r2 = pd.read_csv(C / "C_review_R2_W_trim.csv")
r2["over"] = r2.pfa > 2 * r2.alpha
rows = []
for (arm, W, trim), g in r2.groupby(["arm", "W", "trim"], sort=False):
    v = []
    for al in (1e-2, 1e-3):
        h = g[g.alpha == al]
        v += [f"{h.pd.mean():.3f}", f"{h.pfa.mean() / al:.2f}", f"{int(h.over.sum())}/{len(h)}"]
    rows.append([arm, W, trim] + v)
parts.append(table("Tail-CFAR Settings (IPIX1993 LORO, Grid-Direction Network). $W$: Reference Frames; Trim: Highest Cells Omitted per Reference Frame.",
                   "tab:s_wtrim", ["Arm", "$W$", "Trim", "$\\hat P_d$", "$\\hat P_{fa}/\\alpha$", "Runs $>2\\alpha$",
                                   "$\\hat P_d$", "$\\hat P_{fa}/\\alpha$", "Runs $>2\\alpha$"], rows,
                   note="Columns 4--6: $\\alpha=10^{-2}$; columns 7--9: $\\alpha=10^{-3}$. The design uses $W=64$, trim 1."))

# S3 tail calibration
r5 = pd.read_csv(C / "C_review_R5_tail_calibration.csv")
alphas = sorted(r5.alpha.unique(), reverse=True)
rows = []
for arm, g in r5.groupby("arm", sort=False):
    rows.append([arm] + [f"{g[g.alpha == a].pfa.mean() / a:.2f}" for a in alphas])
parts.append(table("Realized-to-Nominal Ratio $\\hat P_{fa}/\\alpha$ of the Tail CFAR With $c$ Set at $\\alpha=10^{-2}$ (IPIX1993 LORO, Mean Over 30 Runs).",
                   "tab:s_tailcal", ["Arm"] + [f"$\\alpha={a:g}$" for a in alphas], rows,
                   note="A ratio that grows as $\\alpha$ falls indicates a score tail heavier than the exponential model."))

# S4 warm-up and common mask
r3 = pd.read_csv(C / "C_review_R3_warmup.csv").groupby("arm")[["pfa_warm", "pfa_steady"]].mean()
r1 = pd.read_csv(C / "C_review_R1_common_mask.csv").groupby(["arm", "rule"])[["pd", "pfa", "pfa_adj", "pfa_bg"]].mean()
rows = []
for (arm, rule), v in r1.iterrows():
    rows.append([arm, rule, f"{v.pd:.3f}", f"{v.pfa / 1e-2:.2f}", f"{v.pfa_adj:.3f}", f"{v.pfa_bg / 1e-2:.2f}"])
parts.append(table("Tail and Fixed Rules on the Common 1020 Frames per Record ($\\alpha=10^{-2}$, IPIX1993 LORO).",
                   "tab:s_common", ["Arm", "Rule", "$\\hat P_d$", "$\\hat P_{fa}/\\alpha$", "Adjacent $\\hat P_{fa}$", "Background $\\hat P_{fa}/\\alpha$"], rows,
                   note="Warm-up frames (fewer than 64 frames of history) versus steady state, tail rule: " +
                   "; ".join(f"{a} {r.pfa_warm / 1e-2:.2f}$\\alpha$ versus {r.pfa_steady / 1e-2:.2f}$\\alpha$" for a, r in r3.iterrows()) + "."))

# S5 nested selection
ns = pd.read_csv(A / "B_nested_selection.csv")
no = pd.read_csv(A / "B_nested_outer.csv")
rows = []
for _, r in ns.iterrows():
    o = no[no.outer == r.outer].set_index("alpha")
    rows.append([int(r.outer), r.D1, r.D2, r.D3, f"{o.loc[0.01, 'pd']:.3f}", f"{o.loc[0.01, 'pfa'] / 1e-2:.2f}",
                 f"{o.loc[0.001, 'pd']:.3f}", f"{o.loc[0.001, 'pfa'] / 1e-3:.1f}"])
parts.append(table("Nested Selection Within the HGB Development Pipeline. D1: Normalization; D2: Context; D3: Threshold Rule (texp: Exponential Tail; cmap: Empirical Clutter Map).",
                   "tab:s_nested", ["Outer fold", "D1", "D2", "D3", "$\\hat P_d$ ($10^{-2}$)", "$\\hat P_{fa}/\\alpha$",
                                    "$\\hat P_d$ ($10^{-3}$)", "$\\hat P_{fa}/\\alpha$"], rows))

# S6 hull feasibility
h = pd.read_csv(C / "C_classical_hull.csv")
rows = []
for _, r in h.iterrows():
    rows.append([r.protocol.upper() if r.protocol == "loro" else "IPIX1998", f"{r.alpha:g}", int(r.fold) if r.protocol == "loro" else "--",
                 int(r.layer), "yes" if r.feasible else "no", f"{r.val_pfa:.4f}", f"{r.pd:.3f}", f"{r.pfa:.4f}"])
parts.append(table("Convex-Hull Reference: Peeling Layer, Calibration Feasibility, and Test Results.", "tab:s_hull",
                   ["Protocol", "$\\alpha$", "Fold", "Layer", "Feasible", "Val.\\ $P_{fa}$", "$\\hat P_d$", "$\\hat P_{fa}$"], rows,
                   note="Feasible: a peeling depth reaches a validation rate of at most $\\alpha$; otherwise the unpeeled outer hull (layer 0) is used."))

# S7 IPIX1998 micro/macro and matched inputs
mi = pd.read_csv(C / "C_matched_inputs.csv")
rows = []
for (sc, prot, rule, al), g in mi.groupby(["scorer", "protocol", "rule", "alpha"]):
    over = int((g.pfa > 2 * al).sum())
    rows.append([sc, "LORO" if prot == "loro" else "IPIX1998", rule, f"{al:g}", f"{g.pd.mean():.3f}",
                 "--" if rule == "oracle" else f"{g.pfa.mean() / al:.2f}",
                 "--" if rule == "oracle" else f"{over}/{len(g)}"])
parts.append(table("Matched-Input Comparison: Identical 72-Channel Range-Ordered Tensor Given to RadarGrid-CS and HGB.", "tab:s_matched",
                   ["Scorer", "Protocol", "Rule", "$\\alpha$", "$\\hat P_d$", "$\\hat P_{fa}/\\alpha$", "Runs $>2\\alpha$"], rows,
                   note="oracle: $P_d$ at equal realized test $P_{fa}=\\alpha$ (separability, not deployable). texp: tail CFAR; fixed: validation threshold. "
                   "IPIX1998 G7 (grid direction), pooled versus record-averaged: $\\hat P_d$ 0.374 versus 0.348, $\\hat P_{fa}/\\alpha$ 2.36 versus 2.42; G3: 0.247 versus 0.223, 1.84 versus 1.88."))

# S8-S11 final detector (phaseD_temporal)
Dd = ROOT / "phaseD_temporal"
N2 = pd.read_csv(Dd / "N2_duration.csv")
t = N2[(N2.rule == "texp") & (N2.alpha == 1e-2)]
rows = []
for d, g in t.groupby("dur"):
    r = [f"{d:.3f}"]
    for K in (1, 2, 4, 8):
        h = g[g.K == K]
        r.append(f"{h.pd.mean():.3f} ({h.pfa.mean() / 1e-2:.2f}; {int((h.pfa > 0.02).sum())}/30)")
    rows.append(r)
parts.append(table("Final Detector Across Observation Durations (IPIX1993 LORO, G7, Tail CFAR, $\\alpha=10^{-2}$): $\\hat P_d$ ($\\hat P_{fa}/\\alpha$; Runs $>2\\alpha$).",
                   "tab:s_duration", ["$d$ (s)", "$K=1$", "$K=2$", "$K=4$", "$K=8$"], rows))
Q = pd.read_csv(Dd / "CHK_Q_confound.csv")
rows = []
for (T, d, K), g in Q.groupby(["dwell", "dur", "K"]):
    f = g.groupby("fold").pd.mean()
    rows.append([f"{T:.3f}", f"{d:.3f}", K, int(g.n_train_frames.iloc[0]), f"{g.pd.mean():.3f}", f"{g.pfa.mean() / 1e-2:.2f}"])
parts.append(table("Equal Dwell With Matched Training Frames (121 per Sequence; IPIX1993 LORO, G7, Tail CFAR, $\\alpha=10^{-2}$).",
                   "tab:s_dwell", ["Dwell (s)", "$d$ (s)", "$K$", "Train frames", "$\\hat P_d$", "$\\hat P_{fa}/\\alpha$"], rows))
M = pd.read_csv(Dd / "CHK_M_mechanism.csv")
rows = []
for (prot, K), g in M.groupby(["protocol", "K"]):
    rows.append(["IPIX1993 LORO" if prot == "loro" else "IPIX1998", K] + [f"{g[f'ratio_{a:g}'].mean():.2f}" for a in alphas])
parts.append(table("Tail-CFAR Calibration of the Final Detector: $\\hat P_{fa}/\\alpha$ With $c$ Set at $10^{-2}$ (G7, $d=0.512$~s).",
                   "tab:s_tailcal_final", ["Protocol", "$K$"] + [f"$\\alpha={a:g}$" for a in alphas], rows))
N4 = pd.read_csv(Dd / "N4_shift_final.csv")
rows = [[s, f"{g.pd.mean():.3f}", f"{g.pfa.mean() / 1e-2:.2f}"] for s, g in N4.groupby("shift")]
parts.append(table("Cyclic Range-Shift Test of the Final Detector (IPIX1993 LORO, G7, $K=4$, Tail CFAR, $\\alpha=10^{-2}$).",
                   "tab:s_shift_final", ["Shift $s$", "$\\hat P_d$", "$\\hat P_{fa}/\\alpha$"], rows))

# S12 first design at 1e-3 (moved from the manuscript)
parts.append(r"""\begin{table}[!ht]
\centering
\caption{First Design and Reference Detectors, IPIX1993 LORO at $\alpha=10^{-3}$ (Descriptive; Tail CFAR Calibrated at $10^{-2}$ Unless Stated).}
\label{tab:s_arms3}
\footnotesize
\begin{tabular}{@{}lccc@{}}
\toprule
Arm & $\hat P_d$ & $\hat P_{fa}/\alpha$ & Runs $>2\alpha$\\
\midrule
G7 & 0.455 & 2.9 & 9/30\\
G3 & 0.419 & 5.6 & 20/30\\
D7 & 0.460 & 2.9 & 8/30\\
P7 & 0.453 & 2.1 & 11/30\\
G7 w/o SCN & 0.451 & 3.2 & 6/30\\
G7 abs.\ context & 0.462 & 4.0 & 8/30\\
G7 fixed threshold$^{a}$ & 0.465 & 1.7 & 10/30\\
\midrule
Amplitude ratio (RAA) & 0.028 & 0.6 & 1/10\\
Convex hull$^{a,b}$ & 0.078 & 6.2 & 7/10\\
HGB-G3 & 0.434 & 3.2 & 5/10\\
HGB-G7 & 0.524 & 1.5 & 3/10\\
\midrule
Range-ordered RadarGrid-CS & 0.514 & 2.0 & 14/30\\
Range-ordered HGB & 0.509 & 1.5 & 2/10\\
\bottomrule
\end{tabular}
\\[3pt]
\parbox{0.9\columnwidth}{\scriptsize $^{a}$Threshold calibrated on the validation record at $10^{-3}$. $^{b}$No peeling depth reaches a validation rate of at most $10^{-3}$ in 7 of 10 folds; the unpeeled outer hull (layer 0) is then used, so these entries are not calibrated operating points.}
\end{table}
""")
# S13 IPIX1998 record x seed, final detector
RS = pd.read_csv(Dd / "R2_1998_record_seed.csv")
x = RS[(RS.scorer == "net-G7") & (RS.K == 4)]
rows = []
for r, g in x.groupby("record"):
    g = g.sort_values("seed")
    rows.append([f"R{r}"] + [f"{v.pd:.3f} / {v.pfa / 1e-2:.2f}" for _, v in g.iterrows()])
parts.append(table("Final Detector on IPIX1998 per Record and Seed: $\\hat P_d$ / $\\hat P_{fa}/\\alpha$ (G7, $K=4$, Tail CFAR, $\\alpha=10^{-2}$).",
                   "tab:s_recseed", ["Record", "Seed 0", "Seed 1", "Seed 2"], rows,
                   note="Decision frames 8--116 of each window. Five of the 27 record--seed rates exceed $2\\alpha$, all on R6, R8, and R9; pooled over the nine records the rate is 1.15--1.18$\\alpha$ per seed."))
shutil.copy(Path(__file__).resolve().parent / "figures" / "fig_loro.pdf", OUT / "fig_loro.pdf")
shutil.copy(Path(__file__).resolve().parent / "figures" / "fig_crosssite.pdf", OUT / "fig_crosssite.pdf")

doc = r"""\documentclass[journal,onecolumn]{IEEEtran}
\usepackage[T1]{fontenc}
\usepackage{newtxtext,newtxmath}
\usepackage{booktabs,graphicx,amsmath}
\newcommand{\sargb}{SA-RGB}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\begin{document}
\title{Supplementary Material for ``Short-Observation Seven-Feature Detection of Small Floating Targets in Sea Clutter Under Record-Disjoint Evaluation''}
\author{Shiji Fan, Daniel Albuquerque, Feng Luo, Pedro Pinho, and Ming Shen}
\maketitle
This document collects the ablation figure of the first design, its sensitivity and audit tables (S1--S7), and the duration, equal-dwell, calibration, and shift tables of the final detector (S8--S11), the first-design results moved from the manuscript (Figs.~S2--S3 and Table~S12), and the IPIX1998 record-by-seed results of the final detector (Table~S13). All values are recomputed from the per-run files in the supplementary archive; table and figure numbers carry the prefix S.

\begin{figure}[!ht]
\centering
\includegraphics[width=\textwidth]{fig_ablation.pdf}
\caption{Arms at $\alpha=10^{-2}$ (IPIX1993 LORO, tail CFAR unless stated). (a) Test $P_d$, mean and standard error over the ten fold means. (b) Runs, of 30, with test $P_{fa}>2\alpha$. (c) Adjacent-cell $P_{fa}$.}
\label{fig:s_ablation}
\end{figure}

\begin{figure}[!ht]
\centering
\includegraphics[width=\textwidth]{fig_loro.pdf}
\caption{First design, per-fold IPIX1993 LORO results at $\alpha=10^{-2}$ (mean over three seeds; the convex hull is deterministic). (a) Realized test $P_{fa}$; $P_{fa}=0$ is drawn below the axis break. (b) Test $P_d$. Shaded: folds 1 and 6, whose target cell index is absent from the training targets.}
\label{fig:s_loro}
\end{figure}

\begin{figure}[!ht]
\centering
\includegraphics[width=\textwidth]{fig_crosssite.pdf}
\caption{First design on IPIX1998 at $\alpha=10^{-2}$, trained and calibrated on IPIX1993. (a) Test $P_d$ per record; shaded: R8 (15 m) and R9 (9 m). (b) Realized test $P_{fa}$ per record. (c) Second stage: IPIX1993 LORO $P_d$ against IPIX1998 $P_{fa}/\alpha$ for G7 with different context inputs to the cell head (single observations); green: within $2\alpha$.}
\label{fig:s_crosssite}
\end{figure}

""" + "\n".join(parts) + "\n\\end{document}\n"
(OUT / "Supplementary_Material.tex").write_text(doc, encoding="utf-8")
print("written", OUT / "Supplementary_Material.tex")
