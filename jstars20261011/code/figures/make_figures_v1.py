"""Figures for the JSTARS rewrite. Every panel is read from phaseA_20261008/out or phaseC outputs."""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "phaseA_20261008" / "out"
C = ROOT / "phaseC"
F = Path(__file__).resolve().parent / "figures"
F.mkdir(exist_ok=True)

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
                     "lines.linewidth": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False, "mathtext.fontset": "stix"})
BLUE, ORANGE, AQUA, VIOLET, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#898781"
FEATS = ["Hurst", "RAA", "RDPH", "RVE", "RI", "NR", "MCS"]


def save(fig, name):
    fig.savefig(F / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(F / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def tag(ax, s):
    ax.text(-0.16, 1.04, f"({s})", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


# ---------------------------------------------------------------- F1 pipeline
def fig_pipeline():
    sys.path.insert(0, str(C)); sys.path.insert(0, str(ROOT / "phaseA_20261008"))
    from common import load
    from rgcs import fit_percentiles, quantize, render
    data = load()
    lo, hi = fit_percentiles(np.concatenate([data[g]["X"].reshape(-1, 7) for g in range(2, 10)]))
    img = render(quantize(data[5]["X"][0, 100][None], lo, hi), "G7")[0].transpose(1, 2, 0)
    fig = plt.figure(figsize=(7.16, 1.65))
    ax0 = fig.add_axes([0.0, 0.18, 0.30, 0.62])
    ax0.imshow(img, interpolation="nearest", aspect="equal")
    for k in range(14):
        r, c = k % 2, k // 2
        ax0.text(c * 16 + 15.2, r * 16 + 15.2, str(k + 1), fontsize=5.5, ha="right", va="bottom", color="white")
    t = data[5]["t"]
    ax0.add_patch(plt.Rectangle(((t // 2) * 16 - 0.5, (t % 2) * 16 - 0.5), 16, 16, fill=False, ec="white", lw=1.4,
                                ls="--"))
    ax0.set_xticks([]); ax0.set_yticks([])
    ax0.set_title("SA-RGB G7 grid (record 5, HH, frame 101)", fontsize=7)
    ax0.text(0, -0.17, "s1 = (RAA, RDPH, RVE)   s2 = (Hurst, RI, NR)\ns3 = (MCS, a, a)   s4 = (a, a, a)",
             transform=ax0.transAxes, fontsize=6, va="top")
    ax = fig.add_axes([0.31, 0.0, 0.69, 1.0]); ax.set_axis_off(); ax.set_xlim(0, 10); ax.set_ylim(0.25, 2.95)
    boxes = [(0.2, "Slot readout", "anchor lift\n$[I, I-a, |I-a|]$\n12 addresses/cell"),
             (2.15, "SCN", "per-frame,\nper-address\nmedian / IQR"),
             (4.1, "Relative context", "CUT minus\nneighbours,\npeak margin"),
             (6.05, "Shared cell head", "1$\\times$1 conv\n64-64-1, no\nabsolute position"),
             (8.0, "Tail CFAR", "clutter map (64 fr.),\nexp. tail, $c$ set\nat $10^{-2}$")]
    for x, title, body in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.75), 1.75, 1.6, boxstyle="round,pad=0.03,rounding_size=0.08",
                                    fc="#f4f4f2", ec="#52514e", lw=0.7))
        ax.text(x + 0.875, 2.12, title, ha="center", va="center", fontsize=7.5, fontweight="bold")
        ax.text(x + 0.875, 1.4, body, ha="center", va="center", fontsize=6.3, linespacing=1.25)
    for x in (1.95, 3.9, 5.85, 7.8):
        ax.add_patch(FancyArrowPatch((x, 1.55), (x + 0.2, 1.55), arrowstyle="-|>", mutation_scale=7, lw=0.8))
    ax.add_patch(FancyArrowPatch((-0.15, 1.55), (0.2, 1.55), arrowstyle="-|>", mutation_scale=7, lw=0.8))
    ax.text(8.875, 0.45, "decision per range cell:\nlogit $\\geq \\tau_n(\\alpha)$", ha="center", va="center",
            fontsize=6.3)
    ax.text(5.0, 2.75, "RadarGrid-CS: 27,841 parameters; 0.56 ms per frame on a desktop CPU", ha="center",
            fontsize=7, color="#52514e")
    save(fig, "fig1_pipeline")


# ---------------------------------------------------------------- F2 information vs controllability
def fig_diagnosis():
    s = pd.read_csv(A / "A2_per_subset.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.3))
    for rep, ls in (("framenorm", "-"), ("raw", "--")):
        lr = s[(s.model == "LR") & (s.rep == rep) & (~s.ctx)].groupby("k")
        hg = s[(s.model == "HGB") & (s.rep == rep) & (~s.ctx)].sort_values("k")
        lab = "frame-normalized" if rep == "framenorm" else "raw"
        axs[0].plot(lr.pdor_001.mean().index, lr.pdor_001.mean(), ls=ls, color=GRAY, marker="o", ms=3.5,
                    label=f"Linear, {lab} (mean over subsets)")
        axs[0].plot(hg.k, hg.pdor_001, ls=ls, color=BLUE, marker="s", ms=4, label=f"Nonlinear, {lab} (G3/G4/G5/G7)")
        axs[1].plot(lr.over2_001.mean().index, lr.over2_001.mean(), ls=ls, color=GRAY, marker="o", ms=3.5)
        axs[1].plot(hg.k, hg.over2_001, ls=ls, color=BLUE, marker="s", ms=4)
    hg = s[(s.model == "HGB") & (s.rep == "framenorm") & (~s.ctx)].sort_values("k")
    offs = {"G3": (-16, -1), "G4": (5, -9), "G5": (-6, 6), "G7": (-8, 6)}
    for _, r in hg.iterrows():
        axs[0].annotate(r.subset, (r.k, r.pdor_001), textcoords="offset points", xytext=offs[r.subset], fontsize=6.5)
    axs[0].set_xlabel("Number of features"); axs[0].set_ylabel(r"$P_d$ at test $P_{fa}=10^{-2}$")
    axs[1].set_xlabel("Number of features"); axs[1].set_ylabel(r"Folds with test $P_{fa}>2\alpha$ (of 10)")
    axs[0].set_ylim(0, 0.6); axs[1].set_ylim(0, 3.5)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.06))
    for ax, t in zip(axs, "ab"):
        ax.set_xticks(range(1, 8)); tag(ax, t); ax.grid(axis="y", lw=0.3, color="#e0e0dc")
    save(fig, "fig2_information_controllability")


# ---------------------------------------------------------------- F3 failure anatomy
def fig_anatomy():
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.3), gridspec_kw=dict(wspace=0.45))
    d = pd.read_csv(A / "A6_drift_1998_vs_1993.csv"); d = d[d.rep == "raw"].set_index("feature").loc[FEATS]
    o93 = np.where(d.auc93 >= 0.5, d.auc93, 1 - d.auc93)
    o98 = np.where(d.auc93 >= 0.5, d.auc98, 1 - d.auc98)
    y = np.arange(7)
    axs[0].hlines(y, o93, o98, color="#c3c2b7", lw=1.5)
    axs[0].plot(o93, y, "o", color=BLUE, ms=5, label="IPIX1993")
    axs[0].plot(o98, y, "D", color=ORANGE, ms=4.5, label="IPIX1998")
    axs[0].axvline(0.5, color=GRAY, lw=0.6, ls=":")
    axs[0].set_yticks(y, FEATS); axs[0].invert_yaxis(); axs[0].set_xlim(0.1, 1.0)
    axs[0].set_xlabel("Target vs. clutter AUC\n(1993 orientation)"); axs[0].legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.45, 1.0), handletextpad=0.2, columnspacing=0.8)
    a3 = pd.read_csv(A / "A3_detector_pdpfa.csv"); g = a3[a3.method == "SA-RGB G7"].sort_values("fold")
    adj = g["pfa_adj_0.01"] * 2 / 13; oth = g["pfa_0.01"] - adj
    axs[1].bar(g.fold, adj, color=ORANGE, width=0.7, label="adjacent cells", edgecolor="white", lw=0.5)
    axs[1].bar(g.fold, oth, bottom=adj, color="#c3c2b7", width=0.7, label="other $H_0$ cells", edgecolor="white", lw=0.5)
    axs[1].axhline(0.02, color="k", lw=0.6, ls="--"); axs[1].text(10.6, 0.021, r"$2\alpha$", fontsize=6.5, va="bottom", ha="right")
    axs[1].set_xticks(range(1, 11)); axs[1].set_xlabel("LORO fold (test record)")
    axs[1].set_ylabel(r"Test $P_{fa}$ ($\alpha=10^{-2}$)"); axs[1].legend(frameon=False, loc="upper right")
    axs[1].set_title("YOLO11n SA-RGB G7 (historical)", fontsize=7)
    h = pd.read_csv(A / "A4_frame_hits.csv"); h = h[~h.fold.isin([1, 6])]
    bins = [-60, 0, 3, 6, 10, 15, 60]; h["b"] = pd.cut(h.scr_db, bins)
    lab = ["≤0", "0–3", "3–6", "6–10", "10–15", ">15"]
    for m, col, mk, nm in (("YOLO11n SA-RGB G7", GRAY, "o", "YOLO11n (position-bound)"),
                           ("HGB-G7", BLUE, "s", "position-free scorer")):
        v = h[h.method == m].groupby("b", observed=True).hit.mean().values
        axs[2].plot(range(6), v, marker=mk, color=col, ms=4, label=nm)
    axs[2].set_xticks(range(6), lab); axs[2].set_xlabel("Per-frame SCR proxy (dB)")
    axs[2].set_ylabel(r"$P_d$ at calibrated $\tau$"); axs[2].set_ylim(0, 1.05)
    axs[2].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    for ax, t in zip(axs, "abc"):
        tag(ax, t)
    save(fig, "fig3_failure_anatomy")


# ---------------------------------------------------------------- F4 pre-registered LORO
def fig_loro():
    L = pd.read_csv(C / "C_loro_long.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.3))
    d = L[(L.rule == "texp64") & (L.alpha == 0.01)]
    for arm, col, mk, dx in (("G7", BLUE, "o", 0.12), ("G3", ORANGE, "D", -0.12)):
        q = d[d.arm == arm].groupby("fold").pd
        axs[0].errorbar(q.mean().index + dx, q.mean(), yerr=[q.mean() - q.min(), q.max() - q.mean()], fmt=mk,
                        color=col, ms=4, capsize=1.5, lw=0.8, label=arm)
    axs[0].axvspan(0.5, 1.5, color="#f0efec", zorder=0); axs[0].axvspan(5.5, 6.5, color="#f0efec", zorder=0)
    axs[0].set_xticks(range(1, 11)); axs[0].set_xlabel("LORO fold (test record)")
    axs[0].set_ylabel(r"Test $P_d$ (tail CFAR, $\alpha=10^{-2}$)"); axs[0].legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    axs[0].set_ylim(-0.03, 1.05)
    d = L[(L.arm == "G7") & (L.alpha == 0.01)]
    for rule, col, mk, dx, nm in (("texp64", BLUE, "o", 0.12, "tail CFAR"), ("fixed", VIOLET, "^", -0.12, "fixed threshold")):
        q = d[d.rule == rule].groupby("fold").pfa
        m = q.mean() / 0.01
        axs[1].errorbar(m.index + dx, m, yerr=[m - q.min() / 0.01, q.max() / 0.01 - m], fmt=mk, color=col, ms=4,
                        capsize=1.5, lw=0.8, label=nm)
    axs[1].axhline(1, color=GRAY, lw=0.6, ls=":"); axs[1].axhline(2, color="k", lw=0.6, ls="--")
    axs[1].set_yscale("log"); axs[1].set_xticks(range(1, 11)); axs[1].set_xlabel("LORO fold (test record)")
    axs[1].set_ylabel(r"G7 test $P_{fa}/\alpha$"); axs[1].legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    for ax, t in zip(axs, "ab"):
        tag(ax, t)
    save(fig, "fig4_loro_preregistered")


# ---------------------------------------------------------------- F5 ablations
def fig_ablation():
    L = pd.read_csv(C / "C_loro_long.csv"); L = L[L.alpha == 0.01]
    rows = [("G7", "texp64", "G7 (full)"), ("G3", "texp64", "G3"), ("D7", "texp64", "D7 (vector)"),
            ("P7", "texp64", r"P7 (learned 7$\rightarrow$3)"), ("G7-noSCN", "texp64", "G7 w/o SCN"),
            ("G7-absC", "texp64", "G7 abs. context"), ("G7", "fixed", "G7 fixed threshold")]
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.4), sharey=True, gridspec_kw=dict(wspace=0.12))
    y = np.arange(len(rows))
    for i, (arm, rule, nm) in enumerate(rows):
        d = L[(L.arm == arm) & (L.rule == rule)]
        f = d.groupby("fold")
        col = BLUE if i == 0 else (ORANGE if arm == "G3" else GRAY)
        se = f.pd.mean().std(ddof=1) / np.sqrt(10)
        axs[0].barh(i, f.pd.mean().mean(), xerr=se, color=col, height=0.6, error_kw=dict(lw=0.7, capsize=1.5))
        axs[0].text(f.pd.mean().mean() + se + 0.02, i, f"{f.pd.mean().mean():.3f}", va="center", fontsize=6.5)
        axs[1].barh(i, int((d.pfa > 0.02).sum()), color=col, height=0.6)
        axs[1].text(int((d.pfa > 0.02).sum()) + 0.2, i, f"{int((d.pfa > 0.02).sum())}/30", va="center", fontsize=6.5)
        axs[2].barh(i, d.pfa_adj.mean(), color=col, height=0.6)
        axs[2].text(d.pfa_adj.mean() + 0.002, i, f"{d.pfa_adj.mean():.3f}", va="center", fontsize=6.5)
    axs[0].set_yticks(y, [r[2] for r in rows]); axs[0].invert_yaxis()
    axs[0].set_xlabel(r"Test $P_d$ ($\pm$ s.e. over folds)"); axs[0].set_xlim(0, 0.9)
    axs[1].set_xlabel(r"Runs with $P_{fa}>2\alpha$"); axs[1].set_xlim(0, 9)
    axs[2].set_xlabel("Adjacent-cell $P_{fa}$"); axs[2].set_xlim(0, 0.085)
    for ax, t in zip(axs, "abc"):
        tag(ax, t)
    save(fig, "fig5_ablation")


# ---------------------------------------------------------------- F6 cross-site
def fig_crosssite():
    R = json.loads((C / "runs_1998" / "summary.json").read_text())
    meta = json.load(open(A / "ipix1998_7feat.json", encoding="utf8"))
    res_m = {"19980223_173317": 15, "19980223_173950": 9}
    recs = sorted(R[0]["0.01"]["pd_by_record"], key=int)
    lab = [f"R{k}" + (f"\n{res_m[meta['records'][int(k) - 1]['stem']]} m" if meta['records'][int(k) - 1]['stem'] in res_m else "") for k in recs]
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.5), gridspec_kw=dict(width_ratios=[1.35, 1.35, 1], wspace=0.38))
    x = np.arange(len(recs))
    for arm, col, mk, dx in (("G7", BLUE, "o", 0.15), ("G3", ORANGE, "D", -0.15)):
        pdv = np.array([np.mean([r["0.01"]["pd_by_record"][k] for r in R if r["arm"] == arm]) for k in recs])
        pfv = np.array([np.mean([r["0.01"]["pfa_by_record"][k] for r in R if r["arm"] == arm]) for k in recs])
        axs[0].plot(x + dx, pdv, mk, color=col, ms=4, label=arm)
        axs[1].plot(x + dx, pfv / 0.01, mk, color=col, ms=4, label=arm)
    axs[1].axhline(2, color="k", lw=0.6, ls="--"); axs[1].axhline(1, color=GRAY, lw=0.6, ls=":")
    for ax in axs[:2]:
        ax.set_xticks(x, lab, fontsize=6.3)
    axs[0].set_ylabel(r"Test $P_d$ ($\alpha=10^{-2}$)"); axs[1].set_ylabel(r"Test $P_{fa}/\alpha$")
    axs[0].legend(frameon=False, loc="upper right"); axs[0].set_ylim(0, 0.8)
    e = pd.read_csv(C / "explore_context" / "summary.csv")
    e = e[e.arm == "G7"].pivot_table(index="variant", columns="protocol", values=["pd1e2", "pfa1e2"])
    names = {"grid3x3": "3$\\times$3 grid\n(fixed design)", "none": "none", "r1": "$\\pm$1", "r2": "$\\pm$2",
             "r12": "$\\pm$1,$\\pm$2", "margin1": "peak margin\n$\\pm$1"}
    off = {"grid3x3": (-46, -40), "none": (5, 3), "r1": (4, 2), "r2": (4, 2), "r12": (4, 2), "margin1": (4, -4)}
    for v, r in e.iterrows():
        col = BLUE if v == "grid3x3" else (AQUA if v == "margin1" else GRAY)
        axs[2].plot(r[("pd1e2", "loro")], r[("pfa1e2", "1998")] / 0.01, "o", color=col, ms=5)
        axs[2].annotate(names[v], (r[("pd1e2", "loro")], r[("pfa1e2", "1998")] / 0.01), textcoords="offset points",
                        xytext=off[v], fontsize=5.8,
                        arrowprops=dict(arrowstyle="-", lw=0.5, color=GRAY) if v == "grid3x3" else None)
    axs[2].axhline(2, color="k", lw=0.6, ls="--")
    axs[2].set_xlabel(r"IPIX1993 LORO $P_d$"); axs[2].set_ylabel(r"IPIX1998 $P_{fa}/\alpha$")
    axs[2].set_title("G7 context variants (exploratory)", fontsize=7); axs[2].set_xlim(0.5, 0.72)
    for ax, t in zip(axs, "abc"):
        tag(ax, t)
    save(fig, "fig6_cross_site")


if __name__ == "__main__":
    for f in (fig_pipeline, fig_diagnosis, fig_anatomy, fig_loro, fig_ablation, fig_crosssite):
        f(); print(f.__name__, "ok")






