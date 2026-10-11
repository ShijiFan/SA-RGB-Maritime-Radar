"""Figure style of the TAES resubmission (tools/taes_r5_style.py), reused unchanged except for the output folder.

TRS palette (slate, coral, teal, sand, sage), Times text, boxed axes with outward ticks, light horizontal grid,
frameless legends above the axes, (a)/(b) labels centred below each panel, printed size 3.5 / 7.16 in.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # bundled copy of taes_r5_style.py
sys.path.insert(1, r"E:\multipletrifeatures\tools")
import taes_r5_style as S  # noqa: E402  (sets rcParams on import)

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
S.FIG = FIG
SLATE, CORAL, TEAL, SAND, SAGE, BLACK, TINT, W1, W2 = S.SLATE, S.CORAL, S.TEAL, S.SAND, S.SAGE, S.BLACK, S.TINT, S.W1, S.W2
CORAL_E, TEAL_E, SAND_E = "#C4502F", "#1F7A6F", "#B8901F"

# Larger lettering than the 10-pt body text, Times New Roman everywhere (including math)
S.plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 10.5, "axes.labelsize": 10.5, "axes.titlesize": 10.5,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold", "mathtext.sf": "Times New Roman", "mathtext.cal": "Times New Roman:italic",
    "lines.markersize": 5.5, "lines.linewidth": 1.5, "axes.linewidth": 0.9,
})
FS, FS_S = 10.5, 9.5  # main and annotation font sizes
save, plt = S.save, S.plt


def panel_labels(fig, axes, letters="abcdefgh", gap=0.012, y=None):
    """(a), (b), ... centred below each axes at the main figure font size."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    for ax, l in zip(axes, letters):
        bb = ax.get_tightbbox(r).transformed(inv); pos = ax.get_position()
        fig.text((pos.x0 + pos.x1) / 2, (y if y is not None else bb.y0 - gap), f"({l})", ha="center", va="top",
                 fontsize=FS)