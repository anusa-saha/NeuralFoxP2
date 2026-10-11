"""Figures fig:app-sensitivity (E31) and fig:app-sample-efficiency (E32).

usage: python fig_app_sensitivity_sample.py <data_dir> <out_dir>
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

data_dir, out_dir = sys.argv[1], sys.argv[2]
for f in ("LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf", "LiberationSerif-Italic.ttf"):
    font_manager.fontManager.addfont(f"/usr/share/fonts/truetype/liberation/{f}")
plt.rcParams.update({
    "font.family": "Liberation Serif", "mathtext.fontset": "stix", "font.size": 7,
    "axes.linewidth": 0.5, "axes.edgecolor": "#52514e", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "xtick.major.width": 0.4, "ytick.major.width": 0.4,
    "xtick.major.size": 2, "ytick.major.size": 2, "pdf.fonttype": 42, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
})
INK2, GRID, BLUE = "#52514e", "#e6e5e1", "#2a78d6"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARK = ["o", "s", "^", "D"]

# ------------------------------------------------ Figure 18: sensitivity
s = pd.read_csv(f"{data_dir}/e31_sensitivity.csv")
params = [("K", r"features per layer $K$", True), ("r", r"subspace rank $r$", True),
          ("ratio", r"$\beta/\lambda$", False), ("gamma", r"strength multiple of $\gamma$", False),
          ("width", r"window width ($\pm$ layers)", False)]
# Leakage is pending recomputation under the revised definition; plot it only when available.
rows = [("default", "DEFAULT", (0.15, 0.68))]
if s["leak"].notna().sum() > 2 * len(params):
    rows.append(("leak", "leakage", (0, 0.10)))
fig, axes = plt.subplots(len(rows), 5, figsize=(6.3, 1.45 * len(rows)), squeeze=False,
                         gridspec_kw={"wspace": 0.42, "hspace": 0.55})
for j, (p, xl, logx) in enumerate(params):
    g = s[s.param == p].sort_values("value")
    x = np.arange(len(g)) if p == "width" else g.value.values
    opx = (x[g.op.values == 1])[0]
    for i, (col, yl, ylim) in enumerate(rows):
        ax = axes[i, j]
        ax.plot(x, g[col], color=BLUE, marker="o", ms=2.6, lw=1.2, zorder=3)
        ax.axvline(opx, color=INK2, lw=0.6, ls=":", zorder=1)
        if col == "default":  # band within 0.05 of the operating point
            opy = g[g.op == 1][col].iloc[0]
            ax.axhspan(opy - 0.05, opy + 0.05, color=GRID, lw=0, zorder=0)
        if logx:
            ax.set_xscale("log", base=2)
            keep = {8, 32, 64, 256, 1, 2, 4, 16}
            ax.set_xticks(g.value.values, [str(int(v)) if (p == "r" or int(v) in {8, 32, 64, 256}) else "" for v in g.value.values])
            ax.minorticks_off()
        elif p == "width":
            ax.set_xticks(x, ["−4", "−2", "0", "+2", "+4"])
        elif p == "gamma":
            ax.set_xticks([0, 1, 2, 3])
        elif p == "ratio":
            ax.set_xticks([0, 0.5, 1, 2], ["0", "0.5", "1", "2"])
        ax.set_ylim(*ylim)
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.set_ylabel(yl, labelpad=1)
        if i == len(rows) - 1:
            ax.set_xlabel(xl, labelpad=1, fontsize=6.5)
fig.text(0.5, 0.995, "dotted line: operating point; shaded band: DEFAULT within 0.05 of it",
         ha="center", va="top", fontsize=6.3, color=INK2)
fig.savefig(f"{out_dir}/fig_app_sensitivity.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{out_dir}/fig_app_sensitivity.png", dpi=300, bbox_inches="tight", pad_inches=0.02)

# ------------------------------------------------ Figure 19: sample efficiency
e = pd.read_csv(f"{data_dir}/e32_sample.csv")
fig, (a, b) = plt.subplots(1, 2, figsize=(3.03, 1.55), gridspec_kw={"wspace": 0.5})
for k, (c, lab) in enumerate(zip(["L", "G9", "G2", "Q"], ["L-8B", "G-9B", "G-2B", "Q-8B"])):
    a.plot(e.N, e[c], color=SLOT[k], marker=MARK[k], ms=2.6, lw=1.0, label=lab)
a.set_ylabel("gain retained (%)", labelpad=1)
a.set_ylim(60, 102)
a.legend(loc="lower right", fontsize=5.6, handlelength=1.2, labelspacing=0.2)
b.plot(e.N, e.jaccard, color=INK2, marker="o", ms=2.6, lw=1.0)
b.set_ylabel("support Jaccard", labelpad=1)
b.set_ylim(0.2, 1.04)
for xi, yi, ang in zip(e.N, e.jaccard, e.angle):
    if ang > 0:
        b.text(xi, yi - 0.06, f"{ang:.0f}°", ha="center", va="top", fontsize=5.6, color=INK2)
for ax, tag in ((a, "(a)"), (b, "(b)")):
    ax.set_xscale("log")
    ax.set_xticks([25, 50, 100, 200, 300], ["25", "50", "100", "200", "300"])
    ax.minorticks_off()
    ax.tick_params(labelsize=6)
    ax.set_xlabel("discovery pairs $N$", labelpad=1)
    ax.grid(axis="y", color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.text(-0.3, 1.04, tag, transform=ax.transAxes, fontweight="bold", fontsize=7)
fig.savefig(f"{out_dir}/fig_app_sample_efficiency.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{out_dir}/fig_app_sample_efficiency.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
