"""Figures fig:app-responsiveness (E27 + E8/E9 configurations), fig:app-disagreement (E29),
fig:app-horizon (E30).

usage: python fig_app_d13_d15_horizon.py <data_dir> <out_dir>
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
INK, INK2, GRID, GREY = "#0b0b0b", "#52514e", "#e6e5e1", "#8a8984"
# Method colours as in Figure 5
METH = {"M9": ("#2a78d6", "o", "Neural FOXP2"), "M4": ("#eb6834", "s", "Diff.-in-means"), "M5": ("#1baf7a", "^", "DiM + projection"),
        "M6": ("#eda100", "D", "LAPE neurons"), "M7": ("#e87ba4", "v", "SAE features")}


def style(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)


# ------------------------------------------------ Figure 12: responsiveness
sw = pd.read_csv(f"{data_dir}/e9_sweeps.csv")
main = pd.read_csv(f"{data_dir}/e8_main.csv")
main["dM"] = main[["L", "G9", "G2", "Q"]].mean(axis=1)
tau = pd.read_csv(f"{data_dir}/e27_responsiveness.csv").set_index("model")
fig, ax = plt.subplots(figsize=(3.03, 1.85))
for m in ["M4", "M5", "M6", "M7", "M9"]:
    d = sw[sw.method == m]
    col, mk, lab = METH[m]
    ax.plot(d.dM, d.full, ls="none", marker=mk, ms=3.2, color=col, label=lab, alpha=0.9,
            zorder=3 if m == "M9" else 2)
others = main[main.method.isin(["M0", "M1", "M2", "M3", "M8", "M10"])]
ax.plot(others.dM, others.full, ls="none", marker="X", ms=4, color="#6f6e69", mec="white", mew=0.4,
        label="single-setting methods")
ax.set_xlabel(r"mean $\Delta M$ of configuration", labelpad=1)
ax.set_ylabel("full-response target rate (%)", labelpad=1)
ax.set_ylim(-2, 70)
r1, r2 = tau.loc["All (vs full-response)"], tau.loc["All (vs human correctness)"]
ax.text(0.03, 0.97, rf"Kendall $\tau$ = {r1.tau:.2f} [{r1.lo:.2f}, {r1.hi:.2f}]" "\n"
        rf"vs human correctness: {r2.tau:.2f} [{r2.lo:.2f}, {r2.hi:.2f}]",
        transform=ax.transAxes, va="top", ha="left", fontsize=6.2, color=INK)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.82), fontsize=5.6, handlelength=1.0, ncol=1, labelspacing=0.25)
style(ax)
fig.savefig(f"{out_dir}/fig_app_responsiveness.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{out_dir}/fig_app_responsiveness.png", dpi=300, bbox_inches="tight", pad_inches=0.02)

# ------------------------------------------------ Figure 13: disagreement rate
dg = pd.read_csv(f"{data_dir}/e29_disagreement.csv")
fig, ax = plt.subplots(figsize=(3.03, 1.5))
x = np.arange(len(dg))
ax.bar(x, dg.rate, width=0.62, color="#2a78d6", zorder=2)
for xi, v in zip(x, dg.rate):
    ax.text(xi, v + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=6.2, color=INK2)
ax.set_xticks(x, dg.lang)
ax.set_ylabel("disagreement rate (%)", labelpad=1)
ax.set_ylim(0, 16)
style(ax)
fig.savefig(f"{out_dir}/fig_app_disagreement.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{out_dir}/fig_app_disagreement.png", dpi=300, bbox_inches="tight", pad_inches=0.02)

# ------------------------------------------------ Figure 16: persistence over the response
h = pd.read_csv(f"{data_dir}/e30_horizon.csv")
labels = list(h.horizon.astype(str))
x = np.arange(len(h))
# Leakage is pending recomputation under the revised definition; panel (b) only when available.
has_leak = h.leakage.notna().all()
if has_leak:
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 1.6), gridspec_kw={"wspace": 0.3})
else:
    fig, a = plt.subplots(figsize=(3.03, 1.6)); b = None
a.plot(x, h.rate, color="#2a78d6", marker="o", ms=3, lw=1.4)
for xi, v in zip(x, h.rate):
    below = xi == 0
    a.text(xi + (0.12 if below else 0), v + (-1.5 if below else 1.5), f"{v:.1f}", ha="left" if below else "center",
           va="top" if below else "bottom", fontsize=5.8, color=INK2)
a.set_ylabel("target-language rate (%)", labelpad=1)
a.set_ylim(40, 82)
if has_leak:
    b.plot(x, h.leakage, color="#2a78d6", marker="o", ms=3, lw=1.4)
    b.set_ylabel("leakage", labelpad=1)
    b.set_ylim(0, 0.03)
for ax, tag in (((a, "(a)"), (b, "(b)")) if has_leak else ((a, ""),)):
    ax.set_xticks(x, labels)
    ax.set_xlabel("horizon (tokens)", labelpad=1)
    style(ax)
    ax.text(-0.13, 1.04, tag, transform=ax.transAxes, fontweight="bold", fontsize=7)
fig.savefig(f"{out_dir}/fig_app_horizon.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{out_dir}/fig_app_horizon.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
