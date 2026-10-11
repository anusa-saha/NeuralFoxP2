"""Figures fig:metric-validity (E2-A, E4-B) and fig:commitment (E5-A).

usage: python fig_metric_and_commitment.py <data_dir> <out_dir>
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

data_dir, out_dir = sys.argv[1], sys.argv[2]
for f in ("LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf", "LiberationSerif-Italic.ttf"):
    font_manager.fontManager.addfont(f"/usr/share/fonts/truetype/liberation/{f}")
plt.rcParams.update({
    "font.family": "Liberation Serif", "mathtext.fontset": "stix", "font.size": 6.5,
    "axes.linewidth": 0.5, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "xtick.major.width": 0.4,
    "ytick.major.width": 0.4, "xtick.major.size": 2, "ytick.major.size": 2,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    "legend.frameon": False, "legend.fontsize": 5.8, "lines.linewidth": 1.1,
})
# Categorical slots in fixed order (reference palette, light mode).
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
GRID = "#e6e5e1"
MODELS = ["Llama-3.1-8B-Instruct", "Gemma-2-9B-it", "Gemma-2-2B-it", "Qwen3-8B"]
MODEL_LABEL = ["Llama-3.1-8B", "Gemma-2-9B", "Gemma-2-2B", "Qwen3-8B"]
MARK = ["o", "s", "^", "D", "v", "P"]  # secondary encoding so identity is never colour-only


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- Figure 2
auc = pd.read_csv(f"{data_dir}/e2a_auc_by_k.csv")
cue = pd.read_csv(f"{data_dir}/e4b_cue_levels.csv", index_col=0)
fig = plt.figure(figsize=(3.03, 2.40))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.75, wspace=0.42)
ax_a = fig.add_subplot(gs[0, :])
ax_b = fig.add_subplot(gs[1, 0])
ax_c = fig.add_subplot(gs[1, 1])

preds = [("dM", r"$\Delta M$"), ("dL", r"$\Delta L$ (log-odds)"), ("S", r"$S$ (share)"),
         ("LID", "Prefix LID"), ("argmax", "First-token argmax")]
for i, (key, lab) in enumerate(preds):
    d = auc[auc.predictor == key].sort_values("k")
    lw = 1.5 if key == "dM" else 1.0
    ax_a.fill_between(d.k, d.lo, d.hi, color=SLOT[i], alpha=0.12, linewidth=0)
    ax_a.plot(d.k, d.auc, color=SLOT[i], lw=lw, marker=MARK[i], ms=2.6, label=lab,
              zorder=3 if key == "dM" else 2)
ax_a.axvline(3, color="#52514e", lw=0.6, ls=":", zorder=1)
ax_a.text(3.1, 0.648, r"selected $k{=}3$", fontsize=5.6, color="#52514e", va="bottom")
ax_a.set_xticks(range(1, 11))
ax_a.set_xlim(0.7, 10.3)
ax_a.set_ylim(0.64, 0.90)
ax_a.set_xlabel(r"horizon $k$ (decoding steps)", labelpad=1)
ax_a.set_ylabel("pooled AUC", labelpad=1)
ax_a.legend(ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.0), columnspacing=0.9,
            handlelength=1.6, handletextpad=0.4)
style(ax_a)
ax_a.text(-0.11, 1.09, "(a)", transform=ax_a.transAxes, fontsize=6.5, fontweight="bold")

# (b) reliability diagram: awaiting binned data
ax_b.set_xlim(-1, 1)
ax_b.set_ylim(0, 1)
ax_b.plot([-1, 1], [0, 1], color=GRID, lw=0.8)
ax_b.text(0, 0.5, "reliability-bin\ndata pending", ha="center", va="center",
          fontsize=6, color="#8a8984", style="italic")
ax_b.set_xlabel(r"$\Delta M$ bin", labelpad=1)
ax_b.set_ylabel("observed target rate", labelpad=1)
style(ax_b)
ax_b.set_title("(b)", loc="left", fontsize=6.5, fontweight="bold", x=-0.30, y=1.0)

# (c) cue strength
levels = ["L0", "L1", "L2", "L3", "L4", "L5"]
langs = ["es", "fr", "zh", "ru", "hi", "ar"]
x = range(6)
for i, l in enumerate(langs):
    ax_c.plot(x, cue.loc[l, levels].values, color=SLOT[i], lw=0.8, marker=MARK[i], ms=2.2,
              label=l, alpha=0.95)
ax_c.plot(x, cue.loc["mean", levels].values, color="#0b0b0b", lw=1.3, ls="--", label="mean",
          zorder=4)
ax_c.axhline(0, color="#8a8984", lw=0.5)
ax_c.set_xticks(list(x), levels)
ax_c.set_xlabel("cue level", labelpad=1)
ax_c.set_ylabel(r"mean $\Delta M$", labelpad=1)
ax_c.set_ylim(-0.66, 0.30)
ax_c.legend(ncol=4, loc="upper left", fontsize=5.0, handlelength=1.1, columnspacing=0.5,
            handletextpad=0.3, labelspacing=0.2)
style(ax_c)
ax_c.set_title("(c)", loc="left", fontsize=6.5, fontweight="bold", x=-0.30, y=1.0)
fig.savefig(f"{out_dir}/fig_metric_validity.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{out_dir}/fig_metric_validity.png", dpi=300, bbox_inches="tight", pad_inches=0.02)

# ---------------------------------------------------------------- Figure 4
com = pd.read_csv(f"{data_dir}/e5a_commitment.csv")
fig, ax = plt.subplots(figsize=(3.03, 1.55))
for i, (m, lab) in enumerate(zip(MODELS, MODEL_LABEL)):
    ax.plot(com.t, com[m], color=SLOT[i], marker=MARK[i], ms=3, label=lab)
# t=0 equals the Experiment 1 full-response rate (protocol identity); per-model t=0 values pending
ax.plot(com.t, com["macro"], color="#0b0b0b", ls="--", lw=1.0, marker="x", ms=3, label="mean", zorder=4)
ax.axvline(3, color="#52514e", lw=0.6, ls=":")
ax.text(3.15, 0.93, r"selected $k{=}3$", fontsize=5.6, color="#52514e")
ax.set_xticks([0, 1, 2, 3, 5, 10])
ax.set_xlim(-0.3, 10.3)
ax.set_ylim(0, 1)
ax.set_xlabel(r"forced target-language tokens $t$", labelpad=1)
ax.set_ylabel(r"$P$(stay in target)", labelpad=1)
ax.legend(loc="lower right", ncol=2, columnspacing=0.9, handlelength=1.6)
style(ax)
fig.savefig(f"{out_dir}/fig_commitment.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{out_dir}/fig_commitment.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
