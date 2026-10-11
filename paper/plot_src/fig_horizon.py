"""Figures on the choice of the defaultness horizon T = {1, 2, 3}:
fig:horizon (main text), fig:app-horizon-selection, fig:app-horizon-commitment,
fig:app-horizon-synthesis (appendix). Uses only E2-A, E2-B, E5-A and E30.

usage: python fig_horizon.py <data_dir> <out_dir>
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
    "font.family": "Liberation Serif", "mathtext.fontset": "stix", "font.size": 6.5,
    "axes.linewidth": 0.5, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "xtick.major.width": 0.4,
    "ytick.major.width": 0.4, "xtick.major.size": 2, "ytick.major.size": 2,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    "legend.frameon": False, "legend.fontsize": 5.8, "lines.linewidth": 1.1,
})
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
GRID, INK2 = "#e6e5e1", "#52514e"
MODELS = ["Llama-3.1-8B-Instruct", "Gemma-2-9B-it", "Gemma-2-2B-it", "Qwen3-8B"]
MODEL_LABEL = ["Llama-3.1-8B", "Gemma-2-9B", "Gemma-2-2B", "Qwen3-8B"]
MARK = ["o", "s", "^", "D", "v", "P"]


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)


def mark_k3(ax, y, text=r"$k{=}3$"):
    ax.axvline(3, color=INK2, lw=0.6, ls=":", zorder=1)
    if text:
        ax.text(3.15, y, text, fontsize=5.6, color=INK2, va="bottom")


def save(fig, name):
    fig.savefig(f"{out_dir}/{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{out_dir}/{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


auc = pd.read_csv(f"{data_dir}/e2a_auc_by_k.csv")
dev = pd.read_csv(f"{data_dir}/e2b_dev_selection.csv")
com = pd.read_csv(f"{data_dir}/e5a_commitment.csv")
per = pd.read_csv(f"{data_dir}/e30_horizon.csv")
com["mean"] = com[MODELS].mean(axis=1)
per_num = per[per.horizon != "full"].copy()
per_num["horizon"] = per_num.horizon.astype(int)

# ------------------------------------------------------------ main-text figure
fig, (a, b) = plt.subplots(1, 2, figsize=(3.03, 1.42), gridspec_kw={"wspace": 0.42})
preds = [("dM", r"$\Delta M$", 0), ("LID", "prefix LID", 3), ("argmax", "first-token argmax", 4)]
for key, lab, c in preds:
    d = auc[auc.predictor == key].sort_values("k")
    a.fill_between(d.k, d.lo, d.hi, color=SLOT[c], alpha=0.13, linewidth=0)
    a.plot(d.k, d.auc, color=SLOT[c], marker=MARK[c], ms=2.3, lw=1.5 if key == "dM" else 1.0,
           label=lab, zorder=3 if key == "dM" else 2)
mark_k3(a, 0.655, None)
a.set_xticks([1, 2, 3, 5, 10])
a.set_xlim(0.6, 10.4)
a.set_ylim(0.65, 0.89)
a.set_xlabel(r"horizon $k$ (decoding steps)", labelpad=1)
a.set_ylabel("AUC, finished response", labelpad=1)
a.legend(loc="lower right", bbox_to_anchor=(1.0, 0.2), fontsize=5.2, handlelength=1.2, labelspacing=0.2,
         borderaxespad=0.1)
a.set_title("(a) prediction", fontsize=6.5, loc="left", pad=2)
style(a)

for i, m in enumerate(MODELS):
    b.plot(com.t, com[m], color="#b9b8b3", lw=0.6, zorder=1)
b.plot(com.t, com["mean"], color=SLOT[0], marker="o", ms=2.3, lw=1.5, zorder=3,
       label="stays in target\n(forced prefix)")
b.plot(per_num.horizon[per_num.horizon <= 10], per_num.rate[per_num.horizon <= 10] / 100,
       color=SLOT[1], marker="s", ms=2.3, lw=1.0, ls="--", zorder=2, label="steered prefix\nin target")
mark_k3(b, 0.04, None)
b.set_xticks([0, 1, 2, 3, 5, 10])
b.set_xlim(-0.4, 10.4)
b.set_ylim(0, 1.0)
b.set_xlabel(r"tokens $t$", labelpad=1)
b.set_ylabel("probability", labelpad=1)
b.legend(loc="lower right", fontsize=5.0, handlelength=1.4, labelspacing=0.3, borderaxespad=0.1)
b.set_title("(b) commitment", fontsize=6.5, loc="left", pad=2)
style(b)
save(fig, "fig_horizon")

# ------------------------------------------------------------ appendix: selection rule and marginal gain
fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 1.75), gridspec_kw={"wspace": 0.28})
ks = [1, 2, 3, 5, 10]
forms = [("dM", r"$\Delta M$ (uniform)"), ("dM_1/t", r"$\Delta M$ ($1/t$)"), ("dL", r"$\Delta L$"),
         ("S", r"$S$"), ("PrefixLID", "prefix LID")]
a.axhspan(-0.01, 0.0, color=SLOT[0], alpha=0.08, lw=0)
a.axhline(-0.01, color=SLOT[0], lw=0.5, ls="--")
a.text(7.5, -0.0115, "tolerance 0.01", fontsize=5.4, color=SLOT[0], ha="center", va="top")
for i, (key, lab) in enumerate(forms):
    r = dev[dev.form == key].iloc[0]
    vals = np.array([r[f"k{k}"] for k in ks], dtype=float)
    gap = vals - vals.max()
    a.plot(ks, gap, color=SLOT[i], marker=MARK[i], ms=2.6, lw=1.4 if key == "dM" else 1.0, label=lab,
           zorder=3 if key == "dM" else 2)
    sel = int(r.selected_k)
    a.plot([sel], [gap[ks.index(sel)]], marker=MARK[i], ms=6, mfc="none", mec=SLOT[i], mew=1.0, zorder=4)
mark_k3(a, -0.112, None)
a.set_xticks(ks)
a.set_xlim(0.6, 10.5)
a.set_ylim(-0.118, 0.006)
a.set_xlabel(r"horizon $k$", labelpad=1)
a.set_ylabel(r"AUC $-$ row maximum ($\mathcal{D}_{\mathrm{dev}}$)", labelpad=1)
a.legend(loc="lower right", ncol=2, fontsize=5.4, columnspacing=0.8, handlelength=1.3)
a.set_title("(a) selection rule on $\\mathcal{D}_{\\mathrm{dev}}$ (circles: selected $k$)",
            fontsize=6.5, loc="left", pad=2)
style(a)

ev = [("dM", r"$\Delta M$", 0), ("S", r"$S$", 3), ("dL", r"$\Delta L$", 2), ("LID", "prefix LID", 4)]
w = 0.2
for j, (key, lab, c) in enumerate(ev):
    d = auc[auc.predictor == key].sort_values("k")
    gain = np.diff(d.auc.values)
    x = np.arange(2, 11) + (j - 1.5) * w
    b.bar(x, gain, width=w, color=SLOT[c], label=lab, zorder=2)
b.axhline(0, color=INK2, lw=0.5)
b.axvline(3.5, color=INK2, lw=0.6, ls=":")
b.text(3.6, 0.030, r"$\Delta M$ beyond $k{=}3$: every step within $\pm$0.006", fontsize=5.6, color=INK2, va="bottom")
b.set_xticks(range(2, 11), [f"{k-1}→{k}" for k in range(2, 11)])
b.set_xlim(1.4, 10.6)
b.set_ylim(-0.014, 0.052)
b.set_xlabel(r"added decoding step ($\mathcal{D}_{\mathrm{eval}}$)", labelpad=1)
b.set_ylabel("change in AUC", labelpad=1)
b.legend(loc="upper right", ncol=4, fontsize=5.4, columnspacing=0.8, handlelength=1.0)
b.set_title("(b) marginal value of each additional step", fontsize=6.5, loc="left", pad=2)
style(b)
save(fig, "fig_app_horizon_selection")

# ------------------------------------------------------------ appendix: commitment per model
fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 1.75), gridspec_kw={"wspace": 0.28})
ts = com.t.values
mid = (ts[:-1] + ts[1:]) / 2
for i, (m, lab) in enumerate(zip(MODELS, MODEL_LABEL)):
    v = com[m].values
    per_tok = np.diff(v) / np.diff(ts)
    a.plot(mid, per_tok, color=SLOT[i], marker=MARK[i], ms=2.6, lw=1.0, label=lab)
a.plot(mid, np.diff(com["mean"].values) / np.diff(ts), color="#0b0b0b", lw=1.5, ls="--", label="mean", zorder=3)
a.axvline(3, color=INK2, lw=0.6, ls=":")
a.text(3.15, 0.27, r"$t{=}3$", fontsize=5.6, color=INK2)
a.set_xticks(mid, ["0→1", "1→2", "2→3", "3→5", "5→10"])
a.set_ylim(0, 0.31)
a.set_xlabel("interval of forced tokens", labelpad=1)
a.set_ylabel(r"gain in $P$(stay) per forced token", labelpad=1)
a.legend(loc="upper right", fontsize=5.4, ncol=1, handlelength=1.3, labelspacing=0.25)
a.set_title("(a) diminishing returns of each forced token", fontsize=6.5, loc="left", pad=2)
style(a)

for i, (m, lab) in enumerate(zip(MODELS, MODEL_LABEL)):
    v = com[m].values
    share = (v - v[0]) / (v[-1] - v[0])
    s3 = share[list(ts).index(3)] * 100
    b.plot(ts, share * 100, color=SLOT[i], marker=MARK[i], ms=2.6, lw=1.0,
           label=f"{lab}: {s3:.0f}% by $t{{=}}3$")
b.axvline(3, color=INK2, lw=0.6, ls=":")
b.set_xticks([0, 1, 2, 3, 5, 10])
b.set_xlim(-0.4, 10.4)
b.set_ylim(0, 105)
b.set_xlabel(r"forced target-language tokens $t$", labelpad=1)
b.set_ylabel(r"share of the $t{=}0\rightarrow10$ gain (%)", labelpad=1)
b.legend(loc="lower right", fontsize=5.4, handlelength=1.3, labelspacing=0.25)
b.set_title(r"(b) most of the commitment is in place after three tokens", fontsize=6.5, loc="left", pad=2)
style(b)
save(fig, "fig_app_horizon_commitment")

# ------------------------------------------------------------ appendix: the signals side by side
fig, ax = plt.subplots(figsize=(3.03, 1.75))
d = auc[auc.predictor == "dM"].sort_values("k")
auc_share = (d.auc.values - d.auc.values[0]) / (d.auc.values.max() - d.auc.values[0])
ax.plot(d.k, auc_share * 100, color=SLOT[0], marker="o", ms=2.4, lw=1.4,
        label=r"$\Delta M$ AUC gain reached")
c = com[com.t >= 1]
c_share = (c["mean"].values - com["mean"].values[0]) / (com["mean"].values[-1] - com["mean"].values[0])
ax.plot(c.t, c_share * 100, color=SLOT[2], marker="^", ms=2.4, lw=1.1, label="commitment gain reached")
p = per_num[per_num.horizon <= 10]
ax.plot(p.horizon, p.rate / p.rate.max() * 100, color=SLOT[1], marker="s", ms=2.4, lw=1.1, ls="--",
        label="steered prefix rate, % of peak")
kk = np.arange(1, 11)
ax.plot(kk, kk / 10 * 100, color="#8a8984", lw=0.8, ls=":", label=r"scoring cost ($k$ forward steps)")
ax.axvspan(2.6, 3.4, color="#0b0b0b", alpha=0.05, lw=0)
ax.set_xticks(range(1, 11))
ax.set_xlim(0.6, 10.4)
ax.set_ylim(0, 108)
ax.set_xlabel(r"horizon $k$ (decoding steps or tokens)", labelpad=1)
ax.set_ylabel("% of maximum", labelpad=1)
ax.legend(loc="lower right", fontsize=5.2, handlelength=1.6, labelspacing=0.25)
style(ax)
save(fig, "fig_app_horizon_synthesis")
