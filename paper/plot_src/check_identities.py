"""Mandatory cross-table identity checks (protocol, "Finishing the Paper", and revision decisions A1-A9, B1-B19).

usage: python check_identities.py <data_dir> [<latex_dir>]
Prints PASS / FAIL / PENDING / WARN per check. PENDING means a value has been blanked in the data
because it must be recomputed; the check cannot pass until the recomputed value is written back.
Tolerance is the rounding of the reported values. Run before regenerating the abstract and conclusion.
"""
import re
import sys
import numpy as np
import pandas as pd

D = sys.argv[1]
TEX = sys.argv[2] if len(sys.argv) > 2 else None
L = ["es", "fr", "zh", "ru", "hi", "ar"]
MODELS = ["Llama-3.1-8B-Instruct", "Gemma-2-9B-it", "Gemma-2-2B-it", "Qwen3-8B"]
res = []


def check(name, ok, detail, status=None):
    res.append((status or ("PASS" if ok else "FAIL"), name, detail))


def pending(name, detail):
    check(name, False, detail, "PENDING")


e1a = pd.read_csv(f"{D}/e1a_shares.csv"); e1b = pd.read_csv(f"{D}/e1b_deltaM.csv")
e8 = pd.read_csv(f"{D}/e8_main.csv").set_index("method")
e10 = pd.read_csv(f"{D}/e10_by_language.csv", index_col=0)
c1a = e1a[e1a.cond == "C1"].set_index("model"); c1b = e1b[e1b.cond == "C1"].set_index("model")
m9 = e8.loc["M9"]

# 1. Baseline: Table 2 M0 == Experiment 1 (No-Cue), per model and language (A1)
dm_e1 = c1b.loc[MODELS, L].mean(axis=1).round(2).values
dm_t2 = e8.loc["M0", ["L", "G9", "G2", "Q"]].values.astype(float)
check("1a Table 2 M0 dM per model == E1 No-Cue", np.allclose(dm_e1, dm_t2, atol=0.006), f"E1 {dm_e1} vs T2 {dm_t2}")
full_e1 = c1a.loc[MODELS, L].values.mean()
check("1b Table 2 M0 Full% == E1 No-Cue macro target rate", abs(full_e1 - e8.loc["M0", "full"]) < 0.06,
      f"E1 {full_e1:.2f} vs T2 {e8.loc['M0', 'full']} (E1 No-Cue rows must be regenerated to average 1.9)")
lang_e1 = c1a.loc[MODELS, L].mean(axis=0).values
check("1c Table 3 M0 per language == E1 No-Cue per language", np.allclose(lang_e1, e10.loc["M0", L].values, atol=0.06),
      f"E1 {np.round(lang_e1, 2)} vs Table 3 {e10.loc['M0', L].values}")

# 2. Commitment: E5 t=0 == E1 full-response rate (A2)
e5 = pd.read_csv(f"{D}/e5a_commitment.csv").set_index("t")
check("2a E5 macro P(stay) at t=0 == Table 2 M0 Full%", abs(e5.loc[0, "macro"] * 100 - e8.loc["M0", "full"]) < 0.06,
      f"E5 {e5.loc[0, 'macro'] * 100:.1f}% vs T2 {e8.loc['M0', 'full']}%")
if e5.loc[0, MODELS].isna().any():
    pending("2b E5 per-model t=0 == E1 per-model full-response rate", "per-model t=0 values blank; regenerate from E1")
else:
    fr = c1a.loc[MODELS, L].mean(axis=1).values
    check("2b E5 per-model t=0 == E1 per-model full-response rate", np.allclose(e5.loc[0, MODELS].values * 100, fr, atol=0.6),
          f"E5 {np.round(e5.loc[0, MODELS].values * 100, 1)} vs E1 {np.round(fr, 2)}")

# 3. Main comparison: Table 3 average == Table 2 Full%
bad = [m for m in e10.index if abs(e10.loc[m, "avg"] - e8.loc[m, "full"]) > 0.05]
check("3a Table 3 Avg == Table 2 Full%", not bad, f"mismatch: {bad}" if bad else "all methods")
bad = [m for m in e10.index if abs(e10.loc[m, L].mean() - e10.loc[m, "avg"]) > 0.05]
check("3b Table 3 Avg == mean of language columns", not bad, f"mismatch: {bad}" if bad else "all methods")

# 4. Null edit: gamma=0 reproduces M0 within the same split (A6)
sens = pd.read_csv(f"{D}/e31_sensitivity.csv")
g0 = sens[(sens.param == "gamma") & (sens.value == 0)].iloc[0]
check("4a Ddev: sweep at gamma=0 is the unedited model (DEFAULT 0.214, KL 0, dS 0)",
      g0.kl == 0 and g0.dS == 0 and abs(g0.default - 0.214) < 1e-9, f"gamma=0: DEFAULT {g0.default}, KL {g0.kl}, dS {g0.dS}")
if pd.isna(e8.loc["M0", "default"]):
    pending("4b Deval: M0 DEFAULT from the same outputs as M9", "Table 2 M0 DEFAULT blank (do not infer 0.587-0.573)")
else:
    check("4b Deval: M0 DEFAULT present", True, f"{e8.loc['M0', 'default']}")

# 5. Operating point: Table 2 M9 == Deval M9; Ddev tuning values stay Ddev (A9)
rb = pd.read_csv(f"{D}/e34_robustness.csv")
check("5a Table 2 M9 DEFAULT == E34 Deval greedy macro", abs(rb.greedy.mean() - m9.default) < 0.001,
      f"{rb.greedy.mean():.3f} vs {m9.default}")
opx = sens[sens.op == 1]
same = (opx.default == m9.default).all() and (opx.kl == m9.kl).all()
check("5b E31 (Ddev) operating point is a Ddev result, not a copy of Deval", not same,
      "E31 operating rows equal the Deval M9 values (0.587, KL 0.073); regenerate from the Ddev run" if same else "differ",
      None if not same else "WARN")
ab = pd.read_csv(f"{D}/e13_ablations.csv").set_index("variant")
check("5c Table 4 full row == Table 2 M9 (dM, DEFAULT, KL)",
      abs(ab.loc["full", "dM"] - m9[["L", "G9", "G2", "Q"]].mean()) < 0.006 and ab.loc["full", "default"] == m9.default
      and ab.loc["full", "kl"] == m9.kl, f"dM {ab.loc['full', 'dM']}, DEFAULT {ab.loc['full', 'default']}, KL {ab.loc['full', 'kl']}")
b0 = sens[(sens.param == "ratio") & (sens.value == 0)].iloc[0].default
check("5d E31 beta/lambda=0 (beta=0, promotion only) == Table 4 (c) promotion only (A5)",
      abs(b0 - ab.loc["c", "default"]) < 0.01, f"{b0} vs {ab.loc['c', 'default']}")

# 6. Ablation: single-layer row comes from the single-layer run (A9)
it = pd.read_csv(f"{D}/e21_it_sae.csv").set_index("metric")
if it[["base", "instruct"]].isna().any().any():
    pending("6 Table 13 single-layer rows from the single-layer run", "base/instruct rows blank; recompute")
else:
    check("6 Table 13 single-layer base row differs from full multi-layer M9",
          not (it.loc["full_pct", "base"] == m9.full and it.loc["dS", "base"] == m9.dS), "")

# 7. AUC: Table 1 and Figure 2a share metric, k, split and aggregation (A7)
auc = pd.read_csv(f"{D}/e2a_auc_by_k.csv")
a3 = auc[(auc.predictor == "dM") & (auc.k == 3)].auc.iloc[0]
t1 = pd.read_csv(f"{D}/e3a_predictors.csv").set_index("predictor").loc["dM at selected k", "mean_auc"]
check("7 Table 1 (macro) vs Figure 2a (pooled) at k=3", abs(a3 - t1) < 0.0015,
      f"Fig 2a pooled {a3} vs Table 1 macro {t1}: labelled as different aggregations; regenerate Fig 2a as macro-by-k",
      None if abs(a3 - t1) < 0.0015 else "WARN")
sel = pd.read_csv(f"{D}/e2b_dev_selection.csv").set_index("form")
best = sel[["k1", "k2", "k3", "k5", "k10"]].loc[["dM", "S"]]["k3"].idxmax()
check("7b D1 primary form == form with highest Ddev AUC at k=3 (B1)", best == "S", f"highest: {best}; paper uses S as primary")

# 8. Leakage: one formula and unit everywhere (A4)
lk = pd.read_csv(f"{D}/e11_leakage.csv", index_col=0)
off = np.mean([lk.loc[a, b] for a in L for b in L if a != b]) / 100
check("8a Table 2 M9 leakage == Figure 6 off-diagonal mean (fraction)", abs(off - m9.leak) < 0.0005,
      f"Fig 6 {off:.4f} vs Table 2 {m9.leak}")
miss = [m for m in e8.index if pd.isna(e8.loc[m, "leak"])]
if miss:
    pending("8b leakage under the revised definition for all methods, ablations and sweeps",
            f"blank for {miss}, ablation variants, E30 horizons and E31 sweep")

# 9. Utility: Table 2 dS == declared aggregate of benchmark-level values (B5, B17)
u = pd.read_csv(f"{D}/e16_utility.csv").set_index("task")
bench = ["XNLI", "Belebele", "XL-Sum", "FLORES", "MMLU", "GSM8K"]
bad = [m for m in ["M3", "M4", "M7", "M8", "M9"] if abs(round(u.loc[bench, m].mean(), 1) - e8.loc[m, "dS"]) > 0.051]
check("9a Table 2 dS == mean of six utility benchmarks", not bad, f"mismatch: {bad}" if bad else "M3 M4 M7 M8 M9")
um = pd.read_csv(f"{D}/e16_m9_by_model.csv").set_index("model")
check("9b English-task macro == mean of per-model English values", abs(um.loc[MODELS, "english"].mean() - um.loc["Macro", "english"]) < 0.006,
      f"{um.loc[MODELS, 'english'].mean():.3f} vs {um.loc['Macro', 'english']}")
eng = u.loc[["MMLU", "GSM8K"], "M9"].mean()
check("9c Table 5 M9 English benchmarks == per-model English macro", abs(eng - um.loc["Macro", "english"]) < 0.051,
      f"Table 5 MMLU/GSM8K mean {eng:.2f} vs per-model macro {um.loc['Macro', 'english']}")
if um.loc[MODELS, "overall"].isna().any():
    pending("9d per-model overall dS from six benchmark-level values", "Llama and Qwen blank; recompute")

# Necessity baseline == M1 full-response rate (A8)
nec = pd.read_csv(f"{D}/e15_necessity_models.csv").set_index("model")
check("A8 Table 5 'Nothing' macro == Table 2 M1 Full%", abs(nec.loc["Macro", "instr"] - e8.loc["M1", "full"]) < 0.06,
      f"{nec.loc['Macro', 'instr']} vs {e8.loc['M1', 'full']}")
if nec.loc[MODELS, "instr"].isna().any():
    pending("A8b per-model 'Nothing' values on the M1 prompts", "blank; recompute")

# Horizon figure full-response point == Table 2
fr30 = pd.read_csv(f"{D}/e30_fullresp.csv").set_index("method").full
h30 = pd.read_csv(f"{D}/e30_horizon.csv").set_index("horizon")
bad = [m for m in fr30.index if abs(fr30[m] - e8.loc[m, "full"]) > 0.05]
check("H Figure 16 full-response points == Table 2 Full%", not bad and float(h30.loc["full", "rate"]) == m9.full,
      f"mismatch: {bad}" if bad else "all 11 methods")

# Different prompt sets must not be compared as one estimate (A3)
diag = np.array([lk.loc[l, l] for l in L]); gain = (e10.loc["M9", L] - e10.loc["M0", L]).values
check("A3 Figure 6 diagonal (No-Cue DEFAULT) vs Table 3 gains", np.allclose(diag, gain, atol=0.6),
      f"Fig 6 mean {diag.mean():.1f} vs Table 3 mean {gain.mean():.1f}: reported as separate prompt sets; confirm the source of the 56.7 range",
      None if np.allclose(diag, gain, atol=0.6) else "WARN")

# 10. Abstract/conclusion headline numbers copied from canonical tables
if TEX:
    tex = open(f"{TEX}/acl_latex.tex").read()
    ab_txt = tex[tex.index("\\begin{abstract}"):tex.index("\\end{abstract}")]
    concl = tex[tex.index("\\section{Conclusion}"):tex.index("\\section*{Limitations}")]
    mt = pd.read_csv(f"{D}/e8_main.csv").set_index("method")
    canon = {"unedited": f"{mt.loc['M0', 'full']}\\%", "FOXP2": f"{mt.loc['M9', 'full']}\\%",
             "LoRA": f"{mt.loc['M8', 'full']}\\%", "matched": "44.7\\%"}
    for where, txt in (("abstract", ab_txt), ("conclusion", concl)):
        miss = [k for k, v in canon.items() if v not in txt]
        check(f"10 {where} headline numbers match Table 2 / matched-shift table", not miss, f"missing: {miss}" if miss else "ok")

for status, name, detail in res:
    print(f"{status:7s} {name}\n        {detail}")
print("\n" + ", ".join(f"{sum(s == k for s, _, _ in res)} {k.lower()}" for k in ["PASS", "FAIL", "PENDING", "WARN"]))
