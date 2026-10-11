"""Cross-table identity and range checks from 'Finishing the Paper', p. 22-23.

usage: python check_identities.py <data_dir>
Prints PASS / FAIL / WARN per check. Tolerance is the rounding of the reported values.
"""
import sys
import numpy as np
import pandas as pd

D = sys.argv[1]
L = ["es", "fr", "zh", "ru", "hi", "ar"]
MODELS = ["Llama-3.1-8B-Instruct", "Gemma-2-9B-it", "Gemma-2-2B-it", "Qwen3-8B"]
res = []


def check(name, ok, detail, warn=False):
    res.append(("PASS" if ok else ("WARN" if warn else "FAIL"), name, detail))


e1a = pd.read_csv(f"{D}/e1a_shares.csv"); e1b = pd.read_csv(f"{D}/e1b_deltaM.csv")
e8 = pd.read_csv(f"{D}/e8_main.csv").set_index("method")
c1a = e1a[e1a.cond == "C1"].set_index("model"); c1b = e1b[e1b.cond == "C1"].set_index("model")

# 1. Table 2 row M0 == Experiment 1, C1 macro values
dm_e1 = c1b.loc[MODELS, L].mean(axis=1).round(2).values
dm_t2 = e8.loc["M0", ["L", "G9", "G2", "Q"]].values.astype(float)
check("Table 2 M0 dM per model == E1 C1", np.allclose(dm_e1, dm_t2, atol=0.006), f"E1 {dm_e1} vs T2 {dm_t2}")
full_e1 = c1a.loc[MODELS, L].values.mean()
check("Table 2 M0 Full% == E1 C1 macro target rate", abs(full_e1 - e8.loc["M0", "full"]) < 0.06,
      f"E1 {full_e1:.2f} vs T2 {e8.loc['M0', 'full']}")

# 2. Experiment 5 at t=0 == Experiment 1 full-response rate per model
e5 = pd.read_csv(f"{D}/e5a_commitment.csv").set_index("t")
t0 = e5.loc[0, MODELS].values * 100
fr = c1a.loc[MODELS, L].mean(axis=1).values
check("E5 P(stay) at t=0 == E1 full-response rate", np.allclose(t0, fr, atol=0.6),
      f"E5 t=0 {np.round(t0, 1)}% vs E1 {np.round(fr, 2)}%")

# 3. Table 3 Avg == Table 2 Full%
e10 = pd.read_csv(f"{D}/e10_by_language.csv", index_col=0)
bad = [m for m in e10.index if abs(e10.loc[m, "avg"] - e8.loc[m, "full"]) > 0.05]
check("Table 3 Avg == Table 2 Full%", not bad, f"mismatch: {bad}" if bad else "all methods")
bad = [m for m in e10.index if abs(e10.loc[m, L].mean() - e10.loc[m, "avg"]) > 0.05]
check("Table 3 Avg == mean of language columns", not bad, f"mismatch: {bad}" if bad else "all methods")

# 4. Table 4 full row == Table 2 M9
ab = pd.read_csv(f"{D}/e13_ablations.csv").set_index("variant").loc["full"]
m9 = e8.loc["M9"]
check("Table 4 full dM == Table 2 M9 mean dM", abs(ab.dM - m9[["L", "G9", "G2", "Q"]].mean()) < 0.006,
      f"{ab.dM} vs {m9[['L', 'G9', 'G2', 'Q']].mean():.4f}")
check("Table 4 full leakage, KL == Table 2 M9", ab.leak == m9.leak and ab.kl == m9.kl,
      f"leak {ab.leak}/{m9.leak}, KL {ab.kl}/{m9.kl}")

# 5. Table 5 'Nothing' row == M1 full-response rate
nec = pd.read_csv(f"{D}/e15_necessity_models.csv").set_index("model")
check("Table 5 'Nothing' macro == Table 2 M1 Full%", abs(nec.loc["Macro", "instr"] - e8.loc["M1", "full"]) < 0.06,
      f"{nec.loc['Macro', 'instr']} vs {e8.loc['M1', 'full']}")

# 6. Table 2 dS == mean of six benchmark rows
u = pd.read_csv(f"{D}/e16_utility.csv").set_index("task")
bench = ["XNLI", "Belebele", "XL-Sum", "FLORES", "MMLU", "GSM8K"]
bad = [m for m in ["M3", "M4", "M7", "M8", "M9"] if abs(round(u.loc[bench, m].mean(), 1) - e8.loc[m, "dS"]) > 0.051]
check("Table 2 dS == mean of six utility benchmarks", not bad, f"mismatch: {bad}" if bad else "M3 M4 M7 M8 M9")

# 7. Table 1 dM AUC == Figure 2a at selected k
auc = pd.read_csv(f"{D}/e2a_auc_by_k.csv")
a3 = auc[(auc.predictor == "dM") & (auc.k == 3)].auc.iloc[0]
t1 = pd.read_csv(f"{D}/e3a_predictors.csv").set_index("predictor").loc["dM at selected k", "mean_auc"]
check("Table 1 dM AUC == Figure 2a at k=3", abs(a3 - t1) < 0.0015, f"Fig 2a {a3} vs Table 1 {t1}")

# 8. horizon == Table 17 selected k
sel = pd.read_csv(f"{D}/e2b_dev_selection.csv").set_index("form").loc["dM", "selected_k"]
check("Section 3.2 horizon == Table 17 selected k", sel == 3, f"selected k = {sel}; paper uses 3")

# 9. Figure 5 operating points == Table 2
sw = pd.read_csv(f"{D}/e9_sweeps.csv"); op = sw[sw.step == 6].set_index("method")
bad = [m for m in op.index if not (op.loc[m, "full"] == e8.loc[m, "full"] and op.loc[m, "kl"] == e8.loc[m, "kl"])]
check("Figure 5 operating points == Table 2", not bad, f"mismatch: {bad}" if bad else "M4-M7, M9")

# 10. Figure 16 full-response point == Table 2 Full% for each method
fr30 = pd.read_csv(f"{D}/e30_fullresp.csv").set_index("method").full
bad = [m for m in fr30.index if abs(fr30[m] - e8.loc[m, "full"]) > 0.05]
h30 = pd.read_csv(f"{D}/e30_horizon.csv").set_index("horizon")
check("Figure 16 full-response points == Table 2 Full%", not bad and h30.loc["full", "rate"] == e8.loc["M9", "full"]
      and h30.loc["full", "leakage"] == e8.loc["M9", "leak"], f"mismatch: {bad}" if bad else "all 11 methods; M9 leakage too")

# 11. E31-E35 identities
sens = pd.read_csv(f"{D}/e31_sensitivity.csv")
opx = sens[sens.op == 1]
check("E31 operating point == Table 4 full row (DEFAULT, leak, KL)",
      (opx.default == ab.default).all() and (opx.leak == ab.leak).all() and (opx.kl == ab.kl).all(),
      f"{len(opx)} sweeps at the operating point")
g0 = sens[(sens.param == "gamma") & (sens.value == 0)].iloc[0]
m0_default = ab.default - pd.read_csv(f"{D}/e25_constructions.csv").diagnostic.mean()
check("E31 DEFAULT at gamma=0 == unedited DEFAULT (E13 full - E25 gain)", abs(g0.default - m0_default) < 0.01,
      f"gamma=0 gives {g0.default}; implied unedited {m0_default:.3f}")
b0 = sens[(sens.param == "ratio") & (sens.value == 0)].iloc[0].default
abc = pd.read_csv(f"{D}/e13_ablations.csv").set_index("variant").loc["c", "default"]
check("E31 beta/lambda=0 == Table 4 ablation (c) promotion only", abs(b0 - abc) < 0.01, f"{b0} vs {abc}")
rb = pd.read_csv(f"{D}/e34_robustness.csv")
check("E34 greedy DEFAULT macro == Table 4 full DEFAULT", abs(rb.greedy.mean() - ab.default) < 0.001,
      f"{rb.greedy.mean():.3f} vs {ab.default}")
tp = pd.read_csv(f"{D}/e35_template.csv").set_index("variant").loc["Default template"]
check("E35 default-template row == Table 2 (M0, M3, M9)",
      tp.M0 == e8.loc["M0", "full"] and tp.M3 == e8.loc["M3", "full"] and tp.M9 == e8.loc["M9", "full"],
      f"{tp.M0}/{tp.M3}/{tp.M9}")

# 12. Per-model DEFAULT levels (E34) minus per-model gains (E25) must be a valid unedited DEFAULT (>= 0, equal across sources)
c25 = pd.read_csv(f"{D}/e25_constructions.csv").set_index("model").diagnostic
lvl = pd.read_csv(f"{D}/e34_robustness.csv").set_index("model").greedy
imp = (lvl - c25).round(3)
check("E34 DEFAULT - E25 gain gives a valid unedited DEFAULT per model", (imp >= 0).all() and imp.max() - imp.min() < 0.02,
      f"implied unedited DEFAULT {imp.to_dict()}")
# 13. Table 15 r_eff (E6) vs Table 12 pair-level effective rank (E14)
e6 = pd.read_csv(f"{D}/e6_artifacts.csv")
check("Table 15 r_eff median == Table 12 pairs median", abs(e6.reff.median() - 8.3) < 0.1,
      f"E6 median {e6.reff.median()} vs E14 8.3 (state which layer each refers to)", warn=True)
# 14. Cue-pool size vs determined n
q = pd.read_csv(f"{D}/e2c_cue_pool.csv")
check("Cue pool determined n <= items x 6 levels", q.n.max() <= 600, f"max n {q.n.max()}; 200 items x 6 levels would give 1,200", warn=True)

# Extra consistency checks raised during assembly
lk = pd.read_csv(f"{D}/e11_leakage.csv", index_col=0)
diag = np.array([lk.loc[l, l] for l in L]); gain = (e10.loc["M9", L] - e10.loc["M0", L]).values
check("Figure 6 diagonal == Table 3 M9 gain over M0", np.allclose(diag, gain, atol=0.6),
      f"Fig 6 {diag} vs Table 3 {np.round(gain, 1)}")
off = np.mean([lk.loc[a, b] for a in L for b in L if a != b]) / 100
check("Table 2 M9 leakage == Figure 6 off-diagonal mean (fraction)", abs(off - m9.leak) < 0.002,
      f"Fig 6 {off:.4f} vs Table 2 {m9.leak}")
it = pd.read_csv(f"{D}/e21_it_sae.csv").set_index("metric")
same = it.loc["full_pct", "base"] == m9.full and it.loc["dS", "base"] == -0.28
check("Table 13 single-layer base row differs from full multi-layer M9", not same,
      "base-trained single-layer row equals the full M9 result (58.7%, -0.28); confirm it is a single-layer edit", warn=True)

# Interval sanity
iv = [("E16 target", -0.08, -0.25, 0.10), ("E16 English", -0.70, -0.94, -0.46), ("E16 overall", -0.28, -0.42, -0.14),
      ("E18 XSafety", -0.8, -2.7, 1.1), ("E21 DEFAULT", -0.013, -0.031, 0.006), ("E10 weakest", 31.8, 23.1, 40.5)]
r = pd.read_csv(f"{D}/e22_reliability.csv")
iv += [(f"E22 ICC {m}", a, b, c) for m, a, b, c in zip(r.model, r.icc, r.icc_lo, r.icc_hi)]
bad = [n for n, p, lo, hi in iv if not lo <= p <= hi]
check("Every interval contains its point estimate", not bad, f"bad: {bad}" if bad else f"{len(iv)} intervals")

for status, name, detail in res:
    print(f"{status:4s}  {name}\n      {detail}")
print(f"\n{sum(s == 'PASS' for s, _, _ in res)} pass, {sum(s == 'FAIL' for s, _, _ in res)} fail, "
      f"{sum(s == 'WARN' for s, _, _ in res)} warn")
