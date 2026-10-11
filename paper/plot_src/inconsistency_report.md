# Neural FOXP2 — narrative and value inconsistencies

Full read of `acl_latex.tex` and `appendix.tex` plus data cross-checks (`check_identities.py`: 14 pass, 9 fail, 2 warn).
Table/figure numbers refer to the current PDF.

## A. Data contradictions (fix the data; the text follows)

| # | Where | Conflict | What has to be decided |
|---|---|---|---|
| A1 | Abstract, Conclusion, Table 2 M0, Table 26 M0 vs §3.4, Fig 3, Table 21 | Unedited full-response rate is 1.9% in one set and 1.6% (1.55) in the other; per-language M0 also differs (zh 2.2 vs 0.9). | Which run is the unedited baseline; regenerate the other. |
| A2 | §3.4, App. G.1, Fig. on commitment | P(stay) at t=0 is .246 (.154–.303 by model). With nothing forced this must equal the unedited full-response rate (≈1.6–1.9%). The §3.4 sentence "from .246 to .773" depends on it. | Re-check what t=0 measures in E5. |
| A3 | §6.2 (both paragraphs), Fig 6, App. H.1, Table 26 | FOXP2 gain in its own language: +48.3 to +64.4 (Table 26, first §6.2 paragraph) vs +33.1 to +42.8 (Fig 6 diagonal, second §6.2 paragraph, Spanish/French script test). Mean 56.7 vs 38.2. | Same metric and prompt set, or different? Label accordingly. |
| A4 | Table 2, Table 3, App. E definition vs Fig 6 | Leakage 0.026 in Tables 2–3; the Fig 6 off-diagonals give 0.0065 under the App. E definition. Fig 6 is in points, Table 2 in fractions; App. E defines leakage via DEFAULT; Fig 6's en column uses full-response. | One leakage definition and unit everywhere. |
| A5 | Table 3 (c) vs Table 36 (β/λ = 0) | Promotion only = 52.2 in the ablation, 34.8 in the sweep. 34.8 matches suppression only (34.1). If the sweep is right, (c)/(d) are swapped and §6.3's "promotion carries most of the effect" reverses. | Re-check variant labels in E13 and the ratio direction in E31. |
| A6 | Table 36 γ=0 vs E13/E25/E34 | Unedited DEFAULT is 0.214 (sweep at γ=0), 0.014 (0.587 − 0.573), and per model 0.041 / 0.014 / −0.002 / 0.005 (E34 level − E25 gain). The null edit is exact, so these must agree, and −0.002 is impossible. | Recompute unedited DEFAULT once and derive gains from it. |
| A7 | Table 1, §3.3 vs Fig 2a | ΔM AUC .819 vs .837 at k=3. App. F.2 says AUCs use the cue-strength pool; Table 18 and Fig 2 captions say Deval. | State the pool once; make the two AUCs come from the same aggregation. |
| A8 | Table 5 "Nothing" vs Table 2 M1 | Instruction-only compliance 14.8% vs M1 full-response 15.7%. | Same prompts? If not, say so in the caption. |
| A9 | Table 3, Table 13, Table 36, Table 38 | DEFAULT = 0.587 in all four, and equals the full-response rate 58.7. Table 36 is on Ddev, the others on Deval; Table 13's base row is a single-layer edit yet equals the full multi-layer result (58.7, −0.28). | Confirm these are separate runs. |

## B. Text and method conflicts

| # | Where | Conflict |
|---|---|---|
| B1 | §3.2 + App. F.1 | D1 picks the form with the highest dev AUC; S ≥ ΔM at every k (and S .827 > ΔM .814 on human labels). Either S is primary or the tie-break must be stated (orange note). |
| B2 | §3.2 "Outcome measures" | "Outcomes that do not depend on ΔM" — but DEFAULT is defined with ΔM > 0. |
| B3 | Contribution 3, Table 39 (tests), per-model Tables 22–25, C2 table | These report or test DEFAULT for Table 2 comparisons, but Table 2 reports mean ΔM and full-response rate, not DEFAULT. |
| B4 | App. D (M4) vs §5 and Table 17 | M4 strength is "norm-matched to the FOXP2 edit" in App. D but "tuned per pair on Ddev, median γ = 0.75" in Table 17 and §5. |
| B5 | App. E (Utility) vs Table 2, abstract | App. E says ΔS is "reported per task without aggregation", yet Table 2 and the abstract give an aggregate (−0.28 over six benchmarks). §5 says "eight tasks"; ΔS uses six. App. E and the appendix utility table use FLORES chrF++ and COMET; Table 5 has one "composite" row. |
| B6 | Abstract, Contributions, §5 | "Ten baselines" counts M0 (no edit) and M10 (FOXP2 + system prompt); actual baselines are M1–M8 (eight). |
| B7 | Table 1 caption, App. F.2 (D4) | Promise Brier scores, prefix LID at 10 tokens and LCB as a predictor; none is reported. |
| B8 | App. A vs Table 19 vs App. F.5 | Cue pool is "[200] items × 6 levels" (1,200 per pair), but Table 19 has at most 592 determined responses per pair (fits 100 items); D7 uses 50 items. |
| B9 | Limitations | "We report full-response language, LCB and human judgments for every claim" — ablations, sensitivity, transfer and robustness report DEFAULT only. |
| B10 | §5 Statistics | "Every mean carries a 95% bootstrap interval" — most tables have no CIs. |
| B11 | §4 Stage II vs App. C | "Rounded effective rank" vs ⌈r_eff⌉ (ceiling). |
| B12 | Table 15 vs Table 12 | r_eff median 8.55 (IQR 7.75–9.27) vs 8.3 (7.1–9.5); state which layer each uses. |
| B13 | Intro vs §3.4 / App. G | Intro cites that the response language "rarely changes" after an early point; your data show ~21–23% switch away after three target tokens (P(stay) .773; 74.6% → 58.7%). |
| B14 | App. F.7 (D10) | Script-only sets are said to be weaker "for Spanish and French"; no per-language breakdown supports this. |
| B15 | Table 6 caption vs App. K | "2–3 annotators per language" vs "each item rated by [2–3] native speakers". |
| B16 | App. F.7 (D14) vs Table 39 (templates) | "ΔM is insensitive to template" while the minimal template lowers FOXP2's full-response rate by 7.1 points and moves Gemma ΔM upward. Not a contradiction, but say both. |
| B17 | E16 by model | Model rows average −0.57 on English tasks, not −0.70; Llama and Qwen overall ΔS do not follow from their own components. |
| B18 | E9.5 vs E9 sweeps | Matched-KL values are not reproduced by interpolating the macro curves at KL 0.073 (38.2 / 42.2 / 37.9 / 38.0 vs 40.8 / 44.7 / 39.3 / 42.6). State the matching method. |
| B19 | E31 summary (fixed in text) | Your stability ranges for β/λ and γ did not match the sweep values; text uses 1–2 and 1–3×. |

## C. Fixed in this pass (my wording errors)

- Romanized Russian mean: table showed +3.5, text 3.6 → table now +3.6 (E12.3 macro).
- §3.3 "against human labels the ordering is the same" → argmax and perplexity swap; now "the top four keep the same order".
- §6.6 α range omitted the best-output column → now ".57–.78 for the ratings and .54–.61 for the best-output choice".
- §6.2 said native-script edits raise romanized "output" → it is romanized early-token mass.

## D. Status after the editorial rewrite (Oct 12)

Resolved in the text: B2 (DEFAULT no longer called independent of ΔM), B3 (tests table now tests the full-response rate), B4 (M4 described as tuned), B5 (ΔS defined as the six-benchmark mean; FLORES composite flagged), B6 ("eight methods"), B7 (Brier, LID-10 and LCB-as-predictor promises removed), B9 ("every headline comparison"), B10 (no "every mean" claim), B11 ("effective rank, rounded up"), B14 (causal claim removed), B15 ("each rated by 2–3 annotators").

SAE regimes corrected to the published sources: Llama Scope, Gemma Scope and Qwen-Scope are base-trained; Table 13 compares with Gemma Scope's instruction-tuned SAEs for Gemma-2-9B-it (layer still a placeholder).

Disputed values kept out of the main text until the data are fixed: the unedited mean of 1.6% (§3.4 now says "no pair exceeds 3.9%"), P(stay) at t=0 (.246), the Figure 13 diagonal (+38.2) and the English drop (−31.9). They remain in the appendix and in A1–A3 above.

Still open: all of section A, plus B1, B8, B12, B13, B16–B18.
