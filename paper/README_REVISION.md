# Revision notes: Neural FOXP2 (ARR resubmission)

`latex/acl_latex.tex` and `latex/appendix.tex` have been rewritten. `latex/custom.bib` has new entries appended (old entries kept).

## What the source contains

- **Main paper (8 pages):** new narrative, 6-language suite (es, fr, zh, ru, hi, ar), 4 models, 11 methods (M0–M10), a section on the defaultness metric, corrected method equations, 8 empty tables and 6 figure slots.
- **Appendix (sections A–N):** data, dictionaries, method details, baselines, metric definitions, the 15-test metric suite (D1–D15), per-model results, ablations, script control, robustness, utility/intent/safety, human evaluation, cost, statistical tests and example tables. 37 empty tables and 14 figure slots.
- No result values are written anywhere. Every number must come from your runs.

## Placeholders

| Marker | Looks like | Meaning |
| --- | --- | --- |
| `\res{...}` | red bold `[...]` | A value you must fill from the runs. Stays visible even with notes off. |
| `\finding{...}` | orange box | What the paragraph must say once results exist, including how to word it if the result is negative. Delete each box after writing the paragraph. |
| `\figslot{width}{height}{path}{spec}` | grey box | Loads `path` automatically once the file exists; until then shows the spec. |
| Empty table cells | blank | Fill by script from the master results CSV. |

Before submission:
1. Set `\authornotesfalse` in the preamble (hides orange boxes).
2. Search for `\res{` — the count must be zero.
3. Search for `\finding{` — the count must be zero.

## Figures

All expected figure files and their specs are listed in `Figures/revision/FIGURES.md`. Save plots as vector PDF at those paths. The old images in `Figures/` and `latex/*.png` are no longer referenced and can be deleted.

## Items to verify before submission

- `mondal2025language` in `custom.bib` (venue details marked VERIFY).
- XSafety language coverage (the paper states all six languages are covered).
- Whether each SAE release was trained on base or instruction-tuned activations (Table "Models and SAE dictionaries", Limitations paragraph).
- Whether Gemma Scope has instruction-tuned SAEs inside your window (Table "Base- vs instruction-trained dictionaries"; delete it if none exist).
- The old `templeton2026...` entry is no longer cited; the paper cites the 2024 Transformer Circuits publication (`templeton2024scaling`).
- `latex/FAQ.tex` is not included in the paper. Do not add it back.

## Compiling

Same layout as before: main file `latex/acl_latex.tex`, compiled from the project root (Overleaf default for this project). New packages: amsmath, amssymb, booktabs, multirow, xcolor, xspace, enumitem, placeins (all standard on Overleaf).

The example tables (Appendix N) will need native-script text (Chinese, Russian, Hindi, Arabic). pdfLaTeX cannot typeset these scripts. Either compile with LuaLaTeX using `latex/acl_lualatex.tex` as the template with suitable fonts, or insert the examples as images.
