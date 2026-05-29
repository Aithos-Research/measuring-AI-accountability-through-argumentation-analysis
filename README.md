# Measuring Argumentative Moral Competence in LLMs via Walton Schemes

Reproduction code and data for the paper

> *Measuring Argumentative Moral Competence in Large Language Models via Walton Schemes.* (PCAIDE 2026, under review.)

The paper introduces a four-phase dialectical protocol that scores a language model not on which verdict it reaches on a moral dilemma, but on whether the **defence** it can mount for that verdict survives the critical questions of the Walton argumentation scheme it instantiates. Applied to nine frontier LLMs across 200 high-ambiguity MoralChoice dilemmas, the protocol yields 7,176 judge-scored defence cells whose analysis is the basis of the paper's findings.

This repository contains:

- the protocol implementation (`src/moral_eval/tasks/explainability/deliberative_justification_v2.py`)
- the analysis pipeline (`src/moral_eval/analysis/`)
- the scripts used to merge partial runs, rejudge cells, and produce the paper's figures and tables (`scripts/`)
- the 200-dilemma MoralChoice selection used in the paper (`data/selections/`)
- the Walton-scheme prompt set (`data/walton_schemes/`)
- the merged 9-model dataset that produced every paper number (`logs/.../samples.json`)
- the CSV summary tables and PNG figures the paper renders

The repository is licensed [MIT](LICENSE).

## Repository layout

```
walton-schemes-paper/
├── pyproject.toml                          # Python dependencies (uv / pip)
├── run_trial.py                            # Top-level entry point (Inspect AI driver)
├── data/
│   ├── selections/
│   │   ├── moralchoice_high_ambiguity_seed42_001-200.json  # paper selection
│   │   └── moralchoice_blocklist.txt                       # corpus filter
│   └── walton_schemes/                     # Phase-2/3/4 prompts + scheme definitions
├── src/
│   ├── config/
│   │   ├── experiments_walton_moralchoice.yaml             # paper-reproduction config
│   │   ├── test_walton_moralchoice.yaml                    # 5-dilemma smoke test
│   │   └── models.json                                     # provider/model aliases
│   └── moral_eval/
│       ├── tasks/explainability/
│       │   └── deliberative_justification_v2.py            # the four-phase protocol
│       ├── analysis/
│       │   ├── walton_analysis.py                          # aggregate metrics, CSVs
│       │   ├── walton_plots.py                             # paper figures
│       │   ├── walton_report.py                            # human-readable summary
│       │   ├── judge_disagreement.py                       # inter-judge audit
│       │   └── judge_disagreement_curation.py              # polarised-cell curation
│       ├── utils/                                          # config + prompt loaders
│       └── datasets/                                       # MoralChoice loader
├── scripts/
│   ├── merge_runs.py                                       # merge partial .eval runs
│   ├── refill_phase4_judge.py                              # rejudge skipped cells
│   ├── prep_rejudge_cot_panel.py                           # GPT 5.5 → 5.4 prep
│   ├── filter_gpt_and_replot_schemes.py                    # GPT-excluded scheme analyses
│   ├── plot_per_model_int_ext.py                           # Figure 1 generator
│   ├── audit_rejudged_numbers.py                           # cross-check after refill
│   ├── audit_scheme_without_gpt.py                         # GPT-excluded H3 statistics
│   ├── extract_phase2_values.py                            # Phase-2 value lexicon
│   └── backfill_coherence_from_consistency.py              # coherence/consistency fix
└── logs/
    └── deliberative_justification_v2/
        └── 20260527_combined_panels_curation/reasoning/
            ├── samples.json                                # the 9-model merged dataset
            ├── *.csv                                       # per-model / per-scheme summaries
            └── *.png                                       # paper figures
```

## Setup

The project targets Python ≥ 3.11. We recommend [`uv`](https://github.com/astral-sh/uv) for environment management, but plain `pip install -e .` works.

```bash
git clone https://github.com/<your-org>/walton-schemes-paper.git
cd walton-schemes-paper

# Option A: uv
uv sync                                         # creates .venv/ from pyproject.toml

# Option B: pip
python -m venv .venv
source .venv/bin/activate                       # Windows: .venv\Scripts\activate
pip install -e .
```

API keys for the providers used in the paper are read from a `.env` file at the repository root:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
FIREWORKS_API_KEY=...
MISTRAL_API_KEY=...
```

If you only want to reproduce the paper's *analysis* (regenerate the figures and tables from `samples.json`), no API keys are needed.

## Reproducing the paper's numbers from `samples.json`

The merged 9-model dataset that backs every number in the paper is included at:

```
logs/deliberative_justification_v2/20260527_combined_panels_curation/reasoning/samples.json
```

Re-run the analysis pipeline:

```bash
# Recompute per-model / per-scheme / failure-room CSVs
python -m moral_eval.analysis.walton_analysis \
    logs/deliberative_justification_v2/20260527_combined_panels_curation/reasoning

# Regenerate paper figures (per-dim heatmap, scheme distribution, etc.)
python -m moral_eval.analysis.walton_plots \
    logs/deliberative_justification_v2/20260527_combined_panels_curation/reasoning

# Inter-judge disagreement audit
python -m moral_eval.analysis.judge_disagreement \
    logs/deliberative_justification_v2/20260527_combined_panels_curation/reasoning

# Figure 1 (per-model int/ext, with GPT 5.4 r_int marked NA)
python scripts/plot_per_model_int_ext.py \
    logs/deliberative_justification_v2/20260527_combined_panels_curation/reasoning

# GPT-excluded scheme analyses (Figures 3-5 and Tables 3, 5)
python scripts/filter_gpt_and_replot_schemes.py \
    logs/deliberative_justification_v2/20260527_combined_panels_curation/reasoning

# Audit script that reports the rejudged hedge / Wilcoxon / inter-judge numbers
python scripts/audit_rejudged_numbers.py
```

The regenerated CSVs and PNGs land alongside `samples.json`.

## Running the protocol from scratch

To reproduce the protocol end-to-end (≈ 800 Phase-4 calls per model × judges, plus the Phase-2/Phase-3 cascade — significant API spend, several hours):

```bash
# Smoke test: 5 dilemmas, 8 models
python run_trial.py --config src/config/test_walton_moralchoice.yaml

# Full paper run: 200 dilemmas, 9 models, 2 judges
python run_trial.py --config src/config/experiments_walton_moralchoice.yaml
```

Each model run produces an Inspect AI `.eval` log under `logs/deliberative_justification_v2/<timestamp>/`. The paper's `samples.json` was assembled by merging multiple per-model runs:

```bash
# Merge partial / per-model runs into a single combined samples.json
python scripts/merge_runs.py \
    --runs logs/.../<run-1> logs/.../<run-2> ... \
    --out logs/.../combined/reasoning \
    --additive
```

If any cells were skipped during the run because of judge refusals or quota errors, refill them:

```bash
python scripts/refill_phase4_judge.py \
    --cot-dir logs/.../combined/reasoning \
    --judge-alias gpt_5_4 \
    --judge-model openai_gpt54
```

## Software versions

The paper's analyses were run with the following stack. Versions matter for the H2 Wilcoxon and H3 $\chi^2$ tests (default switching behaviour, CI methods).

| Package | Version |
|---|---|
| Python | 3.11 |
| Inspect AI | 0.3.144 |
| SciPy | 1.16.3 |
| NumPy | 2.3.4 |
| statsmodels | 0.14.5 |
| matplotlib | 3.10.7 |
| pandas | 2.3.3 |
| seaborn | 0.13.2 |

## Methodological notes

- **GPT 5.4's $r_{\mathrm{int}}$ is encrypted ciphertext.** OpenAI returns the reasoning channel as a Fernet blob that judges cannot read. The paper marks GPT 5.4's $r_{\mathrm{int}}$ as NA in Table 1 / Figure 1 and excludes GPT 5.4 from the H2 paired Wilcoxon test, the H3 scheme-distribution / scheme-transitions / per-scheme-retention analyses, and the local-failure-by-scheme heatmap (Figure 3). The exclusion is reproduced by `scripts/filter_gpt_and_replot_schemes.py`, which writes a GPT-filtered copy of `samples.json` to a `_no_gpt/` subdirectory and re-runs the analysis pipeline on it.
- **Coherence is reported as a diagnostic, not pooled into the global mean.** The `mean_normalized_global_quality` column in `per_model_metrics.csv` averages over acceptability, relevance and sufficiency only; coherence is reported separately in its own column. See `walton_analysis.py` `_GLOBAL_DIMS_NO_COH`.
- **The hedge-rate Mann–Whitney analysis** (Methods §3.7) is computed with the hedge dictionary in `walton_analysis.py` (or the equivalent appendix regex). To reproduce the hedge buckets and per-model hedge rates, run `python scripts/audit_rejudged_numbers.py`.

## Citation

If you use this protocol, the prompts, or the released data, please cite:

```
TODO bibtex
```

## License

Code is released under the MIT License (see [LICENSE](LICENSE)).

The MoralChoice corpus is the property of its original authors \[Scherrer et al., 2023\] and is used under their license terms. Only the dilemma-id selection and per-dilemma blocklist used in the paper are redistributed here; the dilemma texts themselves are loaded from the upstream HuggingFace dataset at runtime.

## Contact

Issues and questions: please open a GitHub issue.
