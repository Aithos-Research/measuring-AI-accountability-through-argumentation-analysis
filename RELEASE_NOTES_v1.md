# Walton-Schemes Paper — v1 (Publication Release)

This release accompanies the paper

> **Measuring AI Accountability Through Argumentation Analysis: Can Model Reasoning Withstand Scrutiny?**
> Arno Libert, Daan Henselmans — Aithos Research Foundation, Amsterdam (PCAIDE 2026).

It contains the canonical evaluation logs, analyses, paper figures, and a downloadable archive of the raw Inspect-AI eval traces backing every figure, table, and statistical claim in the paper.

---

## What's in the repo (browse online)

- `logs/deliberative_justification_v2/20260527_combined_panels_curation/reasoning/` — canonical reasoning panel (7 native-reasoning models, post-curation `samples.json` + analysis CSVs + figures).
  - `_no_gpt/` — scheme analyses with GPT 5.4 excluded (its reasoning channel is encrypted ciphertext).
  - `figure_data/` — exact CSVs used to render each paper figure.
  - `by_judge/` — per-judge audit split (paper appendix).
- `logs/deliberative_justification_v2/20260522_combined_9models/cot/` — canonical CoT panel (Mistral Large 2512, Llama 3.3) with `samples.json` + analyses + `phase2_values_*.csv` (value-vocabulary table, §A.5).
- `paper_figures/` — byte-identical copies of the figures shipped in the paper, plus their underlying CSVs.
- `eval_files_manifest.json` — provenance of every raw `.eval` file (model, timestamp, source folder, size).

## What's in this Release artifact

**`WaltonSchemes_paper_logs.zip`** (~553 MB compressed, ~1.8 GB uncompressed): everything above **plus the raw Inspect-AI `.eval` files + `.json` sidecars** for all nine models. The zip mirrors the repo layout under a single root folder and adds `reasoning/eval_files/` and `cot/eval_files/` subtrees:

```
WaltonSchemes_paper_logs/
├── README.md
├── eval_files_manifest.json
├── paper_figures/                  (34 files)
├── reasoning/
│   ├── samples.json
│   ├── *.csv / *.png               (analysis + figures)
│   ├── figure_data/
│   ├── _no_gpt/
│   ├── by_judge/
│   └── eval_files/                 (RELEASE-ONLY: 56 .eval + 56 .json, 916 MB)
│       ├── claude-sonnet-4-6/      (7 runs, 116 MB)
│       ├── deepseek-v4-pro/        (8 runs, 120 MB)
│       ├── gemini-2.5-pro/         (8 runs, 145 MB)
│       ├── glm-5p1/                (8 runs, 139 MB)
│       ├── gpt-5.4/                (9 runs, 101 MB)
│       ├── kimi-k2p6/              (8 runs, 163 MB)
│       └── qwen3p6-plus/           (8 runs, 176 MB)
└── cot/
    ├── samples.json
    ├── *.csv / *.png
    ├── phase2_values_*.csv
    ├── by_judge/
    └── eval_files/                 (RELEASE-ONLY: 32 .eval + 32 .json, 271 MB)
        ├── llama-3.3/              (15 partial runs, 31 MB)
        └── mistral-large-2512/     (17 runs, 252 MB)
```

The eval files are kept out of the git tree because their combined 1.2 GB pushes a normal clone past the recommended GitHub repo size. Every `.eval` is paired with its `.json` sidecar (standard Inspect-AI layout) so reviewers can re-load them in Inspect-AI's UI or rerun the analysis pipeline.

## Reproducibility

- Inspect-AI 0.3.114
- SciPy 1.16.3 (`scipy.stats`)
- Subject-model generation: `T = 0.2`, `max_tokens = 8192`
- Reasoning elicitation (Inspect-AI `GenerateConfig`): `reasoning_effort=low`, `reasoning_tokens=4096`, applied uniformly across the reasoning-branch models.

## Curation note

The reasoning panel went through a human-curation round to resolve cases where the two Phase-4 judges (Claude Sonnet 4.6 and GPT 5.4) disagreed. Intermediate snapshots (`samples.bak_*.json`) are not included in either the repo or the release; they are available on request and recoverable from the analysis CSVs together with the canonical `samples.json`.
