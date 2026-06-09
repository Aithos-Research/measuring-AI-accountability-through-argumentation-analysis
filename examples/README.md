# Example Inspect-AI `.eval` runs

Small representative Inspect-AI traces from the publication — **one per
(model, elicitation regime) pair** — so anyone browsing the repo or watching a
presentation can immediately open a real trace from any of the 9 frontier
models the paper evaluates.

Each `.eval` file is paired with its `.json` sidecar (standard Inspect-AI
layout). Total folder size is ~35 MB; the full publication set (1.2 GB across
88 files) is shipped separately as a GitHub Release artifact.

## Reasoning panel — `<model>__reasoning.eval`

Native reasoning channel feeds `r_int`. Files cover the 7 reasoning-branch
models the paper evaluates with the `deliberative_justification_v2_reasoning`
task:

| File | Note |
|---|---|
| `claude-sonnet-4.6__reasoning.eval` | `thinking.type.enabled` block as `r_int` |
| `gpt-5.4__reasoning.eval`           | **Encrypted ciphertext** as `r_int` — motivates GPT exclusion from H2 / scheme analyses (paper §3.3) |
| `gemini-2.5-pro__reasoning.eval`    | Native reasoning block |
| `deepseek-v4-pro__reasoning.eval`   | Fireworks-hosted thinking variant |
| `kimi-k2.6__reasoning.eval`         | Fireworks-hosted thinking variant |
| `glm-5.1__reasoning.eval`           | Fireworks-hosted thinking variant |
| `qwen3.6-plus__reasoning.eval`      | Fireworks-hosted thinking variant |

## CoT panel — `<model>__cot.eval`

Visible chain-of-thought elicitation feeds the **justification track**
(`r_ext`) for all 9 models with the `deliberative_justification_v2_cot` task.
The two models without native reasoning (Mistral, Llama) use this regime as
their `r_int` substitute (pre-`<answer>` prose); the other seven appear here
because the paper also includes their CoT-elicited justifications as a
parallel comparison run.

| File | Panel role |
|---|---|
| `claude-sonnet-4.6__cot.eval`       | reasoning-branch model, justification track |
| `gpt-5.4__cot.eval`                 | reasoning-branch model, justification track |
| `gemini-2.5-pro__cot.eval`          | reasoning-branch model, justification track |
| `deepseek-v4-pro__cot.eval`         | reasoning-branch model, justification track |
| `kimi-k2.6__cot.eval`               | reasoning-branch model, justification track |
| `glm-5.1__cot.eval`                 | reasoning-branch model, justification track |
| `qwen3.6-plus__cot.eval`            | reasoning-branch model, justification track |
| `mistral-large-2512__cot.eval`      | **CoT panel** — visible CoT prose serves as `r_int` |
| `llama-3.3__cot.eval`               | **CoT panel** — visible CoT prose serves as `r_int` |

## How to view

### Web viewer (no install)

> https://inspect.aisi.org.uk/log-viewer

Drag-and-drop any `.eval` file into the page. You'll see the per-sample tree:

- **Phase 1** — Response elicitation: the model's verdict + `r_int` + `r_ext`.
- **Phase 2** — Scheme classification: judge picks the Walton scheme and
  generates situated critical questions per track.
- **Phase 3** — Defence: model answers each critical question.
- **Phase 4** — Argument-quality scoring: per-question + per-cell scores by
  two judges on the 7 Govier dimensions.

### Local CLI

```bash
pip install inspect-ai
inspect view examples/gpt-5.4__reasoning.eval
```

## What the file sizes mean

Each example here is the **smallest viable run** for that (model, task) pair —
roughly 0.1–0.6 MB, covering ~5–20 dilemmas per file. They are subsets, not
the full 200-dilemma runs used in the paper. They are sufficient to inspect
the trace structure, prompts, and scoring; they are **not** sufficient to
re-derive paper statistics.

## Full publication set

The 88 `.eval` files (with sidecars) that produced every paper number are at:

> Releases → `v1.0.0` → `WaltonSchemes_paper_logs.zip`

The zip mirrors the repo layout under
`reasoning/eval_files/<model>/` and `cot/eval_files/<model>/`.
Per-file provenance is in
[`eval_files_manifest.json`](../eval_files_manifest.json) at the repo root.
