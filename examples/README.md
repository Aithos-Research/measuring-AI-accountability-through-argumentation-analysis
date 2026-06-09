# Example Inspect-AI `.eval` runs

Two small representative Inspect-AI runs from the publication, included so
anyone browsing the repo (or watching a presentation) can open a real eval
trace immediately without downloading the 1.2 GB release archive.

## Files

| File | Panel | Model | Task | Note |
|---|---|---|---|---|
| `demo_llama3.3_cot.eval` (+ `.json`) | CoT | Llama 3.3 | `deliberative_justification_v2_cot` | Partial run (~10 dilemmas). Useful for a quick look at the CoT elicitation. |
| `demo_gpt5.4_reasoning_encrypted_r_int.eval` (+ `.json`) | Reasoning | GPT 5.4 | `deliberative_justification_v2_reasoning` | Showcases the **encrypted reasoning channel** — `r_int` is returned as ciphertext, motivating GPT 5.4's exclusion from the H2 paired test and the per-scheme analyses. |

## How to view

### Web viewer (no install needed)

Upload the `.eval` file to the public Inspect-AI viewer:

> https://inspect.aisi.org.uk/log-viewer

Drag-and-drop the `.eval` and inspect the prompts, model outputs, and Phase-4
scoring per sample.

### Local CLI

```bash
pip install inspect-ai
inspect view examples/demo_gpt5.4_reasoning_encrypted_r_int.eval
```

Opens an interactive browser tab with the full trace tree:
Phase 1 (response elicitation) → Phase 2 (scheme classification + critical
questions) → Phase 3 (defence) → Phase 4 (ARG scoring by two judges).

## Full publication eval set

The two files here are a tiny illustrative slice. The full set used in the
paper is 88 `.eval` files across 9 models, ~1.2 GB uncompressed, available as
a GitHub Release artifact:

> Releases → `v1.0.0` → `WaltonSchemes_paper_logs.zip` (553 MB compressed)

The zip mirrors the repo layout and adds:
- `reasoning/eval_files/{claude-sonnet-4-6, deepseek-v4-pro, gemini-2.5-pro, glm-5p1, gpt-5.4, kimi-k2p6, qwen3p6-plus}/`
- `cot/eval_files/{llama-3.3, mistral-large-2512}/`

Per-file provenance (timestamp, source folder, size) is documented in
[`eval_files_manifest.json`](../eval_files_manifest.json) at the repo root.
