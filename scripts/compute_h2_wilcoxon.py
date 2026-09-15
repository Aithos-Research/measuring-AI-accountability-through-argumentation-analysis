"""Paired Wilcoxon signed-rank tests for H2 (reasoning vs. justification).

Reproduces Table `tab:int-ext`: for each Govier dimension, tests
H0: r_r =d= r_j, paired by (dilemma, judge), separately per model and
pooled across the eight-model panel with an interpretable r_r (GPT-5.4
excluded -- its reasoning channel is encrypted ciphertext).

No script in either repository previously computed this despite Table
`tab:int-ext` reporting it: `wilcoxon` is imported in
audit_rejudged_numbers.py but never called there. This fills that gap so
the table can be regenerated after any samples.json update (e.g. the
Phase-3 state-isolation fix).
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from scipy.stats import wilcoxon

SAMPLES = Path("logs/deliberative_justification_v2/20260527_combined_panels_curation/reasoning/samples.json")

SCALE_MIN, SCALE_MAX = 1, 3
SCALE_RANGE = SCALE_MAX - SCALE_MIN

LOCAL_DIMS = ("acceptability", "relevance", "grounds")
GLOBAL_DIMS_NO_COH = ("acceptability", "relevance", "sufficiency")

# The seven Govier dimensions the paper counts "dimensions where reasoning
# wins" over, plus the two composite means reported in their own columns.
DIM_LABELS = {
    "local_acceptability": "Local acceptability",
    "local_relevance": "Local relevance",
    "local_grounds": "Local grounds",
    "commitment": "Commitment stability",
    "global_acceptability": "Global acceptability",
    "global_relevance": "Global relevance",
    "global_sufficiency": "Global sufficiency",
}
SEVEN_DIMS = list(DIM_LABELS)
COMPOSITE_LABELS = {"local_mean": "Local-mean", "global_mean": "Global-mean"}

ORDER = [
    "claude-sonnet-4-6", "deepseek-v4-pro", "gemini-2.5-pro", "glm-5p1",
    "gpt-5.4", "kimi-k2p6", "qwen3p6-plus",
    "mistral-large-2512", "Llama 3.3",
]
COT_MODELS = {"mistral-large-2512", "Llama 3.3"}


def norm(v):
    return (v - SCALE_MIN) / SCALE_RANGE


def cell_dims(p4):
    """One phase-4 judge cell -> {dim_key: normalized 0-1 score or None}.

    local_mean / global_mean mirror plot_per_model_int_ext.py's
    cell_local_mean / cell_global_mean_no_coh exactly (flatten-and-average),
    so the composite columns match the figure/CSV convention, not just the
    per-dimension breakdown.
    """
    local = p4.get("local") or {}
    glob = p4.get("global") or {}
    out = {}

    for dim in LOCAL_DIMS:
        vals = [d.get(dim) for d in local.values()
                if isinstance(d, dict) and isinstance(d.get(dim), int)]
        out[f"local_{dim}"] = norm(sum(vals) / len(vals)) if vals else None

    out["commitment"] = norm(glob["coherence"]) if isinstance(glob.get("coherence"), int) else None
    for dim in GLOBAL_DIMS_NO_COH:
        v = glob.get(dim)
        out[f"global_{dim}"] = norm(v) if isinstance(v, int) else None

    local_flat = [v for dims in local.values() if isinstance(dims, dict)
                  for v in dims.values() if isinstance(v, int)]
    out["local_mean"] = norm(sum(local_flat) / len(local_flat)) if local_flat else None
    global_flat = [glob.get(k) for k in GLOBAL_DIMS_NO_COH if isinstance(glob.get(k), int)]
    out["global_mean"] = norm(sum(global_flat) / len(global_flat)) if global_flat else None

    return out


def wilcoxon_row(rj_pairs):
    """(advantage, p, n) for a list of (r_r, r_j) pairs, or (None, None, 0)."""
    if len(rj_pairs) < 2:
        return None, None, len(rj_pairs)
    diffs = [rv - jv for rv, jv in rj_pairs]
    adv = statistics.mean(diffs)
    if all(d == 0 for d in diffs):
        return adv, 1.0, len(diffs)
    try:
        _, p = wilcoxon(diffs, alternative="two-sided")
    except ValueError:
        # All-zero after scipy's internal zero-dropping.
        p = 1.0
    return adv, p, len(diffs)


def stars(p):
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def main():
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))

    all_dim_keys = SEVEN_DIMS + list(COMPOSITE_LABELS)
    # pairs[model][dim] -> [(r_r_value, r_j_value), ...] paired by (dilemma, judge)
    pairs = defaultdict(lambda: defaultdict(list))

    for dil in samples["dilemmas"]:
        for tg in dil["targets"]:
            model = tg["model"].split("/")[-1]
            by_track = {}
            for tk in tg.get("tracks", []) or []:
                track = tk.get("text_kind")
                if track not in ("reasoning", "justification"):
                    continue
                by_judge = {}
                for p4 in tk.get("phase4", []) or []:
                    if p4.get("skipped"):
                        continue
                    judge = p4.get("phase4_judge")
                    if judge:
                        by_judge[judge] = cell_dims(p4)
                by_track[track] = by_judge
            r_cells = by_track.get("reasoning") or {}
            j_cells = by_track.get("justification") or {}
            for judge in set(r_cells) & set(j_cells):
                rc, jc = r_cells[judge], j_cells[judge]
                for dim in all_dim_keys:
                    rv, jv = rc.get(dim), jc.get(dim)
                    if rv is None or jv is None:
                        continue
                    pairs[model][dim].append((rv, jv))

    # ====================================================================
    # Per-model rows
    # ====================================================================
    print("=== Per-model: Local-mean adv., Global-mean adv., dims winning (p<0.05, /7) ===")
    for model in ORDER:
        if model == "gpt-5.4":
            print(f"  {model:20s}  ---  (r_r encrypted, excluded)")
            continue
        local_adv, local_p, n_local = wilcoxon_row(pairs[model]["local_mean"])
        global_adv, global_p, n_global = wilcoxon_row(pairs[model]["global_mean"])
        n_win = 0
        for dim in SEVEN_DIMS:
            adv, p, n = wilcoxon_row(pairs[model][dim])
            if adv is not None and p is not None and adv > 0 and p < 0.05:
                n_win += 1
        print(f"  {model:20s}  local={local_adv:+.4f}{stars(local_p):<3s} "
              f"(n={n_local:4d})  global={global_adv:+.4f}{stars(global_p):<3s} "
              f"(n={n_global:4d})  wins={n_win}/7")

    # ====================================================================
    # Panel-wide pooled rows per Govier dimension (excludes GPT-5.4)
    # ====================================================================
    print()
    print("=== Panel-wide pooled (8 models, GPT-5.4 excluded) ===")
    pooled = defaultdict(list)
    for model in ORDER:
        if model == "gpt-5.4":
            continue
        for dim in all_dim_keys:
            pooled[dim].extend(pairs[model][dim])

    for dim in all_dim_keys:
        adv, p, n = wilcoxon_row(pooled[dim])
        label = DIM_LABELS.get(dim) or COMPOSITE_LABELS.get(dim, dim)
        if adv is None:
            print(f"  {label:22s}  n=0")
            continue
        print(f"  {label:22s}  advantage={adv:+.4f}{stars(p)}  p={p:.3e}  n={n}")


if __name__ == "__main__":
    main()
