"""Combined per-model heatmap: mean score | failure rate, side-by-side.

Merges the two per-(model, Govier sub-dim) heatmaps into a single
figure with two panels that share the y-axis (one model row each).
Left panel: mean score per dim on the normalised [0,1] scale
(sequential green colormap).
Right panel: failure rate per dim (% of cells at score = 1; sequential
red colormap).

Reads:
  - scores_per_model.csv         (mean-score columns per dim)
  - per_model_failure_by_dim.csv (post-analysis; or the consolidated
                                  figure3_failure_rate_per_dim.csv if
                                  the long form has been removed)

Writes:
  - per_model_score_and_failure.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap


PRETTY = {
    "acceptability": "Acc",
    "relevance": "Rel",
    "grounds": "Grnd",
    "coherence": "Cons",  # displayed as "consistency" (paper rename)
    "sufficiency": "Suf",
}
LOCAL_DIMS = ("acceptability", "relevance", "grounds")
# Substantive global dims first (acceptability, relevance, sufficiency);
# consistency (formerly "coherence") last because it is reported as a
# diagnostic rather than pooled into the substantive global mean.
GLOBAL_DIMS = ("acceptability", "relevance", "sufficiency", "coherence")
COT_MODELS = {"mistral-large-2512", "Llama 3.3", "llama-3.3", "llama-3p3"}

PROVIDER_SHORTNAME = {
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gpt-5.4":           "GPT 5.4",
    "kimi-k2p6":         "Kimi K2.6",
    "qwen3p6-plus":      "Qwen 3.6 Plus",
    "glm-5p1":           "GLM 5.1",
    "deepseek-v4-pro":   "DeepSeek V4 Pro",
    "gemini-2.5-pro":    "Gemini 2.5 Pro",
    "mistral-large-2512": "Mistral Large 2512",
    "Llama 3.3":         "Llama 3.3",
}


def _load_scores(run_dir: Path) -> pd.DataFrame:
    """Return wide DataFrame indexed by model with columns L.Acc, L.Rel,
    L.Grd, G.Coh, G.Acc, G.Rel, G.Suf containing mean scores."""
    df = pd.read_csv(run_dir / "scores_per_model.csv")
    # Global consistency (formerly coherence) is reported separately as a
    # diagnostic and intentionally omitted from this heatmap.
    cols = {
        "local_acceptability":   "L.Acc",
        "local_relevance":       "L.Rel",
        "local_grounds":         "L.Grnd",
        "global_acceptability":  "G.Acc",
        "global_relevance":      "G.Rel",
        "global_sufficiency":    "G.Suf",
    }
    out = df[["model"] + list(cols.keys())].copy()
    out = out.rename(columns=cols).set_index("model")
    return out


def _load_failure(run_dir: Path) -> pd.DataFrame:
    """Return wide DataFrame indexed by model with the same column order,
    containing the per-(model, dim) failure rate in %.
    """
    consolidated = run_dir / "figure3_failure_rate_per_dim.csv"
    long_form = run_dir / "per_model_failure_by_dim.csv"
    if consolidated.exists():
        df = pd.read_csv(consolidated)
        # Drop G.Cons (consistency diagnostic) — reported in its own table.
        cols = ["L.Acc_pct", "L.Rel_pct", "L.Grnd_pct",
                "G.Acc_pct", "G.Rel_pct", "G.Suf_pct"]
        out = df[["model"] + cols].copy()
        out.columns = ["model", "L.Acc", "L.Rel", "L.Grnd",
                       "G.Acc", "G.Rel", "G.Suf"]
        return out.set_index("model")
    if long_form.exists():
        df = pd.read_csv(long_form)
        by_model = {}
        for _, r in df.iterrows():
            col = ("L." if r["scope"] == "local" else "G.") + PRETTY[r["dim"]]
            by_model.setdefault(r["model"], {})[col] = float(r["pct"])
        cols = ["L." + PRETTY[d] for d in LOCAL_DIMS] + \
               ["G." + PRETTY[d] for d in GLOBAL_DIMS]
        return pd.DataFrame(by_model).T.reindex(columns=cols)
    raise SystemExit(
        f"need either {consolidated.name} or {long_form.name} in {run_dir}"
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("-o", "--output", type=Path, default=None)
    args = p.parse_args(argv)

    scores = _load_scores(args.run_dir)
    fail = _load_failure(args.run_dir)

    cols = ["L.Acc", "L.Rel", "L.Grnd", "G.Acc", "G.Rel", "G.Suf"]
    scores = scores.reindex(columns=cols)
    fail = fail.reindex(columns=cols)

    # Sort each panel alphabetically by display name; reasoning-branch
    # panel renders above the CoT panel, separated by the dashed line.
    def _alpha_key(m):
        return PROVIDER_SHORTNAME.get(m, m).lower()

    reasoning = sorted([m for m in fail.index if m not in COT_MODELS],
                       key=_alpha_key)
    cot = sorted([m for m in fail.index if m in COT_MODELS],
                 key=_alpha_key)
    row_order = reasoning + cot
    split_index = len(reasoning)
    scores = scores.reindex(row_order)
    fail = fail.reindex(row_order)
    pretty_index = [PROVIDER_SHORTNAME.get(m, m) for m in row_order]
    scores.index = pretty_index
    fail.index = pretty_index

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 0.55 * len(scores) + 1.8),
        gridspec_kw={"wspace": 0.28},
    )

    # Left panel: failure rate (bad = red); carries the model labels.
    vmax_fail = max(50.0, float(np.nanmax(fail.values)) if fail.size else 50.0)
    sns.heatmap(
        fail.values, annot=True, fmt=".1f",
        cmap="Reds", vmin=0, vmax=vmax_fail,
        cbar_kws={"label": "% cells with score = 1", "fraction": 0.045, "pad": 0.02},
        xticklabels=fail.columns, yticklabels=fail.index,
        ax=ax1, linewidths=0.4, linecolor="white",
    )
    ax1.set_title("Failure rate per Govier sub-dimension")
    ax1.set_ylabel("")
    ax1.tick_params(axis="x", rotation=0)

    # Right panel: mean score (good = green); shares y-axis, no labels.
    sns.heatmap(
        scores.values, annot=True, fmt=".2f",
        cmap="RdYlGn", vmin=0, vmax=1,
        cbar_kws={"label": "Mean normalized score (0--1)", "fraction": 0.045, "pad": 0.02},
        xticklabels=scores.columns, yticklabels=False,
        ax=ax2, linewidths=0.4, linecolor="white",
    )
    ax2.set_title("Mean normalized score per Govier sub-dimension")
    ax2.tick_params(axis="x", rotation=0)

    # Panel separator: dashed horizontal line between the reasoning-branch
    # and CoT-panel rows on both heatmaps.
    if 0 < split_index < len(row_order):
        for ax in (ax1, ax2):
            ax.axhline(split_index, color="black", linestyle="--",
                       linewidth=1.0, alpha=0.85)
        # Dashed vertical line marking local vs global split.
        for ax in (ax1, ax2):
            ax.axvline(3, color="black", linestyle="--",
                       linewidth=0.8, alpha=0.55)

    out = args.output or (args.run_dir / "per_model_score_and_failure.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote: {out}")


if __name__ == "__main__":
    raise SystemExit(main())
