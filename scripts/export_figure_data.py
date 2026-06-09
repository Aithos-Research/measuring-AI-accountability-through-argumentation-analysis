"""Produce one CSV per figure in the paper's Figures/combined folder.

Each output CSV is named exactly like its corresponding PNG and contains
only the data the figure plots, in the same row/column layout the figure
uses.  Nothing extra, no picking-and-choosing across multiple sources
within a single CSV: each figure's data is derived from its own
canonical source.

Usage:
    python scripts/export_figure_data.py <run_dir> <out_dir>

<run_dir> is the analysis run directory that holds samples.json and all
per-figure source CSVs.  <out_dir> is where the figure CSVs are written
(typically the paper's Figures/combined folder).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------- #

COT_MODELS = {"mistral-large-2512", "Llama 3.3", "llama-3.3", "llama-3p3"}

PROVIDER_PRETTY = {
    "claude-sonnet-4-6":  "Claude Sonnet 4.6",
    "gpt-5.4":            "GPT 5.4",
    "kimi-k2p6":          "Kimi K2.6",
    "qwen3p6-plus":       "Qwen 3.6 Plus",
    "glm-5p1":            "GLM 5.1",
    "deepseek-v4-pro":    "DeepSeek V4 Pro",
    "gemini-2.5-pro":     "Gemini 2.5 Pro",
    "mistral-large-2512": "Mistral Large 2512",
    "Llama 3.3":          "Llama 3.3",
}

DIM_PRETTY = {
    "acceptability": "Acc",
    "relevance":     "Rel",
    "grounds":       "Grnd",
    "coherence":     "Cons",  # paper rename: "consistency"
    "sufficiency":   "Suf",
}

LOCAL_DIMS  = ("acceptability", "relevance", "grounds")
# Substantive global dims only; consistency (coherence) is reported in
# its own separate table, not on the per-model heatmaps.
GLOBAL_DIMS = ("acceptability", "relevance", "sufficiency")
SEVEN_COLS  = [f"L.{DIM_PRETTY[d]}" for d in LOCAL_DIMS] + \
              [f"G.{DIM_PRETTY[d]}" for d in GLOBAL_DIMS]


def _split_reasoning_cot(models):
    reasoning = [m for m in models if m not in COT_MODELS]
    cot = [m for m in models if m in COT_MODELS]
    return reasoning, cot


def _alpha_panel_order(models):
    """Reasoning-panel alphabetical, then CoT-panel alphabetical."""
    def key(m):
        return PROVIDER_PRETTY.get(m, m).lower()
    reasoning = sorted([m for m in models if m not in COT_MODELS], key=key)
    cot = sorted([m for m in models if m in COT_MODELS], key=key)
    return reasoning + cot


def _apply_panel_order(df, model_col="model"):
    """Return df reindexed in alphabetical-within-panel order."""
    order = _alpha_panel_order(list(df[model_col]))
    return df.set_index(model_col).reindex(order).reset_index()


# --------------------------------------------------------------------- #
# 1.  failure_room_heatmap.csv
# --------------------------------------------------------------------- #

def export_failure_room_heatmap(run: Path, out: Path) -> None:
    """Per-(model, sub-dim) failure-rate heatmap.

    Rows: panel models, sorted by L.Grnd ascending (best at top).
    Columns: L.Acc, L.Rel, L.Grnd, G.Coh, G.Acc, G.Rel, G.Suf (% cells
    with score = 1 on that sub-dimension).
    """
    df = pd.read_csv(run / "figure3_failure_rate_per_dim.csv")
    keep = ["model"] + [f"{c}_pct" for c in SEVEN_COLS]
    df = df[keep].copy()
    df.columns = ["model"] + SEVEN_COLS
    df = _apply_panel_order(df)
    df["model"] = df["model"].map(lambda m: PROVIDER_PRETTY.get(m, m))
    df.to_csv(out / "failure_room_heatmap.csv", index=False,
              float_format="%.2f")


# --------------------------------------------------------------------- #
# 2.  model_score_heatmap.csv
# --------------------------------------------------------------------- #

def export_model_score_heatmap(run: Path, out: Path) -> None:
    """Per-(model, sub-dim) mean-score heatmap (normalised 0--1).

    Reasoning-branch models first, sorted by overall mean descending,
    then the CoT models (same order), separated by the dashed panel
    line in the figure.
    """
    df = pd.read_csv(run / "scores_per_model.csv")
    src_cols = ["local_acceptability", "local_relevance", "local_grounds",
                "global_acceptability", "global_relevance",
                "global_sufficiency"]
    df = df[["model"] + src_cols].copy()
    df = _apply_panel_order(df)
    df.columns = ["model"] + SEVEN_COLS
    df["model"] = df["model"].map(lambda m: PROVIDER_PRETTY.get(m, m))
    df.to_csv(out / "model_score_heatmap.csv", index=False,
              float_format="%.3f")


# --------------------------------------------------------------------- #
# 3.  per_model_score_and_failure.csv  (combined Fig 2+3)
# --------------------------------------------------------------------- #

def export_per_model_score_and_failure(run: Path, out: Path) -> None:
    """The combined figure has two panels (failure left, mean score
    right) sharing the row order.  Single CSV with both sets of columns
    side-by-side so the reader can re-derive both panels from one file.
    """
    fail = pd.read_csv(run / "figure3_failure_rate_per_dim.csv")
    fail = fail[["model"] + [f"{c}_pct" for c in SEVEN_COLS]].copy()
    fail.columns = ["model"] + [f"fail_{c}" for c in SEVEN_COLS]

    scores = pd.read_csv(run / "scores_per_model.csv")
    src_cols = ["local_acceptability", "local_relevance", "local_grounds",
                "global_acceptability", "global_relevance",
                "global_sufficiency"]
    scores = scores[["model"] + src_cols].copy()
    scores.columns = ["model"] + [f"score_{c}" for c in SEVEN_COLS]

    df = fail.merge(scores, on="model")
    # Same row order as the figure: reasoning first (alphabetical),
    # then CoT (alphabetical).
    df["__cot"] = df["model"].isin(COT_MODELS).astype(int)
    df["__alpha"] = df["model"].map(lambda m: PROVIDER_PRETTY.get(m, m).lower())
    df = df.sort_values(["__cot", "__alpha"])
    df = df.drop(columns=["__cot", "__alpha"])
    df["model"] = df["model"].map(lambda m: PROVIDER_PRETTY.get(m, m))
    df.to_csv(out / "per_model_score_and_failure.csv", index=False,
              float_format="%.3f")


# --------------------------------------------------------------------- #
# 4.  per_model_int_ext.csv
# --------------------------------------------------------------------- #

def export_per_model_int_ext(run: Path, out: Path) -> None:
    """Per-model r_int / r_ext bar chart.  Each cell is the mean of
    (local + global_no_coh) / 2 normalised to [0,1].  GPT 5.4's r_int
    is NA (encrypted reasoning channel)."""
    SCALE_MIN, SCALE_MAX = 1, 3
    SCALE_RANGE = SCALE_MAX - SCALE_MIN

    def cell_local_mean(local):
        nums = [v for q, dims in (local or {}).items()
                if isinstance(dims, dict)
                for v in dims.values() if isinstance(v, int)]
        return (sum(nums) / len(nums)) if nums else None

    def cell_global_mean_no_coh(g):
        nums = [v for k, v in (g or {}).items()
                if k in ("acceptability", "relevance", "sufficiency")
                and isinstance(v, int)]
        return (sum(nums) / len(nums)) if nums else None

    cells = defaultdict(list)
    samples = json.loads((run / "samples.json").read_text(encoding="utf-8"))
    for dil in samples["dilemmas"]:
        for tg in dil["targets"]:
            model = tg["model"].split("/")[-1]
            for tk in tg.get("tracks", []) or []:
                track = tk.get("text_kind")
                if track not in ("reasoning", "justification"):
                    continue
                for p4 in tk.get("phase4", []) or []:
                    if p4.get("skipped"):
                        continue
                    lm = cell_local_mean(p4.get("local"))
                    gm = cell_global_mean_no_coh(p4.get("global"))
                    if lm is None or gm is None:
                        continue
                    avg = ((lm + gm) / 2 - SCALE_MIN) / SCALE_RANGE
                    cells[(model, track)].append(avg)

    def stat(model, track):
        xs = cells.get((model, track), [])
        if not xs:
            return None, None
        m = sum(xs) / len(xs)
        if len(xs) < 2:
            return m, 0.0
        var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
        return m, (var / len(xs)) ** 0.5

    order = _alpha_panel_order(list(PROVIDER_PRETTY.keys()))
    rows = []
    for k in order:
        mi, sei = stat(k, "reasoning")
        me, see = stat(k, "justification")
        rows.append({
            "model": PROVIDER_PRETTY[k],
            "r_int":      "NA" if k == "gpt-5.4" else f"{mi:.4f}",
            "r_int_se":   "NA" if k == "gpt-5.4" else f"{sei:.4f}",
            "r_ext":      f"{me:.4f}",
            "r_ext_se":   f"{see:.4f}",
        })
    with (out / "per_model_int_ext.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------- #
# 5.  model_means_bar.csv
# --------------------------------------------------------------------- #

def export_model_means_bar(run: Path, out: Path) -> None:
    """Horizontal bar: per-model mean ± stderr (raw 1--3 scale).
    Sorted ascending so highest mean is at the top of the chart.
    """
    df = pd.read_csv(run / "scores_per_model.csv")
    df = df[["model", "mean", "stderr"]].copy()
    df = _apply_panel_order(df)
    df["model"] = df["model"].map(lambda m: PROVIDER_PRETTY.get(m, m))
    df.to_csv(out / "model_means_bar.csv", index=False, float_format="%.4f")


# --------------------------------------------------------------------- #
# 6--9.  per_model_<something> bar / heatmap charts
# --------------------------------------------------------------------- #

def export_per_model_failure_rates(run: Path, out: Path) -> None:
    df = pd.read_csv(run / "per_model_metrics.csv")
    keep = ["model", "overall_failure_rate_pct",
            "local_failure_rate_pct", "global_failure_rate_pct"]
    df = df[keep].copy()
    df = _apply_panel_order(df)
    df.columns = ["model", "Any", "Local", "Global"]
    df["model"] = df["model"].map(lambda m: PROVIDER_PRETTY.get(m, m))
    df.to_csv(out / "per_model_failure_rates.csv", index=False,
              float_format="%.2f")


def export_per_model_rfi_rates(run: Path, out: Path) -> None:
    df = pd.read_csv(run / "per_model_metrics.csv")
    keep = ["model", "overall_rfi_rate_pct",
            "local_rfi_rate_pct", "global_rfi_rate_pct"]
    df = df[keep].copy()
    df = _apply_panel_order(df)
    df.columns = ["model", "Any", "Local", "Global"]
    df["model"] = df["model"].map(lambda m: PROVIDER_PRETTY.get(m, m))
    df.to_csv(out / "per_model_rfi_rates.csv", index=False,
              float_format="%.2f")


def export_per_model_normalized_quality(run: Path, out: Path) -> None:
    df = pd.read_csv(run / "per_model_metrics.csv")
    keep = ["model", "mean_normalized_local_quality",
            "mean_normalized_global_quality"]
    df = df[keep].copy()
    df = _apply_panel_order(df)
    df.columns = ["model", "Local", "Global"]
    df["model"] = df["model"].map(lambda m: PROVIDER_PRETTY.get(m, m))
    df.to_csv(out / "per_model_normalized_quality.csv", index=False,
              float_format="%.4f")


def export_per_model_scheme_consistency(run: Path, out: Path) -> None:
    df = pd.read_csv(run / "per_model_metrics.csv")
    keep = ["model", "scheme_consistency_pct", "n_dilemmas_with_both_tracks"]
    df = df[keep].copy()
    df = _apply_panel_order(df)
    df.columns = ["model", "scheme_consistency_pct", "n_dilemmas"]
    df["model"] = df["model"].map(lambda m: PROVIDER_PRETTY.get(m, m))
    df.to_csv(out / "per_model_scheme_consistency.csv", index=False,
              float_format="%.2f")


def export_per_model_metrics_heatmap(run: Path, out: Path) -> None:
    """Two-panel heatmap: % rates on the left, normalised quality on the
    right.  Single CSV contains every cell shown in the figure.
    """
    df = pd.read_csv(run / "per_model_metrics.csv")
    keep = [
        "model",
        "scheme_consistency_pct",
        "overall_failure_rate_pct", "local_failure_rate_pct", "global_failure_rate_pct",
        "overall_rfi_rate_pct", "local_rfi_rate_pct", "global_rfi_rate_pct",
        "mean_normalized_local_quality", "mean_normalized_global_quality",
    ]
    df = df[keep].copy()
    df = _apply_panel_order(df)
    df.columns = [
        "model", "scheme_consist", "fail_any", "fail_local", "fail_global",
        "rfi_any", "rfi_local", "rfi_global", "norm_local", "norm_global",
    ]
    df["model"] = df["model"].map(lambda m: PROVIDER_PRETTY.get(m, m))
    df.to_csv(out / "per_model_metrics_heatmap.csv", index=False,
              float_format="%.4f")


# --------------------------------------------------------------------- #
# 10.  scheme_distribution.csv
# --------------------------------------------------------------------- #

def export_scheme_distribution(run: Path, out: Path) -> None:
    """Per-model stacked bar (top-8 schemes + Other, by track).
    GPT 5.4's reasoning track has zero classifications (encrypted r_int);
    its justification track is shown normally.
    """
    src = pd.read_csv(run / "scheme_distribution_per_model.csv")
    src = src[src["model"] != "__ALL__"].copy()

    # Long form so we can decide the top-8 schemes by panel-wide total.
    long = pd.concat([
        src.assign(track="reasoning", count=src["reasoning"]),
        src.assign(track="justification", count=src["justification"]),
    ])[["model", "scheme", "track", "count"]]

    top = (long.groupby("scheme")["count"].sum()
                .sort_values(ascending=False).head(8).index.tolist())
    long.loc[~long["scheme"].isin(top), "scheme"] = "Other"

    cols = list(top) + ["Other"]
    pivot_r = long[long["track"] == "reasoning"].pivot_table(
        index="model", columns="scheme", values="count",
        aggfunc="sum", fill_value=0).reindex(columns=cols, fill_value=0)
    pivot_j = long[long["track"] == "justification"].pivot_table(
        index="model", columns="scheme", values="count",
        aggfunc="sum", fill_value=0).reindex(columns=cols, fill_value=0)

    pivot_r.columns = [f"reasoning_{c}" for c in pivot_r.columns]
    pivot_j.columns = [f"justification_{c}" for c in pivot_j.columns]
    out_df = pivot_r.join(pivot_j, how="outer").fillna(0).astype(int)
    out_df.index.name = "model"
    out_df = out_df.reset_index()
    out_df = _apply_panel_order(out_df)
    out_df["model"] = out_df["model"].map(lambda m: PROVIDER_PRETTY.get(m, m))
    out_df.to_csv(out / "scheme_distribution.csv", index=False)


# --------------------------------------------------------------------- #
# 11.  scheme_transitions_heatmap.csv
# --------------------------------------------------------------------- #

def export_scheme_transitions_heatmap(run: Path, out: Path) -> None:
    """Reasoning → Justification transition matrix (top-8 × top-8, raw
    counts).  Rows: reasoning_scheme, cols: justification_scheme.
    """
    df = pd.read_csv(run / "scheme_transitions.csv")
    all_schemes = pd.concat([df["reasoning_scheme"],
                              df["justification_scheme"]]).value_counts().index
    top = list(all_schemes[:8])
    df = df[df["reasoning_scheme"].isin(top) & df["justification_scheme"].isin(top)]
    pivot = df.pivot_table(index="reasoning_scheme",
                            columns="justification_scheme",
                            values="count", aggfunc="sum", fill_value=0)
    pivot = pivot.reindex(index=top, columns=top, fill_value=0).astype(int)
    pivot.index.name = "reasoning_scheme"
    pivot.to_csv(out / "scheme_transitions_heatmap.csv")


# --------------------------------------------------------------------- #
# 12.  scores_per_scheme.csv
# --------------------------------------------------------------------- #

def export_scores_per_scheme(run: Path, out: Path) -> None:
    df = pd.read_csv(run / "scores_per_scheme.csv")
    df = df[["scheme", "mean_overall", "n_total"]].dropna(subset=["mean_overall"])
    df = df.sort_values("mean_overall", ascending=True)
    df.to_csv(out / "scores_per_scheme.csv", index=False, float_format="%.4f")


# --------------------------------------------------------------------- #
# 13.  local_failure_room_by_scheme.csv
# --------------------------------------------------------------------- #

def export_local_failure_room_by_scheme(run: Path, out: Path) -> None:
    """Scheme × Govier local-dim failure rate + mean normalised score
    (the two panels of the figure). Schemes ordered by
    n_cells_under_scheme descending. Column order follows the figure:
    L.Grnd, L.Acc, L.Rel."""
    df = pd.read_csv(run / "per_argument_failure_room_by_scheme.csv")
    local = df[df["level"] == "local"].copy()
    f_pct = local.pivot_table(index="scheme", columns="dimension",
                               values="pct_cells_with_failure",
                               aggfunc="first", fill_value=0)
    n_by_scheme = (local.groupby("scheme")["n_cells_under_scheme"]
                        .first().astype(int))
    keep = [s for s in n_by_scheme.index if n_by_scheme[s] > 0]
    keep = sorted(keep, key=lambda s: -n_by_scheme[s])
    dims = ["acceptability", "relevance", "grounds"]
    f_pct = f_pct.reindex(index=keep, columns=dims, fill_value=0)

    SCALE_MIN, SCALE_MAX = 1, 3
    RANGE = SCALE_MAX - SCALE_MIN
    samples = json.loads((run / "samples.json").read_text(encoding="utf-8"))
    acc = {s: {d: [] for d in dims} for s in keep}
    for d in samples.get("dilemmas", []):
        for tg in d.get("targets", []) or []:
            short_model = tg.get("model", "").split("/")[-1]
            for tk in tg.get("tracks", []) or []:
                track = tk.get("text_kind")
                if short_model == "gpt-5.4" and track == "reasoning":
                    continue
                p2 = (tk.get("phase2", []) or [None])[0]
                if not p2:
                    continue
                sch = (p2.get("scheme") or "").strip()
                if sch not in acc:
                    continue
                for p4 in tk.get("phase4", []) or []:
                    if p4.get("skipped"):
                        continue
                    loc = p4.get("local") or {}
                    for dim in dims:
                        vals = [dq.get(dim) for q, dq in loc.items()
                                if isinstance(dq, dict)
                                and isinstance(dq.get(dim), int)]
                        if vals:
                            cell_mean = sum(vals) / len(vals)
                            acc[sch][dim].append(
                                (cell_mean - SCALE_MIN) / RANGE
                            )

    pretty = {"grounds": "L.Grnd", "acceptability": "L.Acc",
              "relevance": "L.Rel"}
    out_rows = []
    for s in keep:
        row = {"scheme": s, "n_cells_under_scheme": int(n_by_scheme[s])}
        for d in dims:
            row[f"fail_{pretty[d]}_pct"] = round(float(f_pct.loc[s, d]), 2)
        for d in dims:
            xs = acc[s][d]
            row[f"score_{pretty[d]}"] = round(sum(xs) / len(xs), 3) if xs else 0.0
        out_rows.append(row)
    with (out / "local_failure_room_by_scheme.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)


# --------------------------------------------------------------------- #
# 14.  global_failure_room.csv
# --------------------------------------------------------------------- #

def export_global_failure_room(run: Path, out: Path) -> None:
    df = pd.read_csv(run / "failure_room_global_metrics.csv")
    metrics = ["coherence", "acceptability", "relevance", "sufficiency"]
    out_rows = []
    for m in metrics:
        f_row = df[(df["category"] == "failure") & (df["metric"] == m)]
        r_row = df[(df["category"] == "room") & (df["metric"] == m)]
        out_rows.append({
            "metric": m,
            "failure_pct": float(f_row["pct"].iloc[0]) if not f_row.empty else 0.0,
            "room_pct":    float(r_row["pct"].iloc[0]) if not r_row.empty else 0.0,
        })
    with (out / "global_failure_room.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)


# --------------------------------------------------------------------- #
# 15--16.  judge_disagreement_*.csv
# --------------------------------------------------------------------- #

def export_judge_disagreement_per_metric(run: Path, out: Path) -> None:
    df = pd.read_csv(run / "judge_disagreement_per_metric.csv")
    df = df[["metric", "n_compared", "n_disagree", "pct_disagree"]]
    df = df.sort_values("pct_disagree", ascending=False)
    df.to_csv(out / "judge_disagreement_per_metric.csv", index=False,
              float_format="%.3f")


def export_judge_disagreement_per_model(run: Path, out: Path) -> None:
    df = pd.read_csv(run / "judge_disagreement_per_model.csv")
    keep = ["model", "n_cells_with_both_judges",
            "n_cells_with_any_disagreement", "pct_cells_with_disagreement",
            "n_disagreement_points", "n_compared_points", "pct_points_disagree"]
    df = df[keep].copy()
    df = df.sort_values("pct_cells_with_disagreement", ascending=False)
    df["model"] = df["model"].map(lambda m: PROVIDER_PRETTY.get(m, m))
    df.to_csv(out / "judge_disagreement_per_model.csv", index=False,
              float_format="%.3f")


# --------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------- #

EXPORTERS = [
    export_failure_room_heatmap,
    export_model_score_heatmap,
    export_per_model_score_and_failure,
    export_per_model_int_ext,
    export_model_means_bar,
    export_per_model_failure_rates,
    export_per_model_rfi_rates,
    export_per_model_normalized_quality,
    export_per_model_scheme_consistency,
    export_per_model_metrics_heatmap,
    export_scheme_distribution,
    export_scheme_transitions_heatmap,
    export_scores_per_scheme,
    export_local_failure_room_by_scheme,
    export_global_failure_room,
    export_judge_disagreement_per_metric,
    export_judge_disagreement_per_model,
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for fn in EXPORTERS:
        try:
            fn(args.run_dir, args.out_dir)
            print(f"ok: {fn.__name__}")
        except Exception as e:
            print(f"FAIL: {fn.__name__}: {e}")


if __name__ == "__main__":
    main()
