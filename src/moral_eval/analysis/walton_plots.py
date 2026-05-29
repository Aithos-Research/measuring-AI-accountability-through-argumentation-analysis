"""Generate PNG visualisations alongside the walton_analysis CSVs.

Plots produced (saved next to ``samples.json`` in the cot directory):

- ``model_score_heatmap.png``         — model × ARG dimension heatmap (0-2)
- ``model_means_bar.png``             — model mean ± stderr (bar with errbars)
- ``failure_room_heatmap.png``        — model × {failure, room} × scope %
- ``scheme_distribution.png``         — model × scheme stacked bar by track
- ``scheme_transitions_heatmap.png``  — reasoning→justification matrix
- ``scores_per_scheme.png``           — mean Phase-4 score per scheme (with n)
- ``local_failure_room_by_scheme.png`` — scheme × Govier-dim failure & room counts
- ``global_failure_room.png``         — failure + room counts per global metric

The module is callable from inside ``analyze_run`` (called automatically at
trial end) and from the CLI for ad-hoc replotting::

    python -m moral_eval.analysis.walton_plots logs/.../<ts>/cot
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# Non-interactive backend so headless / Windows-service runs don't try to pop
# windows.  Must be set before importing pyplot.
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import ListedColormap, LogNorm  # noqa: E402

logger = logging.getLogger(__name__)

# Consistent palette for heatmaps where higher = better
_DIVERGING_GOOD = "RdYlGn"
_DIVERGING_BAD = "Reds"   # higher = worse (failure / room rates)
_SEQUENTIAL = "viridis"


def _infer_scale(df: pd.DataFrame, cols: list[str]) -> tuple[float, float]:
    """Return (min, max) inclusive bounds spanning the score columns,
    rounded out to the nearest integer.  Used to keep heatmap colorbars and
    bar-chart axes correct when the underlying Phase-4 scale changes from
    0-2 to 1-3 or beyond."""
    sub = df[[c for c in cols if c in df.columns]]
    if sub.empty:
        return 0.0, 2.0
    vmin = float(np.floor(np.nanmin(sub.values)))
    vmax = float(np.ceil(np.nanmax(sub.values)))
    # Defensive — avoid degenerate range
    if vmax - vmin < 0.5:
        return vmin, vmin + 2
    return vmin, vmax


def _safe_read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------

def _plot_model_score_heatmap(cot_dir: Path) -> Path | None:
    df = _safe_read(cot_dir / "scores_per_model.csv")
    if df is None:
        return None
    score_cols = [
        "local_acceptability", "local_relevance", "local_grounds",
        "global_coherence", "global_acceptability", "global_relevance", "global_sufficiency",
    ]
    cols_present = [c for c in score_cols if c in df.columns]
    if not cols_present:
        return None
    # CoT-panel models (visible pre-<answer> prose elicitation) are grouped
    # below the reasoning-branch panel so the two panels are not visually
    # interleaved; the dashed separator marks the split.
    cot_models = {"mistral-large-2512", "Llama 3.3", "llama-3.3", "llama-3p3"}
    is_cot = df["model"].isin(cot_models)
    df_reason = df[~is_cot].sort_values("mean", ascending=False)
    df_cot = df[is_cot].sort_values("mean", ascending=False)
    df = pd.concat([df_reason, df_cot], ignore_index=True)
    split_index = len(df_reason)  # the row index just above the CoT block
    data = df[cols_present].copy()
    data.index = df["model"]
    short_cols = [c.replace("local_", "L.").replace("global_", "G.").replace("acceptability", "Acc")
                  .replace("relevance", "Rel").replace("grounds", "Grnd")
                  .replace("coherence", "Coh").replace("sufficiency", "Suf")
                  for c in cols_present]
    vmin, vmax = _infer_scale(df, cols_present)
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(df) + 1.2))
    sns.heatmap(
        data.values, annot=True, fmt=".2f", cmap=_DIVERGING_GOOD,
        vmin=vmin, vmax=vmax,
        cbar_kws={"label": f"Mean score ({int(vmin)}–{int(vmax)})"},
        xticklabels=short_cols, yticklabels=data.index, ax=ax,
        linewidths=0.4, linecolor="white",
    )
    # Panel separator: a dashed black horizontal line between the
    # reasoning-branch block and the CoT block.
    if 0 < split_index < len(df):
        ax.axhline(split_index, color="black", linestyle="--",
                   linewidth=1.0, alpha=0.8)
    ax.set_title(f"Mean score per model × Govier dimension\n(L. = local per-question, G. = global; scale {int(vmin)}–{int(vmax)})")
    ax.set_ylabel("")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = cot_dir / "model_score_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_model_means_bar(cot_dir: Path) -> Path | None:
    df = _safe_read(cot_dir / "scores_per_model.csv")
    if df is None:
        return None
    df = df.sort_values("mean", ascending=True)
    vmin, vmax = _infer_scale(df, ["mean"])
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(df) + 1.2))
    ax.barh(df["model"], df["mean"], xerr=df["stderr"],
            color="#4477AA", edgecolor="black", linewidth=0.5, capsize=4)
    for i, (m, v) in enumerate(zip(df["model"], df["mean"])):
        ax.text(v + 0.02 * vmax, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(vmin, vmax * 1.05)
    ax.set_xlabel(f"Mean Phase-4 score ({int(vmin)}–{int(vmax)})")
    ax.set_title("Per-model mean ± stderr")
    ax.axvline(vmax, color="grey", linestyle=":", linewidth=0.8)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = cot_dir / "model_means_bar.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_per_model_failures_grouped(cot_dir: Path) -> Path | None:
    """Grouped horizontal bar chart of per-model failure rates: any / local /
    global side by side, ordered by overall_failure_rate_pct.
    """
    df = _safe_read(cot_dir / "per_model_metrics.csv")
    if df is None or df.empty:
        return None
    df = df.sort_values("overall_failure_rate_pct", ascending=True)
    metrics = [("overall_failure_rate_pct", "Any"),
               ("local_failure_rate_pct", "Local"),
               ("global_failure_rate_pct", "Global")]
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(df) + 1.5))
    y = np.arange(len(df))
    height = 0.27
    colors = ["#882255", "#cc6677", "#ddaa99"]
    for i, (col, label) in enumerate(metrics):
        offset = (i - 1) * height
        bars = ax.barh(y + offset, df[col], height=height, color=colors[i],
                       edgecolor="black", linewidth=0.4, label=label)
        for bar, v in zip(bars, df[col]):
            if v >= 0.5:
                ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height()/2,
                        f"{v:.1f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["model"])
    ax.set_xlabel("% of arguments with ≥1 failure (score = minimum)")
    ax.set_title("Per-model failure rates (any / local / global)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = cot_dir / "per_model_failure_rates.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_per_model_rfi_grouped(cot_dir: Path) -> Path | None:
    """Grouped horizontal bar chart of per-model RFI rates."""
    df = _safe_read(cot_dir / "per_model_metrics.csv")
    if df is None or df.empty:
        return None
    df = df.sort_values("overall_rfi_rate_pct", ascending=True)
    metrics = [("overall_rfi_rate_pct", "Any"),
               ("local_rfi_rate_pct", "Local"),
               ("global_rfi_rate_pct", "Global")]
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(df) + 1.5))
    y = np.arange(len(df))
    height = 0.27
    colors = ["#117733", "#44aa99", "#88ccbb"]
    for i, (col, label) in enumerate(metrics):
        offset = (i - 1) * height
        bars = ax.barh(y + offset, df[col], height=height, color=colors[i],
                       edgecolor="black", linewidth=0.4, label=label)
        for bar, v in zip(bars, df[col]):
            if v >= 1:
                ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height()/2,
                        f"{v:.1f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["model"])
    ax.set_xlim(0, 105)
    ax.set_xlabel("% of arguments with ≥1 score below maximum (room-for-improvement)")
    ax.set_title("Per-model RFI rates (any / local / global)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = cot_dir / "per_model_rfi_rates.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_per_model_normalized_quality(cot_dir: Path) -> Path | None:
    """Grouped bar chart of normalized local vs global quality (0-1)."""
    df = _safe_read(cot_dir / "per_model_metrics.csv")
    if df is None or df.empty:
        return None
    # Sort by combined quality so the strongest model is on top
    df = df.assign(_combined=df[["mean_normalized_local_quality", "mean_normalized_global_quality"]].mean(axis=1))
    df = df.sort_values("_combined", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(df) + 1.5))
    y = np.arange(len(df))
    height = 0.38
    bars_l = ax.barh(y - height/2, df["mean_normalized_local_quality"], height=height,
                     color="#4477AA", edgecolor="black", linewidth=0.4, label="Local")
    bars_g = ax.barh(y + height/2, df["mean_normalized_global_quality"], height=height,
                     color="#EE6677", edgecolor="black", linewidth=0.4, label="Global")
    for bars, vals in ((bars_l, df["mean_normalized_local_quality"]),
                       (bars_g, df["mean_normalized_global_quality"])):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{v:.3f}", va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(df["model"])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Mean normalized quality (0 = minimum, 1 = maximum)")
    ax.set_title("Per-model mean normalized argument quality")
    ax.axvline(1.0, color="grey", linestyle=":", linewidth=0.8)
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = cot_dir / "per_model_normalized_quality.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_per_model_scheme_consistency(cot_dir: Path) -> Path | None:
    """Horizontal bar chart of reasoning↔justification scheme consistency %."""
    df = _safe_read(cot_dir / "per_model_metrics.csv")
    if df is None or df.empty:
        return None
    df = df.sort_values("scheme_consistency_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(df) + 1.2))
    bars = ax.barh(df["model"], df["scheme_consistency_pct"],
                   color="#332288", edgecolor="black", linewidth=0.5)
    for bar, v, n in zip(bars, df["scheme_consistency_pct"], df["n_dilemmas_with_both_tracks"]):
        ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height()/2,
                f"{v:.1f}%  (n={int(n)})", va="center", fontsize=9)
    ax.set_xlim(0, 105)
    ax.set_xlabel("% of dilemmas where reasoning and justification used the same Walton scheme")
    ax.set_title("Per-model reasoning↔justification scheme consistency")
    ax.axvline(100, color="grey", linestyle=":", linewidth=0.8)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = cot_dir / "per_model_scheme_consistency.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_per_model_metrics(cot_dir: Path) -> Path | None:
    """Per-model metric heatmap: scheme consistency, failure/RFI rates,
    normalised quality.  Two-panel layout — '% rates' (failure/RFI/consistency)
    on the left, '0-1 normalised quality' on the right — so the two scales
    don't fight on one colour bar."""
    df = _safe_read(cot_dir / "per_model_metrics.csv")
    if df is None or df.empty:
        return None
    df = df.sort_values("mean_normalized_local_quality", ascending=False)

    pct_cols = [
        "scheme_consistency_pct",
        "overall_failure_rate_pct", "local_failure_rate_pct", "global_failure_rate_pct",
        "overall_rfi_rate_pct", "local_rfi_rate_pct", "global_rfi_rate_pct",
    ]
    qual_cols = ["mean_normalized_local_quality", "mean_normalized_global_quality"]
    pct_short = ["scheme\nconsist.", "fail\nany", "fail\nlocal", "fail\nglobal",
                 "RFI\nany", "RFI\nlocal", "RFI\nglobal"]
    qual_short = ["norm.\nlocal", "norm.\nglobal"]

    pct_data = df[pct_cols].copy()
    qual_data = df[qual_cols].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 0.55 * len(df) + 1.5),
                             gridspec_kw={"width_ratios": [7, 2]})

    sns.heatmap(
        pct_data.values, annot=True, fmt=".1f", cmap="RdYlGn_r",
        vmin=0, vmax=100, cbar_kws={"label": "% of cells"},
        xticklabels=pct_short, yticklabels=df["model"], ax=axes[0],
        linewidths=0.4, linecolor="white",
    )
    axes[0].set_title("Per-model rates (%)")
    axes[0].set_ylabel("")

    sns.heatmap(
        qual_data.values, annot=True, fmt=".3f", cmap="RdYlGn",
        vmin=0, vmax=1, cbar_kws={"label": "Mean score / 2"},
        xticklabels=qual_short, yticklabels=False, ax=axes[1],
        linewidths=0.4, linecolor="white",
    )
    axes[1].set_title("Normalised quality (0–1)")

    # Dynamic subtitle — read n from n_dilemmas_with_both_tracks if present
    if "n_dilemmas_with_both_tracks" in df.columns and not df.empty:
        n_d = int(df["n_dilemmas_with_both_tracks"].max())
        fig.suptitle(f"Per-model summary (n={n_d} dilemmas)")
    else:
        fig.suptitle("Per-model summary")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = cot_dir / "per_model_metrics_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_failure_room_heatmap(cot_dir: Path) -> Path | None:
    df = _safe_read(cot_dir / "failure_room_summary.csv")
    if df is None:
        return None
    # Extract per-model rows: category like "failure:<model>" / "room:<model>"
    df_pm = df[df["category"].str.contains(":", na=False)].copy()
    if df_pm.empty:
        return None
    df_pm[["kind", "model"]] = df_pm["category"].str.split(":", n=1, expand=True)
    # Pivot to rows=model, cols=(kind, scope)
    pivot = df_pm.pivot_table(index="model", columns=["kind", "scope"], values="pct", aggfunc="first")
    # Reorder columns
    desired = [("failure", "any"), ("failure", "local"), ("failure", "global"),
               ("room", "any"), ("room", "local"), ("room", "global")]
    cols_present = [c for c in desired if c in pivot.columns]
    if not cols_present:
        return None
    pivot = pivot[cols_present]
    # Sort models by overall failure rate desc for readability
    sort_key = pivot.get(("failure", "any"))
    if sort_key is not None:
        pivot = pivot.sort_values(("failure", "any"), ascending=True)
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(pivot) + 1.5))
    sns.heatmap(
        pivot.values, annot=True, fmt=".1f", cmap=_DIVERGING_BAD,
        vmin=0, vmax=max(50.0, np.nanmax(pivot.values) if pivot.size else 50.0),
        cbar_kws={"label": "% of cells"},
        xticklabels=[f"{k}\n{s}" for k, s in pivot.columns],
        yticklabels=pivot.index, ax=ax, linewidths=0.4, linecolor="white",
    )
    ax.set_title("Per-model failure (≥1 zero) and room-for-improvement (≥1 one) rates")
    ax.set_ylabel("")
    plt.tight_layout()
    out = cot_dir / "failure_room_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_scheme_distribution(cot_dir: Path) -> Path | None:
    df = _safe_read(cot_dir / "scheme_distribution_per_model.csv")
    if df is None:
        return None
    df = df[df["model"] != "__ALL__"].copy()
    if df.empty:
        return None
    # Aggregate to long form: model, scheme, track, count
    long = pd.concat([
        df.assign(track="reasoning", count=df["reasoning"]),
        df.assign(track="justification", count=df["justification"]),
    ])[["model", "scheme", "track", "count"]]
    long = long[long["count"] > 0]
    # Limit to the top-N schemes by total and roll the long tail into "Other"
    # so every per-model row in the stacked bar sums to the same dilemma count.
    top_n = 8
    top_schemes = (long.groupby("scheme")["count"].sum().sort_values(ascending=False).head(top_n).index.tolist())
    long.loc[~long["scheme"].isin(top_schemes), "scheme"] = "Other"
    cols = list(top_schemes) + ["Other"]

    # Pivot to model x scheme per track and stack-plot the two tracks side-by-side
    pivot_r = long[long["track"] == "reasoning"].pivot_table(index="model", columns="scheme", values="count", aggfunc="sum", fill_value=0)
    pivot_j = long[long["track"] == "justification"].pivot_table(index="model", columns="scheme", values="count", aggfunc="sum", fill_value=0)
    # Align columns + rows
    models = sorted(set(pivot_r.index) | set(pivot_j.index))
    pivot_r = pivot_r.reindex(index=models, columns=cols, fill_value=0)
    pivot_j = pivot_j.reindex(index=models, columns=cols, fill_value=0)

    # Cap = max per-model count = number of dilemmas (each model classifies
    # once per dilemma per track).  Inferred from the row sums so the chart
    # auto-adjusts to whatever batch size the run was: ~25, ~100, ~200, ...
    cap = int(max(pivot_r.sum(axis=1).max(), pivot_j.sum(axis=1).max(), 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.5 * len(models) + 1.5), sharey=True)
    for ax, pivot, title in ((axes[0], pivot_r, "Reasoning track"), (axes[1], pivot_j, "Justification track")):
        bottoms = np.zeros(len(pivot))
        cmap = plt.get_cmap("tab10")
        for i, sch in enumerate(cols):
            color = "#bdbdbd" if sch == "Other" else cmap(i % 10)
            vals = pivot[sch].values
            ax.barh(pivot.index, vals, left=bottoms, color=color,
                    label=sch if ax is axes[0] else None, edgecolor="white", linewidth=0.4)
            bottoms += vals
        ax.set_xlabel(f"Count (max {cap} per model)")
        ax.set_title(title)
        ax.set_xlim(0, cap)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.legend(loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Scheme distribution per model, by track (top-{top_n} schemes + Other; n={cap} dilemmas)")
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    out = cot_dir / "scheme_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_scheme_transitions_heatmap(cot_dir: Path) -> Path | None:
    """Reasoning -> Justification transition table (raw counts).

    Each cell is the number of (model, dilemma) pairs whose reasoning track
    used the row scheme and whose justification track used the column
    scheme.  Row and column marginals (n=...) are printed on the axis
    labels so the reader can read the directional shift for each scheme
    directly (row n vs.\ column n).
    """
    df = _safe_read(cot_dir / "scheme_transitions.csv")
    if df is None or df.empty:
        return None

    # Cover all schemes that appear on either side
    all_schemes = pd.concat([df["reasoning_scheme"], df["justification_scheme"]]).value_counts().index
    top_schemes = list(all_schemes[:8])
    df_top = df[df["reasoning_scheme"].isin(top_schemes) & df["justification_scheme"].isin(top_schemes)]
    pivot = df_top.pivot_table(
        index="reasoning_scheme", columns="justification_scheme",
        values="count", aggfunc="sum", fill_value=0,
    )
    pivot = pivot.reindex(index=top_schemes, columns=top_schemes, fill_value=0)

    row_totals = pivot.sum(axis=1).astype(int)
    col_totals = pivot.sum(axis=0).astype(int)
    counts = pivot.values.astype(int)
    annot = np.where(counts > 0, np.vectorize(lambda v: f"{v}")(counts), "")

    # Colour by log10(count) so the VBPR->VBPR diagonal (~1000) doesn't
    # flatten the rest of the gradient.  Zero-count cells are rendered as
    # the bottom of the viridis ramp (dark purple) — the absent annotation
    # already conveys "no transitions here", so the colour doesn't need to.
    colour_data = counts.astype(float)
    masked = np.ma.masked_where(colour_data <= 0, colour_data)
    cmap = plt.get_cmap(_SEQUENTIAL).copy()
    cmap.set_bad(color=cmap(0.0))

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(
        masked, annot=annot, fmt="",
        cmap=cmap,
        norm=LogNorm(vmin=1, vmax=max(counts.max(), 1)),
        cbar_kws={"label": "count (log scale)"},
        xticklabels=[f"{s}\n(n={n})" for s, n in zip(pivot.columns, col_totals)],
        yticklabels=[f"{s}  (n={n})" for s, n in zip(pivot.index, row_totals)],
        ax=ax, linewidths=0.4, linecolor="lightgrey",
    )
    ax.set_xlabel("Justification-track scheme  (n = times used in justification)")
    ax.set_ylabel("Reasoning-track scheme  (n = times used in reasoning)")
    ax.set_title(
        "Reasoning -> Justification scheme transitions (raw counts; colour = log10 count)"
    )
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    out = cot_dir / "scheme_transitions_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_scores_per_scheme(cot_dir: Path) -> Path | None:
    df = _safe_read(cot_dir / "scores_per_scheme.csv")
    if df is None:
        return None
    df = df.dropna(subset=["mean_overall"]).copy()
    df = df.sort_values("mean_overall", ascending=True)
    vmin, vmax = _infer_scale(df, ["mean_overall"])
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(df) + 1.5))
    bars = ax.barh(df["scheme"], df["mean_overall"], color="#88CCAA",
                   edgecolor="black", linewidth=0.5)
    for bar, n, m in zip(bars, df["n_total"], df["mean_overall"]):
        ax.text(bar.get_width() + 0.02 * vmax, bar.get_y() + bar.get_height()/2,
                f"{m:.2f}  (n={n})", va="center", fontsize=9)
    ax.set_xlim(vmin, vmax * 1.08)
    ax.set_xlabel(f"Mean Phase-4 score ({int(vmin)}–{int(vmax)})")
    ax.set_title("Per-scheme mean defense score (Phase-4)")
    ax.axvline(vmax, color="grey", linestyle=":", linewidth=0.8)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = cot_dir / "scores_per_scheme.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_local_by_scheme_and_dim(cot_dir: Path) -> Path | None:
    """Two-panel heatmap of failure / room rates per (scheme × Govier dim).

    Values are **% of arguments under that scheme** that had at least one
    failure (or room) on the dimension — not raw zero counts, which would
    just track how many cells got classified under each scheme.  The row
    labels include the n of cells under each scheme so the reader can tell
    when a row is a robust signal versus a thin one.
    """
    df = _safe_read(cot_dir / "per_argument_failure_room_by_scheme.csv")
    if df is None or df.empty:
        return None
    local = df[df["level"] == "local"].copy()
    if local.empty:
        return None
    # Pivot to (scheme x dimension) for failure and room rates
    f_pct = local.pivot_table(index="scheme", columns="dimension",
                              values="pct_cells_with_failure", aggfunc="first", fill_value=0)
    r_pct = local.pivot_table(index="scheme", columns="dimension",
                              values="pct_cells_with_room", aggfunc="first", fill_value=0)
    n_by_scheme = (local.groupby("scheme")["n_cells_under_scheme"]
                   .first().astype(int))
    # Drop dead rows (n=0)
    keep_schemes = [s for s in n_by_scheme.index if n_by_scheme[s] > 0]
    keep_schemes = sorted(keep_schemes, key=lambda s: -n_by_scheme[s])
    metrics_order = ["grounds", "acceptability", "relevance"]
    f_pct = f_pct.reindex(index=keep_schemes, columns=metrics_order, fill_value=0)
    r_pct = r_pct.reindex(index=keep_schemes, columns=metrics_order, fill_value=0)

    row_labels = [f"{s}  (n={n_by_scheme[s]})" for s in keep_schemes]

    fig, axes = plt.subplots(1, 2, figsize=(13, 0.5 * len(keep_schemes) + 2),
                             gridspec_kw={"width_ratios": [1, 1]})

    def annot_pct(vals):
        out = np.empty_like(vals, dtype=object)
        for i, row in enumerate(vals):
            for j, v in enumerate(row):
                out[i, j] = f"{v:.0f}%" if v >= 0.5 else ""
        return out

    fail_vmax = max(50.0, float(np.nanmax(f_pct.values)) if f_pct.size else 50.0)
    room_vmax = max(50.0, float(np.nanmax(r_pct.values)) if r_pct.size else 50.0)

    sns.heatmap(f_pct.values, annot=annot_pct(f_pct.values), fmt="",
                cmap=_DIVERGING_BAD, vmin=0, vmax=fail_vmax,
                xticklabels=metrics_order, yticklabels=row_labels, ax=axes[0],
                linewidths=0.4, linecolor="white",
                cbar_kws={"label": "% of arguments with ≥1 failure on dim"})
    axes[0].set_title("Local failures (score = 1)", fontsize=10)
    axes[0].set_ylabel("Walton scheme  (n = cells classified under scheme)")

    sns.heatmap(r_pct.values, annot=annot_pct(r_pct.values), fmt="",
                cmap="YlOrBr", vmin=0, vmax=room_vmax,
                xticklabels=metrics_order, yticklabels=False, ax=axes[1],
                linewidths=0.4, linecolor="white",
                cbar_kws={"label": "% of arguments with ≥1 dim partially met"})
    axes[1].set_title("Local room for improvement (score = 2)", fontsize=10)

    fig.suptitle("Per-scheme Govier-criterion weaknesses — normalised by n cells under scheme")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = cot_dir / "local_failure_room_by_scheme.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_global_failure_room(cot_dir: Path) -> Path | None:
    df = _safe_read(cot_dir / "failure_room_global_metrics.csv")
    if df is None:
        return None
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    metrics = ["coherence", "acceptability", "relevance", "sufficiency"]
    x = np.arange(len(metrics))
    width = 0.4
    fail = [df[(df.category == "failure") & (df.metric == m)]["pct"].iloc[0]
            if not df[(df.category == "failure") & (df.metric == m)].empty else 0 for m in metrics]
    room = [df[(df.category == "room") & (df.metric == m)]["pct"].iloc[0]
            if not df[(df.category == "room") & (df.metric == m)].empty else 0 for m in metrics]
    ax.bar(x - width/2, fail, width, color="#cc4444", label="failure (≥1 cell scored 0)")
    ax.bar(x + width/2, room, width, color="#cccc44", label="room (≥1 cell scored 1)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("% of Phase-4 cells")
    ax.set_title("Per global metric: failure rate vs room-for-improvement rate")
    ax.set_ylim(0, max(100, max(fail + room) * 1.1))
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for i, v in enumerate(fail):
        ax.text(i - width/2, v + 1, f"{v:.1f}", ha="center", fontsize=8)
    for i, v in enumerate(room):
        ax.text(i + width/2, v + 1, f"{v:.1f}", ha="center", fontsize=8)
    plt.tight_layout()
    out = cot_dir / "global_failure_room.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------

def plot_run(cot_dir: Path) -> list[Path]:
    cot_dir = Path(cot_dir)
    plotters = [
        _plot_model_score_heatmap,
        _plot_model_means_bar,
        _plot_per_model_metrics,
        _plot_per_model_failures_grouped,
        _plot_per_model_rfi_grouped,
        _plot_per_model_normalized_quality,
        _plot_per_model_scheme_consistency,
        _plot_failure_room_heatmap,
        _plot_scheme_distribution,
        _plot_scheme_transitions_heatmap,
        _plot_scores_per_scheme,
        _plot_local_by_scheme_and_dim,
        _plot_global_failure_room,
    ]
    written: list[Path] = []
    for fn in plotters:
        try:
            out = fn(cot_dir)
        except Exception as exc:
            logger.warning("Plot %s failed: %s", fn.__name__, exc)
            out = None
        if out is not None:
            written.append(out)
    return written


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cot_dir", type=Path,
                   help="cot directory containing the walton_analysis CSVs")
    args = p.parse_args(argv)
    out = plot_run(args.cot_dir)
    for fp in out:
        print(f"wrote: {fp}")
    print(f"({len(out)} plots written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
