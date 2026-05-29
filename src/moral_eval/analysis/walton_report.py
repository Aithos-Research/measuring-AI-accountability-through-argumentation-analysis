"""Generate a single consolidated markdown report from the CSVs that
``walton_analysis`` writes.  Produces ``report.md`` next to samples.json
with every per-model and per-scheme table, all failure / RFI rankings,
and inline links to the PNG plots.

Callable from the CLI for ad-hoc rebuilds::

    python -m moral_eval.analysis.walton_report logs/.../<ts>/cot
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd


def _md_table(df: pd.DataFrame, formatters: dict | None = None) -> str:
    if df is None or df.empty:
        return "_(no data)_\n"
    df = df.copy()
    if formatters:
        for col, fn in formatters.items():
            if col in df.columns:
                df[col] = df[col].apply(fn)
    headers = "| " + " | ".join(str(c) for c in df.columns) + " |"
    align_row = "|" + "|".join(["---:" if df[c].dtype.kind in "fiub" else "---"
                                for c in df.columns]) + "|"
    rows = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([headers, align_row, *rows]) + "\n"


def _safe_read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def build_report(cot_dir: Path) -> Path:
    cot_dir = Path(cot_dir)
    out = cot_dir / "report.md"
    parts: list[str] = []

    # ---- header ----
    parts.append(f"# Walton Phase-4 evaluation report")
    parts.append(f"")
    parts.append(f"Source: `{cot_dir}`")
    parts.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    parts.append(f"")

    # ---- 1) Per-model metrics ----
    pmm = _safe_read(cot_dir / "per_model_metrics.csv")
    if pmm is not None:
        pmm = pmm.sort_values("mean_normalized_local_quality", ascending=False)
        parts.append("## 1. Per-model metrics")
        parts.append("")
        parts.append(
            "Failure = at least one score at the **minimum** (1 on the 1-3 scale).  "
            "RFI = at least one score **below maximum** (= 1 or 2).  "
            "Normalised quality = `(mean_score − 1) / 2` (0 = always at minimum, 1 = always at max)."
        )
        parts.append("")
        rename = {
            "model": "Model", "n_cells": "n",
            "scheme_consistency_pct": "Scheme cons %",
            "overall_failure_rate_pct": "Fail any %",
            "local_failure_rate_pct": "Fail local %",
            "global_failure_rate_pct": "Fail global %",
            "overall_rfi_rate_pct": "RFI any %",
            "local_rfi_rate_pct": "RFI local %",
            "global_rfi_rate_pct": "RFI global %",
            "mean_normalized_local_quality": "Norm local",
            "mean_normalized_global_quality": "Norm global",
        }
        cols_keep = [c for c in rename if c in pmm.columns]
        parts.append(_md_table(pmm[cols_keep].rename(columns=rename)))
        parts.append("![per-model heatmap](per_model_metrics_heatmap.png)")
        parts.append("![normalized quality](per_model_normalized_quality.png)")
        parts.append("![failure rates](per_model_failure_rates.png)")
        parts.append("![RFI rates](per_model_rfi_rates.png)")
        parts.append("![scheme consistency](per_model_scheme_consistency.png)")
        parts.append("")

    # ---- 2) Per-model scores breakdown ----
    spm = _safe_read(cot_dir / "scores_per_model.csv")
    if spm is not None:
        spm = spm.sort_values("mean", ascending=False)
        parts.append("## 2. Per-model score breakdown by Govier dimension")
        parts.append("")
        parts.append(
            "Mean Phase-4 score on the 1-3 scale (1 = not met, 2 = partially met, 3 = fully met).  "
            "Local scores are averaged across each cell's per-question trio; global is one score per cell."
        )
        parts.append("")
        rename = {
            "model": "Model", "mean": "Mean", "stderr": "StdErr",
            "mean_int": "Mean reasoning", "mean_ext": "Mean justif.",
            "r_int_available": "r_int avail",
            "local_acceptability": "L.Acc", "local_relevance": "L.Rel", "local_grounds": "L.Grnd",
            "global_coherence": "G.Coh", "global_acceptability": "G.Acc",
            "global_relevance": "G.Rel", "global_sufficiency": "G.Suf",
            "n_cells": "n",
        }
        cols_keep = [c for c in rename if c in spm.columns]
        parts.append(_md_table(spm[cols_keep].rename(columns=rename)))
        parts.append("![model score heatmap](model_score_heatmap.png)")
        parts.append("![model means](model_means_bar.png)")
        parts.append("")

    # ---- 3) Argumentation scheme distribution ----
    sdt = _safe_read(cot_dir / "scheme_distribution_by_track.csv")
    if sdt is not None:
        parts.append("## 3. Argumentation scheme distribution")
        parts.append("")
        parts.append("### Reasoning track")
        r = sdt[sdt["track"] == "reasoning"].sort_values("count", ascending=False)
        parts.append(_md_table(r[["scheme", "count", "pct_of_track"]]
                               .rename(columns={"scheme":"Scheme","count":"Count","pct_of_track":"% of track"})))
        parts.append("")
        parts.append("### Justification track")
        e = sdt[sdt["track"] == "justification"].sort_values("count", ascending=False)
        parts.append(_md_table(e[["scheme", "count", "pct_of_track"]]
                               .rename(columns={"scheme":"Scheme","count":"Count","pct_of_track":"% of track"})))
        parts.append("![scheme distribution per model](scheme_distribution.png)")
        parts.append("")

    trans = _safe_read(cot_dir / "scheme_transitions.csv")
    if trans is not None:
        parts.append("### Reasoning → Justification transitions (top 15)")
        parts.append("")
        top = trans.sort_values("count", ascending=False).head(15)
        parts.append(_md_table(top.rename(columns={
            "reasoning_scheme":"From (reasoning)", "justification_scheme":"To (justification)",
            "count":"Count","pct":"% of transitions","same":"Same"
        })))
        parts.append("![transition heatmap](scheme_transitions_heatmap.png)")
        parts.append("")

    # ---- 4) Failure distribution ----
    frs = _safe_read(cot_dir / "failure_room_summary.csv")
    if frs is not None:
        parts.append("## 4. Failure distribution")
        parts.append("")
        # Top-level (no colon in category)
        top = frs[~frs["category"].str.contains(":", na=False) & (frs["category"]=="failure")]
        parts.append("### Aggregate (across the full panel)")
        parts.append(_md_table(top.rename(columns={
            "category":"Category","scope":"Scope","n_cells":"n cells","n_total":"n total","pct":"%"
        })))
        parts.append("")
        # Per-model
        pm = frs[frs["category"].str.startswith("failure:", na=False)].copy()
        if not pm.empty:
            pm["model"] = pm["category"].str.replace("failure:", "", regex=False)
            wide = pm.pivot_table(index="model", columns="scope", values="pct", aggfunc="first")
            wide = wide.reindex(columns=[c for c in ("any","local","global") if c in wide.columns]).reset_index()
            parts.append("### Per-model failure rates (%)")
            parts.append(_md_table(wide.rename(columns={
                "model":"Model","any":"Any","local":"Local","global":"Global"
            })))
            parts.append("")

    glob_metrics = _safe_read(cot_dir / "failure_room_global_metrics.csv")
    if glob_metrics is not None:
        fail = glob_metrics[glob_metrics["category"] == "failure"]
        if not fail.empty:
            parts.append("### Global failures by metric (which dimension fails most often)")
            parts.append(_md_table(fail[["metric","n_cells","n_total","pct"]].rename(columns={
                "metric":"Global metric","n_cells":"n with score=1","n_total":"n total","pct":"%"
            })))
            parts.append("![global failure & room](global_failure_room.png)")
            parts.append("")

    fail_triples = _safe_read(cot_dir / "failures_local_by_scheme_q_metric.csv")
    if fail_triples is not None and not fail_triples.empty:
        top15 = fail_triples.head(15)
        parts.append("### Top local failure triples (scheme × question × metric)")
        parts.append(_md_table(top15.rename(columns={
            "scheme":"Scheme","question":"Question","metric":"Metric","n_zeros":"# cells score=1"
        })))
        parts.append("![local failure/RFI by scheme](local_failure_room_by_scheme.png)")
        parts.append("")

    # ---- 5) Room-for-improvement distribution ----
    if frs is not None:
        top = frs[~frs["category"].str.contains(":", na=False) & (frs["category"]=="room")]
        parts.append("## 5. Room-for-improvement distribution")
        parts.append("")
        parts.append("**Definition used in this section**: room = at least one score equal to *partial* (= 2 on the 1-3 scale).  "
                     "For the *broader* RFI rate (any score below maximum) see column 'RFI any %' in section 1.")
        parts.append("")
        parts.append("### Aggregate (across the full panel)")
        parts.append(_md_table(top.rename(columns={
            "category":"Category","scope":"Scope","n_cells":"n cells","n_total":"n total","pct":"%"
        })))
        parts.append("")
        pm = frs[frs["category"].str.startswith("room:", na=False)].copy()
        if not pm.empty:
            pm["model"] = pm["category"].str.replace("room:", "", regex=False)
            wide = pm.pivot_table(index="model", columns="scope", values="pct", aggfunc="first")
            wide = wide.reindex(columns=[c for c in ("any","local","global") if c in wide.columns]).reset_index()
            parts.append("### Per-model room rates (%)")
            parts.append(_md_table(wide.rename(columns={
                "model":"Model","any":"Any","local":"Local","global":"Global"
            })))
            parts.append("")

    if glob_metrics is not None:
        room = glob_metrics[glob_metrics["category"] == "room"]
        if not room.empty:
            parts.append("### Global room by metric (which dimension has partial score most often)")
            parts.append(_md_table(room[["metric","n_cells","n_total","pct"]].rename(columns={
                "metric":"Global metric","n_cells":"n with score=2","n_total":"n total","pct":"%"
            })))
            parts.append("")

    room_triples = _safe_read(cot_dir / "room_local_by_scheme_q_metric.csv")
    if room_triples is not None and not room_triples.empty:
        top15 = room_triples.head(15)
        parts.append("### Top local room triples (scheme × question × metric)")
        parts.append(_md_table(top15.rename(columns={
            "scheme":"Scheme","question":"Question","metric":"Metric","n_ones":"# cells score=2"
        })))
        parts.append("")

    # ---- 6) Per-scheme scores ----
    sps = _safe_read(cot_dir / "scores_per_scheme.csv")
    if sps is not None:
        sps = sps.sort_values("n_total", ascending=False)
        parts.append("## 6. Per-Walton-scheme scores")
        parts.append("")
        cols = [c for c in ["scheme","n_total","n_reasoning","n_justification",
                            "mean_overall","mean_overall_reasoning","mean_overall_justification",
                            "mean_local_overall","mean_global_overall"] if c in sps.columns]
        parts.append(_md_table(sps[cols].rename(columns={
            "scheme":"Scheme","n_total":"n","n_reasoning":"n reasoning","n_justification":"n justif.",
            "mean_overall":"Mean overall","mean_overall_reasoning":"Mean r","mean_overall_justification":"Mean j",
            "mean_local_overall":"Mean local","mean_global_overall":"Mean global"
        })))
        parts.append("![per-scheme scores](scores_per_scheme.png)")
        parts.append("")

    parts.append("---")
    parts.append("All CSVs and PNGs referenced above are in this directory.  Regenerate with:")
    parts.append("```")
    parts.append(f"python -m moral_eval.analysis.walton_analysis {cot_dir}")
    parts.append(f"python -m moral_eval.analysis.walton_plots {cot_dir}")
    parts.append(f"python -m moral_eval.analysis.walton_report {cot_dir}")
    parts.append("```")

    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cot_dir", type=Path,
                   help="cot directory containing walton_analysis CSVs.")
    args = p.parse_args(argv)
    out = build_report(args.cot_dir)
    print(f"wrote: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
