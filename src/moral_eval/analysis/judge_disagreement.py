"""Inter-judge disagreement analysis for a combined samples.json.

Disagreement criterion (per the paper's definition):
  For each comparable score (a local critical-question × dimension entry, or
  a global metric entry) where both Phase-4 judges produced a number,
  the judges disagree when one assigned the minimum failure score (1 on the
  1-3 scale) and the other did not.  Symmetric in both directions.

This produces:

- ``judge_disagreement_per_metric.csv``  — counts and % per dimension
- ``judge_disagreement_per_model.csv``   — % of cells with ≥1 disagreement per target
- ``judge_disagreement_per_scheme.csv``  — same shape, grouped by Walton scheme
- ``judge_disagreement_pairs.csv``       — one row per disagreeing data-point
                                           for ad-hoc inspection
- ``judge_disagreement_summary.csv``     — aggregate counts
- ``judge_disagreement_per_model.png``   — bar chart
- ``judge_disagreement_per_metric.png``  — bar chart

Usage::

    python -m moral_eval.analysis.judge_disagreement \\
        logs/deliberative_justification_v2/20260520_combined/cot
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# 1 = minimum on the current 1-3 scale.  If the scoring scale ever changes
# back to 0-2 we can flip this; auto-detect would be tidier but the user's
# definition is explicit ("score of one").
_FAIL = 1
_LOCAL_DIMS = ("acceptability", "relevance", "grounds")
_GLOBAL_DIMS = ("coherence", "acceptability", "relevance", "sufficiency")


def _safe_int(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    return None


def _walk_pairs(samples: dict):
    """Yield one pair-record per (target, dilemma, track, judge_A, judge_B).
    Each track is expected to contain >=2 phase4 entries with different
    phase4_judge aliases; we pair them lexicographically so (j1, j2) is the
    same pair as (j2, j1)."""
    for dil in samples["dilemmas"]:
        did = dil["dilemma_id"]
        for tg in dil["targets"]:
            model = tg["model"]
            for tk in tg["tracks"]:
                track = tk.get("text_kind", "?")
                # Phase-2 entries (1 per j2 judge).  We need the scheme for
                # this track — use the only/first one.
                p2_entries = tk.get("phase2") or []
                scheme = "(unspecified)"
                if p2_entries:
                    scheme = (p2_entries[0].get("scheme") or "").strip() or "(unspecified)"
                p4s = [p for p in (tk.get("phase4") or [])
                       if not p.get("skipped")]
                if len(p4s) < 2:
                    continue
                # Pair every unique combination
                aliases = sorted({p.get("phase4_judge", "?") for p in p4s})
                by_alias = {p.get("phase4_judge"): p for p in p4s}
                for i, a in enumerate(aliases):
                    for b in aliases[i + 1:]:
                        yield {
                            "dilemma_id": did,
                            "model": model.split("/")[-1],
                            "track": track,
                            "scheme": scheme,
                            "judge_a": a,
                            "judge_b": b,
                            "p4_a": by_alias[a],
                            "p4_b": by_alias[b],
                        }


def _enumerate_metrics(p4_a: dict, p4_b: dict):
    """For each comparable metric in this (p4_a, p4_b) pair, yield
    (kind, key, value_a, value_b) where kind is 'local' or 'global'.

    'local' keys look like ``q1.grounds`` etc.  We only compare points where
    both judges produced an integer value."""
    # Local: union of question keys
    la = p4_a.get("local") or {}
    lb = p4_b.get("local") or {}
    qs = sorted(set(la.keys()) | set(lb.keys()))
    for q in qs:
        dims_a = la.get(q) or {}
        dims_b = lb.get(q) or {}
        if not isinstance(dims_a, dict) or not isinstance(dims_b, dict):
            continue
        for d in _LOCAL_DIMS:
            va, vb = _safe_int(dims_a.get(d)), _safe_int(dims_b.get(d))
            if va is None or vb is None:
                continue
            yield ("local", f"{q}.{d}", va, vb)
    # Global
    ga = p4_a.get("global") or {}
    gb = p4_b.get("global") or {}
    for d in _GLOBAL_DIMS:
        va, vb = _safe_int(ga.get(d)), _safe_int(gb.get(d))
        if va is None or vb is None:
            continue
        yield ("global", d, va, vb)


def analyse(cot_dir: Path) -> dict:
    cot_dir = Path(cot_dir)
    samples = json.loads((cot_dir / "samples.json").read_text(encoding="utf-8"))

    # Accumulators
    n_compared = 0
    n_disagree = 0
    by_metric_total: Counter = Counter()
    by_metric_disagree: Counter = Counter()
    by_model_cells: Counter = Counter()
    by_model_cells_any_disagree: Counter = Counter()
    by_model_disagreements: Counter = Counter()
    by_model_compared: Counter = Counter()
    by_scheme_cells: Counter = Counter()
    by_scheme_cells_any_disagree: Counter = Counter()
    by_scheme_disagreements: Counter = Counter()
    by_scheme_compared: Counter = Counter()
    pair_rows = []

    for pair in _walk_pairs(samples):
        any_disagree = False
        cell_compared = 0
        cell_disagree = 0
        for kind, key, va, vb in _enumerate_metrics(pair["p4_a"], pair["p4_b"]):
            metric_label = f"{kind}.{key}"
            by_metric_total[metric_label] += 1
            n_compared += 1
            cell_compared += 1
            # The disagreement test: one judge scored the minimum, the other did not
            disagree = (va == _FAIL) != (vb == _FAIL)
            if disagree:
                n_disagree += 1
                by_metric_disagree[metric_label] += 1
                any_disagree = True
                cell_disagree += 1
                pair_rows.append([
                    pair["dilemma_id"], pair["model"], pair["track"], pair["scheme"],
                    pair["judge_a"], pair["judge_b"], kind, key, va, vb,
                ])
        # Per-model + per-scheme aggregates
        m = pair["model"]
        sch = pair["scheme"]
        by_model_cells[m] += 1
        by_model_compared[m] += cell_compared
        by_model_disagreements[m] += cell_disagree
        if any_disagree:
            by_model_cells_any_disagree[m] += 1
        by_scheme_cells[sch] += 1
        by_scheme_compared[sch] += cell_compared
        by_scheme_disagreements[sch] += cell_disagree
        if any_disagree:
            by_scheme_cells_any_disagree[sch] += 1

    # -- write CSVs --
    def write(name, header, rows):
        (cot_dir / name).write_text(
            "\n".join([",".join(map(str, header))]
                      + [",".join(map(lambda v: str(v) if not isinstance(v, str)
                                      or "," not in v else f'"{v}"', r)) for r in rows]) + "\n",
            encoding="utf-8",
        )

    # 1) Headline summary
    write("judge_disagreement_summary.csv",
          ["category", "value"],
          [["n_points_compared", n_compared],
           ["n_disagreements", n_disagree],
           ["pct_disagreement_per_point", round(100 * n_disagree / n_compared, 3) if n_compared else 0.0]])

    # 2) Per metric — local + global flattened
    rows = []
    for label, n_total in by_metric_total.most_common():
        nd = by_metric_disagree.get(label, 0)
        rows.append([label, n_total, nd, round(100 * nd / n_total, 3) if n_total else 0.0])
    rows.sort(key=lambda r: -r[3])  # by % desc
    write("judge_disagreement_per_metric.csv",
          ["metric", "n_compared", "n_disagree", "pct_disagree"], rows)

    # 3) Per model
    rows = []
    for m in sorted(by_model_cells):
        n_cells = by_model_cells[m]
        n_any = by_model_cells_any_disagree[m]
        n_cmp = by_model_compared[m]
        n_dis = by_model_disagreements[m]
        rows.append([
            m, n_cells,
            n_any, round(100 * n_any / n_cells, 3) if n_cells else 0.0,
            n_dis, n_cmp,
            round(100 * n_dis / n_cmp, 3) if n_cmp else 0.0,
        ])
    rows.sort(key=lambda r: -r[3])
    write("judge_disagreement_per_model.csv",
          ["model", "n_cells_with_both_judges",
           "n_cells_with_any_disagreement", "pct_cells_with_disagreement",
           "n_disagreement_points", "n_compared_points", "pct_points_disagree"], rows)

    # 4) Per Walton scheme
    rows = []
    for s in sorted(by_scheme_cells):
        n_cells = by_scheme_cells[s]
        n_any = by_scheme_cells_any_disagree[s]
        n_cmp = by_scheme_compared[s]
        n_dis = by_scheme_disagreements[s]
        rows.append([
            s, n_cells,
            n_any, round(100 * n_any / n_cells, 3) if n_cells else 0.0,
            n_dis, n_cmp,
            round(100 * n_dis / n_cmp, 3) if n_cmp else 0.0,
        ])
    rows.sort(key=lambda r: -r[6])
    write("judge_disagreement_per_scheme.csv",
          ["scheme", "n_cells", "n_cells_with_any_disagreement", "pct_cells_with_disagreement",
           "n_disagreement_points", "n_compared_points", "pct_points_disagree"], rows)

    # 5) Every disagreeing data-point
    write("judge_disagreement_pairs.csv",
          ["dilemma_id", "model", "track", "scheme",
           "judge_a", "judge_b", "kind", "key", "score_a", "score_b"],
          pair_rows)

    # -- plots --
    _plot_per_model(cot_dir)
    _plot_per_metric(cot_dir)

    summary = {
        "n_compared": n_compared,
        "n_disagreements": n_disagree,
        "pct_disagreement_per_point": 100 * n_disagree / n_compared if n_compared else 0,
        "n_pairs": sum(by_model_cells.values()),
        "n_cells_with_any_disagreement": sum(by_model_cells_any_disagree.values()),
    }
    return summary


def _safe_read(path: Path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _plot_per_model(cot_dir: Path):
    df = _safe_read(cot_dir / "judge_disagreement_per_model.csv")
    if df is None or df.empty:
        return
    df = df.sort_values("pct_cells_with_disagreement", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(df) + 1.5))
    bars = ax.barh(df["model"], df["pct_cells_with_disagreement"],
                   color="#cc6677", edgecolor="black", linewidth=0.5)
    for bar, v_cell, v_pt in zip(bars, df["pct_cells_with_disagreement"], df["pct_points_disagree"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{v_cell:.1f}% cells  •  {v_pt:.2f}% points",
                va="center", fontsize=9)
    ax.set_xlim(0, max(105.0, df["pct_cells_with_disagreement"].max() * 1.2))
    ax.set_xlabel("% of (target × dilemma × track) cells with ≥1 inter-judge disagreement")
    ax.set_title("Per-model inter-judge disagreement rate\n"
                 "(one judge scored 1 on a metric where the other did not)")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(cot_dir / "judge_disagreement_per_model.png", dpi=150)
    plt.close(fig)


def _plot_per_metric(cot_dir: Path):
    df = _safe_read(cot_dir / "judge_disagreement_per_metric.csv")
    if df is None or df.empty:
        return
    # Split into local vs global for readability
    df["kind"] = df["metric"].str.split(".").str[0]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [3, 1]})

    # Local: aggregate by dimension (drop q-index for the headline figure)
    local = df[df["kind"] == "local"].copy()
    if not local.empty:
        local["dim"] = local["metric"].str.split(".").str[-1]
        agg = local.groupby("dim").agg({"n_compared": "sum", "n_disagree": "sum"}).reset_index()
        agg["pct"] = 100 * agg["n_disagree"] / agg["n_compared"]
        agg = agg.sort_values("pct", ascending=True)
        axes[0].barh(agg["dim"], agg["pct"], color="#4477aa", edgecolor="black", linewidth=0.5)
        for y, (dim, n_cmp, n_dis, pct) in enumerate(zip(agg["dim"], agg["n_compared"],
                                                          agg["n_disagree"], agg["pct"])):
            axes[0].text(pct + 0.1, y, f"{pct:.2f}% ({int(n_dis)}/{int(n_cmp)})", va="center", fontsize=9)
        axes[0].set_xlabel("% of compared points where one judge says 1 and the other doesn't")
        axes[0].set_title("Local Govier dimensions (per-question scores aggregated)")
        axes[0].grid(axis="x", linestyle="--", alpha=0.4)

    # Global
    glob = df[df["kind"] == "global"].copy()
    if not glob.empty:
        glob["dim"] = glob["metric"].str.split(".").str[-1]
        glob = glob.sort_values("pct_disagree", ascending=True)
        axes[1].barh(glob["dim"], glob["pct_disagree"], color="#ee6677", edgecolor="black", linewidth=0.5)
        for y, (d, n_cmp, n_dis, pct) in enumerate(zip(glob["dim"], glob["n_compared"],
                                                         glob["n_disagree"], glob["pct_disagree"])):
            axes[1].text(pct + 0.2, y, f"{pct:.1f}%", va="center", fontsize=9)
        axes[1].set_xlabel("% disagree")
        axes[1].set_title("Global dimensions")
        axes[1].grid(axis="x", linestyle="--", alpha=0.4)

    fig.suptitle("Where the judges disagree (1 vs not-1, by Govier dimension)")
    plt.tight_layout()
    fig.savefig(cot_dir / "judge_disagreement_per_metric.png", dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cot_dir", type=Path,
                   help="cot directory containing the combined samples.json")
    args = p.parse_args(argv)
    summary = analyse(args.cot_dir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
