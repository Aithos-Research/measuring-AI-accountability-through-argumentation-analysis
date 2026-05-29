"""Post-run analysis for deliberative_justification_v2 samples.json.

Generates CSV summaries answering three questions about every Phase-4 cell
in a run:

1. **Scheme distribution** — which Walton schemes Sonnet (Phase-2 judge)
   assigned to each track, and the reasoning-to-justification transition
   pattern.
2. **Failure distribution** — rate of arguments with at least one 0 score
   (local critical question or global metric), broken down by scheme /
   question index / metric.
3. **Room-for-improvement distribution** — same shape but for 1 scores
   (where models are competent but not excellent).
4. **Per-model and per-scheme scores** — the headline numbers for the
   paper tables.

The module is callable from inside the dj_v2 ``run()`` (called automatically
at the end of every trial) and also from the CLI for ad-hoc re-analysis::

    python -m moral_eval.analysis.walton_analysis logs/deliberative_justification_v2/<run>/cot

All outputs are written next to ``samples.json``; nothing is mutated.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Output filenames (relative to the cot dir)
# ---------------------------------------------------------------------
_OUT_FILES = [
    "scheme_distribution_by_track.csv",
    "scheme_distribution_per_model.csv",
    "scheme_transitions.csv",
    "scores_per_model.csv",
    "per_model_metrics.csv",
    "scores_per_scheme.csv",
    "failure_room_summary.csv",
    "failure_room_global_metrics.csv",
    "failures_local_by_scheme_q_metric.csv",
    "room_local_by_scheme_q_metric.csv",
    "per_argument_failure_room_by_scheme.csv",
    "analysis_summary.txt",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _pct(n: int, total: int) -> float:
    return 100.0 * n / total if total else 0.0


def _avg(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _collect_cells(samples: Dict[str, Any]):
    """Walk samples.json and emit one cell per non-skipped Phase-4 entry,
    plus a parallel scheme-by-(dilemma,model,track) map for transitions."""
    cells = []
    scheme_by_dmt: Dict[tuple, str] = {}
    for dil in samples["dilemmas"]:
        for tg in dil["targets"]:
            model = tg["model"]
            short_model = model.split("/")[-1]
            for tk in tg["tracks"]:
                track = "reasoning" if tk["text_kind"] == "reasoning" else "justification"
                # Single Phase-2 judge in current configs; if there are more
                # we just record the first scheme for the transition map and
                # iterate over all when building cells.
                p2_list = tk.get("phase2") or []
                if p2_list:
                    first_p2 = p2_list[0]
                    sch_first = (first_p2.get("scheme") or "").strip() or "(unspecified)"
                    scheme_by_dmt[(dil["dilemma_id"], short_model, track)] = sch_first
                for p2 in p2_list:
                    if p2.get("skipped"):
                        continue
                    scheme = (p2.get("scheme") or "").strip() or "(unspecified)"
                    for p4 in tk.get("phase4") or []:
                        if p4.get("skipped"):
                            continue
                        if p4.get("phase2_judge") and p4.get("phase2_judge") != p2.get("judge"):
                            # Only count Phase-4 cells whose phase2_judge matches this p2 entry,
                            # in case future runs add multiple j2 judges.
                            continue
                        cells.append({
                            "dilemma": dil["dilemma_id"],
                            "model": short_model,
                            "full_model": model,
                            "track": track,
                            "phase4_judge": p4.get("phase4_judge"),
                            "scheme": scheme,
                            "local": p4.get("local") or {},
                            "global": p4.get("global") or {},
                        })
    return cells, scheme_by_dmt


def _local_overall(local_block):
    vals = []
    for _, dims in (local_block or {}).items():
        if not isinstance(dims, dict):
            continue
        nums = [v for v in dims.values() if isinstance(v, int)]
        if nums:
            vals.append(sum(nums) / len(nums))
    return _avg(vals) if vals else None


# Global metrics excluding coherence.  Per the 2026-05 setup change,
# coherence (Phase-4 global "do the local CQ answers cohere?") is reported
# as its own column and is NOT counted in global means or in failure / room
# rollups — the panel should not be penalised on a metric we treat as
# diagnostic rather than substantive.
_GLOBAL_DIMS_NO_COH = ("acceptability", "relevance", "sufficiency")


def _global_overall(g):
    vs = [v for k, v in (g or {}).items()
          if k in _GLOBAL_DIMS_NO_COH and isinstance(v, int)]
    return _avg(vs) if vs else None


def _detect_score_scale(cells) -> tuple[int, int]:
    """Find the min and max integer scores observed across all local + global
    score blocks.  The Phase-4 prompt's scale has been 0-2 in some prompt
    revisions and 1-3 in others; auto-detection means the analyser stays
    correct without manual config flips.
    """
    seen: set[int] = set()
    for c in cells:
        for q, dims in (c.get("local") or {}).items():
            if isinstance(dims, dict):
                for v in dims.values():
                    if isinstance(v, int):
                        seen.add(v)
        for v in (c.get("global") or {}).values():
            if isinstance(v, int):
                seen.add(v)
    if not seen:
        return 0, 2  # fallback
    smin, smax = min(seen), max(seen)
    # Defensive: if the observed range is degenerate (all same value), assume
    # a 1-3 scale (the current prompt's scale).
    if smax == smin:
        return 1, 3
    return smin, smax


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def analyze_run(cot_dir: Path) -> Dict[str, Any]:
    """Run the full analysis suite for a trial's cot directory.

    Reads ``cot_dir/samples.json`` (and optionally ``final_task_results.json``)
    and writes CSV + text summaries alongside.  Returns a dict of headline
    counts so the caller can log them.
    """
    cot_dir = Path(cot_dir)
    samples_path = cot_dir / "samples.json"
    if not samples_path.exists():
        raise FileNotFoundError(f"samples.json not found at {samples_path}")
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    final_results_path = cot_dir / "final_task_results.json"
    final_results = None
    if final_results_path.exists():
        try:
            final_results = json.loads(final_results_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not parse final_task_results.json: %s", exc)

    cells, scheme_by_dmt = _collect_cells(samples)
    if not cells:
        logger.warning("No non-skipped Phase-4 cells found in %s; nothing to analyse.", samples_path)
        return {"n_cells": 0}
    n_cells = len(cells)

    # Auto-detect the score scale (was 0-2, now 1-3 in current prompts).
    SCALE_MIN, SCALE_MAX = _detect_score_scale(cells)
    SCALE_MID = SCALE_MIN + 1  # the "partial" score in a 3-point scale
    SCALE_RANGE = SCALE_MAX - SCALE_MIN
    logger.info("Detected Phase-4 score scale: %d..%d", SCALE_MIN, SCALE_MAX)

    # ---- 1) scheme distribution ----------------------------------------------
    schemes_int = Counter()
    schemes_ext = Counter()
    schemes_per_model_track: Dict[tuple, Counter] = {}
    for (_did, model, track), sch in scheme_by_dmt.items():
        if track == "reasoning":
            schemes_int[sch] += 1
        else:
            schemes_ext[sch] += 1
        schemes_per_model_track.setdefault((model, track), Counter())[sch] += 1

    n_int = sum(schemes_int.values())
    n_ext = sum(schemes_ext.values())

    # transitions
    transitions = Counter()
    pairs = set((d, m) for d, m, _ in scheme_by_dmt)
    for d, m in pairs:
        r = scheme_by_dmt.get((d, m, "reasoning"))
        e = scheme_by_dmt.get((d, m, "justification"))
        if r and e:
            transitions[(r, e)] += 1
    n_trans = sum(transitions.values())
    same_cnt = sum(n for (r, e), n in transitions.items() if r == e)

    # ---- 2) failure distribution (≥1 zero) -----------------------------------
    n_any_fail = n_local_fail = n_global_fail = 0
    fail_triples = Counter()
    fail_by_scheme = Counter()
    fail_by_q = Counter()
    fail_by_metric = Counter()
    global_fail_by_metric = Counter()

    # Per-argument (per-cell) rates under each scheme.  These tell us "% of
    # cells classified under scheme S that have at least one zero (or one)
    # on dimension D", which is comparable across schemes regardless of
    # n_questions or sample size.
    cells_per_scheme: Counter = Counter()
    cells_with_local_zero_by_scheme_dim: Dict[tuple, int] = Counter()
    cells_with_local_one_by_scheme_dim: Dict[tuple, int] = Counter()
    cells_with_global_zero_by_scheme_metric: Dict[tuple, int] = Counter()
    cells_with_global_one_by_scheme_metric: Dict[tuple, int] = Counter()

    # ---- 3) room (≥1 one) ----------------------------------------------------
    n_any_room = n_local_room = n_global_room = 0
    room_triples = Counter()
    room_by_metric = Counter()
    room_by_q = Counter()
    global_room_by_metric = Counter()

    # ---- per-model accumulators ---------------------------------------------
    pm_score_dims: Dict[str, Dict[str, list]] = {}
    pm_n_cells: Counter = Counter()
    pm_local_fail: Counter = Counter()       # has ≥1 local 0
    pm_global_fail: Counter = Counter()      # has ≥1 global 0
    pm_any_fail: Counter = Counter()         # has ≥1 0 anywhere
    pm_local_room: Counter = Counter()       # has ≥1 local 1 (narrow)
    pm_global_room: Counter = Counter()      # has ≥1 global 1 (narrow)
    pm_any_room: Counter = Counter()         # has ≥1 (0|1) anywhere → narrow room
    # Broader "RFI" = at least one score below maximum (0 or 1)
    pm_local_rfi: Counter = Counter()
    pm_global_rfi: Counter = Counter()
    pm_any_rfi: Counter = Counter()
    # Means for normalised quality (0-1 = score/2)
    pm_local_scores: Dict[str, list] = {}    # per-cell mean local score (0-2)
    # Per-cell mean global score, computed over acceptability / relevance /
    # sufficiency only (coherence is excluded — see _GLOBAL_DIMS_NO_COH).
    pm_global_scores: Dict[str, list] = {}
    # Per-cell coherence score, kept as its own series so it can be reported
    # in a dedicated column without affecting the global mean.
    pm_coherence_scores: Dict[str, list] = {}

    # ---- per-scheme accumulators --------------------------------------------
    psc_overall: Dict[str, Dict[str, list]] = {}

    for c in cells:
        m = c["model"]
        track = c["track"]
        scheme = c["scheme"]
        local = c["local"]
        glob = c["global"]
        pm_n_cells[m] += 1
        cells_per_scheme[scheme] += 1
        cell_dim_zero = set()  # dims with at least one 0 in this cell
        cell_dim_one = set()   # dims with at least one 1 in this cell
        cell_glob_zero = set()
        cell_glob_one = set()

        # per-model scores
        s = pm_score_dims.setdefault(m, {
            "local_acc": [], "local_rel": [], "local_grnd": [],
            "glob_coh": [], "glob_acc": [], "glob_rel": [], "glob_suf": [],
        })
        accs, rels, grnds = [], [], []
        had_local_zero = had_local_one = False
        had_global_zero = had_global_one = False
        for q, dims in local.items():
            if not isinstance(dims, dict):
                continue
            if isinstance(dims.get("acceptability"), int):
                accs.append(dims["acceptability"])
            if isinstance(dims.get("relevance"), int):
                rels.append(dims["relevance"])
            if isinstance(dims.get("grounds"), int):
                grnds.append(dims["grounds"])
            for k, v in dims.items():
                if not isinstance(v, int):
                    continue
                if v == SCALE_MIN:
                    had_local_zero = True
                    fail_triples[(scheme, q, k)] += 1
                    fail_by_scheme[scheme] += 1
                    fail_by_q[q] += 1
                    fail_by_metric[k] += 1
                    cell_dim_zero.add(k)
                # "Room for improvement" = the partially-met middle
                # rating only (mutually exclusive with failure).  A
                # cell with v == SCALE_MIN is a failure; v == SCALE_MID
                # is room; v == SCALE_MAX is full credit.
                elif v == SCALE_MID:
                    had_local_one = True
                    room_triples[(scheme, q, k)] += 1
                    room_by_metric[k] += 1
                    room_by_q[q] += 1
                    cell_dim_one.add(k)
        if accs:
            s["local_acc"].append(sum(accs) / len(accs))
        if rels:
            s["local_rel"].append(sum(rels) / len(rels))
        if grnds:
            s["local_grnd"].append(sum(grnds) / len(grnds))
        if "coherence" in glob:
            s["glob_coh"].append(glob["coherence"])
        if "acceptability" in glob:
            s["glob_acc"].append(glob["acceptability"])
        if "relevance" in glob:
            s["glob_rel"].append(glob["relevance"])
        if "sufficiency" in glob:
            s["glob_suf"].append(glob["sufficiency"])

        for k, v in glob.items():
            if not isinstance(v, int):
                continue
            # Track per-metric fail/room counts for every global dim
            # (coherence INCLUDED) so the by-metric breakdown CSV stays
            # complete — but do NOT promote coherence flags into the
            # cell-level had_global_zero / had_global_one rollups that
            # drive the overall failure/room rates.
            if v == SCALE_MIN:
                global_fail_by_metric[k] += 1
                cell_glob_zero.add(k)
                if k in _GLOBAL_DIMS_NO_COH:
                    had_global_zero = True
            elif v == SCALE_MID:
                global_room_by_metric[k] += 1
                cell_glob_one.add(k)
                if k in _GLOBAL_DIMS_NO_COH:
                    had_global_one = True

        # Per-cell rollups -> per-scheme-dim counters
        for dim in cell_dim_zero:
            cells_with_local_zero_by_scheme_dim[(scheme, dim)] += 1
        for dim in cell_dim_one:
            cells_with_local_one_by_scheme_dim[(scheme, dim)] += 1
        for metric in cell_glob_zero:
            cells_with_global_zero_by_scheme_metric[(scheme, metric)] += 1
        for metric in cell_glob_one:
            cells_with_global_one_by_scheme_metric[(scheme, metric)] += 1

        if had_local_zero:
            n_local_fail += 1
            pm_local_fail[m] += 1
        if had_global_zero:
            n_global_fail += 1
            pm_global_fail[m] += 1
        if had_local_zero or had_global_zero:
            n_any_fail += 1
            pm_any_fail[m] += 1
        if had_local_one:
            n_local_room += 1
            pm_local_room[m] += 1
        if had_global_one:
            n_global_room += 1
            pm_global_room[m] += 1
        if had_local_one or had_global_one:
            n_any_room += 1
            pm_any_room[m] += 1

        # Broader RFI: any score below maximum (0 or 1).  A 0 implies "below
        # max", so failure → rfi.  A 1 also counts.  Only a cell that scored
        # all 2s everywhere is RFI-free.
        if had_local_zero or had_local_one:
            pm_local_rfi[m] += 1
        if had_global_zero or had_global_one:
            pm_global_rfi[m] += 1
        if had_local_zero or had_local_one or had_global_zero or had_global_one:
            pm_any_rfi[m] += 1

        # Mean normalised quality: per-cell average across local trios + global
        # triplet (coherence excluded), on a 0-2 scale.  Divide by 2 to get 0-1.
        local_nums = [
            v for q, dims in local.items()
            if isinstance(dims, dict)
            for k, v in dims.items() if isinstance(v, int)
        ]
        if local_nums:
            pm_local_scores.setdefault(m, []).append(sum(local_nums) / len(local_nums))
        global_nums = [v for k, v in glob.items()
                       if k in _GLOBAL_DIMS_NO_COH and isinstance(v, int)]
        if global_nums:
            pm_global_scores.setdefault(m, []).append(sum(global_nums) / len(global_nums))
        # Coherence reported as its own per-cell value so it can be averaged
        # into a dedicated mean_normalized_coherence column.
        coh_v = glob.get("coherence")
        if isinstance(coh_v, int):
            pm_coherence_scores.setdefault(m, []).append(coh_v)

        # per-scheme aggregator (track-aware)
        lo = _local_overall(local)
        go = _global_overall(glob)
        psc = psc_overall.setdefault(scheme, {
            "overall_int": [], "overall_ext": [],
            "local_int": [], "local_ext": [],
            "global_int": [], "global_ext": [],
        })
        if lo is not None and go is not None:
            psc[f"overall_{('int' if track=='reasoning' else 'ext')}"].append((lo + go) / 2)
        if lo is not None:
            psc[f"local_{('int' if track=='reasoning' else 'ext')}"].append(lo)
        if go is not None:
            psc[f"global_{('int' if track=='reasoning' else 'ext')}"].append(go)

    # ---- write CSVs ----------------------------------------------------------

    def write_csv(name: str, header: list, rows: list):
        with (cot_dir / name).open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in rows:
                w.writerow(r)

    # 1a) scheme distribution by track
    write_csv("scheme_distribution_by_track.csv",
              ["track", "scheme", "count", "pct_of_track"],
              [["reasoning", s, n, round(_pct(n, n_int), 2)] for s, n in schemes_int.most_common()]
              + [["justification", s, n, round(_pct(n, n_ext), 2)] for s, n in schemes_ext.most_common()])

    # 1b) scheme distribution per model (track-split, all canonical schemes
    # appear; zero rows for unused).  Canonical list = union of observed.
    observed_schemes = sorted(set(schemes_int.keys()) | set(schemes_ext.keys()))
    pm_rows = []
    for m in sorted(pm_n_cells.keys()):
        int_ct = schemes_per_model_track.get((m, "reasoning"), Counter())
        ext_ct = schemes_per_model_track.get((m, "justification"), Counter())
        triples = [(s, int_ct.get(s, 0), ext_ct.get(s, 0)) for s in observed_schemes]
        triples.sort(key=lambda r: (-(r[1] + r[2]), r[0]))
        for s, ri, ej in triples:
            pm_rows.append([m, s, ri, ej, ri + ej, ej - ri])
    # __ALL__ block
    agg_int = Counter(); agg_ext = Counter()
    for (m, t), ct in schemes_per_model_track.items():
        for s, n in ct.items():
            if t == "reasoning":
                agg_int[s] += n
            else:
                agg_ext[s] += n
    triples = [(s, agg_int.get(s, 0), agg_ext.get(s, 0)) for s in observed_schemes]
    triples.sort(key=lambda r: (-(r[1] + r[2]), r[0]))
    for s, ri, ej in triples:
        pm_rows.append(["__ALL__", s, ri, ej, ri + ej, ej - ri])
    write_csv("scheme_distribution_per_model.csv",
              ["model", "scheme", "reasoning", "justification", "total", "shift_ext_minus_int"],
              pm_rows)

    # 1c) scheme transitions
    write_csv("scheme_transitions.csv",
              ["reasoning_scheme", "justification_scheme", "count", "pct", "same"],
              [[r, e, n, round(_pct(n, n_trans), 2), r == e] for (r, e), n in transitions.most_common()])

    # 2) scores per model (uses final_task_results for headline stats when available,
    # otherwise computed from cells)
    sp_model_rows = []
    # Compute headline mean/stderr/mean_int/mean_ext from the cells themselves
    # so merged runs (which have no aggregated final_task_results.json) still
    # populate these columns.  If a final_task_results.json IS present and
    # provides the headline numbers, prefer those (they include inspect-ai's
    # metric reducers).
    import math
    fr_by_short = {}
    if final_results is not None:
        fr_by_short = {k.split("/")[-1]: v for k, v in final_results.get("models", {}).items()}
    pm_cell_overall: Dict[str, list] = {}
    pm_cell_int: Dict[str, list] = {}
    pm_cell_ext: Dict[str, list] = {}
    pm_r_int_dilemmas: Dict[str, set] = {}
    pm_total_dilemmas: Dict[str, set] = {}
    for c in cells:
        m = c["model"]
        # Per-cell mean across all individual scores.  Coherence is held out
        # of the global half so the headline mean / mean_int / mean_ext
        # columns are coherence-free (consistent with the 2026-05 setup
        # change — coherence is reported as a diagnostic, not pooled in).
        local_nums = [v for q, dims in (c.get("local") or {}).items()
                      if isinstance(dims, dict)
                      for v in dims.values() if isinstance(v, int)]
        global_nums = [v for k, v in (c.get("global") or {}).items()
                       if k in _GLOBAL_DIMS_NO_COH and isinstance(v, int)]
        all_nums = local_nums + global_nums
        if all_nums:
            cell_overall = sum(all_nums) / len(all_nums)
            pm_cell_overall.setdefault(m, []).append(cell_overall)
            if c["track"] == "reasoning":
                pm_cell_int.setdefault(m, []).append(cell_overall)
            else:
                pm_cell_ext.setdefault(m, []).append(cell_overall)
    # r_int_available proxy: fraction of (model, dilemma) pairs that had a
    # non-empty reasoning track.  Inferred from scheme_by_dmt — present only
    # when target_text was non-empty.  We pre-built scheme_by_dmt earlier.
    for (did, m, track), _ in scheme_by_dmt.items():
        pm_total_dilemmas.setdefault(m, set()).add(did)
        if track == "reasoning":
            pm_r_int_dilemmas.setdefault(m, set()).add(did)

    def _stderr(xs):
        n = len(xs)
        if n < 2:
            return 0.0
        mean = sum(xs) / n
        var = sum((x - mean) ** 2 for x in xs) / (n - 1)
        return (var / n) ** 0.5

    model_order = sorted(pm_n_cells.keys())
    # Sort by computed mean desc
    def _sort_key(m):
        if m in fr_by_short:
            return -fr_by_short[m].get("mean_ignore_refused", 0)
        return -_avg(pm_cell_overall.get(m, [0]))
    model_order = sorted(model_order, key=_sort_key)

    for m in model_order:
        s = pm_score_dims.get(m, {})
        fr = fr_by_short.get(m, {})
        overall_cells = pm_cell_overall.get(m, [])
        int_cells = pm_cell_int.get(m, [])
        ext_cells = pm_cell_ext.get(m, [])
        total_d = pm_total_dilemmas.get(m, set())
        rint_d = pm_r_int_dilemmas.get(m, set())
        # Always use the locally-computed cell means so the mean / mean_int /
        # mean_ext columns are coherence-free.  We do NOT fall back to
        # final_task_results.json here even when present: inspect-ai's
        # metric reducer pools all four global dims (incl. coherence) and
        # would re-introduce coherence into the headline number.  stderr is
        # taken from inspect-ai when available (a refusal-aware count we
        # can't easily reproduce here) and otherwise computed.
        mean_val = _avg(overall_cells)
        stderr_val = fr.get("stderr_ignore_refused")
        if stderr_val is None:
            stderr_val = _stderr(overall_cells)
        mean_int = _avg(int_cells) if int_cells else float("nan")
        mean_ext = _avg(ext_cells) if ext_cells else float("nan")
        r_int_avail = fr.get("r_int_available")
        if r_int_avail is None:
            r_int_avail = len(rint_d) / len(total_d) if total_d else 0.0

        def _safe_round(v, n):
            try:
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    return ""
                return round(v, n)
            except Exception:
                return ""

        # Normalise every score to 0-1 (raw → (raw - SCALE_MIN) / SCALE_RANGE)
        # so all numbers in scores_per_model.csv share one common scale.
        # stderr is a linear measure on the same scale so it divides by RANGE.
        def _norm(v):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return float("nan")
            try:
                return (float(v) - SCALE_MIN) / SCALE_RANGE
            except Exception:
                return float("nan")

        def _norm_stderr(v):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return float("nan")
            try:
                return float(v) / SCALE_RANGE
            except Exception:
                return float("nan")

        sp_model_rows.append([
            m,
            _safe_round(_norm(mean_val), 4),
            _safe_round(_norm_stderr(stderr_val), 4),
            _safe_round(_norm(mean_int), 4),
            _safe_round(_norm(mean_ext), 4),
            _safe_round(r_int_avail, 3),
            _safe_round(_norm(_avg(s.get("local_acc", []))), 3),
            _safe_round(_norm(_avg(s.get("local_rel", []))), 3),
            _safe_round(_norm(_avg(s.get("local_grnd", []))), 3),
            _safe_round(_norm(_avg(s.get("glob_coh", []))), 3),
            _safe_round(_norm(_avg(s.get("glob_acc", []))), 3),
            _safe_round(_norm(_avg(s.get("glob_rel", []))), 3),
            _safe_round(_norm(_avg(s.get("glob_suf", []))), 3),
            pm_n_cells[m],
        ])
    write_csv("scores_per_model.csv",
              ["model", "mean", "stderr", "mean_int", "mean_ext", "r_int_available",
               "local_acceptability", "local_relevance", "local_grounds",
               "global_coherence", "global_acceptability", "global_relevance", "global_sufficiency",
               "n_cells"],
              sp_model_rows)

    # -- per-model metrics: scheme consistency, failure & RFI rates, normalised quality --
    # Scheme consistency: for each (model, dilemma) pair, do reasoning_scheme
    # and justification_scheme match?  Computed from scheme_by_dmt.
    pm_consistency: Dict[str, Dict[str, int]] = {}
    for (did, m, track), sch in scheme_by_dmt.items():
        d = pm_consistency.setdefault(m, {})
        if track == "reasoning":
            d.setdefault("reasoning", {})[did] = sch
        else:
            d.setdefault("justification", {})[did] = sch

    pmm_rows = []
    for m in model_order:
        n_m = pm_n_cells[m] or 0
        # Scheme consistency: count dilemmas where reasoning_scheme == justification_scheme
        d = pm_consistency.get(m, {})
        r_map = d.get("reasoning", {}); j_map = d.get("justification", {})
        common = set(r_map) & set(j_map)
        consistent = sum(1 for did in common if r_map[did] == j_map[did])
        consistency_pct = round(_pct(consistent, len(common)), 2) if common else 0.0

        # Mean normalised quality (0-1): (raw - SCALE_MIN) / SCALE_RANGE.
        # global_mean here is computed from _GLOBAL_DIMS_NO_COH only;
        # coherence_mean is its own averaged column so the reader can see
        # the coherence signal without it being mixed into the global mean.
        local_mean = _avg(pm_local_scores.get(m, []))
        global_mean = _avg(pm_global_scores.get(m, []))
        coh_mean = _avg(pm_coherence_scores.get(m, []))
        def _norm_or_blank(v):
            return (
                round((v - SCALE_MIN) / SCALE_RANGE, 4)
                if v == v and SCALE_RANGE > 0 else ""
            )
        norm_local = _norm_or_blank(local_mean)
        norm_global = _norm_or_blank(global_mean)
        norm_coh = _norm_or_blank(coh_mean)

        # Per-cell counts of coherence flags (1=fail, 2=room) — reported
        # separately so the reader sees the coherence signal we excluded
        # from the global-failure rate.
        coh_scores = pm_coherence_scores.get(m, [])
        n_coh_fail = sum(1 for v in coh_scores if v == SCALE_MIN)
        n_coh_room = sum(1 for v in coh_scores if v == SCALE_MID)
        coh_fail_pct = round(_pct(n_coh_fail, n_m), 2) if n_m else 0.0
        coh_room_pct = round(_pct(n_coh_room, n_m), 2) if n_m else 0.0

        pmm_rows.append([
            m, n_m,
            consistency_pct, len(common), consistent,
            round(_pct(pm_any_fail[m], n_m), 2),
            round(_pct(pm_local_fail[m], n_m), 2),
            round(_pct(pm_global_fail[m], n_m), 2),
            round(_pct(pm_any_rfi[m], n_m), 2),
            round(_pct(pm_local_rfi[m], n_m), 2),
            round(_pct(pm_global_rfi[m], n_m), 2),
            norm_local, norm_global, norm_coh,
            coh_fail_pct, coh_room_pct,
        ])
    write_csv("per_model_metrics.csv",
              ["model", "n_cells",
               "scheme_consistency_pct", "n_dilemmas_with_both_tracks", "n_dilemmas_consistent",
               "overall_failure_rate_pct", "local_failure_rate_pct", "global_failure_rate_pct",
               "overall_rfi_rate_pct", "local_rfi_rate_pct", "global_rfi_rate_pct",
               "mean_normalized_local_quality", "mean_normalized_global_quality",
               "mean_normalized_coherence",
               "coherence_failure_rate_pct", "coherence_room_rate_pct"],
              pmm_rows)

    # 3) scores per scheme
    sps_rows = []
    schemes_by_n = sorted(psc_overall, key=lambda s: -(len(psc_overall[s]["overall_int"]) + len(psc_overall[s]["overall_ext"])))
    for sch in schemes_by_n:
        p = psc_overall[sch]
        n_int_o = len(p["overall_int"]); n_ext_o = len(p["overall_ext"])
        all_overall = p["overall_int"] + p["overall_ext"]
        all_local = p["local_int"] + p["local_ext"]
        all_global = p["global_int"] + p["global_ext"]
        # Normalise to 0-1 for consistency with scores_per_model.csv.
        def _n(v): return (v - SCALE_MIN) / SCALE_RANGE
        sps_rows.append([
            sch, n_int_o + n_ext_o, n_int_o, n_ext_o,
            round(_n(_avg(all_overall)), 3) if all_overall else "",
            round(_n(_avg(p["overall_int"])), 3) if n_int_o else "",
            round(_n(_avg(p["overall_ext"])), 3) if n_ext_o else "",
            round(_n(_avg(all_local)), 3) if all_local else "",
            round(_n(_avg(all_global)), 3) if all_global else "",
        ])
    write_csv("scores_per_scheme.csv",
              ["scheme", "n_total", "n_reasoning", "n_justification",
               "mean_overall", "mean_overall_reasoning", "mean_overall_justification",
               "mean_local_overall", "mean_global_overall"],
              sps_rows)

    # 4) failure/room headline summary.  "global" scope here now excludes
    # coherence; coherence is reported as its own scope row.
    n_coh_fail_total = global_fail_by_metric.get("coherence", 0)
    n_coh_room_total = global_room_by_metric.get("coherence", 0)
    summary_rows = [
        ["failure", "any", n_any_fail, n_cells, round(_pct(n_any_fail, n_cells), 2)],
        ["failure", "local", n_local_fail, n_cells, round(_pct(n_local_fail, n_cells), 2)],
        ["failure", "global", n_global_fail, n_cells, round(_pct(n_global_fail, n_cells), 2)],
        ["failure", "coherence", n_coh_fail_total, n_cells, round(_pct(n_coh_fail_total, n_cells), 2)],
        ["room", "any", n_any_room, n_cells, round(_pct(n_any_room, n_cells), 2)],
        ["room", "local", n_local_room, n_cells, round(_pct(n_local_room, n_cells), 2)],
        ["room", "global", n_global_room, n_cells, round(_pct(n_global_room, n_cells), 2)],
        ["room", "coherence", n_coh_room_total, n_cells, round(_pct(n_coh_room_total, n_cells), 2)],
    ]
    # Per-model failure/room counts.  Coherence is added as its own scope
    # row alongside any/local/global so the reader can see the coherence
    # flag rate without it being folded into global_failure.
    for m in model_order:
        n_m = pm_n_cells[m]
        coh_scores_m = pm_coherence_scores.get(m, [])
        n_coh_fail_m = sum(1 for v in coh_scores_m if v == SCALE_MIN)
        n_coh_room_m = sum(1 for v in coh_scores_m if v == SCALE_MID)
        summary_rows.append([f"failure:{m}", "any", pm_any_fail[m], n_m, round(_pct(pm_any_fail[m], n_m), 2)])
        summary_rows.append([f"failure:{m}", "local", pm_local_fail[m], n_m, round(_pct(pm_local_fail[m], n_m), 2)])
        summary_rows.append([f"failure:{m}", "global", pm_global_fail[m], n_m, round(_pct(pm_global_fail[m], n_m), 2)])
        summary_rows.append([f"failure:{m}", "coherence", n_coh_fail_m, n_m, round(_pct(n_coh_fail_m, n_m), 2)])
        summary_rows.append([f"room:{m}", "any", pm_any_room[m], n_m, round(_pct(pm_any_room[m], n_m), 2)])
        summary_rows.append([f"room:{m}", "local", pm_local_room[m], n_m, round(_pct(pm_local_room[m], n_m), 2)])
        summary_rows.append([f"room:{m}", "global", pm_global_room[m], n_m, round(_pct(pm_global_room[m], n_m), 2)])
        summary_rows.append([f"room:{m}", "coherence", n_coh_room_m, n_m, round(_pct(n_coh_room_m, n_m), 2)])
    write_csv("failure_room_summary.csv",
              ["category", "scope", "n_cells", "n_total", "pct"], summary_rows)

    # 5) failure/room global metric breakdown
    grm_rows = []
    for k in ("coherence", "acceptability", "relevance", "sufficiency"):
        grm_rows.append(["failure", k, global_fail_by_metric.get(k, 0), n_cells, round(_pct(global_fail_by_metric.get(k, 0), n_cells), 2)])
    for k in ("coherence", "acceptability", "relevance", "sufficiency"):
        grm_rows.append(["room", k, global_room_by_metric.get(k, 0), n_cells, round(_pct(global_room_by_metric.get(k, 0), n_cells), 2)])
    write_csv("failure_room_global_metrics.csv",
              ["category", "metric", "n_cells", "n_total", "pct"], grm_rows)

    # 6) local failure triples
    write_csv("failures_local_by_scheme_q_metric.csv",
              ["scheme", "question", "metric", "n_zeros"],
              [[s, q, k, n] for (s, q, k), n in fail_triples.most_common()])

    # 7) local room triples
    write_csv("room_local_by_scheme_q_metric.csv",
              ["scheme", "question", "metric", "n_ones"],
              [[s, q, k, n] for (s, q, k), n in room_triples.most_common()])

    # 7b) per-argument failure/room rates by (scheme, dimension)
    # One row per (scheme, dimension); columns: n cells under that scheme,
    # n cells with >=1 zero on dim, % of those cells, same for ones.
    LOCAL_DIMS = ("acceptability", "relevance", "grounds")
    GLOBAL_METRICS = ("coherence", "acceptability", "relevance", "sufficiency")
    pa_rows = []
    for sch in sorted(cells_per_scheme, key=lambda s: -cells_per_scheme[s]):
        n = cells_per_scheme[sch]
        for dim in LOCAL_DIMS:
            fail = cells_with_local_zero_by_scheme_dim.get((sch, dim), 0)
            room = cells_with_local_one_by_scheme_dim.get((sch, dim), 0)
            pa_rows.append([
                sch, "local", dim, n, fail, round(_pct(fail, n), 2),
                room, round(_pct(room, n), 2),
            ])
        for metric in GLOBAL_METRICS:
            fail = cells_with_global_zero_by_scheme_metric.get((sch, metric), 0)
            room = cells_with_global_one_by_scheme_metric.get((sch, metric), 0)
            pa_rows.append([
                sch, "global", metric, n, fail, round(_pct(fail, n), 2),
                room, round(_pct(room, n), 2),
            ])
    write_csv("per_argument_failure_room_by_scheme.csv",
              ["scheme", "level", "dimension", "n_cells_under_scheme",
               "n_cells_with_failure", "pct_cells_with_failure",
               "n_cells_with_room", "pct_cells_with_room"],
              pa_rows)

    # 8) readable summary text
    lines = []
    lines.append(f"# Walton-analysis summary for {cot_dir}")
    lines.append(f"Total non-skipped Phase-4 cells: {n_cells}")
    lines.append("")
    lines.append("## Scheme distribution")
    lines.append(f"  Reasoning (n={n_int}): top = {schemes_int.most_common(1)[0] if schemes_int else 'n/a'}")
    lines.append(f"  Justification (n={n_ext}): top = {schemes_ext.most_common(1)[0] if schemes_ext else 'n/a'}")
    lines.append(f"  Transitions (n={n_trans}): same-scheme {same_cnt} ({_pct(same_cnt, n_trans):.1f}%), change {n_trans-same_cnt} ({_pct(n_trans-same_cnt, n_trans):.1f}%)")
    lines.append("")
    lines.append("## Failure distribution")
    lines.append(f"  any  : {n_any_fail}/{n_cells} = {_pct(n_any_fail, n_cells):.1f}%")
    lines.append(f"  local: {n_local_fail}/{n_cells} = {_pct(n_local_fail, n_cells):.1f}%")
    lines.append(f"  global: {n_global_fail}/{n_cells} = {_pct(n_global_fail, n_cells):.1f}%")
    if fail_by_metric:
        lines.append(f"  local zeros by metric: {dict(fail_by_metric.most_common())}")
    if global_fail_by_metric:
        lines.append(f"  global zeros by metric: {dict(global_fail_by_metric.most_common())}")
    lines.append("")
    lines.append("## Room-for-improvement distribution")
    lines.append(f"  any  : {n_any_room}/{n_cells} = {_pct(n_any_room, n_cells):.1f}%")
    lines.append(f"  local: {n_local_room}/{n_cells} = {_pct(n_local_room, n_cells):.1f}%")
    lines.append(f"  global: {n_global_room}/{n_cells} = {_pct(n_global_room, n_cells):.1f}%")
    if room_by_metric:
        lines.append(f"  local ones by metric: {dict(room_by_metric.most_common())}")
    if global_room_by_metric:
        lines.append(f"  global ones by metric: {dict(global_room_by_metric.most_common())}")
    lines.append("")
    lines.append("## CSV outputs")
    for fn in _OUT_FILES[:-1]:
        if (cot_dir / fn).exists():
            lines.append(f"  - {fn}")
    (cot_dir / "analysis_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "n_cells": n_cells,
        "n_any_failure": n_any_fail,
        "n_local_failure": n_local_fail,
        "n_global_failure": n_global_fail,
        "n_any_room": n_any_room,
        "n_local_room": n_local_room,
        "n_global_room": n_global_room,
        "outputs": [str(cot_dir / fn) for fn in _OUT_FILES if (cot_dir / fn).exists()],
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cot_dir", type=Path,
                   help="Path to a cot directory containing samples.json (typically "
                        "logs/deliberative_justification_v2/<ts>/cot)")
    args = p.parse_args(argv)
    summary = analyze_run(args.cot_dir)
    print(f"n_cells={summary.get('n_cells')}")
    print(f"failure  any/local/global = "
          f"{summary.get('n_any_failure')}/{summary.get('n_local_failure')}/{summary.get('n_global_failure')}")
    print(f"room     any/local/global = "
          f"{summary.get('n_any_room')}/{summary.get('n_local_room')}/{summary.get('n_global_room')}")
    for path in summary.get("outputs", []):
        print(f"wrote: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
