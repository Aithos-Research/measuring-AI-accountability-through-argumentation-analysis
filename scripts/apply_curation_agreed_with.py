"""Apply human-curator `agreed_with` decisions to samples.json.

For each polarised-failure cell in the curation CSV that the curator
resolved by choosing one of the two judges ('a' or 'b' in the
``agreed_with`` column), this script overrides BOTH Phase-4 entries
on that cell with the chosen judge's scoring matrix.  This makes the
cell agree with the curator's verdict while keeping the cell present
on both judges' tallies (so downstream per-judge analyses stay
well-defined).

Cells with ``agreed_with`` empty or one of ``neither / both / mixed``
are left untouched.

Usage:
    python scripts/apply_curation_agreed_with.py \\
        --curation "C:/Users/arnol/Downloads/curation - Curation_paper.csv" \\
        --run-dir  logs/deliberative_justification_v2/20260527_combined_panels_curation/reasoning \\
        [--dry-run]

The script writes a backup ``samples.bak_curation_<ts>.json`` next to
samples.json before mutating it, and appends one entry per override to
the ``curation_overrides`` block at the top of samples.json for audit.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _find_track(samples: dict, dilemma_id: str, model_short: str, track: str) -> dict | None:
    """Locate the track block matching (dilemma_id, model_short, track)."""
    for dil in samples.get("dilemmas", []):
        if dil.get("dilemma_id") != dilemma_id:
            continue
        for tg in dil.get("targets", []):
            if tg.get("model", "").split("/")[-1] != model_short:
                continue
            for tk in tg.get("tracks", []) or []:
                if tk.get("text_kind") == track:
                    return tk
    return None


def _scores_dict_from_p4(p4: dict) -> dict:
    out_local: dict = {}
    for q, dims in (p4.get("local") or {}).items():
        if isinstance(dims, dict):
            out_local[q] = {d: int(dims[d]) for d in dims if isinstance(dims[d], (int, float))}
    out_global = {d: int(v) for d, v in (p4.get("global") or {}).items()
                  if isinstance(v, (int, float))}
    return {"local": out_local, "global": out_global}


def _apply_scores(p4: dict, scores: dict) -> dict:
    """Replace the p4 entry's local/global scores with the given dict.
    Returns the previous (local, global) for audit."""
    prev_local = (p4.get("local") or {}).copy()
    prev_global = (p4.get("global") or {}).copy()
    p4["local"] = scores.get("local") or {}
    p4["global"] = scores.get("global") or {}
    return {"local": prev_local, "global": prev_global}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--curation", required=True, type=Path,
                   help="Curation CSV from the researcher (with agreed_with column).")
    p.add_argument("--run-dir", required=True, type=Path,
                   help="Run directory containing samples.json.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would change; do not write anything.")
    args = p.parse_args(argv)

    samples_path = args.run_dir / "samples.json"
    if not samples_path.exists():
        raise SystemExit(f"samples.json missing: {samples_path}")
    if not args.curation.exists():
        raise SystemExit(f"curation CSV missing: {args.curation}")

    samples = json.loads(samples_path.read_text(encoding="utf-8"))

    with args.curation.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        curation_rows = list(reader)

    n_total = len(curation_rows)
    n_filled = 0
    n_skipped_no_decision = 0
    n_skipped_unsupported = 0
    n_applied = 0
    n_not_found = 0
    by_choice = {"a": 0, "b": 0}

    overrides = samples.setdefault("curation_overrides", [])
    ts = datetime.now().isoformat(timespec="seconds")

    for row in curation_rows:
        agreed = (row.get("agreed_with") or "").strip().lower()
        if not agreed:
            n_skipped_no_decision += 1
            continue
        n_filled += 1
        if agreed not in ("a", "b"):
            n_skipped_unsupported += 1
            continue

        cell_id = row.get("cell_id", "")
        dilemma_id = row.get("dilemma_id", "")
        model = row.get("model", "")
        track = row.get("track", "")
        ja = row.get("judge_a", "")
        jb = row.get("judge_b", "")

        # The chosen judge's scores come straight from the curation CSV
        # (so the human's review is using the exact numbers they saw).
        # Curation CSVs use either the old column names (judge_a_scores /
        # judge_b_scores) or the new phase-prefixed names
        # (p4_judge_a_scores / p4_judge_b_scores); try both, prefer the
        # prefixed names when both are present.
        col_prefixed = "p4_judge_a_scores" if agreed == "a" else "p4_judge_b_scores"
        col_legacy = "judge_a_scores" if agreed == "a" else "judge_b_scores"
        raw = (row.get(col_prefixed) or row.get(col_legacy) or "").strip()
        if not raw:
            print(f"  skipping {cell_id}: no scores in either "
                  f"{col_prefixed!r} or {col_legacy!r}", file=sys.stderr)
            n_skipped_unsupported += 1
            continue
        try:
            chosen_scores = json.loads(raw)
        except json.JSONDecodeError:
            print(f"  skipping {cell_id}: invalid scores JSON", file=sys.stderr)
            n_skipped_unsupported += 1
            continue

        tk = _find_track(samples, dilemma_id, model, track)
        if tk is None:
            n_not_found += 1
            print(f"  not found in samples.json: {dilemma_id} / {model} / {track}",
                  file=sys.stderr)
            continue

        # Locate both Phase-4 entries for this (cell, judge pair) and
        # overwrite their local/global with the chosen judge's scores.
        ja_p4 = jb_p4 = None
        for p4 in tk.get("phase4", []) or []:
            if p4.get("phase4_judge") == ja:
                ja_p4 = p4
            elif p4.get("phase4_judge") == jb:
                jb_p4 = p4
        if ja_p4 is None or jb_p4 is None:
            n_not_found += 1
            print(f"  judge entries missing in samples.json: {dilemma_id} / {model} / {track}"
                  f" (ja={ja} found={ja_p4 is not None}, jb={jb} found={jb_p4 is not None})",
                  file=sys.stderr)
            continue

        if args.dry_run:
            n_applied += 1
            by_choice[agreed] += 1
            continue

        # Record the pre-override state per judge for audit.
        prev_a = _scores_dict_from_p4(ja_p4)
        prev_b = _scores_dict_from_p4(jb_p4)
        _apply_scores(ja_p4, chosen_scores)
        _apply_scores(jb_p4, chosen_scores)
        ja_p4["curated"] = {"agreed_with": agreed, "previous": prev_a, "ts": ts}
        jb_p4["curated"] = {"agreed_with": agreed, "previous": prev_b, "ts": ts}

        overrides.append({
            "cell_id": cell_id,
            "dilemma_id": dilemma_id,
            "model": model,
            "track": track,
            "judge_a": ja,
            "judge_b": jb,
            "agreed_with": agreed,
            "applied_scores": chosen_scores,
            "previous_a": prev_a,
            "previous_b": prev_b,
            "curator": row.get("curator", ""),
            "human_reasoning": row.get("human_reasoning", ""),
            "applied_at": ts,
        })
        n_applied += 1
        by_choice[agreed] += 1

    print(f"curation rows:            {n_total}")
    print(f"  decision filled in:     {n_filled}")
    print(f"  no decision (skipped):  {n_skipped_no_decision}")
    print(f"  unsupported value:      {n_skipped_unsupported}")
    print(f"  not found in samples:   {n_not_found}")
    print(f"  applied:                {n_applied}")
    print(f"     -> agreed_with=a:    {by_choice['a']}")
    print(f"     -> agreed_with=b:    {by_choice['b']}")

    if args.dry_run:
        print()
        print("--dry-run: samples.json not modified.")
        return 0

    if n_applied == 0:
        print("nothing to apply; samples.json untouched.")
        return 0

    bak = samples_path.with_name(
        f"samples.bak_curation_{ts.replace(':', '-').replace('+', '_')}.json"
    )
    shutil.copy2(samples_path, bak)
    samples_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    print()
    print(f"backup written: {bak}")
    print(f"updated: {samples_path}")
    print()
    print("Re-run analysis + plots to refresh derived files:")
    print(f"  python -m moral_eval.analysis.walton_analysis {args.run_dir}")
    print(f"  python -m moral_eval.analysis.walton_plots    {args.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
