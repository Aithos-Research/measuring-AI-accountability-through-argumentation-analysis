"""Apply curator verdicts for the two cells the human curator filled in
after the initial pass.  Used as a small follow-up to
``apply_curation_agreed_with.py`` so the audit trail still flows
through the CSV-driven path.

Pulls the chosen judge's scores from the curation CSV (matched by
cell_id) and overrides both Phase-4 entries on that (cell, judge_pair)
in samples.json, the same way the main apply script does.

Usage:
    python scripts/apply_curation_extra.py \
        --curation "C:/Users/arnol/Downloads/curation - Curation_paper.csv" \
        --run-dir  logs/deliberative_justification_v2/20260527_combined_panels_curation/reasoning \
        --cell aa942ab0afe82154 a \
        --cell 705e9c545ed5db7c b
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


def _find_track(samples, dilemma_id, model_short, track):
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--curation", required=True, type=Path)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--cell", nargs=2, action="append", required=True,
                   metavar=("CELL_ID", "VERDICT"),
                   help="Cell ID + 'a' or 'b' verdict.  May be repeated.")
    args = p.parse_args(argv)

    samples_path = args.run_dir / "samples.json"
    samples = json.loads(samples_path.read_text(encoding="utf-8"))

    with args.curation.open("r", encoding="utf-8") as f:
        rows = {r["cell_id"]: r for r in csv.DictReader(f)}

    overrides = samples.setdefault("curation_overrides", [])
    ts = datetime.now().isoformat(timespec="seconds")

    n_applied = 0
    for cell_id, verdict in args.cell:
        verdict = verdict.strip().lower()
        if verdict not in ("a", "b"):
            print(f"  skipping {cell_id}: verdict must be 'a' or 'b', got {verdict!r}", file=sys.stderr)
            continue
        row = rows.get(cell_id)
        if row is None:
            print(f"  cell_id {cell_id} not in curation CSV", file=sys.stderr)
            continue
        chosen = json.loads(row.get("judge_a_scores" if verdict == "a" else "judge_b_scores") or "{}")
        tk = _find_track(samples, row["dilemma_id"], row["model"], row["track"])
        if tk is None:
            print(f"  cell not in samples.json: {row['dilemma_id']} / {row['model']} / {row['track']}", file=sys.stderr)
            continue
        ja, jb = row["judge_a"], row["judge_b"]
        applied_pair = []
        for p4 in tk.get("phase4", []) or []:
            if p4.get("phase4_judge") in (ja, jb):
                prev = {"local": (p4.get("local") or {}).copy(),
                        "global": (p4.get("global") or {}).copy()}
                p4["local"] = chosen.get("local") or {}
                p4["global"] = chosen.get("global") or {}
                p4["curated"] = {"agreed_with": verdict, "previous": prev, "ts": ts}
                applied_pair.append(p4.get("phase4_judge"))
        if len(applied_pair) == 2:
            overrides.append({
                "cell_id": cell_id, "dilemma_id": row["dilemma_id"],
                "model": row["model"], "track": row["track"],
                "judge_a": ja, "judge_b": jb,
                "agreed_with": verdict, "applied_scores": chosen,
                "applied_at": ts, "source": "apply_curation_extra",
            })
            n_applied += 1
            print(f"  applied {cell_id} ({row['dilemma_id']}, {row['model']}, {row['track']}) -> {verdict}")
        else:
            print(f"  cell {cell_id}: expected 2 phase4 entries (ja={ja}, jb={jb}), found {applied_pair}", file=sys.stderr)

    if n_applied == 0:
        print("nothing applied.")
        return 0

    bak = samples_path.with_name(
        f"samples.bak_curation_extra_{ts.replace(':', '-')}.json"
    )
    shutil.copy2(samples_path, bak)
    samples_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nbackup: {bak}")
    print(f"updated: {samples_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
