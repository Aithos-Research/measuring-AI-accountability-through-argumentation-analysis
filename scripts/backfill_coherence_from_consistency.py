"""Backfill global.coherence in a dj_v2 run from the raw judge responses.

Background
----------
The Phase-4 prompts asked the judge to score four global dimensions and used
the label ``consistency`` for the first one.  The Phase-4 structured-output
schema, however, declared the field as ``coherence``.  Claude (which honours
structured output strictly) emitted ``"consistency": <1-3>`` in its JSON, the
schema layer dropped that key, and the required ``coherence`` field fell back
to integer 0.  Every Sonnet-judged cell in such a run has
``global.coherence = 0``.  GPT-5.4 (whose adapter is more permissive) mapped
the prompt's consistency wording onto ``coherence`` correctly, so its
``global.coherence`` is already populated.

This script reads the per-task eval JSON logs alongside samples.json, extracts
the ``"consistency": <int>`` value from each affected Sonnet
``judge2_raw_*`` field, and writes it back into the cell's
``global.coherence`` in samples.json.  GPT-5.4 cells are left alone.

A ``.bak`` of samples.json is written first.

Usage
-----
    .\\.venv\\Scripts\\python.exe scripts/backfill_coherence_from_consistency.py \\
        --cot-dir logs/deliberative_justification_v2/20260527_combined_thinking_n200/reasoning \\
        --source-dirs logs/deliberative_justification_v2/20260526_114842/reasoning \\
                       logs/deliberative_justification_v2/20260527_002619/reasoning
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


_CONSISTENCY_RE = re.compile(r'"consistency"\s*:\s*(\d+)')


def _short_model(name: str) -> str:
    return name.rsplit("/", 1)[-1]


_KNOWN_JUDGE_ALIASES = ("claude_sonnet_4_6", "gpt_5_4", "gpt_5_5")


def _build_raw_index(source_dirs: list[Path]) -> dict[tuple[str, str, str, str], str]:
    """Walk per-task eval JSON files and return a map from
    (judge_alias, target_model, dilemma_id, track) -> raw Phase-4 string."""
    index: dict[tuple[str, str, str, str], str] = {}
    for sd in source_dirs:
        for fp in sorted(glob.glob(str(sd / "*.json"))):
            base = os.path.basename(fp)
            if base in {"samples.json", "final_task_results.json"} or "samples.bak" in base:
                continue
            try:
                data = json.loads(Path(fp).read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"  WARN: could not parse {fp}: {exc}")
                continue
            model = (data.get("eval") or {}).get("model", "")
            target_short = _short_model(model)
            for s in data.get("samples") or []:
                did = (s.get("metadata") or {}).get("dilemma_id")
                if not did:
                    continue
                sm = (s.get("scores") or {}).get("dj_v2_scorer", {}).get("metadata") or {}
                for k, v in sm.items():
                    if not k.startswith("judge2_raw_"):
                        continue
                    if not (k.endswith("_int") or k.endswith("_ext")):
                        continue
                    track_short = k.rsplit("_", 1)[-1]
                    track = "reasoning" if track_short == "int" else "justification"
                    # Strip the prefix and the _int|_ext suffix, then split the
                    # remainder into <phase2_judge>_<phase4_judge> by matching
                    # a known judge alias at the end (alias names contain
                    # underscores so a naive split would mis-segment).
                    body = k[len("judge2_raw_"):-len(track_short) - 1]
                    phase4_judge = None
                    for alias in _KNOWN_JUDGE_ALIASES:
                        if body.endswith("_" + alias):
                            phase4_judge = alias
                            break
                    if phase4_judge is None:
                        continue
                    if not isinstance(v, str) or not v.strip():
                        continue
                    index[(phase4_judge, target_short, did, track)] = v
    return index


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cot-dir", required=True, type=Path,
                    help="Directory containing the samples.json to patch.")
    ap.add_argument("--source-dirs", nargs="+", required=True, type=Path,
                    help="Per-task eval log directories that hold the raw "
                         "judge2_raw_* fields (typically the base run and "
                         "the restart run that were merged).")
    ap.add_argument("--judge-alias", default="claude_sonnet_4_6",
                    help="Phase-4 judge whose raw response carries the "
                         "(prompt-wording) 'consistency' value.  Defaults to "
                         "Sonnet because GPT's adapter already mapped "
                         "consistency -> coherence in the structured output.")
    args = ap.parse_args(argv)

    samples_path = args.cot_dir / "samples.json"
    if not samples_path.exists():
        raise SystemExit(f"missing samples.json: {samples_path}")

    print("Indexing raw judge2_raw_* fields across source dirs...")
    raw_index = _build_raw_index(args.source_dirs)
    print(f"  indexed {len(raw_index)} (judge, model, dilemma, track) entries")

    print(f"Loading {samples_path}...")
    samples = json.loads(samples_path.read_text(encoding="utf-8"))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = samples_path.with_name(f"samples.bak_{stamp}.json")
    shutil.copy2(samples_path, bak)
    print(f"  backup: {bak}")

    patched = 0
    no_raw = 0
    no_match = 0
    by_model: dict[str, int] = defaultdict(int)
    for dil in samples["dilemmas"]:
        did = dil.get("dilemma_id")
        for tg in dil.get("targets") or []:
            tm = _short_model(tg.get("model", ""))
            for tk in tg.get("tracks") or []:
                track = "reasoning" if tk.get("text_kind") == "reasoning" else "justification"
                for p4 in (tk.get("phase4") or []):
                    if p4.get("skipped"):
                        continue
                    if p4.get("phase4_judge") != args.judge_alias:
                        continue
                    glob_block = p4.get("global") or {}
                    coh = glob_block.get("coherence")
                    if coh not in (0, None):
                        # already populated -- skip
                        continue
                    key = (args.judge_alias, tm, did, track)
                    raw = raw_index.get(key)
                    if raw is None:
                        no_raw += 1
                        continue
                    m = _CONSISTENCY_RE.search(raw)
                    if not m:
                        no_match += 1
                        continue
                    glob_block["coherence"] = int(m.group(1))
                    p4["global"] = glob_block
                    patched += 1
                    by_model[tm] += 1

    samples_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    print()
    print(f"Patched {patched} cells; {no_raw} missing raw; {no_match} raw without 'consistency'.")
    for m, n in sorted(by_model.items()):
        print(f"  {m}: {n}")
    print(f"Updated {samples_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
