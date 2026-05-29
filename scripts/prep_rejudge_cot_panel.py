"""Flip the CoT-panel Phase-4 cells from gpt_5_5 to gpt_5_4 + skipped.

The CoT-panel models (Mistral Large 2512, Llama 3.3) were originally judged
by GPT 5.5 while the rest of the 9-model panel was judged by GPT 5.4.
This script prepares samples.json for a rejudging pass with GPT 5.4:

  - Every Phase-4 entry on a CoT-panel model with
    ``phase4_judge == "gpt_5_5"`` is renamed to ``phase4_judge: "gpt_5_4"``
    and marked ``skipped: True`` with reason ``rejudge_for_panel_consistency``.
  - ``local``, ``global`` and ``judge_reasoning`` are cleared so the
    refill script writes the GPT-5.4 result into a clean slot.

After this prep, run ``scripts/refill_phase4_judge.py --judge-alias gpt_5_4
--judge-model openai_gpt54`` to actually rejudge.

A backup of samples.json is written before any mutation.

Usage:
    python scripts/prep_rejudge_cot_panel.py \\
        logs/deliberative_justification_v2/20260527_combined_panels_curation/reasoning

For audit, the original gpt_5_5 scores live in the .bak file produced by
this script (and the secondary .bak produced by refill_phase4_judge).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


COT_PANEL_MODELS = {"mistral-large-2512", "Llama 3.3"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path,
                   help="run directory containing samples.json")
    p.add_argument("--from-judge", default="gpt_5_5",
                   help="phase4_judge alias to flip (default: gpt_5_5)")
    p.add_argument("--to-judge", default="gpt_5_4",
                   help="phase4_judge alias to set (default: gpt_5_4)")
    p.add_argument("--dry-run", action="store_true",
                   help="Count and list cells that would be flipped, no writes")
    args = p.parse_args(argv)

    samples_path = args.run_dir / "samples.json"
    if not samples_path.exists():
        raise SystemExit(f"samples.json not found at {samples_path}")
    samples = json.loads(samples_path.read_text(encoding="utf-8"))

    # Walk and collect work
    n_flipped = 0
    by_model_track = {}
    for dil in samples["dilemmas"]:
        for tg in dil["targets"]:
            m = tg["model"].split("/")[-1]
            if m not in COT_PANEL_MODELS:
                continue
            for tk in tg.get("tracks", []) or []:
                track = tk.get("text_kind")
                for p4 in tk.get("phase4", []) or []:
                    if p4.get("phase4_judge") != args.from_judge:
                        continue
                    n_flipped += 1
                    by_model_track[(m, track)] = by_model_track.get((m, track), 0) + 1
                    if args.dry_run:
                        continue
                    # Flip the judge alias, mark skipped, clear scores.
                    p4["phase4_judge"] = args.to_judge
                    p4["skipped"] = True
                    p4["skipped_reason"] = "rejudge_for_panel_consistency"
                    p4["local"] = {}
                    p4["global"] = {}
                    p4["judge_reasoning"] = ""

    print(f"cells to flip: {n_flipped}")
    for (m, t), n in sorted(by_model_track.items()):
        print(f"  {m:30s} {t:15s} {n}")

    if args.dry_run:
        print("\n--dry-run: no file written.")
        return 0
    if n_flipped == 0:
        print("nothing to flip; samples.json untouched.")
        return 0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = samples_path.with_name(f"samples.bak_pre_rejudge_{ts}.json")
    shutil.copy2(samples_path, bak)
    print(f"backup: {bak}")
    samples_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    print(f"updated: {samples_path}")
    print()
    print("Next step — run the refill (≈798 GPT-5.4 API calls):")
    print(f"  python scripts/refill_phase4_judge.py \\")
    print(f"      --cot-dir {args.run_dir} \\")
    print(f"      --judge-alias {args.to_judge} \\")
    print(f"      --judge-model openai_gpt54")
    return 0


if __name__ == "__main__":
    sys.exit(main())
