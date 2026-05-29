"""Merge two dj_v2 runs into a single samples.json.

The base run supplies most of the cot targets; the override run supplies one
or more specific target models (typically a re-run after the original errored
out partway through).  For each dilemma, this script:

1. Takes the dilemma block from the base run.
2. Removes any target whose ``model`` appears in the override run's target
   set (the base data is stale / incomplete for those).
3. Appends the override run's target entries for that dilemma.

The merged samples.json is written to a fresh ``cot`` directory.  After it
lands, run the walton analyser + plotter against the merged dir to regenerate
all CSVs and PNGs.

Usage:

    .\\.venv\\Scripts\\python.exe scripts/merge_runs.py \\
        --base logs/deliberative_justification_v2/20260520_103147/cot \\
        --override logs/deliberative_justification_v2/20260520_142103/cot \\
        --out logs/deliberative_justification_v2/20260520_combined/cot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(samples_path: Path) -> dict:
    if not samples_path.exists():
        raise SystemExit(f"missing samples.json: {samples_path}")
    return json.loads(samples_path.read_text(encoding="utf-8"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True, type=Path,
                   help="cot directory of the base run.")
    p.add_argument("--override", required=True, type=Path,
                   help="cot directory of the override run (supplies one or "
                        "more target models that should replace the base's).")
    p.add_argument("--out", required=True, type=Path,
                   help="Output cot directory.  samples.json is written here.")
    p.add_argument("--additive", action="store_true",
                   help="Per-dilemma additive merge: only replace a base "
                        "target if the override actually has data for that "
                        "(dilemma, model) pair.  Use for partial restarts "
                        "(e.g. a quota-error re-run that only covers the "
                        "dilemmas the base didn't reach) so the base's "
                        "successful entries are preserved.")
    args = p.parse_args(argv)

    base = _load(args.base / "samples.json")
    override = _load(args.override / "samples.json")

    # Models the override run is authoritative for
    override_models: set[str] = set()
    for dil in override["dilemmas"]:
        for tg in dil["targets"]:
            override_models.add(tg["model"])
    print(f"override authoritative for {len(override_models)} model(s):")
    for m in sorted(override_models):
        print(f"  - {m}")

    # Index override dilemmas by id for fast lookup
    override_by_id: dict[str, dict] = {dil["dilemma_id"]: dil for dil in override["dilemmas"]}

    merged_dilemmas: list[dict] = []
    base_ids = [dil["dilemma_id"] for dil in base["dilemmas"]]
    skipped_no_override = 0
    replaced_targets = 0
    added_targets = 0

    for base_dil in base["dilemmas"]:
        did = base_dil["dilemma_id"]
        over_dil = override_by_id.get(did)
        # Models the override actually has data for *on this dilemma*.  Used
        # only in --additive mode to decide whether to replace or keep each
        # base target.
        override_models_for_dil: set[str] = set()
        if over_dil is not None:
            for tg in over_dil.get("targets") or []:
                if tg["model"] in override_models:
                    override_models_for_dil.add(tg["model"])
        # Start with a shallow copy + new targets list
        merged_dil = {
            "dilemma_id": did,
            "dilemma": base_dil.get("dilemma"),
            "targets": [],
        }
        # Carry over base targets EXCEPT those the override is authoritative for
        for tg in base_dil.get("targets") or []:
            should_replace = (
                tg["model"] in override_models_for_dil
                if args.additive
                else tg["model"] in override_models
            )
            if should_replace:
                replaced_targets += 1
                continue
            merged_dil["targets"].append(tg)
        # Append override targets for this dilemma (if present)
        if over_dil is None:
            skipped_no_override += 1
        else:
            for tg in over_dil.get("targets") or []:
                if tg["model"] in override_models:
                    merged_dil["targets"].append(tg)
                    added_targets += 1
        merged_dilemmas.append(merged_dil)

    # Also include any dilemmas that ONLY exist in the override run (unlikely
    # but handled defensively)
    base_ids_set = set(base_ids)
    for did, over_dil in override_by_id.items():
        if did in base_ids_set:
            continue
        merged_dilemmas.append({
            "dilemma_id": did,
            "dilemma": over_dil.get("dilemma"),
            "targets": [tg for tg in (over_dil.get("targets") or [])],
        })

    # Build the merged document
    merged = {
        "task": base.get("task") or override.get("task"),
        "schema_version": base.get("schema_version", 3),
        "summary": {
            "merged_from": [str(args.base), str(args.override)],
            "override_models": sorted(override_models),
            "n_dilemmas": len(merged_dilemmas),
            "n_target_dilemma_pairs": sum(len(d["targets"]) for d in merged_dilemmas),
        },
        "text_kind_descriptions": base.get("text_kind_descriptions")
                                  or override.get("text_kind_descriptions"),
        "dilemmas": merged_dilemmas,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "samples.json"
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print()
    print(f"replaced {replaced_targets} stale base targets")
    print(f"added {added_targets} authoritative override targets")
    if skipped_no_override:
        print(f"WARN: {skipped_no_override} dilemma(s) had no matching dilemma in the override run")
    print(f"wrote merged samples.json to {out_path}")
    print(f"  n_dilemmas: {merged['summary']['n_dilemmas']}")
    print(f"  n_target_dilemma_pairs: {merged['summary']['n_target_dilemma_pairs']}")
    print()
    print("Next: run the walton analyser + plotter on the merged dir:")
    print(f"  .\\.venv\\Scripts\\python.exe -m moral_eval.analysis.walton_analysis {args.out}")
    print(f"  .\\.venv\\Scripts\\python.exe -m moral_eval.analysis.walton_plots {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
