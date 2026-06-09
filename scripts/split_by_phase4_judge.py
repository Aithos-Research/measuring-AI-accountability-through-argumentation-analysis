"""Split a samples.json into per-Phase-4-judge subsets and run the analyser
+ plotter + report on each.

For each Phase-4 judge alias present in the source samples.json, this writes
a filtered samples.json containing only that judge's Phase-4 entries (per
track), keeping the Phase-2 entries intact.  The result lands in a sibling
directory under the parent of ``--cot-dir`` so all three views
(combined / sonnet-only / gpt-only) sit next to each other.

Usage:

    .\\.venv\\Scripts\\python.exe scripts/split_by_phase4_judge.py \\
        --cot-dir logs/deliberative_justification_v2/20260520_combined/cot
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cot-dir", required=True, type=Path,
                   help="cot directory containing samples.json (combined run).")
    p.add_argument("--run-analysis", action="store_true", default=True,
                   help="After writing each subset, run walton_analysis + walton_plots + walton_report.")
    p.add_argument("--no-run-analysis", dest="run_analysis", action="store_false")
    args = p.parse_args(argv)

    src = args.cot_dir / "samples.json"
    if not src.exists():
        raise SystemExit(f"missing samples.json: {src}")
    samples = json.loads(src.read_text(encoding="utf-8"))

    # Enumerate every Phase-4 judge alias that appears
    judges: set[str] = set()
    for dil in samples["dilemmas"]:
        for tg in dil["targets"]:
            for tk in tg["tracks"]:
                for p4 in tk.get("phase4") or []:
                    j = p4.get("phase4_judge")
                    if j:
                        judges.add(j)
    print(f"Found Phase-4 judges: {sorted(judges)}")

    # Parent of cot dir = where we'll create sibling directories
    parent = args.cot_dir.parent.parent  # e.g. .../20260520_combined  → .../
    base_label = args.cot_dir.parent.name  # e.g. "20260520_combined"
    if base_label.endswith("_combined"):
        base_label = base_label[: -len("_combined")]

    output_dirs: list[Path] = []
    for j in sorted(judges):
        # Sibling dir keyed by judge alias
        out_dir = parent / f"{base_label}_by_judge_{j}" / "cot"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Deep-copy then filter Phase-4 to just this judge
        sub = copy.deepcopy(samples)
        kept = 0
        for dil in sub["dilemmas"]:
            for tg in dil["targets"]:
                for tk in tg["tracks"]:
                    new_p4 = [p for p in (tk.get("phase4") or []) if p.get("phase4_judge") == j]
                    tk["phase4"] = new_p4
                    kept += len(new_p4)
        # Update summary
        if isinstance(sub.get("summary"), dict):
            sub["summary"]["judges_phase4"] = [j]
            sub["summary"]["n_records_subset"] = kept
        (out_dir / "samples.json").write_text(
            json.dumps(sub, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        print(f"  wrote {kept} Phase-4 cells → {out_dir}/samples.json")
        output_dirs.append(out_dir)

    if not args.run_analysis:
        return 0

    # Run analyser + plotter + report on each subset
    for d in output_dirs:
        print(f"\n=== running analysis on {d} ===")
        env_args = ["-X", "utf8"]
        cmds = [
            ["moral_eval.analysis.walton_analysis", str(d)],
            ["moral_eval.analysis.walton_plots", str(d)],
            ["moral_eval.analysis.walton_report", str(d)],
        ]
        for module, *rest in cmds:
            res = subprocess.run(
                [sys.executable, *env_args, "-m", module, *rest],
                cwd=PROJECT_ROOT,
                capture_output=True, text=True, encoding="utf-8",
            )
            if res.returncode != 0:
                print(f"  ERROR running {module}: rc={res.returncode}")
                print(res.stderr[-800:])
            else:
                # Show only the last line of stdout (the "wrote: ..." summary)
                last = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else ""
                print(f"  {module}: {last}")

    print()
    print("Done.  Compare results:")
    print(f"  combined : {args.cot_dir}/report.md")
    for d in output_dirs:
        print(f"  subset    : {d}/report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
