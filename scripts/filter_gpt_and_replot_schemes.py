"""Produce GPT-excluded versions of the scheme-related figures.

GPT 5.4's r_int is encrypted ciphertext; the Phase-2 scheme labels
produced against the opaque blob are not interpretable as
classifications of any reasoning content.  The H2 paired test, scheme
distribution, scheme transitions, and per-scheme local-failure-by-scheme
analyses must therefore exclude GPT 5.4 entirely.

This script:
  1. Loads samples.json from the current run dir
  2. Drops every target whose model is gpt-5.4
  3. Writes the filtered samples to a sibling subdirectory `_no_gpt/`
  4. Runs walton_analysis and walton_plots on the filtered dir
  5. Reports the new analysis dir; the caller is responsible for
     copying the relevant PNGs into the paper folder

Usage:
    python scripts/filter_gpt_and_replot_schemes.py \\
        logs/deliberative_justification_v2/20260527_combined_panels_curation/reasoning
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    args = p.parse_args(argv)

    samples_path = args.run_dir / "samples.json"
    if not samples_path.exists():
        raise SystemExit(f"missing samples.json at {samples_path}")
    samples = json.loads(samples_path.read_text(encoding="utf-8"))

    out_dir = args.run_dir / "_no_gpt"
    out_dir.mkdir(exist_ok=True)
    filtered_samples_path = out_dir / "samples.json"

    n_targets_before = 0
    n_targets_after = 0
    for dil in samples["dilemmas"]:
        before = len(dil["targets"])
        dil["targets"] = [
            tg for tg in dil["targets"]
            if tg["model"].split("/")[-1] != "gpt-5.4"
        ]
        n_targets_before += before
        n_targets_after += len(dil["targets"])

    filtered_samples_path.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"filtered {n_targets_before} -> {n_targets_after} target blocks "
        f"(dropped {n_targets_before - n_targets_after} GPT 5.4 targets)"
    )
    print(f"wrote: {filtered_samples_path}")

    # Run analysis + plots on the filtered dir
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    py = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        py = "python"  # fallback

    print()
    print("Running walton_analysis on filtered samples...")
    subprocess.run([
        str(py), "-m", "moral_eval.analysis.walton_analysis", str(out_dir),
    ], check=True, cwd=PROJECT_ROOT)

    print()
    print("Running walton_plots on filtered samples...")
    subprocess.run([
        str(py), "-m", "moral_eval.analysis.walton_plots", str(out_dir),
    ], check=True, cwd=PROJECT_ROOT)

    print()
    print(f"Done.  Figures in: {out_dir}")
    print("Relevant ones to copy into the paper folder:")
    print(f"  {out_dir / 'local_failure_room_by_scheme.png'}")
    print(f"  {out_dir / 'scheme_distribution.png'}")
    print(f"  {out_dir / 'scheme_transitions_heatmap.png'}")


if __name__ == "__main__":
    sys.exit(main())
