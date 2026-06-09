"""Consolidate per_model_failure_by_dim.csv (long) + the existing wide
file into a single CSV that has both percentages and raw counts.

Output: figure3_failure_rate_per_dim.csv with columns
    model, n_cells, L.Acc_pct, L.Acc_n_fail, L.Rel_pct, L.Rel_n_fail, ...
sorted by L.Grnd ascending to match the heatmap row order.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


PRETTY = {
    "acceptability": "Acc",
    "relevance": "Rel",
    "grounds": "Grnd",
    "coherence": "Coh",
    "sufficiency": "Suf",
}
LOCAL_ORDER = ("acceptability", "relevance", "grounds")
GLOBAL_ORDER = ("coherence", "acceptability", "relevance", "sufficiency")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--remove-source", action="store_true",
                   help="Delete the source long-format CSV after merging.")
    args = p.parse_args(argv)

    src = args.run_dir / "per_model_failure_by_dim.csv"
    dst = args.run_dir / "figure3_failure_rate_per_dim.csv"
    if not src.exists():
        raise SystemExit(f"missing {src}")

    with src.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_model = {}
    totals = {}
    for r in rows:
        m = r["model"]
        col = ("L." if r["scope"] == "local" else "G.") + PRETTY[r["dim"]]
        by_model.setdefault(m, {})[col] = {
            "pct": float(r["pct"]),
            "n": int(r["n_cells_with_failure"]),
        }
        totals[m] = int(r["n_total"])

    cols_order = (
        ["L." + PRETTY[d] for d in LOCAL_ORDER]
        + ["G." + PRETTY[d] for d in GLOBAL_ORDER]
    )
    order = sorted(by_model.keys(), key=lambda m: by_model[m]["L.Grnd"]["pct"])

    header = ["model", "n_cells"]
    for c in cols_order:
        header.append(f"{c}_pct")
        header.append(f"{c}_n_fail")

    with dst.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for m in order:
            row = [m, totals[m]]
            for c in cols_order:
                row.append(f"{by_model[m][c]['pct']:.2f}")
                row.append(by_model[m][c]["n"])
            w.writerow(row)

    print(f"wrote {dst}")
    if args.remove_source:
        os.remove(src)
        print(f"removed {src}")


if __name__ == "__main__":
    raise SystemExit(main())
