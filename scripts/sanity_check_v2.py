"""Sanity check v2: read Tables 2 & 4 from the paper directly and
diff against the CSVs (no hardcoded paper-claim values).
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

PAPER = Path("C:/Users/arnol/Downloads/PCAIDE_PAPERS_AITHOS (6)/WaltonSchemes/main_AI_drafted.tex")
RUN = Path("logs/deliberative_justification_v2/20260527_combined_panels_curation/reasoning")

DISPLAY_TO_KEY = {
    "Qwen 3.6 Plus": "qwen3p6-plus",
    "DeepSeek V4 Pro": "deepseek-v4-pro",
    "Gemini 2.5 Pro": "gemini-2.5-pro",
    "GPT 5.4": "gpt-5.4",
    "GLM 5.1": "glm-5p1",
    "Kimi K2.6": "kimi-k2p6",
    "Claude Sonnet 4.6": "claude-sonnet-4-6",
    "Mistral Large 2512": "mistral-large-2512",
    "Llama 3.3": "Llama 3.3",
}


def load_csvs():
    with (RUN / "per_model_metrics.csv").open("r", encoding="utf-8") as f:
        metrics = {r["model"]: r for r in csv.DictReader(f)}
    fr = {}
    with (RUN / "failure_room_summary.csv").open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if ":" not in r["category"]:
                continue
            k, m = r["category"].split(":", 1)
            fr[(m, k, r["scope"])] = float(r["pct"])
    return metrics, fr


def extract_numbers(line: str) -> list[float]:
    """Pull every percentage or decimal number from a LaTeX table row."""
    nums = re.findall(r"\d+\.\d+", line)
    return [float(x) for x in nums]


def main():
    paper = PAPER.read_text(encoding="utf-8")
    metrics, fr = load_csvs()
    issues = []

    for display, key in DISPLAY_TO_KEY.items():
        # Match LaTeX table rows that start with the model name and `&`
        # The model name may be surrounded by trailing whitespace; check both
        # Table 2 (4 numbers) and Table 4 (5 numbers).
        for line in paper.splitlines():
            stripped = line.strip()
            if not stripped.startswith(display):
                continue
            # only count rows that are LaTeX table data (have at least one &)
            if "&" not in line:
                continue
            nums = extract_numbers(line)
            # Skip the model-name occurrences in Table 1 (which has stderr
            # entries like .004 — too many numbers to interpret here).
            # Use the count of `&` to discriminate:
            #   Table 2 row: 4 ampersands → 4 nums
            #   Table 4 row: 5 ampersands → 5 nums (incl scheme-cons %)
            #   Table 1 / Table 3 / Wilcoxon: many ampersands
            n_amp = line.count("&")
            if n_amp == 4 and len(nums) == 4:
                # Table 2: G_fail, G_room, C_fail, C_room
                gf, gr, cf, cr = nums
                expected = (fr[(key, "failure", "global")],
                            fr[(key, "room", "global")],
                            fr[(key, "failure", "coherence")],
                            fr[(key, "room", "coherence")])
                mis = [(name, p, c) for (name, p, c) in zip(("G_fail","G_room","C_fail","C_room"), (gf,gr,cf,cr), expected) if abs(p - c) > 0.05]
                if mis:
                    print(f"Table 2 :: {display:22s}  {[(n, p, c) for n,p,c in mis]}")
                    issues.extend(mis)
                else:
                    print(f"Table 2 :: {display:22s}  ok  ({gf},{gr},{cf},{cr})")
            elif n_amp == 5 and len(nums) == 5:
                # Table 4: norm.quality, coherence, scheme-cons%, any-fail%, global-fail%
                nq, coh, sc, anyf, gf = nums
                L = float(metrics[key]["mean_normalized_local_quality"])
                G = float(metrics[key]["mean_normalized_global_quality"])
                csv_nq = round((L+G)/2, 2)
                csv_coh = round(float(metrics[key]["mean_normalized_coherence"]), 2)
                csv_any = fr[(key, "failure", "any")]
                csv_gf = fr[(key, "failure", "global")]
                mis = []
                if abs(nq - csv_nq) > 0.01: mis.append(("NQ", nq, csv_nq))
                if abs(coh - csv_coh) > 0.01: mis.append(("Coh", coh, csv_coh))
                if abs(anyf - csv_any) > 0.15: mis.append(("AnyF", anyf, csv_any))
                if abs(gf - csv_gf) > 0.15: mis.append(("GF", gf, csv_gf))
                if mis:
                    print(f"Table 4 :: {display:22s}  {mis}")
                    issues.extend(mis)
                else:
                    print(f"Table 4 :: {display:22s}  ok  ({nq},{coh},{sc},{anyf},{gf})")
    print()
    print(f"== total mismatches: {len(issues)} ==")


if __name__ == "__main__":
    main()
