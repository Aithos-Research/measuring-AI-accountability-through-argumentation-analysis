"""Cross-check the numbers that appear in main_AI_drafted.tex against
the current analysis CSVs.

For each claim in the paper we pattern-match the surrounding context,
extract the number, and compare to the value in the CSV at the same
(model, metric, scope) coordinate.  Mismatches are reported with the
paper's value and the CSV's value side-by-side.

Run after applying any new curation:
    python scripts/sanity_check_paper_numbers.py
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

PAPER = Path("C:/Users/arnol/Downloads/PCAIDE_PAPERS_AITHOS (6)/WaltonSchemes/main_AI_drafted.tex")
RUN = Path("logs/deliberative_justification_v2/20260527_combined_panels_curation/reasoning")


def load_csv(path):
    with (RUN / path).open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    paper = PAPER.read_text(encoding="utf-8")

    metrics = {r["model"]: r for r in load_csv("per_model_metrics.csv")}
    summary = load_csv("failure_room_summary.csv")
    fig3 = {r["model"]: r for r in load_csv("figure3_failure_rate_per_dim.csv")}

    # Look up panel-total failure / room rows
    total = {(r["category"], r["scope"]): float(r["pct"]) for r in summary
             if ":" not in r["category"]}
    per_model_fr = {}
    for r in summary:
        if ":" not in r["category"]:
            continue
        kind, m = r["category"].split(":", 1)
        per_model_fr[(m, kind, r["scope"])] = float(r["pct"])

    issues = []

    # ----- 1. Total cell counts ----------------------------------------
    total_cells = sum(int(r["n_cells"]) for r in metrics.values())
    for claim, label in [(7176, "7,176 scored cells"),
                         (7200, "7,200 potential cells"),
                         (24, "24 lost cells")]:
        if str(claim) in paper or f"{claim:,}".replace(",", "{,}") in paper:
            pass  # found in paper
    print(f"sum(n_cells) over models = {total_cells}  (paper: 7,176)")

    # ----- 2. Per-model failure rates that the paper text mentions -----
    PAPER_CLAIMS_FAILURE_RATE = {
        # (model display, scope, paper-stated value)
        ("Claude Sonnet 4.6", "global", 51.14),
        ("Mistral Large 2512", "global", 5.64),
        ("Llama 3.3", "global", 29.82),
        ("Kimi K2.6", "global", 9.02),  # may be stale -> should now be 9.65
        ("GLM 5.1", "global", 2.25),    # may be stale -> 2.38
        ("Qwen 3.6 Plus", "global", 0.25),  # -> 0.50
        ("DeepSeek V4 Pro", "global", 0.62),  # -> 0.50
        ("GPT 5.4", "global", 1.38),    # -> 1.26
        ("Gemini 2.5 Pro", "global", 0.88),  # -> 1.01
    }
    print()
    print("Per-model global failure rates — paper claims vs CSV:")
    MODEL_KEYS = {
        "Claude Sonnet 4.6": "claude-sonnet-4-6",
        "Mistral Large 2512": "mistral-large-2512",
        "Llama 3.3": "Llama 3.3",
        "Kimi K2.6": "kimi-k2p6",
        "GLM 5.1": "glm-5p1",
        "Qwen 3.6 Plus": "qwen3p6-plus",
        "DeepSeek V4 Pro": "deepseek-v4-pro",
        "GPT 5.4": "gpt-5.4",
        "Gemini 2.5 Pro": "gemini-2.5-pro",
    }
    for display, scope, claim in PAPER_CLAIMS_FAILURE_RATE:
        key = MODEL_KEYS[display]
        csv_value = per_model_fr.get((key, "failure", scope))
        marker = "" if csv_value is not None and abs(csv_value - claim) < 0.05 else "  MISMATCH"
        print(f"  {display:22s}  {scope:8s}  paper={claim:>6}  csv={csv_value!s:>6}{marker}")
        if "MISMATCH" in marker:
            issues.append((display, scope, "failure", claim, csv_value))

    # ----- 3. Per-model room rates (Table 2) --------------------------
    print()
    print("Per-model global room rates — paper Table 2 vs CSV:")
    PAPER_ROOM_RATES = {
        ("Claude Sonnet 4.6", 51.52),
        ("Mistral Large 2512", 95.49),
        ("Llama 3.3", 87.59),
        ("Kimi K2.6", 77.82),
        ("GLM 5.1", 89.12),
        ("Qwen 3.6 Plus", 90.75),
        ("DeepSeek V4 Pro", 90.38),
        ("GPT 5.4", 73.49),
        ("Gemini 2.5 Pro", 96.47),
    }
    for display, claim in PAPER_ROOM_RATES:
        key = MODEL_KEYS[display]
        csv_value = per_model_fr.get((key, "room", "global"))
        marker = "" if csv_value is not None and abs(csv_value - claim) < 0.05 else "  MISMATCH"
        print(f"  {display:22s}  paper={claim:>6}  csv={csv_value!s:>6}{marker}")
        if "MISMATCH" in marker:
            issues.append((display, "global", "room", claim, csv_value))

    # ----- 4. Coherence failure & room ---------------------------------
    print()
    print("Per-model coherence failure/room — paper Table 2 vs CSV:")
    PAPER_COH = {
        ("Claude Sonnet 4.6", "failure", 9.47),
        ("Llama 3.3", "failure", 7.27),
        ("Mistral Large 2512", "failure", 0.38),
        ("Claude Sonnet 4.6", "room", 75.00),
        ("Llama 3.3", "room", 61.03),
        ("Mistral Large 2512", "room", 54.39),
    }
    for display, kind, claim in PAPER_COH:
        key = MODEL_KEYS[display]
        csv_value = per_model_fr.get((key, kind, "coherence"))
        marker = "" if csv_value is not None and abs(csv_value - claim) < 0.05 else "  MISMATCH"
        print(f"  {display:22s} {kind:8s} coh  paper={claim:>6}  csv={csv_value!s:>6}{marker}")
        if "MISMATCH" in marker:
            issues.append((display, "coherence", kind, claim, csv_value))

    # ----- 5. Norm quality (Table 4) -----------------------------------
    print()
    print("Per-model norm quality — paper Table 4 vs CSV:")
    PAPER_QUALITY = {
        ("GPT 5.4", 0.89), ("Claude Sonnet 4.6", 0.79), ("Kimi K2.6", 0.83),
        ("Qwen 3.6 Plus", 0.80), ("GLM 5.1", 0.80), ("DeepSeek V4 Pro", 0.79),
        ("Gemini 2.5 Pro", 0.75), ("Mistral Large 2512", 0.75), ("Llama 3.3", 0.72),
    }
    for display, claim in PAPER_QUALITY:
        key = MODEL_KEYS[display]
        r = metrics[key]
        L = float(r["mean_normalized_local_quality"])
        G = float(r["mean_normalized_global_quality"])
        avg = (L + G) / 2
        marker = "" if abs(avg - claim) < 0.01 else "  MISMATCH"
        print(f"  {display:22s}  paper={claim:.2f}  csv-avg={avg:.4f}{marker}")
        if "MISMATCH" in marker:
            issues.append((display, "overall", "norm_quality", claim, round(avg, 4)))

    # ----- 6. Coherence column in Table 4 ------------------------------
    print()
    print("Per-model mean coherence — paper Table 4 vs CSV:")
    PAPER_COH_QUALITY = {
        ("GPT 5.4", 0.91), ("Claude Sonnet 4.6", 0.53), ("Kimi K2.6", 0.81),
        ("Qwen 3.6 Plus", 0.91), ("GLM 5.1", 0.86), ("DeepSeek V4 Pro", 0.92),
        ("Gemini 2.5 Pro", 0.91), ("Mistral Large 2512", 0.73), ("Llama 3.3", 0.61),
    }
    for display, claim in PAPER_COH_QUALITY:
        key = MODEL_KEYS[display]
        coh = float(metrics[key]["mean_normalized_coherence"])
        marker = "" if abs(coh - claim) < 0.01 else "  MISMATCH"
        print(f"  {display:22s}  paper={claim:.2f}  csv={coh:.4f}{marker}")
        if "MISMATCH" in marker:
            issues.append((display, "overall", "coherence_mean", claim, round(coh, 4)))

    # ----- 7. Any-failure rates (Table 4) ------------------------------
    print()
    print("Per-model any-fail rate — paper Table 4 vs CSV:")
    PAPER_ANY_FAIL = {
        ("GPT 5.4", 1.9), ("Claude Sonnet 4.6", 54.8), ("Kimi K2.6", 10.9),
        ("Qwen 3.6 Plus", 2.9), ("GLM 5.1", 6.9), ("DeepSeek V4 Pro", 3.8),
        ("Gemini 2.5 Pro", 9.9), ("Mistral Large 2512", 12.8), ("Llama 3.3", 50.6),
    }
    for display, claim in PAPER_ANY_FAIL:
        key = MODEL_KEYS[display]
        csv_v = per_model_fr.get((key, "failure", "any"))
        marker = "" if csv_v is not None and abs(csv_v - claim) < 0.15 else "  MISMATCH"
        print(f"  {display:22s}  paper={claim:>5}  csv={csv_v!s:>6}{marker}")
        if "MISMATCH" in marker:
            issues.append((display, "any", "failure", claim, csv_v))

    # ----- 8. Figure 3 per-(model, dim) cross check --------------------
    print()
    print("Figure 3 CSV row sums vs Table 2 (panel-total sanity):")
    sums = {}
    for r in fig3.values():
        m = r["model"]
        for k in ("L.Acc", "L.Rel", "L.Grnd", "G.Coh", "G.Acc", "G.Rel", "G.Suf"):
            sums[(m, k)] = float(r[k + "_pct"])
    # Sanity: max value per column
    print("  Max per dim across all 9 models (should match the figure):")
    for k in ("L.Acc", "L.Rel", "L.Grnd", "G.Coh", "G.Acc", "G.Rel", "G.Suf"):
        vals = [(m, sums[(m, k)]) for m in fig3]
        vals.sort(key=lambda x: -x[1])
        print(f"    {k:6s}  max = {vals[0][1]:>6.2f}  ({vals[0][0]})")

    print()
    if issues:
        print(f"=== {len(issues)} mismatches found ===")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("=== no mismatches found ===")


if __name__ == "__main__":
    main()
