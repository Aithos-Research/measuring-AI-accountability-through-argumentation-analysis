"""Recompute every paper number from the post-refill samples.json.

Outputs hedge analysis, per-model hedge rates, inter-judge agreement
(Claude Sonnet 4.6 vs GPT 5.4 only), and the per-model dimension
disagreement table — everything the rewrite needs.
"""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import chi2_contingency, mannwhitneyu, wilcoxon


SAMPLES = Path("logs/deliberative_justification_v2/20260527_combined_panels_curation/reasoning/samples.json")

SCALE_MIN, SCALE_MAX = 1, 3
RANGE = SCALE_MAX - SCALE_MIN
HEDGE_RE = re.compile(
    r"\b(?:might|may|perhaps|possibly|maybe|could|seem|seems|seemingly|"
    r"arguably|somewhat|sometimes|often|usually|generally|typically|tend|"
    r"tends|i think|i believe|uncertain|unclear|depend|depends|to some "
    r"extent|to a degree|not necessarily|in some cases|in many cases|not "
    r"always|on the other hand|however|although|though|whereas|conceivably|"
    r"presumably)\b",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"\b\w+\b")

ORDER = [
    "gpt-5.4", "claude-sonnet-4-6", "kimi-k2p6", "qwen3p6-plus",
    "glm-5p1", "deepseek-v4-pro", "gemini-2.5-pro",
    "mistral-large-2512", "Llama 3.3",
]


def norm(x):
    return (x - SCALE_MIN) / RANGE


def bucket(v):
    if v < 0.5:
        return "LOW"
    if v < 0.8:
        return "MID"
    return "HIGH"


def main():
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))

    # ====================================================================
    # Hedge analysis at the (cell, judge) level
    # ====================================================================
    rows = []
    for dil in samples["dilemmas"]:
        for tg in dil["targets"]:
            for tk in tg.get("tracks", []) or []:
                p2 = (tk.get("phase2", []) or [None])[0]
                if not p2:
                    continue
                defence = p2.get("defense") or ""
                if not defence:
                    continue
                words = WORD_RE.findall(defence)
                if not words:
                    continue
                n_w = len(words)
                n_h = len(HEDGE_RE.findall(defence))
                hr = n_h / n_w
                for p4 in tk.get("phase4", []) or []:
                    if p4.get("skipped"):
                        continue
                    grnd = [
                        dims.get("grounds")
                        for q, dims in (p4.get("local") or {}).items()
                        if isinstance(dims, dict) and isinstance(dims.get("grounds"), int)
                    ]
                    suf = (p4.get("global") or {}).get("sufficiency")
                    if not grnd or not isinstance(suf, int):
                        continue
                    rows.append({
                        "hedge_rate": hr, "word_count": n_w,
                        "bucket_g": bucket(norm(sum(grnd) / len(grnd))),
                        "bucket_s": bucket(norm(suf)),
                    })

    print(f"total scored cells: {len(rows)}")
    gb = Counter(r["bucket_g"] for r in rows)
    print()
    print("=== Local-grounds buckets ===")
    for b in ["LOW", "MID", "HIGH"]:
        xs = [r for r in rows if r["bucket_g"] == b]
        wc = [r["word_count"] for r in xs]
        hr = [r["hedge_rate"] for r in xs]
        print(f"  {b}: n={len(xs)}, med_words={statistics.median(wc):.0f}, med_hedge={statistics.median(hr):.4f}")
    low_g = [r["hedge_rate"] for r in rows if r["bucket_g"] == "LOW"]
    mh_g = [r["hedge_rate"] for r in rows if r["bucket_g"] in ("MID", "HIGH")]
    u, p = mannwhitneyu(low_g, mh_g, alternative="greater")
    print(f"  MW U (LOW vs MID+HIGH): p={p:.2e}")

    print()
    print("=== Global-sufficiency buckets ===")
    for b in ["LOW", "MID", "HIGH"]:
        xs = [r for r in rows if r["bucket_s"] == b]
        hr = [r["hedge_rate"] for r in xs]
        print(f"  {b}: n={len(xs)}, med_hedge={statistics.median(hr):.4f}")
    low_s = [r["hedge_rate"] for r in rows if r["bucket_s"] == "LOW"]
    mh_s = [r["hedge_rate"] for r in rows if r["bucket_s"] in ("MID", "HIGH")]
    u, p = mannwhitneyu(low_s, mh_s, alternative="greater")
    print(f"  MW U (LOW vs MID+HIGH): p={p:.2e}")

    # ====================================================================
    # Per-model hedge rates (unique defences)
    # ====================================================================
    print()
    print("=== Per-model hedge rate ===")
    per_m = defaultdict(list)
    seen = set()
    for dil in samples["dilemmas"]:
        for tg in dil["targets"]:
            m = tg["model"].split("/")[-1]
            for tk in tg.get("tracks", []) or []:
                track = tk.get("text_kind")
                p2 = (tk.get("phase2", []) or [None])[0]
                if not p2:
                    continue
                defence = p2.get("defense") or ""
                if not defence:
                    continue
                key = (m, dil["dilemma_id"], track)
                if key in seen:
                    continue
                seen.add(key)
                words = WORD_RE.findall(defence)
                if not words:
                    continue
                per_m[m].append({
                    "n_words": len(words),
                    "hedge_rate": len(HEDGE_RE.findall(defence)) / len(words),
                })

    for m in ORDER:
        xs = per_m[m]
        if not xs:
            continue
        wc = [r["n_words"] for r in xs]
        hr = [r["hedge_rate"] for r in xs]
        print(f"  {m:25s} n={len(xs):4d}  med_words={statistics.median(wc):4.0f}  med_hedge={statistics.median(hr):.4f}")

    # Claude vs rest
    claude_hr = [r["hedge_rate"] for r in per_m["claude-sonnet-4-6"]]
    rest_hr = []
    for m in ORDER:
        if m == "claude-sonnet-4-6":
            continue
        rest_hr.extend([r["hedge_rate"] for r in per_m[m]])
    u, p = mannwhitneyu(claude_hr, rest_hr, alternative="greater")
    print(f"  Claude vs rest of panel: p={p:.2e}")

    # ====================================================================
    # Inter-judge agreement (Claude vs GPT 5.4 only)
    # ====================================================================
    print()
    print("=== Inter-judge agreement (Claude vs GPT 5.4 only) ===")
    total_points = 0
    dis_points = 0
    per_model_d = defaultdict(lambda: [0, 0])
    per_model_cd = defaultdict(int)
    per_model_cells = defaultdict(int)

    for dil in samples["dilemmas"]:
        for tg in dil["targets"]:
            m = tg["model"].split("/")[-1]
            for tk in tg.get("tracks", []) or []:
                by_j = {}
                for p4 in tk.get("phase4", []) or []:
                    if p4.get("skipped"):
                        continue
                    by_j[p4.get("phase4_judge")] = p4
                if "claude_sonnet_4_6" not in by_j or "gpt_5_4" not in by_j:
                    continue
                ja, jb = by_j["claude_sonnet_4_6"], by_j["gpt_5_4"]
                cell_dis = False
                for q in ja.get("local") or {}:
                    if q not in (jb.get("local") or {}):
                        continue
                    for d in ("acceptability", "relevance", "grounds"):
                        va = (ja["local"][q] or {}).get(d) if isinstance(ja["local"][q], dict) else None
                        vb = (jb["local"][q] or {}).get(d) if isinstance(jb["local"][q], dict) else None
                        if not isinstance(va, int) or not isinstance(vb, int):
                            continue
                        total_points += 1
                        per_model_d[m][1] += 1
                        if va != vb:
                            dis_points += 1
                            per_model_d[m][0] += 1
                            cell_dis = True
                for d in ("coherence", "acceptability", "relevance", "sufficiency"):
                    va = (ja.get("global") or {}).get(d)
                    vb = (jb.get("global") or {}).get(d)
                    if not isinstance(va, int) or not isinstance(vb, int):
                        continue
                    total_points += 1
                    per_model_d[m][1] += 1
                    if va != vb:
                        dis_points += 1
                        per_model_d[m][0] += 1
                        cell_dis = True
                per_model_cells[m] += 1
                if cell_dis:
                    per_model_cd[m] += 1

    print(f"  Total points: {total_points}, disagree: {dis_points} = {dis_points/total_points*100:.2f}%")
    print(f"  Total cells (with both judges): {sum(per_model_cells.values())}")
    print()
    print(f"  {'model':25s}  cells   cell_disagree  point_disagree")
    for m in ORDER:
        if not per_model_cells[m]:
            continue
        pc = per_model_cd[m] / per_model_cells[m] * 100
        pp = per_model_d[m][0] / per_model_d[m][1] * 100 if per_model_d[m][1] else 0
        print(f"  {m:25s}  {per_model_cells[m]:5d}    {pc:5.1f}%        {pp:.2f}%")

    # ====================================================================
    # Scheme totals + chi^2
    # ====================================================================
    print()
    print("=== Scheme distribution ===")
    sch = defaultdict(lambda: defaultdict(int))
    for dil in samples["dilemmas"]:
        for tg in dil["targets"]:
            for tk in tg.get("tracks", []) or []:
                track = tk.get("text_kind")
                p2 = (tk.get("phase2", []) or [None])[0]
                if not p2:
                    continue
                s = (p2.get("scheme") or "").strip()
                if not s:
                    continue
                sch[track][s] += 1

    top = sorted(set(sch["reasoning"]) | set(sch["justification"]),
                 key=lambda s: -(sch["reasoning"][s] + sch["justification"][s]))[:8]
    tot_r = sum(sch["reasoning"].values())
    tot_j = sum(sch["justification"].values())
    print(f"  total: r={tot_r}, j={tot_j}")
    print(f"  {'scheme':35s}  r       j       diff(pp)")
    for s in top:
        r = sch["reasoning"][s]; j = sch["justification"][s]
        rp = r/tot_r*100; jp = j/tot_j*100
        print(f"  {s:35s}  {r:4d}({rp:5.2f}%)  {j:4d}({jp:5.2f}%)  {jp-rp:+.2f}")


if __name__ == "__main__":
    main()
