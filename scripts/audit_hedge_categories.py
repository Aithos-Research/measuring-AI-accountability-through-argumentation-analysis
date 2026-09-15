"""Split the pooled hedge-marker regex (scripts/audit_rejudged_numbers.py)
into linguistically distinct sub-categories and re-run the Mann-Whitney
analysis per category, to see where the hedge-rate effect concentrates:

  1. epistemic modals      -- might, may, perhaps, possibly, maybe, could,
                               seem/seems/seemingly, arguably, conceivably,
                               presumably
  2. contrastive connectives -- however, although, though, whereas,
                                 on the other hand
  3. first-person doubt markers -- i think, i believe, uncertain, unclear,
                                    not necessarily
  4. frequency/scope hedges (residual, not asked for but present in the
     original dictionary) -- somewhat, sometimes, often, usually, generally,
     typically, tend/tends, depend/depends, to some extent, to a degree,
     in some/many cases, not always

Reuses the same samples.json and the same three splits as
audit_rejudged_numbers.py: local-grounds bucket, global-sufficiency bucket,
and Claude-sonnet-4-6 vs rest of panel.
"""
from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from scipy.stats import mannwhitneyu

SAMPLES = Path("logs/deliberative_justification_v2/20260527_combined_panels_curation/reasoning/samples.json")

SCALE_MIN, SCALE_MAX = 1, 3
RANGE = SCALE_MAX - SCALE_MIN
WORD_RE = re.compile(r"\b\w+\b")

CATEGORIES = {
    "epistemic_modals": re.compile(
        r"\b(?:might|may|perhaps|possibly|maybe|could|seem|seems|seemingly|"
        r"arguably|conceivably|presumably)\b",
        re.IGNORECASE,
    ),
    "contrastive_connectives": re.compile(
        r"\b(?:however|although|though|whereas|on the other hand)\b",
        re.IGNORECASE,
    ),
    "first_person_doubt": re.compile(
        r"\b(?:i think|i believe|uncertain|unclear|not necessarily)\b",
        re.IGNORECASE,
    ),
    "frequency_scope_other": re.compile(
        r"\b(?:somewhat|sometimes|often|usually|generally|typically|tend|"
        r"tends|depend|depends|to some extent|to a degree|not always|"
        r"in some cases|in many cases)\b",
        re.IGNORECASE,
    ),
}

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


def rates_for(defence, n_w):
    return {cat: len(rx.findall(defence)) / n_w for cat, rx in CATEGORIES.items()}


def main():
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))

    # ====================================================================
    # (cell, judge)-level rows, one hedge_rate per category
    # ====================================================================
    rows = []
    for dil in samples["dilemmas"]:
        for tg in dil["targets"]:
            m = tg["model"].split("/")[-1]
            for tk in tg.get("tracks", []) or []:
                if m == "gpt-5.4" and tk.get("text_kind") == "reasoning":
                    continue  # r_int is encrypted ciphertext, unscoreable
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
                cat_rates = rates_for(defence, n_w)
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
                        **cat_rates,
                        "bucket_g": bucket(norm(sum(grnd) / len(grnd))),
                        "bucket_s": bucket(norm(suf)),
                    })

    print(f"total scored cells: {len(rows)}")

    def mw_split(label, bucket_key):
        print()
        print(f"=== {label}: LOW vs MID+HIGH, per hedge category ===")
        for cat in CATEGORIES:
            low = [r[cat] for r in rows if r[bucket_key] == "LOW"]
            mh = [r[cat] for r in rows if r[bucket_key] in ("MID", "HIGH")]
            med_low, med_mh = statistics.median(low), statistics.median(mh)
            mean_low, mean_mh = statistics.mean(low), statistics.mean(mh)
            nz_low = sum(1 for x in low if x > 0) / len(low)
            nz_mh = sum(1 for x in mh if x > 0) / len(mh)
            u, p = mannwhitneyu(low, mh, alternative="two-sided")
            print(f"  {cat:25s} med={med_low:.4f}/{med_mh:.4f}  "
                  f"mean={mean_low:.4f}/{mean_mh:.4f}  "
                  f"%nonzero={nz_low*100:5.1f}/{nz_mh*100:5.1f}  p={p:.2e}")

    mw_split("Local-grounds buckets", "bucket_g")
    mw_split("Global-sufficiency buckets", "bucket_s")

    # ====================================================================
    # Per-model rates (unique defences) + Claude vs rest, per category
    # ====================================================================
    print()
    print("=== Per-model hedge rate, per category ===")
    per_m = defaultdict(list)
    seen = set()
    for dil in samples["dilemmas"]:
        for tg in dil["targets"]:
            m = tg["model"].split("/")[-1]
            for tk in tg.get("tracks", []) or []:
                track = tk.get("text_kind")
                if m == "gpt-5.4" and track == "reasoning":
                    continue  # r_int is encrypted ciphertext, unscoreable
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
                per_m[m].append(rates_for(defence, len(words)))

    for cat in CATEGORIES:
        print(f"  -- {cat} --")
        for m in ORDER:
            xs = per_m[m]
            if not xs:
                continue
            hr = [r[cat] for r in xs]
            print(f"    {m:25s} n={len(xs):4d}  med_hedge={statistics.median(hr):.4f}")

    print()
    print("=== Claude vs rest of panel, per category ===")
    for cat in CATEGORIES:
        claude_hr = [r[cat] for r in per_m["claude-sonnet-4-6"]]
        rest_hr = []
        for m in ORDER:
            if m == "claude-sonnet-4-6":
                continue
            rest_hr.extend([r[cat] for r in per_m[m]])
        nz_c = sum(1 for x in claude_hr if x > 0) / len(claude_hr)
        nz_r = sum(1 for x in rest_hr if x > 0) / len(rest_hr)
        u, p = mannwhitneyu(claude_hr, rest_hr, alternative="two-sided")
        print(f"  {cat:25s} med={statistics.median(claude_hr):.4f}/{statistics.median(rest_hr):.4f}  "
              f"mean={statistics.mean(claude_hr):.4f}/{statistics.mean(rest_hr):.4f}  "
              f"%nonzero={nz_c*100:5.1f}/{nz_r*100:5.1f}  p={p:.2e}")


if __name__ == "__main__":
    main()
