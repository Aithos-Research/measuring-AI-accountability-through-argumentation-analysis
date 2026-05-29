"""Recompute scheme-related stats excluding GPT 5.4 entirely.

Phase 2 classified GPT 5.4's encrypted r_int (Fernet ciphertext) anyway,
producing 200 noisy scheme labels per dilemma that propagate into the
scheme distribution, transitions, retention and chi^2 stats.  This
script drops GPT 5.4 and reports what each result becomes.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import chi2_contingency


SAMPLES = Path("logs/deliberative_justification_v2/20260527_combined_panels_curation/reasoning/samples.json")


def main():
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))

    schemes_int = Counter()
    schemes_ext = Counter()
    transitions = Counter()
    scheme_per_pair = {}
    for dil in samples["dilemmas"]:
        did = dil["dilemma_id"]
        for tg in dil["targets"]:
            m = tg["model"].split("/")[-1]
            if m == "gpt-5.4":
                continue
            ri = re_ = None
            for tk in tg.get("tracks", []) or []:
                track = tk.get("text_kind")
                for p2 in tk.get("phase2", []) or []:
                    s = (p2.get("scheme") or "").strip() or "(unspecified)"
                    if track == "reasoning":
                        schemes_int[s] += 1
                        ri = s
                    elif track == "justification":
                        schemes_ext[s] += 1
                        re_ = s
            if ri and re_:
                transitions[(ri, re_)] += 1
                scheme_per_pair[(m, did)] = (ri, re_)

    n_int = sum(schemes_int.values())
    n_ext = sum(schemes_ext.values())
    n_pairs = len(scheme_per_pair)

    print(f"n_int  (reasoning classifications, GPT excluded):     {n_int}")
    print(f"n_ext  (justification classifications, GPT excluded): {n_ext}")
    print(f"paired (model, dilemma) classifications:              {n_pairs}")

    top = sorted(set(schemes_int) | set(schemes_ext),
                 key=lambda s: -(schemes_int[s] + schemes_ext[s]))[:8]

    print()
    print("Marginal scheme distribution (top 8):")
    print(f"  {'scheme':<35}  {'r_int':>15}  {'r_ext':>15}  shift_pp")
    for s in top:
        ri = schemes_int[s]
        re_ = schemes_ext[s]
        rip = ri / n_int * 100
        rep = re_ / n_ext * 100
        print(f"  {s:<35}  {ri:>4} ({rip:5.2f}%)  {re_:>4} ({rep:5.2f}%)  {rep - rip:+.2f}")

    # chi^2 on schemes with mass >= 10 either side
    top_chi = [s for s in top if (schemes_int[s] + schemes_ext[s]) >= 10]
    ctab = [[schemes_int[s] for s in top_chi], [schemes_ext[s] for s in top_chi]]
    chi2, p, df, _ = chi2_contingency(ctab)
    print()
    print(f"chi^2 scheme x track (GPT excluded): chi2={chi2:.2f}, df={df}, p={p:.2e}")

    print()
    print("Per-scheme retention (GPT excluded):")
    for s in top:
        intpairs = [pp for pp in scheme_per_pair.values() if pp[0] == s]
        if not intpairs:
            continue
        same = sum(1 for pp in intpairs if pp[0] == pp[1])
        ret = same / len(intpairs) * 100
        print(f"  {s:<35}  retention = {same:>3}/{len(intpairs):<3} = {ret:5.1f}%")

    # VBPR vs other-scheme retention chi^2
    vbpr = [pp for pp in scheme_per_pair.values()
            if pp[0] == "value-based practical reasoning"]
    other = [pp for pp in scheme_per_pair.values()
             if pp[0] != "value-based practical reasoning"]
    vbpr_same = sum(1 for pp in vbpr if pp[0] == pp[1])
    other_same = sum(1 for pp in other if pp[0] == pp[1])
    chi2v, pv, _, _ = chi2_contingency([
        [vbpr_same, len(vbpr) - vbpr_same],
        [other_same, len(other) - other_same],
    ])
    print()
    print(f"VBPR retention:  {vbpr_same}/{len(vbpr)} = {vbpr_same / len(vbpr) * 100:.1f}%")
    print(f"Other retention: {other_same}/{len(other)} = {other_same / len(other) * 100:.1f}%")
    print(f"VBPR vs other:   chi2 = {chi2v:.2f}, p = {pv:.2e}")

    print()
    print("Per-scheme marginal shift chi^2 (GPT excluded):")
    for s in top:
        a, b = schemes_int[s], schemes_ext[s]
        c, d = n_int - a, n_ext - b
        try:
            chi2s, ps, _, _ = chi2_contingency([[a, c], [b, d]])
            shift = b / n_ext * 100 - a / n_int * 100
            print(f"  {s:<35}  shift = {shift:+5.2f}pp  chi2={chi2s:5.2f}  p={ps:.2e}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
