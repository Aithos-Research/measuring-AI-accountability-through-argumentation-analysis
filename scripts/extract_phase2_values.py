"""Extract the values that Phase-2 situated critical questions invoke when the
identified Walton scheme is `value-based practical reasoning` or
`argumentation from values`.

Background.  The Phase-2 prompt asks the judge to emit a `variables` dict, but
the structured-output schema we run with only retains `reasoning`, `type`, and
`critical_questions`.  Variables are therefore only present *inlined* inside
the situated critical questions, where they appear in stock phrasings such as
`the value of X`, `the values of X, Y and Z`, `values such as X, Y`.

This script walks samples.json, isolates the two value-centric schemes, and
parses value tokens out of those phrasings.  Output is a tidy CSV plus a
console summary of the top-K values overall and per-model.

Usage:
    .\.venv\Scripts\python.exe scripts/extract_phase2_values.py \
        --samples logs/.../cot/samples.json \
        --out-dir logs/.../cot/ \
        --top 30
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


# The phrases the situated CQs use to surface value tokens.  All matches are
# captured as the text following the trigger up to the next clause boundary
# (full stop, semicolon, ?, ), or the words " for "/" while "/" given ").  We
# then split on comma/and/or to get individual value tokens.
_VALUE_PHRASE_RE = re.compile(
    r"\bvalue[s]?\s+(?:of|like|such as|including|that the agent holds|the agent endorses|"
    r"that include|are|encompass[a-z]*)\s+",
    re.IGNORECASE,
)
# Variants without the "of/like" hinge that still bracket values clearly.
_VALUE_PHRASE_RE_2 = re.compile(
    r"\b(?:the\s+)?value[s]?\s+(?=[A-Z])",  # "value Compassion" / "values Compassion, Honesty"
)
# Stop tokens: where the value-list ends.
_VALUE_LIST_STOP = re.compile(
    r"[.;?!\)]|\s+(?:for|while|given|because|since|in order|so that|"
    r"and how|and the|that the|when|as the|over)\b",
    re.IGNORECASE,
)

# Words we strip from candidate tokens (articles, fillers).
_STRIP_TOKENS = {"the", "a", "an", "any", "some"}

# Words that are obviously not values (false positives we keep seeing).
_BLOCKLIST = {
    "agent", "narrator", "person", "people", "user", "model", "ai",
    "decision", "decisions", "action", "actions", "consequence", "consequences",
    "claim", "claims", "response", "responses", "scenario", "situation",
    "argument", "arguments", "premise", "premises", "conclusion", "conclusions",
    "goal", "goals", "outcome", "outcomes", "value", "values",
    # Adverbial / connective filler the regex sometimes captures.
    "particularly", "especially", "specifically", "namely", "primarily",
    "as opposed to", "as well as", "rather than", "instead of",
    "for example", "for instance", "such", "those", "these",
}

# Map common variants onto a canonical form.  Conservative: only collapse
# textbook synonyms so we don't smuggle in our own value taxonomy.
_CANONICAL = {
    "human life": "preservation of life",
    "protection of human life": "preservation of life",
    "sanctity of life": "preservation of life",
    "well being": "well-being",
    "wellbeing": "well-being",
    "self determination": "autonomy",
    "self-determination": "autonomy",
    "personal autonomy": "autonomy",
    "individual autonomy": "autonomy",
    "bodily autonomy": "autonomy",
    "patient autonomy": "autonomy",
    "honesty and transparency": "honesty",
    "transparency": "honesty",
    "truthfulness": "honesty",
    "harm reduction": "non-maleficence",
    "harm prevention": "non-maleficence",
    "minimizing harm": "non-maleficence",
    "not causing harm": "non-maleficence",
    "do no harm": "non-maleficence",
    "non maleficence": "non-maleficence",
    "compassion and empathy": "compassion",
    "empathy": "compassion",
    "mercy": "compassion",
    "kindness": "compassion",
    "care": "compassion",
    "dignity": "human dignity",
    "respect for persons": "human dignity",
    "respect for human dignity": "human dignity",
    "respect for the individual": "human dignity",
    "justice": "fairness",
    "equity": "fairness",
    "equality": "fairness",
    "duty": "duty/obligation",
    "obligation": "duty/obligation",
    "duties": "duty/obligation",
    "obligations": "duty/obligation",
    "duty of care": "duty/obligation",
    "professional obligation": "duty/obligation",
    "professional duty": "duty/obligation",
    "responsibility": "responsibility/accountability",
    "accountability": "responsibility/accountability",
    "personal responsibility": "responsibility/accountability",
    "loyalty": "loyalty",
    "trustworthiness": "trust",
    "integrity": "integrity",
    "privacy": "privacy",
    "freedom": "liberty",
    "liberty": "liberty",
    "individual freedom": "liberty",
    "informed consent": "consent",
    "consent": "consent",
    "safety": "safety",
    "public safety": "safety",
    "security": "safety",
    "community safety": "safety",
    "common good": "common good",
    "public good": "common good",
    "social good": "common good",
    "rule of law": "rule of law",
    "law and order": "rule of law",
    "legality": "rule of law",
    "tradition": "tradition",
    "tradition and continuity": "tradition",
    "family": "family",
    "family loyalty": "family",
    "family welfare": "family",
    "family well-being": "family",
}


def _norm_token(s: str) -> str:
    s = s.strip().strip(",.;:'\"`*–—")
    s = re.sub(r"\s+", " ", s)
    s = s.lower()
    while s.split(" ", 1)[0] in _STRIP_TOKENS:
        parts = s.split(" ", 1)
        if len(parts) == 1:
            return ""
        s = parts[1]
    # Collapse "X integrity" -> "integrity" (academic / professional / personal /
    # moral / maintaining integrity all just mean integrity in this corpus).
    if s.endswith(" integrity") and s != "integrity":
        s = "integrity"
    if s.endswith(" ethics") and s not in ("professional ethics",):
        s = "professional ethics"
    if s.endswith(" honesty") and s != "honesty":
        s = "honesty"
    if s.endswith(" responsibility") and s != "responsibility":
        s = "responsibility/accountability"
    if s.endswith(" autonomy") and s != "autonomy":
        s = "autonomy"
    return s


def _split_list(blob: str) -> list[str]:
    # Split on commas, "and", "or", "&", or "/".  Keep multiword tokens intact.
    blob = re.sub(r"\s+&\s+", ", ", blob)
    blob = re.sub(r"\s+/\s+", ", ", blob)
    blob = re.sub(r"\band\b|\bor\b", ",", blob, flags=re.IGNORECASE)
    return [p.strip() for p in blob.split(",") if p.strip()]


def _extract_from_text(text: str) -> list[str]:
    """Return canonical value tokens parsed out of one situated-CQ string."""
    if not text:
        return []
    out: list[str] = []
    for m in _VALUE_PHRASE_RE.finditer(text):
        tail = text[m.end():]
        stop = _VALUE_LIST_STOP.search(tail)
        blob = tail[: stop.start()] if stop else tail
        for tok in _split_list(blob):
            t = _norm_token(tok)
            if not t or t in _BLOCKLIST:
                continue
            # Drop tokens with more than 5 words: usually descriptive prose.
            if len(t.split()) > 5:
                continue
            out.append(_CANONICAL.get(t, t))
    return out


def _walk_phase2(samples: dict, schemes_lc: set[str]):
    """Yield (model, dilemma_id, track, scheme, [cq_text, ...])."""
    for dil in samples["dilemmas"]:
        did = dil.get("dilemma_id")
        for tg in dil["targets"]:
            model_full = tg["model"]
            model = model_full.split("/")[-1]
            for tk in tg["tracks"]:
                track = "reasoning" if tk["text_kind"] == "reasoning" else "justification"
                for p2 in (tk.get("phase2") or []):
                    if p2.get("skipped"):
                        continue
                    scheme = (p2.get("scheme") or "").strip().lower()
                    if scheme not in schemes_lc:
                        continue
                    cqs = p2.get("critical_questions") or {}
                    if isinstance(cqs, dict):
                        texts = list(cqs.values())
                    elif isinstance(cqs, list):
                        texts = list(cqs)
                    else:
                        texts = []
                    yield model, did, track, scheme, texts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--top", type=int, default=30, help="Top-K to print to console.")
    args = ap.parse_args()

    samples = json.loads(args.samples.read_text(encoding="utf-8"))

    target_schemes = {"value-based practical reasoning", "argumentation from values"}

    # Counters
    overall = Counter()
    by_scheme: dict[str, Counter] = defaultdict(Counter)
    by_model: dict[str, Counter] = defaultdict(Counter)
    by_track: dict[str, Counter] = defaultdict(Counter)
    by_model_scheme: dict[tuple, Counter] = defaultdict(Counter)
    # Per-(model, dilemma, track) deduplicated set, so we count
    # "how many cells mention compassion" not "how many CQs mention it".
    cell_unique: dict[tuple, set[str]] = defaultdict(set)
    n_cells = 0

    for model, did, track, scheme, texts in _walk_phase2(samples, target_schemes):
        n_cells += 1
        for txt in texts:
            for v in _extract_from_text(txt):
                cell_unique[(model, did, track, scheme)].add(v)

    for (model, _did, track, scheme), vs in cell_unique.items():
        for v in vs:
            overall[v] += 1
            by_scheme[scheme][v] += 1
            by_model[model][v] += 1
            by_track[track][v] += 1
            by_model_scheme[(model, scheme)][v] += 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- CSVs ----
    def _write_counter(path: Path, counter: Counter, extra_cols: dict | None = None):
        rows = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["value", "n_cells", "pct_of_value_cells"] + list((extra_cols or {}).keys()))
            for v, n in rows:
                extras = list((extra_cols or {}).values())
                w.writerow([v, n, round(100.0 * n / n_cells, 2) if n_cells else 0.0] + extras)

    _write_counter(args.out_dir / "phase2_values_overall.csv", overall)

    with (args.out_dir / "phase2_values_by_scheme.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "value", "n_cells"])
        for scheme, c in by_scheme.items():
            for v, n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])):
                w.writerow([scheme, v, n])

    with (args.out_dir / "phase2_values_by_model.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "value", "n_cells"])
        for model, c in by_model.items():
            for v, n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])):
                w.writerow([model, v, n])

    with (args.out_dir / "phase2_values_by_track.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["track", "value", "n_cells"])
        for track, c in by_track.items():
            for v, n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])):
                w.writerow([track, v, n])

    with (args.out_dir / "phase2_values_by_model_scheme.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "scheme", "value", "n_cells"])
        for (model, scheme), c in by_model_scheme.items():
            for v, n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])):
                w.writerow([model, scheme, v, n])

    # ---- Console summary ----
    print(f"Scanned {n_cells} value-centric Phase-2 cells across the panel.")
    print(f"Distinct value tokens (post-canonicalisation): {len(overall)}")
    print()
    print(f"Top {args.top} values overall (count = number of cells that mention it):")
    print(f"  {'rank':>4}  {'count':>5}  {'%':>5}  value")
    for i, (v, n) in enumerate(sorted(overall.items(), key=lambda kv: (-kv[1], kv[0]))[: args.top], 1):
        pct = 100.0 * n / n_cells if n_cells else 0.0
        print(f"  {i:>4}  {n:>5}  {pct:5.1f}  {v}")
    print()
    for scheme in sorted(by_scheme):
        c = by_scheme[scheme]
        scheme_cells = sum(1 for k in cell_unique if k[3] == scheme)
        print(f"Top 10 values for scheme = '{scheme}'  ({scheme_cells} cells):")
        for i, (v, n) in enumerate(sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:10], 1):
            print(f"  {i:>2}.  {n:>4}  {v}")
        print()

    print(f"CSVs written to {args.out_dir}:")
    print("  phase2_values_overall.csv")
    print("  phase2_values_by_scheme.csv")
    print("  phase2_values_by_model.csv")
    print("  phase2_values_by_track.csv")
    print("  phase2_values_by_model_scheme.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
