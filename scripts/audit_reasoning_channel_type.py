"""Check whether each reasoning-panel model's r_int text looks like a raw
reasoning trace or a synthesized "thought summary", per model.

Motivated by a spot-check: Gemini 2.5 Pro's r_int consistently opens with a
bolded essay title (e.g. "**My Ethical Calculus: A Response to a Heartbreaking
Request**") and reads as a complete, concluded narrative -- the documented
signature of the Gemini API returning only a model-generated "thought
summary" (via `include_thoughts`), never the model's raw thoughts. Claude
Sonnet 4.6's r_int shows none of that: it is short, informal, and frequently
cuts off mid-thought, consistent with genuine (if budget-truncated) raw
extended thinking.

This quantifies two signals across the whole reasoning panel so the
Gemini/Claude asymmetry (and the already-documented GPT-5.4 encryption, see
README.md "Methodological notes") rests on numbers rather than a few
hand-picked examples:

  1. bold_title -- r_int opens with a markdown-bold "title" line, the
                   structural signature of a synthesized narrative rather
                   than an unstructured raw trace.
  2. encrypted   -- r_int is an opaque Fernet ciphertext blob (OpenAI's
                   encrypted reasoning items), not text at all.

(An "ends on terminal punctuation" check was tried and dropped: nearly every
model's r_int ends on a full stop regardless of whether the trace reads as
truncated or complete, so it does not discriminate here.)

Reuses the same merged samples.json as audit_rejudged_numbers.py /
audit_hedge_categories.py.
"""
from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

SAMPLES = Path("logs/deliberative_justification_v2/20260527_combined_panels_curation/reasoning/samples.json")

BOLD_TITLE_RE = re.compile(r"^\*\*[^*\n]{3,100}\*\*")
ENCRYPTED_RE = re.compile(r"^gAAAAAB")

ORDER = [
    "gpt-5.4", "claude-sonnet-4-6", "kimi-k2p6", "qwen3p6-plus",
    "glm-5p1", "deepseek-v4-pro", "gemini-2.5-pro",
]


def get_track(target, kind):
    for tk in target.get("tracks", []) or []:
        if tk.get("text_kind") == kind:
            return tk.get("target_text")
    return None


def main():
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))

    per_m = defaultdict(list)
    for dil in samples["dilemmas"]:
        for tg in dil["targets"]:
            m = tg["model"].split("/")[-1]
            r_int = get_track(tg, "reasoning")
            if not r_int:
                continue
            per_m[m].append(r_int)

    print(f"{'model':22s} {'n':>4s} {'med_len':>8s} {'bold_title':>11s} {'encrypted':>10s}")
    for m in ORDER:
        texts = per_m[m]
        if not texts:
            continue
        n = len(texts)
        lens = [len(t) for t in texts]
        bold = sum(1 for t in texts if BOLD_TITLE_RE.match(t.strip())) / n
        enc = sum(1 for t in texts if ENCRYPTED_RE.match(t.strip())) / n
        print(f"{m:22s} {n:4d} {statistics.median(lens):8.0f} {bold*100:10.1f}% {enc*100:9.1f}%")


if __name__ == "__main__":
    main()
