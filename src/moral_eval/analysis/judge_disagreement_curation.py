"""Build (and port back) a human-curation file for inter-judge disagreements.

Workflow:

1. After ``judge_disagreement.py`` writes ``judge_disagreement_pairs.csv``,
   run this module's ``build`` command.  It joins the disagreement points
   against the source ``samples.json`` to attach the full context the
   curator needs (dilemma situation, target text, defense, the critical
   question text, each judge's reasoning) and adds empty curator columns
   at the end.

2. Open the resulting ``judge_disagreement_curation.csv`` in a spreadsheet
   tool (Excel / Sheets / Numbers).  For each row, fill in:

       - ``human_score``  (1, 2, or 3)
       - ``human_reasoning`` (free text)
       - ``agreed_with`` ('a', 'b', 'neither', 'both')
       - ``curator`` (your initials / name)

   The ``conflict_id`` column is a stable SHA-1 of the identifying tuple
   so the row stays linkable even if rows are reordered.

3. When you're done, run this module's ``apply`` command to port the
   resolutions back into samples.json (or a derivative).  Two policies:

       - ``--policy human`` (default): for every curated conflict, REPLACE
         both judges' scores at that (cell × metric) with the human score.
       - ``--policy add-human-judge``: append a new Phase-4 entry per
         affected cell with judge alias ``human_curator``; existing
         judges' scores are untouched.

   Either policy preserves an audit trail by recording the original
   scores in a ``curation_overrides`` block at the top of samples.json.

Usage::

    # Step 1: build the curation file
    .\\.venv\\Scripts\\python.exe -m moral_eval.analysis.judge_disagreement_curation build \\
        logs/deliberative_justification_v2/20260520_combined/cot

    # ... (human fills in the curator columns in Excel / Sheets) ...

    # Step 3: port the resolutions back into samples.json
    .\\.venv\\Scripts\\python.exe -m moral_eval.analysis.judge_disagreement_curation apply \\
        logs/deliberative_justification_v2/20260520_combined/cot \\
        --policy human
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path


def _conflict_id(dilemma_id: str, model: str, track: str, kind: str,
                 key: str, judge_a: str, judge_b: str) -> str:
    """Stable identifier for a (cell × metric × judge-pair) disagreement.
    Used to round-trip curator decisions back into samples.json even if rows
    are reordered or columns moved."""
    s = "|".join([dilemma_id, model, track, kind, key, judge_a, judge_b])
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _truncate(text: str, max_chars: int) -> str:
    """Optional cap on long text in the curation CSV.  ``max_chars <= 0``
    disables truncation entirely (default) — the curator gets the full
    target_text / defense / judge reasoning for every conflict."""
    if not isinstance(text, str):
        return ""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + f"\n[...{len(text) - max_chars} more chars]"


def _find_critical_question(p2_entry: dict, key: str) -> str:
    """``key`` for local disagreements looks like ``q6.grounds`` — pull the
    text for that question out of the Phase-2 critical-questions block."""
    if "." not in key:
        return ""
    q_id = key.split(".", 1)[0]
    qs = p2_entry.get("critical_questions") or {}
    if isinstance(qs, dict):
        return str(qs.get(q_id, ""))
    if isinstance(qs, list):
        try:
            i = int(q_id.lstrip("q")) - 1
            return str(qs[i]) if 0 <= i < len(qs) else ""
        except Exception:
            return ""
    return ""


def _all_critical_questions_text(p2_entry: dict) -> str:
    """Render the entire Phase-2 critical-questions block as one
    newline-separated string (``q1: ...\\nq2: ...``).  Used as context for
    global-metric disagreements, which aren't tied to a single q-index."""
    qs_obj = p2_entry.get("critical_questions") or {}
    if isinstance(qs_obj, dict):
        try:
            ordered = sorted(qs_obj.items(),
                             key=lambda kv: int(str(kv[0]).lstrip("q")))
        except Exception:
            ordered = list(qs_obj.items())
        return "\n".join(f"{q}: {text}" for q, text in ordered)
    if isinstance(qs_obj, list):
        return "\n".join(f"q{i+1}: {q}" for i, q in enumerate(qs_obj))
    return ""


def _find_target(samples: dict, dilemma_id: str, model_short: str, track: str) -> dict | None:
    """Locate the target block matching (dilemma_id, model_short, track) in
    samples.json.  ``model_short`` is the trailing path segment (e.g.
    ``gpt-5.4``)."""
    for dil in samples["dilemmas"]:
        if dil["dilemma_id"] != dilemma_id:
            continue
        for tg in dil["targets"]:
            if tg["model"].split("/")[-1] != model_short:
                continue
            return {"dilemma": dil, "target": tg}
    return None


def _track_for(target: dict, track: str) -> dict | None:
    for tk in target.get("tracks") or []:
        if tk.get("text_kind") == track:
            return tk
    return None


def _cell_id(dilemma_id: str, model: str, track: str,
             judge_a: str, judge_b: str) -> str:
    """Stable identifier for a (cell × judge-pair) — coarser than _conflict_id;
    one cell can have many metric-level disagreements but only one cell_id."""
    s = "|".join([dilemma_id, model, track, judge_a, judge_b])
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _judge_failures(p4_entry: dict) -> list[tuple[str, str, int]]:
    """Enumerate every (kind, key, score=1) cell from a single Phase-4 judge entry.
    'kind' is 'local' or 'global'; 'key' is q-index.dim for local or just dim for global.
    Used to count how many failures a judge saw in the entire scoring of one cell.
    """
    out: list[tuple[str, str, int]] = []
    for q, dims in (p4_entry.get("local") or {}).items():
        if isinstance(dims, dict):
            for d in _LOCAL_DIMS:
                v = dims.get(d)
                if isinstance(v, int) and v == _FAIL:
                    out.append(("local", f"{q}.{d}", v))
    for d in _GLOBAL_DIMS:
        v = (p4_entry.get("global") or {}).get(d)
        if isinstance(v, int) and v == _FAIL:
            out.append(("global", d, v))
    return out


# Scale constants used by the failure enumeration above.
_LOCAL_DIMS = ("acceptability", "relevance", "grounds")
_GLOBAL_DIMS = ("coherence", "acceptability", "relevance", "sufficiency")
_FAIL = 1


# Models elicited via visible pre-<answer> prose (CoT panel); everything
# else is treated as a provider-native reasoning-branch model.
_COT_PANEL_MODELS = {"mistral-large-2512", "Llama 3.3", "llama-3.3", "llama-3p3"}


def _panel_for_model(model_short: str) -> str:
    """Return ``"cot"`` for CoT-panel subjects (Mistral Large 2512,
    Llama 3.3) and ``"reasoning"`` for everything else."""
    return "cot" if model_short in _COT_PANEL_MODELS else "reasoning"


def cmd_build(args) -> int:
    cot_dir: Path = args.cot_dir
    pairs_path = cot_dir / "judge_disagreement_pairs.csv"
    samples_path = cot_dir / "samples.json"
    if not pairs_path.exists():
        raise SystemExit(
            f"missing {pairs_path}; run judge_disagreement.py first."
        )
    if not samples_path.exists():
        raise SystemExit(f"missing samples.json: {samples_path}")

    samples = json.loads(samples_path.read_text(encoding="utf-8"))

    # Group metric-level disagreements by (cell × judge-pair) for the per-cell
    # curation file.  We still emit the per-metric file too (downstream
    # ``apply`` reads the metric-level one).
    per_cell: dict[tuple, dict] = {}

    out_rows = []
    # iterate every disagreement row and enrich with context
    with pairs_path.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            did = r["dilemma_id"]; model = r["model"]; track = r["track"]
            scheme = r["scheme"]; ja = r["judge_a"]; jb = r["judge_b"]
            kind = r["kind"]; key = r["key"]
            sa, sb = r["score_a"], r["score_b"]

            cid = _conflict_id(did, model, track, kind, key, ja, jb)
            ctx = _find_target(samples, did, model, track)
            dilemma_situation = ""
            verdict = ""
            target_text = ""
            other_track_text = ""
            defense_text = ""
            crit_q = ""
            ja_reason = ""
            jb_reason = ""
            ja_full_scores = ""
            jb_full_scores = ""
            if ctx:
                dilemma_situation = ctx["dilemma"].get("dilemma") or ""
                verdict = ctx["target"].get("verdict") or ""
                # Also surface the OTHER track's text so the curator sees the
                # full model output (both reasoning and justification).
                other_kind = "justification" if track == "reasoning" else "reasoning"
                tk_other = _track_for(ctx["target"], other_kind)
                if tk_other:
                    other_track_text = tk_other.get("target_text") or ""
                tk = _track_for(ctx["target"], track)
                if tk:
                    target_text = tk.get("target_text") or ""
                    # Phase-2 entry for the judge1 that fed both judges
                    p2_entries = tk.get("phase2") or []
                    p2 = p2_entries[0] if p2_entries else {}
                    defense_text = p2.get("defense") or ""
                    if kind == "local":
                        crit_q = _find_critical_question(p2, key)
                    else:
                        # Global metrics aren't tied to a single CQ, but the
                        # curator still needs the full CQ block as context.
                        crit_q = _all_critical_questions_text(p2)
                    # Each judge's reasoning prose + full score matrix
                    for p4 in tk.get("phase4") or []:
                        if p4.get("phase4_judge") == ja:
                            ja_reason = p4.get("judge_reasoning") or ""
                            ja_full_scores = json.dumps(_scores_dict(p4),
                                                        ensure_ascii=False, indent=2)
                        elif p4.get("phase4_judge") == jb:
                            jb_reason = p4.get("judge_reasoning") or ""
                            jb_full_scores = json.dumps(_scores_dict(p4),
                                                        ensure_ascii=False, indent=2)

            # Column order follows the protocol's logical flow so the curator
            # can read the row top-to-bottom: identifiers → Phase 1 (subject's
            # response to dilemma) → Phase 2 (judge's critical questions) →
            # Phase 3 (subject's defense answering those CQs) → Phase 4 (both
            # judges' scoring of the defense) → curator columns.
            out_rows.append({
                # --- Identifiers ---
                "conflict_id": cid,
                "panel": _panel_for_model(model),
                "dilemma_id": did,
                "model": model,           # the SUBJECT model
                "track": track,
                "scheme": scheme,
                "kind": kind,
                "key": key,
                "judge_a": ja,
                "judge_b": jb,
                "score_a": sa,
                "score_b": sb,
                # --- Phase 1: Subject model's response to the dilemma ---
                "p1_dilemma": dilemma_situation,
                "p1_subject_verdict": verdict,
                "p1_subject_target_text": target_text,
                "p1_subject_other_track_text": other_track_text,
                # --- Phase 2: Phase-2 judge's critical questions for the
                #     target track.  For local conflicts, only the disputed
                #     CQ; for global conflicts, the whole CQ block. ---
                "p2_critical_question": crit_q,
                # --- Phase 3: Subject model's defense answering the CQs ---
                "p3_subject_defense": defense_text,
                # --- Phase 4: Both judges' scoring of the defense ---
                "p4_judge_a_scores": ja_full_scores,
                "p4_judge_b_scores": jb_full_scores,
                "p4_judge_a_reasoning": ja_reason,
                "p4_judge_b_reasoning": jb_reason,
                # --- Curator columns (intentionally empty) ---
                "human_score": "",
                "human_reasoning": "",
                "agreed_with": "",
                "curator": "",
                "decision_at": "",
                "notes": "",
            })

            # (The per-cell file is built independently from samples.json
            # in the polarised-failures walk further below; nothing accumulates
            # here.)

    out_path = cot_dir / "judge_disagreement_curation.csv"
    if out_rows:
        # Sort rows so the most likely interesting conflicts (those involving
        # global metrics, or where the score gap is bigger) bubble to the top.
        # Panel comes first so reasoning-branch and CoT-panel rows are
        # grouped — the curator works through one panel at a time.
        def sort_key(r):
            try:
                gap = abs(int(r["score_a"]) - int(r["score_b"]))
            except Exception:
                gap = 0
            kind_priority = 0 if r["kind"] == "global" else 1
            return (r.get("panel", ""), kind_priority, -gap,
                    r["model"], r["dilemma_id"])
        out_rows.sort(key=sort_key)

        with out_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()),
                               quoting=csv.QUOTE_MINIMAL)
            w.writeheader()
            for row in out_rows:
                w.writerow(row)
    print(f"wrote {len(out_rows)} metric-level conflicts to {out_path}")

    # ============================================================
    # Per-cell curation file — POLARISED FAILURES ONLY
    # ============================================================
    # Filter: include a cell only when ONE judge scored 1 (= failure) at
    # least once anywhere in their scoring (local q×dim OR global dim) AND
    # the OTHER judge scored 1 zero times anywhere in their scoring.  These
    # are the maximally-polarised disagreement cells where one judge sees
    # a hard failure and the other sees none.
    #
    # The output row carries the entire local + global score matrix for
    # both judges inline, plus parallel empty columns for the curator to
    # fill in their own scoring.
    polarised_rows: list[dict] = []
    # First determine the max number of critical questions across all cells
    # in the dataset so the column schema is wide enough for every row.
    max_q = 0
    for dil in samples["dilemmas"]:
        for tg in dil["targets"]:
            for tk in tg["tracks"]:
                for p2 in tk.get("phase2") or []:
                    qs = p2.get("critical_questions") or {}
                    if isinstance(qs, dict):
                        max_q = max(max_q, len(qs))
                    elif isinstance(qs, list):
                        max_q = max(max_q, len(qs))
                for p4 in tk.get("phase4") or []:
                    loc = p4.get("local") or {}
                    if isinstance(loc, dict):
                        max_q = max(max_q, len(loc))
    if max_q == 0:
        max_q = 7  # safe default

    # Walk every (cell, judge-pair).  We DON'T re-use _walk_pairs because
    # that yields *all* pairs whether they disagree or not; we want the
    # polarised filter applied per-cell here.
    for dil in samples["dilemmas"]:
        did = dil["dilemma_id"]
        for tg in dil["targets"]:
            model_short = tg["model"].split("/")[-1]
            panel = _panel_for_model(model_short)
            for tk in tg["tracks"]:
                track = tk.get("text_kind", "?")
                p2_entries = tk.get("phase2") or []
                if not p2_entries:
                    continue
                p2 = p2_entries[0]
                scheme = (p2.get("scheme") or "").strip() or "(unspecified)"
                p4s = [p for p in (tk.get("phase4") or []) if not p.get("skipped")]
                if len(p4s) < 2:
                    continue
                aliases = sorted({p.get("phase4_judge", "?") for p in p4s})
                by_alias = {p.get("phase4_judge"): p for p in p4s}
                for i, a in enumerate(aliases):
                    for b in aliases[i + 1:]:
                        p4_a = by_alias[a]; p4_b = by_alias[b]
                        # Polarised filter: one judge has ≥1 score=1, the other has 0
                        fa = len(_judge_failures(p4_a))
                        fb = len(_judge_failures(p4_b))
                        if (fa > 0 and fb == 0) or (fa == 0 and fb > 0):
                            polarised_rows.append(_build_polarised_row(
                                did, model_short, track, scheme, p2, p4_a, p4_b,
                                a, b, fa, fb,
                                ctx_dilemma=dil, ctx_target=tg, max_q=max_q,
                                panel=panel,
                            ))

    # Sort: panel first (reasoning-branch rows above CoT-panel rows so the
    # curator works through one panel at a time), then failing-judge,
    # then model, then dilemma.
    polarised_rows.sort(key=lambda r: (
        r.get("panel", ""), r["failing_judge"], r["model"], r["dilemma_id"]
    ))

    # Column order follows the protocol's logical flow so the curator can
    # read the row top-to-bottom.  Columns are grouped into five blocks:
    #
    #   1. Identifiers           — what cell this is, which two judges are in conflict
    #   2. Phase 1 (subject)     — dilemma + the subject model's response
    #   3. Phase 2 (judge1)      — critical questions the judge asked
    #   4. Phase 3 (subject)     — subject's defence answering those CQs
    #   5. Phase 4 (judge a/b)   — each judge's score matrix + prose reasoning
    #   6. Curator               — pre-filled human_scores + free-text columns
    #
    # The column keys carry the same `model` / `track` / `dilemma_id` names
    # that `cmd_apply` expects, so the round-trip back into samples.json
    # is unaffected.
    field_order = [
        # --- 1. Identifiers ---
        "cell_id", "panel", "dilemma_id", "model", "track", "scheme",
        "judge_a", "judge_b",
        "judge_a_total_failures", "judge_b_total_failures",
        "failing_judge",
        # --- 2. Phase 1: subject model's response ---
        "p1_dilemma",
        "p1_subject_verdict",
        "p1_subject_target_text",       # the track being evaluated
        "p1_subject_other_track_text",  # the OTHER track, for context
        # --- 3. Phase 2: Phase-2 judge's critical questions ---
        "p2_critical_questions",
        # --- 4. Phase 3: subject model's defence (answering the CQs) ---
        "p3_subject_defense",
        # --- 5. Phase 4: both Phase-4 judges' scoring of the defence ---
        # Three JSON columns. p4_judge_a_scores / p4_judge_b_scores are
        # read-only snapshots; human_scores is pre-filled with the FAILING
        # judge's scores and the curator edits the values they disagree with.
        "p4_judge_a_scores", "p4_judge_a_reasoning",
        "p4_judge_b_scores", "p4_judge_b_reasoning",
        # --- 6. Curator columns ---
        "human_scores",
        "agreed_with", "human_reasoning", "curator", "decision_at", "notes",
    ]

    per_cell_path = cot_dir / "judge_disagreement_curation_per_cell.csv"
    if polarised_rows:
        with per_cell_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=field_order, quoting=csv.QUOTE_MINIMAL)
            w.writeheader()
            for row in polarised_rows:
                w.writerow({k: row.get(k, "") for k in field_order})
    print(f"wrote {len(polarised_rows)} polarised-failure cells to {per_cell_path}")
    print(f"  (one judge saw ≥1 failure in their entire scoring; the other saw 0)")
    print()
    print("== Per-cell file (POLARISED FAILURES) — primary curation input ==")
    print("Each row has THREE JSON score columns:")
    print("  - judge_a_scores    (read-only snapshot of judge A's scoring)")
    print("  - judge_b_scores    (read-only snapshot of judge B's scoring)")
    print("  - human_scores      pre-filled with the FAILING judge's scoring;")
    print("                      edit the values you disagree with.  Apply will only")
    print("                      override metrics whose value you changed from the")
    print("                      pre-fill — untouched scores are left alone.")
    print()
    print("Other columns to fill (optional):")
    print("  - agreed_with         'a', 'b', 'neither', 'both', or 'mixed'")
    print("  - human_reasoning     free-text rationale")
    print("  - curator             initials / name")
    print()
    print("== Metric-level file (granular drill-down, kept for reference) ==")
    print(f"  {out_path}")
    print()
    print("After curating, port the per-cell decisions back:")
    print(f"  python -m moral_eval.analysis.judge_disagreement_curation apply {cot_dir}")
    return 0


def _scores_dict(p4: dict) -> dict:
    """Pull the local + global score blocks from a Phase-4 entry into a
    minimal nested dict ({'local': {...}, 'global': {...}}).  Used to
    pre-fill the JSON columns in the curation CSV."""
    out_local: dict = {}
    for q, dims in (p4.get("local") or {}).items():
        if isinstance(dims, dict):
            out_local[q] = {
                d: int(dims[d]) for d in _LOCAL_DIMS
                if d in dims and isinstance(dims[d], (int, float))
            }
    out_global = {
        d: int(v) for d, v in (p4.get("global") or {}).items()
        if d in _GLOBAL_DIMS and isinstance(v, (int, float))
    }
    return {"local": out_local, "global": out_global}


def _build_polarised_row(dilemma_id: str, model_short: str, track: str,
                         scheme: str, p2_entry: dict,
                         p4_a: dict, p4_b: dict, judge_a: str, judge_b: str,
                         failures_a: int, failures_b: int,
                         *, ctx_dilemma: dict, ctx_target: dict,
                         max_q: int, panel: str = "") -> dict:
    """One polarised-failure row with full context + JSON score blocks for
    both judges + a pre-filled ``human_scores`` JSON that the curator edits.

    Column names use phase prefixes (``p1_``, ``p2_``, ``p3_``, ``p4_``) so
    the curator can tell at a glance which protocol phase each piece of
    text came from and which model produced it.  Identifier columns
    (``model``, ``track``, ``dilemma_id``) keep their plain names so the
    apply step can still round-trip the curated scores back into
    samples.json without further plumbing.
    """
    row: dict = {
        # --- Identifiers ---
        "cell_id": _cell_id(dilemma_id, model_short, track, judge_a, judge_b),
        "panel": panel,
        "dilemma_id": dilemma_id,
        "model": model_short,        # the SUBJECT model
        "track": track,
        "scheme": scheme,
        "judge_a": judge_a,
        "judge_b": judge_b,
        "judge_a_total_failures": failures_a,
        "judge_b_total_failures": failures_b,
        "failing_judge": "a" if failures_a > 0 else "b",
    }

    # --- Phase 1: subject model's response to the dilemma ---
    row["p1_dilemma"] = ctx_dilemma.get("dilemma") or ""
    row["p1_subject_verdict"] = ctx_target.get("verdict") or ""
    target_text = ""
    other_track_text = ""
    for tk in ctx_target.get("tracks") or []:
        if tk.get("text_kind") == track:
            target_text = tk.get("target_text") or ""
        else:
            other_track_text = tk.get("target_text") or ""
    row["p1_subject_target_text"] = target_text
    row["p1_subject_other_track_text"] = other_track_text

    # --- Phase 2: Phase-2 judge's critical questions for the target text ---
    row["p2_critical_questions"] = _all_critical_questions_text(p2_entry)

    # --- Phase 3: subject model's defence answering those CQs ---
    row["p3_subject_defense"] = p2_entry.get("defense") or ""

    # --- Phase 4: both Phase-4 judges' scoring of the defence ---
    # JSON snapshots of both judges' scoring matrices.  json.dumps with
    # indent=2 keeps the column readable as multi-line text in a spreadsheet.
    scores_a = _scores_dict(p4_a)
    scores_b = _scores_dict(p4_b)
    row["p4_judge_a_scores"] = json.dumps(scores_a, ensure_ascii=False, indent=2)
    row["p4_judge_a_reasoning"] = p4_a.get("judge_reasoning") or ""
    row["p4_judge_b_scores"] = json.dumps(scores_b, ensure_ascii=False, indent=2)
    row["p4_judge_b_reasoning"] = p4_b.get("judge_reasoning") or ""

    # --- Curator columns ---
    # Pre-fill the human column with the FAILING judge's scores.  The curator
    # edits the specific values they want to override — typically the
    # disputed 1s where the non-failing judge gave 2 or 3.  Leaving the
    # human_scores column identical to the failing judge's snapshot means
    # "I agree with the failing judge".
    failing = scores_a if failures_a > 0 else scores_b
    row["human_scores"] = json.dumps(failing, ensure_ascii=False, indent=2)
    row["agreed_with"] = ""
    row["human_reasoning"] = ""
    row["curator"] = ""
    row["decision_at"] = ""
    row["notes"] = ""
    return row


def cmd_apply(args) -> int:
    cot_dir: Path = args.cot_dir
    per_cell_path = cot_dir / "judge_disagreement_curation_per_cell.csv"
    samples_path = cot_dir / "samples.json"
    if not per_cell_path.exists():
        raise SystemExit(f"missing {per_cell_path}")
    if not samples_path.exists():
        raise SystemExit(f"missing {samples_path}")

    # Load per-cell curation.  For each row, parse human_scores JSON and
    # diff it against the corresponding judge's scores.  Any value where the
    # human disagrees with the failing judge becomes an override applied to
    # samples.json.
    curated = []
    with per_cell_path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            hs_raw = (r.get("human_scores") or "").strip()
            if not hs_raw:
                continue
            try:
                human_scores = json.loads(hs_raw)
            except json.JSONDecodeError as exc:
                print(f"  skipping {r.get('cell_id')}: invalid JSON in human_scores ({exc})",
                      file=sys.stderr)
                continue
            if not isinstance(human_scores, dict):
                print(f"  skipping {r.get('cell_id')}: human_scores is not an object",
                      file=sys.stderr)
                continue
            # Compare with the "failing judge" pre-fill to detect edits.
            # New column names use a ``p4_`` prefix; old (pre-2026-05-27)
            # files used the unprefixed names — fall back if needed.
            failing = r.get("failing_judge")
            failing_judge_key = "p4_judge_a_scores" if failing == "a" else "p4_judge_b_scores"
            pre_raw = r.get(failing_judge_key)
            if not pre_raw:
                failing_judge_key = "judge_a_scores" if failing == "a" else "judge_b_scores"
                pre_raw = r.get(failing_judge_key) or "{}"
            try:
                pre_filled = json.loads(pre_raw)
            except Exception:
                pre_filled = {}
            # Detect actual edits: any (kind, key, value) where the human's
            # number differs from the pre-fill.
            edits = _diff_scores(human_scores, pre_filled)
            if not edits:
                continue  # human left it identical to the failing judge → no override
            curated.append({**r, "_edits": edits, "_human_scores": human_scores})

    if not curated:
        raise SystemExit("no curated rows with edits (human_scores unchanged).")
    print(f"loaded {len(curated)} curated cells with edits")

    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    overrides = samples.setdefault("curation_overrides", [])

    timestamp = datetime.now().isoformat(timespec="seconds")

    applied_cells = 0
    applied_edits = 0
    not_found = 0
    for c in curated:
        cid = c["cell_id"]
        did = c["dilemma_id"]; model = c["model"]; track = c["track"]
        ja = c["judge_a"]; jb = c["judge_b"]
        edits = c["_edits"]
        ctx = _find_target(samples, did, model, track)
        if ctx is None:
            not_found += 1; continue
        tk = _track_for(ctx["target"], track)
        if tk is None:
            not_found += 1; continue
        # Find both Phase-4 cells (both judges)
        affected = [p for p in (tk.get("phase4") or [])
                    if p.get("phase4_judge") in (ja, jb)]
        if not affected:
            not_found += 1; continue

        for kind, key, new_val in edits:
            if args.policy == "human":
                # Write the curated value to BOTH judges' entries at this metric.
                for p4 in affected:
                    original = _read_metric(p4, kind, key)
                    _write_metric(p4, kind, key, new_val)
                    overrides.append({
                        "cell_id": cid, "applied_at": timestamp,
                        "dilemma_id": did, "model": model, "track": track,
                        "kind": kind, "key": key,
                        "phase4_judge": p4.get("phase4_judge"),
                        "original_score": original, "human_score": new_val,
                        "curator": c.get("curator", ""), "policy": "human",
                        "reasoning": c.get("human_reasoning", ""),
                    })
                applied_edits += 1
            elif args.policy == "add-human-judge":
                target_entry = next(
                    (p for p in (tk.get("phase4") or [])
                     if p.get("phase4_judge") == "human_curator"),
                    None,
                )
                if target_entry is None:
                    target_entry = {
                        "phase2_judge": affected[0].get("phase2_judge"),
                        "phase4_judge": "human_curator",
                        "judge_reasoning": c.get("human_reasoning", ""),
                        "local": {},
                        "global": {},
                        "skipped": False,
                    }
                    tk.setdefault("phase4", []).append(target_entry)
                _write_metric(target_entry, kind, key, new_val)
                overrides.append({
                    "cell_id": cid, "applied_at": timestamp,
                    "dilemma_id": did, "model": model, "track": track,
                    "kind": kind, "key": key,
                    "phase4_judge": "human_curator",
                    "human_score": new_val,
                    "policy": "add-human-judge",
                    "curator": c.get("curator", ""),
                    "reasoning": c.get("human_reasoning", ""),
                })
                applied_edits += 1
            else:
                raise SystemExit(f"unknown policy: {args.policy}")
        applied_cells += 1

    # Backup
    bak = samples_path.with_suffix(f".bak_curation_{timestamp.replace(':', '-')}.json")
    bak.write_text(json.dumps(json.loads(samples_path.read_text(encoding="utf-8")),
                              ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    samples_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    print(f"applied {applied_edits} metric overrides across {applied_cells} cells;"
          f" {not_found} could not be located in samples.json")
    print(f"backup: {bak}")
    print(f"updated: {samples_path}")
    print()
    print("Re-run the analyser + plotter + report to refresh derived files:")
    print(f"  python -m moral_eval.analysis.walton_analysis {cot_dir}")
    print(f"  python -m moral_eval.analysis.walton_plots    {cot_dir}")
    print(f"  python -m moral_eval.analysis.walton_report   {cot_dir}")
    return 0


def _diff_scores(human: dict, pre_filled: dict) -> list[tuple[str, str, int]]:
    """Return every (kind, key, new_value) tuple where the human's JSON
    differs from the pre-filled JSON.  ``kind`` is 'local' or 'global'.
    For local entries, ``key`` is ``q{i}.{dim}`` (matching the format used
    elsewhere); for global, ``key`` is the dim name."""
    edits: list[tuple[str, str, int]] = []
    # Local
    hl = (human.get("local") or {}); pl = (pre_filled.get("local") or {})
    q_keys = sorted(set(hl.keys()) | set(pl.keys()))
    for q in q_keys:
        hd = hl.get(q) if isinstance(hl.get(q), dict) else {}
        pd = pl.get(q) if isinstance(pl.get(q), dict) else {}
        for d in _LOCAL_DIMS:
            hv = hd.get(d); pv = pd.get(d)
            if isinstance(hv, (int, float)) and (
                pv is None or int(hv) != int(pv)
            ):
                edits.append(("local", f"{q}.{d}", int(hv)))
    # Global
    hg = (human.get("global") or {}); pg = (pre_filled.get("global") or {})
    for d in _GLOBAL_DIMS:
        hv = hg.get(d); pv = pg.get(d)
        if isinstance(hv, (int, float)) and (pv is None or int(hv) != int(pv)):
            edits.append(("global", d, int(hv)))
    return edits


def _read_metric(p4: dict, kind: str, key: str):
    if kind == "local":
        if "." not in key:
            return None
        q, dim = key.split(".", 1)
        return ((p4.get("local") or {}).get(q) or {}).get(dim)
    return (p4.get("global") or {}).get(key)


def _write_metric(p4: dict, kind: str, key: str, value: int) -> None:
    if kind == "local":
        if "." not in key:
            return
        q, dim = key.split(".", 1)
        local = p4.setdefault("local", {})
        q_block = local.setdefault(q, {})
        q_block[dim] = int(value)
    else:
        p4.setdefault("global", {})[key] = int(value)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Write the curation CSV with empty curator cols.")
    p_build.add_argument("cot_dir", type=Path,
                         help="cot directory containing samples.json + judge_disagreement_pairs.csv.")
    p_build.set_defaults(func=cmd_build)

    p_apply = sub.add_parser("apply", help="Port curated decisions back into samples.json.")
    p_apply.add_argument("cot_dir", type=Path)
    p_apply.add_argument("--policy",
                         choices=("human", "add-human-judge"),
                         default="human",
                         help="'human' (default): overwrite both judges' scores at the "
                              "conflict point with the curator's number.  "
                              "'add-human-judge': append a 'human_curator' Phase-4 entry "
                              "with the curator's scores, leave existing judges' scores intact.")
    p_apply.set_defaults(func=cmd_apply)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
