"""Refill skipped Phase-4 judge cells in a completed dj_v2 run.

When a Phase-4 judge errors mid-run (e.g. OpenAI ``insufficient_quota``,
Anthropic rate-limit retries exhausted), the cell is flagged
``skipped: True`` in samples.json by the run's failsafe.  This script
reads samples.json directly, finds those errored cells for the chosen
judge, reconstructs the Phase-4 prompt from the same track's
``target_text`` + Phase-2's ``critical_questions`` + ``defense``, calls
the judge fresh, parses the response, and patches samples.json in-place.

It writes a ``.bak`` of samples.json first so you can roll back.

Examples
--------
Dry-run — count how many gpt-5.5 cells need refilling, no API calls::

    .\\.venv\\Scripts\\python.exe scripts/refill_phase4_judge.py \\
        --cot-dir logs/deliberative_justification_v2/20260520_103147/cot \\
        --judge-alias gpt_5_5 --dry-run

Actual refill::

    .\\.venv\\Scripts\\python.exe scripts/refill_phase4_judge.py \\
        --cot-dir logs/deliberative_justification_v2/20260520_103147/cot \\
        --judge-alias gpt_5_5 \\
        --judge-model openai_gpt55

After it finishes, re-run the analysis to refresh CSVs + plots::

    .\\.venv\\Scripts\\python.exe -m moral_eval.analysis.walton_analysis \\
        logs/deliberative_justification_v2/20260520_103147/cot
    .\\.venv\\Scripts\\python.exe -m moral_eval.analysis.walton_plots \\
        logs/deliberative_justification_v2/20260520_103147/cot
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("refill_phase4")


def _bootstrap(experiments_file: Path) -> None:
    """Mirror the bootstrap that run_trial.py does so the dj_v2 module can import
    cleanly: UTF-8 stdio, .env loaded, SSL trust store injected, the experiments
    file env var pointing at a config the module can parse."""
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass

    os.environ["MORAL_EVAL_EXPERIMENTS_FILE"] = str(experiments_file)

    import truststore
    truststore.inject_into_ssl()

    from dotenv import load_dotenv
    load_dotenv(override=True)

    hf_key = os.environ.get("HUGGINGFACE_ACCESS_KEY") or os.environ.get("HF_TOKEN")
    if hf_key:
        os.environ["HF_TOKEN"] = hf_key
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_key


def _is_judge_errored(p4_entry: dict, judge_alias: str) -> bool:
    """True when this Phase-4 cell is the target judge AND was skipped
    because the judge itself errored — not because the upstream Phase-2
    cascade-skipped (those genuinely have no defense to score)."""
    if p4_entry.get("phase4_judge") != judge_alias:
        return False
    if not p4_entry.get("skipped"):
        return False
    reason = (p4_entry.get("skipped_reason") or "").lower()
    # cascade skip messages start with "Defense was not produced..."
    if "defense was not produced" in reason:
        return False
    return True


def _format_questions_block(questions_obj: Any) -> str:
    """Reproduce ``format_questions_block`` from the v2 task.  ``questions_obj``
    may be a list (rare in samples.json) or a {q1: ..., q2: ...} dict (exporter
    shape)."""
    if isinstance(questions_obj, dict):
        # Sort by q-number so the prompt matches what the original run sent
        try:
            ordered = sorted(questions_obj.items(), key=lambda kv: int(str(kv[0]).lstrip("q")))
        except Exception:
            ordered = list(questions_obj.items())
        return "\n".join(f"{i}. {q}" for i, (_, q) in enumerate(ordered, start=1))
    if isinstance(questions_obj, list):
        return "\n".join(f"{i}. {q}" for i, q in enumerate(questions_obj, start=1))
    return ""


async def _refill_one(judge, prompt: str, n_questions: int, PHASE4_JUDGE_CFG,
                       safe_json_parse, _parse_arg_scores, EmptyJudgeCompletionError,
                       semaphore: asyncio.Semaphore) -> dict | None:
    """Issue a single judge call (under the semaphore) and return the parsed
    update payload, or None if the call failed.  Errors are swallowed and
    surfaced to the caller via None so one bad cell doesn't kill the batch."""
    async with semaphore:
        try:
            raw = await judge.generate(prompt, config=PHASE4_JUDGE_CFG, cache=False)
        except Exception as exc:
            return {"_error": f"generate failed: {exc!s}"}
        completion = (raw.completion if raw else "") or ""
        if not completion.strip():
            return {"_error": "empty completion"}
        try:
            obj = safe_json_parse(completion)
        except Exception as exc:
            return {"_error": f"json parse: {exc!s}"}
        if not isinstance(obj, dict):
            return {"_error": "completion did not parse to a dict"}
        # Local block per-question
        local_block: dict = {}
        for qi in range(1, n_questions + 1):
            q_entry = (obj.get("local", {}) or {}).get(f"q{qi}") or {}
            if isinstance(q_entry, dict):
                score_block = q_entry.get("score", q_entry) if isinstance(q_entry, dict) else {}
                if isinstance(score_block, dict):
                    local_block[f"q{qi}"] = {
                        d: int(score_block[d])
                        for d in ("acceptability", "relevance", "grounds")
                        if d in score_block and isinstance(score_block.get(d), (int, float))
                    }
        global_obj = obj.get("global", {}) or {}
        global_block = {
            d: int(global_obj[d])
            for d in ("coherence", "acceptability", "relevance", "sufficiency")
            if d in global_obj and isinstance(global_obj.get(d), (int, float))
        }
        judge_reasoning = ""
        if isinstance(obj.get("reasoning"), str):
            judge_reasoning = obj["reasoning"].strip()
        if not judge_reasoning:
            start = completion.find("{")
            if start > 0:
                judge_reasoning = completion[:start].strip()
        return {"local": local_block, "global": global_block,
                "judge_reasoning": judge_reasoning, "completion": completion}


async def _run_async(cot_dir: Path, judge_alias: str, judge_model_key: str,
                     dry_run: bool, limit: int | None,
                     concurrency: int, persist_every: int) -> dict:
    """Main async refill loop."""
    # Import inside the function so the bootstrap (env + ssl) has already run.
    from moral_eval.tasks.explainability.deliberative_justification_v2 import (
        PROMPT3_REASONING, PROMPT3_JUSTIFICATION, _parse_arg_scores, fill,
        PHASE4_JUDGE_CFG, EmptyJudgeCompletionError,
    )
    from moral_eval.utils.config_loader import resolve_models
    from inspect_ai.model import get_model

    judge_id = resolve_models([judge_model_key])[0]
    samples_path = cot_dir / "samples.json"
    if not samples_path.exists():
        raise SystemExit(f"samples.json not found at {samples_path}")
    samples = json.loads(samples_path.read_text(encoding="utf-8"))

    # Override the YAML's max_connections / max_retries on the imported config
    # so the throttle baked in for the original Sonnet-heavy run doesn't apply
    # to gpt-5.5 refills.  Inspect-ai's internal connection pool reads these
    # off the GenerateConfig, so the asyncio semaphore alone isn't enough.
    try:
        PHASE4_JUDGE_CFG = PHASE4_JUDGE_CFG.model_copy(update={
            "max_connections": concurrency,
            "max_retries": 5,
        })
    except Exception as exc:
        logger.warning("Could not override PHASE4_JUDGE_CFG concurrency: %s", exc)
    logger.info("Judge GenerateConfig: max_connections=%s max_retries=%s",
                getattr(PHASE4_JUDGE_CFG, "max_connections", "?"),
                getattr(PHASE4_JUDGE_CFG, "max_retries", "?"))

    # Walk and gather work items
    work: list[tuple] = []  # (dilemma_obj, target_obj, track_obj, p2_entry, p4_entry)
    for dil in samples["dilemmas"]:
        for tg in dil["targets"]:
            for tk in tg["tracks"]:
                # Map phase2 entries by judge alias so we can pair with p4
                p2_by_alias = {p2.get("judge"): p2 for p2 in (tk.get("phase2") or [])}
                for p4 in tk.get("phase4") or []:
                    if not _is_judge_errored(p4, judge_alias):
                        continue
                    p2 = p2_by_alias.get(p4.get("phase2_judge"))
                    if p2 is None or p2.get("skipped"):
                        # No matching Phase-2 with a real defense; nothing to score.
                        continue
                    work.append((dil, tg, tk, p2, p4))

    logger.info("Found %d errored Phase-4 cells for judge=%s under cot_dir=%s",
                len(work), judge_alias, cot_dir)
    if limit is not None:
        work = work[:limit]
        logger.info("Limiting to first %d cells (per --limit)", len(work))
    if dry_run:
        logger.info("--dry-run: no API calls will be issued.")
        # Surface the spread by dilemma/model/track for visibility
        from collections import Counter
        by_dil = Counter((d["dilemma_id"], tg["model"].split("/")[-1], tk["text_kind"])
                         for d, tg, tk, _, _ in work)
        for (did, m, kind), n in by_dil.most_common(20):
            logger.info("  - %s / %s / %s x%d", did, m, kind, n)
        return {"to_refill": len(work), "refilled": 0, "errors": 0}

    # Backup samples.json before mutating
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = cot_dir / f"samples.bak_{ts}.json"
    shutil.copy2(samples_path, bak)
    logger.info("Backup written to %s", bak)

    judge = get_model(judge_id)
    # Bound the in-flight judge calls.  CLI default (3) matches the
    # judge_generate.max_connections in the YAML; the user can raise it now
    # that OpenAI credits are confirmed.
    semaphore = asyncio.Semaphore(max(1, concurrency))
    from moral_eval.utils.robustness_utils import safe_json_parse

    # Pre-build every Phase-4 prompt so the async dispatcher just feeds them in.
    prepared: list[tuple] = []  # (idx, did, m, kind, p4, n_questions, prompt)
    for i, (dil, tg, tk, p2, p4) in enumerate(work, 1):
        did = dil["dilemma_id"]
        m = tg["model"].split("/")[-1]
        kind = tk["text_kind"]
        target_label = "int" if kind == "reasoning" else "ext"
        dilemma_text = dil.get("dilemma", "") or ""
        target_text = tk.get("target_text", "") or ""
        verdict = tg.get("verdict") or ""
        defense_text = p2.get("defense") or ""
        questions = p2.get("critical_questions") or {}
        questions_block = _format_questions_block(questions)
        n_questions = len(questions) if isinstance(questions, (dict, list)) else 0

        if target_label == "int":
            prompt = fill(PROMPT3_REASONING, {
                "dilemma_with_options_ab": dilemma_text,
                "model_reasoning": target_text,
                "model_answer": verdict,
                "critical_questions": questions_block,
                "model_defense": defense_text,
            })
        else:
            prompt = fill(PROMPT3_JUSTIFICATION, {
                "dilemma_with_options_ab": dilemma_text,
                "model_answer": verdict,
                "model_justification": target_text,
                "critical_questions": questions_block,
                "model_defense": defense_text,
            })
        prepared.append((i, did, m, kind, p4, n_questions, prompt))

    logger.info(
        "Dispatching %d refill calls with concurrency=%d "
        "(persisting every %d completions).",
        len(prepared), concurrency, persist_every,
    )

    refilled = 0
    errors = 0
    completed_since_persist = 0
    pending: list[asyncio.Task] = []

    async def handle(item):
        nonlocal refilled, errors, completed_since_persist
        i, did, m, kind, p4, n_questions, prompt = item
        result = await _refill_one(
            judge, prompt, n_questions, PHASE4_JUDGE_CFG,
            safe_json_parse, _parse_arg_scores, EmptyJudgeCompletionError,
            semaphore,
        )
        if result is None or "_error" in result:
            errors += 1
            logger.warning("refill failed [%d/%d]  %s / %s / %s : %s",
                           i, len(prepared), did, m, kind,
                           result.get("_error") if result else "no result")
            return
        # Patch the Phase-4 cell in samples.json (in-memory mutation; we
        # persist to disk every N completions to keep the cost of a crash
        # bounded.)
        p4["local"] = result["local"]
        p4["global"] = result["global"]
        p4["judge_reasoning"] = result["judge_reasoning"]
        p4["skipped"] = False
        p4.pop("skipped_reason", None)
        p4["refilled_at"] = datetime.now().isoformat(timespec="seconds")
        refilled += 1
        completed_since_persist += 1
        if completed_since_persist >= persist_every:
            samples_path.write_text(
                json.dumps(samples, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            completed_since_persist = 0
            logger.info("checkpoint: %d refilled, %d errors (samples.json saved)",
                        refilled, errors)

    # Dispatch all tasks; the semaphore caps concurrency, asyncio.gather waits.
    for item in prepared:
        pending.append(asyncio.create_task(handle(item)))
    await asyncio.gather(*pending, return_exceptions=False)

    # Final persist
    samples_path.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Refilled %d / %d cells; %d errors.  samples.json updated.",
                refilled, len(work), errors)
    logger.info(
        "Backup of pre-edit samples.json: %s\n"
        "Re-run analysis to regenerate CSVs + plots:\n"
        "  .venv/Scripts/python.exe -m moral_eval.analysis.walton_analysis %s\n"
        "  .venv/Scripts/python.exe -m moral_eval.analysis.walton_plots %s",
        bak, cot_dir, cot_dir,
    )
    return {"to_refill": len(work), "refilled": refilled, "errors": errors}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cot-dir", required=True, type=Path,
                   help="cot directory of the run to refill (contains samples.json).")
    p.add_argument("--judge-alias", default="gpt_5_5",
                   help="Phase-4 judge alias (as written in samples.json's "
                        "phase4_judge field).  Default: gpt_5_5.")
    p.add_argument("--judge-model", default="openai_gpt55",
                   help="models.json key for the judge model.  Default: openai_gpt55.  "
                        "Must match --judge-alias semantically (same model).")
    p.add_argument("--experiments-file", type=Path,
                   default=PROJECT_ROOT / "src" / "config" / "experiments_walton_moralchoice.yaml",
                   help="Experiments YAML the dj_v2 module imports.  Any walton config will "
                        "do — the script overrides the judge target via CLI.")
    p.add_argument("--dry-run", action="store_true",
                   help="Count cells to refill, list them, and exit.  No API calls.")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after this many refills (useful for sanity-testing).")
    p.add_argument("--concurrency", type=int, default=10,
                   help="In-flight judge calls (default 10).  Raise to speed up if the "
                        "provider's rate limit allows; lower if you see 429s.")
    p.add_argument("--persist-every", type=int, default=50,
                   help="Write samples.json to disk every N successful refills "
                        "(default 50).  Limits work lost if the script is interrupted.")
    args = p.parse_args(argv)

    if not args.cot_dir.exists():
        raise SystemExit(f"cot dir not found: {args.cot_dir}")
    if not (args.cot_dir / "samples.json").exists():
        raise SystemExit(f"samples.json missing in {args.cot_dir}")

    _bootstrap(args.experiments_file)
    asyncio.run(_run_async(args.cot_dir, args.judge_alias, args.judge_model,
                           args.dry_run, args.limit,
                           args.concurrency, args.persist_every))
    return 0


if __name__ == "__main__":
    sys.exit(main())
