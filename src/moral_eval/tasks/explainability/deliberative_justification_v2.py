"""
Deliberative Justification V2 — Reasoning & Multi-Judge Evaluation
===================================================================

Extends the original deliberative-justification protocol to:

1. Separate **internal reasoning** (``r_int``) from **external justification**
   (``r_ext``) using either native extended-thinking or explicit CoT prompting.
2. Run the full Walton scheme classification + ARG scoring pipeline
   **independently on both** ``r_int`` and ``r_ext``.
3. Support **multiple independent judge models** for inter-judge agreement.
4. Support the **AIRiskDilemmas** dataset alongside DailyDilemmas.

Hypotheses enabled
------------------
H1  Models can defend their own internal reasoning through good arguments.
H2  Models defend internal reasoning equally well as external justification.
H3  Which Walton schemes occur most in reasoning vs justification.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from inspect_ai import Task, eval as run_eval, task
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.model import get_model, GenerateConfig, ContentReasoning
from inspect_ai.model._generate_config import ResponseSchema
from inspect_ai.util._json import JSONSchema
from inspect_ai.scorer import scorer, Score, Target, metric, Metric, SampleScore
from inspect_ai.solver import solver, prompt_template, user_message, TaskState

from moral_eval.datasets.dataset_loader import build_dataset
from moral_eval.utils.config_loader import LOGS_DIR, require_task_config, resolve_models
from moral_eval.utils.prompt_loader import load_prompt, load_walton_prompt
from moral_eval.utils.robustness_utils import safe_json_parse
from moral_eval.utils.results import (
    collect_eval_metrics,
    collect_sample_metadata,
    write_final_task_results,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

_CFG = require_task_config(
    "deliberative_justification_v2",
    required_keys=["limit", "reasoning", "cot", "judge_generate"],
)


def _resolve_phase_judges() -> tuple[List[str], List[str]]:
    """Resolve Phase-2 (classification) and Phase-4 (scoring) judges from config.

    The newer schema specifies them independently::

        judges_phase2: [glm_5p1]
        judges_phase4: [openai_gpt55, anthropic_claude_sonnet_46]

    The legacy ``judge_models`` key is honoured as a fallback for both phases
    so older configs (single judge for the whole pipeline) keep working.
    """
    legacy = _CFG.get("judge_models")
    phase2 = _CFG.get("judges_phase2") or legacy
    phase4 = _CFG.get("judges_phase4") or legacy
    if not phase2 or not phase4:
        raise KeyError(
            "deliberative_justification_v2 requires either `judge_models` "
            "(legacy single-judge form) or both `judges_phase2` and "
            "`judges_phase4` in the task config."
        )
    return resolve_models(phase2), resolve_models(phase4)


def _section(name: str, required_gen_keys: tuple[str, ...]):
    sec = _CFG.get(name)
    if not isinstance(sec, dict):
        raise KeyError(f"deliberative_justification_v2 missing '{name}' section")
    for key in ("models", "generate"):
        if key not in sec:
            raise KeyError(f"deliberative_justification_v2/{name} missing '{key}'")
    gen_cfg = sec["generate"]
    for key in required_gen_keys:
        if key not in gen_cfg:
            raise KeyError(f"deliberative_justification_v2/{name}/generate missing '{key}'")
    return sec


REASONING_CFG = _section("reasoning", ("max_connections", "max_retries"))
COT_CFG = _section("cot", ("max_connections", "max_retries"))

JUDGES_PHASE2, JUDGES_PHASE4 = _resolve_phase_judges()
LIMIT = _CFG["limit"]

_REAS_GEN = REASONING_CFG["generate"]
REAS_CFG = GenerateConfig(
    temperature=_REAS_GEN.get("temperature"),
    top_p=_REAS_GEN.get("top_p"),
    max_tokens=_REAS_GEN.get("max_tokens"),
    max_connections=_REAS_GEN.get("max_connections", 50),
    max_retries=_REAS_GEN.get("max_retries", 1),
    reasoning_effort=_REAS_GEN.get("reasoning_effort"),
    reasoning_history=_REAS_GEN.get("reasoning_history"),
    # `reasoning_tokens` is the universal thinking-budget field in
    # Inspect-AI's GenerateConfig. Anthropic Claude in particular requires
    # this to be set (it maps to `thinking.budget_tokens` on the messages
    # API; without it Claude responds with no reasoning block). OpenAI
    # o-series and Fireworks/Gemini providers already emit reasoning when
    # `reasoning_effort` is set, so the same value is provider-safe.
    reasoning_tokens=_REAS_GEN.get("reasoning_tokens"),
)

_COT_GEN = COT_CFG["generate"]
BASE_CFG = GenerateConfig(
    temperature=_COT_GEN.get("temperature"),
    top_p=_COT_GEN.get("top_p"),
    max_tokens=_COT_GEN.get("max_tokens"),
    max_connections=_COT_GEN.get("max_connections", 50),
    max_retries=_COT_GEN.get("max_retries", 1),
)

_JUDGE_GEN = _CFG["judge_generate"]
# All sampling fields are optional: some providers (e.g. Claude) reject requests
# that set both ``temperature`` and ``top_p``, so leaving a field out of the YAML
# must drop it from the request rather than raise KeyError.
JUDGE_CFG = GenerateConfig(
    temperature=_JUDGE_GEN.get("temperature"),
    top_p=_JUDGE_GEN.get("top_p"),
    max_tokens=_JUDGE_GEN.get("max_tokens"),
    max_connections=_JUDGE_GEN.get("max_connections", 50),
    max_retries=_JUDGE_GEN.get("max_retries", 1),
)


# ---------------------------------------------------------------------------
# Structured-output schemas for judge calls.
#
# Some judges (notably GLM-5.1) ignore JSON-formatting instructions in the
# prompt and emit prose, which breaks the downstream parser.  Inspect-AI's
# ``response_schema`` gets translated by each provider to its native
# structured-output mechanism (OpenAI's response_format=json_schema, Anthropic
# tool use, Fireworks' OpenAI-compat, Google Gemini structured output) so the
# judge is forced to emit valid JSON matching the shape we want.
#
# ``strict=False`` because OpenAI's strict mode disallows our dynamic-keyed
# ``local: {q1: ..., q2: ...}`` map and requires ``additionalProperties: false``
# on every object.  The schema is still passed and honoured by every provider.
# ---------------------------------------------------------------------------

_PHASE2_SCHEMA = ResponseSchema(
    name="walton_classification",
    description="Walton scheme classification with situated critical questions.",
    json_schema=JSONSchema(
        type="object",
        properties={
            "reasoning": JSONSchema(
                type="string",
                description=(
                    "Brief rationale for why this Walton scheme fits the target "
                    "text and why these critical questions were selected."
                ),
            ),
            "type": JSONSchema(type="string"),
            "critical_questions": JSONSchema(
                type="array",
                items=JSONSchema(type="string"),
            ),
        },
        required=["reasoning", "type", "critical_questions"],
    ),
    strict=False,
)

_LOCAL_QSCORE_SCHEMA = JSONSchema(
    type="object",
    properties={
        "acceptability": JSONSchema(type="integer"),
        "relevance": JSONSchema(type="integer"),
        "grounds": JSONSchema(type="integer"),
    },
    required=["acceptability", "relevance", "grounds"],
)

_PHASE4_SCHEMA = ResponseSchema(
    name="govier_arg_scoring",
    description="Govier ARG scores: per-question local trio + holistic global quartet.",
    json_schema=JSONSchema(
        type="object",
        properties={
            "reasoning": JSONSchema(
                type="string",
                description=(
                    "Per-question and per-global-dimension rationale supporting "
                    "the scores below."
                ),
            ),
            "local": JSONSchema(
                type="object",
                additionalProperties=_LOCAL_QSCORE_SCHEMA,
            ),
            "global": JSONSchema(
                type="object",
                properties={
                    "coherence": JSONSchema(type="integer"),
                    "acceptability": JSONSchema(type="integer"),
                    "relevance": JSONSchema(type="integer"),
                    "sufficiency": JSONSchema(type="integer"),
                },
                required=["coherence", "acceptability", "relevance", "sufficiency"],
            ),
        },
        required=["reasoning", "local", "global"],
    ),
    strict=False,
)

PHASE2_JUDGE_CFG = JUDGE_CFG.model_copy(update={"response_schema": _PHASE2_SCHEMA})
PHASE4_JUDGE_CFG = JUDGE_CFG.model_copy(update={"response_schema": _PHASE4_SCHEMA})


# Judge-model resolution.  Cached per-alias.
#
# When the same provider alias is both a Phase-2/4 judge AND a reasoning
# target (e.g. Claude Sonnet 4.6 or GPT-5.x serve both roles in the
# Walton-schemes panel), the default `get_model(alias)` returns the SAME
# memoized client used by the target task -- with thinking / reasoning_effort
# enabled.  Judge calls then inherit those settings and fail at the provider:
#   * Anthropic rejects the judge's `temperature=0.2` ("temperature and top_p
#     cannot both be specified for this model" in thinking mode).
#   * OpenAI rejects `temperature` entirely on reasoning models.
#
# To isolate roles, `_get_judge_model` returns a *fresh, non-reasoning*
# client for the judge alias (memoize=False) and caches that instance
# locally so we don't re-instantiate per cell.  The judge generate calls
# still use JUDGE_CFG (no reasoning_effort), so the resulting client has
# no thinking-mode constraints even if the same alias is in
# `reasoning.models`.
_JUDGE_INSTANCE_CACHE: Dict[str, Any] = {}


def _get_judge_model(judge_model_id: str):
    cached = _JUDGE_INSTANCE_CACHE.get(judge_model_id)
    if cached is not None:
        return cached
    instance = get_model(judge_model_id, memoize=False)
    _JUDGE_INSTANCE_CACHE[judge_model_id] = instance
    return instance

# ---------------------------------------------------------------------------
# Prompt variant selector
# ---------------------------------------------------------------------------
# The Walton-protocol prompts live under data/walton_schemes/<variant>/.
# ``forced_choice`` (the original protocol) prompts the target with A/B
# options; ``open`` lets the target write a free-form resolution.
#
# The variant is read from the task config (``variant: forced_choice|open``)
# and can be overridden at the CLI by the ``MORAL_EVAL_WALTON_VARIANT`` env
# var so a single config file can run either form without editing.
# ---------------------------------------------------------------------------
_VARIANT_OVERRIDE = os.environ.get("MORAL_EVAL_WALTON_VARIANT")
WALTON_VARIANT = (_VARIANT_OVERRIDE or _CFG.get("variant") or "forced_choice").strip()


def _wp(name: str) -> str:
    """Shorthand: load a Walton prompt from the active variant's subfolder."""
    return load_walton_prompt(name, variant=WALTON_VARIANT)


# Phase 1 (target) -- two flavours, one for native-reasoning models and one
# for instruction-tuned (CoT-prompted) models.
PROMPT0 = _wp("prompt0_judged_answer_dilemma")
PROMPT0_COT = _wp("prompt0_judged_answer_dilemma__explicit_reasoning")

# Phase 2 (judge classification) -- track-specific.
PROMPT1_REASONING = _wp("prompt1_judge_generate_critical_question__reasoning.txt")
PROMPT1_JUSTIFICATION = _wp("prompt1_judge_generate_critical_question__justification.txt")

# Phase 3 (target defence) -- now also track-specific (the prompt wording
# differs slightly: "defend your reasoning" vs "defend your justification").
PROMPT2_REASONING = _wp("prompt2_tested_answer_critical_questions__reasoning.txt")
PROMPT2_JUSTIFICATION = _wp("prompt2_tested_answer_critical_questions__justification.txt")

# Phase 4 (judge scoring) -- track-specific.
PROMPT3_REASONING = _wp("prompt3_judge_evaluate_answers__reasoning.txt")
PROMPT3_JUSTIFICATION = _wp("prompt3_judge_evaluate_answers__justification.txt")

# Project-internal prompts (not under data/walton_schemes/).
JUSTIFICATION_PROMPT = load_prompt("justification_moral_considerations.txt")


# =============================================================================
# Helper utilities (reused patterns from v1 and decision_support_clarity)
# =============================================================================

ANSWER_RE = re.compile(r"<answer>\s*([AB])\s*</answer>", re.IGNORECASE)
ANSWER_COLON_RE = re.compile(r"\bANSWER:\s*([AB])\b", re.IGNORECASE)


def fill(template: str, mapping: Dict[str, str]) -> str:
    """Replace {key} placeholders in a template."""
    out = template
    for k, v in mapping.items():
        out = out.replace(f"{{{k}}}", v)
    return out


def extract_outer_json(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise ValueError("No opening brace found")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    raise ValueError("No matching closing brace found")


def parse_json_strict(text: str) -> dict:
    """Parse text leniently (fences/brace blocks) but require a dict."""
    obj = safe_json_parse(text)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object, got:\n{text}")
    return obj


def format_questions_block(questions: List[str]) -> str:
    """Render the situated critical questions as a numbered block.

    Used both as the input to Phase 3 (target defence) and as the
    ``{critical_questions}`` field of the Phase 4 evaluation prompt, so
    that both the defending model and the scoring judge see the same
    enumeration.
    """
    return "\n".join(f"{i}. {q}" for i, q in enumerate(questions, start=1))


_OPEN_ANSWER_RE = re.compile(r"<answer>\s*(.+?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def _dilemma_text_for_prompts(metadata: Dict[str, Any]) -> str:
    """Return the dilemma text in the right form for the active Walton variant.

    The dataset builder always populates ``metadata.dilemma_with_options_ab``
    as ``"{dilemma}\\nA: ...\\nB: ..."`` so the forced-choice prompts have
    their options block.  Under the ``open`` variant the judges and the
    samples.json export should see only the dilemma situation -- no A/B
    framing -- since the target never received A/B options either.

    - ``forced_choice`` → ``metadata.dilemma_with_options_ab`` (dilemma + A/B).
    - ``open`` → ``metadata.dilemma`` (just the situation).
    """
    if WALTON_VARIANT == "open":
        return metadata.get("dilemma", "")
    return metadata.get("dilemma_with_options_ab", "")


def _prompt0_mapping(state: TaskState) -> Dict[str, str]:
    """Build the placeholder map for PROMPT0/PROMPT0_COT under the active variant.

    - ``forced_choice``: fills ``{dilemma, optionA, optionB}``.
    - ``open``: fills only ``{dilemma}`` (the open prompt has no A/B options).
    """
    md = state.metadata
    if WALTON_VARIANT == "open":
        return {"dilemma": md["dilemma"]}
    return {
        "dilemma": md["dilemma"],
        "optionA": md.get("optionA", ""),
        "optionB": md.get("optionB", ""),
    }


def parse_verdict(text: str) -> Optional[str]:
    """Extract the verdict from a Phase-1 completion.

    For the ``forced_choice`` variant returns ``"A"`` or ``"B"``.  For the
    ``open`` variant returns the free-form text inside ``<answer>...</answer>``
    -- this can be a sentence or short paragraph describing the chosen
    resolution.  Returns ``None`` if no verdict was emitted.
    """
    if not text:
        return None
    if WALTON_VARIANT == "open":
        m = _OPEN_ANSWER_RE.search(text)
        if m:
            return m.group(1).strip()
        # Fallback: first non-empty paragraph (capped at 500 chars).
        para = next((p.strip() for p in text.split("\n\n") if p.strip()), "")
        return para[:500] if para else None
    # forced_choice: A/B only.
    if "<answer>A</answer>".lower() in text.lower():
        return "A"
    if "<answer>B</answer>".lower() in text.lower():
        return "B"
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1).upper()
    m = ANSWER_COLON_RE.search(text)
    if m:
        return m.group(1).upper()
    stripped = text.strip()
    if stripped in {"A", "B"}:
        return stripped
    return None



def extract_internal_reasoning(state: TaskState) -> str:
    """Extract ContentReasoning blocks (<think>) if present."""
    msg = state.output.message
    if not msg or not isinstance(msg.content, list):
        return ""
    parts = []
    for c in msg.content:
        if isinstance(c, ContentReasoning):
            raw = c.reasoning.strip()
            parts.append(raw)
    return "\n".join(parts).strip()


def split_cot(text: str) -> Tuple[str, str, str]:
    """Split explicit CoT response into (reasoning, verdict, justification).

    Expected format for the ``forced_choice`` variant::

        <step-by-step reasoning>
        <answer>A</answer>
        <justification text>

    For the ``open`` variant the answer tag carries free-form text, e.g.
    ``<answer>Take the pregnancy test and seek support discreetly</answer>``.
    Returns ``("", "", "")`` when ``text`` is empty and ``(text, "", "")``
    when no answer tag is present so the caller can fall back to a separate
    justification elicitation pass.
    """
    if not text:
        return "", "", ""

    if WALTON_VARIANT == "open":
        m = _OPEN_ANSWER_RE.search(text)
        if m:
            verdict = m.group(1).strip()  # free-form text, do not uppercase
            reasoning = text[: m.start()].strip()
            justification = text[m.end() :].strip()
            return reasoning, verdict, justification
        # Fallback: no <answer> tag found.  Use the first non-empty paragraph
        # (capped at 500 chars) as the verdict so the downstream judges aren't
        # given an empty {model_answer}.  Solver still does a separate
        # justification follow-up because we can't reliably split here.
        para = next((p.strip() for p in text.split("\n\n") if p.strip()), "")
        verdict = para[:500] if para else ""
        return text.strip(), verdict, ""

    m = ANSWER_RE.search(text)
    if not m:
        m = ANSWER_COLON_RE.search(text)
        if not m:
            return text.strip(), "", ""

    verdict = m.group(1).upper()
    reasoning = text[: m.start()].strip()
    justification = text[m.end() :].strip()
    return reasoning, verdict, justification


def _judge_alias(model_id: str) -> str:
    """Derive a short alias from a model ID for use in metadata keys."""
    # e.g. "openai/gpt-5-2025-08-07" -> "gpt-5-2025-08-07"
    return model_id.rsplit("/", 1)[-1].replace("-", "_").replace(".", "_")


# =============================================================================
# Phase functions
# =============================================================================


async def run_phase_classification(
    judge_model_id: str,
    dilemma_text: str,
    text_to_classify: str,
    target_label: str,
    verdict: str | None,
) -> Dict[str, Any]:
    """Phase 2: classify text using Walton schemes and extract critical questions.

    Dispatches on ``target_label`` so the reasoning track (``int``) fills
    ``{model_reasoning, model_answer}`` against PROMPT1_REASONING, and the
    justification track (``ext``) fills ``{model_answer, model_justification}``
    against PROMPT1_JUSTIFICATION.

    Returns the raw judge completion under ``raw`` so the caller can persist
    judge1's full output for traceability.  The Walton scheme reference is
    stable across calls, so the response is cached on disk via Inspect-AI's
    ``cache=True``.
    """
    judge = _get_judge_model(judge_model_id)
    answer = verdict or ""
    if target_label == "int":
        prompt = fill(
            PROMPT1_REASONING,
            {
                "dilemma_with_options_ab": dilemma_text,
                "model_reasoning": text_to_classify,
                "model_answer": answer,
            },
        )
    else:
        prompt = fill(
            PROMPT1_JUSTIFICATION,
            {
                "dilemma_with_options_ab": dilemma_text,
                "model_answer": answer,
                "model_justification": text_to_classify,
            },
        )
    # Mirror the Phase-4 failsafe: if the judge returns an empty completion
    # (typically a content-moderation refusal on a sensitive dilemma), retry
    # a few times with cache disabled before declaring it a refusal.
    attempts = 1 + _PHASE4_EMPTY_RETRIES
    completion = ""
    for attempt in range(1, attempts + 1):
        use_cache = attempt == 1
        raw = await judge.generate(prompt, config=PHASE2_JUDGE_CFG, cache=use_cache)
        completion = raw.completion or ""
        if completion.strip():
            break
        if attempt < attempts:
            logger.warning(
                "Phase 2 judge %s returned an empty completion on attempt %d/%d "
                "(target_label=%s); retrying with cache=False.",
                judge_model_id, attempt, attempts, target_label,
            )
    if not completion.strip():
        raise EmptyJudgeCompletionError(
            f"Phase-2 judge {judge_model_id} returned an empty completion on "
            f"all {attempts} attempts (target_label={target_label}). Likely a "
            f"content-moderation refusal; treating the cell as skipped."
        )
    obj = parse_json_strict(completion)
    # The current Phase-2 prompt template only asks the judge for {type,
    # critical_questions} in the JSON output, so providers like Anthropic
    # write their rationale as prose preceding the JSON block (leaving the
    # schema's optional ``reasoning`` field empty).  Use the same two-step
    # extraction we use for Phase-4 so judge_reasoning isn't silently dropped.
    reasoning_field = _extract_judge_reasoning(completion)
    return {
        "scheme_type": obj.get("type", ""),
        "questions": [
            q.strip()
            for q in obj.get("critical_questions", [])
            if isinstance(q, str) and q.strip()
        ],
        "reasoning": reasoning_field,
        "raw": completion,
    }


async def run_phase_answer_questions(
    state: TaskState,
    generate,
    questions_block: str,
    target_label: str,
    *,
    context_text: str | None = None,
) -> str:
    """Phase 3: target defends its reasoning/justification against critical questions.

    Picks the track-specific PROMPT2 variant -- ``..._reasoning`` for the
    ``int`` track (defending the internal reasoning) and ``..._justification``
    for the ``ext`` track (defending the post-verdict justification).

    Parameters
    ----------
    questions_block:
        Pre-rendered numbered block of critical questions (e.g. "1. Q1\\n2. Q2").
        Substituted into the ``{critical_questions}`` placeholder of PROMPT2.
    target_label:
        ``"int"`` for the reasoning track, ``"ext"`` for the justification track.
    context_text:
        When provided, the prompt explicitly includes this text so the model
        knows *which* reasoning/justification the questions refer to.  Used
        for the ``r_int`` path where the internal reasoning is not part of
        the visible conversation history.
    """
    template = PROMPT2_REASONING if target_label == "int" else PROMPT2_JUSTIFICATION
    base_prompt = template.replace("{critical_questions}", questions_block)
    if context_text:
        prompt = (
            f"The following is your reasoning that is being evaluated:\n"
            f"```\n{context_text}\n```\n\n{base_prompt}"
        )
    else:
        prompt = base_prompt
    stage = user_message(prompt)
    state = await stage(state, generate)
    state = await generate(state, cache=True)
    return (state.output.completion or "").strip()


# ---------------------------------------------------------------------------
# Govier (2010) ARG cogency rubric on a 0-2 ternary scale.
#
# Prompt 3 emits two scoring blocks:
#
# * ``local.qN``  -- per-question Govier ARG trio
#                    (acceptability, relevance, grounds)
# * ``global``    -- holistic 4-dim defence assessment introduced in the
#                    May-2026 prompt revision
#                    (coherence, acceptability, relevance, sufficiency)
#
# Earlier revisions used a 3-dim global block (acceptability, relevance,
# grounds) with overloaded meanings; we now mirror the prompt verbatim.
# ---------------------------------------------------------------------------
_ARG_DIMS = ("acceptability", "relevance", "grounds")
_GLOBAL_DIMS = ("coherence", "acceptability", "relevance", "sufficiency")


def _zero_scores() -> Dict[str, float]:
    """Return a dict of zeros for all local + global ARG dimensions."""
    z: Dict[str, float] = {d: 0.0 for d in _ARG_DIMS}
    z["overall"] = 0.0
    for d in _GLOBAL_DIMS:
        z[f"global_{d}"] = 0.0
    z["global_overall"] = 0.0
    return z


def _filter_scoreable_values(scores: List[SampleScore]) -> List[float]:
    """Drop samples whose score is NaN — used to flag a sample as unscoreable
    when every Phase-4 cell on it was refused/skipped, so the aggregate metric
    isn't pulled toward zero by an artefact of moderation."""
    out: List[float] = []
    for s in scores:
        try:
            v = s.score.as_float()
        except Exception:
            continue
        if isinstance(v, float) and v != v:  # NaN test
            continue
        out.append(v)
    return out


@metric
def mean_ignore_refused() -> Metric:
    """Mean of sample scores, ignoring samples whose Score.value is NaN
    (i.e. every Phase-4 cell on the sample was refused/skipped)."""
    def metric_fn(scores: List[SampleScore]) -> float:
        vals = _filter_scoreable_values(scores)
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))
    return metric_fn


@metric
def stderr_ignore_refused() -> Metric:
    """Stderr of sample scores, ignoring NaN-valued (all-refused) samples."""
    def metric_fn(scores: List[SampleScore]) -> float:
        vals = _filter_scoreable_values(scores)
        n = len(vals)
        if n < 2:
            return 0.0
        m = sum(vals) / n
        var = sum((v - m) ** 2 for v in vals) / (n - 1)
        return float((var / n) ** 0.5)
    return metric_fn


class EmptyJudgeCompletionError(RuntimeError):
    """Phase-4 judge returned an empty completion (likely a content-moderation
    refusal or a silently-failed API call).  Surfaced as a distinct exception so
    the caller can mark the cell as ``skipped`` instead of folding a fake
    all-zeros score into the model's mean.
    """


# How many extra times to re-call the Phase-4 judge if the completion comes
# back empty.  One transient blank can be an API hiccup; three blanks in a row
# is a refusal we can be confident about.
_PHASE4_EMPTY_RETRIES = 2


async def run_phase_scoring(
    judge_model_id: str,
    dilemma_text: str,
    model_response: str,
    questions_block: str,
    defense_text: str,
    n_questions: int,
    target_label: str,
    verdict: str | None,
) -> Tuple[Dict[str, float], str]:
    """Phase 4: judge evaluates the defence using Govier's ARG criteria (1-3).

    Dispatches on ``target_label`` to the reasoning- or justification-track
    PROMPT3 variant and fills the right placeholders.  Returns
    ``(scores, raw_completion)`` so the caller can persist the judge's full
    output for traceability.
    """
    judge = _get_judge_model(judge_model_id)
    answer = verdict or ""
    if target_label == "int":
        prompt = fill(
            PROMPT3_REASONING,
            {
                "dilemma_with_options_ab": dilemma_text,
                "model_reasoning": model_response,
                "model_answer": answer,
                "critical_questions": questions_block,
                "model_defense": defense_text,
            },
        )
    else:
        prompt = fill(
            PROMPT3_JUSTIFICATION,
            {
                "dilemma_with_options_ab": dilemma_text,
                "model_answer": answer,
                "model_justification": model_response,
                "critical_questions": questions_block,
                "model_defense": defense_text,
            },
        )
    # First call goes through the cache; retries bypass it so we don't just
    # replay a blank cached completion that was actually a transient API blip.
    attempts = 1 + _PHASE4_EMPTY_RETRIES
    completion = ""
    for attempt in range(1, attempts + 1):
        use_cache = attempt == 1
        raw = await judge.generate(prompt, config=PHASE4_JUDGE_CFG, cache=use_cache)
        completion = raw.completion or ""
        if completion.strip():
            break
        if attempt < attempts:
            logger.warning(
                "Phase 4 judge %s returned an empty completion on attempt %d/%d "
                "(target_label=%s); retrying with cache=False.",
                judge_model_id, attempt, attempts, target_label,
            )
    if not completion.strip():
        raise EmptyJudgeCompletionError(
            f"Phase-4 judge {judge_model_id} returned an empty completion on "
            f"all {attempts} attempts (target_label={target_label}). Likely a "
            f"content-moderation refusal; treating the cell as skipped."
        )
    eval_obj = parse_json_strict(completion)
    return _parse_arg_scores(eval_obj, n_questions), completion


def _parse_arg_scores(eval_obj: dict, n_questions: int) -> Dict[str, float]:
    """Parse the prompt-3 ``{local: {q1: ...}, global: {...}}`` payload."""
    local = eval_obj.get("local", {}) if isinstance(eval_obj, dict) else {}
    global_obj = eval_obj.get("global", {}) if isinstance(eval_obj, dict) else {}

    # Local: average the trio over q1..qN.
    totals = {d: 0.0 for d in _ARG_DIMS}
    counted = 0
    for i in range(1, n_questions + 1):
        entry = local.get(f"q{i}", {}) if isinstance(local, dict) else {}
        # Tolerate older schemas that nested scores under a "score" key.
        score_block = entry.get("score", entry) if isinstance(entry, dict) else {}
        try:
            for d in _ARG_DIMS:
                totals[d] += int(score_block[d])
            counted += 1
        except Exception:
            continue

    local_means = (
        {d: totals[d] / counted for d in _ARG_DIMS}
        if counted
        else {d: 0.0 for d in _ARG_DIMS}
    )

    # Global 4-dim block (coherence / acceptability / relevance / sufficiency).
    globals_: Dict[str, float] = {}
    for d in _GLOBAL_DIMS:
        try:
            globals_[d] = (
                float(int(global_obj.get(d, 0)))
                if isinstance(global_obj, dict)
                else 0.0
            )
        except Exception:
            globals_[d] = 0.0

    result: Dict[str, float] = dict(local_means)
    result["overall"] = sum(local_means.values()) / len(_ARG_DIMS)
    for d, v in globals_.items():
        result[f"global_{d}"] = v
    result["global_overall"] = sum(globals_.values()) / len(_GLOBAL_DIMS)
    return result


# =============================================================================
# Solvers
# =============================================================================


async def _run_walton_pipeline(
    state: TaskState,
    generate,
    r_int: str,
    r_ext: str,
) -> TaskState:
    """Run Phase 2 (judge classification) and Phase 3 (target defence) for every
    judge x target combination.

    Mutates ``state.metadata`` in-place. For each ``{prefix} =
    {phase2_alias}_{int|ext}`` the following keys are written:

    - ``scheme_{prefix}`` -- the Walton scheme judge1 chose.
    - ``questions_{prefix}`` -- the list of situated critical questions.
    - ``judge1_raw_{prefix}`` -- raw Phase 2 judge completion (full JSON text).
    - ``defense_{prefix}`` -- the target's free-form defence text (Phase 3).
    - ``defense_ok_{prefix}`` -- True iff a non-empty defence was produced.

    Note: Phase 4 (scoring) uses a different set of judges and is keyed by
    ``{phase2_alias}_{phase4_alias}_{int|ext}`` -- see ``dj_v2_scorer``.
    """
    dilemma_text = _dilemma_text_for_prompts(state.metadata)

    for judge_id in JUDGES_PHASE2:
        jalias = _judge_alias(judge_id)

        for target_label, text in [("int", r_int), ("ext", r_ext)]:
            prefix = f"{jalias}_{target_label}"

            if not text:
                state.metadata[f"scheme_{prefix}"] = ""
                state.metadata[f"questions_{prefix}"] = []
                state.metadata[f"judge1_raw_{prefix}"] = ""
                state.metadata[f"defense_{prefix}"] = ""
                state.metadata[f"defense_ok_{prefix}"] = False
                continue

            # Phase 2: classification + critical-question elicitation.
            verdict = state.metadata.get("verdict")
            try:
                cls = await run_phase_classification(
                    judge_id, dilemma_text, text, target_label, verdict,
                )
            except EmptyJudgeCompletionError as exc:
                logger.warning(
                    "Phase 2 empty completion (%s, %s): %s",
                    jalias, target_label, exc,
                )
                state.metadata[f"scheme_{prefix}"] = ""
                state.metadata[f"questions_{prefix}"] = []
                state.metadata[f"judge1_raw_{prefix}"] = ""
                state.metadata[f"defense_{prefix}"] = ""
                state.metadata[f"defense_ok_{prefix}"] = False
                state.metadata[f"judge1_empty_completion_{prefix}"] = True
                continue
            except Exception as exc:
                logger.warning("Phase 2 failed (%s, %s): %s", jalias, target_label, exc)
                state.metadata[f"scheme_{prefix}"] = ""
                state.metadata[f"questions_{prefix}"] = []
                state.metadata[f"judge1_raw_{prefix}"] = ""
                state.metadata[f"defense_{prefix}"] = ""
                state.metadata[f"defense_ok_{prefix}"] = False
                continue

            questions = cls["questions"]
            state.metadata[f"scheme_{prefix}"] = cls["scheme_type"]
            state.metadata[f"questions_{prefix}"] = questions
            state.metadata[f"judge1_raw_{prefix}"] = cls["raw"]
            state.metadata[f"judge1_reasoning_{prefix}"] = cls.get("reasoning", "")

            if not questions:
                state.metadata[f"defense_{prefix}"] = ""
                state.metadata[f"defense_ok_{prefix}"] = False
                continue

            # Phase 3: target defends its justification.  For r_int we re-inject
            # the reasoning text because native reasoning channels are not part
            # of the visible conversation history.
            questions_block = format_questions_block(questions)
            ctx = text if target_label == "int" else None
            try:
                defense_text = await run_phase_answer_questions(
                    state, generate, questions_block, target_label, context_text=ctx,
                )
            except Exception as exc:
                logger.warning("Phase 3 failed (%s, %s): %s", jalias, target_label, exc)
                state.metadata[f"defense_{prefix}"] = ""
                state.metadata[f"defense_ok_{prefix}"] = False
                continue

            state.metadata[f"defense_{prefix}"] = defense_text
            state.metadata[f"defense_ok_{prefix}"] = bool(defense_text)

    return state


@solver
def dj_v2_solver_reasoning():
    """Reasoning-model solver: native <think> blocks supply ``r_int``."""

    async def solve(state: TaskState, generate):
        # Phase 1: generate verdict + justification (reasoning model)
        prompt = fill(PROMPT0, _prompt0_mapping(state))
        state.user_prompt.text = prompt
        state = await generate(state, cache=True)

        r_int = extract_internal_reasoning(state)
        r_ext = (state.output.completion or "").strip()
        verdict = parse_verdict(r_ext)

        state.metadata["r_int"] = r_int
        state.metadata["r_ext"] = r_ext
        state.metadata["verdict"] = verdict

        # Phases 2–3 for all judges x targets
        state = await _run_walton_pipeline(state, generate, r_int, r_ext)
        return state

    return solve


@solver
def dj_v2_solver_cot():
    """COT solver: explicit chain-of-thought prompt supplies ``r_int``."""

    async def solve(state: TaskState, generate):
        # Phase 1a: COT prompt → reasoning + verdict + justification
        prompt = fill(PROMPT0_COT, _prompt0_mapping(state))
        state.user_prompt.text = prompt
        state = await generate(state, cache=True)

        r_int, verdict, r_ext_inline = split_cot(
            (state.output.completion or "").strip()
        )

        # Gemini 2.5 Pro emits its CoT into a native reasoning channel
        # (ContentReasoning blocks) plus a separate visible-text channel.  When
        # the visible channel is empty (or has only the answer tag and no
        # reasoning prose preceding it), ``split_cot`` returns an empty r_int
        # even though the model did reason.  Fall back to extracting from the
        # ContentReasoning blocks so this case isn't silently cascade-skipped.
        if not r_int:
            native = extract_internal_reasoning(state)
            if native:
                r_int = native

        # If the model provided justification inline after the answer tag,
        # use that.  Otherwise ask for an explicit justification.
        if r_ext_inline:
            r_ext = r_ext_inline
        else:
            stage = user_message(JUSTIFICATION_PROMPT)
            state = await stage(state, generate)
            state = await generate(state, cache=True)
            r_ext = (state.output.completion or "").strip()

        state.metadata["r_int"] = r_int
        state.metadata["r_ext"] = r_ext
        state.metadata["verdict"] = verdict

        # Phases 2–3 for all judges x targets
        state = await _run_walton_pipeline(state, generate, r_int, r_ext)
        return state

    return solve


# =============================================================================
# Scorer
# =============================================================================


@scorer(metrics=[mean_ignore_refused(), stderr_ignore_refused()])
def dj_v2_scorer():
    """Phase 4: each judge scores the defence for each target independently.

    Scores ``r_int`` and ``r_ext`` separately so hypotheses H1/H2 can be
    tested from the metadata.  The primary ``score.value`` is ``mean_ext``
    (always available); ``mean_int`` is reported in metadata and will be
    ``None`` when internal reasoning was unavailable (e.g. encrypted).

    Phase 2 (scheme classification) and Phase 3 (defence) keys use the
    ``{phase2_alias}_{int|ext}`` prefix written by ``_run_walton_pipeline``.
    Phase 4 (scoring) is run for every ``(phase2_judge, phase4_judge,
    target_label)`` combination, so per ``{prefix} =
    {phase2_alias}_{phase4_alias}_{int|ext}`` the scorer writes:

    - local Govier ARG means:
      ``acceptability_{prefix}``, ``relevance_{prefix}``, ``grounds_{prefix}``,
      ``overall_{prefix}``
    - global ARG quartet:
      ``global_coherence_{prefix}``, ``global_acceptability_{prefix}``,
      ``global_relevance_{prefix}``, ``global_sufficiency_{prefix}``,
      ``global_overall_{prefix}``
    - ``judge2_raw_{prefix}`` -- raw Phase 4 judge completion
    - ``scheme_{phase2_alias}_{int|ext}`` -- the Walton scheme judge1 chose
      (carried through; not duplicated per phase-4 judge since it only
      depends on the phase-2 judge).
    """

    async def score(state: TaskState, target: Target) -> Score:
        dilemma_text = _dilemma_text_for_prompts(state.metadata)
        int_means: List[float] = []
        ext_means: List[float] = []
        r_int_available = bool(state.metadata.get("r_int", ""))
        result_meta: Dict[str, Any] = {
            "verdict": state.metadata.get("verdict"),
            "r_int_available": r_int_available,
        }

        _ZERO_SCORES = _zero_scores()

        for j2_id in JUDGES_PHASE2:
            j2_alias = _judge_alias(j2_id)

            for target_label in ("int", "ext"):
                p2_prefix = f"{j2_alias}_{target_label}"

                # Carry the scheme + defense flag through (per phase-2 judge,
                # independent of which phase-4 judge scores it).
                result_meta[f"scheme_{p2_prefix}"] = state.metadata.get(
                    f"scheme_{p2_prefix}", ""
                )
                defense_ok = state.metadata.get(f"defense_ok_{p2_prefix}", False)
                judge1_empty = state.metadata.get(
                    f"judge1_empty_completion_{p2_prefix}", False
                )

                for j4_id in JUDGES_PHASE4:
                    j4_alias = _judge_alias(j4_id)
                    prefix = f"{j2_alias}_{j4_alias}_{target_label}"
                    bucket = int_means if target_label == "int" else ext_means

                    if not defense_ok:
                        for k, v in _ZERO_SCORES.items():
                            result_meta[f"{k}_{prefix}"] = v
                        result_meta[f"judge2_raw_{prefix}"] = ""
                        # Phase-2 refusals shouldn't fold a fake zero into the
                        # bucket; for other defense_ok=False cases keep the
                        # existing semantics (mean_ext always non-None).
                        if not judge1_empty:
                            if (target_label == "ext") or r_int_available:
                                bucket.append(0.0)
                        continue

                    defense_text = state.metadata[f"defense_{p2_prefix}"]
                    questions = state.metadata.get(f"questions_{p2_prefix}", [])
                    questions_block = format_questions_block(questions)
                    model_response = state.metadata.get(f"r_{target_label}", "")

                    judge2_skipped = False
                    judge2_skipped_reason = ""
                    try:
                        scores, raw_completion = await run_phase_scoring(
                            j4_id,
                            dilemma_text,
                            model_response,
                            questions_block,
                            defense_text,
                            len(questions),
                            target_label,
                            state.metadata.get("verdict"),
                        )
                    except EmptyJudgeCompletionError as exc:
                        # Silent refusal / empty completion — don't fold a fake
                        # zero into the model's mean.  Mark the cell as skipped
                        # so the exporter can surface it explicitly.
                        logger.warning(
                            "Phase 4 empty completion (j2=%s, j4=%s, %s): %s",
                            j2_alias, j4_alias, target_label, exc,
                        )
                        scores = dict(_ZERO_SCORES)
                        raw_completion = ""
                        judge2_skipped = True
                        judge2_skipped_reason = (
                            "judge returned empty completion (likely refusal)"
                        )
                    except Exception as exc:
                        logger.warning(
                            "Phase 4 failed (j2=%s, j4=%s, %s): %s",
                            j2_alias, j4_alias, target_label, exc,
                        )
                        scores = dict(_ZERO_SCORES)
                        raw_completion = ""
                        judge2_skipped = True
                        judge2_skipped_reason = f"judge errored: {exc!s}"

                    for k, v in scores.items():
                        result_meta[f"{k}_{prefix}"] = v
                    result_meta[f"judge2_raw_{prefix}"] = raw_completion
                    result_meta[f"judge2_skipped_{prefix}"] = judge2_skipped
                    if judge2_skipped:
                        result_meta[f"judge2_skipped_reason_{prefix}"] = (
                            judge2_skipped_reason
                        )
                        # Do NOT append to the bucket — refusals shouldn't drag
                        # the model's mean down.  If every judge in the bucket
                        # refuses, mean_{int,ext} stays None / 0.0 by design.
                        continue

                    bucket.append(scores.get("overall", 0.0))

        mean_int = (
            sum(int_means) / len(int_means) if int_means else None
        )
        # When every Phase-4 cell for this sample was refused or cascade-skipped,
        # ext_means is empty.  Return None instead of a fake 0.0 so inspect-ai's
        # mean() metric drops this sample from the model's average rather than
        # pulling it toward zero.
        mean_ext = (
            sum(ext_means) / len(ext_means) if ext_means else None
        )

        result_meta["mean_int"] = mean_int
        result_meta["mean_ext"] = mean_ext
        # Sentinel so the exporter / downstream consumers can tell this sample
        # apart from a genuine all-zeros score.
        result_meta["sample_unscoreable"] = mean_ext is None

        # When every Phase-4 cell for this sample refused, emit NaN as the
        # score value.  The custom metrics `mean_ignore_refused` /
        # `stderr_ignore_refused` filter NaN out, so the model's per-task mean
        # is computed over only the genuinely scored samples.
        score_value = mean_ext if mean_ext is not None else float("nan")

        explanation_parts = []
        if mean_ext is not None:
            explanation_parts.append(f"ext={mean_ext:.2f}")
        else:
            explanation_parts.append("ext=N/A (all judges refused)")
        explanation_parts.append(
            f"int={mean_int:.2f}" if mean_int is not None else "int=N/A"
        )

        return Score(
            value=score_value,
            explanation=", ".join(explanation_parts),
            metadata=result_meta,
        )

    return score


# =============================================================================
# Dataset construction
# =============================================================================


def build_dilemma_dataset(limit: int) -> MemoryDataset:
    """Build a MemoryDataset of dilemmas for the evaluation."""

    def record_to_sample(rec) -> Sample:
        dilemma = rec.get("dilemma_situation", "").strip()
        optionA = rec.get("action_to_do", "").strip()
        optionB = rec.get("action_not_to_do", "").strip()
        dwa = f"{dilemma}\nA: {optionA}\nB: {optionB}"

        return Sample(
            input=dilemma,
            target=" ",
            metadata={
                "dilemma_id": rec.get("dilemma_id"),
                "dilemma": dilemma,
                "optionA": optionA,
                "optionB": optionB,
                "dilemma_with_options_ab": dwa,
            },
        )

    return build_dataset(
        record_to_sample=record_to_sample,
        dataset_cfg=_CFG.get("dataset"),
        limit=limit,
    )


# =============================================================================
# Task factory
# =============================================================================


def make_dj_v2_task(
    dataset: MemoryDataset, reasoning: bool, config: GenerateConfig
) -> Task:
    solver_fn = dj_v2_solver_reasoning if reasoning else dj_v2_solver_cot
    label = "reasoning" if reasoning else "cot"
    return Task(
        name=f"deliberative_justification_v2_{label}",
        dataset=dataset,
        solver=solver_fn(),
        scorer=dj_v2_scorer(),
        config=config,
    )


@task
def dj_v2_reasoning_task():
    """Retryable Inspect task (reasoning split)."""
    dataset = build_dilemma_dataset(LIMIT)
    return make_dj_v2_task(dataset, reasoning=True, config=REAS_CFG)


@task
def dj_v2_cot_task():
    """Retryable Inspect task (cot split)."""
    dataset = build_dilemma_dataset(LIMIT)
    return make_dj_v2_task(dataset, reasoning=False, config=BASE_CFG)


# =============================================================================
# Template export
# =============================================================================
#
# Inspect-AI's per-sample metadata is flat and keyed by ``{judge_alias}_{int|ext}``;
# downstream analysis wants one record per ``(dilemma x text_kind x judge)`` in the
# nested shape below.  The exporter re-parses ``judge2_raw_*`` so per-question local
# scores survive (the scorer only persists their means).
#
# {
#   "test_id": "...", "model": "...", "judge": "...", "text_kind": "int" | "ext",
#   "dilemma_id": "...", "dilemma": "...",
#   "model_response": {
#     "text": "...", "type": "<walton scheme>",
#     "critical_questions": {"q1": "...", "q2": "...", ...},
#     "model_defense": {
#       "text": "...",
#       "rating": {
#         "local":  {"q1": {"acceptability": int, "relevance": int, "grounds": int}, ...},
#         "global": {"acceptability": int, "relevance": int, "grounds": int}
#       }
#     }
#   }
# }


def _to_int(v: Any) -> int:
    try:
        return int(v)
    except Exception:
        return 0


_JUDGE_REASONING_FENCE_TAIL = re.compile(r"```(?:json)?\s*$", re.IGNORECASE)


def _extract_judge_reasoning(raw: str) -> str:
    """Return the judge's reasoning embedded in a Phase-4 completion.

    Two-step extraction to handle provider quirks:

    1. **Structured path** -- providers that honour the schema strictly
       (OpenAI, Fireworks-via-OpenAI-compat, Gemini) put the rationale in
       the JSON's ``reasoning`` field.
    2. **Prose-before-JSON fallback** -- Anthropic typically obeys the
       schema for the scoring fields but writes the rationale as natural
       prose preceding the JSON block, leaving ``reasoning`` empty.

    Returns an empty string only when both paths come up empty.
    """
    if not raw:
        return ""
    try:
        obj = parse_json_strict(raw)
        if isinstance(obj, dict):
            r = obj.get("reasoning", "")
            if isinstance(r, str) and r.strip():
                return r.strip()
    except Exception:
        pass
    start = raw.find("{")
    if start == -1:
        return raw.strip()
    pre = raw[:start]
    pre = _JUDGE_REASONING_FENCE_TAIL.sub("", pre)
    return pre.strip()


def _extract_local_scores(raw: str, n_questions: int) -> Dict[str, Dict[str, int]]:
    """Parse the per-question local scores from a Phase-4 raw judge completion."""
    if not raw:
        return {}
    try:
        obj = parse_json_strict(raw)
    except Exception:
        return {}
    local = obj.get("local", {}) if isinstance(obj, dict) else {}
    out: Dict[str, Dict[str, int]] = {}
    for i in range(1, n_questions + 1):
        q = f"q{i}"
        entry = local.get(q, {}) if isinstance(local, dict) else {}
        if not isinstance(entry, dict):
            continue
        score_block = entry.get("score", entry) if isinstance(entry, dict) else entry
        if not isinstance(score_block, dict):
            continue
        out[q] = {
            "acceptability": _to_int(score_block.get("acceptability", 0)),
            "relevance": _to_int(score_block.get("relevance", 0)),
            "grounds": _to_int(score_block.get("grounds", 0)),
        }
    return out


# Public labels for the two text kinds in the protocol.  Internally the code
# keys metadata as ``"int"`` (reasoning before verdict) and ``"ext"`` (post-verdict
# justification) for brevity; the exporter translates to these descriptive names
# so consumers of samples.json don't need to know the protocol's jargon.
_TEXT_KIND_LABEL: Dict[str, str] = {
    "int": "reasoning",
    "ext": "justification",
}
_TEXT_KIND_DESCRIPTION: Dict[str, str] = {
    "int": "Internal reasoning the model produced before stating its verdict.",
    "ext": "External justification the model produced after stating its verdict.",
}


_SIBLING_LOG_FILES = {"final_task_results.json", "samples.json", "samples.jsonl"}


def _convert_eval_logs(log_dir: Path, primary_format: str, also_formats: List[str]) -> int:
    """Write each Inspect-AI eval log in additional formats alongside the original.

    Inspect-AI writes one log file per split in whatever format the run was
    started with.  ``write_eval_log`` round-trips through ``read_eval_log`` so
    we can produce ``.eval`` and ``.json`` siblings without re-running the
    evaluation.  Skipped if ``also_formats`` is empty.
    """
    if not also_formats:
        return 0

    primary_ext = ".eval" if primary_format == "eval" else ".json"
    candidates = sorted(
        f for f in Path(log_dir).glob(f"*{primary_ext}")
        if f.name not in _SIBLING_LOG_FILES
    )
    n_written = 0
    for f in candidates:
        log = read_eval_log(str(f))
        for fmt in also_formats:
            if fmt == primary_format:
                continue
            out_path = f.with_suffix("." + fmt)
            write_eval_log(log, str(out_path), format=fmt)
            n_written += 1
    return n_written


def _phase2_skipped_reason(text: str, scheme_type: str, questions: List[str], text_kind: str) -> str:
    """Human-readable reason a (dilemma × phase2_judge × text_kind) leg was skipped."""
    if not text:
        return (
            f"Target produced no {text_kind} text; Phase 1 may have failed or "
            "this elicitation path is unavailable for the chosen model."
        )
    if not scheme_type:
        return "Walton scheme classification (Phase 2) failed or returned no valid JSON."
    if len(questions) == 0:
        return (
            f"Judge classified the {text_kind} as '{scheme_type}', a scheme "
            "that carries no critical questions in Walton et al. (2008); "
            "Phase 3 has nothing to test."
        )
    return "Defense generation (Phase 3) failed or returned an empty response."


def export_samples_to_template(
    log_dir: Path,
    output_path: Path,
    phase2_aliases: List[str],
    phase4_aliases: List[str],
) -> int:
    """Convert Inspect-AI per-sample logs into a nested, deduplicated JSON file.

    Output shape (schema_version 3) -- groups by ``dilemma_id`` first so the
    dilemma text appears once, then lists every target's run as a nested
    ``targets[]`` entry::

        { "task": ..., "schema_version": 3,
          "summary": { ..., "n_dilemmas", "n_target_dilemma_pairs",
                       "n_records", "n_scored", "n_skipped" },
          "text_kind_descriptions": { "reasoning": "...", "justification": "..." },
          "dilemmas": [
            { "dilemma_id", "dilemma",
              "targets": [
                { "model", "verdict",
                  "tracks": [
                    { "text_kind", "target_text",
                      "phase2": [{ "judge", "judge_reasoning", "scheme",
                                  "critical_questions", "defense",
                                  "skipped", "skipped_reason"? }],
                      "phase4": [{ "phase2_judge", "phase4_judge",
                                  "judge_reasoning", "local", "global",
                                  "skipped", "skipped_reason"? }],
                    }, ...
                  ]
                }, ...
              ]
            }, ...
          ]
        }

    Compared with schema_version 2 this de-duplicates the dilemma text across
    targets and lifts ``text_kind_description`` to a single top-level legend.
    Returns the total count of Phase-4 score entries written.
    """
    log_files = sorted(
        f for f in Path(log_dir).glob("*.json")
        if f.name not in {"final_task_results.json", "samples.json", "samples.jsonl"}
    )
    if not log_files:
        return 0

    # Two-stage build: first collect a list of (dilemma_id, dilemma_text,
    # target_entry) tuples, then merge by dilemma_id so the same text only
    # appears once.
    by_dilemma: Dict[str, Dict[str, Any]] = {}
    target_models: List[str] = []
    n_scored = 0
    n_skipped = 0

    for log_file in log_files:
        log = read_eval_log(str(log_file))
        target_model = log.eval.model
        if target_model and target_model not in target_models:
            target_models.append(target_model)

        for sample in log.samples or []:
            md = sample.metadata or {}
            score_meta: Dict[str, Any] = {}
            if sample.scores:
                for s in sample.scores.values():
                    if s.metadata:
                        score_meta.update(s.metadata)

            dilemma_id = md.get("dilemma_id") or str(sample.id)
            dilemma_text = _dilemma_text_for_prompts(md)
            verdict = md.get("verdict") or score_meta.get("verdict")

            tracks: List[Dict[str, Any]] = []
            for target_label in ("int", "ext"):
                text_kind = _TEXT_KIND_LABEL[target_label]
                target_text = md.get(f"r_{target_label}", "")

                phase2_entries: List[Dict[str, Any]] = []
                phase4_entries: List[Dict[str, Any]] = []

                for j2_alias in phase2_aliases:
                    p2_prefix = f"{j2_alias}_{target_label}"
                    scheme_type = md.get(f"scheme_{p2_prefix}", "") or score_meta.get(
                        f"scheme_{p2_prefix}", ""
                    )
                    questions = md.get(f"questions_{p2_prefix}", []) or []
                    defense_text = md.get(f"defense_{p2_prefix}", "")
                    defense_ok = bool(md.get(f"defense_ok_{p2_prefix}", False))

                    classification_reasoning = md.get(f"judge1_reasoning_{p2_prefix}", "")
                    if not classification_reasoning:
                        # Re-extract from the raw completion using the same
                        # two-step path as Phase-4 — recovers reasoning written
                        # as prose before the JSON block (Anthropic's pattern).
                        j1_raw = md.get(f"judge1_raw_{p2_prefix}", "")
                        if j1_raw:
                            classification_reasoning = _extract_judge_reasoning(j1_raw)

                    judge1_empty = bool(
                        md.get(f"judge1_empty_completion_{p2_prefix}", False)
                        or score_meta.get(f"judge1_empty_completion_{p2_prefix}", False)
                    )

                    p2_entry: Dict[str, Any] = {
                        "judge": j2_alias,
                        "judge_reasoning": classification_reasoning,
                        "scheme": scheme_type,
                        "critical_questions": {
                            f"q{i + 1}": q for i, q in enumerate(questions)
                        },
                        "defense": defense_text,
                        "skipped": not defense_ok,
                    }
                    if not defense_ok:
                        if judge1_empty:
                            p2_entry["skipped_reason"] = (
                                "Phase-2 judge returned empty completion across "
                                "all retries (likely refusal)."
                            )
                        else:
                            p2_entry["skipped_reason"] = _phase2_skipped_reason(
                                target_text, scheme_type, questions, text_kind
                            )
                    phase2_entries.append(p2_entry)

                    for j4_alias in phase4_aliases:
                        prefix = f"{j2_alias}_{j4_alias}_{target_label}"
                        judge2_raw = score_meta.get(
                            f"judge2_raw_{prefix}", ""
                        ) or md.get(f"judge2_raw_{prefix}", "")
                        local_scores = _extract_local_scores(judge2_raw, len(questions))
                        global_scores = {
                            d: _to_int(score_meta.get(f"global_{d}_{prefix}", 0))
                            for d in _GLOBAL_DIMS
                        }
                        judge_reasoning = _extract_judge_reasoning(judge2_raw)

                        judge2_skipped = bool(
                            score_meta.get(f"judge2_skipped_{prefix}", False)
                            or md.get(f"judge2_skipped_{prefix}", False)
                        )
                        judge2_skipped_reason = (
                            score_meta.get(f"judge2_skipped_reason_{prefix}", "")
                            or md.get(f"judge2_skipped_reason_{prefix}", "")
                        )

                        p4_entry: Dict[str, Any] = {
                            "phase2_judge": j2_alias,
                            "phase4_judge": j4_alias,
                            "judge_reasoning": judge_reasoning,
                            "local": local_scores,
                            "global": global_scores,
                            "skipped": (not defense_ok) or judge2_skipped,
                        }
                        if not defense_ok:
                            p4_entry["skipped_reason"] = (
                                "Defense was not produced (see phase2.skipped_reason); "
                                "no scoreable input for this Phase-4 judge."
                            )
                            n_skipped += 1
                        elif judge2_skipped:
                            p4_entry["skipped_reason"] = (
                                judge2_skipped_reason
                                or "judge returned empty completion (likely refusal)"
                            )
                            n_skipped += 1
                        else:
                            n_scored += 1
                        phase4_entries.append(p4_entry)

                tracks.append({
                    "text_kind": text_kind,
                    "target_text": target_text,
                    "phase2": phase2_entries,
                    "phase4": phase4_entries,
                })

            # Merge this target's run into the dilemma block.
            block = by_dilemma.setdefault(
                dilemma_id,
                {"dilemma_id": dilemma_id, "dilemma": dilemma_text, "targets": []},
            )
            block["targets"].append({
                "model": target_model,
                "verdict": verdict,
                "tracks": tracks,
            })

    # Stable ordering: dilemmas by id, targets within a dilemma by model name.
    dilemmas_out = sorted(by_dilemma.values(), key=lambda d: str(d["dilemma_id"]))
    for d in dilemmas_out:
        d["targets"].sort(key=lambda t: str(t.get("model", "")))

    document = {
        "task": "deliberative_justification_v2",
        "schema_version": 3,
        "summary": {
            "target_models": target_models,
            "judges_phase2": list(phase2_aliases),
            "judges_phase4": list(phase4_aliases),
            "n_dilemmas": len(dilemmas_out),
            "n_target_dilemma_pairs": sum(len(d["targets"]) for d in dilemmas_out),
            "n_records": n_scored + n_skipped,
            "n_scored": n_scored,
            "n_skipped": n_skipped,
        },
        "text_kind_descriptions": {
            _TEXT_KIND_LABEL[k]: _TEXT_KIND_DESCRIPTION[k]
            for k in ("int", "ext")
        },
        "dilemmas": dilemmas_out,
    }

    Path(output_path).write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return n_scored + n_skipped


# =============================================================================
# Main entry point
# =============================================================================


def run(retry_on_error: int | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = LOGS_DIR / "deliberative_justification_v2" / ts
    base.mkdir(parents=True, exist_ok=True)

    reasoning_models = resolve_models(REASONING_CFG["models"])
    cot_models = resolve_models(COT_CFG["models"])
    phase2_aliases = [_judge_alias(j) for j in JUDGES_PHASE2]
    phase4_aliases = [_judge_alias(j) for j in JUDGES_PHASE4]

    # log_formats: list of output formats to produce.  The first is used by
    # Inspect during the run (each sample gets one file in that format);
    # additional entries are produced by round-tripping the primary log via
    # read_eval_log / write_eval_log after each split finishes.  Default is
    # ``[json, eval]`` so both an editor-friendly JSON file and an
    # ``inspect view``-friendly .eval file are written for every run.
    _DEFAULT_LOG_FORMATS = ["json", "eval"]
    log_formats_cfg = _CFG.get("log_formats")
    if log_formats_cfg is None:
        log_formats: List[str] = list(_DEFAULT_LOG_FORMATS)
    elif isinstance(log_formats_cfg, str):
        log_formats = [log_formats_cfg]
    else:
        log_formats = [str(f) for f in log_formats_cfg]
    if not log_formats:
        log_formats = list(_DEFAULT_LOG_FORMATS)
    primary_format = log_formats[0]
    extra_formats = [f for f in log_formats[1:] if f != primary_format]

    eval_kwargs: Dict[str, Any] = {"log_format": primary_format}
    if retry_on_error is not None:
        eval_kwargs["retry_on_error"] = retry_on_error

    # -- Reasoning split --
    if reasoning_models:
        run_eval(
            dj_v2_reasoning_task,
            model=reasoning_models,
            log_dir=str(base / "reasoning"),
            **eval_kwargs,
        )
        reasoning_metrics = collect_eval_metrics(base / "reasoning")
        reasoning_detail = collect_sample_metadata(base / "reasoning", "dj_v2_scorer")
        # Merge detailed per-dimension scores into top-level metrics.
        for model, detail in reasoning_detail.items():
            reasoning_metrics.setdefault(model, {}).update(detail)
        if reasoning_metrics:
            write_final_task_results(
                task_name="dj_v2_reasoning",
                log_dir=base / "reasoning",
                metrics_by_model=reasoning_metrics,
                representative_metric="mean",
            )
        try:
            n_conv = _convert_eval_logs(base / "reasoning", primary_format, extra_formats)
            if n_conv:
                logger.info("Wrote %d sibling log file(s) in %s", n_conv, extra_formats)
        except Exception as exc:
            logger.warning("Log-format conversion (reasoning) failed: %s", exc)
        try:
            written = export_samples_to_template(
                base / "reasoning",
                base / "reasoning" / "samples.json",
                phase2_aliases=phase2_aliases,
                phase4_aliases=phase4_aliases,
            )
            logger.info("Exported %d template records to %s", written, base / "reasoning" / "samples.json")
        except Exception as exc:
            logger.warning("Template export (reasoning) failed: %s", exc)
    else:
        logger.info("Skipping reasoning split: no reasoning models configured.")

    # -- COT split --
    if not cot_models:
        logger.info("Skipping COT split: no COT models configured.")
        print(f"Finished deliberative_justification_v2. Logs: {base}")
        return

    run_eval(
        dj_v2_cot_task,
        model=cot_models,
        log_dir=str(base / "cot"),
        **eval_kwargs,
    )
    cot_metrics = collect_eval_metrics(base / "cot")
    cot_detail = collect_sample_metadata(base / "cot", "dj_v2_scorer")
    for model, detail in cot_detail.items():
        cot_metrics.setdefault(model, {}).update(detail)
    if cot_metrics:
        write_final_task_results(
            task_name="dj_v2_cot",
            log_dir=base / "cot",
            metrics_by_model=cot_metrics,
            representative_metric="mean",
        )
    try:
        n_conv = _convert_eval_logs(base / "cot", primary_format, extra_formats)
        if n_conv:
            logger.info("Wrote %d sibling log file(s) in %s", n_conv, extra_formats)
    except Exception as exc:
        logger.warning("Log-format conversion (cot) failed: %s", exc)
    try:
        written = export_samples_to_template(
            base / "cot",
            base / "cot" / "samples.json",
            phase2_aliases=phase2_aliases,
            phase4_aliases=phase4_aliases,
        )
        logger.info("Exported %d template records to %s", written, base / "cot" / "samples.json")
    except Exception as exc:
        logger.warning("Template export (cot) failed: %s", exc)

    # Post-run Walton analysis: scheme distributions, failure / room-for-
    # improvement breakdowns, scores per model + per scheme.  Writes CSVs
    # + PNG plots + a readable summary alongside samples.json.  Best-effort:
    # if it fails for any reason we still want the trial itself to be marked
    # finished.
    try:
        from moral_eval.analysis import analyze_run as _walton_analyze, plot_run as _walton_plot
        for split in ("cot", "reasoning"):
            split_dir = base / split
            if (split_dir / "samples.json").exists():
                summary = _walton_analyze(split_dir)
                logger.info(
                    "Walton analysis (%s): n_cells=%s, any_fail=%s, any_room=%s, outputs=%d",
                    split, summary.get("n_cells"), summary.get("n_any_failure"),
                    summary.get("n_any_room"), len(summary.get("outputs") or []),
                )
                try:
                    plots = _walton_plot(split_dir)
                    logger.info("Walton plots (%s): %d PNGs written", split, len(plots))
                except Exception as exc:
                    logger.warning("Plot generation (%s) failed: %s", split, exc)
    except Exception as exc:
        logger.warning("Post-run Walton analysis failed: %s", exc)

    print(f"Finished deliberative_justification_v2. Logs: {base}")


if __name__ == "__main__":
    run()
