from __future__ import annotations

import ast
import json
import os
from typing import Any, Callable, Dict, Iterable, List, Tuple

from inspect_ai.dataset import MemoryDataset, Sample

from moral_eval.datasets.daily_dilemmas.base import DailyDilemmas
from moral_eval.datasets.daily_dilemmas.utils import combine_outcome
from moral_eval.datasets.daily_dilemmas.perspectives import DailyDilemmasPerspectives
from moral_eval.datasets.daily_dilemmas.perturbations import DailyDilemmasPerturbations
from moral_eval.datasets.moralchoice.moralchoice import MoralChoice
from moral_eval.datasets.moralchoice.perspectives import MoralChoicePerspectives
from moral_eval.datasets.moralchoice.perturbations import MoralChoicePerturbations
from moral_eval.datasets.ai_risk_dilemmas.loader import load_ai_risk_dilemmas

DatasetRecord = Dict[str, Any]
RecordToSample = Callable[[DatasetRecord], Sample]

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DEFAULT_DATASET = "daily_dilemmas"

_ALIASES = {
    "daily_dilemmas": "daily_dilemmas",
    "daily-dilemmas": "daily_dilemmas",
    "dd": "daily_dilemmas",
    "daily_dilemmas_perspectives": "daily_dilemmas_perspectives",
    "daily-dilemmas-perspectives": "daily_dilemmas_perspectives",
    "dd_perspectives": "daily_dilemmas_perspectives",
    "daily_dilemmas_perturbations": "daily_dilemmas_perturbations",
    "daily-dilemmas-perturbations": "daily_dilemmas_perturbations",
    "dd_perturbations": "daily_dilemmas_perturbations",
    "moral_choice": "moral_choice",
    "moralchoice": "moral_choice",
    "moral_choice_perspectives": "moral_choice_perspectives",
    "moralchoice_perspectives": "moral_choice_perspectives",
    "moral_choice_perturbations": "moral_choice_perturbations",
    "moralchoice_perturbations": "moral_choice_perturbations",
    "ai_risk_dilemmas": "ai_risk_dilemmas",
    "airisk": "ai_risk_dilemmas",
    "air": "ai_risk_dilemmas",
}


def _ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, (int, float)):
        return [str(value)]

    text = str(value).strip()
    if not text:
        return []

    # Try JSON / ast parsing first
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except Exception:
            continue

    # Fall back to single-item list
    return [text]


def _normalize_daily(record: Dict[str, Any]) -> DatasetRecord:
    return {
        "dataset": "daily_dilemmas",
        "dilemma_id": str(record.get("dilemma_idx", "")).strip(),
        "dilemma_idx": str(record.get("dilemma_idx", "")).strip(),
        "dilemma_situation": str(record.get("dilemma_situation", "")).strip(),
        "action_to_do": str(record.get("action_to_do", "")).strip(),
        "action_not_to_do": str(record.get("action_not_to_do", "")).strip(),
        "neg_consequence_to_do": str(record.get("neg_consequence_to_do", "")).strip(),
        "neg_consequence_not_to_do": str(record.get("neg_consequence_not_to_do", "")).strip(),
        "values_to_do": _ensure_list(record.get("values_to_do")),
        "values_not_to_do": _ensure_list(record.get("values_not_to_do")),
        "topic": str(record.get("topic", "")).strip(),
        "topic_group": str(record.get("topic_group", "")).strip(),
        "raw": record,
    }


def _normalize_moral_choice(record: Dict[str, Any]) -> DatasetRecord:
    return {
        "dataset": "moral_choice",
        "dilemma_id": str(record.get("scenario_id", "")).strip(),
        "dilemma_idx": str(record.get("scenario_id", "")).strip(),
        "dilemma_situation": str(record.get("context", "")).strip(),
        "action_to_do": str(record.get("action1", "")).strip(),
        "action_not_to_do": str(record.get("action2", "")).strip(),
        "neg_consequence_to_do": str(record.get("neg_consequence_to_do", "") or "").strip(),
        "neg_consequence_not_to_do": str(record.get("neg_consequence_not_to_do", "") or "").strip(),
        "values_to_do": _ensure_list(record.get("values_to_do")),
        "values_not_to_do": _ensure_list(record.get("values_not_to_do")),
        "topic": str(record.get("generation_rule", "") or record.get("topic", "")).strip(),
        "topic_group": str(record.get("generation_type", "") or record.get("topic_group", "")).strip(),
        "raw": record,
    }


def _load_daily_dilemmas(params: Dict[str, Any], limit: int | None) -> List[DatasetRecord]:
    daily = DailyDilemmas(
        split=params.get("split", "train"),
        data_file=params.get("data_file", DailyDilemmas().data_file),
    )
    dataset = daily.build_merged(limit=limit)
    return [_normalize_daily(s.metadata) for s in dataset]


def _load_moral_choice(params: Dict[str, Any], limit: int | None) -> List[DatasetRecord]:
    mc = MoralChoice(
        ambiguity=params.get("ambiguity", "low"),
        template=params.get("template", "ab"),
        include_system_instruction=params.get("include_system_instruction", True),
        include_both_orderings=params.get("include_both_orderings", True),
        split=params.get("split", "train"),
    )
    dataset = mc.build_merged(limit=limit)
    return [_normalize_moral_choice(s.metadata) for s in dataset]


DATASET_LOADERS: Dict[str, Callable[[Dict[str, Any], int | None], List[DatasetRecord]]] = {
    "daily_dilemmas": _load_daily_dilemmas,
    "moral_choice": _load_moral_choice,
    "daily_dilemmas_perspectives": lambda params, limit: [
        _normalize_perspective_record({**s.metadata, "scenario": s.input}, "daily_dilemmas_perspectives")
        for s in DailyDilemmasPerspectives(
            data_file=params.get("data_file", DailyDilemmasPerspectives().data_file)
        ).load(limit=limit)
    ],
    "moral_choice_perspectives": lambda params, limit: [
        _normalize_perspective_record({**s.metadata, "scenario": s.input}, "moral_choice_perspectives")
        for s in MoralChoicePerspectives(
            ambiguity=params.get("ambiguity", "low"),
            data_file=params.get("data_file"),
        ).load(limit=limit)
    ],
    "daily_dilemmas_perturbations": lambda params, limit: [
        _normalize_perturbation_record(s.metadata, "daily_dilemmas_perturbations")
        for s in DailyDilemmasPerturbations(
            data_file=params.get("data_file", DailyDilemmasPerturbations().data_file)
        ).load(limit=limit, max_variants=params.get("max_variants"))
    ],
    "moral_choice_perturbations": lambda params, limit: [
        _normalize_perturbation_record(s.metadata, "moral_choice_perturbations")
        for s in MoralChoicePerturbations(
            ambiguity=params.get("ambiguity", "low"),
            data_file=params.get("data_file"),
        ).load(limit=limit, max_variants=params.get("max_variants"))
    ],
    "ai_risk_dilemmas": load_ai_risk_dilemmas,
}


def _canonical_name(name: str | None) -> str:
    if name is None:
        return DEFAULT_DATASET
    return _ALIASES.get(name, name)


def resolve_dataset_spec(dataset_cfg: Dict[str, Any] | None, default: str = DEFAULT_DATASET) -> Tuple[str, Dict[str, Any]]:
    """
    Resolve dataset name/params from config and environment override.
    """
    cfg = dataset_cfg or {}
    env_override = os.environ.get("MORAL_EVAL_DATASET")
    name = _canonical_name(env_override or cfg.get("name") or default)
    params = cfg.get("params", {}) if isinstance(cfg, dict) else {}
    return name, params


def _load_selection_file(selection_path: str | os.PathLike, limit: int | None) -> List[DatasetRecord]:
    """Read pre-sampled records produced by scripts/prepare_selections.py.

    The selection file is a JSON document with a ``records`` array.  Bypasses
    the dataset-specific HF loaders entirely so runs are deterministic across
    upstream dataset changes and require no network access.
    """
    path = os.fspath(selection_path)
    if not os.path.isabs(path):
        # __file__ is .../src/moral_eval/datasets/dataset_loader.py; resolve
        # relative selection_file paths against the project root (three levels up).
        project_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        path = os.path.join(project_root, path)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    records = payload.get("records") or []
    if not isinstance(records, list):
        raise ValueError(f"selection file '{path}' has no 'records' list")
    if limit is not None:
        records = records[:limit]
    return records


def load_dataset_records(
    *,
    dataset_cfg: Dict[str, Any] | None = None,
    name: str | None = None,
    limit: int | None = None,
) -> List[DatasetRecord]:
    """
    Load canonicalized dilemma records for a dataset.

    If ``dataset_cfg["selection_file"]`` is set, records are read verbatim
    from that pre-sampled file (see ``scripts/prepare_selections.py``).
    Otherwise the dataset's registered loader is invoked normally.
    """
    cfg = dataset_cfg or {}
    selection_file = cfg.get("selection_file")
    if selection_file:
        return _load_selection_file(selection_file, limit)

    resolved_name, params = resolve_dataset_spec(dataset_cfg, default=name or DEFAULT_DATASET)
    if resolved_name not in DATASET_LOADERS:
        raise KeyError(f"Unknown dataset '{resolved_name}'. Available: {sorted(DATASET_LOADERS)}")

    loader = DATASET_LOADERS[resolved_name]
    return loader(params, limit)


def _normalize_perspective_record(record: Dict[str, Any], dataset_name: str) -> DatasetRecord:
    return {
        "dataset": dataset_name,
        "dilemma_id": str(record.get("dilemma_id") or record.get("scenario_id") or record.get("dilemma_idx") or "").strip(),
        "perspective": str(record.get("perspective", "")).strip(),
        "scenario": str(record.get("scenario") or record.get("dilemma_situation") or "").strip(),
        "action_to_do": str(record.get("to_do_action") or record.get("action_to_do") or "").strip(),
        "action_not_to_do": str(record.get("not_to_do_action") or record.get("action_not_to_do") or "").strip(),
        "topic": str(record.get("topic", "")).strip(),
        "topic_group": str(record.get("topic_group", "")).strip(),
        "raw": record,
    }


def _normalize_perturbation_record(record: Dict[str, Any], dataset_name: str) -> DatasetRecord:
    return {
        "dataset": dataset_name,
        "dilemma_id": str(record.get("dilemma_id") or record.get("scenario_id") or record.get("original_dilemma_idx") or "").strip(),
        "variant": int(record.get("variant", 0)) if str(record.get("variant", "")).strip() else 0,
        "dilemma_situation": str(record.get("dilemma_situation", "")).strip(),
        "action_to_do": str(record.get("action_to_do", "")).strip(),
        "action_not_to_do": str(record.get("action_not_to_do", "")).strip(),
        "neg_consequence_to_do": str(record.get("neg_consequence_to_do", record.get("negative_consequence_to_do", record.get("negative_consequence")) or "")).strip(),
        "neg_consequence_not_to_do": str(record.get("neg_consequence_not_to_do", record.get("negative_consequence_not_to_do", record.get("negative_consequence")) or "")).strip(),
        "topic": str(record.get("topic", "")).strip(),
        "topic_group": str(record.get("topic_group", "")).strip(),
        "raw": record,
    }


def build_dataset(
    *,
    record_to_sample: RecordToSample | None = None,
    dataset_cfg: Dict[str, Any] | None = None,
    name: str | None = None,
    limit: int | None = None,
) -> MemoryDataset:
    """
    Build a MemoryDataset from any registered dataset by name.

    - record_to_sample is called with canonicalized records.
    - If record_to_sample is None, a simple Sample(input=scenario) is created.
    """
    records = load_dataset_records(dataset_cfg=dataset_cfg, name=name, limit=limit)

    if record_to_sample is None:
        samples = [
            Sample(
                id=rec["dilemma_id"] or None,
                input=rec["dilemma_situation"],
                target=" ",
                metadata=rec,
            )
            for rec in records
        ]
        return MemoryDataset(samples)

    samples = [record_to_sample(rec) for rec in records]
    return MemoryDataset(samples)


def build_flat_options(
    records: Iterable[DatasetRecord],
    *,
    use_topic_group: bool = True,
) -> List[Tuple[str, str, DatasetRecord]]:
    """
    Flatten canonical records into (category, option_text, metadata) tuples.
    """
    cat_field = "topic_group" if use_topic_group else "topic"
    out: List[Tuple[str, str, DatasetRecord]] = []
    for rec in records:
        cat = str(rec.get(cat_field, "")).strip()
        option = combine_outcome(rec.get("action_to_do", ""), rec.get("neg_consequence_to_do", ""))
        if option:
            out.append((cat, option, rec))
        option_b = combine_outcome(rec.get("action_not_to_do", ""), rec.get("neg_consequence_not_to_do", ""))
        if option_b:
            out.append((cat, option_b, rec))
    return out


__all__ = [
    "DatasetRecord",
    "RecordToSample",
    "build_dataset",
    "build_flat_options",
    "load_dataset_records",
    "resolve_dataset_spec",
    "DEFAULT_DATASET",
]
