from __future__ import annotations

import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable


def _load_eval_log(path: Path) -> Dict[str, Any] | None:
    """
    Load a single eval log.

    Supports plain JSON logs and zipped `.eval` archives (Inspect default).
    Returns the decoded header dict or None if the file cannot be parsed.
    """
    try:
        if path.suffix == ".eval":
            with zipfile.ZipFile(path) as zf:
                return json.loads(zf.read("header.json"))

        data = json.loads(path.read_text(encoding="utf-8"))
        # Eval logs always contain an "eval" block.
        return data if isinstance(data, dict) and "eval" in data else None
    except Exception:
        return None


def _load_eval_samples(path: Path) -> tuple:
    """Load model ID and sample dicts from a zipped ``.eval`` archive.

    Returns ``(model_id, [sample_dict, ...])``."""
    with zipfile.ZipFile(path) as zf:
        header = json.loads(zf.read("header.json"))
        model = (header.get("eval") or {}).get("model")
        samples = []
        for name in zf.namelist():
            if name.startswith("samples/") and name.endswith(".json"):
                samples.append(json.loads(zf.read(name)))
    return model, samples


def _extract_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Flatten InspectAI results into a metric -> value mapping.

    If multiple scorer entries are present, metric keys are prefixed with the
    scorer name to avoid collisions (e.g., `scorer_mean`).
    """
    scores = results.get("scores") or []
    multiple_scorers = len(scores) > 1

    metrics: Dict[str, float] = {}
    for score in scores:
        metric_block = score.get("metrics") or {}
        prefix = f"{score.get('name') or score.get('scorer')}_"

        for name, meta in metric_block.items():
            key = f"{prefix}{name}" if multiple_scorers else name
            val = meta.get("value") if isinstance(meta, dict) else None
            if isinstance(val, (int, float)):
                metrics[key] = float(val)

    return metrics


def collect_eval_metrics(log_dir: Path) -> Dict[str, Dict[str, float]]:
    """
    Walk a log directory and aggregate metrics per model.

    Returns:
      {model_id: {metric_name: mean_value}}
    """
    totals: Dict[str, Dict[str, list[float]]] = {}

    for path in log_dir.rglob("*"):
        if path.suffix not in (".json", ".eval") or not path.is_file():
            continue
        if path.name == "final_task_results.json":
            continue

        data = _load_eval_log(path)
        if not data:
            continue

        model = (data.get("eval") or {}).get("model")
        results = data.get("results") or {}
        if not model or not results:
            continue

        metrics = _extract_metrics(results)
        if not metrics:
            continue

        model_totals = totals.setdefault(model, {})
        for metric_name, value in metrics.items():
            model_totals.setdefault(metric_name, []).append(value)

    # Mean-aggregate any repeated metrics for the same model.
    aggregated: Dict[str, Dict[str, float]] = {}
    for model, metric_lists in totals.items():
        aggregated[model] = {name: sum(vals) / len(vals) for name, vals in metric_lists.items() if vals}

    return aggregated


def collect_sample_metadata(
    log_dir: Path,
    scorer_name: str = "dj_v2_scorer",
    *,
    numeric_only: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Aggregate per-sample scorer metadata into per-model means.

    InspectAI's top-level ``results.scores[].metrics`` only reports the
    primary score statistics (mean/stderr).  This function digs into the
    sample-level ``scores.<scorer>.metadata`` to extract the detailed
    dimension scores stored there.

    Parameters
    ----------
    log_dir:
        Directory containing ``.json`` or ``.eval`` InspectAI log files.
    scorer_name:
        Name of the scorer whose sample metadata to extract.
    numeric_only:
        If True (default), only aggregate keys with numeric values.

    Returns
    -------
    ``{model_id: {metric_key: mean_value}}``
    """
    accumulators: Dict[str, Dict[str, list]] = {}

    for path in log_dir.rglob("*"):
        if path.suffix not in (".json", ".eval") or not path.is_file():
            continue
        if path.name == "final_task_results.json":
            continue

        try:
            if path.suffix == ".eval":
                model, samples = _load_eval_samples(path)
            else:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict) or "eval" not in data:
                    continue
                model = (data.get("eval") or {}).get("model")
                samples = data.get("samples") or []
        except Exception:
            continue

        if not model or not samples:
            continue

        model_acc = accumulators.setdefault(model, {})

        for sample in samples:
            scores_block = sample.get("scores") or {}
            scorer_data = scores_block.get(scorer_name) or {}
            meta = scorer_data.get("metadata") or {}

            for key, val in meta.items():
                if numeric_only and not isinstance(val, (int, float)):
                    continue
                if val is None:
                    continue
                model_acc.setdefault(key, []).append(float(val))

    # Compute means.
    aggregated: Dict[str, Dict[str, float]] = {}
    for model, metric_lists in accumulators.items():
        aggregated[model] = {
            name: sum(vals) / len(vals)
            for name, vals in metric_lists.items()
            if vals
        }

    return aggregated


def write_final_task_results(
    task_name: str,
    log_dir: Path,
    metrics_by_model: Dict[str, Dict[str, float]],
    representative_metric: str,
    *,
    timestamp: str | None = None,
    metadata: Dict[str, Any] | None = None,
) -> Path:
    """
    Persist a standardized per-task results file.
    """
    payload: Dict[str, Any] = {
        "task": task_name,
        "timestamp": timestamp or log_dir.name,
        "log_dir": str(log_dir),
        "representative_metric": representative_metric,
        "models": metrics_by_model,
    }

    if metadata:
        payload["metadata"] = metadata

    out_path = log_dir / "final_task_results.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _timestamp_key(ts: str | None) -> datetime:
    if not ts:
        return datetime.min
    for fmt in ("%Y%m%d_%H%M%S", "%Y-%m-%dT%H-%M-%S"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return datetime.min


def aggregate_run_results(run_root: Path, *, task_order: Iterable[str] | None = None) -> Dict[str, Any]:
    """
    Build a run-level table from all `final_task_results.json` files
    under a run directory.
    """
    if not run_root.exists():
        return {}

    files = sorted(run_root.rglob("final_task_results.json"))
    tasks: Dict[str, Dict[str, Any]] = {}

    for file in files:
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except Exception:
            continue

        task = data.get("task")
        if not task:
            continue

        ts = data.get("timestamp")
        existing = tasks.get(task)
        if existing and _timestamp_key(existing.get("timestamp")) > _timestamp_key(ts):
            continue

        tasks[task] = {
            "timestamp": ts,
            "representative_metric": data.get("representative_metric"),
            "path": str(file.relative_to(run_root)),
            "models": data.get("models", {}),
            "log_dir": data.get("log_dir"),
        }

    if not tasks:
        return {}

    task_names = [t for t in task_order if t in tasks] if task_order else []
    extras = [t for t in sorted(tasks.keys()) if t not in task_names]
    task_names.extend(extras)
    models = set()
    for info in tasks.values():
        models.update(info.get("models", {}).keys())
    sorted_models = sorted(models)

    table = []
    for model in sorted_models:
        row = {"model": model}
        for task in task_names:
            info = tasks.get(task, {})
            metrics = (info.get("models") or {}).get(model, {})
            rep = info.get("representative_metric")
            row[task] = metrics.get(rep)
        table.append(row)

    return {
        "run_dir": str(run_root),
        "generated_at": datetime.now().isoformat(),
        "tasks": tasks,
        "table": table,
    }
