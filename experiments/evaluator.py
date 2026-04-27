from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from experiments.runner import ExperimentRunRecord, load_experiment_results


def evaluate_results(
    results: list[ExperimentRunRecord] | list[dict[str, Any]] | str | Path,
    output_path: str | Path | None = None,
    reference_algorithm: str = "dijkstra",
) -> dict[str, Any]:
    """Evaluate experiment runs and compute summary metrics."""
    records = _normalize_results_input(results)
    annotated = _annotate_with_delta_j(records, reference_algorithm)
    summary = {
        "reference_algorithm": reference_algorithm,
        "runs": annotated,
        "by_algorithm": _aggregate_by_algorithm(annotated),
    }

    if output_path is not None:
        save_evaluation_summary(output_path, summary)

    return summary


def save_evaluation_summary(output_path: str | Path, summary: dict[str, Any]) -> None:
    """Save evaluation summary as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _normalize_results_input(
    results: list[ExperimentRunRecord] | list[dict[str, Any]] | str | Path,
) -> list[dict[str, Any]]:
    if isinstance(results, (str, Path)):
        loaded = load_experiment_results(results)
        return [dict(item) for item in loaded]

    normalized: list[dict[str, Any]] = []
    for item in results:
        if isinstance(item, ExperimentRunRecord):
            normalized.append(item.to_dict())
        else:
            normalized.append(dict(item))
    return normalized


def _annotate_with_delta_j(
    records: list[dict[str, Any]],
    reference_algorithm: str,
) -> list[dict[str, Any]]:
    reference_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        if record["algorithm"] == reference_algorithm:
            key = (str(record["environment_name"]), int(record["seed"]))
            reference_by_key[key] = record

    annotated: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        key = (str(item["environment_name"]), int(item["seed"]))
        reference = reference_by_key.get(key)
        if reference is None or item.get("total_cost") is None or reference.get("total_cost") is None:
            item["delta_j"] = None
        else:
            item["delta_j"] = float(item["total_cost"]) - float(reference["total_cost"])
        annotated.append(item)
    return annotated


def _aggregate_by_algorithm(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["algorithm"]), []).append(record)

    summary: dict[str, dict[str, Any]] = {}
    for algorithm, items in grouped.items():
        solved = [item for item in items if item["found"]]
        delta_values = [float(item["delta_j"]) for item in items if item["delta_j"] is not None]
        total_costs = [float(item["total_cost"]) for item in solved if item["total_cost"] is not None]

        summary[algorithm] = {
            "runs": len(items),
            "solved_runs": len(solved),
            "success_rate": len(solved) / len(items) if items else 0.0,
            "mean_total_cost": _safe_mean(total_costs),
            "mean_delta_j": _safe_mean(delta_values),
            "mean_steps": _safe_mean([float(item["steps"]) for item in items]),
            "mean_plan_time": _safe_mean([float(item["plan_time"]) for item in items]),
            "mean_training_time": _safe_mean([float(item["training_time"]) for item in items]),
            "mean_inference_time": _safe_mean([float(item["inference_time"]) for item in items]),
            "mean_total_time": _safe_mean([float(item["total_time"]) for item in items]),
            "angle_valid_rate": _safe_mean([
                1.0 if item["angle_valid"] else 0.0 for item in items
            ]),
        }
    return summary


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(mean(values))
