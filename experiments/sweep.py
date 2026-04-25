"""Sweep runner – iterates over a parameter grid defined in ``configs/sweep.yaml``.

Each point in the grid corresponds to one experiment bundle (all algorithms ×
all seeds) and is stored as its own ``R-{id}.json`` / ``EV-{id}.json`` pair.
Already-computed results are skipped (cache-hit), so the sweep is resumable.

Public API
----------
``load_sweep_config(path)``     – load and validate the YAML sweep config.
``sweep_points(sweep_cfg)``     – generator of :class:`SweepPoint` instances.
``run_sweep(...)``              – execute the full sweep, yielding progress dicts.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.config_loader import load_algorithm_config
from experiments.evaluator import evaluate_results
from experiments.experiment_id import build_experiment_id, results_paths
from experiments.runner import (
    load_experiment_results,
    run_all_experiments,
    save_experiment_results,
)


# ---------------------------------------------------------------------------
# Data class describing one sweep point
# ---------------------------------------------------------------------------


@dataclass
class SweepPoint:
    """One fully-specified parameter combination in the sweep."""

    flow_type: str          # "gaussian" or "constant"
    flow_dir: str           # "up" or "down"
    flow_strength: float    # vector magnitude 0.0 – 1.0
    sigma: float            # Gaussian sigma (only meaningful if flow_type == "gaussian")
    floor: float
    alpha: float
    beta: float
    inertia: int
    turn_penalty: float
    ql_episodes: int
    grid_nx: int
    grid_ny: int
    seeds: list[int]
    algorithms: list[str]
    # derived
    flow_cfg: dict
    algo_cfg_common: dict   # the "common" section with overrides applied

    def build_algo_override(self, base_algo_cfg: dict) -> dict:
        """Return a copy of *base_algo_cfg* with this point's common-section values."""
        effective = copy.deepcopy(base_algo_cfg)
        effective["common"] = dict(effective.get("common", {}))
        effective["common"]["alpha"] = self.alpha
        effective["common"]["beta"] = self.beta
        effective["common"]["inertia"] = self.inertia
        effective["common"]["turn_penalty"] = self.turn_penalty
        # Propagate ql_episodes into the q_learning section
        effective["q_learning"] = dict(effective.get("q_learning", {}))
        effective["q_learning"]["episodes"] = self.ql_episodes
        return effective


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_sweep_config(path: str | Path) -> dict[str, Any]:
    """Load and lightly validate the sweep YAML configuration."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Sweep config not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Sweep config must be a mapping: {p}")
    return cfg


# ---------------------------------------------------------------------------
# Sweep-point generator
# ---------------------------------------------------------------------------


def sweep_points(sweep_cfg: dict[str, Any]) -> Iterator[SweepPoint]:
    """Yield one :class:`SweepPoint` per parameter combination."""
    seeds: list[int] = [int(s) for s in sweep_cfg.get("seeds", [1])]
    algorithms: list[str] = list(sweep_cfg.get("algorithms", ["dijkstra"]))
    flow_types: list[str] = list(sweep_cfg.get("flow_types", ["constant"]))
    flow_dirs: list[str] = list(sweep_cfg.get("flow_directions", ["up"]))
    strengths: list[float] = [float(v) for v in sweep_cfg.get("flow_strengths", [1.0])]
    sigmas: list[float] = [float(v) for v in sweep_cfg.get("sigmas", [5.0])]
    floor: float = float(sweep_cfg.get("floor", 0.1))
    alpha: float = float(sweep_cfg.get("alpha", 1.0))
    beta: float = float(sweep_cfg.get("beta", 1.0))
    inertias: list[int] = [int(v) for v in sweep_cfg.get("inertias", [0])]
    ql_episodes_list: list[int] = [int(v) for v in sweep_cfg.get("ql_episodes", [1200])]
    turn_penalty: float = float(sweep_cfg.get("turn_penalty", 1.0))
    grid_nx: int = int(sweep_cfg.get("grid_nx", 40))
    grid_ny: int = int(sweep_cfg.get("grid_ny", 20))

    for ft in flow_types:
        # sigma sweep only applies to gaussian; for constant use a single dummy value
        sigma_range = sigmas if ft == "gaussian" else [5.0]
        for fd in flow_dirs:
            vj_sign = 1.0 if fd == "up" else -1.0
            for strength in strengths:
                vj = vj_sign * strength
                flow_cfg = {
                    "type": ft,
                    "vector": [0.0, vj],
                }
                if ft == "gaussian":
                    for sigma in sigma_range:
                        flow_cfg_g = dict(flow_cfg)
                        flow_cfg_g["sigma"] = sigma
                        flow_cfg_g["floor"] = floor
                        for inertia in inertias:
                            for ql_ep in ql_episodes_list:
                                yield SweepPoint(
                                    flow_type=ft,
                                    flow_dir=fd,
                                    flow_strength=strength,
                                    sigma=sigma,
                                    floor=floor,
                                    alpha=alpha,
                                    beta=beta,
                                    inertia=inertia,
                                    turn_penalty=turn_penalty,
                                    ql_episodes=ql_ep,
                                    grid_nx=grid_nx,
                                    grid_ny=grid_ny,
                                    seeds=seeds,
                                    algorithms=algorithms,
                                    flow_cfg=dict(flow_cfg_g),
                                    algo_cfg_common={
                                        "alpha": alpha,
                                        "beta": beta,
                                        "inertia": inertia,
                                        "turn_penalty": turn_penalty,
                                    },
                                )
                else:
                    for inertia in inertias:
                        for ql_ep in ql_episodes_list:
                            yield SweepPoint(
                                flow_type=ft,
                                flow_dir=fd,
                                flow_strength=strength,
                                sigma=5.0,
                                floor=floor,
                                alpha=alpha,
                                beta=beta,
                                inertia=inertia,
                                turn_penalty=turn_penalty,
                                ql_episodes=ql_ep,
                                grid_nx=grid_nx,
                                grid_ny=grid_ny,
                                seeds=seeds,
                                algorithms=algorithms,
                                flow_cfg=dict(flow_cfg),
                                algo_cfg_common={
                                    "alpha": alpha,
                                    "beta": beta,
                                    "inertia": inertia,
                                    "turn_penalty": turn_penalty,
                                },
                            )


def count_sweep_points(sweep_cfg: dict[str, Any]) -> int:
    """Return the total number of (algorithm × parameter) steps in the sweep."""
    n_points = sum(1 for _ in sweep_points(sweep_cfg))
    n_algorithms = len(list(sweep_cfg.get("algorithms", ["dijkstra"])))
    return n_points * max(1, n_algorithms)


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------


@dataclass
class SweepResult:
    """Result record for one completed or cached (algorithm, sweep-point) step.

    Each step produces one result file per algorithm, named with the algorithm
    abbreviation as a prefix (e.g. ``R-dijk-V0100-...json``).
    ``raw_path`` / ``ev_path`` refer to the current algorithm's files.
    ``raw_paths`` / ``ev_paths`` accumulate all algorithms processed so far
    within the current sweep point.
    """

    point: SweepPoint
    exp_id: str          # base exp_id (no algo prefix) for display
    raw_path: Path       # last algorithm's raw-results file
    ev_path: Path        # last algorithm's evaluation file
    raw_paths: list[Path]  # all per-algorithm raw-results files
    ev_paths: list[Path]   # all per-algorithm evaluation files
    cached: bool         # True only when every algorithm was already cached
    records: list[dict[str, Any]]


def run_sweep(
    sweep_cfg: dict[str, Any],
    base_algo_cfg: dict[str, Any],
    config_dir: Path,
    analysis_dir: Path,
) -> Generator[dict[str, Any], None, SweepResult | None]:
    """Execute the full parameter sweep.

    Yields one progress dict **per algorithm per sweep point**::

        {"done": int, "total": int, "cached": bool,
         "exp_id": str,   # algorithm-specific ID, e.g. "dijk-V0100-..."
         "algorithm": str,
         "raw_path": str,
         "point": SweepPoint}

    The generator's return value (accessible via ``StopIteration.value``) is the
    :class:`SweepResult` for the *last* processed (algorithm, point) step, or
    ``None`` if the sweep was empty.
    """
    points = list(sweep_points(sweep_cfg))
    n_algorithms = len(points[0].algorithms) if points else 1
    total = len(points) * n_algorithms
    global_step = 0
    last_result: SweepResult | None = None

    for point in points:
        effective_algo = point.build_algo_override(base_algo_cfg)

        all_records: list[dict[str, Any]] = []
        all_raw_paths: list[Path] = []
        all_ev_paths: list[Path] = []

        for algorithm in point.algorithms:
            global_step += 1
            algo_exp_id = build_experiment_id(
                flow_cfg=point.flow_cfg,
                algo_cfg=effective_algo,
                seeds=point.seeds,
                algorithm=algorithm,
                grid_nx=point.grid_nx,
                grid_ny=point.grid_ny,
                ql_episodes=point.ql_episodes,
            )
            algo_raw_path, algo_ev_path = results_paths(algo_exp_id, analysis_dir)
            all_raw_paths.append(algo_raw_path)
            all_ev_paths.append(algo_ev_path)

            if algo_raw_path.exists():
                algo_cached = True
                algo_records = load_experiment_results(algo_raw_path)
                if not algo_ev_path.exists():
                    evaluate_results(algo_records, output_path=algo_ev_path)
            else:
                algo_cached = False
                result_records = run_all_experiments(
                    config_dir=config_dir,
                    output_path=algo_raw_path,
                    flow_config_override=dict(point.flow_cfg),
                    impact_override=point.beta,
                    inertia_override=point.inertia,
                    rl_episodes=point.ql_episodes,
                    algorithms_override=[algorithm],
                )
                evaluate_results(result_records, output_path=algo_ev_path)
                algo_records = [r.to_dict() for r in result_records]

            all_records.extend(algo_records)

            last_result = SweepResult(
                point=point,
                exp_id=algo_exp_id,
                raw_path=algo_raw_path,
                ev_path=algo_ev_path,
                raw_paths=list(all_raw_paths),
                ev_paths=list(all_ev_paths),
                cached=algo_cached,
                records=list(all_records),
            )
            yield {
                "done": global_step,
                "total": total,
                "cached": algo_cached,
                "exp_id": algo_exp_id,
                "algorithm": algorithm,
                "raw_path": str(algo_raw_path),
                "point": point,
            }

    return last_result
