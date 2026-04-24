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

from core.config_loader import load_algorithm_config, load_env_config
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
    """Return the total number of sweep points without running anything."""
    return sum(1 for _ in sweep_points(sweep_cfg))


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------


@dataclass
class SweepResult:
    """Result record for one completed or cached sweep point."""

    point: SweepPoint
    exp_id: str
    raw_path: Path
    ev_path: Path
    cached: bool
    records: list[dict[str, Any]]


def run_sweep(
    sweep_cfg: dict[str, Any],
    base_algo_cfg: dict[str, Any],
    config_dir: Path,
    analysis_dir: Path,
) -> Generator[dict[str, Any], None, SweepResult | None]:
    """Execute the full parameter sweep.

    This is a generator that yields progress dicts for each point:

        {"done": int, "total": int, "cached": bool, "exp_id": str,
         "raw_path": str, "point": SweepPoint}

    The generator's return value (accessible via ``StopIteration.value``) is the
    :class:`SweepResult` for the *last* computed point, or ``None`` if the sweep
    was empty.

    Usage::

        gen = run_sweep(...)
        last_result = None
        for progress in gen:
            ...  # update UI
        try:
            last_result = gen.send(None)  # should not happen after StopIteration
        except StopIteration as exc:
            last_result = exc.value
    """
    points = list(sweep_points(sweep_cfg))
    total = len(points)
    last_result: SweepResult | None = None

    for idx, point in enumerate(points):
        effective_algo = point.build_algo_override(base_algo_cfg)
        exp_id = build_experiment_id(
            flow_cfg=point.flow_cfg,
            algo_cfg=effective_algo,
            seeds=point.seeds,
            grid_nx=point.grid_nx,
            grid_ny=point.grid_ny,
            ql_episodes=point.ql_episodes,
        )
        raw_path, ev_path = results_paths(exp_id, analysis_dir)

        if raw_path.exists():
            records = load_experiment_results(raw_path)
            if not ev_path.exists():
                evaluate_results(records, output_path=ev_path)
            cached = True
        else:
            # Build a minimal env config for this sweep point
            base_env_cfg = load_env_config(config_dir / "env.yaml")
            env_cfg = copy.deepcopy(base_env_cfg)
            env_cfg["flow"] = dict(point.flow_cfg)
            env_cfg["grid"] = {"nx": point.grid_nx, "ny": point.grid_ny}

            # Temporarily swap experiment.yaml algorithms+seeds by using
            # run_all_experiments with overrides
            result_records = run_all_experiments(
                config_dir=config_dir,
                output_path=raw_path,
                flow_config_override=dict(point.flow_cfg),
                impact_override=point.beta,
                inertia_override=point.inertia,
                rl_episodes=point.ql_episodes,
            )
            evaluate_results(result_records, output_path=ev_path)
            records = [r.to_dict() for r in result_records]
            cached = False

        last_result = SweepResult(
            point=point,
            exp_id=exp_id,
            raw_path=raw_path,
            ev_path=ev_path,
            cached=cached,
            records=records,
        )
        yield {
            "done": idx + 1,
            "total": total,
            "cached": cached,
            "exp_id": exp_id,
            "raw_path": str(raw_path),
            "point": point,
        }

    return last_result
