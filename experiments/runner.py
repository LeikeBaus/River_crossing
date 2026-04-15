from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from algorithms.graph_search import (
    apf_from_config,
    astar_from_config,
    dijkstra,
    extract_path,
    value_iteration,
    weighted_astar_from_config,
)
from algorithms.rl import q_learning_from_config, rollout_policy
from core.config_loader import load_algorithm_config, load_env_config, load_experiment_config
from core.cost.cost_function import CostFunction
from core.environment.environment import RiverEnvironment


@dataclass
class ExperimentRunRecord:
    """Normalized, JSON-serializable result for one experiment run."""

    environment_name: str
    algorithm: str
    seed: int
    found: bool
    total_cost: float | None
    path_length: int
    steps: int
    angle_valid: bool
    plan_time: float
    training_time: float
    inference_time: float
    total_time: float
    nodes_expanded: int
    path: list[list[int]] = field(default_factory=list)
    actions: list[list[int]] = field(default_factory=list)
    best_path: list[list[int]] = field(default_factory=list)
    best_actions: list[list[int]] = field(default_factory=list)
    run_trace: list[dict[str, Any]] = field(default_factory=list)
    reward_history: list[float] = field(default_factory=list)
    success_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_all_experiments(
    config_dir: str | Path = "configs",
    output_path: str | Path | None = None,
    rl_episodes: int | None = None,
) -> list[ExperimentRunRecord]:
    """Run the configured batch of experiments and optionally save raw results.

    Results are stored as a flat list of per-run records to keep downstream
    evaluation and serialization straightforward.
    """
    config_dir = Path(config_dir)
    algo_cfg = load_algorithm_config(config_dir / "algorithm.yaml")
    exp_cfg = load_experiment_config(config_dir / "experiment.yaml")
    default_env_cfg = load_env_config(config_dir / "env.yaml")

    if rl_episodes is None:
        rl_episodes = int(exp_cfg.get("q_learning_episodes", exp_cfg.get("episodes", 200)))

    results: list[ExperimentRunRecord] = []

    for env_entry in exp_cfg["environments"]:
        env_name, env_cfg = _resolve_environment_config(config_dir, env_entry, default_env_cfg)

        for algorithm in exp_cfg["algorithms"]:
            for seed in exp_cfg["seeds"]:
                env = RiverEnvironment.from_config(env_cfg)
                cost_fn = CostFunction.from_config(algo_cfg, env.flow)
                record = _run_single_experiment(
                    env=env,
                    env_name=env_name,
                    algorithm=algorithm,
                    seed=int(seed),
                    cost_fn=cost_fn,
                    algo_cfg=algo_cfg,
                    rl_episodes=rl_episodes,
                )
                results.append(record)

    if output_path is not None:
        save_experiment_results(output_path, results)

    return results


def save_experiment_results(
    output_path: str | Path,
    results: list[ExperimentRunRecord],
) -> None:
    """Save raw experiment records as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [record.to_dict() for record in results]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_experiment_results(output_path: str | Path) -> list[dict[str, Any]]:
    """Load raw experiment results from JSON."""
    path = Path(output_path)
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_environment_config(
    config_dir: Path,
    env_entry: dict[str, Any],
    default_env_cfg: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    env_name = str(env_entry.get("name", "environment"))
    env_path = env_entry.get("env_config")
    if env_path is None:
        return env_name, default_env_cfg
    return env_name, load_env_config(config_dir.parent / str(env_path))


def _run_single_experiment(
    env: RiverEnvironment,
    env_name: str,
    algorithm: str,
    seed: int,
    cost_fn: CostFunction,
    algo_cfg: dict[str, Any],
    rl_episodes: int,
) -> ExperimentRunRecord:
    if algorithm == "dijkstra":
        t0 = time.perf_counter()
        plan = dijkstra(env, cost_fn)
        plan_time = time.perf_counter() - t0
        return _record_from_plan(env, env_name, algorithm, seed, plan, plan_time=plan_time)

    if algorithm == "a_star":
        t0 = time.perf_counter()
        plan = astar_from_config(env, cost_fn, algo_cfg["a_star"])
        plan_time = time.perf_counter() - t0
        return _record_from_plan(env, env_name, algorithm, seed, plan, plan_time=plan_time)

    if algorithm == "weighted_a_star":
        t0 = time.perf_counter()
        plan = weighted_astar_from_config(env, cost_fn, algo_cfg["weighted_a_star"])
        plan_time = time.perf_counter() - t0
        return _record_from_plan(env, env_name, algorithm, seed, plan, plan_time=plan_time)

    if algorithm == "dynamic_programming":
        dp_cfg = algo_cfg["dynamic_programming"]
        t0 = time.perf_counter()
        vi = value_iteration(
            env,
            cost_fn,
            gamma=float(dp_cfg.get("gamma", 1.0)),
            tolerance=float(dp_cfg.get("tolerance", 1e-6)),
            max_iterations=int(dp_cfg.get("max_iterations", 10_000)),
        )
        path, actions = extract_path(env, vi.policy)
        plan_time = time.perf_counter() - t0
        plan = _plan_result_from_path(env, cost_fn, path, actions, nodes_expanded=vi.iterations)
        return _record_from_plan(env, env_name, algorithm, seed, plan, plan_time=plan_time)

    if algorithm == "apf":
        t0 = time.perf_counter()
        plan = apf_from_config(env, cost_fn, algo_cfg["apf"])
        plan_time = time.perf_counter() - t0
        return _record_from_plan(env, env_name, algorithm, seed, plan, plan_time=plan_time)

    if algorithm == "q_learning":
        t0 = time.perf_counter()
        train_result = q_learning_from_config(
            env,
            cost_fn,
            algo_cfg["q_learning"],
            episodes=rl_episodes,
            seed=seed,
        )
        training_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        plan = rollout_policy(env, cost_fn, train_result.policy)
        inference_time = time.perf_counter() - t1

        return _record_from_plan(
            env,
            env_name,
            algorithm,
            seed,
            plan,
            training_time=training_time,
            inference_time=inference_time,
            reward_history=train_result.reward_history,
            success_rate=train_result.success_rate,
        )

    raise ValueError(f"Unsupported algorithm: {algorithm!r}")


def _plan_result_from_path(
    env: RiverEnvironment,
    cost_fn: CostFunction,
    path: list[Any],
    actions: list[Any],
    nodes_expanded: int,
) -> Any:
    from algorithms.graph_search.dijkstra import PlanResult

    total_cost = 0.0
    for state, action in zip(path, actions):
        total_cost += cost_fn(state, action)

    found = bool(path) and path[-1] == env.goal and bool(actions) and env.is_goal(path[-1], actions[-1])
    return PlanResult(
        path=path,
        actions=actions,
        total_cost=total_cost if found else float("inf"),
        nodes_expanded=nodes_expanded,
        found=found,
    )


def _record_from_plan(
    env: RiverEnvironment,
    environment_name: str,
    algorithm: str,
    seed: int,
    plan: Any,
    plan_time: float = 0.0,
    training_time: float = 0.0,
    inference_time: float = 0.0,
    reward_history: list[float] | None = None,
    success_rate: float | None = None,
) -> ExperimentRunRecord:
    path = [[state.i, state.j] for state in plan.path]
    actions = [[action[0], action[1]] for action in plan.actions]
    run_trace = _build_run_trace(env, plan.path, plan.actions)
    angle_valid = bool(plan.found and plan.path and plan.actions)
    total_time = plan_time + training_time + inference_time
    total_cost = None if not plan.found else float(plan.total_cost)

    return ExperimentRunRecord(
        environment_name=environment_name,
        algorithm=algorithm,
        seed=seed,
        found=bool(plan.found),
        total_cost=total_cost,
        path_length=len(plan.path),
        steps=len(plan.actions),
        angle_valid=angle_valid,
        plan_time=plan_time,
        training_time=training_time,
        inference_time=inference_time,
        total_time=total_time,
        nodes_expanded=int(plan.nodes_expanded),
        path=path,
        actions=actions,
        best_path=path,
        best_actions=actions,
        run_trace=run_trace,
        reward_history=reward_history or [],
        success_rate=success_rate,
    )


def _build_run_trace(
    env: RiverEnvironment,
    path: list[Any],
    actions: list[Any],
) -> list[dict[str, Any]]:
    """Build a per-step decision trace for the UI Run view."""
    trace: list[dict[str, Any]] = []

    for index, state in enumerate(path):
        best_action = actions[index] if index < len(actions) else None
        step: dict[str, Any] = {
            "step": index,
            "state": [state.i, state.j],
            "best_action": None if best_action is None else [best_action[0], best_action[1]],
            "valid_actions": [],
            "invalid_actions": [],
        }

        valid_actions = set(env.valid_actions(state))
        for action in env.actions:
            encoded = [int(action[0]), int(action[1])]
            if action not in valid_actions:
                step["invalid_actions"].append(encoded)
                continue

            next_state = env.transition(state, action)
            if next_state == env.goal and not env.is_goal(next_state, action):
                step["invalid_actions"].append(encoded)
            elif action != best_action:
                step["valid_actions"].append(encoded)

        trace.append(step)

    return trace
