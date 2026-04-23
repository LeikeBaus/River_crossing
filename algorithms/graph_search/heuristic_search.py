from __future__ import annotations

import heapq
import math
from collections.abc import Iterable
from typing import Any

from algorithms.graph_search.dijkstra import PlanResult
from core.cost.cost_function import CostFunction
from core.environment.actions import Action
from core.environment.environment import RiverEnvironment
from core.environment.grid import State


def astar(
    env: RiverEnvironment,
    cost_fn: CostFunction,
    heuristic: str = "euclidean",
) -> PlanResult:
    """Run A* search with admissible heuristic weighting (w = 1).

    Parameters
    ----------
    env:
        River-crossing environment.
    cost_fn:
        Transition-cost function.
    heuristic:
        Name of heuristic function. Currently supports ``"euclidean"``.
    """
    return weighted_astar(env, cost_fn, weight=1.0, heuristic=heuristic)


def weighted_astar(
    env: RiverEnvironment,
    cost_fn: CostFunction,
    weight: float = 1.5,
    heuristic: str = "euclidean",
) -> PlanResult:
    """Run Weighted A* search with ``f(s) = g(s) + w * h(s)``.

    For ``w=1`` this reduces to standard A*.

    Parameters
    ----------
    env:
        River-crossing environment.
    cost_fn:
        Transition-cost function.
    weight:
        Heuristic inflation factor (must be >= 1).
    heuristic:
        Name of heuristic function. Currently supports ``"euclidean"``.
    """
    if weight < 1.0:
        raise ValueError(f"weight must be >= 1, got {weight}.")

    inertia: int = getattr(cost_fn, "inertia", 0)
    start = env.start
    goal = env.goal

    start_hist: tuple = ()
    start_node = (start, start_hist)

    g_score: dict[tuple, float] = {start_node: 0.0}
    prev: dict[tuple, tuple] = {}

    # For admissibility with mixed costs, scale euclidean distance by alpha.
    heuristic_scale = max(0.0, getattr(cost_fn, "alpha", 1.0))

    counter = 0
    start_h = _heuristic(start, goal, heuristic, heuristic_scale)
    # Heap entry: (f, g, counter, state, history) – counter ensures State is never compared.
    heap: list[tuple] = [(start_h, 0.0, counter, start, start_hist)]

    nodes_expanded = 0
    exploration_path: list[list[State]] = []

    while heap:
        f_curr, g_curr, _, state, history = heapq.heappop(heap)
        node = (state, history)

        # Skip stale queue entries.
        if g_curr > g_score.get(node, float("inf")):
            continue

        nodes_expanded += 1

        # Record partial path snapshot from start to this expansion frontier.
        snapshot: list[State] = []
        cur_node = node
        while cur_node in prev:
            snapshot.append(cur_node[0])
            cur_node, _ = prev[cur_node]
        snapshot.append(start)
        snapshot.reverse()
        exploration_path.append(snapshot)

        if state == goal:
            return _reconstruct_result_hist(start, node, prev, g_score[node], nodes_expanded, exploration_path)

        for action in env.valid_actions(state):
            next_state = env.transition(state, action)

            if next_state == goal and not env.is_goal(next_state, action):
                continue

            tentative_g = g_curr + cost_fn.with_history(state, action, history)
            new_history: tuple = (history + (action,))[-inertia:] if inertia > 0 else ()
            next_node = (next_state, new_history)

            if tentative_g < g_score.get(next_node, float("inf")):
                g_score[next_node] = tentative_g
                prev[next_node] = (node, action)

                h = _heuristic(next_state, goal, heuristic, heuristic_scale)
                counter += 1
                heapq.heappush(
                    heap,
                    (tentative_g + weight * h, tentative_g, counter, next_state, new_history),
                )

    return PlanResult(nodes_expanded=nodes_expanded, found=False, exploration_path=exploration_path)


def astar_from_config(
    env: RiverEnvironment,
    cost_fn: CostFunction,
    a_star_config: dict[str, Any],
) -> PlanResult:
    """Run A* using config section ``a_star`` from algorithm config."""
    heuristic = str(a_star_config.get("heuristic", "euclidean"))
    return astar(env, cost_fn, heuristic=heuristic)


def weighted_astar_from_config(
    env: RiverEnvironment,
    cost_fn: CostFunction,
    weighted_cfg: dict[str, Any],
) -> PlanResult:
    """Run Weighted A* using config section ``weighted_a_star``."""
    heuristic = str(weighted_cfg.get("heuristic", "euclidean"))
    weight = float(weighted_cfg.get("weight", 1.5))
    return weighted_astar(env, cost_fn, weight=weight, heuristic=heuristic)


def run_weighted_astar_sweep(
    env: RiverEnvironment,
    cost_fn: CostFunction,
    weights: Iterable[float],
    heuristic: str = "euclidean",
) -> dict[float, PlanResult]:
    """Run Weighted A* for multiple ``w`` values.

    This utility prepares parameter studies over the speed/optimality trade-off.
    """
    results: dict[float, PlanResult] = {}
    for w in weights:
        results[float(w)] = weighted_astar(env, cost_fn, weight=float(w), heuristic=heuristic)
    return results


def _heuristic(
    state: State,
    goal: State,
    heuristic: str,
    scale: float,
) -> float:
    if heuristic == "euclidean":
        return scale * math.hypot(goal.i - state.i, goal.j - state.j)
    raise ValueError(f"Unknown heuristic: {heuristic!r}")


def _reconstruct_result_hist(
    start: State,
    goal_node: tuple,
    prev: dict[tuple, tuple],
    total_cost: float,
    nodes_expanded: int,
    exploration_path: list[list[State]] | None = None,
) -> PlanResult:
    """Reconstruct a PlanResult from history-augmented predecessor map."""
    path: list[State] = []
    actions: list[Action] = []
    cur_node = goal_node

    while cur_node in prev:
        pre_node, act = prev[cur_node]
        path.append(cur_node[0])
        actions.append(act)
        cur_node = pre_node

    path.append(start)
    path.reverse()
    actions.reverse()

    return PlanResult(
        path=path,
        actions=actions,
        total_cost=total_cost,
        nodes_expanded=nodes_expanded,
        found=True,
        exploration_path=exploration_path or [],
    )


def _reconstruct_result(
    start: State,
    goal: State,
    prev: dict[State, tuple[State, Action]],
    total_cost: float,
    nodes_expanded: int,
    exploration_path: list[list[State]] | None = None,
) -> PlanResult:
    path: list[State] = []
    actions: list[Action] = []
    current = goal

    while current in prev:
        pre, act = prev[current]
        path.append(current)
        actions.append(act)
        current = pre

    path.append(start)
    path.reverse()
    actions.reverse()

    return PlanResult(
        path=path,
        actions=actions,
        total_cost=total_cost,
        nodes_expanded=nodes_expanded,
        found=True,
        exploration_path=exploration_path or [],
    )
