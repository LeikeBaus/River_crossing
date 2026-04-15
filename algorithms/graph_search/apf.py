from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from algorithms.graph_search.dijkstra import PlanResult
from core.cost.cost_function import CostFunction
from core.environment.actions import Action
from core.environment.environment import RiverEnvironment
from core.environment.grid import State


@dataclass
class APFResult(PlanResult):
    """Result of the artificial potential field planner.

    Attributes
    ----------
    termination_reason:
        One of ``"goal"``, ``"local_minimum"`` or ``"max_steps"``.
    local_minimum_detected:
        ``True`` when the greedy descent gets stuck or starts cycling.
    """

    termination_reason: str = "local_minimum"
    local_minimum_detected: bool = False


def attractive_potential(
    state: State,
    goal: State,
    k_att: float,
    target: State | None = None,
) -> float:
    """Return the attractive potential ``0.5 * k_att * ||s - s_target||^2``."""
    target_state = goal if target is None else target
    di = state.i - target_state.i
    dj = state.j - target_state.j
    return 0.5 * k_att * (di * di + dj * dj)


def flow_potential(env: RiverEnvironment, state: State, lambda_flow: float) -> float:
    """Return the flow contribution ``-lambda * <s, u_flow(s)>``."""
    vi, vj = env.flow_at(state)
    return -lambda_flow * (state.i * vi + state.j * vj)


def _valid_goal_predecessors(env: RiverEnvironment) -> list[State]:
    """Return in-grid states from which the goal can be reached validly in one step."""
    predecessors: list[State] = []
    for action in env.actions:
        candidate = State(env.goal.i - action[0], env.goal.j - action[1])
        if not env.grid.contains(candidate):
            continue
        if env.transition(candidate, action) == env.goal and env.is_goal(env.goal, action):
            predecessors.append(candidate)
    return list(dict.fromkeys(predecessors))


def _guidance_target(env: RiverEnvironment, state: State) -> State:
    """Choose a docking-aware subgoal that guides APF into a valid approach corridor."""
    predecessors = _valid_goal_predecessors(env)
    if not predecessors or state in predecessors:
        return env.goal

    return min(
        predecessors,
        key=lambda item: (
            (item.i - state.i) ** 2 + (item.j - state.j) ** 2,
            abs(item.j - env.goal.j),
            item.i,
            item.j,
        ),
    )


def _docking_distance_penalty(env: RiverEnvironment, state: State) -> float:
    """Penalize states that are still far away from a valid final docking predecessor."""
    if state == env.goal:
        return 0.0

    predecessors = _valid_goal_predecessors(env)
    if not predecessors:
        return 0.0

    best_sq_distance = min((state.i - item.i) ** 2 + (state.j - item.j) ** 2 for item in predecessors)
    return 0.05 * float(best_sq_distance)


def total_potential(
    env: RiverEnvironment,
    state: State,
    k_att: float,
    lambda_flow: float,
    target: State | None = None,
) -> float:
    """Return the total potential ``Phi_att + Phi_flow`` at *state*."""
    return attractive_potential(state, env.goal, k_att, target=target) + flow_potential(env, state, lambda_flow)


def potential_gradient(
    env: RiverEnvironment,
    state: State,
    k_att: float,
    lambda_flow: float,
    target: State | None = None,
) -> tuple[float, float]:
    """Return the discrete APF gradient under the current constant-flow model.

    With the current flow implementation the gradient is approximated as

        grad Phi(s) = k_att * (s - s_goal) - lambda_flow * u_flow(s)

    which is exact for spatially constant flow fields.
    """
    target_state = env.goal if target is None else target
    vi, vj = env.flow_at(state)
    grad_i = k_att * (state.i - target_state.i) - lambda_flow * vi
    grad_j = k_att * (state.j - target_state.j) - lambda_flow * vj
    return (grad_i, grad_j)


def select_apf_action(
    env: RiverEnvironment,
    state: State,
    k_att: float,
    lambda_flow: float,
) -> Action | None:
    """Select the best discrete action by descending along the potential field.

    The chosen action minimizes the directional derivative ``<a, grad Phi>``.
    Ties are broken by lower next-state potential and then lexicographically.
    Invalid arrivals at the goal are excluded.
    """
    target = _guidance_target(env, state)
    gradient = potential_gradient(env, state, k_att, lambda_flow, target=target)
    candidates: list[tuple[float, float, Action]] = []

    for action in env.valid_actions(state):
        next_state = env.transition(state, action)
        if next_state == env.goal:
            if env.is_goal(next_state, action):
                return action
            continue

        directional_derivative = action[0] * gradient[0] + action[1] * gradient[1]
        next_potential = total_potential(env, next_state, k_att, lambda_flow, target=target)
        docking_penalty = _docking_distance_penalty(env, next_state)
        candidates.append(
            (directional_derivative + docking_penalty, next_potential + docking_penalty, action)
        )

    if not candidates:
        return None

    _, _, best_action = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    return best_action


def apf_plan(
    env: RiverEnvironment,
    cost_fn: CostFunction,
    k_att: float = 1.0,
    lambda_flow: float = 0.2,
    max_steps: int | None = None,
) -> APFResult:
    """Plan greedily with an artificial potential field.

    The planner repeatedly selects the valid action that best aligns with the
    negative potential gradient. Because greedy APF can get trapped in local
    minima or cycles, the implementation explicitly detects these cases and
    returns ``found=False`` with ``local_minimum_detected=True``.
    """
    if k_att <= 0:
        raise ValueError(f"k_att must be > 0, got {k_att}.")
    if lambda_flow < 0:
        raise ValueError(f"lambda_flow must be >= 0, got {lambda_flow}.")

    if max_steps is None:
        max_steps = 4 * env.grid.nx * env.grid.ny
    if max_steps <= 0:
        raise ValueError(f"max_steps must be > 0, got {max_steps}.")

    state = env.start
    path = [state]
    actions: list[Action] = []
    total_cost = 0.0
    visited: set[State] = {state}

    for _ in range(max_steps):
        if state == env.goal:
            return APFResult(
                path=path,
                actions=actions,
                total_cost=total_cost,
                nodes_expanded=len(actions),
                found=True,
                termination_reason="goal",
                local_minimum_detected=False,
            )

        current_potential = total_potential(env, state, k_att, lambda_flow)
        action = select_apf_action(env, state, k_att, lambda_flow)
        if action is None:
            return APFResult(
                path=path,
                actions=actions,
                total_cost=total_cost,
                nodes_expanded=len(actions),
                found=False,
                termination_reason="local_minimum",
                local_minimum_detected=True,
            )

        next_state = env.transition(state, action)
        next_potential = total_potential(env, next_state, k_att, lambda_flow)

        # APF can oscillate on plateaus or cycles. Treat revisits or non-
        # descending moves as a detected local minimum unless the goal is hit.
        if next_state != env.goal and (next_state in visited or next_potential >= current_potential):
            return APFResult(
                path=path,
                actions=actions,
                total_cost=total_cost,
                nodes_expanded=len(actions),
                found=False,
                termination_reason="local_minimum",
                local_minimum_detected=True,
            )

        total_cost += cost_fn(state, action)
        actions.append(action)
        path.append(next_state)
        visited.add(next_state)
        state = next_state

    return APFResult(
        path=path,
        actions=actions,
        total_cost=total_cost,
        nodes_expanded=len(actions),
        found=(state == env.goal),
        termination_reason="goal" if state == env.goal else "max_steps",
        local_minimum_detected=False,
    )


def apf_from_config(
    env: RiverEnvironment,
    cost_fn: CostFunction,
    apf_config: dict[str, Any],
) -> APFResult:
    """Run the APF planner from the ``apf`` section of the algorithm config."""
    return apf_plan(
        env,
        cost_fn,
        k_att=float(apf_config.get("k_att", 1.0)),
        lambda_flow=float(apf_config.get("lambda_flow", 0.2)),
    )