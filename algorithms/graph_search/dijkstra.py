from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any

from core.cost.cost_function import CostFunction
from core.environment.actions import Action
from core.environment.environment import RiverEnvironment
from core.environment.grid import State


@dataclass
class PlanResult:
    """Result returned by any graph-search or planning algorithm.

    Attributes
    ----------
    path:
        Sequence of states from start (inclusive) to goal (inclusive).
        Empty if no solution was found.
    actions:
        Sequence of actions taken – ``len(actions) == len(path) - 1``.
    total_cost:
        Accumulated cost ``J(π) = Σ c(s_t, a_t)``.
    nodes_expanded:
        Number of states popped from the priority queue (search effort).
    found:
        ``True`` when a valid solution was found.
    """

    path: list[State] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    total_cost: float = float("inf")
    nodes_expanded: int = 0
    found: bool = False


def dijkstra(
    env: RiverEnvironment,
    cost_fn: CostFunction,
) -> PlanResult:
    """Find the cost-optimal path using Dijkstra's algorithm.

    The algorithm performs a shortest-path search over the discrete grid while
    respecting the docking approach-angle constraint: only a state transition
    that arrives at ``env.goal`` via an action satisfying the angle bounds is
    accepted as a terminal state.

    The search uses a *state-action* goal check because the validity of reaching
    the goal depends on the **last action**, not the state alone.  To correctly
    handle this, goal states are only recorded when ``env.is_goal(next_state,
    action)`` is ``True``.

    Parameters
    ----------
    env:
        The fully-specified river-crossing environment.
    cost_fn:
        Transition-cost function ``c(s, a)``.

    Returns
    -------
    PlanResult
        Optimal path, actions, total cost and search statistics.
        ``found=False`` when no valid path exists.
    """
    # Priority queue entries: (accumulated_cost, tie_breaker, state)
    start = env.start
    dist: dict[State, float] = {start: 0.0}
    # predecessor map: state → (previous_state, action_taken)
    prev: dict[State, tuple[State, Action]] = {}

    counter = 0  # tie-breaker for heap stability
    heap: list[tuple[float, int, State]] = [(0.0, counter, start)]

    nodes_expanded = 0

    goal = env.goal

    while heap:
        g, _, state = heapq.heappop(heap)

        # Skip stale entries
        if g > dist.get(state, float("inf")):
            continue

        nodes_expanded += 1

        for action in env.valid_actions(state):
            next_state = env.transition(state, action)

            # The goal cell is only reachable via a valid approach angle.
            # Transitions that arrive at the goal with an invalid angle are
            # discarded so that dist[goal] always reflects a valid docking.
            if next_state == goal and not env.is_goal(next_state, action):
                continue

            edge_cost = cost_fn(state, action)
            new_g = g + edge_cost

            if new_g < dist.get(next_state, float("inf")):
                dist[next_state] = new_g
                prev[next_state] = (state, action)
                counter += 1
                heapq.heappush(heap, (new_g, counter, next_state))

    if goal not in dist:
        return PlanResult(nodes_expanded=nodes_expanded, found=False)

    # Reconstruct path
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
        total_cost=dist[goal],
        nodes_expanded=nodes_expanded,
        found=True,
    )
