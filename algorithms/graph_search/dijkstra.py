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
    exploration_path:
        For algorithms that support it: one partial-path snapshot per
        expansion step, ordered by expansion time.  Each snapshot is the
        chain from ``start`` to the currently-expanded node reconstructed
        through the predecessor map.  Empty for algorithms that do not
        populate it.
    """

    path: list[State] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    total_cost: float = float("inf")
    nodes_expanded: int = 0
    found: bool = False
    exploration_path: list[list[State]] = field(default_factory=list)


def dijkstra(
    env: RiverEnvironment,
    cost_fn: CostFunction,
    max_snapshots: int = 500,
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
    max_snapshots:
        Maximum number of exploration-path snapshots to store.  Snapshots are
        sampled evenly so memory and reconstruction time stay bounded even with
        large inertia state spaces.

    Returns
    -------
    PlanResult
        Optimal path, actions, total cost and search statistics.
        ``found=False`` when no valid path exists.
    """
    # Priority queue entries: (accumulated_cost, tie_breaker, state, history_tuple)
    # history_tuple = last `inertia` actions taken; () when inertia == 0.
    inertia: int = getattr(cost_fn, "inertia", 0)
    start = env.start
    start_hist: tuple = ()
    start_node = (start, start_hist)

    # Distance map keyed by (state, history_tuple).
    dist: dict[tuple, float] = {start_node: 0.0}
    # Predecessor map: node → (parent_node, action_taken)
    prev: dict[tuple, tuple] = {}

    counter = 0  # tie-breaker for heap stability
    # Heap entry: (cost, counter, state, history) – counter ensures State is never compared.
    heap: list[tuple] = [(0.0, counter, start, start_hist)]

    nodes_expanded = 0
    exploration_path: list[list[State]] = []
    # Sample snapshots at most every `_snap_stride` expansions so the total
    # stored count stays at most max_snapshots regardless of state-space size.
    _snap_stride = max(1, inertia ** 2) if inertia > 1 else 1

    goal = env.goal

    while heap:
        g, _, state, history = heapq.heappop(heap)
        node = (state, history)

        # Skip stale entries
        if g > dist.get(node, float("inf")):
            continue

        nodes_expanded += 1

        # Reconstruct partial path and store as a snapshot, sampled so the
        # total count stays bounded at max_snapshots.
        if nodes_expanded % _snap_stride == 0 and len(exploration_path) < max_snapshots:
            snapshot: list[State] = []
            cur_node = node
            while cur_node in prev:
                snapshot.append(cur_node[0])
                cur_node, _ = prev[cur_node]
            snapshot.append(start)
            snapshot.reverse()
            exploration_path.append(snapshot)

        for action in env.valid_actions(state):
            next_state = env.transition(state, action)

            # The goal cell is only reachable via a valid approach angle.
            if next_state == goal and not env.is_goal(next_state, action):
                continue

            edge_cost = cost_fn.with_history(state, action, history)
            new_g = g + edge_cost

            new_history: tuple = (history + (action,))[-inertia:] if inertia > 0 else ()
            next_node = (next_state, new_history)

            if new_g < dist.get(next_node, float("inf")):
                dist[next_node] = new_g
                prev[next_node] = (node, action)
                counter += 1
                heapq.heappush(heap, (new_g, counter, next_state, new_history))

    # Find the cheapest arrival tuple at goal (across all possible histories).
    goal_arrivals = [(d, n) for n, d in dist.items() if n[0] == goal]
    if not goal_arrivals:
        return PlanResult(
            nodes_expanded=nodes_expanded,
            found=False,
            exploration_path=exploration_path,
        )

    _, best_goal_node = min(goal_arrivals)

    # Reconstruct path
    path: list[State] = []
    actions: list[Action] = []
    cur_node = best_goal_node
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
        total_cost=dist[best_goal_node],
        nodes_expanded=nodes_expanded,
        found=True,
        exploration_path=exploration_path,
    )
