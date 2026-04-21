from __future__ import annotations

from typing import Any

from core.cost.cost_function import CostFunction
from core.environment.actions import Action
from core.environment.environment import RiverEnvironment
from core.environment.grid import State


def value_iteration(
    env: RiverEnvironment,
    cost_fn: CostFunction,
    gamma: float = 1.0,
    tolerance: float = 1e-6,
    max_iterations: int = 10_000,
) -> ValueIterationResult:
    """Compute the optimal value function via Value Iteration.

    Solves the Bellman optimality equation for the deterministic MDP:

        V*(s) = min_{a ∈ A} [ c(s, a) + γ · V*(T(s, a)) ]

    with V*(goal) = 0 and the approach-angle constraint enforced by treating
    the goal cell as absorbing only when entered via a valid action.

    Parameters
    ----------
    env:
        The fully-specified river-crossing environment.
    cost_fn:
        Transition-cost function ``c(s, a)``.
    gamma:
        Discount factor in (0, 1].
    tolerance:
        Convergence threshold (max |V_{k+1} − V_k|).
    max_iterations:
        Safety cap on the number of sweeps.

    Returns
    -------
    ValueIterationResult
        Optimal value function, greedy policy, iteration count, and final
        Bellman residual.
    """
    if not (0 < gamma <= 1.0):
        raise ValueError(f"gamma must be in (0, 1], got {gamma}.")
    if tolerance <= 0:
        raise ValueError(f"tolerance must be > 0, got {tolerance}.")

    goal = env.goal

    # Enumerate all states
    states: list[State] = [
        State(i, j) for i in range(env.grid.nx) for j in range(env.grid.ny)
    ]

    # Initialise value function to 0.0 everywhere
    V: dict[State, float] = {s: 0.0 for s in states}
    # Policy: maps every state to the currently greedy action
    policy: dict[State, Action] = {}

    iterations = 0
    residual = float("inf")
    iteration_paths: list[list[State]] = []
    _max_path_steps = 4 * env.grid.nx * env.grid.ny

    for iterations in range(1, max_iterations + 1):
        max_delta = 0.0

        new_V: dict[State, float] = {}

        for state in states:
            if state == goal:
                # Absorbing terminal state: V*(goal) = 0
                new_V[state] = 0.0
                continue

            best_value = float("inf")
            best_action: Action | None = None

            for action in env.valid_actions(state):
                next_state = env.transition(state, action)

                # Transitions into the goal cell are only valid with a correct
                # approach angle.  Invalid transitions are treated as if the
                # action moves the ship to a boundary-clipped non-goal state
                # instead of the goal — i.e. we simply skip them.
                if next_state == goal and not env.is_goal(next_state, action):
                    continue

                c = cost_fn(state, action)
                value = c + gamma * V[next_state]

                if value < best_value:
                    best_value = value
                    best_action = action

            if best_action is None:
                # No valid action leads to an improvable state (shouldn't
                # normally happen on a connected grid).
                new_V[state] = V[state]
            else:
                new_V[state] = best_value
                policy[state] = best_action

            delta = abs(new_V[state] - V[state])
            if delta > max_delta:
                max_delta = delta

        V = new_V
        residual = max_delta

        # Snapshot: follow current policy greedily from start.
        snap: list[State] = [env.start]
        _s = env.start
        for _ in range(_max_path_steps):
            if _s == goal or _s not in policy:
                break
            _s = env.transition(_s, policy[_s])
            snap.append(_s)
            if _s == goal:
                break
        iteration_paths.append(snap)

        if residual < tolerance:
            break

    return ValueIterationResult(
        value_function=V,
        policy=policy,
        iterations=iterations,
        residual=residual,
        converged=(residual < tolerance),
        iteration_paths=iteration_paths,
    )


def extract_path(
    env: RiverEnvironment,
    policy: dict[State, Action],
    max_steps: int | None = None,
) -> tuple[list[State], list[Action]]:
    """Follow *policy* greedily from ``env.start`` to ``env.goal``.

    Parameters
    ----------
    env:
        Environment (provides start, goal, and transition function).
    policy:
        Greedy policy mapping states to actions.
    max_steps:
        Maximum number of steps before aborting (default: 4 × grid area).

    Returns
    -------
    path, actions
        ``path`` includes both endpoints; ``actions`` has length
        ``len(path) - 1``.  If the goal is not reached within ``max_steps``
        the returned path ends at the last visited state.
    """
    if max_steps is None:
        max_steps = 4 * env.grid.nx * env.grid.ny

    path: list[State] = [env.start]
    actions: list[Action] = []
    state = env.start

    for _ in range(max_steps):
        if state == env.goal:
            break
        action = policy.get(state)
        if action is None:
            break
        next_state = env.transition(state, action)
        actions.append(action)
        path.append(next_state)
        state = next_state

    return path, actions


class ValueIterationResult:
    """Result of a Value Iteration run.

    Attributes
    ----------
    value_function:
        Mapping from every grid state to its optimal value ``V*(s)``.
    policy:
        Greedy policy ``π*(s) = argmin_a [ c(s,a) + γ V*(T(s,a)) ]``.
    iterations:
        Number of full sweeps performed.
    residual:
        Final max |V_{k+1} − V_k| (Bellman residual).
    converged:
        ``True`` if the residual fell below *tolerance* before *max_iterations*.
    iteration_paths:
        One greedy policy-path snapshot per sweep, ordered by iteration.
        Each snapshot is the path followed from ``env.start`` under the
        current (partially-converged) policy.  Grows toward the optimal path
        as value iteration converges.
    """

    def __init__(
        self,
        value_function: dict[State, float],
        policy: dict[State, Action],
        iterations: int,
        residual: float,
        converged: bool,
        iteration_paths: list[list[State]] | None = None,
    ) -> None:
        self.value_function = value_function
        self.policy = policy
        self.iterations = iterations
        self.residual = residual
        self.converged = converged
        self.iteration_paths: list[list[State]] = iteration_paths or []

    @classmethod
    def from_config(
        cls,
        env: RiverEnvironment,
        cost_fn: CostFunction,
        dp_config: dict[str, Any],
    ) -> ValueIterationResult:
        """Run value iteration using parameters from the algorithm config.

        Parameters
        ----------
        env:
            River-crossing environment.
        cost_fn:
            Transition-cost function.
        dp_config:
            The ``dynamic_programming`` sub-section of the algorithm config.
        """
        return value_iteration(
            env,
            cost_fn,
            gamma=float(dp_config.get("gamma", 1.0)),
            tolerance=float(dp_config.get("tolerance", 1e-6)),
            max_iterations=int(dp_config.get("max_iterations", 10_000)),
        )

    def __repr__(self) -> str:
        return (
            f"ValueIterationResult("
            f"iterations={self.iterations}, "
            f"residual={self.residual:.2e}, "
            f"converged={self.converged})"
        )
