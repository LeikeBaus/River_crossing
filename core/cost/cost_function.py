from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.dynamics.flow import FlowField
    from core.environment.actions import Action
    from core.environment.grid import State


class TimeCost(ABC):
    """Abstract base for time-cost components ``t(s, a)``."""

    @abstractmethod
    def __call__(self, state: State, action: Action) -> float:
        """Return the time cost of taking *action* from *state*."""


class EnergyCost(ABC):
    """Abstract base for energy-cost components ``E(s, a)``."""

    @abstractmethod
    def __call__(self, state: State, action: Action) -> float:
        """Return the energy cost of taking *action* from *state*."""


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


class EuclideanTimeCost(TimeCost):
    """Time cost equals the Euclidean length of the action vector.

    ``t(s, a) = ||a||_2``

    Physically this models that a longer step takes proportionally more time.
    """

    def __call__(self, state: State, action: Action) -> float:  # noqa: ARG002
        di, dj = action
        return math.sqrt(di * di + dj * dj)


class FlowEnergyCost(EnergyCost):
    """Energy cost based on the effort required to overcome the flow.

    ``E(s, a) = ||a - u_flow(s)||_2^2``

    A larger deviation between the chosen action and the local flow vector
    implies more thrust — and therefore more energy.

    Parameters
    ----------
    flow:
        Flow-field model used to look up the local velocity at any state.
    """

    def __init__(self, flow: FlowField) -> None:
        self._flow = flow

    def __call__(self, state: State, action: Action) -> float:
        vi, vj = self._flow.at(state)
        di, dj = action
        action_mag_sq = di * di + dj * dj
        if action_mag_sq == 0.0:
            return 0.0
        action_mag = math.sqrt(action_mag_sq)
        # Signed projection of flow onto the action direction.
        flow_along = (di * vi + dj * vj) / action_mag
        # Thrust needed along the action direction; clamped to 0 when flow
        # already carries the agent at least as far as needed.
        parallel_residual = max(0.0, action_mag - flow_along)
        # Squared magnitude of the flow component perpendicular to the action;
        # this must be counteracted regardless of flow strength.
        perp_sq = max(0.0, vi * vi + vj * vj - flow_along * flow_along)
        # Return L2 norm of required thrust so the base cost (no flow) matches
        # the Euclidean time cost: 1 for h/v moves, sqrt(2) for diagonals.
        return math.sqrt(parallel_residual * parallel_residual + perp_sq)


class TurnCost:
    """Penalty for changing heading, proportional to ``(1 - cos θ) / 2``.

    The reference heading is the component-wise average of the last *N*
    actions in *history* (where N = ``inertia``).  This gives:

    * 0.0  for continuing straight (θ = 0°)
    * 0.5  for a 90° turn
    * 1.0  for a 180° reversal

    Parameters
    ----------
    turn_penalty:
        Overall scaling factor applied to the (1 - cos θ) / 2 value.
    """

    def __init__(self, turn_penalty: float) -> None:
        self._penalty = turn_penalty

    def __call__(self, history: "tuple[Action, ...]", action: "Action") -> float:
        """Return the turn cost given recent action *history* and current *action*."""
        if not history:
            return 0.0
        n = len(history)
        avg_i = sum(a[0] for a in history) / n
        avg_j = sum(a[1] for a in history) / n
        mag_h = math.sqrt(avg_i * avg_i + avg_j * avg_j)
        di, dj = action
        mag_a = math.sqrt(di * di + dj * dj)
        if mag_h < 1e-9 or mag_a < 1e-9:
            return 0.0
        cos_t = max(-1.0, min(1.0, (di * avg_i + dj * avg_j) / (mag_a * mag_h)))
        return self._penalty * (1.0 - cos_t) / 2.0


# ---------------------------------------------------------------------------
# Combined cost function
# ---------------------------------------------------------------------------


class CostFunction:
    """Combined transition-cost function.

    ``c(s, a) = alpha * t(s, a) + beta * E(s, a)``

    Parameters
    ----------
    alpha:
        Weight for the time component (must be ≥ 0).
    beta:
        Weight for the energy component (must be ≥ 0).
    time_cost:
        A :class:`TimeCost` instance.
    energy_cost:
        An :class:`EnergyCost` instance.
    turn_cost:
        Optional :class:`TurnCost` instance.  ``None`` disables turn penalty.
    inertia:
        Number of past actions whose average forms the reference heading.
        0 = no memory (turn cost always 0), 1 = previous step only, etc.
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        time_cost: TimeCost,
        energy_cost: EnergyCost,
        turn_cost: TurnCost | None = None,
        inertia: int = 0,
    ) -> None:
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}.")
        if beta < 0:
            raise ValueError(f"beta must be >= 0, got {beta}.")
        self.alpha = alpha
        self.beta = beta
        self.inertia = max(0, int(inertia))
        self._time_cost = time_cost
        self._energy_cost = energy_cost
        self._turn_cost: TurnCost = turn_cost if turn_cost is not None else TurnCost(0.0)

    def __call__(self, state: State, action: Action) -> float:
        """Return ``alpha * t(s, a) + beta * E(s, a)`` (no turn cost; history-free)."""
        t = self._time_cost(state, action)
        e = self._energy_cost(state, action)
        return self.alpha * t + self.beta * e

    def with_history(
        self,
        state: State,
        action: Action,
        history: "tuple[Action, ...]",
    ) -> float:
        """Return the full cost including the turn-cost term.

        ``c(s, a, hist) = alpha * t(s,a) + beta * E(s,a) + turn(hist, a)``
        """
        return self(state, action) + self._turn_cost(history, action)

    def time(self, state: State, action: Action) -> float:
        """Return the raw time component ``t(s, a)`` (unweighted)."""
        return self._time_cost(state, action)

    def energy(self, state: State, action: Action) -> float:
        """Return the raw energy component ``E(s, a)`` (unweighted)."""
        return self._energy_cost(state, action)

    def turn(self, history: "tuple[Action, ...]", action: Action) -> float:
        """Return the raw turn cost ``T(hist, a)`` (unweighted)."""
        return self._turn_cost(history, action)

    @classmethod
    def from_config(cls, algo_config: dict[str, Any], flow: FlowField) -> CostFunction:
        """Build a :class:`CostFunction` from the algorithm config.

        Parameters
        ----------
        algo_config:
            Dictionary produced by :func:`core.config_loader.load_algorithm_config`.
        flow:
            Flow-field instance (required for :class:`FlowEnergyCost`).
        """
        common = algo_config.get("common", {})
        alpha = float(common.get("alpha", 1.0))
        beta = float(common.get("beta", 0.0))
        inertia = max(0, int(common.get("inertia", 0)))
        turn_penalty = float(common.get("turn_penalty", 1.0))
        turn_cost = TurnCost(turn_penalty) if inertia > 0 and turn_penalty > 0.0 else None
        return cls(
            alpha=alpha,
            beta=beta,
            time_cost=EuclideanTimeCost(),
            energy_cost=FlowEnergyCost(flow),
            turn_cost=turn_cost,
            inertia=inertia,
        )

    def __repr__(self) -> str:
        return (
            f"CostFunction(alpha={self.alpha}, beta={self.beta}, "
            f"time={self._time_cost.__class__.__name__}, "
            f"energy={self._energy_cost.__class__.__name__})"
        )
