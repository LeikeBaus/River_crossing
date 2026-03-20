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
        diff_i = di - vi
        diff_j = dj - vj
        return diff_i * diff_i + diff_j * diff_j


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
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        time_cost: TimeCost,
        energy_cost: EnergyCost,
    ) -> None:
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}.")
        if beta < 0:
            raise ValueError(f"beta must be >= 0, got {beta}.")
        self.alpha = alpha
        self.beta = beta
        self._time_cost = time_cost
        self._energy_cost = energy_cost

    def __call__(self, state: State, action: Action) -> float:
        """Return ``alpha * t(s, a) + beta * E(s, a)``."""
        t = self._time_cost(state, action)
        e = self._energy_cost(state, action)
        return self.alpha * t + self.beta * e

    def time(self, state: State, action: Action) -> float:
        """Return the raw time component ``t(s, a)`` (unweighted)."""
        return self._time_cost(state, action)

    def energy(self, state: State, action: Action) -> float:
        """Return the raw energy component ``E(s, a)`` (unweighted)."""
        return self._energy_cost(state, action)

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
        return cls(
            alpha=alpha,
            beta=beta,
            time_cost=EuclideanTimeCost(),
            energy_cost=FlowEnergyCost(flow),
        )

    def __repr__(self) -> str:
        return (
            f"CostFunction(alpha={self.alpha}, beta={self.beta}, "
            f"time={self._time_cost.__class__.__name__}, "
            f"energy={self._energy_cost.__class__.__name__})"
        )
