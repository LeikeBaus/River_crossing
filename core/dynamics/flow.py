from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.environment.grid import State

# A flow velocity is a 2-D real-valued vector (vi, vj).
FlowVector = tuple[float, float]


class FlowField(ABC):
    """Abstract base class for all flow-field models."""

    @abstractmethod
    def at(self, state: State) -> FlowVector:
        """Return the flow velocity vector at *state*.

        Parameters
        ----------
        state:
            Grid position to query.

        Returns
        -------
        FlowVector
            ``(vi, vj)`` – velocity components along the i- and j-axes.
        """


class ConstantFlow(FlowField):
    """Spatially uniform, time-invariant flow field.

    Parameters
    ----------
    vi:
        Flow velocity along the i-axis (columns).
    vj:
        Flow velocity along the j-axis (rows).
    """

    def __init__(self, vi: float, vj: float) -> None:
        self._vi = vi
        self._vj = vj

    def at(self, state: State) -> FlowVector:  # noqa: ARG002
        return (self._vi, self._vj)

    def __repr__(self) -> str:
        return f"ConstantFlow(vi={self._vi}, vj={self._vj})"


def make_flow(config: dict[str, Any]) -> FlowField:
    """Instantiate a :class:`FlowField` from a configuration dictionary.

    The dictionary must contain a ``"type"`` key.  Currently supported types:

    ``"constant"``
        Requires a ``"vector"`` key with a two-element list ``[vi, vj]``.

    Parameters
    ----------
    config:
        Flow sub-section of the environment configuration.

    Returns
    -------
    FlowField
        Concrete flow-field instance.

    Raises
    ------
    ValueError
        If the flow type is unknown or required keys are missing.
    """
    flow_type = config.get("type")
    if flow_type == "constant":
        vec = config.get("vector")
        if vec is None or len(vec) != 2:
            raise ValueError("Constant flow requires 'vector' with two elements.")
        return ConstantFlow(float(vec[0]), float(vec[1]))
    raise ValueError(f"Unknown flow type: {flow_type!r}")
