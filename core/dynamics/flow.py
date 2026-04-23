from __future__ import annotations

import math
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


class GaussianFlow(FlowField):
    """Flow field with Gaussian cross-section – maximum at channel centre.

    The magnitude profile across the i-axis (horizontal / column index) follows:

        m(i) = floor + (1 - floor) * exp(-0.5 * ((i - i_c) / sigma)^2)

    where ``i_c`` is the centre column.  Every cell in the same column shares
    the same magnitude, so the "fast lane" runs as a full vertical stripe down
    the middle of the grid.  The direction vector (vi, vj) is scaled by
    ``m(i)`` so the flow is always > 0 at every cell.

    Parameters
    ----------
    vi, vj:
        Base direction vector components (will be scaled by the profile).
    nx:
        Number of grid columns – used to locate the centre column.
    sigma:
        Gaussian width in grid cells.  Larger values = wider fast zone.
    floor_frac:
        Minimum flow fraction at the shore edges (0 < floor_frac < 1).
    """

    def __init__(
        self,
        vi: float,
        vj: float,
        nx: int,
        sigma: float,
        floor_frac: float,
    ) -> None:
        self._vi = vi
        self._vj = vj
        self._ic = (nx - 1) / 2.0
        self._sigma = max(sigma, 1e-6)
        self._floor = max(1e-3, min(0.999, floor_frac))

    def at(self, state: "State") -> FlowVector:
        g = math.exp(-0.5 * ((state.i - self._ic) / self._sigma) ** 2)
        scale = self._floor + (1.0 - self._floor) * g
        return (self._vi * scale, self._vj * scale)

    def __repr__(self) -> str:
        return (
            f"GaussianFlow(vi={self._vi}, vj={self._vj}, "
            f"ic={self._ic}, sigma={self._sigma}, floor={self._floor})"
        )


def make_flow(config: dict[str, Any], nx: int = 20) -> FlowField:
    """Instantiate a :class:`FlowField` from a configuration dictionary.

    The dictionary must contain a ``"type"`` key.  Currently supported types:

    ``"constant"``
        Requires a ``"vector"`` key with a two-element list ``[vi, vj]``.

    ``"gaussian"``
        Requires a ``"vector"`` key.  Optionally accepts ``"sigma"`` (default
        5.0) and ``"floor"`` (default 0.1) to control the Gaussian profile.

    Parameters
    ----------
    config:
        Flow sub-section of the environment configuration.
    nx:
        Number of grid columns – passed through to :class:`GaussianFlow` so it
        can locate the centre column.

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
    vec = config.get("vector")
    if vec is None or len(vec) != 2:
        raise ValueError(f"{flow_type!r} flow requires 'vector' with two elements.")
    vi, vj = float(vec[0]), float(vec[1])

    if flow_type == "constant":
        return ConstantFlow(vi, vj)

    if flow_type == "gaussian":
        sigma = float(config.get("sigma", 5.0))
        floor_frac = float(config.get("floor", 0.1))
        return GaussianFlow(vi, vj, nx=nx, sigma=sigma, floor_frac=floor_frac)

    raise ValueError(f"Unknown flow type: {flow_type!r}")
