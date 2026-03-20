from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class State:
    """Discrete position on the 2-D grid.

    Attributes
    ----------
    i:
        Column index – runs from 0 to nx-1 (horizontal / x-axis).
    j:
        Row index – runs from 0 to ny-1 (vertical / y-axis).
    """

    i: int
    j: int


class Grid:
    """Discrete 2-D grid of shape nx × ny.

    Parameters
    ----------
    nx:
        Number of columns (must be > 0).
    ny:
        Number of rows (must be > 0).
    """

    def __init__(self, nx: int, ny: int) -> None:
        if nx <= 0 or ny <= 0:
            raise ValueError(f"Grid dimensions must be > 0, got nx={nx}, ny={ny}.")
        self.nx = nx
        self.ny = ny

    def contains(self, state: State) -> bool:
        """Return True if *state* lies within the grid boundaries."""
        return 0 <= state.i < self.nx and 0 <= state.j < self.ny

    def clip(self, state: State) -> State:
        """Clamp *state* so that it lies inside the grid.

        States already inside the grid are returned unchanged.
        """
        i = max(0, min(self.nx - 1, state.i))
        j = max(0, min(self.ny - 1, state.j))
        return State(i, j)

    def __repr__(self) -> str:
        return f"Grid(nx={self.nx}, ny={self.ny})"
