from __future__ import annotations

from typing import Final

from core.environment.grid import Grid, State

# A single action is a 2-D integer displacement vector (di, dj).
Action = tuple[int, int]

# All eight neighbours in a Moore neighbourhood.
# Order: N, NE, E, SE, S, SW, W, NW  (j increases upward by convention).
ACTIONS: Final[tuple[Action, ...]] = (
    (0, 1),   # N
    (1, 1),   # NE
    (1, 0),   # E
    (1, -1),  # SE
    (0, -1),  # S
    (-1, -1), # SW
    (-1, 0),  # W
    (-1, 1),  # NW
)


def apply_action(state: State, action: Action, grid: Grid) -> State:
    """Apply *action* to *state* and clip the result to the grid.

    Parameters
    ----------
    state:
        Current position.
    action:
        Displacement vector chosen from :data:`ACTIONS`.
    grid:
        Grid used for boundary clipping.

    Returns
    -------
    State
        Next position, guaranteed to be inside *grid*.
    """
    raw = State(state.i + action[0], state.j + action[1])
    return grid.clip(raw)
