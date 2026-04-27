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


def is_action_valid(state: State, action: Action, grid: Grid) -> bool:
    """Return True when applying *action* keeps the vessel inside *grid*."""
    next_state = State(state.i + action[0], state.j + action[1])
    return grid.contains(next_state)


def apply_action(state: State, action: Action, grid: Grid) -> State:
    """Apply *action* to *state* and return the exact next state.

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
        Next position.

    Raises
    ------
    ValueError
        If the action would leave the grid.
    """
    if not is_action_valid(state, action, grid):
        raise ValueError(
            f"Action {action} from state {state} would leave the grid {grid}."
        )
    return State(state.i + action[0], state.j + action[1])
