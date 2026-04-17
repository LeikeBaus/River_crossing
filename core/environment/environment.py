from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from core.dynamics.flow import FlowField, FlowVector, make_flow
from core.environment.actions import ACTIONS, Action, apply_action, is_action_valid
from core.environment.grid import Grid, State


@dataclass(frozen=True)
class DockingConfig:
    """Geometric specification of the target docking berth.

    Attributes
    ----------
    normal:
        Unit normal vector of the berth face ``(ni, nj)``.  The ship must
        approach along a direction whose angle with this vector lies within
        ``[angle_min_deg, angle_max_deg]``.
    angle_min_deg:
        Minimum valid approach angle in degrees (inclusive).
    angle_max_deg:
        Maximum valid approach angle in degrees (inclusive).
    """

    normal: tuple[float, float]
    angle_min_deg: float
    angle_max_deg: float


class RiverEnvironment:
    """Fully-specified river-crossing environment.

    Combines the discrete grid, flow field, start position, goal position and
    docking constraints into a single object that the planning algorithms can
    query.

    Parameters
    ----------
    grid:
        Discrete 2-D grid defining the state space.
    flow:
        Flow-field model.
    start:
        Initial position of the ship.
    goal:
        Target docking position.
    docking:
        Approach-angle and berth-normal specifications.
    """

    def __init__(
        self,
        grid: Grid,
        flow: FlowField,
        start: State,
        goal: State,
        docking: DockingConfig,
    ) -> None:
        if not grid.contains(start):
            raise ValueError(f"Start {start} lies outside the grid {grid}.")
        if not grid.contains(goal):
            raise ValueError(f"Goal {goal} lies outside the grid {grid}.")
        if start == goal:
            raise ValueError("Start and goal must be different states.")
        if start.i >= goal.i:
            raise ValueError("Start must be left of goal (start.i < goal.i).")

        self.grid = grid
        self.flow = flow
        self.start = start
        self.goal = goal
        self.docking = docking

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, env_config: dict[str, Any]) -> RiverEnvironment:
        """Build a :class:`RiverEnvironment` from a loaded environment config.

        Parameters
        ----------
        env_config:
            Dictionary produced by :func:`core.config_loader.load_env_config`.
        """
        grid_cfg = env_config["grid"]
        grid = Grid(grid_cfg["nx"], grid_cfg["ny"])

        flow = make_flow(env_config["flow"])

        start = State(*env_config["start"])
        goal = State(*env_config["goal"])

        docking_cfg = env_config["docking"]
        n = docking_cfg["normal"]
        angle_range = docking_cfg["approach_angle_deg"]
        docking = DockingConfig(
            normal=(float(n[0]), float(n[1])),
            angle_min_deg=float(angle_range["min"]),
            angle_max_deg=float(angle_range["max"]),
        )

        return cls(grid, flow, start, goal, docking)

    # ------------------------------------------------------------------
    # Core MDP interface
    # ------------------------------------------------------------------

    @property
    def actions(self) -> tuple[Action, ...]:
        """Return the full action set (8 Moore-neighbourhood directions)."""
        return ACTIONS

    def valid_actions(self, state: State) -> tuple[Action, ...]:
        """Return valid actions that stay in-bounds and in the water corridor."""
        actions = tuple(a for a in self.actions if is_action_valid(state, a, self.grid))
        actions = tuple(a for a in actions if not self.is_land(apply_action(state, a, self.grid)))
        if state == self.start:
            return tuple(a for a in actions if self._approach_angle_valid(a))
        return actions

    def transition(self, state: State, action: Action) -> State:
        """Apply *action* to *state* and return the next state.

        Only in-bounds transitions are allowed.

        Raises
        ------
        ValueError
            If the action would leave the grid.
        """
        if state == self.start and not self._approach_angle_valid(action):
            raise ValueError(f"Action {action} from {state} is not a valid diagonal start departure.")
        next_state = apply_action(state, action, self.grid)
        if self.is_land(next_state):
            raise ValueError(f"Action {action} from {state} enters land at {next_state}.")
        return next_state

    def flow_at(self, state: State) -> FlowVector:
        """Return the flow velocity vector at *state*."""
        return self.flow.at(state)

    # ------------------------------------------------------------------
    # Goal / terminal condition
    # ------------------------------------------------------------------

    def is_goal(self, state: State, last_action: Action) -> bool:
        """Return True if *state* is the goal reached with a valid final move.

        Goal arrival is valid only via diagonal actions. The docking angle
        configuration is retained for compatibility but no longer constrains
        orientation relative to a berth normal.
        """
        if state != self.goal:
            return False
        return self._approach_angle_valid(last_action)

    def is_land(self, state: State) -> bool:
        """Return True if *state* lies outside the water corridor.

        All columns strictly left of the start or strictly right of the goal are
        land.
        """
        return state.i < self.start.i or state.i > self.goal.i

    def approach_angle_deg(self, action: Action) -> float | None:
        """Return the approach angle in degrees for *action*, or None if the action is the zero vector."""
        ai, aj = float(action[0]), float(action[1])
        norm = math.sqrt(ai * ai + aj * aj)
        if norm == 0.0:
            return None
        ni, nj = self.docking.normal
        n_norm = math.sqrt(ni * ni + nj * nj)
        if n_norm == 0.0:
            return None
        cos_theta = (ai * ni + aj * nj) / (norm * n_norm)
        cos_theta = max(-1.0, min(1.0, cos_theta))  # numerical safety
        return math.degrees(math.acos(cos_theta))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _approach_angle_valid(self, action: Action) -> bool:
        return abs(int(action[0])) == 1 and abs(int(action[1])) == 1

    def __repr__(self) -> str:
        return (
            f"RiverEnvironment(grid={self.grid!r}, "
            f"start={self.start!r}, goal={self.goal!r})"
        )
