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
        # Cache for valid_actions: the grid and land/water layout are static,
        # so the result for a given state never changes.
        self._valid_actions_cache: dict[State, tuple[Action, ...]] = {}

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

        flow = make_flow(env_config["flow"], nx=grid.nx)

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
        """Return valid actions that stay in-bounds and in the water corridor.

        Additional geometric constraints (cached together with boundary checks):
        - From the **start** position only diagonal actions are permitted
          (the ship must enter the river at an angle).
        - The **goal** cell can only be reached via diagonal actions
          (docking approach angle requirement).
        """
        cached = self._valid_actions_cache.get(state)
        if cached is not None:
            return cached

        is_start = (state == self.start)
        result_list: list[Action] = []
        for a in self.actions:
            di, dj = a
            # Diagonal-only departure from start
            if is_start and (di == 0 or dj == 0):
                continue
            if not is_action_valid(state, a, self.grid):
                continue
            next_s = apply_action(state, a, self.grid)
            if self.is_land(next_s):
                continue
            # Diagonal-only arrival at goal
            if next_s == self.goal and (di == 0 or dj == 0):
                continue
            result_list.append(a)

        result = tuple(result_list)
        self._valid_actions_cache[state] = result
        return result

    def transition(self, state: State, action: Action) -> State:
        """Apply *action* to *state* and return the next state.

        Only in-bounds transitions are allowed.

        Raises
        ------
        ValueError
            If the action would leave the grid.
        """
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
        theta = self.approach_angle_deg(action)
        if theta is None:
            return False
        return self.docking.angle_min_deg <= theta <= self.docking.angle_max_deg

    def __repr__(self) -> str:
        return (
            f"RiverEnvironment(grid={self.grid!r}, "
            f"start={self.start!r}, goal={self.goal!r})"
        )
