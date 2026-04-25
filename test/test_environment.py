"""Tests for Step 2: core model – grid, actions, flow field, and environment."""

from __future__ import annotations

import math
import unittest

from core.dynamics.flow import ConstantFlow, make_flow
from core.environment.actions import ACTIONS, apply_action, is_action_valid
from core.environment.environment import DockingConfig, RiverEnvironment
from core.environment.grid import Grid, State


# ---------------------------------------------------------------------------
# Grid tests
# ---------------------------------------------------------------------------


class TestGrid(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = Grid(10, 5)

    def test_dimensions_stored(self) -> None:
        self.assertEqual(self.grid.nx, 10)
        self.assertEqual(self.grid.ny, 5)

    def test_rejects_zero_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            Grid(0, 5)
        with self.assertRaises(ValueError):
            Grid(10, 0)

    def test_rejects_negative_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            Grid(-1, 5)

    def test_contains_interior(self) -> None:
        self.assertTrue(self.grid.contains(State(0, 0)))
        self.assertTrue(self.grid.contains(State(9, 4)))
        self.assertTrue(self.grid.contains(State(5, 2)))

    def test_contains_rejects_out_of_bounds(self) -> None:
        self.assertFalse(self.grid.contains(State(10, 0)))   # i == nx
        self.assertFalse(self.grid.contains(State(0, 5)))    # j == ny
        self.assertFalse(self.grid.contains(State(-1, 0)))
        self.assertFalse(self.grid.contains(State(0, -1)))

    def test_clip_interior_state_unchanged(self) -> None:
        s = State(3, 2)
        self.assertEqual(self.grid.clip(s), s)

    def test_clip_clamps_i_low(self) -> None:
        self.assertEqual(self.grid.clip(State(-5, 2)), State(0, 2))

    def test_clip_clamps_i_high(self) -> None:
        self.assertEqual(self.grid.clip(State(20, 2)), State(9, 2))

    def test_clip_clamps_j_low(self) -> None:
        self.assertEqual(self.grid.clip(State(3, -3)), State(3, 0))

    def test_clip_clamps_j_high(self) -> None:
        self.assertEqual(self.grid.clip(State(3, 10)), State(3, 4))

    def test_clip_clamps_both_axes(self) -> None:
        self.assertEqual(self.grid.clip(State(-1, 100)), State(0, 4))


# ---------------------------------------------------------------------------
# State tests
# ---------------------------------------------------------------------------


class TestState(unittest.TestCase):
    def test_equality(self) -> None:
        self.assertEqual(State(1, 2), State(1, 2))
        self.assertNotEqual(State(1, 2), State(2, 1))

    def test_hashable(self) -> None:
        seen = {State(1, 2), State(3, 4)}
        self.assertIn(State(1, 2), seen)

    def test_frozen(self) -> None:
        s = State(1, 2)
        with self.assertRaises((AttributeError, TypeError)):
            s.i = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Action space tests
# ---------------------------------------------------------------------------


class TestActions(unittest.TestCase):
    def test_eight_actions(self) -> None:
        self.assertEqual(len(ACTIONS), 8)

    def test_all_unit_or_diagonal(self) -> None:
        for a in ACTIONS:
            norm_sq = a[0] ** 2 + a[1] ** 2
            # Either |a| = 1 (axis-aligned) or |a| = √2 (diagonal)
            self.assertIn(norm_sq, (1, 2), msg=f"Unexpected action vector: {a}")

    def test_actions_cover_all_directions(self) -> None:
        # Each of the four axis-aligned and four diagonal directions present
        axis = {(0, 1), (1, 0), (0, -1), (-1, 0)}
        diag = {(1, 1), (1, -1), (-1, -1), (-1, 1)}
        action_set = set(ACTIONS)
        self.assertTrue(axis.issubset(action_set))
        self.assertTrue(diag.issubset(action_set))


# ---------------------------------------------------------------------------
# apply_action tests
# ---------------------------------------------------------------------------


class TestApplyAction(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = Grid(5, 5)

    def test_move_north(self) -> None:
        result = apply_action(State(2, 2), (0, 1), self.grid)
        self.assertEqual(result, State(2, 3))

    def test_move_east(self) -> None:
        result = apply_action(State(2, 2), (1, 0), self.grid)
        self.assertEqual(result, State(3, 2))

    def test_move_diagonal(self) -> None:
        result = apply_action(State(2, 2), (1, 1), self.grid)
        self.assertEqual(result, State(3, 3))

    def test_invalid_at_north_wall_raises(self) -> None:
        with self.assertRaises(ValueError):
            apply_action(State(2, 4), (0, 1), self.grid)

    def test_invalid_at_east_wall_raises(self) -> None:
        with self.assertRaises(ValueError):
            apply_action(State(4, 2), (1, 0), self.grid)

    def test_invalid_at_south_wall_raises(self) -> None:
        with self.assertRaises(ValueError):
            apply_action(State(2, 0), (0, -1), self.grid)

    def test_invalid_at_west_wall_raises(self) -> None:
        with self.assertRaises(ValueError):
            apply_action(State(0, 2), (-1, 0), self.grid)

    def test_result_in_grid_for_valid_actions_only(self) -> None:
        # Exhaustive check: valid state-action pairs always stay inside
        for i in range(self.grid.nx):
            for j in range(self.grid.ny):
                for action in ACTIONS:
                    state = State(i, j)
                    if is_action_valid(state, action, self.grid):
                        result = apply_action(state, action, self.grid)
                        self.assertTrue(
                            self.grid.contains(result),
                            msg=f"apply_action({state}, {action}) → {result} outside grid",
                        )
                    else:
                        with self.assertRaises(ValueError):
                            apply_action(state, action, self.grid)


# ---------------------------------------------------------------------------
# Flow field tests
# ---------------------------------------------------------------------------


class TestConstantFlow(unittest.TestCase):
    def test_returns_configured_vector(self) -> None:
        flow = ConstantFlow(vi=1.0, vj=0.0)
        self.assertEqual(flow.at(State(0, 0)), (1.0, 0.0))
        self.assertEqual(flow.at(State(5, 3)), (1.0, 0.0))

    def test_uniform_across_all_states(self) -> None:
        flow = ConstantFlow(vi=2.5, vj=-0.7)
        grid = Grid(4, 4)
        for i in range(grid.nx):
            for j in range(grid.ny):
                self.assertEqual(flow.at(State(i, j)), (2.5, -0.7))


class TestMakeFlow(unittest.TestCase):
    def test_constant_flow(self) -> None:
        config = {"type": "constant", "vector": [3.0, -1.0]}
        flow = make_flow(config)
        self.assertIsInstance(flow, ConstantFlow)
        self.assertEqual(flow.at(State(0, 0)), (3.0, -1.0))

    def test_unknown_type_raises(self) -> None:
        with self.assertRaises(ValueError):
            make_flow({"type": "turbulent"})

    def test_constant_missing_vector_raises(self) -> None:
        with self.assertRaises((ValueError, KeyError)):
            make_flow({"type": "constant"})


# ---------------------------------------------------------------------------
# DockingConfig / approach angle tests
# ---------------------------------------------------------------------------


class TestApproachAngle(unittest.TestCase):
    """Test diagonal-only final-move validity."""

    def setUp(self) -> None:
        grid = Grid(40, 20)
        flow = ConstantFlow(1.0, 0.0)
        docking = DockingConfig(
            normal=(0.0, 1.0),
            angle_min_deg=30.0,
            angle_max_deg=60.0,
        )
        self.env = RiverEnvironment(grid, flow, State(2, 10), State(37, 10), docking)

    def test_diagonal_ne_valid(self) -> None:
        # Any diagonal entry into the goal is valid.
        theta = self.env.approach_angle_deg((1, 1))
        self.assertAlmostEqual(theta, 45.0, places=10)
        self.assertTrue(self.env._approach_angle_valid((1, 1)))

    def test_diagonal_nw_valid(self) -> None:
        # Orientation-independent diagonal validity.
        theta = self.env.approach_angle_deg((-1, 1))
        self.assertAlmostEqual(theta, 45.0, places=10)
        self.assertTrue(self.env._approach_angle_valid((-1, 1)))

    def test_straight_north_invalid(self) -> None:
        # (0, 1) → θ = 0°
        theta = self.env.approach_angle_deg((0, 1))
        self.assertAlmostEqual(theta, 0.0, places=10)
        self.assertFalse(self.env._approach_angle_valid((0, 1)))

    def test_straight_east_invalid(self) -> None:
        # (1, 0) → θ = 90°
        self.assertFalse(self.env._approach_angle_valid((1, 0)))

    def test_zero_action_invalid(self) -> None:
        self.assertIsNone(self.env.approach_angle_deg((0, 0)))
        self.assertFalse(self.env._approach_angle_valid((0, 0)))


# ---------------------------------------------------------------------------
# RiverEnvironment tests
# ---------------------------------------------------------------------------


class TestRiverEnvironment(unittest.TestCase):
    def _make_env(self, start: State = State(2, 10), goal: State = State(37, 10)) -> RiverEnvironment:
        grid = Grid(40, 20)
        flow = ConstantFlow(1.0, 0.0)
        docking = DockingConfig(normal=(0.0, 1.0), angle_min_deg=30.0, angle_max_deg=60.0)
        return RiverEnvironment(grid, flow, start, goal, docking)

    def test_from_config(self) -> None:
        from core.config_loader import load_env_config
        from pathlib import Path

        env_cfg = load_env_config(Path("configs/env.yaml"))
        env = RiverEnvironment.from_config(env_cfg)
        self.assertEqual(env.grid.nx, 40)
        self.assertEqual(env.grid.ny, 20)
        self.assertEqual(env.start, State(2, 10))
        self.assertEqual(env.goal, State(37, 10))

    def test_rejects_start_outside_grid(self) -> None:
        with self.assertRaises(ValueError):
            self._make_env(start=State(100, 10))

    def test_rejects_goal_outside_grid(self) -> None:
        with self.assertRaises(ValueError):
            self._make_env(goal=State(100, 10))

    def test_rejects_equal_start_and_goal(self) -> None:
        with self.assertRaises(ValueError):
            self._make_env(start=State(5, 5), goal=State(5, 5))

    def test_rejects_start_not_left_of_goal(self) -> None:
        with self.assertRaises(ValueError):
            self._make_env(start=State(10, 5), goal=State(10, 6))
        with self.assertRaises(ValueError):
            self._make_env(start=State(11, 5), goal=State(10, 6))

    def test_transition_moves_ship(self) -> None:
        env = self._make_env()
        next_state = env.transition(State(2, 10), (1, 1))
        self.assertEqual(next_state, State(3, 11))

    def test_transition_clips_at_boundary(self) -> None:
        env = self._make_env()
        s = State(39, 19)
        with self.assertRaises(ValueError):
            env.transition(s, (1, 1))

    def test_valid_actions_excludes_out_of_grid_moves(self) -> None:
        env = self._make_env()
        corner = State(2, 0)
        valid = set(env.valid_actions(corner))
        self.assertNotIn((-1, 0), valid)
        self.assertNotIn((0, -1), valid)
        self.assertNotIn((-1, -1), valid)
        self.assertIn((1, 0), valid)
        self.assertIn((0, 1), valid)
        self.assertIn((1, 1), valid)

    def test_valid_actions_at_start_exclude_land(self) -> None:
        env = self._make_env(start=State(2, 10))
        valid = set(env.valid_actions(env.start))

        # All in-bounds, non-land directions are valid (no diagonal-only constraint).
        self.assertIn((1, 1), valid)
        self.assertIn((1, -1), valid)
        self.assertIn((1, 0), valid)
        self.assertIn((0, 1), valid)
        self.assertIn((0, -1), valid)
        self.assertNotIn((-1, 0), valid)  # left side is land

    def test_valid_actions_non_start_do_not_use_angle_filter(self) -> None:
        env = self._make_env(start=State(2, 10))
        valid = set(env.valid_actions(State(3, 10)))

        self.assertIn((1, 0), valid)
        self.assertIn((0, 1), valid)
        self.assertIn((1, 1), valid)

    def test_land_cells_are_invalid_targets(self) -> None:
        env = self._make_env(start=State(2, 10), goal=State(37, 10))
        self.assertTrue(env.is_land(State(1, 10)))
        self.assertFalse(env.is_land(State(2, 10)))
        self.assertFalse(env.is_land(State(37, 10)))
        self.assertTrue(env.is_land(State(38, 10)))

    def test_transition_to_land_raises(self) -> None:
        env = self._make_env(start=State(2, 10), goal=State(37, 10))
        with self.assertRaises(ValueError):
            env.transition(State(2, 10), (-1, 0))

    def test_transition_non_diagonal_from_start_permitted(self) -> None:
        # Non-diagonal departure from start is permitted; goal-arrival is
        # validated by angle bounds in is_goal(), not in transition().
        env = self._make_env(start=State(2, 10), goal=State(37, 10))
        next_state = env.transition(env.start, (1, 0))
        self.assertEqual(next_state, State(3, 10))

    def test_is_goal_requires_correct_position_and_angle(self) -> None:
        env = self._make_env()
        goal = State(37, 10)
        # Valid: at goal, approach angle within docking range [30°, 60°].
        self.assertTrue(env.is_goal(goal, (1, 1)))    # 45° — in range
        self.assertTrue(env.is_goal(goal, (-1, 1)))   # 45° — in range
        # Invalid: wrong position.
        self.assertFalse(env.is_goal(State(36, 10), (1, 1)))
        # Invalid: approach angle outside [30°, 60°].
        self.assertFalse(env.is_goal(goal, (1, -1)))  # 135° — out of range
        self.assertFalse(env.is_goal(goal, (0, 1)))   # 0°  — out of range
        self.assertFalse(env.is_goal(goal, (1, 0)))   # 90° — out of range

    def test_actions_property(self) -> None:
        env = self._make_env()
        self.assertEqual(env.actions, ACTIONS)
        self.assertEqual(len(env.actions), 8)

    def test_flow_at_returns_configured_flow(self) -> None:
        env = self._make_env()
        vi, vj = env.flow_at(State(5, 5))
        self.assertAlmostEqual(vi, 1.0)
        self.assertAlmostEqual(vj, 0.0)


if __name__ == "__main__":
    unittest.main()
