"""Tests for Step 3: cost function and approach-angle hard constraint."""

from __future__ import annotations

import math
import unittest

from core.cost.cost_function import (
    CostFunction,
    EuclideanTimeCost,
    FlowEnergyCost,
)
from core.dynamics.flow import ConstantFlow, make_flow
from core.environment.actions import ACTIONS
from core.environment.environment import DockingConfig, RiverEnvironment
from core.environment.grid import Grid, State


def _make_env(
    start: State = State(2, 10),
    goal: State = State(37, 10),
    normal: tuple[float, float] = (0.0, 1.0),
    angle_min: float = 30.0,
    angle_max: float = 60.0,
) -> RiverEnvironment:
    grid = Grid(40, 20)
    flow = ConstantFlow(1.0, 0.0)
    docking = DockingConfig(normal=normal, angle_min_deg=angle_min, angle_max_deg=angle_max)
    return RiverEnvironment(grid, flow, start, goal, docking)


def _make_cost(alpha: float = 1.0, beta: float = 0.0) -> CostFunction:
    flow = ConstantFlow(1.0, 0.0)
    return CostFunction(
        alpha=alpha,
        beta=beta,
        time_cost=EuclideanTimeCost(),
        energy_cost=FlowEnergyCost(flow),
    )


# ---------------------------------------------------------------------------
# EuclideanTimeCost
# ---------------------------------------------------------------------------


class TestEuclideanTimeCost(unittest.TestCase):
    def setUp(self) -> None:
        self.cost = EuclideanTimeCost()
        self.s = State(5, 5)

    def test_axis_action_length_one(self) -> None:
        self.assertAlmostEqual(self.cost(self.s, (1, 0)), 1.0)
        self.assertAlmostEqual(self.cost(self.s, (0, 1)), 1.0)
        self.assertAlmostEqual(self.cost(self.s, (-1, 0)), 1.0)
        self.assertAlmostEqual(self.cost(self.s, (0, -1)), 1.0)

    def test_diagonal_action_length_sqrt2(self) -> None:
        for a in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            with self.subTest(action=a):
                self.assertAlmostEqual(self.cost(self.s, a), math.sqrt(2))

    def test_state_independent(self) -> None:
        a = (1, 0)
        self.assertAlmostEqual(
            self.cost(State(0, 0), a),
            self.cost(State(9, 9), a),
        )

    def test_all_eight_actions(self) -> None:
        for action in ACTIONS:
            val = self.cost(self.s, action)
            self.assertGreater(val, 0.0)


# ---------------------------------------------------------------------------
# FlowEnergyCost
# ---------------------------------------------------------------------------


class TestFlowEnergyCost(unittest.TestCase):
    def test_action_equals_flow_gives_zero_energy(self) -> None:
        # flow = (1, 0), action = (1, 0) → diff = 0 → E = 0
        flow = ConstantFlow(vi=1.0, vj=0.0)
        cost = FlowEnergyCost(flow)
        self.assertAlmostEqual(cost(State(0, 0), (1, 0)), 0.0)

    def test_perpendicular_action(self) -> None:
        # flow = (1, 0), action = (0, 1)
        # flow_along=0, parallel_residual=1, perp_sq=1 → E = sqrt(2)
        flow = ConstantFlow(vi=1.0, vj=0.0)
        cost = FlowEnergyCost(flow)
        self.assertAlmostEqual(cost(State(0, 0), (0, 1)), math.sqrt(2))

    def test_opposing_action(self) -> None:
        # flow = (1, 0), action = (-1, 0)
        # flow_along=-1, parallel_residual=2, perp_sq=0 → E = sqrt(4) = 2
        flow = ConstantFlow(vi=1.0, vj=0.0)
        cost = FlowEnergyCost(flow)
        self.assertAlmostEqual(cost(State(0, 0), (-1, 0)), 2.0)

    def test_zero_flow_energy_equals_action_norm(self) -> None:
        flow = ConstantFlow(vi=0.0, vj=0.0)
        cost = FlowEnergyCost(flow)
        # No flow: energy equals Euclidean action length
        # h/v action (1, 0) → E = 1
        self.assertAlmostEqual(cost(State(0, 0), (1, 0)), 1.0)
        # diagonal action (1, 1) → E = sqrt(2)
        self.assertAlmostEqual(cost(State(0, 0), (1, 1)), math.sqrt(2))

    def test_spatially_uniform(self) -> None:
        flow = ConstantFlow(vi=0.5, vj=0.3)
        cost = FlowEnergyCost(flow)
        a = (1, 0)
        # Same cost at every position
        self.assertAlmostEqual(cost(State(0, 0), a), cost(State(5, 7), a))

    def test_nonnegative_for_all_actions(self) -> None:
        flow = ConstantFlow(vi=0.7, vj=-0.3)
        cost = FlowEnergyCost(flow)
        s = State(3, 3)
        for action in ACTIONS:
            self.assertGreaterEqual(cost(s, action), 0.0)

    def test_strong_aligned_flow_gives_zero_energy(self) -> None:
        # flow = (3, 0), action = (1, 0) — flow overshoots along action dir → E = 0
        flow = ConstantFlow(vi=3.0, vj=0.0)
        cost = FlowEnergyCost(flow)
        self.assertAlmostEqual(cost(State(0, 0), (1, 0)), 0.0)

    def test_energy_minimum_at_max_aligned_flow(self) -> None:
        # Increasing aligned flow from 0 to 5: cost must be non-increasing
        s = State(0, 0)
        prev_cost = None
        for strength in [0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0]:
            flow = ConstantFlow(vi=strength, vj=0.0)
            cost = FlowEnergyCost(flow)
            val = cost(s, (1, 0))
            if prev_cost is not None:
                self.assertLessEqual(round(val, 10), round(prev_cost, 10))
            prev_cost = val


# ---------------------------------------------------------------------------
# CostFunction
# ---------------------------------------------------------------------------


class TestCostFunction(unittest.TestCase):
    def test_alpha_only_equals_time_cost(self) -> None:
        cf = _make_cost(alpha=1.0, beta=0.0)
        s = State(5, 5)
        # Axis action: expected = 1.0 * 1.0 + 0.0 * anything = 1.0
        self.assertAlmostEqual(cf(s, (1, 0)), 1.0)
        # Diagonal: expected = 1.0 * sqrt(2)
        self.assertAlmostEqual(cf(s, (1, 1)), math.sqrt(2))

    def test_beta_only_equals_energy_cost(self) -> None:
        # flow = (1, 0), action = (-1, 0): E = 2
        cf = _make_cost(alpha=0.0, beta=1.0)
        s = State(5, 5)
        self.assertAlmostEqual(cf(s, (-1, 0)), 2.0)

    def test_combined_weights(self) -> None:
        # alpha=2, beta=3, flow=(1,0), action=(0,1)
        # t = 1.0, E = sqrt(2) → c = 2*1 + 3*sqrt(2)
        cf = _make_cost(alpha=2.0, beta=3.0)
        s = State(5, 5)
        self.assertAlmostEqual(cf(s, (0, 1)), 2.0 + 3.0 * math.sqrt(2))

    def test_time_component_accessor(self) -> None:
        cf = _make_cost(alpha=1.0, beta=0.0)
        s = State(0, 0)
        self.assertAlmostEqual(cf.time(s, (1, 0)), 1.0)

    def test_energy_component_accessor(self) -> None:
        # flow=(1,0), action=(1,0): E=0
        cf = _make_cost(alpha=0.0, beta=1.0)
        s = State(0, 0)
        self.assertAlmostEqual(cf.energy(s, (1, 0)), 0.0)

    def test_rejects_negative_alpha(self) -> None:
        flow = ConstantFlow(1.0, 0.0)
        with self.assertRaises(ValueError):
            CostFunction(
                alpha=-0.1,
                beta=0.0,
                time_cost=EuclideanTimeCost(),
                energy_cost=FlowEnergyCost(flow),
            )

    def test_rejects_negative_beta(self) -> None:
        flow = ConstantFlow(1.0, 0.0)
        with self.assertRaises(ValueError):
            CostFunction(
                alpha=1.0,
                beta=-1.0,
                time_cost=EuclideanTimeCost(),
                energy_cost=FlowEnergyCost(flow),
            )

    def test_nonnegative_for_all_actions(self) -> None:
        cf = _make_cost(alpha=1.0, beta=0.5)
        s = State(3, 3)
        for action in ACTIONS:
            self.assertGreaterEqual(cf(s, action), 0.0)

    def test_from_config(self) -> None:
        from pathlib import Path
        from core.config_loader import load_algorithm_config

        algo_cfg = load_algorithm_config(Path("configs/algorithm.yaml"))
        flow = ConstantFlow(1.0, 0.0)
        cf = CostFunction.from_config(algo_cfg, flow)
        # Default: alpha=1.0, beta=0.0 → cost of (1,0) = 1.0
        self.assertAlmostEqual(cf(State(0, 0), (1, 0)), 1.0)


# ---------------------------------------------------------------------------
# Path cost integration test
# ---------------------------------------------------------------------------


class TestPathCost(unittest.TestCase):
    """Verify total path cost J(π) = Σ c(s_t, a_t)."""

    def test_path_of_east_steps(self) -> None:
        cf = _make_cost(alpha=1.0, beta=0.0)
        # 3 steps east: each costs 1.0 → total = 3.0
        states = [State(0, 0), State(1, 0), State(2, 0), State(3, 0)]
        action = (1, 0)
        total = sum(cf(s, action) for s in states[:-1])
        self.assertAlmostEqual(total, 3.0)

    def test_diagonal_path(self) -> None:
        cf = _make_cost(alpha=1.0, beta=0.0)
        # 2 diagonal steps: each costs sqrt(2) → total = 2*sqrt(2)
        states = [State(0, 0), State(1, 1), State(2, 2)]
        action = (1, 1)
        total = sum(cf(s, action) for s in states[:-1])
        self.assertAlmostEqual(total, 2 * math.sqrt(2))


# ---------------------------------------------------------------------------
# Approach-angle hard constraint (integration with RiverEnvironment)
# ---------------------------------------------------------------------------


class TestApproachAngleConstraint(unittest.TestCase):
    """Ensure is_goal enforces the approach-angle hard constraint."""

    def setUp(self) -> None:
        self.env = _make_env()

    def test_goal_reached_with_valid_angle_accepted(self) -> None:
        # (1, 1) → 45° relative to normal (0, 1) → valid
        self.assertTrue(self.env.is_goal(State(37, 10), (1, 1)))
        self.assertTrue(self.env.is_goal(State(37, 10), (-1, 1)))

    def test_goal_reached_with_invalid_angle_rejected(self) -> None:
        # (0, 1) → 0°, (1, 0) → 90° → both outside [30°, 60°]
        self.assertFalse(self.env.is_goal(State(37, 10), (0, 1)))
        self.assertFalse(self.env.is_goal(State(37, 10), (1, 0)))
        self.assertFalse(self.env.is_goal(State(37, 10), (0, -1)))

    def test_wrong_position_never_accepted(self) -> None:
        for action in ACTIONS:
            self.assertFalse(self.env.is_goal(State(36, 10), action))

    def test_angle_boundary_min(self) -> None:
        # Construct action at exactly 30°: a = (sin30, cos30) = (0.5, sqrt(3)/2)
        # Nearest discrete action: check boundary is inclusive
        # Use float precision: arccos of 30deg
        theta = self.env.approach_angle_deg((1, 1))
        self.assertGreaterEqual(theta, 30.0)
        self.assertLessEqual(theta, 60.0)

    def test_all_eight_actions_classified_correctly(self) -> None:
        """Verify each action is correctly accepted or rejected at the goal."""
        env = self.env
        goal = env.goal
        for action in ACTIONS:
            theta = env.approach_angle_deg(action)
            expected = 30.0 <= theta <= 60.0
            result = env.is_goal(goal, action)
            self.assertEqual(
                result,
                expected,
                msg=f"action={action}, theta={theta:.2f}°, expected is_goal={expected}",
            )


if __name__ == "__main__":
    unittest.main()
