"""Tests for Step 4: Dijkstra and Value Iteration (Dynamic Programming)."""

from __future__ import annotations

import math
import unittest

from algorithms.graph_search.dijkstra import PlanResult, dijkstra
from algorithms.graph_search.dynamic_programming import (
    ValueIterationResult,
    extract_path,
    value_iteration,
)
from core.cost.cost_function import CostFunction, EuclideanTimeCost, FlowEnergyCost
from core.dynamics.flow import ConstantFlow
from core.environment.environment import DockingConfig, RiverEnvironment
from core.environment.grid import Grid, State


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_env(
    nx: int = 10,
    ny: int = 10,
    start: State = State(0, 5),
    goal: State = State(9, 5),
    normal: tuple[float, float] = (0.0, 1.0),
    angle_min: float = 30.0,
    angle_max: float = 60.0,
    flow_vi: float = 0.0,
    flow_vj: float = 0.0,
) -> RiverEnvironment:
    grid = Grid(nx, ny)
    flow = ConstantFlow(vi=flow_vi, vj=flow_vj)
    docking = DockingConfig(normal=normal, angle_min_deg=angle_min, angle_max_deg=angle_max)
    return RiverEnvironment(grid, flow, start, goal, docking)


def _make_cost(alpha: float = 1.0, beta: float = 0.0, flow_vi: float = 0.0) -> CostFunction:
    flow = ConstantFlow(vi=flow_vi, vj=0.0)
    return CostFunction(
        alpha=alpha,
        beta=beta,
        time_cost=EuclideanTimeCost(),
        energy_cost=FlowEnergyCost(flow),
    )


# ---------------------------------------------------------------------------
# Small deterministic scenario
# ---------------------------------------------------------------------------
# 5×5 grid, start=(0,2), goal=(4,2).
# Normal = (0,1), angle in [30°,60°] → only actions (1,1) and (-1,1) are valid.
#
# With alpha=1, beta=0 (pure time cost) and no flow:
#   Optimal path goes diagonally NE then SE (or similar) so the last step
#   uses action (1,1) or (-1,1).
# ---------------------------------------------------------------------------

SMALL_NX, SMALL_NY = 5, 5
SMALL_START = State(0, 2)
SMALL_GOAL = State(4, 2)


def _small_env() -> RiverEnvironment:
    return _make_env(
        nx=SMALL_NX, ny=SMALL_NY,
        start=SMALL_START, goal=SMALL_GOAL,
    )


def _small_cost() -> CostFunction:
    return _make_cost(alpha=1.0, beta=0.0)


# ---------------------------------------------------------------------------
# PlanResult dataclass tests
# ---------------------------------------------------------------------------


class TestPlanResult(unittest.TestCase):
    def test_default_not_found(self) -> None:
        r = PlanResult()
        self.assertFalse(r.found)
        self.assertEqual(r.total_cost, float("inf"))
        self.assertEqual(r.path, [])
        self.assertEqual(r.actions, [])
        self.assertEqual(r.nodes_expanded, 0)


# ---------------------------------------------------------------------------
# Dijkstra – structural checks
# ---------------------------------------------------------------------------


class TestDijkstraStructure(unittest.TestCase):
    def setUp(self) -> None:
        self.env = _small_env()
        self.cost_fn = _small_cost()

    def test_found_is_true(self) -> None:
        result = dijkstra(self.env, self.cost_fn)
        self.assertTrue(result.found)

    def test_path_starts_at_start(self) -> None:
        result = dijkstra(self.env, self.cost_fn)
        self.assertEqual(result.path[0], SMALL_START)

    def test_path_ends_at_goal(self) -> None:
        result = dijkstra(self.env, self.cost_fn)
        self.assertEqual(result.path[-1], SMALL_GOAL)

    def test_actions_length_matches_path(self) -> None:
        result = dijkstra(self.env, self.cost_fn)
        self.assertEqual(len(result.actions), len(result.path) - 1)

    def test_path_is_connected(self) -> None:
        """Consecutive states must differ by exactly the recorded action."""
        result = dijkstra(self.env, self.cost_fn)
        for s, a, s_next in zip(result.path, result.actions, result.path[1:]):
            expected = self.env.transition(s, a)
            self.assertEqual(s_next, expected, msg=f"Broken transition at {s} via {a}")

    def test_last_action_valid_approach_angle(self) -> None:
        result = dijkstra(self.env, self.cost_fn)
        last_action = result.actions[-1]
        self.assertTrue(
            self.env.is_goal(SMALL_GOAL, last_action),
            msg=f"Last action {last_action} violates approach-angle constraint.",
        )

    def test_all_states_inside_grid(self) -> None:
        result = dijkstra(self.env, self.cost_fn)
        for s in result.path:
            self.assertTrue(self.env.grid.contains(s), msg=f"{s} outside grid")

    def test_total_cost_matches_sum(self) -> None:
        result = dijkstra(self.env, self.cost_fn)
        computed = sum(
            self.cost_fn(s, a) for s, a in zip(result.path, result.actions)
        )
        self.assertAlmostEqual(result.total_cost, computed, places=10)

    def test_nodes_expanded_positive(self) -> None:
        result = dijkstra(self.env, self.cost_fn)
        self.assertGreater(result.nodes_expanded, 0)


# ---------------------------------------------------------------------------
# Dijkstra – optimality
# ---------------------------------------------------------------------------


class TestDijkstraOptimality(unittest.TestCase):
    def test_cost_is_positive(self) -> None:
        env = _small_env()
        cost_fn = _small_cost()
        result = dijkstra(env, cost_fn)
        self.assertGreater(result.total_cost, 0.0)
        self.assertLess(result.total_cost, float("inf"))

    def test_known_minimum_on_tiny_grid(self) -> None:
        # 3×3 grid, start=(0,1), goal=(2,1).
        # Goal is valid only with a diagonal final move.
        # Shortest valid path: (0,1)→(1,0)→(2,1), cost=2√2
        env = _make_env(nx=3, ny=3, start=State(0, 1), goal=State(2, 1))
        cost_fn = _make_cost(alpha=1.0, beta=0.0)
        result = dijkstra(env, cost_fn)
        self.assertTrue(result.found)
        self.assertAlmostEqual(result.total_cost, 2 * math.sqrt(2), places=8)

    def test_no_path_when_goal_unreachable_via_valid_angle(self) -> None:
        # Start and goal adjacent, only straight east (1,0) is possible,
        # but angle constraint requires 30-60° from normal (0,1) – (1,0) = 90° invalid.
        # In a 2×1 grid the only reachable path is start→goal via (1,0), which is invalid.
        env = _make_env(
            nx=2, ny=1,
            start=State(0, 0), goal=State(1, 0),
            angle_min=30.0, angle_max=60.0,
        )
        cost_fn = _make_cost()
        result = dijkstra(env, cost_fn)
        self.assertFalse(result.found)

    def test_wider_angle_finds_path(self) -> None:
        # Widening the angle to [0°, 90°] makes every action valid at the goal.
        env = _make_env(
            nx=5, ny=5,
            start=State(0, 2), goal=State(4, 2),
            angle_min=0.0, angle_max=90.0,
        )
        cost_fn = _make_cost()
        result = dijkstra(env, cost_fn)
        self.assertTrue(result.found)


# ---------------------------------------------------------------------------
# ValueIterationResult – structural checks
# ---------------------------------------------------------------------------


class TestValueIterationStructure(unittest.TestCase):
    def setUp(self) -> None:
        self.env = _small_env()
        self.cost_fn = _small_cost()
        self.vi = value_iteration(self.env, self.cost_fn)

    def test_converged(self) -> None:
        self.assertTrue(self.vi.converged)

    def test_residual_below_tolerance(self) -> None:
        self.assertLess(self.vi.residual, 1e-6)

    def test_iterations_positive(self) -> None:
        self.assertGreater(self.vi.iterations, 0)

    def test_value_function_covers_all_states(self) -> None:
        for i in range(self.env.grid.nx):
            for j in range(self.env.grid.ny):
                self.assertIn(State(i, j), self.vi.value_function)

    def test_goal_value_is_zero(self) -> None:
        goal = self.env.goal
        self.assertAlmostEqual(self.vi.value_function[goal], 0.0, places=10)

    def test_value_function_nonnegative(self) -> None:
        for v in self.vi.value_function.values():
            self.assertGreaterEqual(v, 0.0)

    def test_policy_covers_non_goal_states(self) -> None:
        goal = self.env.goal
        for i in range(self.env.grid.nx):
            for j in range(self.env.grid.ny):
                s = State(i, j)
                if s != goal:
                    self.assertIn(s, self.vi.policy)

    def test_value_strictly_greater_than_zero_away_from_goal(self) -> None:
        """States far from the goal must have positive value."""
        v_start = self.vi.value_function[self.env.start]
        self.assertGreater(v_start, 0.0)


# ---------------------------------------------------------------------------
# Value Iteration – input validation
# ---------------------------------------------------------------------------


class TestValueIterationValidation(unittest.TestCase):
    def test_rejects_gamma_out_of_range(self) -> None:
        env = _small_env()
        cost_fn = _small_cost()
        with self.assertRaises(ValueError):
            value_iteration(env, cost_fn, gamma=0.0)
        with self.assertRaises(ValueError):
            value_iteration(env, cost_fn, gamma=1.1)

    def test_rejects_non_positive_tolerance(self) -> None:
        env = _small_env()
        cost_fn = _small_cost()
        with self.assertRaises(ValueError):
            value_iteration(env, cost_fn, tolerance=0.0)


# ---------------------------------------------------------------------------
# extract_path
# ---------------------------------------------------------------------------


class TestExtractPath(unittest.TestCase):
    def setUp(self) -> None:
        self.env = _small_env()
        self.cost_fn = _small_cost()
        self.vi = value_iteration(self.env, self.cost_fn)

    def test_path_starts_at_start(self) -> None:
        path, _ = extract_path(self.env, self.vi.policy)
        self.assertEqual(path[0], self.env.start)

    def test_path_ends_at_goal(self) -> None:
        path, _ = extract_path(self.env, self.vi.policy)
        self.assertEqual(path[-1], self.env.goal)

    def test_actions_length_matches_path(self) -> None:
        path, actions = extract_path(self.env, self.vi.policy)
        self.assertEqual(len(actions), len(path) - 1)

    def test_path_is_connected(self) -> None:
        path, actions = extract_path(self.env, self.vi.policy)
        for s, a, s_next in zip(path, actions, path[1:]):
            expected = self.env.transition(s, a)
            self.assertEqual(s_next, expected)

    def test_last_action_valid_approach_angle(self) -> None:
        path, actions = extract_path(self.env, self.vi.policy)
        self.assertTrue(self.env.is_goal(path[-1], actions[-1]))

    def test_all_states_inside_grid(self) -> None:
        path, _ = extract_path(self.env, self.vi.policy)
        for s in path:
            self.assertTrue(self.env.grid.contains(s))


# ---------------------------------------------------------------------------
# Consistency: Dijkstra vs Value Iteration
# ---------------------------------------------------------------------------


class TestDijkstraVsValueIteration(unittest.TestCase):
    """Both algorithms must agree on optimal cost (within numerical tolerance)."""

    def _run_both(
        self,
        nx: int,
        ny: int,
        start: State,
        goal: State,
        angle_min: float = 30.0,
        angle_max: float = 60.0,
    ) -> tuple[float, float]:
        env = _make_env(
            nx=nx, ny=ny,
            start=start, goal=goal,
            angle_min=angle_min, angle_max=angle_max,
        )
        cost_fn = _make_cost(alpha=1.0, beta=0.0)
        d_result = dijkstra(env, cost_fn)
        vi_result = value_iteration(env, cost_fn, gamma=1.0)
        _, vi_actions = extract_path(env, vi_result.policy)
        vi_cost = sum(cost_fn(s, a) for s, a in zip(
            extract_path(env, vi_result.policy)[0], vi_actions
        ))
        return d_result.total_cost, vi_cost

    def test_small_grid_agreement(self) -> None:
        d_cost, vi_cost = self._run_both(5, 5, State(0, 2), State(4, 2))
        self.assertAlmostEqual(d_cost, vi_cost, places=4)

    def test_asymmetric_grid_agreement(self) -> None:
        d_cost, vi_cost = self._run_both(8, 4, State(0, 2), State(7, 2))
        self.assertAlmostEqual(d_cost, vi_cost, places=4)

    def test_wider_angle_agreement(self) -> None:
        d_cost, vi_cost = self._run_both(
            5, 5, State(0, 2), State(4, 2),
            angle_min=0.0, angle_max=90.0,
        )
        self.assertAlmostEqual(d_cost, vi_cost, places=4)

    def test_from_project_config(self) -> None:
        """Agreement on the default project scenario (40×20 grid)."""
        from pathlib import Path
        from core.config_loader import load_all_configs

        env_cfg, algo_cfg, _ = load_all_configs(Path("configs"))
        env = RiverEnvironment.from_config(env_cfg)
        cost_fn = CostFunction.from_config(algo_cfg, env.flow)

        d_result = dijkstra(env, cost_fn)
        vi_result = ValueIterationResult.from_config(env, cost_fn, algo_cfg["dynamic_programming"])

        self.assertTrue(d_result.found)
        self.assertTrue(vi_result.converged)

        vi_path, vi_actions = extract_path(env, vi_result.policy)
        vi_cost = sum(cost_fn(s, a) for s, a in zip(vi_path, vi_actions))

        self.assertAlmostEqual(d_result.total_cost, vi_cost, places=2)


if __name__ == "__main__":
    unittest.main()
