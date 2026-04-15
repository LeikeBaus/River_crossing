"""Tests for Step 6: Artificial Potential Field (APF)."""

from __future__ import annotations

import unittest

from algorithms.graph_search.apf import (
    APFResult,
    apf_from_config,
    apf_plan,
    attractive_potential,
    flow_potential,
    potential_gradient,
    select_apf_action,
    total_potential,
)
from algorithms.graph_search.dijkstra import dijkstra
from core.cost.cost_function import CostFunction, EuclideanTimeCost, FlowEnergyCost
from core.dynamics.flow import ConstantFlow
from core.environment.environment import DockingConfig, RiverEnvironment
from core.environment.grid import Grid, State


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


class TestPotentials(unittest.TestCase):
    def test_attractive_potential_zero_at_goal(self) -> None:
        goal = State(4, 2)
        self.assertEqual(attractive_potential(goal, goal, k_att=1.0), 0.0)

    def test_attractive_potential_increases_with_distance(self) -> None:
        goal = State(4, 2)
        near = attractive_potential(State(3, 2), goal, k_att=1.0)
        far = attractive_potential(State(0, 2), goal, k_att=1.0)
        self.assertLess(near, far)

    def test_flow_potential_reflects_flow_alignment(self) -> None:
        env = _make_env(flow_vi=1.0, flow_vj=0.0)
        self.assertLess(flow_potential(env, State(5, 0), lambda_flow=0.2), 0.0)

    def test_total_potential_combines_both_terms(self) -> None:
        env = _make_env(goal=State(4, 2), flow_vi=1.0)
        state = State(2, 2)
        total = total_potential(env, state, k_att=1.0, lambda_flow=0.2)
        expected = attractive_potential(state, env.goal, 1.0) + flow_potential(env, state, 0.2)
        self.assertAlmostEqual(total, expected)

    def test_gradient_points_toward_goal_without_flow(self) -> None:
        env = _make_env(start=State(0, 2), goal=State(4, 2))
        grad_i, grad_j = potential_gradient(env, State(0, 2), k_att=1.0, lambda_flow=0.0)
        self.assertLess(grad_i, 0.0)
        self.assertEqual(grad_j, 0.0)


class TestActionSelection(unittest.TestCase):
    def test_selects_east_when_goal_is_directly_east(self) -> None:
        env = _make_env(start=State(0, 2), goal=State(4, 2), angle_min=0.0, angle_max=90.0)
        action = select_apf_action(env, State(0, 2), k_att=1.0, lambda_flow=0.0)
        self.assertEqual(action, (1, 0))

    def test_excludes_invalid_goal_arrivals(self) -> None:
        env = _make_env(nx=2, ny=1, start=State(0, 0), goal=State(1, 0), angle_min=30.0, angle_max=60.0)
        action = select_apf_action(env, State(0, 0), k_att=1.0, lambda_flow=0.0)
        self.assertIsNone(action)


class TestAPFPlanner(unittest.TestCase):
    def test_result_type(self) -> None:
        result = APFResult()
        self.assertFalse(result.found)
        self.assertEqual(result.termination_reason, "local_minimum")

    def test_finds_path_in_simple_case(self) -> None:
        env = _make_env(nx=7, ny=5, start=State(0, 2), goal=State(6, 2), angle_min=0.0, angle_max=90.0)
        cost_fn = _make_cost(alpha=1.0, beta=0.0)
        result = apf_plan(env, cost_fn, k_att=1.0, lambda_flow=0.0)

        self.assertTrue(result.found)
        self.assertEqual(result.path[0], env.start)
        self.assertEqual(result.path[-1], env.goal)
        self.assertEqual(result.termination_reason, "goal")

    def test_last_action_respects_goal_angle(self) -> None:
        env = _make_env(nx=5, ny=5, start=State(0, 0), goal=State(4, 4), angle_min=30.0, angle_max=60.0)
        cost_fn = _make_cost(alpha=1.0, beta=0.0)
        result = apf_plan(env, cost_fn, k_att=1.0, lambda_flow=0.0)

        self.assertTrue(result.found)
        self.assertTrue(env.is_goal(result.path[-1], result.actions[-1]))

    def test_finds_path_with_docking_constraint_in_baseline_case(self) -> None:
        env = _make_env(
            nx=40,
            ny=20,
            start=State(2, 10),
            goal=State(37, 10),
            angle_min=30.0,
            angle_max=60.0,
            flow_vi=1.0,
            flow_vj=0.0,
        )
        cost_fn = _make_cost(alpha=1.0, beta=0.0, flow_vi=1.0)
        result = apf_plan(env, cost_fn, k_att=1.0, lambda_flow=0.2)

        self.assertTrue(result.found)
        self.assertEqual(result.path[-1], env.goal)
        self.assertTrue(env.is_goal(result.path[-1], result.actions[-1]))

    def test_total_cost_matches_path_sum(self) -> None:
        env = _make_env(nx=7, ny=5, start=State(0, 2), goal=State(6, 2), angle_min=0.0, angle_max=90.0)
        cost_fn = _make_cost(alpha=1.0, beta=0.0)
        result = apf_plan(env, cost_fn, k_att=1.0, lambda_flow=0.0)

        computed = sum(cost_fn(s, a) for s, a in zip(result.path, result.actions))
        self.assertAlmostEqual(result.total_cost, computed, places=10)

    def test_detects_local_minimum_when_no_valid_goal_arrival_exists(self) -> None:
        env = _make_env(nx=2, ny=1, start=State(0, 0), goal=State(1, 0), angle_min=30.0, angle_max=60.0)
        cost_fn = _make_cost(alpha=1.0, beta=0.0)
        result = apf_plan(env, cost_fn, k_att=1.0, lambda_flow=0.0)

        self.assertFalse(result.found)
        self.assertTrue(result.local_minimum_detected)
        self.assertEqual(result.termination_reason, "local_minimum")

    def test_rejects_invalid_parameters(self) -> None:
        env = _make_env()
        cost_fn = _make_cost()

        with self.assertRaises(ValueError):
            apf_plan(env, cost_fn, k_att=0.0)
        with self.assertRaises(ValueError):
            apf_plan(env, cost_fn, lambda_flow=-0.1)
        with self.assertRaises(ValueError):
            apf_plan(env, cost_fn, max_steps=0)

    def test_cost_not_better_than_dijkstra_reference(self) -> None:
        env = _make_env(nx=8, ny=5, start=State(0, 2), goal=State(7, 2), angle_min=0.0, angle_max=90.0)
        cost_fn = _make_cost(alpha=1.0, beta=0.0)

        reference = dijkstra(env, cost_fn)
        result = apf_plan(env, cost_fn, k_att=1.0, lambda_flow=0.0)

        self.assertTrue(reference.found)
        self.assertTrue(result.found)
        self.assertGreaterEqual(result.total_cost, reference.total_cost)


class TestAPFConfig(unittest.TestCase):
    def test_apf_from_config(self) -> None:
        from pathlib import Path

        from core.config_loader import load_all_configs

        env_cfg, algo_cfg, _ = load_all_configs(Path("configs"))
        env = RiverEnvironment.from_config(env_cfg)
        cost_fn = CostFunction.from_config(algo_cfg, env.flow)

        result = apf_from_config(env, cost_fn, algo_cfg["apf"])
        self.assertIsInstance(result, APFResult)


if __name__ == "__main__":
    unittest.main()