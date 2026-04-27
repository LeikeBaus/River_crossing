"""Tests for Step 5: A* and Weighted A*."""

from __future__ import annotations

import unittest

from algorithms.graph_search.dijkstra import dijkstra
from algorithms.graph_search.heuristic_search import (
    astar,
    astar_from_config,
    run_weighted_astar_sweep,
    weighted_astar,
    weighted_astar_from_config,
)
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


class TestAStarStructure(unittest.TestCase):
    def setUp(self) -> None:
        self.env = _make_env(nx=7, ny=5, start=State(0, 2), goal=State(6, 2))
        self.cost_fn = _make_cost(alpha=1.0, beta=0.0)

    def test_finds_path(self) -> None:
        result = astar(self.env, self.cost_fn)
        self.assertTrue(result.found)
        self.assertEqual(result.path[0], self.env.start)
        self.assertEqual(result.path[-1], self.env.goal)

    def test_last_action_respects_approach_angle(self) -> None:
        result = astar(self.env, self.cost_fn)
        self.assertTrue(self.env.is_goal(result.path[-1], result.actions[-1]))

    def test_cost_matches_path_sum(self) -> None:
        result = astar(self.env, self.cost_fn)
        computed = sum(self.cost_fn(s, a) for s, a in zip(result.path, result.actions))
        self.assertAlmostEqual(result.total_cost, computed, places=10)


class TestAStarOptimality(unittest.TestCase):
    def test_matches_dijkstra_cost(self) -> None:
        env = _make_env(nx=8, ny=5, start=State(0, 2), goal=State(7, 2))
        cost_fn = _make_cost(alpha=1.0, beta=0.0)

        d_res = dijkstra(env, cost_fn)
        a_res = astar(env, cost_fn)

        self.assertTrue(d_res.found)
        self.assertTrue(a_res.found)
        self.assertAlmostEqual(a_res.total_cost, d_res.total_cost, places=8)

    def test_weighted_with_w1_equals_astar(self) -> None:
        env = _make_env(nx=8, ny=5, start=State(0, 2), goal=State(7, 2))
        cost_fn = _make_cost(alpha=1.0, beta=0.0)

        a_res = astar(env, cost_fn)
        w_res = weighted_astar(env, cost_fn, weight=1.0)

        self.assertTrue(a_res.found)
        self.assertTrue(w_res.found)
        self.assertAlmostEqual(w_res.total_cost, a_res.total_cost, places=8)


class TestWeightedAStarBehavior(unittest.TestCase):
    def test_rejects_weight_below_one(self) -> None:
        env = _make_env()
        cost_fn = _make_cost()
        with self.assertRaises(ValueError):
            weighted_astar(env, cost_fn, weight=0.99)

    def test_suboptimality_bound_direction(self) -> None:
        """Weighted A* should not outperform the Dijkstra optimum."""
        env = _make_env(nx=12, ny=8, start=State(0, 4), goal=State(11, 4))
        cost_fn = _make_cost(alpha=1.0, beta=0.0)

        d_res = dijkstra(env, cost_fn)
        w_res = weighted_astar(env, cost_fn, weight=2.0)

        self.assertTrue(d_res.found)
        self.assertTrue(w_res.found)
        self.assertGreaterEqual(w_res.total_cost, d_res.total_cost)

    def test_no_path_case_matches_reference(self) -> None:
        env = _make_env(
            nx=2,
            ny=1,
            start=State(0, 0),
            goal=State(1, 0),
            angle_min=30.0,
            angle_max=60.0,
        )
        cost_fn = _make_cost()

        a_res = astar(env, cost_fn)
        w_res = weighted_astar(env, cost_fn, weight=2.0)

        self.assertFalse(a_res.found)
        self.assertFalse(w_res.found)


class TestConfigAndSweep(unittest.TestCase):
    def test_astar_from_config(self) -> None:
        from pathlib import Path

        from core.config_loader import load_all_configs

        env_cfg, algo_cfg, _ = load_all_configs(Path("configs"))
        env = RiverEnvironment.from_config(env_cfg)
        cost_fn = CostFunction.from_config(algo_cfg, env.flow)

        result = astar_from_config(env, cost_fn, algo_cfg["a_star"])
        self.assertTrue(result.found)

    def test_weighted_astar_from_config(self) -> None:
        from pathlib import Path

        from core.config_loader import load_all_configs

        env_cfg, algo_cfg, _ = load_all_configs(Path("configs"))
        env = RiverEnvironment.from_config(env_cfg)
        cost_fn = CostFunction.from_config(algo_cfg, env.flow)

        result = weighted_astar_from_config(env, cost_fn, algo_cfg["weighted_a_star"])
        self.assertTrue(result.found)

    def test_weight_sweep_returns_all_requested_keys(self) -> None:
        env = _make_env(nx=9, ny=6, start=State(0, 3), goal=State(8, 3))
        cost_fn = _make_cost(alpha=1.0, beta=0.0)

        weights = [1.0, 1.2, 1.5, 2.0]
        sweep = run_weighted_astar_sweep(env, cost_fn, weights)

        self.assertEqual(set(sweep.keys()), {1.0, 1.2, 1.5, 2.0})
        self.assertTrue(all(res.found for res in sweep.values()))

    def test_unknown_heuristic_raises(self) -> None:
        env = _make_env()
        cost_fn = _make_cost()

        with self.assertRaises(ValueError):
            astar(env, cost_fn, heuristic="manhattan")


if __name__ == "__main__":
    unittest.main()
