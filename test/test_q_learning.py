"""Tests for Step 7: Q-learning agent, training loop, and policy rollout."""

from __future__ import annotations

import unittest

from algorithms.graph_search.dijkstra import dijkstra
from algorithms.rl.q_learning import (
    QLearningResult,
    extract_policy,
    q_learning_from_config,
    q_learning_train,
    rollout_policy,
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
    angle_min: float = 0.0,
    angle_max: float = 90.0,
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


class TestQLearningValidation(unittest.TestCase):
    def test_invalid_hyperparameters_raise(self) -> None:
        env = _make_env()
        cost_fn = _make_cost()

        with self.assertRaises(ValueError):
            q_learning_train(env, cost_fn, episodes=0)
        with self.assertRaises(ValueError):
            q_learning_train(env, cost_fn, episodes=10, learning_rate=0.0)
        with self.assertRaises(ValueError):
            q_learning_train(env, cost_fn, episodes=10, gamma=1.1)
        with self.assertRaises(ValueError):
            q_learning_train(env, cost_fn, episodes=10, epsilon_start=0.1, epsilon_end=0.2)
        with self.assertRaises(ValueError):
            q_learning_train(env, cost_fn, episodes=10, epsilon_decay=0.0)


class TestQLearningTraining(unittest.TestCase):
    def test_result_structure(self) -> None:
        env = _make_env(nx=6, ny=4, start=State(0, 2), goal=State(5, 2))
        cost_fn = _make_cost(alpha=1.0, beta=0.0)

        result = q_learning_train(
            env,
            cost_fn,
            episodes=30,
            seed=123,
            max_steps_per_episode=80,
        )

        self.assertIsInstance(result, QLearningResult)
        self.assertEqual(len(result.reward_history), 30)
        self.assertEqual(len(result.success_history), 30)
        self.assertEqual(len(result.steps_history), 30)
        self.assertEqual(len(result.epsilon_history), 30)

    def test_epsilon_decays_monotonically(self) -> None:
        env = _make_env(nx=6, ny=4, start=State(0, 2), goal=State(5, 2))
        cost_fn = _make_cost()

        result = q_learning_train(
            env,
            cost_fn,
            episodes=20,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.8,
            seed=7,
        )

        eps = result.epsilon_history
        for a, b in zip(eps, eps[1:]):
            self.assertGreaterEqual(a, b)
        self.assertGreaterEqual(eps[-1], 0.1)

    def test_same_seed_reproducible_histories(self) -> None:
        env = _make_env(nx=6, ny=4, start=State(0, 2), goal=State(5, 2))
        cost_fn = _make_cost()

        r1 = q_learning_train(env, cost_fn, episodes=25, seed=42)
        r2 = q_learning_train(env, cost_fn, episodes=25, seed=42)

        self.assertEqual(r1.reward_history, r2.reward_history)
        self.assertEqual(r1.success_history, r2.success_history)
        self.assertEqual(r1.steps_history, r2.steps_history)

    def test_policy_rollout_returns_valid_path(self) -> None:
        env = _make_env(nx=6, ny=4, start=State(0, 2), goal=State(5, 2))
        cost_fn = _make_cost()

        result = q_learning_train(
            env,
            cost_fn,
            episodes=120,
            seed=13,
            max_steps_per_episode=120,
        )
        rollout = rollout_policy(env, cost_fn, result.policy, max_steps=120)

        self.assertEqual(rollout.path[0], env.start)
        self.assertTrue(all(env.grid.contains(s) for s in rollout.path))

    def test_unreachable_goal_yields_zero_success_rate(self) -> None:
        env = _make_env(
            nx=2,
            ny=1,
            start=State(0, 0),
            goal=State(1, 0),
            angle_min=30.0,
            angle_max=60.0,
        )
        cost_fn = _make_cost()

        result = q_learning_train(env, cost_fn, episodes=30, seed=5)
        self.assertEqual(result.success_rate, 0.0)


class TestPolicyHelpers(unittest.TestCase):
    def test_extract_policy_returns_actions_for_states(self) -> None:
        env = _make_env(nx=4, ny=3, start=State(0, 1), goal=State(3, 1))
        q_table = {
            State(0, 1): {(1, 0): 2.0, (1, 1): 1.0},
            State(1, 1): {(1, 0): 3.0},
        }

        policy = extract_policy(env, q_table)
        self.assertIn(State(0, 1), policy)
        self.assertEqual(policy[State(0, 1)], (1, 0))

    def test_rollout_stops_if_policy_missing_state(self) -> None:
        env = _make_env(nx=5, ny=3, start=State(0, 1), goal=State(4, 1))
        cost_fn = _make_cost()
        policy = {State(0, 1): (1, 0)}

        result = rollout_policy(env, cost_fn, policy, max_steps=20)
        self.assertFalse(result.found)
        self.assertGreaterEqual(len(result.path), 2)

    def test_rollout_breaks_out_of_simple_loops(self) -> None:
        env = _make_env(nx=5, ny=3, start=State(0, 1), goal=State(4, 1))
        cost_fn = _make_cost()
        policy = {
            State(0, 1): (1, 0),
            State(1, 1): (-1, 0),
        }

        result = rollout_policy(env, cost_fn, policy, max_steps=50)
        self.assertFalse(result.found)
        self.assertLess(len(result.path), 10)


class TestConfigIntegration(unittest.TestCase):
    def test_q_learning_from_config(self) -> None:
        from pathlib import Path

        from core.config_loader import load_all_configs

        env_cfg, algo_cfg, _ = load_all_configs(Path("configs"))
        env = RiverEnvironment.from_config(env_cfg)
        cost_fn = CostFunction.from_config(algo_cfg, env.flow)

        result = q_learning_from_config(
            env,
            cost_fn,
            algo_cfg["q_learning"],
            episodes=20,
            seed=99,
        )
        self.assertIsInstance(result, QLearningResult)

    def test_q_learning_reference_gap_non_negative(self) -> None:
        """Delta J = J_algo - J_optimal should be >= 0 when rollout reaches goal."""
        env = _make_env(nx=7, ny=5, start=State(0, 2), goal=State(6, 2))
        cost_fn = _make_cost(alpha=1.0, beta=0.0)

        ref = dijkstra(env, cost_fn)
        train_result = q_learning_train(
            env,
            cost_fn,
            episodes=180,
            seed=23,
            max_steps_per_episode=100,
        )
        rollout = rollout_policy(env, cost_fn, train_result.policy, max_steps=100)

        self.assertTrue(ref.found)
        if rollout.found:
            self.assertGreaterEqual(rollout.total_cost, ref.total_cost)


if __name__ == "__main__":
    unittest.main()
