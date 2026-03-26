from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.evaluator import evaluate_results
from experiments.runner import load_experiment_results, run_all_experiments


def _write_minimal_configs(base: Path) -> Path:
    config_dir = base / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    (config_dir / "env.yaml").write_text(
        """
grid:
  nx: 6
  ny: 4

flow:
  type: constant
  vector: [0.0, 0.0]

start: [0, 2]
goal: [5, 2]

docking:
  normal: [0.0, 1.0]
  approach_angle_deg:
    min: 0.0
    max: 90.0
""",
        encoding="utf-8",
    )

    (config_dir / "algorithm.yaml").write_text(
        """
common:
  alpha: 1.0
  beta: 0.0

dijkstra: {}

a_star:
  heuristic: euclidean

weighted_a_star:
  heuristic: euclidean
  weight: 1.5

dynamic_programming:
  gamma: 1.0
  tolerance: 1.0e-6
  max_iterations: 1000

apf:
  k_att: 1.0
  lambda_flow: 0.0

q_learning:
  learning_rate: 0.1
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_decay: 0.95
""",
        encoding="utf-8",
    )

    (config_dir / "experiment.yaml").write_text(
        """
seeds: [1, 2]

environments:
  - name: baseline
    env_config: configs/env.yaml

algorithms:
  - dijkstra
  - a_star
  - q_learning

metrics:
  - total_time
  - path_length
  - plan_time
  - training_time
  - inference_time
  - angle_valid
  - delta_j

q_learning_episodes: 30
""",
        encoding="utf-8",
    )

    return config_dir


class ExperimentRunnerTests(unittest.TestCase):
    def test_run_all_experiments_returns_expected_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = _write_minimal_configs(Path(tmp))
            results = run_all_experiments(config_dir)

            self.assertEqual(len(results), 1 * 3 * 2)
            self.assertEqual({record.environment_name for record in results}, {"baseline"})
            self.assertEqual({record.algorithm for record in results}, {"dijkstra", "a_star", "q_learning"})

    def test_run_all_experiments_can_save_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config_dir = _write_minimal_configs(base)
            output_path = base / "analysis" / "raw_results.json"

            results = run_all_experiments(config_dir, output_path=output_path)

            self.assertTrue(output_path.exists())
            loaded = load_experiment_results(output_path)
            self.assertEqual(len(loaded), len(results))

    def test_records_contain_required_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = _write_minimal_configs(Path(tmp))
            results = run_all_experiments(config_dir)

            sample = results[0]
            self.assertIsInstance(sample.plan_time, float)
            self.assertIsInstance(sample.training_time, float)
            self.assertIsInstance(sample.inference_time, float)
            self.assertIsInstance(sample.total_time, float)
            self.assertIsInstance(sample.steps, int)
            self.assertIsInstance(sample.angle_valid, bool)


class ExperimentEvaluatorTests(unittest.TestCase):
    def test_evaluate_results_computes_delta_j(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = _write_minimal_configs(Path(tmp))
            results = run_all_experiments(config_dir)
            summary = evaluate_results(results)

            dijkstra_runs = [run for run in summary["runs"] if run["algorithm"] == "dijkstra"]
            self.assertTrue(all(run["delta_j"] == 0.0 for run in dijkstra_runs if run["total_cost"] is not None))

    def test_evaluate_results_aggregates_by_algorithm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = _write_minimal_configs(Path(tmp))
            results = run_all_experiments(config_dir)
            summary = evaluate_results(results)

            by_algorithm = summary["by_algorithm"]
            self.assertIn("dijkstra", by_algorithm)
            self.assertIn("a_star", by_algorithm)
            self.assertIn("q_learning", by_algorithm)
            self.assertEqual(by_algorithm["dijkstra"]["runs"], 2)

    def test_evaluate_results_can_save_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config_dir = _write_minimal_configs(base)
            results = run_all_experiments(config_dir)
            output_path = base / "analysis" / "evaluation_summary.json"

            summary = evaluate_results(results, output_path=output_path)

            self.assertTrue(output_path.exists())
            loaded = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["reference_algorithm"], summary["reference_algorithm"])


if __name__ == "__main__":
    unittest.main()