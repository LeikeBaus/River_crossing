import unittest
from pathlib import Path

from experiments.evaluator import evaluate_results
from experiments.runner import run_all_experiments
from ui.visualization import launch_ui, parse_seed_options


class ScaffoldContractTests(unittest.TestCase):
    def test_expected_project_paths_exist(self) -> None:
        expected_paths = [
            Path("configs/env.yaml"),
            Path("configs/algorithm.yaml"),
            Path("configs/experiment.yaml"),
            Path("core/config_loader.py"),
            Path("experiments/runner.py"),
            Path("experiments/evaluator.py"),
            Path("ui/visualization.py"),
        ]

        for path in expected_paths:
            self.assertTrue(path.exists(), f"Missing required scaffold file: {path}")

    def test_experiment_entrypoints_exist(self) -> None:
        self.assertTrue(callable(run_all_experiments))
        self.assertTrue(callable(evaluate_results))

    def test_visualization_module_exposes_pyqt_entrypoints(self) -> None:
        self.assertTrue(callable(launch_ui))
        self.assertEqual(parse_seed_options([1, 3, 5]), ["1", "3", "5"])


if __name__ == "__main__":
    unittest.main()
