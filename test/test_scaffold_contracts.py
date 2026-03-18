import unittest
from pathlib import Path

from experiments.evaluator import evaluate_results
from experiments.runner import run_all_experiments
from ui.visualization import render


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

    def test_placeholders_raise_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            run_all_experiments()

        with self.assertRaises(NotImplementedError):
            evaluate_results()

        with self.assertRaises(NotImplementedError):
            render()


if __name__ == "__main__":
    unittest.main()
