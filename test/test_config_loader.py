import tempfile
import unittest
from pathlib import Path

from core.config_loader import ConfigError, load_all_configs, load_algorithm_config, load_env_config


class ConfigLoaderTests(unittest.TestCase):
    def test_load_all_project_configs(self) -> None:
        env_cfg, algo_cfg, exp_cfg = load_all_configs(Path("configs"))

        self.assertIn("grid", env_cfg)
        self.assertIn("weighted_a_star", algo_cfg)
        self.assertIn("algorithms", exp_cfg)

    def test_env_config_rejects_non_positive_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "env.yaml"
            config_path.write_text(
                """
                grid:
                  nx: 0
                  ny: 10
                flow:
                  type: constant
                  vector: [1.0, 0.0]
                start: [0, 0]
                goal: [1, 1]
                docking:
                  normal: [0.0, 1.0]
                  approach_angle_deg:
                    min: 30
                    max: 60
                """,
                encoding="utf-8",
            )

            with self.assertRaises(ConfigError):
                load_env_config(config_path)

    def test_algorithm_config_rejects_invalid_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "algorithm.yaml"
            config_path.write_text(
                """
                common:
                  alpha: 1.0
                  beta: 0.0
                dijkstra: {}
                a_star:
                  heuristic: euclidean
                weighted_a_star:
                  heuristic: euclidean
                  weight: 0.9
                dynamic_programming:
                  gamma: 1.0
                  tolerance: 1.0e-6
                  max_iterations: 100
                apf:
                  k_att: 1.0
                  lambda_flow: 0.2
                q_learning:
                  learning_rate: 0.1
                  gamma: 0.99
                  epsilon_start: 1.0
                  epsilon_end: 0.1
                  epsilon_decay: 0.99
                """,
                encoding="utf-8",
            )

            with self.assertRaises(ConfigError):
                load_algorithm_config(config_path)


if __name__ == "__main__":
    unittest.main()
