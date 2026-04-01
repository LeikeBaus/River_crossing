from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from main import main


def _write_env_yaml(path: Path) -> None:
    path.write_text(
        """
grid:
  nx: 6
  ny: 4

flow:
  type: constant
  vector: [1.0, 0.0]

start: [0, 1]
goal: [5, 2]

docking:
  normal: [0.0, 1.0]
  approach_angle_deg:
    min: 30.0
    max: 60.0
""",
        encoding="utf-8",
    )


def _write_raw_results_json(path: Path) -> None:
    path.write_text(
        """
[
  {
    "environment_name": "baseline",
    "algorithm": "dijkstra",
    "seed": 1,
    "found": true,
    "total_cost": 12.3,
    "path_length": 4,
    "steps": 3,
    "angle_valid": true,
    "plan_time": 0.01,
    "training_time": 0.0,
    "inference_time": 0.0,
    "total_time": 0.01,
    "nodes_expanded": 42,
    "path": [[0, 1], [1, 1], [3, 2], [5, 2]],
    "actions": [[1, 0], [2, 1], [2, 0]],
    "reward_history": [],
    "success_rate": null
  },
  {
    "environment_name": "baseline",
    "algorithm": "a_star",
    "seed": 1,
    "found": true,
    "total_cost": 12.9,
    "path_length": 4,
    "steps": 3,
    "angle_valid": true,
    "plan_time": 0.005,
    "training_time": 0.0,
    "inference_time": 0.0,
    "total_time": 0.005,
    "nodes_expanded": 21,
    "path": [[0, 1], [2, 1], [4, 2], [5, 2]],
    "actions": [[2, 0], [2, 1], [1, 0]],
    "reward_history": [],
    "success_rate": null
  }
]
""",
        encoding="utf-8",
    )


class MainCliTests(unittest.TestCase):
  def test_check_command_returns_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            env_path = base / "env.yaml"
            results_path = base / "raw_results.json"
            _write_env_yaml(env_path)
            _write_raw_results_json(results_path)

            argv = [
                "main.py",
        "check",
            ]

            with patch("sys.argv", argv):
                code = main()

            self.assertEqual(code, 0)

  def test_ui_command_invokes_launcher(self) -> None:
    argv = ["main.py", "ui"]
    with patch("sys.argv", argv), patch("main.launch_ui", return_value=0) as launch:
      code = main()

    self.assertEqual(code, 0)
    launch.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
