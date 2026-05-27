"""Tests that the sweep runner respects the results cache.

Key contract being verified
---------------------------
- A sweep point whose ``R-{id}.json`` already exists is *not* re-run;
  the cached file is loaded and the ``cached`` flag is ``True``.
- A sweep point whose file does not exist *is* executed (``run_all_experiments``
  is called) and the result is written to disk.
- After a full sweep the ``R-*.json`` / ``EV-*.json`` pair is present for
  every point.
"""

from __future__ import annotations

import json
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from experiments.experiment_id import build_experiment_id, results_paths
from experiments.sweep import (
    SweepPoint,
    count_sweep_points,
    load_sweep_config,
    run_sweep,
    sweep_points,
)


# ---------------------------------------------------------------------------
# Helpers – minimal configs for a tiny 2-point sweep
# ---------------------------------------------------------------------------

_MINIMAL_SWEEP_CFG: dict = {
    "seeds": [1],
    "algorithms": ["dijkstra"],
    "flow_types": ["constant"],
    "flow_directions": ["up"],
    "flow_strengths": [0.0, 1.0],   # 2 strength steps → 2 points
    "sigmas": [5.0],
    "floor": 0.1,
    "alpha": 1.0,
    "beta": 1.0,
    "inertias": [0],
    "turn_penalty": 1.0,
    "grid_nx": 40,
    "grid_ny": 20,
}

_MINIMAL_ALGO_CFG: dict = {
    "common": {
        "alpha": 1.0,
        "beta": 1.0,
        "inertia": 0,
        "turn_penalty": 1.0,
        "version": 0.1,
    },
    "dijkstra": {},
    "a_star": {"heuristic": "euclidean"},
    "weighted_a_star": {"heuristic": "euclidean", "weight": 1.5},
    "dynamic_programming": {"gamma": 1.0, "tolerance": 1e-6, "max_iterations": 100},
    "apf": {"k_att": 1.0, "lambda_flow": 0.0},
    "q_learning": {
        "learning_rate": 0.1,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.95,
        "goal_reward": 50.0,
        "progress_reward_scale": 2.5,
        "revisit_penalty": 0.3,
        "approach_bonus": 5.0,
        "max_steps_per_episode": 100,
        "min_success_rate_for_decay": 0.05,
        "success_window": 10,
        "max_snapshots": 10,
    },
}

_FAKE_RECORD = [
    {
        "environment_name": "env",
        "algorithm": "dijkstra",
        "seed": 1,
        "found": True,
        "total_cost": 1.0,
        "path_length": 1,
        "steps": 1,
        "angle_valid": True,
        "plan_time": 0.0,
        "training_time": 0.0,
        "inference_time": 0.0,
        "total_time": 0.0,
        "nodes_expanded": 1,
        "path": [],
        "actions": [],
        "best_path": [],
        "best_actions": [],
        "run_trace": [],
        "reward_history": [],
        "success_rate": None,
        "exploration_snapshots": [],
    }
]


def _exp_id_for_point(point: SweepPoint, algorithm: str | None = None) -> str:
    """Return the per-algorithm exp_id used by run_sweep for cache look-up."""
    effective_algo = point.build_algo_override(_MINIMAL_ALGO_CFG)
    algo = algorithm if algorithm is not None else point.algorithms[0]
    return build_experiment_id(
        flow_cfg=point.flow_cfg,
        algo_cfg=effective_algo,
        seeds=point.seeds,
        algorithm=algo,
        grid_nx=point.grid_nx,
        grid_ny=point.grid_ny,
        ql_episodes=point.ql_episodes,
    )


def _write_minimal_env_config(config_dir: Path) -> None:
    (config_dir / "env.yaml").write_text(
        textwrap.dedent("""\
            grid:
              nx: 40
              ny: 20
            flow:
              type: constant
              vector: [0.0, 1.0]
            start: [2, 5]
            goal: [37, 15]
            docking:
              normal: [0.0, 1.0]
              approach_angle_deg:
                min: 30.0
                max: 60.0
        """),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class SweepCacheTests(unittest.TestCase):
    """Verify that run_sweep skips points whose R-*.json already exists."""

    def _collect_sweep(
        self,
        sweep_cfg: dict,
        analysis_dir: Path,
        config_dir: Path,
    ) -> list[dict]:
        """Drain the sweep generator and return the list of progress dicts."""
        results = []
        gen = run_sweep(
            sweep_cfg=sweep_cfg,
            base_algo_cfg=_MINIMAL_ALGO_CFG,
            config_dir=config_dir,
            analysis_dir=analysis_dir,
        )
        try:
            while True:
                results.append(next(gen))
        except StopIteration:
            pass
        return results

    def test_all_points_cached_when_files_exist(self) -> None:
        """All points should be reported as cached when R-*.json files already exist."""
        with tempfile.TemporaryDirectory() as tmp:
            analysis_dir = Path(tmp) / "analysis"
            analysis_dir.mkdir(parents=True)
            config_dir = Path(tmp) / "configs"
            config_dir.mkdir(parents=True)
            _write_minimal_env_config(config_dir)

            # Pre-write a per-algorithm results file for every sweep point
            for point in sweep_points(_MINIMAL_SWEEP_CFG):
                for algorithm in point.algorithms:
                    raw_path, _ = results_paths(_exp_id_for_point(point, algorithm), analysis_dir)
                    raw_path.write_text(json.dumps(_FAKE_RECORD), encoding="utf-8")

            # Patch run_all_experiments so we can detect if it is called
            with patch(
                "experiments.sweep.run_all_experiments",
                side_effect=AssertionError("run_all_experiments must NOT be called for cached points"),
            ):
                progress_list = self._collect_sweep(
                    _MINIMAL_SWEEP_CFG, analysis_dir, config_dir
                )

            self.assertEqual(len(progress_list), 2)
            self.assertTrue(all(p["cached"] for p in progress_list))

    def test_non_cached_points_are_executed(self) -> None:
        """Points without an existing file must call run_all_experiments exactly once."""
        with tempfile.TemporaryDirectory() as tmp:
            analysis_dir = Path(tmp) / "analysis"
            analysis_dir.mkdir(parents=True)
            config_dir = Path(tmp) / "configs"
            config_dir.mkdir(parents=True)
            _write_minimal_env_config(config_dir)

            # No pre-existing files → all 2 points must be executed
            fake_record_obj = MagicMock()
            fake_record_obj.to_dict.return_value = _FAKE_RECORD[0]

            with patch(
                "experiments.sweep.run_all_experiments",
                return_value=[fake_record_obj],
            ) as mock_run, patch(
                "experiments.sweep.evaluate_results",
            ):
                progress_list = self._collect_sweep(
                    _MINIMAL_SWEEP_CFG, analysis_dir, config_dir
                )

            self.assertEqual(len(progress_list), 2)
            self.assertTrue(all(not p["cached"] for p in progress_list))
            self.assertEqual(mock_run.call_count, 2)

    def test_mixed_cache_calls_runner_only_for_missing(self) -> None:
        """Only the first point is pre-cached; run_all_experiments fires once for the second."""
        with tempfile.TemporaryDirectory() as tmp:
            analysis_dir = Path(tmp) / "analysis"
            analysis_dir.mkdir(parents=True)
            config_dir = Path(tmp) / "configs"
            config_dir.mkdir(parents=True)
            _write_minimal_env_config(config_dir)

            pts = list(sweep_points(_MINIMAL_SWEEP_CFG))
            self.assertEqual(len(pts), 2)

            # Cache only the first point (all its algorithms)
            for algorithm in pts[0].algorithms:
                first_raw, _ = results_paths(_exp_id_for_point(pts[0], algorithm), analysis_dir)
                first_raw.write_text(json.dumps(_FAKE_RECORD), encoding="utf-8")

            fake_record_obj = MagicMock()
            fake_record_obj.to_dict.return_value = _FAKE_RECORD[0]

            with patch(
                "experiments.sweep.run_all_experiments",
                return_value=[fake_record_obj],
            ) as mock_run, patch(
                "experiments.sweep.evaluate_results",
            ):
                progress_list = self._collect_sweep(
                    _MINIMAL_SWEEP_CFG, analysis_dir, config_dir
                )

            self.assertEqual(len(progress_list), 2)
            self.assertTrue(progress_list[0]["cached"])
            self.assertFalse(progress_list[1]["cached"])
            self.assertEqual(mock_run.call_count, 1)

    def test_result_files_written_for_new_points(self) -> None:
        """After a sweep, R-*.json must exist for every non-cached point.

        run_all_experiments writes the file when given output_path.  The mock
        replicates this side-effect so we can verify the files end up on disk.
        """
        with tempfile.TemporaryDirectory() as tmp:
            analysis_dir = Path(tmp) / "analysis"
            analysis_dir.mkdir(parents=True)
            config_dir = Path(tmp) / "configs"
            config_dir.mkdir(parents=True)
            _write_minimal_env_config(config_dir)

            fake_record_obj = MagicMock()
            fake_record_obj.to_dict.return_value = _FAKE_RECORD[0]

            def _fake_run_all(*, output_path=None, **_kwargs):
                # Simulate what the real runner does: write the JSON to output_path
                if output_path is not None:
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(output_path).write_text(
                        json.dumps([_FAKE_RECORD[0]]), encoding="utf-8"
                    )
                return [fake_record_obj]

            with patch(
                "experiments.sweep.run_all_experiments",
                side_effect=_fake_run_all,
            ), patch("experiments.sweep.evaluate_results"):
                self._collect_sweep(_MINIMAL_SWEEP_CFG, analysis_dir, config_dir)

            pts = list(sweep_points(_MINIMAL_SWEEP_CFG))
            for point in pts:
                for algorithm in point.algorithms:
                    raw_path, _ = results_paths(_exp_id_for_point(point, algorithm), analysis_dir)
                    self.assertTrue(
                        raw_path.exists(),
                        f"Expected results file not found: {raw_path.name}",
                    )

    def test_done_counter_increments_correctly(self) -> None:
        """The 'done' field must run 1 … total in order."""
        with tempfile.TemporaryDirectory() as tmp:
            analysis_dir = Path(tmp) / "analysis"
            analysis_dir.mkdir(parents=True)
            config_dir = Path(tmp) / "configs"
            config_dir.mkdir(parents=True)
            _write_minimal_env_config(config_dir)

            # Pre-cache all so no real work is done
            for point in sweep_points(_MINIMAL_SWEEP_CFG):
                for algorithm in point.algorithms:
                    raw_path, _ = results_paths(_exp_id_for_point(point, algorithm), analysis_dir)
                    raw_path.write_text(json.dumps(_FAKE_RECORD), encoding="utf-8")

            with patch("experiments.sweep.run_all_experiments"):
                progress_list = self._collect_sweep(
                    _MINIMAL_SWEEP_CFG, analysis_dir, config_dir
                )

            total = count_sweep_points(_MINIMAL_SWEEP_CFG)
            done_values = [p["done"] for p in progress_list]
            self.assertEqual(done_values, list(range(1, total + 1)))
            self.assertTrue(all(p["total"] == total for p in progress_list))


class SweepConfigLoadTests(unittest.TestCase):
    """load_sweep_config validation."""

    def test_loads_real_sweep_yaml(self) -> None:
        cfg = load_sweep_config("configs/sweep.yaml")
        self.assertIn("env_sizes", cfg)
        self.assertIn("inertias", cfg)
        self.assertIn("flow_strengths", cfg)

    def test_raises_on_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_sweep_config("configs/nonexistent_sweep.yaml")


if __name__ == "__main__":
    unittest.main()
