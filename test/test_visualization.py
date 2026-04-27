from __future__ import annotations

import unittest

from ui.visualization import (
    UISelectionState,
    angle_strength_to_vector,
    build_action_overlay,
    build_render_run_data,
    coerce_path,
    compute_delta_j,
    find_single_run,
    format_metrics,
    max_frame_count,
    parse_seed_options,
    vector_to_angle_strength,
    visible_path_for_frame,
)
from core.environment.environment import DockingConfig, RiverEnvironment
from core.dynamics.flow import ConstantFlow
from core.environment.grid import Grid, State


class VisualizationTests(unittest.TestCase):
    def test_flow_vector_round_trip_is_consistent(self) -> None:
        angle, strength = vector_to_angle_strength(1.0, 1.0)
        vx, vy = angle_strength_to_vector(angle, strength)

        self.assertAlmostEqual(vx, 1.0, places=4)
        self.assertAlmostEqual(vy, 1.0, places=4)

    def test_build_action_overlay_marks_best_valid_and_invalid(self) -> None:
        env = RiverEnvironment(
            grid=Grid(5, 5),
            flow=ConstantFlow(vi=0.0, vj=0.0),
            start=State(0, 2),
            goal=State(4, 2),
            docking=DockingConfig(normal=(0.0, 1.0), angle_min_deg=30.0, angle_max_deg=60.0),
        )

        overlay = build_action_overlay(env, State(0, 2), best_action=(1, -1))

        self.assertIn((1, -1), overlay["best"])
        self.assertIn((1, 0), overlay["valid"])
        self.assertIn((-1, 0), overlay["invalid"])

    def test_parse_seed_options_uses_defaults_for_empty(self) -> None:
        self.assertEqual(parse_seed_options([]), ["1"])

    def test_parse_seed_options_formats_items(self) -> None:
        self.assertEqual(parse_seed_options([1, 2, 10]), ["1", "2", "10"])

    def test_coerce_path_filters_invalid_entries(self) -> None:
        self.assertEqual(coerce_path([[0, 1], [2, 3], [1], "x"]), [(0, 1), (2, 3)])

    def test_visible_path_for_frame_truncates_path(self) -> None:
        path = [(0, 0), (1, 0), (1, 1)]
        self.assertEqual(visible_path_for_frame(path, 0), [])
        self.assertEqual(visible_path_for_frame(path, 2), [(0, 0), (1, 0)])
        self.assertEqual(visible_path_for_frame(path, 99), path)

    def test_ui_selection_state_values(self) -> None:
        state = UISelectionState(
            environment="baseline",
            algorithm="dijkstra",
            seed=1,
            show_flow=True,
            show_labels=False,
            speed=1.5,
        )
        self.assertEqual(state.environment, "baseline")
        self.assertEqual(state.algorithm, "dijkstra")
        self.assertEqual(state.seed, 1)
        self.assertTrue(state.show_flow)
        self.assertFalse(state.show_labels)
        self.assertEqual(state.speed, 1.5)

    def test_find_single_run_by_selection(self) -> None:
        records = [
            {"environment_name": "baseline", "algorithm": "dijkstra", "seed": 1},
            {"environment_name": "baseline", "algorithm": "a_star", "seed": 1},
        ]
        selection = UISelectionState(
            environment="baseline",
            algorithm="a_star",
            seed=1,
            show_flow=True,
            show_labels=True,
            speed=1.0,
        )
        found = find_single_run(records, selection)
        self.assertIsNotNone(found)
        self.assertEqual(found["algorithm"], "a_star")

    def test_build_render_run_data_and_max_frame_count(self) -> None:
        runs = build_render_run_data(
            [
                {"algorithm": "dijkstra", "environment_name": "baseline", "seed": 1, "path": [[0, 0], [1, 0]]},
                {"algorithm": "a_star", "environment_name": "baseline", "seed": 1, "path": [[0, 0], [1, 1], [2, 2]]},
            ]
        )
        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[0].path, [(0, 0), (1, 0)])
        self.assertEqual(max_frame_count(runs), 3)

    def test_compute_delta_j_against_dijkstra_reference(self) -> None:
        records = [
            {"environment_name": "baseline", "algorithm": "dijkstra", "seed": 2, "total_cost": 10.0},
            {"environment_name": "baseline", "algorithm": "a_star", "seed": 2, "total_cost": 12.5},
        ]
        delta = compute_delta_j(records[1], records)
        self.assertEqual(delta, 2.5)

    def test_format_metrics_includes_seconds_and_flags(self) -> None:
        records = [
            {"environment_name": "baseline", "algorithm": "dijkstra", "seed": 1, "total_cost": 5.0},
        ]
        run = {
            "environment_name": "baseline",
            "algorithm": "dijkstra",
            "seed": 1,
            "total_cost": 5.0,
            "steps": 6,
            "plan_time": 0.015,
            "training_time": 0.0,
            "inference_time": 0.0,
            "angle_valid": True,
        }
        metrics = format_metrics(run, records)
        self.assertEqual(metrics["total_cost"], "5.0000")
        self.assertEqual(metrics["delta_j"], "0.0000")
        self.assertEqual(metrics["steps"], "6")
        self.assertEqual(metrics["angle_valid"], "yes")


if __name__ == "__main__":
    unittest.main()
