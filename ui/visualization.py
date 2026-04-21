from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, QPointF, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QBrush, QColor, QFont, QPainter, QPolygonF, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDial,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSlider,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from core.config_loader import load_algorithm_config, load_env_config, load_experiment_config
from core.cost.cost_function import CostFunction
from core.environment.actions import ACTIONS, Action
from core.environment.environment import RiverEnvironment
from core.environment.grid import State
from experiments.evaluator import evaluate_results
from experiments.runner import load_experiment_results, run_all_experiments, run_single_experiment


@dataclass(frozen=True)
class UISelectionState:
    environment: str
    algorithm: str
    seed: int
    show_flow: bool
    show_labels: bool
    speed: float
    flow_vector: tuple[float, float] = (1.0, 0.0)
    impact: float = 0.0


@dataclass(frozen=True)
class RenderRunData:
    title: str
    run: dict[str, Any]
    path: list[tuple[int, int]]
    actions: list[tuple[int, int]]
    best_path: list[tuple[int, int]] = field(default_factory=list)
    exploration_snapshots: list[list[tuple[int, int]]] = field(default_factory=list)


def parse_seed_options(values: list[int] | None) -> list[str]:
    if not values:
        return ["1"]
    return [str(value) for value in values]


def coerce_path(raw_path: Any) -> list[tuple[int, int]]:
    if not isinstance(raw_path, list):
        return []

    result: list[tuple[int, int]] = []
    for item in raw_path:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            result.append((int(item[0]), int(item[1])))
    return result


def coerce_actions(raw_actions: Any) -> list[tuple[int, int]]:
    if not isinstance(raw_actions, list):
        return []

    result: list[tuple[int, int]] = []
    for item in raw_actions:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            result.append((int(item[0]), int(item[1])))
    return result


def angle_strength_to_vector(angle_deg: float, strength: float) -> tuple[float, float]:
    radians = math.radians(float(angle_deg))
    magnitude = max(0.0, float(strength))
    return (magnitude * math.cos(radians), magnitude * math.sin(radians))


def vector_to_angle_strength(vi: float, vj: float) -> tuple[float, float]:
    strength = math.hypot(float(vi), float(vj))
    angle = math.degrees(math.atan2(float(vj), float(vi)))
    if angle < 0.0:
        angle += 360.0
    return (angle, strength)


def dial_to_math_angle(dial_deg: float) -> float:
    """Map dial value to vector angle so dial "up" points north.

    QDial increases clockwise; math angles increase counterclockwise.
    """
    return (270.0 - float(dial_deg)) % 360.0


def math_to_dial_angle(angle_deg: float) -> float:
    """Inverse mapping for setting the dial from a vector."""
    return (270.0 - float(angle_deg)) % 360.0


def build_action_overlay(
    env: RiverEnvironment,
    state: State,
    best_action: Action | None,
) -> dict[str, list[Action]]:
    valid_actions = set(env.valid_actions(state))
    overlay: dict[str, list[Action]] = {"best": [], "valid": [], "invalid": []}

    for action in ACTIONS:
        next_is_valid_goal = True
        if action in valid_actions:
            next_state = env.transition(state, action)
            if next_state == env.goal and not env.is_goal(next_state, action):
                next_is_valid_goal = False

        if action == best_action and action in valid_actions and next_is_valid_goal:
            overlay["best"].append(action)
        elif action in valid_actions and next_is_valid_goal:
            overlay["valid"].append(action)
        else:
            overlay["invalid"].append(action)

    return overlay


def visible_path_for_frame(path: list[tuple[int, int]], frame_index: int) -> list[tuple[int, int]]:
    if frame_index <= 0:
        return []
    return path[: min(frame_index, len(path))]


def max_frame_count(runs: list[RenderRunData]) -> int:
    if not runs:
        return 0
    return max(len(run.path) for run in runs)


def find_single_run(records: list[dict[str, Any]], selection: UISelectionState) -> dict[str, Any] | None:
    for record in records:
        if (
            str(record.get("environment_name")) == selection.environment
            and str(record.get("algorithm")) == selection.algorithm
            and int(record.get("seed", -1)) == selection.seed
        ):
            return record
    return None


def find_comparison_runs(records: list[dict[str, Any]], selection: UISelectionState) -> list[dict[str, Any]]:
    runs = [
        record
        for record in records
        if str(record.get("environment_name")) == selection.environment
        and int(record.get("seed", -1)) == selection.seed
    ]
    return sorted(runs, key=lambda item: str(item.get("algorithm", "")))


def compute_delta_j(
    run: dict[str, Any],
    records: list[dict[str, Any]],
    reference_algorithm: str = "dijkstra",
) -> float | None:
    run_cost = run.get("total_cost")
    if run_cost is None:
        return None

    env_name = str(run.get("environment_name"))
    seed = int(run.get("seed", -1))
    for item in records:
        if (
            str(item.get("environment_name")) == env_name
            and int(item.get("seed", -2)) == seed
            and str(item.get("algorithm")) == reference_algorithm
            and item.get("total_cost") is not None
        ):
            return float(run_cost) - float(item["total_cost"])
    return None


def format_metrics(run: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, str]:
    delta_j = compute_delta_j(run, records)
    return {
        "total_cost": _format_number(run.get("total_cost")),
        "delta_j": _format_number(delta_j),
        "steps": str(run.get("steps", "-")),
        "plan_time": _format_seconds(run.get("plan_time")),
        "training_time": _format_seconds(run.get("training_time")),
        "inference_time": _format_seconds(run.get("inference_time")),
        "angle_valid": "yes" if bool(run.get("angle_valid", False)) else "no",
    }


def build_render_run_data(runs: list[dict[str, Any]]) -> list[RenderRunData]:
    result = []
    for run in runs:
        best_path = coerce_path(run.get("best_path", run.get("path", [])))
        actions = coerce_actions(run.get("best_actions", run.get("actions", [])))
        raw_snaps = run.get("exploration_snapshots", [])
        exploration_snapshots: list[list[tuple[int, int]]] = [
            coerce_path(snap) for snap in raw_snaps
        ]
        # Run tab path: last exploration snapshot (most complete) or best path.
        run_path = exploration_snapshots[-1] if exploration_snapshots else best_path
        result.append(
            RenderRunData(
                title=(
                    f"{run.get('algorithm')} | env={run.get('environment_name')} | "
                    f"seed={run.get('seed')}"
                ),
                run=run,
                path=run_path,
                actions=actions,
                best_path=best_path,
                exploration_snapshots=exploration_snapshots,
            )
        )
    return result


def _format_number(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _format_seconds(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}s"
    except (TypeError, ValueError):
        return str(value)


class SimulationWorker(QObject):
    progress = pyqtSignal(int)
    message = pyqtSignal(str)
    result = pyqtSignal(dict)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(
        self,
        mode: str,
        records: list[dict[str, Any]],
        selection: UISelectionState,
        config_dir: Path,
        results_path: Path,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.records = records
        self.selection = selection
        self.config_dir = config_dir
        self.results_path = results_path

    def run(self) -> None:
        try:
            self.message.emit(f"Worker started in mode '{self.mode}'.")
            self.progress.emit(15)

            if self.mode == "run_single_experiment":
                self.message.emit("Running selected experiment for all configured seeds...")
                exp_cfg = load_experiment_config(self.config_dir / "experiment.yaml")
                seeds = [int(s) for s in exp_cfg.get("seeds", [self.selection.seed])]
                runs: list[dict] = []
                for idx, seed in enumerate(seeds):
                    self.message.emit(f"Running seed {seed} ({idx + 1}/{len(seeds)})...")
                    record = run_single_experiment(
                        environment_name=self.selection.environment,
                        algorithm=self.selection.algorithm,
                        seed=seed,
                        config_dir=self.config_dir,
                        flow_vector_override=self.selection.flow_vector,
                        impact_override=self.selection.impact,
                    )
                    runs.append(record.to_dict())
                    self.progress.emit(int(15 + 85 * (idx + 1) / len(seeds)))
                self.result.emit({
                    "mode": "run_single_experiment",
                    "runs": runs,
                    "selected_seed": self.selection.seed,
                })
                return

            if self.mode == "run_batch_experiment":
                self.message.emit("Running experiment batch and evaluation...")
                records = run_all_experiments(
                    config_dir=self.config_dir,
                    output_path=self.results_path,
                    flow_vector_override=self.selection.flow_vector,
                    impact_override=self.selection.impact,
                )
                summary_path = self.results_path.parent / "evaluation_summary.json"
                evaluate_results(records, output_path=summary_path)
                self.progress.emit(100)
                self.result.emit(
                    {
                        "mode": "run_batch_experiment",
                        "records": [record.to_dict() for record in records],
                        "results_path": str(self.results_path),
                        "summary_path": str(summary_path),
                    }
                )
                return

            if self.mode == "show_results":
                run = find_single_run(self.records, self.selection)
                if run is None:
                    raise ValueError(
                        "No run found for current selection. Load or generate matching results first."
                    )
                self.progress.emit(100)
                self.result.emit({"mode": "show_results", "run": run})
                return

            if self.mode == "compare":
                runs = find_comparison_runs(self.records, self.selection)
                if not runs:
                    raise ValueError(
                        "No runs found for compare selection. Load or generate matching results first."
                    )
                self.progress.emit(100)
                self.result.emit({"mode": "compare", "runs": runs})
                return

            self.progress.emit(100)
            self.result.emit({"mode": self.mode, "status": "not_implemented"})
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))
        finally:
            self.finished.emit()


class GridCanvas(QGraphicsView):
    def __init__(self, title: str, nx: int = 40, ny: int = 20, minimum_width: int = 540) -> None:
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._title = title
        self._nx = nx
        self._ny = ny
        self._cell_size = 18.0
        self.setMinimumSize(minimum_width, 320)
        self._draw_placeholder()

    def _draw_placeholder(self) -> None:
        self.draw_run(
            start=(2, min(10, self._ny - 1)),
            goal=(min(37, self._nx - 1), min(10, self._ny - 1)),
            path=[],
            flow_vector=(1.0, 0.0),
            show_flow=True,
            show_labels=False,
            title_suffix="placeholder",
        )

    def set_grid_shape(self, nx: int, ny: int) -> None:
        self._nx = max(1, nx)
        self._ny = max(1, ny)

    def draw_run(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        path: list[tuple[int, int]],
        flow_vector: tuple[float, float],
        show_flow: bool,
        show_labels: bool,
        title_suffix: str,
        overlay_state: tuple[int, int] | None = None,
        action_overlay: dict[str, list[Action]] | None = None,
    ) -> None:
        self._scene.clear()
        border_pen = QPen(QColor("#D0D7DE"))
        border_pen.setWidth(1)

        for i in range(self._nx):
            for j in range(self._ny):
                x, y = self._cell_origin(i, j)
                self._scene.addRect(
                    x,
                    y,
                    self._cell_size,
                    self._cell_size,
                    border_pen,
                    QBrush(self._base_tile_color(i, j, start, goal)),
                )

        if show_flow:
            self._draw_flow(flow_vector, start, goal)

        if overlay_state is not None and action_overlay is not None:
            self._draw_action_overlay(overlay_state, action_overlay)

        self._draw_path(path, show_labels)
        self._draw_marker(start, QColor("#1F77B4"), "S")
        self._draw_marker(goal, QColor("#2CA02C"), "G")
        self._draw_active_position(path, start)

        text = self._scene.addText(f"{self._title} ({title_suffix})")
        text.setDefaultTextColor(QColor("#24292F"))
        text.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        text.setPos(8, 8)

        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.fitInView(self._scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _cell_origin(self, i: int, j: int) -> tuple[float, float]:
        x = i * self._cell_size
        y = (self._ny - 1 - j) * self._cell_size
        return x, y

    def _cell_center(self, i: int, j: int) -> tuple[float, float]:
        x, y = self._cell_origin(i, j)
        return x + self._cell_size / 2.0, y + self._cell_size / 2.0

    def _draw_marker(self, cell: tuple[int, int], color: QColor, label: str) -> None:
        i, j = cell
        if not (0 <= i < self._nx and 0 <= j < self._ny):
            return
        x, y = self._cell_origin(i, j)
        self._scene.addEllipse(x + 2, y + 2, self._cell_size - 4, self._cell_size - 4, QPen(), QBrush(color))
        text = self._scene.addText(label)
        text.setDefaultTextColor(QColor("#FFFFFF"))
        text.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        text.setPos(x + self._cell_size * 0.28, y + self._cell_size * 0.12)

    def _fill_cell(self, cell: tuple[int, int], color: QColor) -> None:
        i, j = cell
        if not (0 <= i < self._nx and 0 <= j < self._ny):
            return
        x, y = self._cell_origin(i, j)
        fill = QColor(color)
        fill.setAlpha(160)
        pen = QPen(fill.darker(120))
        pen.setCosmetic(True)
        self._scene.addRect(x + 1, y + 1, self._cell_size - 2, self._cell_size - 2, pen, QBrush(fill))

    def _base_tile_color(
        self,
        i: int,
        j: int,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> QColor:
        if (i, j) == start or (i, j) == goal:
            return QColor("#E2E8F0")
        if i < start[0] or i > goal[0]:
            # Low-saturation dark green for land.
            land = QColor("#3F5A4E")
            land.setAlpha(110)
            return land
        # Low-saturation light blue for water.
        return QColor("#D6E6F2")

    def _draw_action_overlay(
        self,
        state: tuple[int, int],
        action_overlay: dict[str, list[Action]],
    ) -> None:
        si, sj = state
        palette = {
            "invalid": QColor("#DC2626"),
            "valid": QColor("#EAB308"),
            "best": QColor("#16A34A"),
        }

        for key in ("invalid", "valid", "best"):
            for action in action_overlay.get(key, []):
                cell = (si + int(action[0]), sj + int(action[1]))
                self._fill_cell(cell, palette[key])

    def _draw_active_position(self, path: list[tuple[int, int]], start: tuple[int, int]) -> None:
        cell = path[-1] if path else start
        i, j = cell
        if not (0 <= i < self._nx and 0 <= j < self._ny):
            return
        x, y = self._cell_origin(i, j)
        pen = QPen(QColor("#B91C1C"))
        pen.setWidthF(2.0)
        pen.setCosmetic(True)
        self._scene.addRect(x + 1.5, y + 1.5, self._cell_size - 3, self._cell_size - 3, pen)

    def _draw_arrow_head(
        self,
        ex: float,
        ey: float,
        vi: float,
        vj: float,
        size: float,
        pen: QPen,
    ) -> None:
        """Draw a small arrow head at (ex, ey) pointing in direction (vi, vj)."""
        magnitude = math.hypot(vi, vj)
        if magnitude == 0.0:
            return

        di = vi / magnitude
        dj = -vj / magnitude
        perp_i = dj
        perp_j = di

        p1 = QPointF(ex, ey)
        p2 = QPointF(ex - di * size - perp_i * size / 2, ey - dj * size + perp_j * size / 2)
        p3 = QPointF(ex - di * size + perp_i * size / 2, ey - dj * size - perp_j * size / 2)

        arrow = QPolygonF([p1, p2, p3])
        brush = QBrush(pen.color())
        self._scene.addPolygon(arrow, pen, brush)

    def _draw_path(self, path: list[tuple[int, int]], show_labels: bool) -> None:
        if len(path) < 2:
            return

        path_pen = QPen(QColor("#D97706"))
        path_pen.setWidthF(2.4)
        path_pen.setCosmetic(True)

        for idx in range(len(path) - 1):
            i0, j0 = path[idx]
            i1, j1 = path[idx + 1]
            x0, y0 = self._cell_center(i0, j0)
            x1, y1 = self._cell_center(i1, j1)
            self._scene.addLine(x0, y0, x1, y1, path_pen)

            if show_labels and idx % max(1, len(path) // 12) == 0:
                label = self._scene.addText(str(idx))
                label.setDefaultTextColor(QColor("#7C2D12"))
                label.setFont(QFont("Segoe UI", 7))
                label.setPos(x0 + 2, y0 + 2)

    def _draw_flow(
        self,
        flow_vector: tuple[float, float],
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> None:
        vi, vj = flow_vector
        if abs(vi) + abs(vj) == 0.0:
            return

        step = 1 if self._nx * self._ny <= 1500 else 2
        flow_pen = QPen(QColor("#94A3B8"))
        flow_pen.setWidthF(1.0)
        flow_pen.setCosmetic(True)

        scale = self._cell_size * 0.35
        arrow_size = 3.0
        for i in range(0, self._nx, step):
            if i < start[0] or i > goal[0]:
                continue
            for j in range(0, self._ny, step):
                cx, cy = self._cell_center(i, j)
                ex = cx + vi * scale
                ey = cy - vj * scale
                self._scene.addLine(cx, cy, ex, ey, flow_pen)
                self._draw_arrow_head(ex, ey, vi, vj, arrow_size, flow_pen)

    def resizeEvent(self, event: Any) -> None:  # noqa: ANN401
        super().resizeEvent(event)
        self.fitInView(self._scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)


class CompareView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._runs: list[RenderRunData] = []
        self._canvases: list[GridCanvas] = []
        self._nx = 40
        self._ny = 20

        self._container = QWidget()
        self._layout = QGridLayout(self._container)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(12)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(self._container)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._scroll)

    def set_grid_shape(self, nx: int, ny: int) -> None:
        self._nx = max(1, nx)
        self._ny = max(1, ny)
        for canvas in self._canvases:
            canvas.set_grid_shape(self._nx, self._ny)

    def set_runs(self, runs: list[RenderRunData]) -> None:
        self._runs = runs

        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self._canvases = []
        columns = 2
        for idx, _run in enumerate(runs):
            canvas = GridCanvas("Compare View", self._nx, self._ny, minimum_width=420)
            self._canvases.append(canvas)
            self._layout.addWidget(canvas, idx // columns, idx % columns)

    def draw_runs(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        flow_vector: tuple[float, float],
        show_flow: bool,
        show_labels: bool,
        frame_index: int,
    ) -> None:
        for canvas, run in zip(self._canvases, self._runs):
            visible_path = visible_path_for_frame(run.path, frame_index)
            canvas.draw_run(
                start=start,
                goal=goal,
                path=visible_path,
                flow_vector=flow_vector,
                show_flow=show_flow,
                show_labels=show_labels,
                title_suffix=f"{run.title} | frame={len(visible_path)}/{len(run.path)}",
            )

    def is_empty(self) -> bool:
        return not self._runs


class ControlPanel(QWidget):
    play_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    step_clicked = pyqtSignal()
    step_reverse_clicked = pyqtSignal()
    reset_clicked = pyqtSignal()
    redraw_requested = pyqtSignal()

    def __init__(self, initial_impact: float = 0.0) -> None:
        super().__init__()
        self._updating_flow_controls = False

        self.environment_combo = QComboBox()
        self.environment_combo.addItems(["baseline"])

        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems([
            "dijkstra",
            "a_star",
            "weighted_a_star",
            "dynamic_programming",
            "apf",
            "q_learning",
        ])

        self.seed_combo = QComboBox()
        self.seed_combo.addItems(parse_seed_options([1, 2, 3, 4, 5]))

        self.show_flow = QCheckBox("Show flow vectors")
        self.show_flow.setChecked(True)

        self.show_labels = QCheckBox("Show path labels")
        self.show_labels.setChecked(True)

        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.1, 10.0)
        self.speed.setSingleStep(0.1)
        self.speed.setValue(1.0)
        self.speed.setSuffix("x")

        self.flow_direction_dial = QDial()
        self.flow_direction_dial.setRange(0, 359)
        self.flow_direction_dial.setWrapping(True)
        self.flow_direction_dial.setNotchesVisible(True)

        self.flow_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.flow_strength_slider.setRange(0, 100)
        self.flow_strength_slider.setValue(100)

        self.flow_x_edit = QLineEdit("1.000")
        self.flow_y_edit = QLineEdit("0.000")
        self.flow_strength_edit = QLineEdit("1.000")
        self.flow_impact_spin = QDoubleSpinBox()
        self.flow_impact_spin.setRange(0.0, 100.0)
        self.flow_impact_spin.setSingleStep(1.0)
        self.flow_impact_spin.setDecimals(0)
        self.flow_impact_spin.setSuffix("%")
        self.flow_impact_spin.setValue(max(0.0, min(100.0, float(initial_impact) * 100.0)))
        for widget in (self.flow_x_edit, self.flow_y_edit, self.flow_strength_edit, self.flow_impact_spin):
            widget.setMaximumWidth(100)

        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")
        self.step_button = QPushButton("Step forward")
        self.step_reverse_button = QPushButton("Step reverse")
        self.reset_button = QPushButton("Reset")

        self.play_button.clicked.connect(self.play_clicked.emit)
        self.pause_button.clicked.connect(self.pause_clicked.emit)
        self.step_button.clicked.connect(self.step_clicked.emit)
        self.step_reverse_button.clicked.connect(self.step_reverse_clicked.emit)
        self.reset_button.clicked.connect(self.reset_clicked.emit)

        self.show_flow.stateChanged.connect(lambda _: self.redraw_requested.emit())
        self.show_labels.stateChanged.connect(lambda _: self.redraw_requested.emit())
        self.speed.valueChanged.connect(lambda _: self.redraw_requested.emit())

        self.flow_direction_dial.valueChanged.connect(self._on_flow_direction_changed)
        self.flow_strength_slider.valueChanged.connect(self._on_flow_strength_changed)
        self.flow_x_edit.editingFinished.connect(self._on_vector_line_edits_changed)
        self.flow_y_edit.editingFinished.connect(self._on_vector_line_edits_changed)
        self.flow_strength_edit.editingFinished.connect(self._on_strength_line_edit_changed)
        self.flow_impact_spin.valueChanged.connect(lambda _: self.redraw_requested.emit())

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.addRow("Environment", self.environment_combo)
        form.addRow("Algorithm", self.algorithm_combo)
        form.addRow("Seed", self.seed_combo)
        form.addRow("Animation speed", self.speed)
        layout.addLayout(form)

        layout.addWidget(QLabel("Flow direction"))
        layout.addWidget(self.flow_direction_dial)
        layout.addWidget(QLabel("Flow strength"))
        layout.addWidget(self.flow_strength_slider)

        flow_values = QFormLayout()
        flow_values.addRow("flow x", self.flow_x_edit)
        flow_values.addRow("flow y", self.flow_y_edit)
        flow_values.addRow("strength", self.flow_strength_edit)
        flow_values.addRow("impact", self.flow_impact_spin)
        layout.addLayout(flow_values)

        layout.addWidget(self.show_flow)
        layout.addWidget(self.show_labels)
        layout.addWidget(self.play_button)
        layout.addWidget(self.pause_button)
        layout.addWidget(self.step_button)
        layout.addWidget(self.step_reverse_button)
        layout.addWidget(self.reset_button)
        layout.addStretch(1)

        self.set_flow_vector(1.0, 0.0)

    def set_flow_vector(self, vi: float, vj: float) -> None:
        self._updating_flow_controls = True
        angle, strength = vector_to_angle_strength(vi, vj)
        self.flow_direction_dial.setValue(int(round(math_to_dial_angle(angle))) % 360)
        self.flow_strength_slider.setValue(int(round(strength * 100.0)))
        self.flow_x_edit.setText(f"{float(vi):.3f}")
        self.flow_y_edit.setText(f"{float(vj):.3f}")
        self.flow_strength_edit.setText(f"{strength:.3f}")
        self._updating_flow_controls = False

    def flow_vector(self) -> tuple[float, float]:
        try:
            return (float(self.flow_x_edit.text()), float(self.flow_y_edit.text()))
        except ValueError:
            return (1.0, 0.0)

    def _on_flow_direction_changed(self, value: int) -> None:
        if self._updating_flow_controls:
            return
        strength = self.flow_strength_slider.value() / 100.0
        vi, vj = angle_strength_to_vector(dial_to_math_angle(float(value)), strength)
        self.set_flow_vector(vi, vj)
        self.redraw_requested.emit()

    def _on_flow_strength_changed(self, value: int) -> None:
        if self._updating_flow_controls:
            return
        vi, vj = angle_strength_to_vector(
            dial_to_math_angle(float(self.flow_direction_dial.value())),
            value / 100.0,
        )
        self.set_flow_vector(vi, vj)
        self.redraw_requested.emit()

    def _on_vector_line_edits_changed(self) -> None:
        if self._updating_flow_controls:
            return
        try:
            vi = float(self.flow_x_edit.text())
            vj = float(self.flow_y_edit.text())
        except ValueError:
            return
        self.set_flow_vector(vi, vj)
        self.redraw_requested.emit()

    def _on_strength_line_edit_changed(self) -> None:
        if self._updating_flow_controls:
            return
        try:
            strength = max(0.0, float(self.flow_strength_edit.text()))
        except ValueError:
            return
        vi, vj = angle_strength_to_vector(
            dial_to_math_angle(float(self.flow_direction_dial.value())),
            strength,
        )
        self.set_flow_vector(vi, vj)
        self.redraw_requested.emit()

    def selection_state(self) -> UISelectionState:
        return UISelectionState(
            environment=self.environment_combo.currentText(),
            algorithm=self.algorithm_combo.currentText(),
            seed=int(self.seed_combo.currentText()),
            show_flow=self.show_flow.isChecked(),
            show_labels=self.show_labels.isChecked(),
            speed=float(self.speed.value()),
            flow_vector=self.flow_vector(),
            impact=float(self.flow_impact_spin.value()) / 100.0,
        )


class MetricsPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.labels: dict[str, QLabel] = {}
        self._neighborhood_cells: dict[tuple[int, int], QLabel] = {}

        layout = QFormLayout(self)
        for key in [
            "total_cost",
            "delta_j",
            "steps",
            "plan_time",
            "training_time",
            "inference_time",
            "angle_valid",
        ]:
            label = QLabel("-")
            self.labels[key] = label
            layout.addRow(key, label)

        neighborhood = QWidget()
        grid = QGridLayout(neighborhood)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(2)
        grid.setVerticalSpacing(2)

        for row in range(3):
            for col in range(3):
                di = col - 1
                dj = 1 - row
                cell = QLabel("")
                cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
                cell.setMinimumSize(76, 52)
                cell.setStyleSheet("background-color: #F8F9FA; border: 1px solid #D0D7DE; font-size: 9px;")
                self._neighborhood_cells[(di, dj)] = cell
                grid.addWidget(cell, row, col)

        spacer = QLabel("")
        spacer.setFixedHeight(24)
        layout.addRow(spacer)

        local_label = QLabel("Local 3x3")
        local_label.setStyleSheet("font-weight: 600;")
        layout.addRow(local_label)
        layout.addRow(neighborhood)

    def update_metrics(self, metrics: dict[str, Any]) -> None:
        for key, label in self.labels.items():
            label.setText(str(metrics.get(key, "-")))

    def update_neighborhood(
        self,
        env: RiverEnvironment | None,
        state: State | None,
        action_overlay: dict[str, list[Action]] | None,
        cost_fn: CostFunction | None,
    ) -> None:
        if env is None or state is None:
            for cell in self._neighborhood_cells.values():
                cell.setText("")
                cell.setStyleSheet("background-color: #F8F9FA; border: 1px solid #D0D7DE; font-size: 9px;")
            return

        overlay = action_overlay or {"best": [], "valid": [], "invalid": []}
        best_actions = set(overlay.get("best", []))
        valid_actions = set(overlay.get("valid", []))
        invalid_actions = set(overlay.get("invalid", []))

        for (di, dj), cell in self._neighborhood_cells.items():
            target = State(state.i + di, state.j + dj)
            if di == 0 and dj == 0:
                cell.setText("Current\nposition")
                cell.setStyleSheet("background-color: #E2E8F0; border: 1px solid #94A3B8; font-size: 9px; font-weight: 600;")
                continue

            action = (di, dj)
            if not env.grid.contains(target):
                cell.setText("")
                cell.setStyleSheet("background-color: #D1D5DB; border: 1px solid #9CA3AF; font-size: 9px;")
                continue

            if env.is_land(target):
                cell.setText("")
                cell.setStyleSheet("background-color: rgba(63, 90, 78, 110); border: 1px solid #4B5F55; font-size: 9px;")
                continue

            background = "#D6E6F2"
            border = "#8EA6B8"
            text = ""

            if action in best_actions:
                background = "#16A34A"
                border = "#166534"
            elif action in valid_actions:
                background = "#EAB308"
                border = "#A16207"
            elif action in invalid_actions:
                background = "#DC2626"
                border = "#991B1B"

            if (action in best_actions or action in valid_actions) and cost_fn is not None:
                try:
                    time_cost = cost_fn.time(state, action)
                    energy_cost = cost_fn.energy(state, action)
                    total_cost = cost_fn(state, action)
                    text = (
                        f"Time {time_cost:.2f}\n"
                        f"Energy {energy_cost:.2f}\n"
                        f"Total {total_cost:.2f}"
                    )
                except Exception:  # noqa: BLE001
                    text = ""

            cell.setText(text)
            cell.setStyleSheet(
                "background-color: "
                + background
                + "; border: 1px solid "
                + border
                + "; font-size: 9px;"
            )


class RiverCrossingMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("River Crossing - PyQt6 UI (M3)")
        self.resize(1360, 860)

        self._worker_thread: QThread | None = None
        self._worker: SimulationWorker | None = None
        self._records: list[dict[str, Any]] = []
        self._results_path = Path("analysis/raw_results.json")
        self._single_run: RenderRunData | None = None
        self._compare_runs: list[RenderRunData] = []
        self._single_frame = 0
        self._compare_frame = 0

        self._env_cfg = load_env_config("configs/env.yaml")
        self._algo_cfg = load_algorithm_config("configs/algorithm.yaml")
        self._exp_cfg = load_experiment_config("configs/experiment.yaml")

        self._animation_timer = QTimer(self)
        self._animation_timer.timeout.connect(self._advance_animation)

        default_impact = float(self._algo_cfg.get("common", {}).get("beta", 0.0))
        self.control_panel = ControlPanel(initial_impact=default_impact)
        self.metrics_panel = MetricsPanel()

        self.run_canvas = GridCanvas("Run View")
        self.single_canvas = GridCanvas("Best Path View")
        self.compare_view = CompareView()

        self.tabs = QTabWidget()
        self.tabs.addTab(self.run_canvas, "Run")
        self.tabs.addTab(self.single_canvas, "Best path")
        self.tabs.addTab(self.compare_view, "Compare")
        self.tabs.currentChanged.connect(lambda _: self._render_current_tab())

        central = QWidget()
        central_layout = QHBoxLayout(central)
        central_layout.addWidget(self.control_panel, 1)
        central_layout.addWidget(self.tabs, 3)
        central_layout.addWidget(self.metrics_panel, 1)
        self.setCentralWidget(central)

        self._build_toolbar()
        self._build_log_panel()
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")

        self.control_panel.environment_combo.currentIndexChanged.connect(self._on_selection_changed)
        self.control_panel.algorithm_combo.currentIndexChanged.connect(self._on_selection_changed)
        self.control_panel.seed_combo.currentIndexChanged.connect(self._on_selection_changed)
        self.control_panel.play_clicked.connect(self._start_animation)
        self.control_panel.pause_clicked.connect(self._pause_animation)
        self.control_panel.step_clicked.connect(self._step_animation)
        self.control_panel.step_reverse_clicked.connect(self._step_animation_reverse)
        self.control_panel.reset_clicked.connect(self._reset_animation)
        self.control_panel.redraw_requested.connect(self._handle_redraw_request)

        self._apply_env_to_views()
        self._load_records_if_available(self._results_path)

    def _apply_env_to_views(self) -> None:
        nx = int(self._env_cfg["grid"]["nx"])
        ny = int(self._env_cfg["grid"]["ny"])
        self.run_canvas.set_grid_shape(nx, ny)
        self.single_canvas.set_grid_shape(nx, ny)
        self.compare_view.set_grid_shape(nx, ny)
        self._populate_control_options()
        flow = self._env_cfg.get("flow", {}).get("vector", [1.0, 0.0])
        self.control_panel.set_flow_vector(float(flow[0]), float(flow[1]))
        self._render_current_tab()

    def _populate_control_options(self) -> None:
        environments = [str(item.get("name", "baseline")) for item in self._exp_cfg["environments"]]
        algorithms = [str(item) for item in self._exp_cfg["algorithms"]]
        seeds = [int(item) for item in self._exp_cfg["seeds"]]

        self.control_panel.environment_combo.clear()
        self.control_panel.environment_combo.addItems(environments)
        self.control_panel.algorithm_combo.clear()
        self.control_panel.algorithm_combo.addItems(algorithms)
        self.control_panel.seed_combo.clear()
        self.control_panel.seed_combo.addItems(parse_seed_options(seeds))

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        load_action = QAction("Load Results", self)
        load_action.triggered.connect(self._pick_results_file)
        toolbar.addAction(load_action)

        run_single_action = QAction("Run Single Experiment", self)
        run_single_action.triggered.connect(lambda: self._run_worker("run_single_experiment"))
        toolbar.addAction(run_single_action)

        run_batch_action = QAction("Run Experiment Batch", self)
        run_batch_action.triggered.connect(lambda: self._run_worker("run_batch_experiment"))
        toolbar.addAction(run_batch_action)

        show_results_action = QAction("Show Results", self)
        show_results_action.triggered.connect(lambda: self._run_worker("show_results"))
        toolbar.addAction(show_results_action)

        compare_action = QAction("Compare", self)
        compare_action.triggered.connect(lambda: self._run_worker("compare"))
        toolbar.addAction(compare_action)

        export_png_action = QAction("Export PNG", self)
        export_png_action.triggered.connect(self._export_current_tab_png)
        toolbar.addAction(export_png_action)

        export_report_action = QAction("Export Report", self)
        export_report_action.triggered.connect(
            lambda: self._append_log("Export Report clicked (not implemented yet).")
        )
        toolbar.addAction(export_report_action)

    def _build_log_panel(self) -> None:
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)

        dock = QDockWidget("Status Log", self)
        dock.setWidget(self.log)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

    def _append_log(self, message: str) -> None:
        self.log.appendPlainText(message)
        self.statusBar().showMessage(message, 3000)

    def _pick_results_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Raw Results JSON",
            str(Path.cwd()),
            "JSON files (*.json)",
        )
        if path:
            self._load_records_if_available(Path(path))

    def _load_records_if_available(self, path: Path) -> None:
        if not path.exists():
            self._append_log(f"Results file not found: {path}. Use runner first or load manually.")
            return
        try:
            self._records = load_experiment_results(path)
            self._results_path = path
            self._append_log(f"Loaded {len(self._records)} runs from {path}.")
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"Failed to load results from {path}: {exc}")

    def _export_current_tab_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export PNG",
            str(Path("analysis") / "ui_export.png"),
            "PNG files (*.png)",
        )
        if not path:
            return

        widget = self.tabs.currentWidget()
        if widget is None or not widget.grab().save(path, "PNG"):
            self._append_log(f"Failed to save PNG: {path}")
            return
        self._append_log(f"Exported PNG: {path}")

    def _run_worker(self, mode: str) -> None:
        if self._worker_thread is not None:
            self._append_log("A worker is already running.")
            return
        if not self._records and mode in {"show_results", "compare"}:
            self._append_log("No results loaded. Generate results first or load a JSON file.")
            return

        selection = self.control_panel.selection_state()
        self._append_log(
            f"Starting {mode} with env={selection.environment}, algorithm={selection.algorithm}, seed={selection.seed}."
        )

        self._worker_thread = QThread(self)
        self._worker = SimulationWorker(
            mode,
            list(self._records),
            selection,
            config_dir=Path("configs"),
            results_path=self._results_path,
        )
        self._worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(lambda value: self._append_log(f"Progress: {value}%"))
        self._worker.message.connect(self._append_log)
        self._worker.result.connect(self._on_worker_result)
        self._worker.error.connect(self._on_worker_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        self._worker_thread.start()

    def _on_worker_result(self, payload: dict[str, Any]) -> None:
        self._append_log(f"Worker result ready for mode={payload.get('mode')}")
        mode = str(payload.get("mode", ""))

        if mode == "run_single_experiment":
            runs_list = [dict(r) for r in payload.get("runs", [])]
            for run in runs_list:
                self._upsert_record(run)
            # Display the run matching the currently-selected seed if available.
            selected_seed = int(payload.get("selected_seed", self.control_panel.selection_state().seed))
            display_run = next(
                (r for r in runs_list if int(r.get("seed", -1)) == selected_seed),
                runs_list[0] if runs_list else None,
            )
            if display_run is not None:
                self._set_single_run(display_run)
            self._append_log(
                f"Single experiment finished: {len(runs_list)} seed(s) run, rendering seed={selected_seed}."
            )
            return

        if mode == "run_batch_experiment":
            self._records = [dict(item) for item in payload.get("records", [])]
            self._append_log(f"Batch finished: {len(self._records)} runs saved to {payload.get('results_path')}")
            self._append_log(f"Evaluation summary saved to {payload.get('summary_path')}")
            self.metrics_panel.update_metrics({})
            return

        if mode == "show_results":
            self._set_single_run(payload["run"], switch_to_run_tab=False)
            return
        if mode == "compare":
            self._set_compare_runs(payload["runs"])
            return

        self._append_log(f"Unhandled worker mode: {mode}")
        self.metrics_panel.update_metrics({})

    def _upsert_record(self, run: dict[str, Any]) -> None:
        key = (
            str(run.get("environment_name")),
            str(run.get("algorithm")),
            int(run.get("seed", -1)),
        )
        for index, existing in enumerate(self._records):
            existing_key = (
                str(existing.get("environment_name")),
                str(existing.get("algorithm")),
                int(existing.get("seed", -2)),
            )
            if existing_key == key:
                self._records[index] = run
                return
        self._records.append(run)

    def _set_single_run(self, run: dict[str, Any], switch_to_run_tab: bool = True) -> None:
        self._pause_animation(log_message=False)
        self._single_run = build_render_run_data([run])[0]
        self._single_frame = 1 if self._single_run.path else 0
        if switch_to_run_tab:
            self.tabs.setCurrentIndex(0)
        self._render_current_tab()

    def _on_selection_changed(self, _: int) -> None:
        if not self._records or self._worker_thread is not None:
            return
        self._run_worker("show_results")

    def _set_compare_runs(self, runs: list[dict[str, Any]]) -> None:
        self._pause_animation(log_message=False)
        self._compare_runs = build_render_run_data(runs)
        self.compare_view.set_runs(self._compare_runs)
        self._compare_frame = max_frame_count(self._compare_runs)
        self.tabs.setCurrentIndex(2)
        self._render_compare_frame()

    def _render_current_tab(self) -> None:
        if self.tabs.currentIndex() == 0:
            self._render_run_frame()
        elif self.tabs.currentIndex() == 1:
            self._render_single_frame()
        else:
            self._render_compare_frame()

    def _handle_redraw_request(self) -> None:
        flow = self.control_panel.selection_state().flow_vector
        self._env_cfg.setdefault("flow", {})["vector"] = [float(flow[0]), float(flow[1])]
        if self._animation_timer.isActive():
            self._restart_animation_timer()
        self._render_current_tab()

    def _reset_animation(self) -> None:
        self._pause_animation(log_message=False)
        if self._single_run is not None:
            self._single_frame = 1 if self._single_run.path else 0
        if self._compare_runs:
            self._compare_frame = 1
        self._render_current_tab()
        self._append_log("Reset to frame 1.")

    def _current_environment(self) -> RiverEnvironment:
        env_cfg = dict(self._env_cfg)
        env_cfg["flow"] = dict(env_cfg.get("flow", {}))
        flow = self.control_panel.selection_state().flow_vector
        env_cfg["flow"]["vector"] = [float(flow[0]), float(flow[1])]
        return RiverEnvironment.from_config(env_cfg)

    def _current_cost_function(self, env: RiverEnvironment) -> CostFunction:
        selection = self.control_panel.selection_state()
        effective_algo_cfg = dict(self._algo_cfg)
        effective_algo_cfg["common"] = dict(effective_algo_cfg.get("common", {}))
        effective_algo_cfg["common"]["beta"] = float(selection.impact)
        return CostFunction.from_config(effective_algo_cfg, env.flow)

    def _env_start(self) -> tuple[int, int]:
        return (int(self._env_cfg["start"][0]), int(self._env_cfg["start"][1]))

    def _env_goal(self) -> tuple[int, int]:
        return (int(self._env_cfg["goal"][0]), int(self._env_cfg["goal"][1]))

    def _env_flow(self) -> tuple[float, float]:
        flow = self.control_panel.selection_state().flow_vector
        return (float(flow[0]), float(flow[1]))

    def _render_run_frame(self) -> None:
        start = self._env_start()
        goal = self._env_goal()
        flow = self._env_flow()
        state = self.control_panel.selection_state()

        if self._single_run is None:
            env = self._current_environment()
            overlay = build_action_overlay(env, State(start[0], start[1]), best_action=None)
            cost_fn = self._current_cost_function(env)
            self.run_canvas.draw_run(
                start=start,
                goal=goal,
                path=[],
                flow_vector=flow,
                show_flow=state.show_flow,
                show_labels=state.show_labels,
                title_suffix="ready",
            )
            self.metrics_panel.update_metrics({})
            self.metrics_panel.update_neighborhood(env, State(start[0], start[1]), overlay, cost_fn)
            return

        env = self._current_environment()
        if self._single_run.exploration_snapshots:
            snap_idx = max(0, min(self._single_frame - 1, len(self._single_run.exploration_snapshots) - 1))
            visible_path = self._single_run.exploration_snapshots[snap_idx]
            total = len(self._single_run.exploration_snapshots)
            frame_label = f"explore {snap_idx + 1}/{total}"
        else:
            visible_path = visible_path_for_frame(self._single_run.path, self._single_frame)
            frame_label = f"run frame={len(visible_path)}/{len(self._single_run.path)}"
        current_cell = visible_path[-1] if visible_path else start
        best_action = None
        next_action_index = max(0, len(visible_path) - 1)
        if next_action_index < len(self._single_run.actions):
            best_action = self._single_run.actions[next_action_index]

        overlay = build_action_overlay(
            env,
            State(int(current_cell[0]), int(current_cell[1])),
            best_action=best_action,
        )

        self.run_canvas.draw_run(
            start=start,
            goal=goal,
            path=visible_path,
            flow_vector=flow,
            show_flow=state.show_flow,
            show_labels=state.show_labels,
            title_suffix=f"{self._single_run.title} | {frame_label}",
            overlay_state=current_cell,
            action_overlay=overlay,
        )
        self.metrics_panel.update_metrics(format_metrics(self._single_run.run, self._records))
        self.metrics_panel.update_neighborhood(
            env,
            State(int(current_cell[0]), int(current_cell[1])),
            overlay,
            self._current_cost_function(env),
        )

    def _render_single_frame(self) -> None:
        start = self._env_start()
        goal = self._env_goal()
        flow = self._env_flow()
        state = self.control_panel.selection_state()

        if self._single_run is None:
            env = self._current_environment()
            overlay = build_action_overlay(env, State(start[0], start[1]), best_action=None)
            self.single_canvas.draw_run(
                start=start,
                goal=goal,
                path=[],
                flow_vector=flow,
                show_flow=state.show_flow,
                show_labels=state.show_labels,
                title_suffix="ready",
            )
            self.metrics_panel.update_metrics({})
            self.metrics_panel.update_neighborhood(
                env,
                State(start[0], start[1]),
                overlay,
                self._current_cost_function(env),
            )
            return

        visible_path = visible_path_for_frame(self._single_run.best_path, self._single_frame)
        env = self._current_environment()
        current_cell = visible_path[-1] if visible_path else start
        best_action = None
        next_action_index = max(0, len(visible_path) - 1)
        if next_action_index < len(self._single_run.actions):
            best_action = self._single_run.actions[next_action_index]
        overlay = build_action_overlay(
            env,
            State(int(current_cell[0]), int(current_cell[1])),
            best_action=best_action,
        )
        self.single_canvas.draw_run(
            start=start,
            goal=goal,
            path=visible_path,
            flow_vector=flow,
            show_flow=state.show_flow,
            show_labels=state.show_labels,
            title_suffix=f"{self._single_run.title} | frame={len(visible_path)}/{len(self._single_run.best_path)}",
        )
        self.metrics_panel.update_metrics(format_metrics(self._single_run.run, self._records))
        self.metrics_panel.update_neighborhood(
            env,
            State(int(current_cell[0]), int(current_cell[1])),
            overlay,
            self._current_cost_function(env),
        )

    def _render_compare_frame(self) -> None:
        start = self._env_start()
        goal = self._env_goal()
        flow = self._env_flow()
        state = self.control_panel.selection_state()

        if not self._compare_runs:
            self.compare_view.set_runs([])
            self.metrics_panel.update_metrics({})
            self.metrics_panel.update_neighborhood(None, None, None, None)
            return

        self.compare_view.draw_runs(
            start=start,
            goal=goal,
            flow_vector=flow,
            show_flow=state.show_flow,
            show_labels=state.show_labels,
            frame_index=self._compare_frame,
        )

        selected = next(
            (item for item in self._compare_runs if item.run.get("algorithm") == state.algorithm),
            self._compare_runs[0],
        )
        self.metrics_panel.update_metrics(format_metrics(selected.run, self._records))
        env = self._current_environment()
        visible_path = visible_path_for_frame(selected.path, self._compare_frame)
        current_cell = visible_path[-1] if visible_path else start
        best_action = None
        next_action_index = max(0, len(visible_path) - 1)
        if next_action_index < len(selected.actions):
            best_action = selected.actions[next_action_index]
        overlay = build_action_overlay(
            env,
            State(int(current_cell[0]), int(current_cell[1])),
            best_action=best_action,
        )
        self.metrics_panel.update_neighborhood(
            env,
            State(int(current_cell[0]), int(current_cell[1])),
            overlay,
            self._current_cost_function(env),
        )

    def _active_total_frames(self) -> int:
        if self.tabs.currentIndex() in {0, 1}:
            if self._single_run is None:
                return 0
            if self.tabs.currentIndex() == 0 and self._single_run.exploration_snapshots:
                return len(self._single_run.exploration_snapshots)
            path = self._single_run.best_path if self.tabs.currentIndex() == 1 else self._single_run.path
            return len(path)
        return max_frame_count(self._compare_runs)

    def _start_animation(self) -> None:
        total_frames = self._active_total_frames()
        if total_frames <= 0:
            self._append_log("No rendered path available for animation.")
            return

        if self.tabs.currentIndex() in {0, 1} and self._single_frame >= total_frames:
            self._single_frame = 0
        if self.tabs.currentIndex() == 2 and self._compare_frame >= total_frames:
            self._compare_frame = 0

        self._restart_animation_timer()
        self._append_log("Animation started.")

    def _pause_animation(self, log_message: bool = True) -> None:
        if self._animation_timer.isActive():
            self._animation_timer.stop()
            if log_message:
                self._append_log("Animation paused.")

    def _step_animation(self) -> None:
        total_frames = self._active_total_frames()
        if total_frames <= 0:
            self._append_log("No rendered path available for stepping.")
            return
        self._advance_animation(step_only=True)

    def _step_animation_reverse(self) -> None:
        total_frames = self._active_total_frames()
        if total_frames <= 0:
            self._append_log("No rendered path available for stepping.")
            return
        if self.tabs.currentIndex() in {0, 1}:
            self._single_frame = max(0, self._single_frame - 1)
        else:
            self._compare_frame = max(0, self._compare_frame - 1)
        self._render_current_tab()

    def _restart_animation_timer(self) -> None:
        speed = max(0.1, self.control_panel.selection_state().speed)
        interval_ms = max(40, int(300 / speed))
        self._animation_timer.start(interval_ms)

    def _advance_animation(self, step_only: bool = False) -> None:
        total_frames = self._active_total_frames()
        if total_frames <= 0:
            self._pause_animation(log_message=False)
            return

        if self.tabs.currentIndex() in {0, 1}:
            self._single_frame = min(self._single_frame + 1, total_frames)
            self._render_current_tab()
            at_end = self._single_frame >= total_frames
        else:
            self._compare_frame = min(self._compare_frame + 1, total_frames)
            self._render_compare_frame()
            at_end = self._compare_frame >= total_frames

        if at_end or step_only:
            self._animation_timer.stop()

    def _on_worker_error(self, message: str) -> None:
        self._append_log(f"Error: {message}")
        QMessageBox.warning(self, "Worker Error", message)

    def _on_worker_finished(self) -> None:
        self._append_log("Worker finished.")
        self._worker = None
        self._worker_thread = None


def launch_ui() -> int:
    app = QApplication.instance() or QApplication([])
    window = RiverCrossingMainWindow()
    window.show()
    return app.exec()
