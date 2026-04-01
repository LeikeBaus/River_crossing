from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QBrush, QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from core.config_loader import load_env_config, load_experiment_config
from experiments.runner import load_experiment_results


@dataclass(frozen=True)
class UISelectionState:
    environment: str
    algorithm: str
    seed: int
    show_flow: bool
    show_labels: bool
    speed: float


@dataclass(frozen=True)
class RenderRunData:
    title: str
    run: dict[str, Any]
    path: list[tuple[int, int]]


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
    return [
        RenderRunData(
            title=(
                f"{run.get('algorithm')} | env={run.get('environment_name')} | "
                f"seed={run.get('seed')}"
            ),
            run=run,
            path=coerce_path(run.get("path", [])),
        )
        for run in runs
    ]


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

    def __init__(self, mode: str, records: list[dict[str, Any]], selection: UISelectionState) -> None:
        super().__init__()
        self.mode = mode
        self.records = records
        self.selection = selection

    def run(self) -> None:
        try:
            self.message.emit(f"Worker started in mode '{self.mode}'.")
            self.progress.emit(15)

            if self.mode == "single":
                run = find_single_run(self.records, self.selection)
                if run is None:
                    raise ValueError(
                        "No run found for current selection. Load or generate matching results first."
                    )
                self.progress.emit(100)
                self.result.emit({"mode": "single", "run": run})
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
    ) -> None:
        self._scene.clear()
        border_pen = QPen(QColor("#D0D7DE"))
        border_pen.setWidth(1)
        fill_brush = QBrush(QColor("#F8F9FA"))

        for i in range(self._nx):
            for j in range(self._ny):
                x, y = self._cell_origin(i, j)
                self._scene.addRect(x, y, self._cell_size, self._cell_size, border_pen, fill_brush)

        if show_flow:
            self._draw_flow(flow_vector)

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

    def _draw_flow(self, flow_vector: tuple[float, float]) -> None:
        vi, vj = flow_vector
        if abs(vi) + abs(vj) == 0.0:
            return

        step = 1 if self._nx * self._ny <= 1500 else 2
        flow_pen = QPen(QColor("#94A3B8"))
        flow_pen.setWidthF(1.0)
        flow_pen.setCosmetic(True)

        scale = self._cell_size * 0.35
        for i in range(0, self._nx, step):
            for j in range(0, self._ny, step):
                cx, cy = self._cell_center(i, j)
                ex = cx + vi * scale
                ey = cy - vj * scale
                self._scene.addLine(cx, cy, ex, ey, flow_pen)

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
    run_single_clicked = pyqtSignal()
    run_batch_clicked = pyqtSignal()
    compare_clicked = pyqtSignal()
    play_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    step_clicked = pyqtSignal()
    redraw_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
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

        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")
        self.step_button = QPushButton("Step")

        self.run_single_button = QPushButton("Run Single")
        self.run_batch_button = QPushButton("Run Batch")
        self.compare_button = QPushButton("Compare")

        self.run_single_button.clicked.connect(self.run_single_clicked.emit)
        self.run_batch_button.clicked.connect(self.run_batch_clicked.emit)
        self.compare_button.clicked.connect(self.compare_clicked.emit)
        self.play_button.clicked.connect(self.play_clicked.emit)
        self.pause_button.clicked.connect(self.pause_clicked.emit)
        self.step_button.clicked.connect(self.step_clicked.emit)

        self.show_flow.stateChanged.connect(self.redraw_requested.emit)
        self.show_labels.stateChanged.connect(self.redraw_requested.emit)
        self.speed.valueChanged.connect(self.redraw_requested.emit)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.addRow("Environment", self.environment_combo)
        form.addRow("Algorithm", self.algorithm_combo)
        form.addRow("Seed", self.seed_combo)
        form.addRow("Animation speed", self.speed)
        layout.addLayout(form)
        layout.addWidget(self.show_flow)
        layout.addWidget(self.show_labels)
        layout.addWidget(self.play_button)
        layout.addWidget(self.pause_button)
        layout.addWidget(self.step_button)
        layout.addSpacing(12)
        layout.addWidget(self.run_single_button)
        layout.addWidget(self.run_batch_button)
        layout.addWidget(self.compare_button)
        layout.addStretch(1)

    def selection_state(self) -> UISelectionState:
        return UISelectionState(
            environment=self.environment_combo.currentText(),
            algorithm=self.algorithm_combo.currentText(),
            seed=int(self.seed_combo.currentText()),
            show_flow=self.show_flow.isChecked(),
            show_labels=self.show_labels.isChecked(),
            speed=float(self.speed.value()),
        )


class MetricsPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.labels: dict[str, QLabel] = {}

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

    def update_metrics(self, metrics: dict[str, Any]) -> None:
        for key, label in self.labels.items():
            label.setText(str(metrics.get(key, "-")))


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
        self._exp_cfg = load_experiment_config("configs/experiment.yaml")

        self._animation_timer = QTimer(self)
        self._animation_timer.timeout.connect(self._advance_animation)

        self.control_panel = ControlPanel()
        self.metrics_panel = MetricsPanel()

        self.single_canvas = GridCanvas("Single View")
        self.compare_view = CompareView()

        self.tabs = QTabWidget()
        self.tabs.addTab(self.single_canvas, "Single")
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

        self.control_panel.run_single_clicked.connect(lambda: self._run_worker("single"))
        self.control_panel.run_batch_clicked.connect(lambda: self._run_worker("batch"))
        self.control_panel.compare_clicked.connect(lambda: self._run_worker("compare"))
        self.control_panel.play_clicked.connect(self._start_animation)
        self.control_panel.pause_clicked.connect(self._pause_animation)
        self.control_panel.step_clicked.connect(self._step_animation)
        self.control_panel.redraw_requested.connect(self._handle_redraw_request)

        self._apply_env_to_views()
        self._load_records_if_available(self._results_path)

    def _apply_env_to_views(self) -> None:
        nx = int(self._env_cfg["grid"]["nx"])
        ny = int(self._env_cfg["grid"]["ny"])
        self.single_canvas.set_grid_shape(nx, ny)
        self.compare_view.set_grid_shape(nx, ny)
        self._populate_control_options()
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

        run_single_action = QAction("Run Single", self)
        run_single_action.triggered.connect(lambda: self._run_worker("single"))
        toolbar.addAction(run_single_action)

        run_batch_action = QAction("Run Batch", self)
        run_batch_action.triggered.connect(lambda: self._run_worker("batch"))
        toolbar.addAction(run_batch_action)

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
        if not self._records and mode in {"single", "compare"}:
            self._append_log("No results loaded. Generate results first or load a JSON file.")
            return

        selection = self.control_panel.selection_state()
        self._append_log(
            f"Starting {mode} with env={selection.environment}, algorithm={selection.algorithm}, seed={selection.seed}."
        )

        self._worker_thread = QThread(self)
        self._worker = SimulationWorker(mode, self._records, selection)
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

        if mode == "single":
            self._set_single_run(payload["run"])
            return
        if mode == "compare":
            self._set_compare_runs(payload["runs"])
            return

        self.metrics_panel.update_metrics({})

    def _set_single_run(self, run: dict[str, Any]) -> None:
        self._pause_animation(log_message=False)
        self._single_run = build_render_run_data([run])[0]
        self._single_frame = len(self._single_run.path)
        self.tabs.setCurrentIndex(0)
        self._render_single_frame()

    def _set_compare_runs(self, runs: list[dict[str, Any]]) -> None:
        self._pause_animation(log_message=False)
        self._compare_runs = build_render_run_data(runs)
        self.compare_view.set_runs(self._compare_runs)
        self._compare_frame = max_frame_count(self._compare_runs)
        self.tabs.setCurrentIndex(1)
        self._render_compare_frame()

    def _render_current_tab(self) -> None:
        if self.tabs.currentIndex() == 0:
            self._render_single_frame()
        else:
            self._render_compare_frame()

    def _handle_redraw_request(self) -> None:
        if self._animation_timer.isActive():
            self._restart_animation_timer()
        self._render_current_tab()

    def _env_start(self) -> tuple[int, int]:
        return (int(self._env_cfg["start"][0]), int(self._env_cfg["start"][1]))

    def _env_goal(self) -> tuple[int, int]:
        return (int(self._env_cfg["goal"][0]), int(self._env_cfg["goal"][1]))

    def _env_flow(self) -> tuple[float, float]:
        return (
            float(self._env_cfg["flow"]["vector"][0]),
            float(self._env_cfg["flow"]["vector"][1]),
        )

    def _render_single_frame(self) -> None:
        start = self._env_start()
        goal = self._env_goal()
        flow = self._env_flow()
        state = self.control_panel.selection_state()

        if self._single_run is None:
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
            return

        visible_path = visible_path_for_frame(self._single_run.path, self._single_frame)
        self.single_canvas.draw_run(
            start=start,
            goal=goal,
            path=visible_path,
            flow_vector=flow,
            show_flow=state.show_flow,
            show_labels=state.show_labels,
            title_suffix=f"{self._single_run.title} | frame={len(visible_path)}/{len(self._single_run.path)}",
        )
        self.metrics_panel.update_metrics(format_metrics(self._single_run.run, self._records))

    def _render_compare_frame(self) -> None:
        start = self._env_start()
        goal = self._env_goal()
        flow = self._env_flow()
        state = self.control_panel.selection_state()

        if not self._compare_runs:
            self.compare_view.set_runs([])
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

    def _active_total_frames(self) -> int:
        if self.tabs.currentIndex() == 0:
            return 0 if self._single_run is None else len(self._single_run.path)
        return max_frame_count(self._compare_runs)

    def _start_animation(self) -> None:
        total_frames = self._active_total_frames()
        if total_frames <= 0:
            self._append_log("No rendered path available for animation.")
            return

        if self.tabs.currentIndex() == 0 and self._single_frame >= total_frames:
            self._single_frame = 0
        if self.tabs.currentIndex() == 1 and self._compare_frame >= total_frames:
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

    def _restart_animation_timer(self) -> None:
        speed = max(0.1, self.control_panel.selection_state().speed)
        interval_ms = max(40, int(300 / speed))
        self._animation_timer.start(interval_ms)

    def _advance_animation(self, step_only: bool = False) -> None:
        total_frames = self._active_total_frames()
        if total_frames <= 0:
            self._pause_animation(log_message=False)
            return

        if self.tabs.currentIndex() == 0:
            self._single_frame = min(self._single_frame + 1, total_frames)
            self._render_single_frame()
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
