# UI Artifact - River Crossing (PyQt6)

## 1. Purpose
Define a concrete PyQt6 rewrite plan for the project UI. This artifact serves as implementation target, review checklist, and acceptance baseline.

## 2. Fixed Decisions
These decisions are final and should not be revisited during M1-M3:

1. Rendering backend: QGraphicsView + QPainter.
2. Concurrency model: QThread worker objects with Qt signals.
3. Style system: native platform style (no global custom theme in M1).
4. Comparison mode default: tab-based views.
5. Export priority: PNG first (SVG optional later).

## 3. Product Goals
- Replace text UI with a desktop UI.
- Visualize river grid, flow vectors, start/goal, and trajectories.
- Support single-run and multi-algorithm comparison workflows.
- Keep domain logic independent from UI classes.

## 4. Main User Workflows
1. Open app and load config context.
2. Choose environment, algorithm, seed.
3. Trigger single run and inspect trajectory + metrics.
4. Switch to compare tab and compare algorithms for same scenario.
5. Export current visualization to PNG.

## 5. Target Layout
## 5.1 Main Window
- Top toolbar actions:
  - Load Config
  - Run Single
  - Run Batch
  - Compare
  - Export PNG
  - Export Report
- Left control panel:
  - Environment selector
  - Algorithm selector
  - Seed selector
  - Show flow checkbox
  - Show labels checkbox
  - Animation controls (play, pause, step, speed)
- Center panel:
  - Tab widget with:
    - Single View tab (single canvas)
    - Compare View tab (multi-canvas comparison grid)
- Right metrics panel:
  - total_cost
  - delta_j
  - steps
  - plan_time
  - training_time
  - inference_time
  - angle_valid
- Bottom dock:
  - status log (append-only)

## 6. Milestone Status
M1 delivered the shell and wiring.

M2 delivered:
1. Single-run rendering from stored run records.
2. Metrics panel population.
3. Results loading and PNG export.

M3 delivers:
1. Compare tab with multiple algorithm canvases.
2. Play, pause, and step controls.
3. Synchronized frame-based animation in Single and Compare tabs.
4. Redraw behavior when display toggles change.

Still deferred after M3:
1. Running simulations directly from UI buttons.
2. Export report implementation.
3. Advanced compare interactions such as per-canvas selection/focus.

## 7. Architecture
- ui/visualization.py:
  - PyQt6 widgets, single-run rendering, compare grid, and animation flow.
  - Main window orchestration and placeholder interactions.
- main.py:
  - Command-line entrypoint with `ui` command to launch desktop app.
- experiments/, algorithms/, core/:
  - unchanged domain/services consumed by later UI stages.

Separation rule:
- UI never mutates core algorithm code.
- Data exchange through stable dict-based run records and config objects.

## 8. Concurrency Contract (QThread)
- Long-running tasks execute in worker QObject moved to QThread.
- Worker emits:
  - progress(int)
  - message(str)
  - result(dict)
  - error(str)
  - finished()
- Main window updates controls and logs only from the UI thread.

## 9. Error Handling Policy
- Invalid file/config paths: non-blocking dialog + status log message.
- Missing run selections: clear warning in status area.
- Worker exceptions: catch in worker, emit error signal, keep app alive.

## 10. Test Strategy
Current focus through M3:
1. CLI contract for launching UI command.
2. Pure helper logic in UI module (formatting, path coercion, animation frames, run selection).
3. Scaffold contract checks for PyQt6 UI entrypoints.

## 11. Definition of Done
M3 is done when:
1. Running `python main.py ui` opens the native-style app shell.
2. Single tab renders one selected run with metrics.
3. Compare tab renders multiple algorithms for one scenario.
4. Play, pause, and step animate both tabs.
5. Updated tests and docs reflect the PyQt6 direction.
