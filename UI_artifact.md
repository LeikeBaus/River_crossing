# UI Artifact - River Crossing (PyQt6)

## 1. Purpose
This document describes the current UI artifact that is now implemented in the project and serves as the documentation baseline for the desktop interface.

## 2. Core UI Decisions
The interface currently uses:

1. QGraphicsView and QPainter for grid rendering.
2. QThread worker objects with Qt signals for background UI tasks.
3. Native platform styling.
4. Tab-based navigation for the three main analysis modes.
5. PNG export as the primary output format.

## 3. Current User-Facing Features
The application supports:

- loading stored experiment results,
- selecting environment, algorithm, and seed,
- animating trajectories,
- resetting the animation to frame 1,
- comparing best paths across algorithms,
- inspecting run creation step by step,
- interactively changing the displayed flow vector.

## 4. Current Layout
### 4.1 Main Window
- Top toolbar actions:
  - Load Results
  - Run Single
  - Run Batch
  - Compare
  - Export PNG
  - Export Report

### 4.2 Left Control Panel
- Environment selector
- Algorithm selector
- Seed selector
- Animation speed control
- Show flow checkbox
- Show labels checkbox
- Play button
- Pause button
- Step button
- Reset button

### 4.3 Flow Controls
The flow section contains:

- a QDial for choosing direction,
- a horizontal slider for flow strength,
- three synchronized float line edits for:
  - x-direction,
  - y-direction,
  - strength.

All controls update each other bidirectionally.

### 4.4 Center Tabs
The center panel contains three tabs:

1. Run
   - shows the progressive creation of a path,
   - highlights action categories with color.

2. Best path
   - shows the final chosen path for one selected run,
   - supports animation over the best trajectory.

3. Compare
   - shows best-path comparison across algorithms for the same scenario.

### 4.5 Metrics Panel
The right side displays:

- total_cost
- delta_j
- steps
- plan_time
- training_time
- inference_time
- angle_valid

### 4.6 Status Area
A bottom dock logs load events, worker messages, warnings, and UI status changes.

## 5. Run View Semantics
The Run tab uses color-coded overlays to show stepwise decision quality:

- red = invalid action,
- yellow = valid but not selected,
- green = selected best action.

This view is intended to make the search or rollout process interpretable rather than only showing the final result.

## 6. Current Limitations
The following items remain intentionally incomplete:

1. Run Batch is still a placeholder in the UI.
2. Export Report is still a placeholder.
3. The Run view currently derives its visualization from stored run traces and best-action information rather than launching a live algorithm debugger.

## 7. Data Contract with Analysis Files
The UI now relies on a richer run record format that distinguishes between:

- best_path and best_actions,
- path and actions for compatibility,
- run_trace for stepwise rendering,
- timing and quality metrics,
- reward history and success metrics for RL.

This separation enables the three-tab design without overloading a single generic path field.

## 8. Acceptance State
The current UI artifact is considered achieved when:

1. the application launches from the command line,
2. the three tabs are visible as Run, Best path, and Compare,
3. Reset returns the visualization to frame 1,
4. flow controls stay synchronized,
5. Compare uses best-path data,
6. the Run tab shows action coloring.
