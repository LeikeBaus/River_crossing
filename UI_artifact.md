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
- running a single experiment from the toolbar,
- running an experiment batch from the toolbar (full parameter sweep over all combinations in `configs/sweep.yaml`; cached results are skipped),
- showing the selected result record from the toolbar,
- selecting environment, algorithm, and seed,
- animating trajectories,
- stepping trajectories forwards and backwards,
- resetting the animation to frame 1,
- comparing best paths across algorithms,
- inspecting run creation step by step,
- interactively changing the flow direction via radio buttons (top→bottom / bottom→top),
- interactively changing the flow strength via slider,
- interactively changing Gaussian flow profile parameters (sigma and floor) when the Gaussian environment type is selected,
- interactively changing heading memory depth via the inertia spinbox (0–5),
- interactively changing the Q-learning episode count via the QL episodes combo (200 / 500 / 1500),
- interactively changing flow impact in percent,
- inspecting a Local 3x3 neighborhood with action costs.

## 4. Current Layout
### 4.1 Main Window
- Top toolbar actions:
  - Load Results
   - Run Single Experiment
   - Run Experiment Batch
   - Show Results
  - Compare
  - Export PNG
  - Export Report

### 4.2 Left Control Panel
- Environment selector
- Algorithm selector
- Seed selector
- Animation speed control
- Inertia spinbox (0–5)
- QL episodes combo (200 / 500 / 1500)
- Flow direction radio buttons (↓ Top → Bottom / ↑ Bottom → Top)
- Flow strength slider
- Flow sigma spinbox (visible for Gaussian environment only)
- Flow floor spinbox (visible for Gaussian environment only)
- Impact spinbox
- Show flow checkbox
- Show labels checkbox
- Play button
- Pause button
- Step forward button
- Step reverse button
- Reset button

### 4.3 Flow Controls
The flow section contains:

- two QRadioButton controls for flow direction:
  - ↓ Top → Bottom (flow vector (0, +1)),
  - ↑ Bottom → Top (flow vector (0, −1)).
- a horizontal slider for flow strength (scales the direction vector magnitude).
- a QDoubleSpinBox for Gaussian sigma (width of the fast-lane profile in columns), visible only when the environment type is "gaussian".
- a QDoubleSpinBox for Gaussian floor (minimum flow fraction at the shores), visible only when the environment type is "gaussian".
- an impact spinbox displayed as 0% to 100%, mapped internally to the float parameter β.
- a QL episodes QComboBox with options 200, 500, and 1500, controlling the number of training episodes used by Q-learning during single or batch runs.

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
- Local 3x3 neighborhood cells with action coloring
- per-cell time, energy, and total costs for valid actions

### 4.6 Status Area
A bottom dock logs load events, worker messages, warnings, and UI status changes.

## 5. Run View Semantics
The Run tab uses color-coded overlays to show stepwise decision quality:

- red = invalid action,
- yellow = valid but not selected,
- green = selected best action.

This view is intended to make the search or rollout process interpretable rather than only showing the final result.

Additionally, base-cell colors distinguish water and land, and flow vectors are rendered as arrows only inside the valid water corridor.

## 6. Current Limitations
The following items remain intentionally incomplete:

1. Export Report is still a placeholder.
2. The Run view currently derives its visualization from stored run traces and best-action information rather than launching a live algorithm debugger.

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
4. flow direction radio buttons switch the flow vector between top→bottom and bottom→top,
5. impact is shown in percent and applied internally as β,
6. Compare uses best-path data,
7. the Run tab shows action coloring,
8. the Local 3x3 panel reflects local action costs,
9. the inertia spinbox updates the heading-memory depth used by all compatible algorithms,
10. Gaussian profile controls (sigma, floor) appear only when the Gaussian environment type is selected,
11. the QL episodes combo sets the Q-learning training depth and is included in the experiment ID and sweep.
