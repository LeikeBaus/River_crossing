# Current State - River Crossing Project

## 1. Scope and Status

This document summarizes the currently implemented state as of April 2026.

Implemented and working:

- configuration-driven environment and experiment setup,
- strict in-grid dynamics with water-corridor and land exclusion,
- diagonal-only departure from the start and diagonal-only arrival at the goal,
- Dijkstra, A*, Weighted A*, Dynamic Programming, APF, and Q-learning,
- experiment runner and evaluator with JSON outputs,
- PyQt6 UI with Run, Best path, and Compare tabs,
- animation controls including Play, Pause, Step forward, Step reverse, and Reset,
- interactive flow controls via direction dial, strength slider, synchronized float inputs, and an impact control.

Still intentionally deferred:

- Export Report implementation,
- extended robustness-study workflows from inside the UI.

---

## 2. Functional Overview

### 2.1 Environment

The river-crossing environment currently supports:

- configurable grid size,
- constant flow vectors,
- configurable start and goal positions,
- a water corridor between start and goal columns,
- land tiles outside that corridor,
- diagonal-only start departure,
- diagonal-only final move into the goal.

A run is only successful if the vessel reaches the goal with a diagonal final move. The start must also be left via a diagonal move.

### 2.2 Cost Model

The optimization target remains

$$
c(s,a) = \alpha \cdot t(s,a) + \beta \cdot E(s,a)
$$

with Euclidean time cost and flow-dependent energy cost.

### 2.3 Algorithm State

All six planning approaches are implemented and available in the current configuration:

- Dijkstra
- A*
- Weighted A*
- Dynamic Programming
- APF
- Q-learning

Recent improvements:

- APF now uses guidance compatible with the diagonal arrival constraint and the valid approach corridor.
- Q-learning now uses reward shaping and loop reduction to avoid random wandering and repeated revisits.

---

## 3. Analysis Outputs

The result format has been extended to distinguish between final solution data and run-construction data.

Per-run records now include:

- path and actions,
- best_path and best_actions,
- run_trace for the Run tab,
- timing and quality metrics,
- reward_history and success_rate for reinforcement learning.

This allows the UI to cleanly separate:

- the run creation process,
- the final best path,
- and algorithm comparison.

---

## 4. UI State

The current PyQt6 UI includes:

- Run tab,
- Best path tab,
- Compare tab,
- metrics panel,
- Local 3x3 neighborhood panel with per-action costs,
- status log,
- PNG export,
- flow vector visualization with arrows,
- flow controls with bidirectional synchronization,
- toolbar-based actions for Run Single Experiment, Run Experiment Batch, Show Results, and Compare,
- Reset behavior returning the animation to frame 1.

Run view coloring is defined as:

- red = invalid action,
- yellow = valid but not selected,
- green = selected best action.

The left control panel now focuses on selection, flow tuning, and animation. Experiment execution and result-display actions are provided through the top toolbar only.

---

## 5. Verified State

The refreshed default experiment outputs confirm that all algorithms now solve the baseline scenario, including APF and Q-learning.

Qualitatively:

- Dijkstra, A*, Weighted A*, and Dynamic Programming remain the optimal references.
- APF now reaches the target while respecting the diagonal arrival constraint.
- Q-learning now converges to valid solutions, though it is still less cost-efficient than the optimal graph-search baselines.

---

## 6. Remaining Work

The remaining tasks are primarily product-completion items:

- implement report export,
- expand robustness and sensitivity-study tooling,
- deepen run-trace instrumentation if more algorithm introspection is needed.