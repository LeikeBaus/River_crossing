# Current State - River Crossing Project

## 1. Scope and Status

This document summarizes the currently implemented state as of April 2026.

Implemented and working:

- configuration-driven environment and experiment setup,
- strict in-grid dynamics with docking-angle goal validation,
- Dijkstra, A*, Weighted A*, Dynamic Programming, APF, and Q-learning,
- experiment runner and evaluator with JSON outputs,
- PyQt6 UI with Run, Best path, and Compare tabs,
- animation controls including Play, Pause, Step, and Reset,
- interactive flow controls via direction dial, strength slider, and synchronized float inputs.

Still intentionally deferred:

- Run Batch execution directly from the UI button,
- Export Report implementation,
- extended robustness-study workflows from inside the UI.

---

## 2. Functional Overview

### 2.1 Environment

The river-crossing environment currently supports:

- configurable grid size,
- constant flow vectors,
- configurable start and goal positions,
- a hard docking-angle constraint on the final move.

A run is only successful if the vessel reaches the goal with an admissible final approach angle.

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

- APF now uses docking-aware guidance so it can enter the valid approach corridor instead of failing at the goal boundary.
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
- status log,
- PNG export,
- flow vector visualization,
- flow controls with bidirectional synchronization,
- Reset behavior returning the animation to frame 1.

Run view coloring is defined as:

- red = invalid action,
- yellow = valid but not selected,
- green = selected best action.

The Run Batch button remains a deliberate placeholder for now.

---

## 5. Verified State

The refreshed default experiment outputs confirm that all algorithms now solve the baseline scenario, including APF and Q-learning.

Qualitatively:

- Dijkstra, A*, Weighted A*, and Dynamic Programming remain the optimal references.
- APF now reaches the target while respecting the docking constraint.
- Q-learning now converges to valid solutions, though it is still less cost-efficient than the optimal graph-search baselines.

---

## 6. Remaining Work

The remaining tasks are primarily product-completion items:

- wire actual batch execution into the UI,
- implement report export,
- expand robustness and sensitivity-study tooling,
- deepen run-trace instrumentation if more algorithm introspection is needed.