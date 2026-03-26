# Current State - River Crossing Project

## 1. Scope and Status

This document describes the **currently implemented state** of the project.

Implemented so far:

- Step 1: Project/config scaffold and config loading.
- Step 2: Core environment model (grid, state, action model, flow abstraction, docking-angle constraint).
- Step 3: Cost model (`time + energy`) and hard goal-angle validation.
- Step 4: Reference planners: Dijkstra + Dynamic Programming (Value Iteration).
- Step 5: Heuristic planners: A* + Weighted A*.
- Step 6: Artificial Potential Field (APF).
- Step 7: Reinforcement Learning: tabular Q-learning.
- Step 8: Experiment runner + evaluator pipeline with JSON outputs.

Not implemented yet:

- Step 9+ (notably visualization and final analysis/reporting pipeline extensions).
- `ui/visualization.py` is still a placeholder.

---

## 2. Configuration-Driven Architecture

The project is configuration-first. Core runtime behavior is controlled by:

- `configs/env.yaml`
- `configs/algorithm.yaml`
- `configs/experiment.yaml`

### 2.1 Environment Config (`env.yaml`)

Defines state space and scenario:

- `grid.nx`, `grid.ny`: grid width/height.
- `flow.type`, `flow.vector`: flow model (currently constant vector only).
- `start`, `goal`: start and target state.
- `docking.normal`: docking normal vector.
- `docking.approach_angle_deg.min/max`: required final approach angle range.

### 2.2 Algorithm Config (`algorithm.yaml`)

Defines shared and algorithm-specific hyperparameters:

- `common.alpha`, `common.beta`: global cost weights.
- `a_star.heuristic`: currently `euclidean`.
- `weighted_a_star.weight`: heuristic inflation factor `w >= 1`.
- `dynamic_programming.gamma`, `tolerance`, `max_iterations`.
- `apf.k_att`, `apf.lambda_flow`.
- `q_learning.learning_rate`, `gamma`, `epsilon_start`, `epsilon_end`, `epsilon_decay`.

### 2.3 Experiment Config (`experiment.yaml`)

Defines batch execution dimensions:

- `seeds`: random seeds for repeated runs.
- `environments`: named environment definitions.
- `algorithms`: algorithm IDs to execute.
- `metrics`: target metrics to report (currently interpreted by evaluator/runner fields).

### 2.4 Config Loader

`core/config_loader.py` provides:

- `load_env_config(...)`
- `load_algorithm_config(...)`
- `load_experiment_config(...)`
- `load_all_configs(...)`

It validates required keys and basic constraints (e.g. positive grid sizes, weighted A* weight bound).

---

## 3. Core Domain Model

## 3.1 State and Grid

File: `core/environment/grid.py`

- `State(i, j)`: immutable discrete coordinate.
- `Grid(nx, ny)`:
  - `contains(state)` for bounds checks.
  - `clip(state)` exists, but the transition model now uses **strict in-bounds actions** instead of clipped transitions.

## 3.2 Action Space and Boundary Semantics

File: `core/environment/actions.py`

- `ACTIONS`: 8-neighborhood actions.
- `is_action_valid(state, action, grid)`: true if next state remains in-grid.
- `apply_action(...)`: returns exact next state, raises `ValueError` if action leaves grid.

Important semantic decision:

- The vessel is **not allowed to leave the grid**.
- Algorithms should iterate via `env.valid_actions(state)`.

## 3.3 Flow Model

File: `core/dynamics/flow.py`

- Abstract `FlowField.at(state)`.
- Implemented model: `ConstantFlow(vi, vj)`.
- Factory: `make_flow(config)` with `type: constant`.

## 3.4 Environment and Goal Constraint

File: `core/environment/environment.py`

`RiverEnvironment` encapsulates:

- Grid, flow, start/goal, docking config.
- `actions`: full action set.
- `valid_actions(state)`: boundary-safe action subset.
- `transition(state, action)`: strict transition (raises on invalid move).
- `flow_at(state)`.
- `is_goal(state, last_action)`: requires state == goal **and** final approach angle valid.
- `approach_angle_deg(action)`: computes angle between last action and docking normal.

### Docking Constraint Meaning

Goal is not purely positional. A run only truly reaches goal if final action satisfies:

$$
\theta_{min} \leq \theta(a_{final}, n_{dock}) \leq \theta_{max}
$$

---

## 4. Cost Model

File: `core/cost/cost_function.py`

## 4.1 Components

- `TimeCost` interface.
- `EnergyCost` interface.
- `EuclideanTimeCost`: $t(s,a)=||a||_2$.
- `FlowEnergyCost`: $E(s,a)=||a-u_{flow}(s)||_2^2$.

## 4.2 Combined Cost

`CostFunction(alpha, beta, time_cost, energy_cost)` computes:

$$
c(s,a) = \alpha \cdot t(s,a) + \beta \cdot E(s,a)
$$

Also exposes `time(...)` and `energy(...)` for component-level access.

Config factory:

- `CostFunction.from_config(algo_config, flow)` uses `common.alpha`, `common.beta`.

### Hyperparameter Meaning

- `alpha`: emphasizes traversal time/step length.
- `beta`: emphasizes effort against flow.

---

## 5. Implemented Algorithms

All planners operate on `RiverEnvironment` and consume `CostFunction`.

## 5.1 Dijkstra (Reference Optimum)

File: `algorithms/graph_search/dijkstra.py`

Core function:

- `dijkstra(env, cost_fn) -> PlanResult`

How it works:

- Standard uniform-cost graph search on grid states.
- Expands by current lowest accumulated `g` cost.
- Ignores transitions that enter goal with invalid approach angle.
- Returns `PlanResult(path, actions, total_cost, nodes_expanded, found)`.

Hyperparameters:

- None beyond environment + cost model.

## 5.2 Dynamic Programming (Value Iteration)

File: `algorithms/graph_search/dynamic_programming.py`

Core functions:

- `value_iteration(...) -> ValueIterationResult`
- `extract_path(env, policy, max_steps=None)`

How it works:

- Iterates Bellman backups over all states.
- Terminal value fixed at goal (`V(goal)=0`).
- Invalid goal-entry actions are skipped.
- Returns converged value function + greedy policy.

Hyperparameters:

- `gamma`: discount factor.
- `tolerance`: convergence threshold on max update delta.
- `max_iterations`: hard cap on sweeps.

Meaning:

- Higher `gamma` values weight long-term cost more strongly.
- Lower `tolerance` demands stricter convergence.

## 5.3 A* and Weighted A*

File: `algorithms/graph_search/heuristic_search.py`

Core functions:

- `astar(env, cost_fn, heuristic='euclidean')`
- `weighted_astar(env, cost_fn, weight=1.5, heuristic='euclidean')`
- Config wrappers and sweep helper.

How they work:

- Priority function:
  - A*: $f=g+h$
  - Weighted A*: $f=g+w\cdot h$
- Uses Euclidean heuristic scaled by `alpha` to remain conservative relative to cost definition.
- Respects valid actions and docking-angle goal validity.

Hyperparameters:

- `heuristic`: currently only `euclidean`.
- `weight` (`w >= 1`): search aggressiveness.

Meaning:

- `w=1`: standard A* (optimal if heuristic admissible).
- Larger `w`: faster/bias toward goal, potentially less optimal path.

## 5.4 Artificial Potential Field (APF)

File: `algorithms/graph_search/apf.py`

Core functions:

- `apf_plan(...) -> APFResult`
- `apf_from_config(...)`
- potential utilities (`attractive_potential`, `flow_potential`, etc.)

How it works:

- Defines potential

$$
\Phi(s) = \Phi_{att}(s) + \Phi_{flow}(s)
$$

with:

- attractive term toward goal,
- flow term from local flow vector.

Action policy:

- Chooses discrete action that minimizes directional derivative $\langle a, \nabla\Phi \rangle$.

Robustness behavior:

- Detects local minima/cycles.
- Returns explicit failure with `local_minimum_detected=True` when stuck.

Hyperparameters:

- `k_att > 0`: strength of attraction to goal.
- `lambda_flow >= 0`: strength of flow influence.
- `max_steps` optional safety bound.

Meaning:

- Higher `k_att`: stronger pull to target.
- Higher `lambda_flow`: more flow-following behavior.

## 5.5 Q-Learning (Tabular RL)

File: `algorithms/rl/q_learning.py`

Core functions:

- `q_learning_train(...) -> QLearningResult`
- `extract_policy(...)`
- `rollout_policy(...)`
- `q_learning_from_config(...)`

How it works:

- Tabular Q update:

$$
Q(s,a) \leftarrow Q(s,a) + \eta\left[r + \gamma\max_{a'}Q(s',a') - Q(s,a)\right]
$$

- Reward is `r = -cost_fn(s,a)` (min-cost objective mapped to max-reward RL).
- Epsilon-greedy exploration with multiplicative decay.
- Trains only on valid actions, excluding invalid goal arrivals.

Produced outputs:

- `q_table`
- greedy `policy`
- histories: reward/success/steps/epsilon
- derived `success_rate`

Hyperparameters:

- `episodes`: number of training episodes.
- `learning_rate` (`eta`): update speed.
- `gamma`: discount factor.
- `epsilon_start`: initial exploration probability.
- `epsilon_end`: minimum exploration floor.
- `epsilon_decay`: per-episode decay factor.
- `max_steps_per_episode`: truncation bound.
- `seed`: reproducibility control.

Meaning:

- Larger `learning_rate`: faster but less stable updates.
- Larger `gamma`: more far-sighted behavior.
- Higher/longer epsilon: more exploration before exploitation.

---

## 6. Shared Result Types

## 6.1 `PlanResult`

Used by graph/planning modules and policy rollouts:

- `path`
- `actions`
- `total_cost`
- `nodes_expanded`
- `found`

## 6.2 `APFResult`

Extends `PlanResult` with:

- `termination_reason`
- `local_minimum_detected`

## 6.3 `QLearningResult`

Training-focused bundle:

- learned `q_table`, `policy`
- history arrays
- `success_rate`

---

## 7. Experiment Runner

File: `experiments/runner.py`

## 7.1 Purpose

Batch orchestration over:

- environments
- algorithm list
- random seeds

It executes each combination and emits normalized run records.

## 7.2 Core Functions

- `run_all_experiments(config_dir='configs', output_path=None, rl_episodes=None)`
- `save_experiment_results(...)`
- `load_experiment_results(...)`

Internal dispatch (`_run_single_experiment`) handles algorithm-specific execution:

- Graph/APF methods: planning time only.
- Q-learning: separate training and inference timing.

## 7.3 Result Schema (`ExperimentRunRecord`)

Each run includes:

- identity: `environment_name`, `algorithm`, `seed`
- success: `found`, `angle_valid`
- quality: `total_cost`, `path_length`, `steps`
- performance: `plan_time`, `training_time`, `inference_time`, `total_time`
- search effort: `nodes_expanded`
- trajectories: serialized `path`, `actions`
- RL diagnostics: `reward_history`, `success_rate`

---

## 8. Evaluator

File: `experiments/evaluator.py`

## 8.1 Purpose

Post-processes raw run records into:

- per-run annotated metrics (notably `delta_j`)
- per-algorithm aggregate summaries

## 8.2 Core Function

- `evaluate_results(results, output_path=None, reference_algorithm='dijkstra')`

Input can be:

- in-memory `ExperimentRunRecord` list,
- list of dicts,
- path to saved JSON results.

## 8.3 Computed Metrics

Per run:

- `delta_j = total_cost_algo - total_cost_reference` for same `(environment_name, seed)`.

Per algorithm summary:

- run counts and solved counts,
- success rate,
- means of total cost, delta J, steps,
- means of plan/training/inference/total time,
- docking-angle validity rate.

---

## 9. Entry Points and Package Exports

- `main.py`: currently only validates/prints loaded configs.
- `algorithms/graph_search/__init__.py`: exports all graph/APF APIs.
- `algorithms/rl/__init__.py`: exports Q-learning APIs.

---

## 10. Testing Status and Coverage Snapshot

Implemented unit-test modules cover:

- config loading,
- environment/actions/flow,
- cost model + angle constraints,
- reference algorithms (Dijkstra/DP consistency),
- heuristic algorithms (A*/Weighted A*),
- APF behavior and local-minimum handling,
- Q-learning training/policy helpers,
- experiment runner and evaluator.

`test_scaffold_contracts.py` currently confirms scaffold files and that visualization remains a placeholder.

---

## 11. Known Gaps / Next Work

- Visualization (`ui/visualization.py`) is not implemented.
- Analysis/reporting artifacts and richer plotting are pending later steps.
- `Grid.clip(...)` remains available as utility but is not used by active transition semantics.

This state is suitable for running reproducible algorithm batches and producing structured evaluation outputs for further analysis.

---

## 12. Quick Start (Commands)

Run these commands from the project root.

### 12.1 Activate Virtual Environment

Git Bash (Windows):

```bash
source .venv/Scripts/activate
```

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 12.2 Install Dependencies

```bash
pip install -r requirements.txt
```

### 12.3 Sanity Check Config Loading

```bash
python main.py
```

### 12.4 Run Unit Tests

Run all currently available test modules:

```bash
python -m test.test_config_loader
python -m test.test_environment
python -m test.test_cost
python -m test.test_reference_algorithms
python -m test.test_heuristic_algorithms
python -m test.test_apf
python -m test.test_q_learning
python -m test.test_experiments
python -m test.test_scaffold_contracts
```

### 12.5 Run Full Experiment Batch and Save Raw Results

```bash
python -c "from experiments.runner import run_all_experiments; run_all_experiments(config_dir='configs', output_path='analysis/raw_results.json')"
```

### 12.6 Evaluate Raw Results and Save Summary

```bash
python -c "from experiments.evaluator import evaluate_results; evaluate_results('analysis/raw_results.json', output_path='analysis/evaluation_summary.json', reference_algorithm='dijkstra')"
```

### 12.7 Optional: Run Fewer RL Episodes for Faster Batch Turnaround

```bash
python -c "from experiments.runner import run_all_experiments; run_all_experiments(config_dir='configs', output_path='analysis/raw_results_fast.json', rl_episodes=50)"
```

### 12.8 Inspect Output Files

Raw run records:

- `analysis/raw_results.json`

Aggregated evaluation:

- `analysis/evaluation_summary.json`
