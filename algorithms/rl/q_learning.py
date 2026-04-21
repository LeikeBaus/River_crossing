from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from algorithms.graph_search.dijkstra import PlanResult
from core.cost.cost_function import CostFunction
from core.environment.actions import Action
from core.environment.environment import RiverEnvironment
from core.environment.grid import State


def _available_actions(env: RiverEnvironment, state: State) -> tuple[Action, ...]:
    """Return valid RL actions from *state*.

    Actions that enter the goal with an invalid approach angle are excluded so
    the learned policy aligns with the hard docking constraint.
    """
    actions: list[Action] = []
    for action in env.valid_actions(state):
        next_state = env.transition(state, action)
        if next_state == env.goal and not env.is_goal(next_state, action):
            continue
        actions.append(action)
    return tuple(actions)


def _guidance_targets(env: RiverEnvironment) -> tuple[State, ...]:
    """Return the goal and valid one-step docking predecessors used for shaping."""
    targets: dict[State, None] = {env.goal: None}
    for action in env.actions:
        candidate = State(env.goal.i - action[0], env.goal.j - action[1])
        if not env.grid.contains(candidate):
            continue
        if env.transition(candidate, action) == env.goal and env.is_goal(env.goal, action):
            targets[candidate] = None
    return tuple(sorted(targets.keys(), key=lambda item: (item.i, item.j)))


def _guidance_distance(env: RiverEnvironment, state: State) -> float:
    """Distance to the goal or its valid docking corridor predecessors."""
    targets = _guidance_targets(env)
    return min(((state.i - target.i) ** 2 + (state.j - target.j) ** 2) ** 0.5 for target in targets)


@dataclass
class QLearningResult:
    """Result bundle produced by Q-learning training."""

    q_table: dict[State, dict[Action, float]] = field(default_factory=dict)
    policy: dict[State, Action] = field(default_factory=dict)
    reward_history: list[float] = field(default_factory=list)
    success_history: list[bool] = field(default_factory=list)
    steps_history: list[int] = field(default_factory=list)
    epsilon_history: list[float] = field(default_factory=list)
    episode_paths: list[list[State]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if not self.success_history:
            return 0.0
        return sum(1 for ok in self.success_history if ok) / len(self.success_history)


def q_learning_train(
    env: RiverEnvironment,
    cost_fn: CostFunction,
    episodes: int,
    learning_rate: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    max_steps_per_episode: int | None = None,
    seed: int | None = None,
    goal_reward: float = 25.0,
    progress_reward_scale: float = 1.5,
    revisit_penalty: float = 0.75,
    approach_bonus: float = 2.5,
    min_success_rate_for_decay: float = 0.0,
    success_window: int = 50,
) -> QLearningResult:
    """Train a tabular Q-learning agent.

    Reward is defined as ``r = -c(s, a)`` to match the project objective of
    minimizing traversal cost.

    Parameters
    ----------
    min_success_rate_for_decay:
        Minimum recent success rate required before epsilon is decayed.
        Set to e.g. 0.05 when opposing flow makes the goal hard to reach:
        exploration stays high until the agent finds the goal reliably,
        preventing premature exploitation of an empty Q-table.  Default 0.0
        preserves the original fixed-schedule behaviour.
    success_window:
        Number of recent episodes used to compute the success rate that
        gates epsilon decay.
    """
    if episodes <= 0:
        raise ValueError(f"episodes must be > 0, got {episodes}.")
    if not (0 < learning_rate <= 1.0):
        raise ValueError(f"learning_rate must be in (0,1], got {learning_rate}.")
    if not (0 < gamma <= 1.0):
        raise ValueError(f"gamma must be in (0,1], got {gamma}.")
    if epsilon_start < 0 or epsilon_end < 0:
        raise ValueError("epsilon values must be >= 0.")
    if epsilon_start < epsilon_end:
        raise ValueError("epsilon_start must be >= epsilon_end.")
    if not (0 < epsilon_decay <= 1.0):
        raise ValueError(f"epsilon_decay must be in (0,1], got {epsilon_decay}.")
    if goal_reward < 0 or progress_reward_scale < 0 or revisit_penalty < 0 or approach_bonus < 0:
        raise ValueError("reward-shaping parameters must be >= 0.")
    if not (0.0 <= min_success_rate_for_decay <= 1.0):
        raise ValueError(f"min_success_rate_for_decay must be in [0, 1], got {min_success_rate_for_decay}.")
    if success_window < 1:
        raise ValueError(f"success_window must be >= 1, got {success_window}.")

    if max_steps_per_episode is None:
        max_steps_per_episode = 4 * env.grid.nx * env.grid.ny
    if max_steps_per_episode <= 0:
        raise ValueError("max_steps_per_episode must be > 0.")

    rng = random.Random(seed)

    q_table: dict[State, dict[Action, float]] = {}

    def get_q(s: State, a: Action) -> float:
        return q_table.get(s, {}).get(a, 0.0)

    def set_q(s: State, a: Action, value: float) -> None:
        q_table.setdefault(s, {})[a] = value

    epsilon = epsilon_start
    reward_history: list[float] = []
    success_history: list[bool] = []
    steps_history: list[int] = []
    epsilon_history: list[float] = []
    episode_paths_list: list[list[State]] = []
    recent_successes: list[bool] = []

    for _ in range(episodes):
        state = env.start
        episode_reward = 0.0
        success = False
        steps = 0
        visited_counts: dict[State, int] = {state: 1}
        episode_path: list[State] = [state]

        for _step in range(max_steps_per_episode):
            steps += 1
            actions = _available_actions(env, state)
            if not actions:
                break

            action = _epsilon_greedy_action(q_table, state, actions, epsilon, rng)
            next_state = env.transition(state, action)

            current_distance = _guidance_distance(env, state)
            next_distance = _guidance_distance(env, next_state)
            reward = -cost_fn(state, action)
            reward += progress_reward_scale * (current_distance - next_distance)
            reward -= revisit_penalty * visited_counts.get(next_state, 0)
            if next_state != env.goal and next_distance < 1e-9:
                reward += approach_bonus

            done = next_state == env.goal
            if done:
                reward += goal_reward
                success = True

            episode_reward += reward
            visited_counts[next_state] = visited_counts.get(next_state, 0) + 1

            next_actions = _available_actions(env, next_state)
            max_next_q = max((get_q(next_state, a2) for a2 in next_actions), default=0.0)

            old_q = get_q(state, action)
            target = reward + gamma * (0.0 if done else max_next_q)
            new_q = old_q + learning_rate * (target - old_q)
            set_q(state, action, new_q)

            state = next_state
            episode_path.append(next_state)
            if done:
                break

        reward_history.append(episode_reward)
        success_history.append(success)
        steps_history.append(steps)
        epsilon_history.append(epsilon)
        episode_paths_list.append(episode_path)

        # Adaptive epsilon decay: only reduce exploration once the agent is
        # finding the goal often enough.  With min_success_rate_for_decay=0.0
        # (default) this always decays, matching the original behaviour.
        recent_successes.append(success)
        if len(recent_successes) > success_window:
            recent_successes.pop(0)
        recent_rate = sum(recent_successes) / len(recent_successes)
        if recent_rate >= min_success_rate_for_decay:
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

    policy = extract_policy(env, q_table)
    return QLearningResult(
        q_table=q_table,
        policy=policy,
        reward_history=reward_history,
        success_history=success_history,
        steps_history=steps_history,
        epsilon_history=epsilon_history,
        episode_paths=episode_paths_list,
    )


def extract_policy(
    env: RiverEnvironment,
    q_table: dict[State, dict[Action, float]],
) -> dict[State, Action]:
    """Extract greedy policy from a Q-table for all reachable states."""
    policy: dict[State, Action] = {}
    for i in range(env.grid.nx):
        for j in range(env.grid.ny):
            s = State(i, j)
            actions = _available_actions(env, s)
            if not actions:
                continue
            best_action = max(actions, key=lambda a: q_table.get(s, {}).get(a, 0.0))
            policy[s] = best_action
    return policy


def rollout_policy(
    env: RiverEnvironment,
    cost_fn: CostFunction,
    policy: dict[State, Action],
    max_steps: int | None = None,
) -> PlanResult:
    """Run one deterministic inference rollout of a learned policy."""
    if max_steps is None:
        max_steps = 4 * env.grid.nx * env.grid.ny
    if max_steps <= 0:
        raise ValueError("max_steps must be > 0.")

    state = env.start
    path: list[State] = [state]
    actions: list[Action] = []
    total_cost = 0.0
    visited_counts: dict[State, int] = {state: 1}

    for _ in range(max_steps):
        if state == env.goal:
            return PlanResult(
                path=path,
                actions=actions,
                total_cost=total_cost,
                nodes_expanded=len(actions),
                found=True,
            )

        action = policy.get(state)
        if action is None:
            break

        next_state = env.transition(state, action)
        if next_state == env.goal and not env.is_goal(next_state, action):
            break
        if visited_counts.get(next_state, 0) >= 2:
            break

        total_cost += cost_fn(state, action)
        actions.append(action)
        path.append(next_state)
        visited_counts[next_state] = visited_counts.get(next_state, 0) + 1
        state = next_state

    return PlanResult(
        path=path,
        actions=actions,
        total_cost=total_cost,
        nodes_expanded=len(actions),
        found=(state == env.goal),
    )


def q_learning_from_config(
    env: RiverEnvironment,
    cost_fn: CostFunction,
    q_config: dict[str, Any],
    episodes: int,
    seed: int | None = None,
    max_steps_per_episode: int | None = None,
) -> QLearningResult:
    """Train Q-learning from ``q_learning`` config section."""
    cfg_max_steps = q_config.get("max_steps_per_episode")
    if max_steps_per_episode is None and cfg_max_steps is not None:
        max_steps_per_episode = int(cfg_max_steps)
    return q_learning_train(
        env=env,
        cost_fn=cost_fn,
        episodes=episodes,
        learning_rate=float(q_config.get("learning_rate", 0.1)),
        gamma=float(q_config.get("gamma", 0.99)),
        epsilon_start=float(q_config.get("epsilon_start", 1.0)),
        epsilon_end=float(q_config.get("epsilon_end", 0.05)),
        epsilon_decay=float(q_config.get("epsilon_decay", 0.995)),
        max_steps_per_episode=max_steps_per_episode,
        seed=seed,
        goal_reward=float(q_config.get("goal_reward", 25.0)),
        progress_reward_scale=float(q_config.get("progress_reward_scale", 1.5)),
        revisit_penalty=float(q_config.get("revisit_penalty", 0.75)),
        approach_bonus=float(q_config.get("approach_bonus", 2.5)),
        min_success_rate_for_decay=float(q_config.get("min_success_rate_for_decay", 0.0)),
        success_window=int(q_config.get("success_window", 50)),
    )


def _epsilon_greedy_action(
    q_table: dict[State, dict[Action, float]],
    state: State,
    actions: tuple[Action, ...],
    epsilon: float,
    rng: random.Random,
) -> Action:
    if rng.random() < epsilon:
        return rng.choice(actions)

    q_state = q_table.get(state, {})
    best_q = max(q_state.get(a, 0.0) for a in actions)
    best_actions = [a for a in actions if q_state.get(a, 0.0) == best_q]
    return rng.choice(best_actions)