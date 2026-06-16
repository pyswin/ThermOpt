from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from thermopt.layout.objects import FloorplanCase, Layout
from thermopt.objective.cost import CostResult
from thermopt.optimizer.rl_environment import ThermalFloorplanEnv


@dataclass(frozen=True)
class RLResult:
    best_layout: Layout
    best_cost: CostResult
    best_curve: list[float]
    episode_returns: list[float]
    policy_weights: np.ndarray
    training_episodes: int
    rollout_steps: int


def optimize(
    case: FloorplanCase,
    initial_layout: Layout,
    objective: Callable[[Layout], CostResult],
    config: dict,
    seed: int,
) -> RLResult:
    rng = np.random.default_rng(seed)
    episodes = max(1, int(config.get("episodes", 80)))
    max_steps = max(1, int(config.get("max_steps", 80)))
    rollout_steps = max(1, int(config.get("rollout_steps", max_steps)))
    learning_rate = float(config.get("learning_rate", 0.035))
    gamma = float(config.get("gamma", 0.97))
    entropy_coef = float(config.get("entropy_coef", 0.01))
    epsilon = float(config.get("epsilon", 0.08))
    move_scale = float(config.get("move_scale", 10.0))
    accept_worse_probability = float(config.get("accept_worse_probability", 0.02))
    log_every = max(1, int(config.get("log_every", 10)))
    verbose = bool(config.get("verbose", False))

    env = ThermalFloorplanEnv(
        case,
        initial_layout,
        objective,
        max_steps=max_steps,
        move_scale=move_scale,
        accept_worse_probability=accept_worse_probability,
        seed=seed,
    )
    feature_dim = _features(env.reset(), max_steps).shape[0]
    action_count = len(env.action_names)
    weights = rng.normal(0.0, 0.01, size=(feature_dim, action_count))

    initial_cost = objective(initial_layout)
    best_layout = initial_layout
    best_cost = initial_cost
    best_curve = [best_cost.total]
    episode_returns: list[float] = []
    baseline = 0.0

    for episode in range(episodes):
        observation = env.reset(seed=seed + episode + 1)
        features: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []

        for _ in range(max_steps):
            feature = _features(observation, max_steps)
            probs = _policy(feature, weights)
            if rng.random() < epsilon:
                action = int(rng.integers(0, action_count))
            else:
                action = int(rng.choice(action_count, p=probs))
            step = env.step(action)
            features.append(feature)
            actions.append(action)
            rewards.append(step.reward)
            observation = step.observation
            if env.cost.total < best_cost.total:
                best_layout = env.layout
                best_cost = env.cost
            if step.done:
                break

        returns = _discounted_returns(rewards, gamma)
        total_return = float(sum(rewards))
        episode_returns.append(total_return)
        if returns.size:
            baseline = 0.9 * baseline + 0.1 * float(np.mean(returns))
            advantages = returns - baseline
            std = float(np.std(advantages))
            if std > 1e-9:
                advantages = advantages / std
            for feature, action, advantage in zip(features, actions, advantages):
                probs = _policy(feature, weights)
                grad = -probs
                grad[action] += 1.0
                entropy_grad = -np.log(np.clip(probs, 1e-9, 1.0)) - 1.0
                weights += learning_rate * (np.outer(feature, advantage * grad) + entropy_coef * np.outer(feature, entropy_grad))

        best_curve.append(best_cost.total)
        if verbose and ((episode + 1) % log_every == 0 or episode == 0 or episode + 1 == episodes):
            print(
                f"[rl] episode={episode + 1}/{episodes} "
                f"return={total_return:.4f} best_cost={best_cost.total:.4f}"
            )

    rollout_layout, rollout_cost, rollout_curve = _greedy_rollout(
        case, initial_layout, objective, weights, rollout_steps, move_scale, accept_worse_probability, seed + 10_000
    )
    if rollout_cost.total < best_cost.total:
        best_layout = rollout_layout
        best_cost = rollout_cost
    best_curve.extend(rollout_curve)

    return RLResult(
        best_layout=best_layout,
        best_cost=best_cost,
        best_curve=best_curve,
        episode_returns=episode_returns,
        policy_weights=weights,
        training_episodes=episodes,
        rollout_steps=rollout_steps,
    )


def _greedy_rollout(
    case: FloorplanCase,
    initial_layout: Layout,
    objective: Callable[[Layout], CostResult],
    weights: np.ndarray,
    steps: int,
    move_scale: float,
    accept_worse_probability: float,
    seed: int,
) -> tuple[Layout, CostResult, list[float]]:
    env = ThermalFloorplanEnv(
        case,
        initial_layout,
        objective,
        max_steps=steps,
        move_scale=move_scale,
        accept_worse_probability=accept_worse_probability,
        seed=seed,
    )
    observation = env.reset(seed=seed)
    best_layout = env.layout
    best_cost = env.cost
    curve = [best_cost.total]
    for _ in range(steps):
        action = int(np.argmax(_policy(_features(observation, steps), weights)))
        step = env.step(action)
        observation = step.observation
        if env.cost.total < best_cost.total:
            best_layout = env.layout
            best_cost = env.cost
        curve.append(best_cost.total)
        if step.done:
            break
    return best_layout, best_cost, curve


def _features(observation: dict, max_steps: int) -> np.ndarray:
    metrics = observation.get("metrics", {})
    cost = float(observation["cost"])
    return np.array(
        [
            1.0,
            float(observation["step"]) / max(max_steps, 1),
            np.log1p(max(cost, 0.0)),
            np.log1p(max(float(metrics.get("wirelength", 0.0)), 0.0)),
            float(metrics.get("tmax", 0.0)) / 100.0,
            float(metrics.get("top5", 0.0)) / 100.0,
            float(metrics.get("outline_penalty", 0.0)),
            float(metrics.get("overlap_penalty", 0.0)),
        ],
        dtype=float,
    )


def _policy(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    logits = features @ weights
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)


def _discounted_returns(rewards: list[float], gamma: float) -> np.ndarray:
    returns = np.zeros(len(rewards), dtype=float)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + gamma * running
        returns[index] = running
    return returns
