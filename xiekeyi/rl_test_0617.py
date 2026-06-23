from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from thermopt.layout.geometry import total_outline_penalty, total_overlap_penalty
from thermopt.layout.objects import FloorplanCase, Layout, Placement
from thermopt.objective.cost import CostResult


@dataclass(frozen=True)
class RLResult:
    best_layout: Layout
    best_cost: CostResult
    best_curve: list[float]
    episode_returns: list[float]
    action_acceptance_curve: list[float]
    policy_state_dict: dict[str, torch.Tensor]
    training_episodes: int
    rollout_steps: int
    final_layout: Layout
    final_cost: CostResult
    attempted_actions: int
    accepted_actions: int

    @property
    def accepted_ratio(self) -> float:
        return self.accepted_actions / max(1, self.attempted_actions)


@dataclass
class _RolloutBuffer:
    states: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    old_log_probs: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)

    def append(
        self,
        state: np.ndarray,
        action: int,
        old_log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.old_log_probs.append(old_log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def __len__(self) -> int:
        return len(self.states)


@dataclass(frozen=True)
class _ActionResult:
    layout: Layout
    valid: bool


class _ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.backbone(state)
        logits = self.actor(feature)
        value = self.critic(feature).squeeze(-1)
        return logits, value


def optimize(
    case: FloorplanCase,
    initial_layout: Layout,
    objective: Callable[[Layout], CostResult],
    config: dict,
    seed: int,
) -> RLResult:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _select_device()

    episodes = max(1, int(config.get("episodes", 100)))
    max_steps = max(1, int(config.get("max_steps", 70)))
    rollout_steps = max(1, int(config.get("rollout_steps", max_steps)))

    move_scale = float(config.get("move_scale", 10.0))
    hidden_dim = int(config.get("hidden_dim", 128))
    learning_rate = float(config.get("learning_rate", 3e-4))
    gamma = float(config.get("gamma", 0.95))
    gae_lambda = float(config.get("gae_lambda", 0.95))
    ppo_epochs = max(1, int(config.get("ppo_epochs", 4)))
    batch_size = max(1, int(config.get("batch_size", 64)))
    clip_epsilon = float(config.get("clip_epsilon", 0.2))
    entropy_coef = float(config.get("entropy_coef", 1e-3))
    critic_coef = float(config.get("critic_coef", 0.5))
    reward_scale = float(config.get("reward_scale", 1.0))
    max_grad_norm = float(config.get("max_grad_norm", 1.0))
    invalid_placement_penalty = float(config.get("invalid_placement_penalty", 1.0))

    initial_cost = objective(initial_layout)
    reference = _reference_metrics(initial_cost)
    state_dim = len(_layout_state(case, initial_layout, initial_cost, reference, step=0, max_steps=max_steps))
    action_dim = max(1, len(initial_layout.placements) * 10)
    model = _ActorCritic(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_layout = initial_layout
    best_cost = initial_cost
    best_score = _score(initial_cost, reference, config)
    best_curve = [best_cost.total]
    episode_returns: list[float] = []
    action_acceptance_curve: list[float] = []
    attempted_moves = 0
    accepted_moves = 0
    final_layout = initial_layout
    final_cost = initial_cost
    started = time.perf_counter()

    for episode in range(episodes):
        current_layout = initial_layout
        current_cost = objective(current_layout)
        current_score = _score(current_cost, reference, config)
        buffer = _RolloutBuffer()
        episode_return = 0.0

        for step in range(max_steps):
            state_np = _layout_state(case, current_layout, current_cost, reference, step, max_steps)
            state = torch.as_tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, value = model(state)
                distribution = Categorical(logits=logits)
                action = distribution.sample()
                log_prob = distribution.log_prob(action)

            action_result = _apply_action(case, current_layout, int(action.item()), move_scale)
            candidate_layout = action_result.layout
            candidate_cost = objective(candidate_layout)
            candidate_score = _score(candidate_cost, reference, config)
            reward = _reward(current_cost, candidate_cost, config) / max(reward_scale, 1e-9)
            if not action_result.valid:
                reward -= invalid_placement_penalty
            done = step + 1 >= max_steps

            buffer.append(
                state=state_np,
                action=int(action.item()),
                old_log_prob=float(log_prob.item()),
                reward=float(reward),
                value=float(value.item()),
                done=done,
            )

            if action_result.valid:
                current_layout = candidate_layout
                current_cost = candidate_cost
                current_score = candidate_score
            episode_return += float(reward)
            attempted_moves += 1
            if action_result.valid:
                accepted_moves += 1

            if current_score < best_score:
                best_layout = current_layout
                best_cost = current_cost
                best_score = current_score

            if len(buffer) >= rollout_steps or done:
                last_value = 0.0
                if not done:
                    next_state_np = _layout_state(case, current_layout, current_cost, reference, step + 1, max_steps)
                    next_state = torch.as_tensor(next_state_np, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        _, next_value = model(next_state)
                    last_value = float(next_value.item())
                _ppo_update(
                    model=model,
                    optimizer=optimizer,
                    buffer=buffer,
                    device=device,
                    last_value=last_value,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    ppo_epochs=ppo_epochs,
                    batch_size=batch_size,
                    clip_epsilon=clip_epsilon,
                    entropy_coef=entropy_coef,
                    critic_coef=critic_coef,
                    max_grad_norm=max_grad_norm,
                )
                buffer = _RolloutBuffer()

            if done:
                break

        final_layout = current_layout
        final_cost = current_cost
        episode_returns.append(episode_return)
        action_acceptance_curve.append(accepted_moves / max(1, attempted_moves))
        best_curve.append(best_cost.total)
        elapsed = time.perf_counter() - started
        print(
            f"[rl-ppo] time={elapsed:.2f}s episode={episode + 1}/{episodes} "
            f"objective={current_score:.4f} best_objective={best_score:.4f}",
            flush=True,
        )

    rollout_layout, rollout_cost, rollout_curve, rollout_count = _greedy_rollout(
        case=case,
        initial_layout=initial_layout,
        objective=objective,
        config=config,
        model=model,
        reference=reference,
        device=device,
        steps=rollout_steps,
        move_scale=move_scale,
    )
    rollout_score = _score(rollout_cost, reference, config)
    if rollout_score < best_score:
        best_layout = rollout_layout
        best_cost = rollout_cost
    final_layout = rollout_layout
    final_cost = rollout_cost
    best_curve.extend(rollout_curve)
    attempted_moves += rollout_count
    accepted_moves += rollout_count

    if best_curve[-1] != best_cost.total:
        best_curve.append(best_cost.total)

    return RLResult(
        best_layout=best_layout,
        best_cost=best_cost,
        best_curve=best_curve,
        episode_returns=episode_returns,
        action_acceptance_curve=action_acceptance_curve,
        policy_state_dict={name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        training_episodes=episodes,
        rollout_steps=rollout_steps,
        final_layout=final_layout,
        final_cost=final_cost,
        attempted_actions=attempted_moves,
        accepted_actions=accepted_moves,
    )


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _reference_metrics(cost: CostResult) -> dict[str, float]:
    return {
        "wirelength": max(float(cost.metrics.get("wirelength", cost.total)), 1e-9),
        "thermal": max(float(cost.metrics.get("thermal", cost.total)), 1e-9),
        "total": max(float(cost.total), 1e-9),
    }


def _layout_state(
    case: FloorplanCase,
    layout: Layout,
    cost: CostResult,
    reference: dict[str, float],
    step: int,
    max_steps: int,
) -> np.ndarray:
    chiplets = case.chiplet_by_id
    max_power = max((chiplet.power for chiplet in case.chiplets), default=1.0)
    outline_width = max(case.outline_width, 1e-9)
    outline_height = max(case.outline_height, 1e-9)
    net_features = _chiplet_net_features(case, layout)

    features: list[float] = []
    for placement in layout.placements:
        chiplet = chiplets[placement.chiplet_id]
        width, height = placement.rotated_size(chiplet)
        chiplet_net_features = net_features[placement.chiplet_id]
        features.extend(
            [
                placement.x / outline_width,
                placement.y / outline_height,
                width / outline_width,
                height / outline_height,
                chiplet.power / max(max_power, 1e-9),
                1.0 if placement.rotation % 180 == 90 else 0.0,
                chiplet.width / outline_width,
                chiplet.height / outline_height,
                *chiplet_net_features,
            ]
        )

    metrics = cost.metrics
    features.extend(
        [
            float(step) / max(max_steps, 1),
            float(metrics.get("wirelength", cost.total)) / reference["wirelength"],
            float(metrics.get("thermal", cost.total)) / reference["thermal"],
            float(metrics.get("tmax", 0.0)) / 100.0,
            float(metrics.get("top5", 0.0)) / 100.0,
            float(metrics.get("outline_penalty", 0.0)),
            float(metrics.get("overlap_penalty", 0.0)),
            cost.total / reference["total"],
        ]
    )
    return np.asarray(features, dtype=np.float32)


def _chiplet_net_features(case: FloorplanCase, layout: Layout) -> dict[str, tuple[float, ...]]:
    placement_by_id = layout.by_id
    outline_width = max(case.outline_width, 1e-9)
    outline_height = max(case.outline_height, 1e-9)
    centers: dict[str, tuple[float, float]] = {}
    for chiplet in case.chiplets:
        placement = placement_by_id[chiplet.id]
        width, height = placement.rotated_size(chiplet)
        centers[chiplet.id] = (placement.x + 0.5 * width, placement.y + 0.5 * height)

    incident_counts = {chiplet.id: 0 for chiplet in case.chiplets}
    degree_sums = {chiplet.id: 0.0 for chiplet in case.chiplets}
    neighbors: dict[str, set[str]] = {chiplet.id: set() for chiplet in case.chiplets}
    max_net_degree = 1
    for net in case.nets:
        chiplet_ids = tuple(dict.fromkeys(chiplet_id for chiplet_id in net.chiplets if chiplet_id in incident_counts))
        degree = len(chiplet_ids)
        if degree == 0:
            continue
        max_net_degree = max(max_net_degree, degree)
        for chiplet_id in chiplet_ids:
            incident_counts[chiplet_id] += 1
            degree_sums[chiplet_id] += degree
            neighbors[chiplet_id].update(other_id for other_id in chiplet_ids if other_id != chiplet_id)

    max_incident_count = max(max(incident_counts.values(), default=0), 1)
    max_neighbor_count = max(len(case.chiplets) - 1, 1)
    features: dict[str, tuple[float, ...]] = {}
    for chiplet_id in incident_counts:
        chiplet_neighbors = neighbors[chiplet_id]
        if chiplet_neighbors:
            mean_neighbor_x = float(np.mean([centers[neighbor_id][0] for neighbor_id in chiplet_neighbors]))
            mean_neighbor_y = float(np.mean([centers[neighbor_id][1] for neighbor_id in chiplet_neighbors]))
        else:
            mean_neighbor_x, mean_neighbor_y = centers[chiplet_id]
        incident_count = incident_counts[chiplet_id]
        average_net_degree = degree_sums[chiplet_id] / max(incident_count, 1)
        features[chiplet_id] = (
            incident_count / max_incident_count,
            len(chiplet_neighbors) / max_neighbor_count,
            average_net_degree / max_net_degree,
            mean_neighbor_x / outline_width,
            mean_neighbor_y / outline_height,
        )
    return features


def _score(cost: CostResult, reference: dict[str, float], config: dict) -> float:
    metrics = cost.metrics
    wirelength_weight = float(config.get("wirelength_weight", 1.0))
    outline_weight = float(config.get("outline_penalty_weight", 10.0))
    overlap_weight = float(config.get("overlap_penalty_weight", 10.0))
    return (
        wirelength_weight * float(metrics.get("wirelength", cost.total)) / reference["wirelength"]
        + outline_weight * float(metrics.get("outline_penalty", 0.0))
        + overlap_weight * float(metrics.get("overlap_penalty", 0.0))
    )


def _reward(previous: CostResult, current: CostResult, config: dict) -> float:
    previous_metrics = previous.metrics
    current_metrics = current.metrics
    wirelength_weight = float(config.get("wirelength_weight", 1.0))
    return wirelength_weight * (
        float(previous_metrics.get("wirelength", previous.total))
        - float(current_metrics.get("wirelength", current.total))
    )


def _apply_action(case: FloorplanCase, layout: Layout, action: int, move_scale: float) -> _ActionResult:
    placements = list(layout.placements)
    num_placements = len(placements)
    if num_placements == 0:
        return _ActionResult(layout=layout, valid=True)

    action_kinds = 10
    idx = int(action // action_kinds) % num_placements
    kind = int(action % action_kinds)

    placement = placements[idx]
    chiplet = case.chiplet_by_id[placement.chiplet_id]

    if kind == 8:
        placements[idx] = placement.rotated()
        candidate = Layout(tuple(placements))
        return _constrain_action(case, layout, candidate)

    if kind == 9 and num_placements >= 2:
        other_idx = (idx + 1) % num_placements
        other = placements[other_idx]
        placements[idx] = Placement(placement.chiplet_id, other.x, other.y, placement.rotation)
        placements[other_idx] = Placement(other.chiplet_id, placement.x, placement.y, other.rotation)
        candidate = Layout(tuple(placements))
        return _constrain_action(case, layout, candidate)

    directions = (
        (-1.0, 0.0),
        (1.0, 0.0),
        (0.0, -1.0),
        (0.0, 1.0),
        (-1.0, -1.0),
        (-1.0, 1.0),
        (1.0, -1.0),
        (1.0, 1.0),
    )
    dx_unit, dy_unit = directions[kind % len(directions)]
    step_scale = move_scale * (0.7 if kind >= 4 else 1.0)
    width, height = placement.rotated_size(chiplet)
    x = float(np.clip(placement.x + dx_unit * step_scale, 0.0, max(0.0, case.outline_width - width)))
    y = float(np.clip(placement.y + dy_unit * step_scale, 0.0, max(0.0, case.outline_height - height)))
    placements[idx] = placement.moved(x=x, y=y)
    candidate = Layout(tuple(placements))
    return _constrain_action(case, layout, candidate)


def _constrain_action(case: FloorplanCase, current_layout: Layout, candidate_layout: Layout) -> _ActionResult:
    tolerance = 1e-9
    current_outline = total_outline_penalty(case, current_layout)
    current_overlap = total_overlap_penalty(case, current_layout)
    candidate_outline = total_outline_penalty(case, candidate_layout)
    candidate_overlap = total_overlap_penalty(case, candidate_layout)

    candidate_legal = candidate_outline <= tolerance and candidate_overlap <= tolerance
    if candidate_legal:
        return _ActionResult(layout=candidate_layout, valid=True)

    current_legal = current_outline <= tolerance and current_overlap <= tolerance
    if current_legal:
        return _ActionResult(layout=current_layout, valid=False)

    outline_worse = candidate_outline > current_outline + tolerance
    overlap_worse = candidate_overlap > current_overlap + tolerance
    improves_invalidity = (
        candidate_outline < current_outline - tolerance or candidate_overlap < current_overlap - tolerance
    )
    if not outline_worse and not overlap_worse and improves_invalidity:
        return _ActionResult(layout=candidate_layout, valid=True)
    return _ActionResult(layout=current_layout, valid=False)


def _ppo_update(
    model: _ActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: _RolloutBuffer,
    device: torch.device,
    last_value: float,
    gamma: float,
    gae_lambda: float,
    ppo_epochs: int,
    batch_size: int,
    clip_epsilon: float,
    entropy_coef: float,
    critic_coef: float,
    max_grad_norm: float,
) -> None:
    if len(buffer) == 0:
        return

    states = torch.as_tensor(np.stack(buffer.states), dtype=torch.float32, device=device)
    actions = torch.as_tensor(buffer.actions, dtype=torch.long, device=device)
    old_log_probs = torch.as_tensor(buffer.old_log_probs, dtype=torch.float32, device=device)
    values_np = np.asarray(buffer.values, dtype=np.float32)
    rewards_np = np.asarray(buffer.rewards, dtype=np.float32)
    dones_np = np.asarray(buffer.dones, dtype=np.float32)
    returns_np, advantages_np = _compute_gae(rewards_np, values_np, dones_np, last_value, gamma, gae_lambda)

    returns = torch.as_tensor(returns_np, dtype=torch.float32, device=device)
    advantages = torch.as_tensor(advantages_np, dtype=torch.float32, device=device)
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    sample_count = states.shape[0]
    for _ in range(ppo_epochs):
        permutation = torch.randperm(sample_count, device=device)
        for start in range(0, sample_count, batch_size):
            idx = permutation[start : start + batch_size]
            logits, value = model(states[idx])
            distribution = Categorical(logits=logits)
            new_log_probs = distribution.log_prob(actions[idx])
            entropy = distribution.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs[idx])
            surrogate_1 = ratio * advantages[idx]
            surrogate_2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages[idx]
            actor_loss = -torch.min(surrogate_1, surrogate_2).mean()
            critic_loss = F.smooth_l1_loss(value, returns[idx])
            loss = actor_loss + critic_coef * critic_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()


def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        next_non_terminal = 1.0 - dones[index]
        next_value = last_value if index == len(rewards) - 1 else values[index + 1]
        delta = rewards[index] + gamma * next_value * next_non_terminal - values[index]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[index] = last_gae
    returns = advantages + values
    return returns.astype(np.float32), advantages.astype(np.float32)


def _greedy_rollout(
    case: FloorplanCase,
    initial_layout: Layout,
    objective: Callable[[Layout], CostResult],
    config: dict,
    model: _ActorCritic,
    reference: dict[str, float],
    device: torch.device,
    steps: int,
    move_scale: float,
) -> tuple[Layout, CostResult, list[float], int]:
    current_layout = initial_layout
    current_cost = objective(current_layout)
    best_layout = current_layout
    best_cost = current_cost
    best_score = _score(current_cost, reference, config)
    curve: list[float] = []

    model.eval()
    with torch.no_grad():
        for step in range(steps):
            state_np = _layout_state(case, current_layout, current_cost, reference, step, steps)
            state = torch.as_tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = model(state)
            action_result = None
            for action in torch.argsort(logits.squeeze(0), descending=True).tolist():
                candidate_result = _apply_action(case, current_layout, int(action), move_scale)
                if candidate_result.valid:
                    action_result = candidate_result
                    break
            if action_result is None:
                curve.append(best_cost.total)
                continue
            current_layout = action_result.layout
            current_cost = objective(current_layout)
            current_score = _score(current_cost, reference, config)
            if current_score < best_score:
                best_layout = current_layout
                best_cost = current_cost
                best_score = current_score
            curve.append(best_cost.total)
    model.train()
    return best_layout, best_cost, curve, steps
