from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from thermopt.layout.geometry import hpwl, total_outline_penalty, total_overlap_penalty
from thermopt.layout.objects import FloorplanCase, Layout, Placement
from thermopt.objective.cost import CostResult


_DATA_MISMATCHES = (
    "EfficientPlace reads a PlaceDB benchmark with ranked macros and detailed node/net metadata; "
    "ThermOpt provides FloorplanCase chiplets/nets directly.",
    "EfficientPlace places on a native square grid; ThermOpt layouts use continuous outline units, "
    "so this optimizer discretizes the outline to a grid and maps actions back to continuous x/y.",
    "EfficientPlace constructs a placement from an empty canvas; ThermOpt passes an initial_layout for "
    "the optimizer interface, which is used here as the baseline/fallback, not as the policy state.",
    "EfficientPlace optimizes HPWL-style wirelength reward; ThermOpt Objective can include thermal cost, "
    "but this file uses Objective only to return CostResult for the selected wirelength-best layout.",
    "EfficientPlace does not model chiplet rotation; this migration keeps rotation at 0 during placement.",
)


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
    policy_order: tuple[str, ...]
    objective_name: str = "wirelength"
    data_mismatches: tuple[str, ...] = _DATA_MISMATCHES

    @property
    def accepted_ratio(self) -> float:
        return self.accepted_actions / max(1, self.attempted_actions)


@dataclass
class _RolloutBuffer:
    states: list[np.ndarray] = field(default_factory=list)
    time_steps: list[int] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    old_log_probs: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)

    def append(
        self,
        state: np.ndarray,
        time_step: int,
        action: int,
        old_log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        self.states.append(state)
        self.time_steps.append(time_step)
        self.actions.append(action)
        self.old_log_probs.append(old_log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def __len__(self) -> int:
        return len(self.states)

    def clone(self) -> "_RolloutBuffer":
        return _RolloutBuffer(
            states=[np.array(state, copy=True) for state in self.states],
            time_steps=list(self.time_steps),
            actions=list(self.actions),
            old_log_probs=list(self.old_log_probs),
            rewards=list(self.rewards),
            values=list(self.values),
            dones=list(self.dones),
        )


@dataclass(eq=False)
class _SearchNode:
    parent: "_SearchNode | None"
    action: int | None
    depth: int
    placements: dict[str, tuple[int, int]]
    net_bounds: dict[int, tuple[float, float, float, float]]
    state: np.ndarray
    children: dict[int, "_SearchNode"] = field(default_factory=dict)
    visits: int = 0
    best_value: float = -float("inf")
    best_wirelength: float = float("inf")


class _EfficientPlaceActorCritic(nn.Module):
    """Small EfficientPlace-style actor/critic for ThermOpt's variable grid sizes."""

    def __init__(self, grid: int, num_time_steps: int, hidden_dim: int):
        super().__init__()
        self.grid = grid
        self.num_time_steps = num_time_steps
        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.time_embedding = nn.Embedding(num_time_steps + 1, hidden_dim)
        self.actor = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        self.critic = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, time_step: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        time_step = torch.clamp(time_step, min=0, max=self.num_time_steps)
        feature = self.encoder(state)
        time_feature = self.time_embedding(time_step).unsqueeze(-1).unsqueeze(-1)
        feature = feature + time_feature
        logits = self.actor(feature).reshape(-1, self.grid * self.grid)
        value = self.critic(feature).squeeze(-1)
        return logits, value


class _RunLogger:
    def __init__(self, output_dir: Path | None, verbose: bool):
        self.verbose = verbose
        self.path: Path | None = None
        if output_dir is None:
            return
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.path = output_dir / "rl_effplace.log"
        except OSError:
            self.path = None

    def log(self, message: str, *, console: bool | None = None) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        if self.path is not None:
            try:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                self.path = None
        if self.verbose if console is None else console:
            print(line, flush=True)


class _EfficientPlaceEnv:
    def __init__(
        self,
        case: FloorplanCase,
        order: tuple[str, ...],
        grid: int,
        wire_mask_scale: float,
        reward_scale: float,
        invalid_placement_penalty: float,
    ):
        self.case = case
        self.order = order
        self.grid = grid
        self.cell_width = case.outline_width / max(1, grid)
        self.cell_height = case.outline_height / max(1, grid)
        self.wire_mask_scale = max(wire_mask_scale, 1e-9)
        self.reward_scale = max(reward_scale, 1e-9)
        self.invalid_placement_penalty = invalid_placement_penalty
        self.chiplets = case.chiplet_by_id
        self.size_cells = {
            chiplet.id: (
                max(1, math.ceil(chiplet.width / max(self.cell_width, 1e-9))),
                max(1, math.ceil(chiplet.height / max(self.cell_height, 1e-9))),
            )
            for chiplet in case.chiplets
        }
        self.net_indices_by_chiplet = {chiplet.id: [] for chiplet in case.chiplets}
        self.pin_offsets: dict[tuple[int, str], tuple[float, float]] = {}
        for net_index, net in enumerate(case.nets):
            offsets = net.pin_offsets or tuple((0.0, 0.0) for _ in net.chiplets)
            for pin_index, chiplet_id in enumerate(net.chiplets):
                if chiplet_id not in self.net_indices_by_chiplet:
                    continue
                self.net_indices_by_chiplet[chiplet_id].append(net_index)
                offset = offsets[pin_index] if pin_index < len(offsets) else (0.0, 0.0)
                self.pin_offsets[(net_index, chiplet_id)] = (float(offset[0]), float(offset[1]))
        self.reset()

    def reset(self) -> np.ndarray:
        self.t = 0
        self.canvas = np.zeros((self.grid, self.grid), dtype=np.float32)
        self.placements: dict[str, tuple[int, int]] = {}
        self.net_bounds: dict[int, tuple[float, float, float, float]] = {}
        self.state = self._observation()
        return self.state

    def restore(self, node: _SearchNode) -> np.ndarray:
        self.t = node.depth
        self.placements = dict(node.placements)
        self.net_bounds = dict(node.net_bounds)
        self.state = np.array(node.state, copy=True)
        self.canvas = np.array(self.state[0], copy=True)
        return self.state

    def search_node(self, parent: _SearchNode | None, action: int | None) -> _SearchNode:
        return _SearchNode(
            parent=parent,
            action=action,
            depth=self.t,
            placements=dict(self.placements),
            net_bounds=dict(self.net_bounds),
            state=np.array(self.state, copy=True),
        )

    def is_done(self) -> bool:
        return self.t >= len(self.order)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, float | bool]]:
        if self.t >= len(self.order):
            self.state = self._observation()
            return self.state, 0.0, True, {"valid": False, "wirelength_increment": 0.0}

        chiplet_id = self.order[self.t]
        cell_x = int(action // self.grid)
        cell_y = int(action % self.grid)
        position_mask = self.state[2]
        has_valid_action = bool(np.any(position_mask < 0.5))
        valid = 0 <= cell_x < self.grid and 0 <= cell_y < self.grid and position_mask[cell_x, cell_y] < 0.5
        wirelength_increment = self._wire_increment(chiplet_id, cell_x, cell_y) if valid else 0.0

        if valid:
            self._place(chiplet_id, cell_x, cell_y)
            reward = -wirelength_increment / self.reward_scale
            self.t += 1
        elif not has_valid_action:
            self._place_with_fallback(chiplet_id)
            reward = -self.invalid_placement_penalty
            self.t += 1
        else:
            reward = -self.invalid_placement_penalty

        done = self.is_done()
        self.state = self._observation()
        return (
            self.state,
            float(reward),
            done,
            {"valid": bool(valid), "wirelength_increment": float(wirelength_increment)},
        )

    def layout(self, initial_layout: Layout | None = None) -> Layout:
        initial_by_id = initial_layout.by_id if initial_layout is not None else {}
        placements: list[Placement] = []
        for chiplet in self.case.chiplets:
            if chiplet.id in self.placements:
                cell_x, cell_y = self.placements[chiplet.id]
                x, y = self._to_continuous(cell_x, cell_y)
                x = min(max(0.0, x), max(0.0, self.case.outline_width - chiplet.width))
                y = min(max(0.0, y), max(0.0, self.case.outline_height - chiplet.height))
                placements.append(Placement(chiplet.id, x, y, 0))
            elif chiplet.id in initial_by_id:
                placements.append(initial_by_id[chiplet.id])
            else:
                placements.append(Placement(chiplet.id, 0.0, 0.0, 0))
        return Layout(tuple(placements))

    def _observation(self) -> np.ndarray:
        if self.t >= len(self.order):
            wire_mask = np.zeros((self.grid, self.grid), dtype=np.float32)
            position_mask = np.ones((self.grid, self.grid), dtype=np.float32)
            return np.stack([self.canvas.copy(), wire_mask, position_mask], axis=0).astype(np.float32)

        chiplet_id = self.order[self.t]
        size_x, size_y = self.size_cells[chiplet_id]
        position_mask = self._position_mask(size_x, size_y)
        wire_mask = self._wire_mask(chiplet_id, position_mask)
        return np.stack([self.canvas.copy(), wire_mask, position_mask], axis=0).astype(np.float32)

    def _position_mask(self, size_x: int, size_y: int) -> np.ndarray:
        mask = np.ones((self.grid, self.grid), dtype=np.float32)
        if size_x > self.grid or size_y > self.grid:
            return mask
        for cell_x in range(0, self.grid - size_x + 1):
            for cell_y in range(0, self.grid - size_y + 1):
                if not np.any(self.canvas[cell_x : cell_x + size_x, cell_y : cell_y + size_y] > 0.0):
                    mask[cell_x, cell_y] = 0.0
        return mask

    def _wire_mask(self, chiplet_id: str, position_mask: np.ndarray) -> np.ndarray:
        mask = np.zeros((self.grid, self.grid), dtype=np.float32)
        valid_cells = np.argwhere(position_mask < 0.5)
        for cell_x, cell_y in valid_cells:
            mask[cell_x, cell_y] = self._wire_increment(chiplet_id, int(cell_x), int(cell_y)) / self.wire_mask_scale
        if valid_cells.size:
            max_valid = float(np.max(mask[position_mask < 0.5]))
            mask[position_mask >= 0.5] = max_valid + 1.0
        return mask

    def _wire_increment(self, chiplet_id: str, cell_x: int, cell_y: int) -> float:
        increment = 0.0
        for net_index in self.net_indices_by_chiplet[chiplet_id]:
            if net_index not in self.net_bounds:
                continue
            min_x, max_x, min_y, max_y = self.net_bounds[net_index]
            pin_x, pin_y = self._pin_position(net_index, chiplet_id, cell_x, cell_y)
            old_hpwl = (max_x - min_x) + (max_y - min_y)
            new_hpwl = (max(max_x, pin_x) - min(min_x, pin_x)) + (max(max_y, pin_y) - min(min_y, pin_y))
            increment += max(0.0, new_hpwl - old_hpwl)
        return float(increment)

    def _place(self, chiplet_id: str, cell_x: int, cell_y: int) -> None:
        size_x, size_y = self.size_cells[chiplet_id]
        self.canvas[cell_x : cell_x + size_x, cell_y : cell_y + size_y] = 1.0
        self.canvas[cell_x : cell_x + size_x, cell_y] = 0.5
        self.canvas[cell_x, cell_y : cell_y + size_y] = 0.5
        if cell_x + size_x - 1 < self.grid:
            self.canvas[cell_x + size_x - 1, cell_y : cell_y + size_y] = 0.5
        if cell_y + size_y - 1 < self.grid:
            self.canvas[cell_x : cell_x + size_x, cell_y + size_y - 1] = 0.5
        self.placements[chiplet_id] = (cell_x, cell_y)
        self._update_net_bounds(chiplet_id, cell_x, cell_y)

    def _place_with_fallback(self, chiplet_id: str) -> None:
        size_x, size_y = self.size_cells[chiplet_id]
        max_x = max(0, self.grid - min(size_x, self.grid))
        max_y = max(0, self.grid - min(size_y, self.grid))
        self._place(chiplet_id, max_x, max_y)

    def _update_net_bounds(self, chiplet_id: str, cell_x: int, cell_y: int) -> None:
        for net_index in self.net_indices_by_chiplet[chiplet_id]:
            pin_x, pin_y = self._pin_position(net_index, chiplet_id, cell_x, cell_y)
            if net_index not in self.net_bounds:
                self.net_bounds[net_index] = (pin_x, pin_x, pin_y, pin_y)
                continue
            min_x, max_x, min_y, max_y = self.net_bounds[net_index]
            self.net_bounds[net_index] = (
                min(min_x, pin_x),
                max(max_x, pin_x),
                min(min_y, pin_y),
                max(max_y, pin_y),
            )

    def _pin_position(self, net_index: int, chiplet_id: str, cell_x: int, cell_y: int) -> tuple[float, float]:
        chiplet = self.chiplets[chiplet_id]
        x, y = self._to_continuous(cell_x, cell_y)
        offset_x, offset_y = self.pin_offsets.get((net_index, chiplet_id), (0.0, 0.0))
        return x + 0.5 * chiplet.width + offset_x, y + 0.5 * chiplet.height + offset_y

    def _to_continuous(self, cell_x: int, cell_y: int) -> tuple[float, float]:
        return cell_x * self.cell_width, cell_y * self.cell_height


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
    rng = np.random.default_rng(seed)
    device = _select_device()

    order = _placement_order(case, config)
    episodes = max(1, int(config.get("episodes", 100)))
    grid = _config_grid(config)
    rollout_steps = len(order)
    hidden_dim = max(8, int(config.get("hidden_dim", 64)))
    learning_rate = float(config.get("learning_rate", 3e-4))
    gamma = float(config.get("gamma", 0.99))
    gae_lambda = float(config.get("gae_lambda", 0.95))
    ppo_epochs = max(1, int(config.get("ppo_epochs", 4)))
    batch_size = max(1, int(config.get("batch_size", 64)))
    clip_epsilon = float(config.get("clip_epsilon", 0.2))
    entropy_coef = float(config.get("entropy_coef", 1e-3))
    critic_coef = float(config.get("critic_coef", 0.5))
    max_grad_norm = float(config.get("max_grad_norm", 0.5))
    wire_mask_scale = float(config.get("wire_mask_scale", max(case.outline_width + case.outline_height, 1.0)))
    reward_scale = float(config.get("reward_scale", max(wire_mask_scale, 1.0)))
    invalid_placement_penalty = float(config.get("invalid_placement_penalty", 1.0))
    wire_mask_bias = float(config.get("wire_mask_bias", 1.0))
    greedy_wire_mask = bool(config.get("greedy_wire_mask", True))
    terminal_reward_coef = float(config.get("terminal_reward_coef", 10.0))
    illegal_terminal_penalty = float(config.get("illegal_terminal_penalty", invalid_placement_penalty * len(order)))
    elite_replay_coef = float(config.get("elite_replay_coef", 0.05))
    elite_replay_epochs = max(0, int(config.get("elite_replay_epochs", 1)))
    default_final_replay_epochs = max(5, elite_replay_epochs) if elite_replay_epochs > 0 else 0
    elite_replay_final_epochs = max(0, int(config.get("elite_replay_final_epochs", default_final_replay_epochs)))
    default_match_epochs = max(20, elite_replay_final_epochs) if elite_replay_final_epochs > 0 else 0
    elite_replay_match_epochs = max(0, int(config.get("elite_replay_match_epochs", default_match_epochs)))
    elite_replay_match_coef = float(config.get("elite_replay_match_coef", max(1.0, elite_replay_coef)))
    elite_replay_margin = max(0.0, float(config.get("elite_replay_margin", 1.0)))
    tree_search_enabled = bool(config.get("tree_search_enabled", True))
    frontier_capacity = max(1, int(config.get("frontier_capacity", min(8, len(order)))))
    frontier_update_every = max(1, int(config.get("frontier_update_every", 1)))
    tree_exploration_weight = float(config.get("tree_exploration_weight", max(case.outline_width + case.outline_height, 1.0)))
    verbose = bool(config.get("verbose", True))
    output_dir = _resolve_output_dir(config, objective)
    logger = _RunLogger(output_dir, verbose)

    model = _EfficientPlaceActorCritic(grid=grid, num_time_steps=len(order), hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    initial_wirelength = _wirelength(case, initial_layout)
    illegal_frontier_wirelength = float(
        config.get("illegal_frontier_wirelength", max(initial_wirelength, 1.0) * 1000.0)
    )
    initial_cost = objective(initial_layout)
    best_layout = initial_layout
    best_cost = initial_cost
    best_wirelength = initial_wirelength
    best_selection_wirelength = initial_wirelength if _is_legal(case, initial_layout) else float("inf")
    best_curve = [best_wirelength]
    episode_returns: list[float] = []
    action_acceptance_curve: list[float] = []
    attempted_actions = 0
    accepted_actions = 0
    final_layout = initial_layout
    final_cost = initial_cost
    best_buffer: _RolloutBuffer | None = None
    best_order = order
    best_policy_state_dict = _clone_state_dict(model)
    search_root, frontiers = _init_search_tree(
        case=case,
        order=order,
        grid=grid,
        wire_mask_scale=wire_mask_scale,
        reward_scale=reward_scale,
        invalid_placement_penalty=invalid_placement_penalty,
    )
    started = time.perf_counter()
    logger.log(
        "training_config "
        f"device={device} episodes={episodes} grid={grid} order_len={len(order)} "
        f"placement_order={config.get('placement_order', 'degree_area')} "
        f"shuffle_placement_order={bool(config.get('shuffle_placement_order', False))} "
        f"learning_rate={learning_rate:.6g} gamma={gamma:.4f} gae_lambda={gae_lambda:.4f} "
        f"ppo_epochs={ppo_epochs} batch_size={batch_size} clip_epsilon={clip_epsilon:.4f} "
        f"entropy_coef={entropy_coef:.6f} critic_coef={critic_coef:.4f} max_grad_norm={max_grad_norm:.4f} "
        f"wire_mask_scale={wire_mask_scale:.4f} reward_scale={reward_scale:.4f} "
        f"wire_mask_bias={wire_mask_bias:.4f} greedy_wire_mask={greedy_wire_mask} "
        f"terminal_reward_coef={terminal_reward_coef:.4f} invalid_placement_penalty={invalid_placement_penalty:.4f} "
        f"illegal_terminal_penalty={illegal_terminal_penalty:.4f} "
        f"illegal_frontier_wirelength={illegal_frontier_wirelength:.4f} "
        f"elite_replay_coef={elite_replay_coef:.4f} elite_replay_epochs={elite_replay_epochs} "
        f"elite_replay_final_epochs={elite_replay_final_epochs} "
        f"elite_replay_match_epochs={elite_replay_match_epochs} "
        f"elite_replay_match_coef={elite_replay_match_coef:.4f} "
        f"elite_replay_margin={elite_replay_margin:.4f} "
        f"tree_search_enabled={tree_search_enabled} frontier_capacity={frontier_capacity} "
        f"frontier_update_every={frontier_update_every} "
        f"tree_exploration_weight={tree_exploration_weight:.4f}",
        console=False,
    )

    for episode in range(episodes):
        frontier = _select_frontier(frontiers, rng, tree_exploration_weight) if tree_search_enabled else search_root
        env = _make_env(
            case=case,
            order=order,
            grid=grid,
            wire_mask_scale=wire_mask_scale,
            reward_scale=reward_scale,
            invalid_placement_penalty=invalid_placement_penalty,
        )
        state_np = env.restore(frontier) if tree_search_enabled else env.reset()
        tree_node = frontier
        buffer = _RolloutBuffer()
        episode_return = 0.0
        done = False

        while not done:
            time_step = env.t
            state = torch.as_tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
            time_tensor = torch.as_tensor([time_step], dtype=torch.long, device=device)
            with torch.no_grad():
                logits, value = model(state, time_tensor)
                distribution = _masked_distribution(
                    logits=logits,
                    state=state,
                    grid=grid,
                    wire_mask_bias=wire_mask_bias,
                    greedy_wire_mask=greedy_wire_mask,
                )
                action = distribution.sample()
                log_prob = distribution.log_prob(action)

            action_index = int(action.item())
            previous_depth = env.t
            next_state_np, reward, done, info = env.step(action_index)
            if tree_search_enabled and env.t > previous_depth:
                child = tree_node.children.get(action_index)
                if child is None:
                    child = env.search_node(parent=tree_node, action=action_index)
                    tree_node.children[action_index] = child
                tree_node = child
            buffer.append(
                state=state_np,
                time_step=time_step,
                action=action_index,
                old_log_prob=float(log_prob.item()),
                reward=float(reward),
                value=float(value.item()),
                done=done,
            )

            attempted_actions += 1
            if bool(info["valid"]):
                accepted_actions += 1
            episode_return += float(reward)
            state_np = next_state_np

        episode_layout = env.layout(initial_layout)
        episode_wirelength = _wirelength(case, episode_layout)
        episode_is_legal = _is_legal(case, episode_layout)
        terminal_hpwl_reward = _terminal_reward(initial_wirelength, episode_wirelength, terminal_reward_coef) if episode_is_legal else 0.0
        illegal_penalty = illegal_terminal_penalty if not episode_is_legal else 0.0
        terminal_reward = terminal_hpwl_reward - illegal_penalty
        if len(buffer) > 0:
            buffer.rewards[-1] += terminal_reward
            episode_return += terminal_reward
        episode_improved = episode_is_legal and episode_wirelength < best_selection_wirelength
        if tree_search_enabled:
            frontier_wirelength = episode_wirelength if episode_is_legal else illegal_frontier_wirelength
            _backup_search(tree_node, frontier_wirelength)
            if (episode + 1) % frontier_update_every == 0:
                frontiers = _update_frontiers(
                    frontiers,
                    capacity=frontier_capacity,
                    exploration_weight=tree_exploration_weight,
                    max_depth=len(order),
                )
        ppo_stats = _ppo_update(
            model=model,
            optimizer=optimizer,
            buffer=buffer,
            device=device,
            grid=grid,
            last_value=0.0,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ppo_epochs=ppo_epochs,
            batch_size=batch_size,
            clip_epsilon=clip_epsilon,
            entropy_coef=entropy_coef,
            critic_coef=critic_coef,
            max_grad_norm=max_grad_norm,
            wire_mask_bias=wire_mask_bias,
            greedy_wire_mask=greedy_wire_mask,
        ) or {}
        if episode_improved:
            best_buffer = _combine_search_path_buffer(frontier, buffer) if tree_search_enabled else buffer.clone()
            best_order = order
        elite_stats = _imitation_update(
            model=model,
            optimizer=optimizer,
            buffer=best_buffer,
            device=device,
            grid=grid,
            coef=elite_replay_coef,
            epochs=elite_replay_epochs,
            batch_size=batch_size,
            max_grad_norm=max_grad_norm,
            wire_mask_bias=wire_mask_bias,
            greedy_wire_mask=greedy_wire_mask,
            margin=elite_replay_margin,
        ) or {}
        episode_cost = objective(episode_layout)
        final_layout = episode_layout
        final_cost = episode_cost
        if episode_improved:
            best_layout = episode_layout
            best_cost = episode_cost
            best_wirelength = episode_wirelength
            best_selection_wirelength = episode_wirelength
            best_policy_state_dict = _clone_state_dict(model)

        episode_returns.append(episode_return)
        action_acceptance_curve.append(accepted_actions / max(1, attempted_actions))
        best_curve.append(best_wirelength)
        elapsed = time.perf_counter() - started
        logger.log(
            f"episode={episode + 1}/{episodes} time={elapsed:.2f}s "
            f"wirelength={episode_wirelength:.2f} best_wirelength={best_wirelength:.2f} "
            f"accepted_ratio={accepted_actions / max(1, attempted_actions):.4f} "
            f"legal={episode_is_legal} frontier_depth={frontier.depth}"
        )
        logger.log(
            f"training_update episode={episode + 1}/{episodes} "
            f"return={episode_return:.6f} terminal_reward={terminal_reward:.6f} "
            f"terminal_hpwl_reward={terminal_hpwl_reward:.6f} illegal_penalty={illegal_penalty:.6f} "
            f"improved={episode_improved} "
            f"ppo_batches={int(ppo_stats.get('batches', 0))} "
            f"ppo_loss={ppo_stats.get('loss', 0.0):.6f} "
            f"ppo_actor={ppo_stats.get('actor_loss', 0.0):.6f} "
            f"ppo_critic={ppo_stats.get('critic_loss', 0.0):.6f} "
            f"ppo_entropy={ppo_stats.get('entropy', 0.0):.6f} "
            f"ppo_grad_norm={ppo_stats.get('grad_norm', 0.0):.6f} "
            f"elite_batches={int(elite_stats.get('batches', 0))} "
            f"elite_loss={elite_stats.get('loss', 0.0):.6f} "
            f"elite_behavior={elite_stats.get('behavior_loss', 0.0):.6f} "
            f"elite_margin={elite_stats.get('margin_loss', 0.0):.6f} "
            f"elite_grad_norm={elite_stats.get('grad_norm', 0.0):.6f}",
            console=False,
        )

        if config.get("shuffle_placement_order", False):
            fixed_first = int(config.get("fixed_first_macros", 0))
            order = order[:fixed_first] + tuple(rng.permutation(order[fixed_first:]).tolist())
            if tree_search_enabled:
                search_root, frontiers = _init_search_tree(
                    case=case,
                    order=order,
                    grid=grid,
                    wire_mask_scale=wire_mask_scale,
                    reward_scale=reward_scale,
                    invalid_placement_penalty=invalid_placement_penalty,
                )

    final_elite_stats = _imitation_update(
        model=model,
        optimizer=optimizer,
        buffer=best_buffer,
        device=device,
        grid=grid,
        coef=elite_replay_coef,
        epochs=elite_replay_final_epochs,
        batch_size=batch_size,
        max_grad_norm=max_grad_norm,
        wire_mask_bias=wire_mask_bias,
        greedy_wire_mask=greedy_wire_mask,
        margin=elite_replay_margin,
    ) or {}
    match_stats = _reinforce_elite_until_greedy_match(
        model=model,
        optimizer=optimizer,
        buffer=best_buffer,
        device=device,
        grid=grid,
        coef=elite_replay_match_coef,
        max_epochs=elite_replay_match_epochs,
        batch_size=batch_size,
        max_grad_norm=max_grad_norm,
        wire_mask_bias=wire_mask_bias,
        greedy_wire_mask=greedy_wire_mask,
        margin=elite_replay_margin,
    )
    elite_policy_reproduced = bool(match_stats.get("reproduced", False))
    if best_buffer is not None and (
        (elite_replay_coef > 0.0 and elite_replay_final_epochs > 0) or elite_policy_reproduced
    ):
        best_policy_state_dict = _clone_state_dict(model)
    logger.log(
        "final_elite_replay "
        f"batches={int(final_elite_stats.get('batches', 0))} "
        f"loss={final_elite_stats.get('loss', 0.0):.6f} "
        f"behavior_loss={final_elite_stats.get('behavior_loss', 0.0):.6f} "
        f"margin_loss={final_elite_stats.get('margin_loss', 0.0):.6f} "
        f"grad_norm={final_elite_stats.get('grad_norm', 0.0):.6f} "
        f"match_reproduced={elite_policy_reproduced} "
        f"match_epochs={int(match_stats.get('epochs', 0))}",
        console=False,
    )

    rollout_layout, rollout_cost, rollout_wirelength = _greedy_rollout(
        case=case,
        initial_layout=initial_layout,
        objective=objective,
        order=best_order,
        grid=grid,
        model=model,
        device=device,
        wire_mask_scale=wire_mask_scale,
        reward_scale=reward_scale,
        invalid_placement_penalty=invalid_placement_penalty,
        wire_mask_bias=wire_mask_bias,
        greedy_wire_mask=greedy_wire_mask,
    )
    if _is_legal(case, rollout_layout) and rollout_wirelength < best_selection_wirelength:
        best_layout = rollout_layout
        best_cost = rollout_cost
        best_wirelength = rollout_wirelength
        best_selection_wirelength = rollout_wirelength
        best_policy_state_dict = _clone_state_dict(model)
    final_layout = rollout_layout
    final_cost = rollout_cost
    if best_curve[-1] != best_wirelength:
        best_curve.append(best_wirelength)
    logger.log(
        "final_greedy_rollout "
        f"wirelength={rollout_wirelength:.2f} best_wirelength={best_wirelength:.2f} "
        f"legal={_is_legal(case, rollout_layout)} reproduced_best={rollout_wirelength <= best_wirelength + 1e-6} "
        f"policy_order_len={len(best_order)}"
    )
    model_path = _save_policy_artifact(
        output_dir=output_dir,
        model=model,
        best_policy_state_dict=best_policy_state_dict,
        case=case,
        grid=grid,
        hidden_dim=hidden_dim,
        policy_order=best_order,
        best_wirelength=best_wirelength,
        rollout_wirelength=rollout_wirelength,
        config=config,
        logger=logger,
    )
    logger.log(f"model_saved path={model_path if model_path is not None else 'not_saved'}", console=False)

    return RLResult(
        best_layout=best_layout,
        best_cost=best_cost,
        best_curve=best_curve,
        episode_returns=episode_returns,
        action_acceptance_curve=action_acceptance_curve,
        policy_state_dict=best_policy_state_dict,
        training_episodes=episodes,
        rollout_steps=rollout_steps,
        final_layout=final_layout,
        final_cost=final_cost,
        attempted_actions=attempted_actions,
        accepted_actions=accepted_actions,
        policy_order=best_order,
    )


def _clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _resolve_output_dir(config: dict, objective: Callable[[Layout], CostResult]) -> Path:
    for key in ("output_dir", "log_dir", "model_dir"):
        value = config.get(key)
        if value:
            return Path(value).expanduser().resolve()

    backend = getattr(objective, "backend", None)
    workspace_root = getattr(backend, "_workspace_root", None)
    if workspace_root is not None:
        path = Path(workspace_root).expanduser().resolve()
        return path.parent if path.name == "_thermal" else path

    output_root = Path("outputs").expanduser()
    try:
        candidates = [path for path in output_root.iterdir() if path.is_dir()]
    except OSError:
        return output_root.resolve()
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime).resolve()
    return output_root.resolve()


def _save_policy_artifact(
    output_dir: Path,
    model: _EfficientPlaceActorCritic,
    best_policy_state_dict: dict[str, torch.Tensor],
    case: FloorplanCase,
    grid: int,
    hidden_dim: int,
    policy_order: tuple[str, ...],
    best_wirelength: float,
    rollout_wirelength: float,
    config: dict,
    logger: _RunLogger,
) -> Path | None:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "rl_effplace_model.pt"
        torch.save(
            {
                "best_policy_state_dict": best_policy_state_dict,
                "final_model_state_dict": _clone_state_dict(model),
                "grid": int(grid),
                "hidden_dim": int(hidden_dim),
                "num_time_steps": len(policy_order),
                "policy_order": tuple(policy_order),
                "case_chiplet_ids": tuple(case.chiplet_ids),
                "best_wirelength": float(best_wirelength),
                "final_greedy_wirelength": float(rollout_wirelength),
                "config": dict(config),
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            },
            path,
        )
        return path
    except Exception:
        return None


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _config_grid(config: dict) -> int:
    value = config.get("grid", 32)
    if isinstance(value, (list, tuple)):
        value = max(value) if value else 32
    return max(4, int(value))


def _placement_order(case: FloorplanCase, config: dict) -> tuple[str, ...]:
    mode = str(config.get("placement_order", "degree_area")).lower()
    chiplets = case.chiplet_by_id
    ids = tuple(chiplet.id for chiplet in case.chiplets)
    degree = {chiplet_id: 0 for chiplet_id in ids}
    for net in case.nets:
        for chiplet_id in set(net.chiplets):
            if chiplet_id in degree:
                degree[chiplet_id] += 1

    if mode == "input":
        return ids
    if mode == "area":
        return tuple(sorted(ids, key=lambda chiplet_id: (-chiplets[chiplet_id].width * chiplets[chiplet_id].height, chiplet_id)))
    if mode == "degree":
        return tuple(sorted(ids, key=lambda chiplet_id: (-degree[chiplet_id], chiplet_id)))
    return tuple(
        sorted(
            ids,
            key=lambda chiplet_id: (
                -degree[chiplet_id],
                -chiplets[chiplet_id].width * chiplets[chiplet_id].height,
                -chiplets[chiplet_id].power,
                chiplet_id,
            ),
        )
    )


def _make_env(
    case: FloorplanCase,
    order: tuple[str, ...],
    grid: int,
    wire_mask_scale: float,
    reward_scale: float,
    invalid_placement_penalty: float,
) -> _EfficientPlaceEnv:
    return _EfficientPlaceEnv(
        case=case,
        order=order,
        grid=grid,
        wire_mask_scale=wire_mask_scale,
        reward_scale=reward_scale,
        invalid_placement_penalty=invalid_placement_penalty,
    )


def _combine_search_path_buffer(frontier: _SearchNode, suffix: _RolloutBuffer) -> _RolloutBuffer:
    path_nodes: list[_SearchNode] = []
    node: _SearchNode | None = frontier
    while node is not None and node.parent is not None:
        path_nodes.append(node)
        node = node.parent
    path_nodes.reverse()

    combined = _RolloutBuffer()
    for path_node in path_nodes:
        parent = path_node.parent
        if parent is None or path_node.action is None:
            continue
        combined.append(
            state=np.array(parent.state, copy=True),
            time_step=parent.depth,
            action=int(path_node.action),
            old_log_prob=0.0,
            reward=0.0,
            value=0.0,
            done=False,
        )

    for index, state in enumerate(suffix.states):
        combined.append(
            state=np.array(state, copy=True),
            time_step=suffix.time_steps[index],
            action=suffix.actions[index],
            old_log_prob=suffix.old_log_probs[index],
            reward=suffix.rewards[index],
            value=suffix.values[index],
            done=suffix.dones[index],
        )
    return combined


def _init_search_tree(
    case: FloorplanCase,
    order: tuple[str, ...],
    grid: int,
    wire_mask_scale: float,
    reward_scale: float,
    invalid_placement_penalty: float,
) -> tuple[_SearchNode, list[_SearchNode]]:
    env = _make_env(
        case=case,
        order=order,
        grid=grid,
        wire_mask_scale=wire_mask_scale,
        reward_scale=reward_scale,
        invalid_placement_penalty=invalid_placement_penalty,
    )
    env.reset()
    root = env.search_node(parent=None, action=None)
    return root, [root]


def _node_score(node: _SearchNode, exploration_weight: float) -> float:
    if node.visits == 0:
        return float("inf")
    parent_visits = node.parent.visits if node.parent is not None else node.visits
    exploration = exploration_weight * math.sqrt(math.log(max(parent_visits, 1) + 1.0) / node.visits)
    return node.best_value + exploration


def _select_frontier(frontiers: list[_SearchNode], rng: np.random.Generator, exploration_weight: float) -> _SearchNode:
    finite_scores = np.array(
        [
            _node_score(node, exploration_weight) if math.isfinite(_node_score(node, exploration_weight)) else 1e9
            for node in frontiers
        ],
        dtype=float,
    )
    finite_scores -= np.max(finite_scores)
    weights = np.exp(np.clip(finite_scores, -30.0, 0.0))
    weights /= np.sum(weights)
    return frontiers[int(rng.choice(len(frontiers), p=weights))]


def _update_frontiers(
    frontiers: list[_SearchNode],
    capacity: int,
    exploration_weight: float,
    max_depth: int,
) -> list[_SearchNode]:
    candidates: dict[int, _SearchNode] = {id(node): node for node in frontiers if node.depth < max_depth}
    for node in frontiers:
        for child in node.children.values():
            if child.depth < max_depth:
                candidates[id(child)] = child

    ordered = sorted(candidates.values(), key=lambda node: _node_score(node, exploration_weight), reverse=True)
    return ordered[: max(1, capacity)] or frontiers[:1]


def _backup_search(node: _SearchNode, wirelength: float) -> None:
    value = -float(wirelength)
    while node is not None:
        node.visits += 1
        if value > node.best_value:
            node.best_value = value
            node.best_wirelength = float(wirelength)
        node = node.parent


def _terminal_reward(initial_wirelength: float, episode_wirelength: float, coef: float) -> float:
    if coef == 0.0:
        return 0.0
    return coef * (initial_wirelength - episode_wirelength) / max(initial_wirelength, 1e-9)


def _masked_distribution(
    logits: torch.Tensor,
    state: torch.Tensor,
    grid: int,
    wire_mask_bias: float,
    greedy_wire_mask: bool,
) -> Categorical:
    position_mask = state[:, 2].reshape(-1, grid * grid) >= 0.5
    wire_mask = state[:, 1].reshape(-1, grid * grid)
    masked_logits = torch.where(position_mask, torch.full_like(logits, -1e9), logits - wire_mask_bias * wire_mask)
    all_invalid = torch.all(position_mask, dim=-1)
    if torch.any(all_invalid):
        masked_logits = masked_logits.clone()
        masked_logits[all_invalid] = 0.0
    if greedy_wire_mask:
        valid_wire = torch.where(position_mask, torch.full_like(wire_mask, 1e9), wire_mask)
        best_wire = torch.min(valid_wire, dim=-1, keepdim=True).values
        greedy_mask = torch.where(valid_wire <= best_wire + 1e-6, torch.zeros_like(valid_wire), torch.full_like(valid_wire, -1e9))
        masked_logits = masked_logits + greedy_mask
        if torch.any(all_invalid):
            masked_logits[all_invalid] = 0.0
    return Categorical(logits=masked_logits)


def _ppo_update(
    model: _EfficientPlaceActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: _RolloutBuffer,
    device: torch.device,
    grid: int,
    last_value: float,
    gamma: float,
    gae_lambda: float,
    ppo_epochs: int,
    batch_size: int,
    clip_epsilon: float,
    entropy_coef: float,
    critic_coef: float,
    max_grad_norm: float,
    wire_mask_bias: float,
    greedy_wire_mask: bool,
) -> dict[str, float]:
    if len(buffer) == 0:
        return {"batches": 0.0}

    states = torch.as_tensor(np.stack(buffer.states), dtype=torch.float32, device=device)
    time_steps = torch.as_tensor(buffer.time_steps, dtype=torch.long, device=device)
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
    totals = {
        "batches": 0.0,
        "loss": 0.0,
        "actor_loss": 0.0,
        "critic_loss": 0.0,
        "entropy": 0.0,
        "grad_norm": 0.0,
    }
    for _ in range(ppo_epochs):
        permutation = torch.randperm(sample_count, device=device)
        for start in range(0, sample_count, batch_size):
            idx = permutation[start : start + batch_size]
            logits, value = model(states[idx], time_steps[idx])
            distribution = _masked_distribution(
                logits=logits,
                state=states[idx],
                grid=grid,
                wire_mask_bias=wire_mask_bias,
                greedy_wire_mask=greedy_wire_mask,
            )
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
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            totals["batches"] += 1.0
            totals["loss"] += float(loss.detach().item())
            totals["actor_loss"] += float(actor_loss.detach().item())
            totals["critic_loss"] += float(critic_loss.detach().item())
            totals["entropy"] += float(entropy.detach().item())
            totals["grad_norm"] += float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)

    batches = max(1.0, totals["batches"])
    for key in ("loss", "actor_loss", "critic_loss", "entropy", "grad_norm"):
        totals[key] /= batches
    return totals


def _imitation_update(
    model: _EfficientPlaceActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: _RolloutBuffer | None,
    device: torch.device,
    grid: int,
    coef: float,
    epochs: int,
    batch_size: int,
    max_grad_norm: float,
    wire_mask_bias: float,
    greedy_wire_mask: bool,
    margin: float,
) -> dict[str, float]:
    if buffer is None or len(buffer) == 0 or coef <= 0.0 or epochs <= 0:
        return {"batches": 0.0}

    states = torch.as_tensor(np.stack(buffer.states), dtype=torch.float32, device=device)
    time_steps = torch.as_tensor(buffer.time_steps, dtype=torch.long, device=device)
    actions = torch.as_tensor(buffer.actions, dtype=torch.long, device=device)
    sample_count = states.shape[0]
    totals = {
        "batches": 0.0,
        "loss": 0.0,
        "behavior_loss": 0.0,
        "margin_loss": 0.0,
        "grad_norm": 0.0,
    }

    for _ in range(epochs):
        permutation = torch.randperm(sample_count, device=device)
        for start in range(0, sample_count, batch_size):
            idx = permutation[start : start + batch_size]
            logits, _ = model(states[idx], time_steps[idx])
            distribution = _masked_distribution(
                logits=logits,
                state=states[idx],
                grid=grid,
                wire_mask_bias=wire_mask_bias,
                greedy_wire_mask=greedy_wire_mask,
            )
            selected_actions = actions[idx]
            behavior_loss = -distribution.log_prob(selected_actions).mean()
            margin_loss = torch.zeros((), dtype=behavior_loss.dtype, device=device)
            if margin > 0.0:
                masked_logits = distribution.logits
                selected_logits = masked_logits.gather(1, selected_actions.unsqueeze(1)).squeeze(1)
                other_logits = masked_logits.clone()
                other_logits.scatter_(1, selected_actions.unsqueeze(1), -1e9)
                margin_loss = F.relu(float(margin) + other_logits.max(dim=1).values - selected_logits).mean()
                behavior_loss = behavior_loss + margin_loss
            loss = coef * behavior_loss

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            totals["batches"] += 1.0
            totals["loss"] += float(loss.detach().item())
            totals["behavior_loss"] += float(behavior_loss.detach().item())
            totals["margin_loss"] += float(margin_loss.detach().item())
            totals["grad_norm"] += float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)

    batches = max(1.0, totals["batches"])
    for key in ("loss", "behavior_loss", "margin_loss", "grad_norm"):
        totals[key] /= batches
    return totals


def _reinforce_elite_until_greedy_match(
    model: _EfficientPlaceActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: _RolloutBuffer | None,
    device: torch.device,
    grid: int,
    coef: float,
    max_epochs: int,
    batch_size: int,
    max_grad_norm: float,
    wire_mask_bias: float,
    greedy_wire_mask: bool,
    margin: float,
) -> dict[str, float | bool]:
    if buffer is None or len(buffer) == 0:
        return {"reproduced": False, "epochs": 0.0}
    if _greedy_matches_buffer(
        model=model,
        buffer=buffer,
        device=device,
        grid=grid,
        wire_mask_bias=wire_mask_bias,
        greedy_wire_mask=greedy_wire_mask,
    ):
        return {"reproduced": True, "epochs": 0.0}
    if coef <= 0.0 or max_epochs <= 0:
        return {"reproduced": False, "epochs": 0.0}

    for epoch in range(max_epochs):
        _imitation_update(
            model=model,
            optimizer=optimizer,
            buffer=buffer,
            device=device,
            grid=grid,
            coef=coef,
            epochs=1,
            batch_size=batch_size,
            max_grad_norm=max_grad_norm,
            wire_mask_bias=wire_mask_bias,
            greedy_wire_mask=greedy_wire_mask,
            margin=margin,
        )
        if _greedy_matches_buffer(
            model=model,
            buffer=buffer,
            device=device,
            grid=grid,
            wire_mask_bias=wire_mask_bias,
            greedy_wire_mask=greedy_wire_mask,
        ):
            return {"reproduced": True, "epochs": float(epoch + 1)}
    return {"reproduced": False, "epochs": float(max_epochs)}


def _greedy_matches_buffer(
    model: _EfficientPlaceActorCritic,
    buffer: _RolloutBuffer,
    device: torch.device,
    grid: int,
    wire_mask_bias: float,
    greedy_wire_mask: bool,
) -> bool:
    states = torch.as_tensor(np.stack(buffer.states), dtype=torch.float32, device=device)
    time_steps = torch.as_tensor(buffer.time_steps, dtype=torch.long, device=device)
    actions = torch.as_tensor(buffer.actions, dtype=torch.long, device=device)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits, _ = model(states, time_steps)
        distribution = _masked_distribution(
            logits=logits,
            state=states,
            grid=grid,
            wire_mask_bias=wire_mask_bias,
            greedy_wire_mask=greedy_wire_mask,
        )
        predicted_actions = torch.argmax(distribution.logits, dim=-1)
        matched = bool(torch.all(predicted_actions == actions).item())
    if was_training:
        model.train()
    return matched


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
    order: tuple[str, ...],
    grid: int,
    model: _EfficientPlaceActorCritic,
    device: torch.device,
    wire_mask_scale: float,
    reward_scale: float,
    invalid_placement_penalty: float,
    wire_mask_bias: float,
    greedy_wire_mask: bool,
) -> tuple[Layout, CostResult, float]:
    env = _make_env(
        case=case,
        order=order,
        grid=grid,
        wire_mask_scale=wire_mask_scale,
        reward_scale=reward_scale,
        invalid_placement_penalty=invalid_placement_penalty,
    )
    state_np = env.reset()
    model.eval()
    with torch.no_grad():
        while not env.is_done():
            state = torch.as_tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
            time_tensor = torch.as_tensor([env.t], dtype=torch.long, device=device)
            logits, _ = model(state, time_tensor)
            distribution = _masked_distribution(
                logits=logits,
                state=state,
                grid=grid,
                wire_mask_bias=wire_mask_bias,
                greedy_wire_mask=greedy_wire_mask,
            )
            action = int(torch.argmax(distribution.logits, dim=-1).item())
            state_np, _, _, _ = env.step(action)
    model.train()
    layout = env.layout(initial_layout)
    return layout, objective(layout), _wirelength(case, layout)


def _wirelength(case: FloorplanCase, layout: Layout) -> float:
    return float(hpwl(case, layout))


def _is_legal(case: FloorplanCase, layout: Layout) -> bool:
    return total_overlap_penalty(case, layout) <= 1e-9 and total_outline_penalty(case, layout) <= 1e-9
