"""RL-guided local search optimizer for chiplet floorplanning.

Starts from a strong initial solution and uses a DQN agent to learn which
coordinate-space perturbation moves to accept -- replacing SA's temperature-based
acceptance criterion.

At each step K candidate perturbations (translate / swap / rotate / perturb)
are sampled.  The DQN scores each candidate and the agent picks the best
(or rejects all).  Training uses prioritized experience replay.

Design references:
- ICCD 2020  "Learn to Floorplan through Acquisition of Effective Local Search Heuristics"
- ICML 2024  EfficientPlace: prioritized replay for sample efficiency
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from thermopt.layout.geometry import hpwl
from thermopt.layout.objects import FloorplanCase, Layout, Placement
from thermopt.objective.cost import CostResult

FEATURE_DIM = 10

# coordinate-space move palette (same as SA, but selected by DQN not temperature)
_MOVE_NAMES = ("translate", "swap", "rotate", "perturb")
_MOVE_PROBS = np.array([0.35, 0.25, 0.15, 0.25])


@dataclass(frozen=True)
class RLLocalSearchResult:
    best_layout: Layout
    best_cost: CostResult
    best_curve: list[float]
    accepted_moves: int
    attempted_moves: int
    training_loss_curve: list[float]
    baseline_wl: float
    final_wl: float

    @property
    def accepted_ratio(self) -> float:
        return self.accepted_moves / max(1, self.attempted_moves)


# ---------------------------------------------------------------------------
# Coordinate-space perturbation
# ---------------------------------------------------------------------------

def _sample_move(
    case: FloorplanCase,
    layout: Layout,
    rng: np.random.Generator,
    move_scale: float,
) -> tuple[Layout, set[str]]:
    """One coordinate-space perturbation.  Returns (new_layout, affected_ids)."""
    move = str(rng.choice(_MOVE_NAMES, p=_MOVE_PROBS))
    placements = list(layout.placements)
    idx = int(rng.integers(0, len(placements)))
    affected: set[str] = set()

    if move == "swap" and len(placements) >= 2:
        j = int(rng.integers(0, len(placements) - 1))
        if j >= idx:
            j += 1
        a, b = placements[idx], placements[j]
        placements[idx] = Placement(a.chiplet_id, b.x, b.y, a.rotation)
        placements[j] = Placement(b.chiplet_id, a.x, a.y, b.rotation)
        affected = {a.chiplet_id, b.chiplet_id}
    elif move == "rotate":
        placements[idx] = placements[idx].rotated()
        affected = {placements[idx].chiplet_id}
    else:
        p = placements[idx]
        chiplet = case.chiplet_by_id[p.chiplet_id]
        scale = move_scale if move == "translate" else move_scale * 0.35
        dx, dy = float(rng.normal(0, scale)), float(rng.normal(0, scale))
        w, h = p.rotated_size(chiplet)
        margin = max(case.outline_width, case.outline_height) * 0.1
        x = float(np.clip(p.x + dx, -margin, case.outline_width - w + margin))
        y = float(np.clip(p.y + dy, -margin, case.outline_height - h + margin))
        placements[idx] = p.moved(x=x, y=y)
        affected = {p.chiplet_id}

    return Layout(tuple(placements)), affected


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _compute_features(
    e_curr: float,
    e_neighbor: float,
    e_best: float,
    e_init: float,
    e_avg: float,
    affected_ratio: float,
    progress: float,
    metrics: dict[str, float],
) -> np.ndarray:
    inv = 1.0 / max(e_init, 1e-9)
    return np.array(
        [
            e_curr * inv,
            e_neighbor * inv,
            e_best * inv,
            (e_curr - e_neighbor) * inv,
            e_avg * inv,
            affected_ratio,
            progress,
            metrics.get("overlap_penalty", 0.0),
            metrics.get("outline_penalty", 0.0),
            1.0,
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Prioritized Replay Buffer
# ---------------------------------------------------------------------------

class _ReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.cap = capacity
        self.alpha = alpha
        self.feat: list[np.ndarray] = []
        self.rew: list[float] = []
        self.mnq: list[float] = []
        self.done: list[bool] = []
        self.prio: list[float] = []
        self._pos = 0

    def push(self, f: np.ndarray, r: float, mnq: float, *, done: bool) -> None:
        d = done
        p = max(self.prio) if self.prio else 1.0
        if len(self.feat) < self.cap:
            self.feat.append(f); self.rew.append(r)
            self.mnq.append(mnq); self.done.append(d); self.prio.append(p)
        else:
            i = self._pos
            self.feat[i] = f; self.rew[i] = r
            self.mnq[i] = mnq; self.done[i] = d; self.prio[i] = p
        self._pos = (self._pos + 1) % self.cap

    def sample(self, bs: int, beta: float, rng: np.random.Generator):
        n = len(self.feat)
        pr = np.array(self.prio[:n]) ** self.alpha
        pr /= pr.sum()
        idx = rng.choice(n, size=min(bs, n), p=pr, replace=False)
        w = (n * pr[idx]) ** (-beta); w /= w.max()
        return (
            np.array([self.feat[i] for i in idx]),
            np.array([self.rew[i] for i in idx], dtype=np.float32),
            np.array([self.mnq[i] for i in idx], dtype=np.float32),
            np.array([float(self.done[i]) for i in idx], dtype=np.float32),
            idx,
            w.astype(np.float32),
        )

    def update_prio(self, idx: np.ndarray, td: np.ndarray) -> None:
        for i, t in zip(idx, td):
            self.prio[int(i)] = float(abs(t)) + 1e-6

    def __len__(self) -> int:
        return len(self.feat)


# ---------------------------------------------------------------------------
# Q-Network  (numpy 2-layer MLP)
# ---------------------------------------------------------------------------

class _QNet:
    def __init__(self, d_in: int, d_h: int, lr: float, seed: int):
        g = np.random.default_rng(seed)
        self.w1 = (g.standard_normal((d_in, d_h)) * np.sqrt(2.0 / d_in)).astype(np.float32)
        self.b1 = np.zeros(d_h, dtype=np.float32)
        self.w2 = (g.standard_normal((d_h, 1)) * np.sqrt(2.0 / d_h)).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)
        self.w1t = self.w1.copy(); self.b1t = self.b1.copy()
        self.w2t = self.w2.copy(); self.b2t = self.b2.copy()
        self.lr = lr

    def q(self, x: np.ndarray) -> np.ndarray:
        return (np.maximum(0, x @ self.w1 + self.b1) @ self.w2 + self.b2).ravel()

    def q_target(self, x: np.ndarray) -> np.ndarray:
        return (np.maximum(0, x @ self.w1t + self.b1t) @ self.w2t + self.b2t).ravel()

    def step(self, x: np.ndarray, tgt: np.ndarray, w: np.ndarray):
        b = x.shape[0]
        pre = x @ self.w1 + self.b1; h = np.maximum(0, pre)
        q = (h @ self.w2 + self.b2).ravel()
        td = q - tgt; loss = float(np.mean(w * td ** 2))
        dq = (2.0 * w * td / b).reshape(-1, 1)
        dw2 = h.T @ dq; db2 = dq.sum(0)
        dh = dq @ self.w2.T; dh *= (pre > 0).astype(np.float32)
        dw1 = x.T @ dh; db1 = dh.sum(0)
        for g in (dw1, db1, dw2, db2):
            np.clip(g, -1, 1, out=g)
        self.w1 -= self.lr * dw1; self.b1 -= self.lr * db1
        self.w2 -= self.lr * dw2; self.b2 -= self.lr * db2
        return loss, td

    def soft_update(self, tau: float) -> None:
        self.w1t += tau * (self.w1 - self.w1t); self.b1t += tau * (self.b1 - self.b1t)
        self.w2t += tau * (self.w2 - self.w2t); self.b2t += tau * (self.b2 - self.b2t)


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

def optimize(
    case: FloorplanCase,
    initial_layout: Layout,
    objective: Callable[[Layout], CostResult],
    config: dict,
    seed: int,
) -> RLLocalSearchResult:
    rng = np.random.default_rng(seed)

    K            = int(config.get("num_candidates", 5))
    total_steps  = int(config.get("total_steps", 800))
    eps0         = float(config.get("epsilon_start", 0.3))
    eps1         = float(config.get("epsilon_end", 0.05))
    hidden       = int(config.get("hidden_dim", 64))
    lr           = float(config.get("learning_rate", 0.001))
    gamma        = float(config.get("gamma", 0.99))
    buf_cap      = int(config.get("buffer_capacity", 5000))
    bs           = int(config.get("batch_size", 32))
    train_after  = int(config.get("train_after", 64))
    tgt_freq     = int(config.get("target_update_freq", 10))
    tau          = float(config.get("tau", 0.01))
    per_a        = float(config.get("per_alpha", 0.6))
    per_b0       = float(config.get("per_beta_start", 0.4))
    per_b1       = float(config.get("per_beta_end", 1.0))
    rej_pen      = float(config.get("reject_penalty", -0.01))
    sub_pen      = float(config.get("suboptimal_accept_penalty", -0.01))
    move_scale   = float(config.get("move_scale", 0.5))
    restart_iv   = int(config.get("restart_interval", 0))
    report_every = max(1, int(config.get("report_every", 50)))

    # --- start directly from the given layout (no SP conversion) ---
    current_layout = initial_layout
    current_cost   = objective(current_layout)
    baseline_wl    = float(hpwl(case, initial_layout))
    e_init         = max(current_cost.total, 1e-9)

    best_layout = current_layout
    best_cost   = current_cost
    best_curve: list[float] = [best_cost.total]

    net = _QNet(FEATURE_DIM, hidden, lr, seed)
    buf = _ReplayBuffer(buf_cap, per_a)
    loss_curve: list[float] = []

    prev_feat: np.ndarray | None = None
    prev_rew = 0.0
    accepted = 0
    improvements = 0

    for step in range(total_steps):
        prog = step / max(1, total_steps - 1)
        eps  = eps0 + (eps1 - eps0) * prog
        pb   = per_b0 + (per_b1 - per_b0) * prog

        # --- periodic restart to best ---
        if restart_iv > 0 and step > 0 and step % restart_iv == 0:
            current_layout = best_layout
            current_cost = best_cost

        # --- sample K candidates ---
        cands: list[tuple[Layout, CostResult, float]] = []
        for _ in range(K):
            lay, affected = _sample_move(case, current_layout, rng, move_scale)
            cost = objective(lay)
            area = sum(
                case.chiplet_by_id[c].width * case.chiplet_by_id[c].height
                for c in affected if c in case.chiplet_by_id
            )
            cands.append((lay, cost, area / max(case.total_chiplet_area, 1e-9)))

        cc = [c[1].total for c in cands]
        e_avg = float(np.mean(cc))

        feats = np.array([
            _compute_features(
                current_cost.total, c[1].total, best_cost.total,
                e_init, e_avg, c[2], prog, c[1].metrics,
            )
            for c in cands
        ] + [
            _compute_features(
                current_cost.total, current_cost.total, best_cost.total,
                e_init, e_avg, 0.0, prog, current_cost.metrics,
            )
        ])

        qv = net.q(feats)
        max_q = float(np.max(qv))

        if prev_feat is not None:
            buf.push(prev_feat, prev_rew, max_q, done=False)

        action = int(rng.integers(0, K + 1)) if rng.random() < eps else int(np.argmax(qv))

        if action < K:
            lay, cost, _ = cands[action]
            reward = (current_cost.total - cost.total) / e_init
            if cost.total > 1.2 * min(cc):
                reward += sub_pen
            if cost.total < best_cost.total:
                reward += (best_cost.total - cost.total) / e_init
                best_layout = lay; best_cost = cost
                improvements += 1
            current_layout = lay; current_cost = cost
            accepted += 1
        else:
            reward = rej_pen
            if min(cc) < current_cost.total:
                reward += rej_pen

        prev_feat = feats[action]
        prev_rew = reward

        if len(buf) >= train_after:
            fb, rb, mb, db, ix, wb = buf.sample(bs, pb, rng)
            tgt = rb + gamma * mb * (1.0 - db)
            loss, td = net.step(fb, tgt, wb)
            buf.update_prio(ix, td)
            loss_curve.append(loss)
            if step % tgt_freq == 0:
                net.soft_update(tau)

        if step % report_every == 0 or step == total_steps - 1:
            best_curve.append(best_cost.total)

    if prev_feat is not None:
        buf.push(prev_feat, prev_rew, 0.0, done=True)

    return RLLocalSearchResult(
        best_layout=best_layout,
        best_cost=best_cost,
        best_curve=best_curve,
        accepted_moves=accepted,
        attempted_moves=total_steps,
        training_loss_curve=loss_curve,
        baseline_wl=baseline_wl,
        final_wl=float(hpwl(case, best_layout)),
    )
