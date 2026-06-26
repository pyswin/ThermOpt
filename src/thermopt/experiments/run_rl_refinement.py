"""Experiment: RL local search refinement from a baseline optimizer.

Pipeline per case:
  1. Run baseline optimizer to get an initial layout.
     Supported baselines: atplace, atmplace, milp_only.
     milp_only = atplace with refine_steps=0 and legal_perturb_iterations=0
     (MILP placement only, no gradient refinement, no SA perturbation).
  2. Run RL local search starting from that layout.
  3. Report wirelength comparison against baseline and paper reference numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from thermopt.data.inputs import CaseInput
from thermopt.experiments.run_v0_sa import load_inputs
from thermopt.layout.geometry import hpwl
from thermopt.layout.objects import FloorplanCase, Layout, Placement
from thermopt.layout.visualization import save_cost_curve, save_layout_figure
from thermopt.objective.cost import Objective
from thermopt.optimizer import atplace, atmplace
from thermopt.optimizer.rl_local_search import optimize as rl_optimize
from thermopt.thermal.backend import build_thermal_backend

# Paper reference WL values (ATMPlace arXiv 2511.17319 Table VI, WL-driven TWL/m).
# Multiply by 1000 to convert to our internal mm units for comparison.
PAPER_ATM_WL_M: dict[str, float] = {
    "Case1": 11.973,
    "Case2": 15.408,
    "Case3": 32.350,
    "Case4": 46.982,
    "Case5": 48.265,
    "Case6": 29.482,
    "Case7": 12.523,
    "Case8": 9.001,
}


def _greedy_connectivity_layout(
    case: FloorplanCase,
    rng: np.random.Generator,
    objective: object | None = None,
    sa_steps: int = 0,
) -> Layout:
    """Connectivity-aware initial layout with domain-prior pairing + optional SA refinement.

    Three phases:
    1. Build pairwise net-weight affinity; detect "strong pairs" (mutual best-partner,
       e.g. GPU↔HBM, CPU↔Analog) -- a classic chiplet co-placement heuristic.
    2. Connectivity-guided ordering with immediate-partner insertion: chips are ordered
       by total affinity to already-placed chips, but when a chip is selected its strong
       partner (if any) is inserted immediately after.  This keeps paired chips adjacent
       while preserving the connectivity-driven global order (avoids the "place all pairs
       first" trap that pushes strongly-connected but unpaired chips far apart).
    3. Strip-pack the ordering into rows (landscape orientation), then optionally run
       SA coordinate refinement to tighten the result before RL takes over.

    SA temperature is calibrated to the normalised-cost scale (~0.001 per 1% WL change)
    so overlap-creating moves (penalty ~0.5) are rejected while WL-improving moves are
    explored freely.
    """
    chip_by_id = case.chiplet_by_id

    # ------------------------------------------------------------------
    # Phase 1 – pairwise affinity
    # ------------------------------------------------------------------
    affinity: dict[str, dict[str, float]] = {c.id: {} for c in case.chiplets}
    for net in case.nets:
        chips = list(set(net.chiplets))
        for i in range(len(chips)):
            for j in range(i + 1, len(chips)):
                a, b = chips[i], chips[j]
                affinity[a][b] = affinity[a].get(b, 0.0) + 1.0
                affinity[b][a] = affinity[b].get(a, 0.0) + 1.0

    total_affinity = {c.id: sum(affinity[c.id].values()) for c in case.chiplets}

    # ------------------------------------------------------------------
    # Phase 2 – detect strong pairs (mutual best-partner)
    # Two chips form a "strong pair" iff each is the other's single highest-affinity
    # neighbor.  Examples: GPU↔HBM (high-bandwidth memory), CPU↔Analog.
    # ------------------------------------------------------------------
    best_partner: dict[str, str] = {}
    for cid, nbrs in affinity.items():
        if nbrs:
            best_partner[cid] = max(nbrs, key=nbrs.__getitem__)

    strong_partner: dict[str, str] = {}  # a→b means a and b are a mutual best-pair
    seen_pairs: set[frozenset[str]] = set()
    for a, b in best_partner.items():
        if best_partner.get(b) == a:
            key = frozenset({a, b})
            if key not in seen_pairs:
                strong_partner[a] = b
                strong_partner[b] = a
                seen_pairs.add(key)

    # ------------------------------------------------------------------
    # Phase 3 – connectivity-guided ordering with partner pull-in
    # Start with the highest-affinity chip; greedily add the chip most connected
    # to the placed set.  Whenever a chip is selected, if it has an unplaced strong
    # partner, that partner is inserted immediately afterwards.
    # ------------------------------------------------------------------
    unplaced = {c.id for c in case.chiplets}
    ordering: list[str] = []
    placed: set[str] = set()

    def _enqueue(cid: str) -> None:
        ordering.append(cid)
        unplaced.discard(cid)
        placed.add(cid)
        # pull-in partner immediately to keep pairs adjacent
        partner = strong_partner.get(cid)
        if partner and partner in unplaced:
            ordering.append(partner)
            unplaced.discard(partner)
            placed.add(partner)

    start = max(unplaced, key=lambda c: total_affinity.get(c, 0.0))
    _enqueue(start)

    while unplaced:
        best = max(
            unplaced,
            key=lambda c: (sum(affinity[c].get(p, 0.0) for p in placed), total_affinity.get(c, 0.0)),
        )
        _enqueue(best)

    # ------------------------------------------------------------------
    # Phase 4 – strip (shelf) packing
    # Place chips left-to-right in landscape orientation; start a new row when
    # the current one is full.
    # ------------------------------------------------------------------
    placements: list[Placement] = []
    cur_x, cur_y, row_h = 0.0, 0.0, 0.0

    for cid in ordering:
        chiplet = chip_by_id[cid]
        w, h, rot = chiplet.width, chiplet.height, 0
        if h > w:   # landscape orientation uses less row height
            w, h, rot = h, w, 90
        if cur_x + w > case.outline_width + 1e-6 and cur_x > 1e-6:
            cur_y += row_h
            cur_x, row_h = 0.0, 0.0
        x = min(cur_x, max(0.0, case.outline_width - w))
        placements.append(Placement(cid, x, cur_y, rot))
        cur_x += w
        row_h = max(row_h, h)

    layout = Layout(tuple(placements))

    # ------------------------------------------------------------------
    # Phase 5 – optional SA-style coordinate refinement
    # Temperature calibrated to normalised-cost scale: typical WL delta per step
    # ~0.001; overlap delta ~0.5.  t_start=0.05 → accept most WL-improving moves
    # but reject almost all overlap-creating moves.
    # ------------------------------------------------------------------
    if sa_steps > 0 and objective is not None:
        layout = _sa_refine(case, layout, objective, rng, sa_steps)

    return layout


def _sa_refine(
    case: FloorplanCase,
    layout: Layout,
    objective: object,
    rng: np.random.Generator,
    steps: int,
) -> Layout:
    """Lightweight SA coordinate refinement to polish the greedy initial layout."""
    from thermopt.optimizer.rl_local_search import _sample_move

    current_layout = layout
    current_cost = objective(current_layout)  # type: ignore[operator]
    best_layout = current_layout
    best_cost = current_cost

    # Temperature calibrated to normalised cost scale.
    # Overlap penalty ~0.5 per tiny overlap >> WL delta ~0.001 per 1% WL improvement.
    # t_start=0.05: P(accept overlap-creating move)≈exp(-0.5/0.05)≈0 → nearly rejected.
    # t_end=0.002: greedy acceptance at the end.
    t_start, t_end = 0.05, 0.002
    move_scale_start, move_scale_end = 1.0, 0.2

    for step in range(steps):
        progress = step / max(steps - 1, 1)
        temp = t_start * (t_end / t_start) ** progress
        move_scale = move_scale_start + (move_scale_end - move_scale_start) * progress

        candidate, _ = _sample_move(case, current_layout, rng, move_scale)
        cand_cost = objective(candidate)  # type: ignore[operator]

        delta = cand_cost.total - current_cost.total
        if delta < 0 or rng.random() < float(np.exp(-delta / max(temp, 1e-9))):
            current_layout = candidate
            current_cost = cand_cost
            if cand_cost.total < best_cost.total:
                best_layout = candidate
                best_cost = cand_cost

    return best_layout


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_output_dir(config: dict) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = config.get("experiment_name", "rl_refinement")
    output_dir = Path(config.get("output_root", "outputs")) / f"{stamp}_{name}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def run_single_case(
    config: dict,
    config_path: Path,
    case_input: CaseInput,
    output_dir: Path,
    seed: int,
) -> dict:
    case = case_input.case
    initial_layout = case_input.layout
    thermal_backend = build_thermal_backend(
        case, config["thermal"], work_dir=output_dir / "_thermal",
    )
    objective = Objective(
        case, config["thermal"], config["objective"],
        initial_layout, thermal_backend=thermal_backend,
    )

    save_layout_figure(
        case, initial_layout, output_dir / "random_initial.png", "Random initial",
    )

    # --- Step 1: baseline optimizer ---
    baseline_name = str(config.get("baseline_optimizer", "atplace"))
    rng_baseline = np.random.default_rng(seed)

    print(f"  [{case_input.name}] running baseline={baseline_name} ...")
    t0 = time.perf_counter()

    if baseline_name == "greedy_connectivity":
        # Connectivity-aware greedy layout + optional SA refinement.
        # sa_steps=0 → pure greedy; sa_steps>0 → greedy then SA polish.
        greedy_cfg = config.get("greedy_connectivity", {})
        sa_steps = int(greedy_cfg.get("sa_steps", 2000))
        baseline_layout = _greedy_connectivity_layout(
            case, rng_baseline, objective=objective, sa_steps=sa_steps,
        )
        baseline_cost = objective(baseline_layout)
        baseline_time = time.perf_counter() - t0
    else:
        if baseline_name == "milp_only":
            baseline_fn = atplace.optimize
            raw_cfg = config.get("milp_only", config.get("atplace", {}))
            baseline_config = dict(raw_cfg)
            baseline_config["refine_steps"] = 0
            baseline_config["legal_perturb_iterations"] = 0
        elif baseline_name == "atmplace":
            baseline_fn = atmplace.optimize
            baseline_config = config.get("atmplace", {})
        else:
            baseline_fn = atplace.optimize
            baseline_config = config.get("atplace", {})
        baseline_result = baseline_fn(case, initial_layout, objective, baseline_config, seed)
        baseline_time = time.perf_counter() - t0
        baseline_layout = baseline_result.best_layout
        baseline_cost = baseline_result.best_cost

    baseline_wl = float(hpwl(case, baseline_layout))
    print(
        f"  [{case_input.name}] baseline done  "
        f"wl={baseline_wl:.2f}  cost={baseline_cost.total:.4f}  "
        f"time={baseline_time:.1f}s"
    )

    save_layout_figure(
        case, baseline_layout,
        output_dir / f"baseline_{baseline_name}.png",
        f"Baseline: {baseline_name}",
    )
    baseline_curve = [baseline_cost.total]
    if baseline_name != "greedy_connectivity":
        baseline_curve = baseline_result.best_curve  # type: ignore[union-attr]
    save_cost_curve(
        baseline_curve,
        output_dir / f"cost_curve_{baseline_name}.png",
        f"Cost curve: {baseline_name}",
    )

    # --- Step 2: RL local search refinement ---
    rl_config = config.get("rl_local_search", {})
    print(f"  [{case_input.name}] running RL local search refinement ...")
    t0 = time.perf_counter()
    rl_result = rl_optimize(
        case, baseline_layout, objective, rl_config, seed + 5000,
    )
    rl_time = time.perf_counter() - t0
    rl_wl = rl_result.final_wl
    print(
        f"  [{case_input.name}] RL done  "
        f"wl={rl_wl:.2f}  cost={rl_result.best_cost.total:.4f}  "
        f"accepted={rl_result.accepted_ratio:.2%}  time={rl_time:.1f}s"
    )

    save_layout_figure(
        case, rl_result.best_layout,
        output_dir / "rl_refined.png", "RL refined",
    )
    save_cost_curve(
        rl_result.best_curve,
        output_dir / "cost_curve_rl.png", "Cost curve: RL local search",
    )
    if rl_result.training_loss_curve:
        save_cost_curve(
            rl_result.training_loss_curve,
            output_dir / "rl_loss_curve.png", "DQN training loss",
        )

    # --- Step 3: comparison ---
    wl_delta = rl_wl - baseline_wl
    wl_pct = wl_delta / max(baseline_wl, 1e-9) * 100.0
    improved = wl_delta < -1e-6
    paper_wl_m = PAPER_ATM_WL_M.get(case_input.name)
    paper_wl = paper_wl_m * 1000.0 if paper_wl_m is not None else None
    paper_vs_rl_pct = (rl_wl - paper_wl) / paper_wl * 100.0 if paper_wl else None
    print(
        f"  [{case_input.name}] WL change: {baseline_wl:.2f} -> {rl_wl:.2f}  "
        f"({wl_pct:+.2f}%)  {'IMPROVED' if improved else 'no improvement'}"
    )
    if paper_wl is not None:
        print(
            f"  [{case_input.name}] vs paper ATMPlace: {paper_wl:.2f}  "
            f"RL is {paper_vs_rl_pct:+.2f}% vs paper"
        )

    summary = {
        "case": case_input.name,
        "num_chiplets": len(case.chiplets),
        "num_nets": len(case.nets),
        "baseline_optimizer": baseline_name,
        "baseline_wl": baseline_wl,
        "baseline_wl_m": baseline_wl / 1000.0,
        "baseline_cost": float(baseline_cost.total),
        "baseline_time_sec": baseline_time,
        "rl_wl": rl_wl,
        "rl_wl_m": rl_wl / 1000.0,
        "rl_cost": float(rl_result.best_cost.total),
        "rl_time_sec": rl_time,
        "rl_accepted_ratio": rl_result.accepted_ratio,
        "rl_total_steps": rl_result.attempted_moves,
        "wl_delta": wl_delta,
        "wl_change_pct": wl_pct,
        "improved": improved,
        "paper_atm_wl_m": paper_wl_m,
        "paper_vs_baseline_pct": (baseline_wl - paper_wl) / paper_wl * 100.0 if paper_wl else None,
        "paper_vs_rl_pct": paper_vs_rl_pct,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def run(config_path: Path) -> Path:
    config = load_config(config_path)
    output_dir = make_output_dir(config)
    shutil.copy2(config_path, output_dir / "config.yaml")

    seed = int(config.get("seed", 17))
    inputs = load_inputs(config, seed)
    summaries: list[dict] = []

    for index, case_input in enumerate(inputs):
        case_dir = output_dir
        if len(inputs) > 1:
            case_dir = output_dir / case_input.name
            case_dir.mkdir(parents=True, exist_ok=False)
        print(f"[rl_refinement] === {case_input.name} ===")
        summary = run_single_case(
            config, config_path, case_input, case_dir, seed + index * 1000,
        )
        summaries.append(summary)

    # --- aggregate ---
    if len(summaries) > 1:
        with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump({"config": str(config_path), "cases": summaries}, f, indent=2)

        fieldnames = list(summaries[0].keys())
        with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summaries:
                writer.writerow(row)

        _print_summary_table(summaries)

    print(f"[rl_refinement] saved outputs to {output_dir}")
    return output_dir


def _print_summary_table(summaries: list[dict]) -> None:
    header = f"{'Case':<8} {'#chips':>6} {'MILP(m)':>10} {'RL(m)':>10} {'ΔRL%':>8} {'Paper(m)':>10} {'ΔPaper%':>9}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for s in summaries:
        paper = s.get("paper_atm_wl_m")
        paper_str = f"{paper:>10.3f}" if paper else f"{'N/A':>10}"
        paper_vs = s.get("paper_vs_rl_pct")
        paper_vs_str = f"{paper_vs:>+9.2f}" if paper_vs is not None else f"{'N/A':>9}"
        print(
            f"{s['case']:<8} {s['num_chiplets']:>6} "
            f"{s['baseline_wl_m']:>10.3f} {s['rl_wl_m']:>10.3f} "
            f"{s['wl_change_pct']:>+8.2f} "
            f"{paper_str} {paper_vs_str}"
        )
    print("=" * len(header))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RL local search refinement from baseline optimizer.",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/rl_refinement.yaml"),
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
