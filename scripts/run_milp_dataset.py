"""
run_milp_dataset.py — Cases 1-8 MILP-init snapshot + perturbation dataset.

Phase A (per case): solve the Phase-1 MILP init with Gurobi (100s budget),
recording a position-only snapshot roughly every second of solve time via a
MIPSOL callback (see thermopt.optimizer.milp_wl_gurobi). No thermal
simulation is run anywhere in this script.

Phase B (per case): pick a handful of seed snapshots spread across the solve
trajectory, then generate legal perturbations around each seed using
swap / move(small) / move(more) / rotate (rotate only applies to
non-square chiplets). Every perturbation is re-legalized via sequence-pair
packing before being recorded.

Output layout (single run timestamp shared by both phases):
    atplace/milp_wl_dataset/<STAMP>/
        run_summary.json
        intermediate/CaseN/snapshots.json, trajectory.png, montage_snapshots.png
        perturb/CaseN/perturbations.json, distribution.png, examples.png, montage_<type>.png

Usage (must run under an env with gurobipy + a valid license, e.g.):
    /data/xinli/anaconda3/envs/atplace_py39/bin/python scripts/run_milp_dataset.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from thermopt.data.atplace import load_atplace_case
from thermopt.data.dataset_io import layout_from_chiplets_json
from thermopt.layout.geometry import hpwl, total_outline_penalty, total_overlap_penalty
from thermopt.layout.objects import FloorplanCase, Layout
from thermopt.layout.visualization import draw_layout, draw_layout_light
from thermopt.optimizer.milp_wl_gurobi import Snapshot, initialize_with_snapshots
from thermopt.optimizer import perturb

CASES_DIR = ROOT / "external/ATPlace_pub/cases"
CASES = [f"Case{i}" for i in range(1, 9)]
HARD_CASES = {"Case6", "Case7", "Case8"}
SAMPLES_DEFAULT = 1000
SAMPLES_HARD = 1500
N_SEEDS = 5
TIME_LIMIT = 100.0
SNAPSHOT_INTERVAL = 1.0
SEED = 42


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "atplace" / "milp_wl_dataset" / stamp
    inter_dir = run_dir / "intermediate"
    perturb_dir = run_dir / "perturb"
    inter_dir.mkdir(parents=True, exist_ok=True)
    perturb_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "milp_time_limit": TIME_LIMIT,
        "mip_rel_gap": 0.001,
        "snapshot_interval": SNAPSHOT_INTERVAL,
        "seed": SEED,
        "verbose": False,
    }
    print(f"Run dir: {run_dir}")

    run_summary: dict[str, dict] = {}
    t_total = time.time()

    for case_name in CASES:
        case_dir = CASES_DIR / case_name
        if not case_dir.exists():
            print(f"{case_name}: SKIP (no case dir)")
            continue

        print(f"\n=== {case_name} ===")
        ci = load_atplace_case(case_dir, {}, seed=SEED)
        case = ci.case
        n = len(case.chiplets)

        t0 = time.time()
        result, snapshots = initialize_with_snapshots(case, config)
        solve_t = time.time() - t0
        print(
            f"  MILP: success={result.solver_success} obj={result.solver_objective:.1f} "
            f"solve_t={solve_t:.1f}s n_snapshots={len(snapshots)}"
        )

        case_inter_dir = inter_dir / case_name
        case_inter_dir.mkdir(parents=True, exist_ok=True)
        snap_records = _save_snapshots(case, snapshots, case_inter_dir)
        _plot_trajectory(case, snapshots, case_inter_dir / "trajectory.png", case_name)
        _plot_snapshot_montage(case, snapshots, case_inter_dir / "montage_snapshots.png", case_name)

        n_samples = SAMPLES_HARD if case_name in HARD_CASES else SAMPLES_DEFAULT
        seed_indices = _pick_seed_indices(snapshots, N_SEEDS)
        case_perturb_dir = perturb_dir / case_name
        case_perturb_dir.mkdir(parents=True, exist_ok=True)
        perturb_records = _generate_perturbations(case, snapshots, seed_indices, n_samples, seed=SEED)
        (case_perturb_dir / "perturbations.json").write_text(json.dumps(perturb_records, indent=2))
        _plot_distribution(case, perturb_records, case_perturb_dir / "distribution.png", case_name)
        _plot_examples(case, perturb_records, case_perturb_dir / "examples.png", case_name)
        for move_type in (perturb.SWAP, perturb.MOVE_SMALL, perturb.MOVE_MORE, perturb.ROTATE):
            _plot_type_montage(case, perturb_records, move_type, case_perturb_dir / f"montage_{move_type}.png", case_name)

        legal_count = sum(1 for r in perturb_records if r["legal"])
        print(f"  Perturb: {len(perturb_records)} samples, {legal_count} legal, seeds@t={[round(snapshots[i].t,1) for i in seed_indices]}")

        run_summary[case_name] = {
            "n_chiplets": n,
            "milp_solver_success": result.solver_success,
            "milp_solver_objective": result.solver_objective,
            "milp_solve_time_s": solve_t,
            "n_snapshots": len(snapshots),
            "n_perturb_samples": len(perturb_records),
            "n_perturb_legal": legal_count,
            "seed_snapshot_indices": seed_indices,
        }

    total_t = time.time() - t_total
    (run_dir / "run_summary.json").write_text(
        json.dumps({"config": config, "n_seeds": N_SEEDS, "cases": run_summary, "total_time_s": total_t}, indent=2)
    )
    print(f"\nDone in {total_t/60:.1f} min. Output: {run_dir}")


def _save_snapshots(case: FloorplanCase, snapshots: list[Snapshot], out_dir: Path) -> list[dict]:
    records = []
    for i, snap in enumerate(snapshots):
        overlap = total_overlap_penalty(case, snap.layout)
        outline = total_outline_penalty(case, snap.layout)
        records.append(
            {
                "index": i,
                "t": snap.t,
                "overlap": overlap,
                "outline": outline,
                "chiplets": _layout_to_json(case, snap.layout),
            }
        )
    (out_dir / "snapshots.json").write_text(json.dumps(records, indent=2))
    return records


def _layout_to_json(case: FloorplanCase, layout: Layout) -> list[dict]:
    out = []
    for p in layout.placements:
        chiplet = case.chiplet_by_id[p.chiplet_id]
        w, h = p.rotated_size(chiplet)
        out.append(
            {
                "name": p.chiplet_id,
                "x_mm": round(p.x, 6),
                "y_mm": round(p.y, 6),
                "cx_mm": round(p.x + 0.5 * w, 6),
                "cy_mm": round(p.y + 0.5 * h, 6),
                "width_mm": round(w, 6),
                "height_mm": round(h, 6),
                "rotation": p.rotation,
                "power": chiplet.power,
            }
        )
    return out


def _layouts_equal(a: Layout, b: Layout, tol: float = 1e-9) -> bool:
    ba, bb = a.by_id, b.by_id
    return all(
        abs(ba[cid].x - bb[cid].x) < tol and abs(ba[cid].y - bb[cid].y) < tol and ba[cid].rotation == bb[cid].rotation
        for cid in ba
    )


def _unique_snapshot_indices(snapshots: list[Snapshot]) -> list[int]:
    """Collapse runs of consecutive identical snapshots (the solver found no better
    incumbent in between, so the same position got re-saved to keep the ~1s cadence)
    down to one representative index each."""
    indices = []
    prev_layout = None
    for i, snap in enumerate(snapshots):
        if prev_layout is None or not _layouts_equal(prev_layout, snap.layout):
            indices.append(i)
        prev_layout = snap.layout
    return indices


def _pick_seed_indices(snapshots: list[Snapshot], n_seeds: int) -> list[int]:
    """Pick seed indices evenly spread across the *distinct* states in the solve
    trajectory, so seeds aren't wasted repeating a plateau where the incumbent
    didn't change for a while."""
    unique = _unique_snapshot_indices(snapshots)
    if len(unique) <= 1:
        return unique or [0]
    if len(unique) <= n_seeds:
        return unique
    fracs = np.linspace(0.0, 1.0, n_seeds)
    return sorted(set(unique[int(round(f * (len(unique) - 1)))] for f in fracs))


def _generate_perturbations(
    case: FloorplanCase,
    snapshots: list[Snapshot],
    seed_indices: list[int],
    n_samples: int,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    n = len(case.chiplets)
    small_k = (1, max(1, n // 6))
    more_k = (max(2, n // 6 + 1), max(2, n // 2))

    records = []
    for sample_id in range(n_samples):
        seed_idx = seed_indices[sample_id % len(seed_indices)]
        seed_snapshot = snapshots[seed_idx]
        result = perturb.generate_one(case, seed_snapshot.layout, rng, small_k, more_k)
        records.append(
            {
                "id": sample_id,
                "seed_snapshot_index": seed_idx,
                "seed_t": seed_snapshot.t,
                "move_type": result.move_type,
                "touched": list(result.touched),
                "overlap": result.overlap,
                "outline": result.outline,
                "legal": result.legal,
                "chiplets": _layout_to_json(case, result.layout),
            }
        )
    return records


def _plot_trajectory(case: FloorplanCase, snapshots: list[Snapshot], path: Path, title: str) -> None:
    if not snapshots:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ids = [p.chiplet_id for p in snapshots[0].layout.placements]
    cmap = plt.get_cmap("tab20", max(len(ids), 1))
    for ci, chiplet_id in enumerate(ids):
        xs = [s.layout.by_id[chiplet_id].x + 0.5 * s.layout.by_id[chiplet_id].rotated_size(case.chiplet_by_id[chiplet_id])[0] for s in snapshots]
        ys = [s.layout.by_id[chiplet_id].y + 0.5 * s.layout.by_id[chiplet_id].rotated_size(case.chiplet_by_id[chiplet_id])[1] for s in snapshots]
        ax.plot(xs, ys, "-o", color=cmap(ci), markersize=3, linewidth=0.8, alpha=0.8, label=chiplet_id)
    draw_layout(ax, case, snapshots[-1].layout, f"{title}  MILP trajectory ({len(snapshots)} snapshots)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_snapshot_montage(case: FloorplanCase, snapshots: list[Snapshot], path: Path, title: str, cols: int = 10) -> None:
    """One thumbnail per MILP snapshot, in solve-time order (companion to trajectory.png)."""
    if not snapshots:
        return
    n = len(snapshots)
    cols = min(cols, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.3, rows * 1.3))
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax, snap in zip(axes_flat, snapshots):
        draw_layout_light(ax, case, snap.layout)
        ax.set_title(f"t={snap.t:.1f}s", fontsize=6)
    for ax in list(axes_flat)[n:]:
        ax.set_visible(False)
    fig.suptitle(f"{title}  MILP snapshots in solve order (n={n})", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_distribution(case: FloorplanCase, records: list[dict], path: Path, title: str) -> None:
    ids = [c.id for c in case.chiplets]
    cmap = plt.get_cmap("tab20", max(len(ids), 1))
    color_by_id = {cid: cmap(i) for i, cid in enumerate(ids)}

    # One scatter() call per chiplet (one per point would be O(n_records * n_chiplets)
    # individual matplotlib Artists and is extremely slow for large datasets).
    points = {cid: ([], []) for cid in ids}
    for record in records:
        for c in record["chiplets"]:
            xs, ys = points[c["name"]]
            xs.append(c["cx_mm"])
            ys.append(c["cy_mm"])

    fig, ax = plt.subplots(figsize=(7, 7))
    for cid in ids:
        xs, ys = points[cid]
        ax.scatter(xs, ys, s=6, color=color_by_id[cid], alpha=0.12, linewidths=0)
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=color_by_id[cid], label=cid) for cid in ids]
    ax.legend(handles=handles, fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0), ncol=1)
    ax.set_xlim(0, case.outline_width)
    ax.set_ylim(0, case.outline_height)
    ax.set_aspect("equal", adjustable="box")
    legal_n = sum(1 for r in records if r["legal"])
    ax.set_title(f"{title}  perturbation center distribution  (n={len(records)}, legal={legal_n})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_examples(case: FloorplanCase, records: list[dict], path: Path, title: str, n_examples: int = 9) -> None:
    if not records:
        return
    rng = np.random.default_rng(0)
    idx = rng.choice(len(records), size=min(n_examples, len(records)), replace=False)
    cols = 3
    rows = (len(idx) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax, i in zip(axes_flat, idx):
        record = records[int(i)]
        layout = _json_to_layout(record["chiplets"])
        legal_str = "LEGAL" if record["legal"] else "ILLEGAL"
        draw_layout(ax, case, layout, f"#{record['id']} {record['move_type']} {legal_str}")
    for ax in list(axes_flat)[len(idx):]:
        ax.set_visible(False)
    fig.suptitle(f"{title}  example perturbations", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_type_montage(
    case: FloorplanCase,
    records: list[dict],
    move_type: str,
    path: Path,
    title: str,
    max_n: int = 100,
    cols: int = 10,
) -> None:
    subset = [r for r in records if r["move_type"] == move_type]
    if not subset:
        return
    rng = np.random.default_rng(0)
    if len(subset) > max_n:
        idx = rng.choice(len(subset), size=max_n, replace=False)
        subset = [subset[i] for i in idx]
    n = len(subset)
    cols = min(cols, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.3, rows * 1.3))
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax, record in zip(axes_flat, subset):
        layout = _json_to_layout(record["chiplets"])
        draw_layout_light(ax, case, layout)
        color = "limegreen" if record["legal"] else "red"
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.3)
    for ax in list(axes_flat)[n:]:
        ax.set_visible(False)
    legal_n = sum(1 for r in subset if r["legal"])
    fig.suptitle(f"{title}  {move_type}  (n={n}, legal={legal_n}, green border=legal)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _json_to_layout(chiplets: list[dict]) -> Layout:
    return layout_from_chiplets_json(chiplets)


if __name__ == "__main__":
    main()
