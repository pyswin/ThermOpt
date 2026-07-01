from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from thermopt.data.atplace import load_atplace_case
from thermopt.layout.geometry import bounds, center
from thermopt.layout.objects import Layout, Placement
from thermopt.thermal.hotspot import HotSpotBackend

SCALE = 0.001
CASES_DIR = ROOT / "external/ATPlace_pub/cases"
MILP_DIR = ROOT / "atplace/20260627_163106_milp150s"
HOTSPOT_BIN = ROOT / "external/ATPlace_pub/thermal/hotspot"


@dataclass
class SampleMetrics:
    sample_file: str
    sample_id: int
    layout_center_shift_mean_mm: float
    layout_center_shift_max_mm: float
    layout_hp_shift_mean_mm: float
    layout_hp_shift_max_mm: float
    layout_rotation_changes: int
    layout_hp_pairwise_min_mm: float
    layout_hp_pairwise_mean_mm: float
    layout_hp_boundary_min_mm: float
    temp_tmax_c: float
    temp_t50_c: float
    temp_tmean_c: float
    temp_dtmax_c: float
    temp_dt50_c: float
    temp_dtmean_c: float


def load_milp_layout(case_name: str) -> Layout:
    data = json.load(open(MILP_DIR / case_name / "layout.json"))
    return Layout(
        placements=tuple(
            Placement(
                chiplet_id=item["name"],
                x=float(item.get("cx_mm", item["x"] * SCALE)),
                y=float(item.get("cy_mm", item["y"] * SCALE)),
                rotation=int(round(math.degrees(float(item["angle_rad"])))) % 360,
            )
            for item in data["chiplets"]
        )
    )


def load_dataset_sample(path: Path) -> tuple[int, Layout, np.ndarray]:
    data = json.load(open(path))
    sample_id = int(data["sample_id"])
    layout = Layout(
        placements=tuple(
            Placement(
                chiplet_id=item["name"],
                x=float(item.get("cx", float(item["x"]) + float(item["width"]) * 0.5)),
                y=float(item.get("cy", float(item["y"]) + float(item["height"]) * 0.5)),
                rotation=int(item.get("rotation", 0)) % 360,
            )
            for item in data["chiplet_configs"]
        )
    )
    temp = np.asarray(data["temperature_map"], dtype=np.float32)
    return sample_id, layout, temp


def temp_stats(grid: np.ndarray) -> dict[str, float]:
    flat = np.sort(np.asarray(grid, dtype=np.float32).ravel())[::-1]
    top50 = flat[:50] if flat.size >= 50 else flat
    return {
        "tmax": float(flat[0]),
        "t50": float(top50.mean()),
        "tmean": float(np.mean(grid)),
        "tmin": float(flat[-1]),
    }


def _center_map(case, layout: Layout) -> dict[str, tuple[float, float]]:
    return {placement.chiplet_id: center(case, placement) for placement in layout.placements}


def _rotation_map(layout: Layout) -> dict[str, int]:
    return {placement.chiplet_id: int(placement.rotation) % 360 for placement in layout.placements}


def _high_power_ids(case, threshold: float = 100.0) -> list[str]:
    return [chiplet.id for chiplet in case.chiplets if chiplet.power >= threshold]


def _pairwise_center_stats(case, layout: Layout, chiplet_ids: list[str]) -> tuple[float, float]:
    if len(chiplet_ids) < 2:
        return 0.0, 0.0
    centers = _center_map(case, layout)
    dists = []
    for i, left_id in enumerate(chiplet_ids):
        for right_id in chiplet_ids[i + 1 :]:
            lx, ly = centers[left_id]
            rx, ry = centers[right_id]
            dists.append(math.hypot(lx - rx, ly - ry))
    return float(min(dists)), float(sum(dists) / len(dists))


def _boundary_clearance(case, layout: Layout, chiplet_ids: list[str]) -> float:
    if not chiplet_ids:
        return 0.0
    centers = _center_map(case, layout)
    values = []
    for chiplet_id in chiplet_ids:
        cx, cy = centers[chiplet_id]
        values.append(min(cx, case.outline_width - cx, cy, case.outline_height - cy))
    return float(min(values))


def compare_layout(case, ref_layout: Layout, layout: Layout) -> dict[str, float]:
    ref_centers = _center_map(case, ref_layout)
    cur_centers = _center_map(case, layout)
    shifts = []
    hp_shifts = []
    rotation_changes = 0
    hp_ids = _high_power_ids(case)
    ref_rot = _rotation_map(ref_layout)
    cur_rot = _rotation_map(layout)
    for chiplet_id in case.chiplet_ids:
        rx, ry = ref_centers[chiplet_id]
        cx, cy = cur_centers[chiplet_id]
        shift = math.hypot(cx - rx, cy - ry)
        shifts.append(shift)
        if ref_rot[chiplet_id] != cur_rot[chiplet_id]:
            rotation_changes += 1
        if chiplet_id in hp_ids:
            hp_shifts.append(shift)

    hp_pairwise_min, hp_pairwise_mean = _pairwise_center_stats(case, layout, hp_ids)
    hp_boundary_min = _boundary_clearance(case, layout, hp_ids)

    return {
        "layout_center_shift_mean_mm": float(sum(shifts) / len(shifts)),
        "layout_center_shift_max_mm": float(max(shifts)),
        "layout_hp_shift_mean_mm": float(sum(hp_shifts) / len(hp_shifts)) if hp_shifts else 0.0,
        "layout_hp_shift_max_mm": float(max(hp_shifts)) if hp_shifts else 0.0,
        "layout_rotation_changes": int(rotation_changes),
        "layout_hp_pairwise_min_mm": hp_pairwise_min,
        "layout_hp_pairwise_mean_mm": hp_pairwise_mean,
        "layout_hp_boundary_min_mm": hp_boundary_min,
    }


def draw_layout_panel(ax: plt.Axes, case, layout: Layout, title: str) -> None:
    chiplets = case.chiplet_by_id
    powers = [chiplets[p.chiplet_id].power for p in layout.placements]
    pmin, pmax = min(powers), max(powers)
    cmap = plt.get_cmap("inferno")
    for placement in layout.placements:
        chiplet = chiplets[placement.chiplet_id]
        x0, y0, x1, y1 = bounds(case, placement)
        frac = (chiplet.power - pmin) / max(pmax - pmin, 1e-9)
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=cmap(frac), edgecolor="white", lw=0.8)
        ax.add_patch(rect)
        if chiplet.power >= 100.0:
            ax.text((x0 + x1) * 0.5, (y0 + y1) * 0.5, chiplet.id, color="white",
                    ha="center", va="center", fontsize=7)
    ax.set_xlim(0, case.outline_width)
    ax.set_ylim(0, case.outline_height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")


def draw_temp_panel(ax: plt.Axes, grid: np.ndarray, title: str, vmin: float, vmax: float, cmap: str = "hot") -> None:
    image = ax.imshow(grid, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    return image


def save_layout_figure(case, ref_layout: Layout, samples: list[tuple[str, Layout, dict[str, float]]], out_path: Path) -> None:
    count = 1 + len(samples)
    cols = 4
    rows = int(math.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.1 * rows), constrained_layout=False)
    axes = np.asarray(axes).reshape(-1)

    draw_layout_panel(axes[0], case, ref_layout, "MILP layout")
    for i, (label, layout, metrics) in enumerate(samples, start=1):
        title = (
            f"{label}\n"
            f"dC={metrics['layout_center_shift_mean_mm']:.2f}mm  "
            f"dTmax={metrics['temp_dtmax_c']:+.2f}C"
        )
        draw_layout_panel(axes[i], case, layout, title)
    for j in range(count, len(axes)):
        axes[j].axis("off")

    fig.suptitle("MILP vs dataset layouts", fontsize=13, y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_temperature_figure(
    ref_grid: np.ndarray,
    samples: list[tuple[str, np.ndarray, dict[str, float]]],
    out_path: Path,
) -> None:
    count = 1 + len(samples)
    cols = 4
    rows = int(math.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.1 * rows), constrained_layout=False)
    axes = np.asarray(axes).reshape(-1)

    grids = [ref_grid] + [grid for _, grid, _ in samples]
    vmin = min(float(np.min(grid)) for grid in grids)
    vmax = max(float(np.max(grid)) for grid in grids)

    im = draw_temp_panel(axes[0], ref_grid, f"MILP\nTmax={temp_stats(ref_grid)['tmax']:.2f}C", vmin, vmax)
    for i, (label, grid, metrics) in enumerate(samples, start=1):
        draw_temp_panel(
            axes[i],
            grid,
            f"{label}\nTmax={metrics['temp_tmax_c']:.2f}C  dTmax={metrics['temp_dtmax_c']:+.2f}C",
            vmin,
            vmax,
        )
    for j in range(count, len(axes)):
        axes[j].axis("off")

    fig.colorbar(im, ax=axes[:count].tolist(), fraction=0.02, pad=0.01, label="temperature (C)")
    fig.suptitle("MILP vs dataset temperature maps", fontsize=13, y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_delta_figure(
    ref_grid: np.ndarray,
    samples: list[tuple[str, np.ndarray, dict[str, float]]],
    out_path: Path,
) -> None:
    count = len(samples)
    cols = 4
    rows = int(math.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.1 * rows), constrained_layout=False)
    axes = np.asarray(axes).reshape(-1)

    delta_max = 0.0
    for _, grid, _ in samples:
        delta = grid - ref_grid
        delta_max = max(delta_max, float(np.max(np.abs(delta))))
    delta_max = max(delta_max, 1e-6)

    im = None
    for i, (label, grid, metrics) in enumerate(samples):
        delta = grid - ref_grid
        im = draw_temp_panel(
            axes[i],
            delta,
            f"{label}\nΔTmax={metrics['temp_dtmax_c']:+.2f}C",
            -delta_max,
            delta_max,
            cmap="RdBu_r",
        )
    for j in range(count, len(axes)):
        axes[j].axis("off")

    if im is not None:
        fig.colorbar(im, ax=axes[:count].tolist(), fraction=0.02, pad=0.01, label="delta temperature (C)")
    fig.suptitle("Dataset temperature minus MILP temperature", fontsize=13, y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_summary_csv(rows: list[SampleMetrics], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        if rows:
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare one MILP layout with a small set of generated thermal samples.")
    parser.add_argument("--case", type=str, default="Case5")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "outputs/thermopt_dataset/case5/json")
    parser.add_argument("--milp-layout", type=Path, default=MILP_DIR / "Case5" / "layout.json")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/case5_milp_vs_dataset")
    parser.add_argument("--hotspot-grid-size", type=int, nargs=2, default=[64, 64])
    args = parser.parse_args()

    case_dir = CASES_DIR / args.case
    case = load_atplace_case(case_dir, {"unit_scale": 0.001, "initial_layout": "pl"}, 42).case
    ref_layout = load_milp_layout(args.case)

    sample_files = sorted(args.dataset_dir.glob("*.json"))
    if not sample_files:
        raise FileNotFoundError(f"no dataset samples found in {args.dataset_dir}")

    sample_count = min(args.samples, len(sample_files))
    if sample_count == len(sample_files):
        selected = sample_files
    else:
        indexes = np.linspace(0, len(sample_files) - 1, sample_count, dtype=int)
        selected = [sample_files[idx] for idx in indexes]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = HotSpotBackend(
        case=case,
        config={
            "hotspot_binary": str(HOTSPOT_BIN),
            "grid_size": list(args.hotspot_grid_size),
        },
    )
    ref_temp = backend.simulate(case, ref_layout)
    ref_metrics = temp_stats(ref_temp)

    sample_layout_rows: list[tuple[str, Layout, dict[str, float]]] = []
    sample_temp_rows: list[tuple[str, np.ndarray, dict[str, float]]] = []
    summary_rows: list[SampleMetrics] = []
    raw_summary: dict[str, object] = {
        "case": args.case,
        "milp_layout": str(args.milp_layout),
        "dataset_dir": str(args.dataset_dir),
        "selected_samples": [str(path.name) for path in selected],
        "milp_temperature_stats": ref_metrics,
        "samples": [],
    }

    for path in selected:
        sample_id, layout, temp = load_dataset_sample(path)
        layout_metrics = compare_layout(case, ref_layout, layout)
        temp_metrics = temp_stats(temp)
        metrics = {
            **layout_metrics,
            "temp_tmax_c": temp_metrics["tmax"],
            "temp_t50_c": temp_metrics["t50"],
            "temp_tmean_c": temp_metrics["tmean"],
            "temp_dtmax_c": temp_metrics["tmax"] - ref_metrics["tmax"],
            "temp_dt50_c": temp_metrics["t50"] - ref_metrics["t50"],
            "temp_dtmean_c": temp_metrics["tmean"] - ref_metrics["tmean"],
        }
        label = f"sample_{sample_id:06d}"
        sample_layout_rows.append((label, layout, metrics))
        sample_temp_rows.append((label, temp, metrics))
        summary_rows.append(
            SampleMetrics(
                sample_file=path.name,
                sample_id=sample_id,
                layout_center_shift_mean_mm=metrics["layout_center_shift_mean_mm"],
                layout_center_shift_max_mm=metrics["layout_center_shift_max_mm"],
                layout_hp_shift_mean_mm=metrics["layout_hp_shift_mean_mm"],
                layout_hp_shift_max_mm=metrics["layout_hp_shift_max_mm"],
                layout_rotation_changes=metrics["layout_rotation_changes"],
                layout_hp_pairwise_min_mm=metrics["layout_hp_pairwise_min_mm"],
                layout_hp_pairwise_mean_mm=metrics["layout_hp_pairwise_mean_mm"],
                layout_hp_boundary_min_mm=metrics["layout_hp_boundary_min_mm"],
                temp_tmax_c=metrics["temp_tmax_c"],
                temp_t50_c=metrics["temp_t50_c"],
                temp_tmean_c=metrics["temp_tmean_c"],
                temp_dtmax_c=metrics["temp_dtmax_c"],
                temp_dt50_c=metrics["temp_dt50_c"],
                temp_dtmean_c=metrics["temp_dtmean_c"],
            )
        )
        raw_summary["samples"].append({"file": path.name, **metrics})

    layout_fig = output_dir / "layout_compare.png"
    temp_fig = output_dir / "temperature_compare.png"
    delta_fig = output_dir / "temperature_delta.png"
    csv_path = output_dir / "comparison_summary.csv"
    json_path = output_dir / "comparison_summary.json"

    save_layout_figure(case, ref_layout, sample_layout_rows, layout_fig)
    save_temperature_figure(ref_temp, sample_temp_rows, temp_fig)
    save_delta_figure(ref_temp, sample_temp_rows, delta_fig)
    write_summary_csv(summary_rows, csv_path)
    json_path.write_text(json.dumps(raw_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    dtemp = [row.temp_dtmax_c for row in summary_rows]
    dshift = [row.layout_center_shift_mean_mm for row in summary_rows]
    print(f"Case: {args.case}")
    print(f"MILP Tmax: {ref_metrics['tmax']:.2f} C")
    print(f"Selected samples: {len(summary_rows)}")
    print(f"Avg layout center shift: {sum(dshift) / len(dshift):.2f} mm")
    print(f"Avg delta Tmax: {sum(dtemp) / len(dtemp):+.2f} C")
    print(f"Min/Max delta Tmax: {min(dtemp):+.2f} C / {max(dtemp):+.2f} C")
    print(f"Saved figures to: {output_dir}")


if __name__ == "__main__":
    main()
