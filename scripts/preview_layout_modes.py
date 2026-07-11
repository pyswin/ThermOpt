#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from thermopt.data.thermal_dataset import ThermalDatasetGenerator  # noqa: E402
from thermopt.layout.visualization import draw_layout  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview layout distributions without thermal simulation.")
    parser.add_argument(
        "--case_dir",
        type=Path,
        default=Path("external/ATPlace_pub/cases/Case5"),
        help="ATPlace case directory, relative to repo root or absolute.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("outputs/case5_layout_preview_20"),
        help="Directory to write montage figures.",
    )
    parser.add_argument("--num_layouts", type=int, default=20, help="Number of layouts to generate per mode.")
    parser.add_argument(
        "--layout_modes",
        type=str,
        default="random,cluster,boundary,dispersed",
        help="Comma-separated layout modes to preview.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Base random seed.")
    parser.add_argument("--min_gap", type=float, default=0.05, help="Minimum gap between chiplets in mm.")
    parser.add_argument("--randomize_position", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--randomize_power", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--randomize_rotation", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _make_generator(
    case_dir: Path,
    mode: str,
    seed: int,
    min_gap: float,
    *,
    randomize_position: bool,
    randomize_power: bool,
    randomize_rotation: bool,
) -> ThermalDatasetGenerator:
    return ThermalDatasetGenerator(
        case_dir,
        {"backend": "heuristic"},
        layout_mode=mode,
        min_gap=min_gap,
        randomize_position=randomize_position,
        randomize_power=randomize_power,
        randomize_rotation=randomize_rotation,
        work_dir=Path("/tmp/thermopt_preview"),
        seed=seed,
    )


def _generate_layouts(generator: ThermalDatasetGenerator, mode: str, count: int):
    layouts = []
    attempts = 0
    max_attempts = max(count * 50, 200)
    while len(layouts) < count and attempts < max_attempts:
        attempts += 1
        try:
            # Reuse the exact layout-generation path used by generate_dataset.
            configs = generator._generate_layout_variation(mode)
        except Exception:
            continue
        layouts.append(generator._build_layout(configs))
    if len(layouts) < count:
        raise RuntimeError(
            f"failed to generate enough valid layouts for mode={mode}: "
            f"{len(layouts)}/{count} after {attempts} attempts"
        )
    return layouts


def _save_mode_montage(case, layouts, out_path: Path, mode: str) -> None:
    count = len(layouts)
    cols = 5
    rows = int(math.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows), constrained_layout=False)
    axes = np.asarray(axes).reshape(-1)

    for index, (ax, layout) in enumerate(zip(axes, layouts), start=1):
        draw_layout(ax, case, layout, f"{mode} #{index:02d}", center_based=True)
        ax.tick_params(labelsize=8)

    for ax in axes[len(layouts) :]:
        ax.axis("off")

    fig.suptitle(f"Case preview: {mode} ({count} layouts)", fontsize=14, y=0.995)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    case_dir = args.case_dir if args.case_dir.is_absolute() else (Path(__file__).resolve().parent.parent / args.case_dir)
    out_dir = args.out_dir if args.out_dir.is_absolute() else (Path(__file__).resolve().parent.parent / args.out_dir)
    modes = [item.strip() for item in args.layout_modes.split(",") if item.strip()]
    if not modes:
        raise ValueError("no layout modes provided")

    out_dir.mkdir(parents=True, exist_ok=True)

    for offset, mode in enumerate(modes):
        generator = _make_generator(
            case_dir,
            mode,
            args.seed + offset * 1000,
            args.min_gap,
            randomize_position=args.randomize_position,
            randomize_power=args.randomize_power,
            randomize_rotation=args.randomize_rotation,
        )
        layouts = _generate_layouts(generator, mode, args.num_layouts)
        _save_mode_montage(generator.case, layouts, out_dir / f"{mode}.png", mode)

    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
