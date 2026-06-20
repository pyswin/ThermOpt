#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from thermopt.data.thermal_dataset import ThermalDatasetGenerator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate thermal datasets from ATPlace-style cases.")
    parser.add_argument("--case_dir", type=Path, required=True, help="Case directory containing .blocks/.nets/.power/.pl")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--variation_type", type=str, default="random", choices=["fixed", "random", "grid"])
    parser.add_argument("--save_formats", type=str, default="pointwise,json", help="Comma-separated list: pointwise, gridwise, json")

    parser.add_argument("--config_name", type=str, default="reproduce.json")
    parser.add_argument("--config_mode", type=str, default="thermal", choices=["wl", "thermal"])
    parser.add_argument("--use_case_config", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--unit_scale", type=float, default=0.001, help="Scale case units into mm")
    parser.add_argument("--initial_layout", type=str, default="pl", choices=["pl", "random"])
    parser.add_argument("--min_gap", type=float, default=0.05, help="Minimum gap between chiplets in mm")

    parser.add_argument("--randomize_position", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--randomize_power", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--randomize_rotation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--power_additive_fraction", type=float, default=0.20)
    parser.add_argument("--power_dropout_prob", type=float, default=0.05)
    parser.add_argument("--power_sleep_ratio", type=float, default=0.05)
    parser.add_argument("--power_shutdown_prob", type=float, default=0.02)
    parser.add_argument("--min_power_density", type=float, default=0.0)
    parser.add_argument("--max_power_density", type=float, default=None)
    parser.add_argument("--tdp_limit", type=float, default=None)
    parser.add_argument("--tdp_limit_ratio", type=float, default=1.25)

    parser.add_argument("--backend", type=str, default="hotspot", choices=["hotspot", "heuristic"])
    parser.add_argument("--hotspot_binary", type=str, default="external/ATPlace_pub/thermal/hotspot")
    parser.add_argument("--hotspot_required", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hotspot_allow_fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grid_size", type=int, nargs=2, default=[64, 64], metavar=("NX", "NY"))
    parser.add_argument("--ambient", type=float, default=25.0)
    parser.add_argument("--scale", type=float, default=0.05)
    parser.add_argument("--sigma_factor", type=float, default=1.0)
    parser.add_argument("--thermal_threshold", type=float, default=None)
    parser.add_argument("--work_dir", type=Path, default=None, help="Optional workspace for HotSpot temp files")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    thermal_config = {
        "backend": args.backend,
        "hotspot_binary": args.hotspot_binary,
        "hotspot_required": args.hotspot_required,
        "hotspot_allow_fallback": args.hotspot_allow_fallback,
        "grid_size": [int(args.grid_size[0]), int(args.grid_size[1])],
        "ambient": float(args.ambient),
        "scale": float(args.scale),
        "sigma_factor": float(args.sigma_factor),
    }
    if args.thermal_threshold is not None:
        thermal_config["thermal_threshold"] = float(args.thermal_threshold)

    generator = ThermalDatasetGenerator(
        args.case_dir,
        thermal_config,
        config_name=args.config_name,
        config_mode=args.config_mode,
        use_case_config=args.use_case_config,
        unit_scale=args.unit_scale,
        initial_layout=args.initial_layout,
        min_gap=args.min_gap,
        randomize_position=args.randomize_position,
        randomize_power=args.randomize_power,
        randomize_rotation=args.randomize_rotation,
        power_additive_fraction=args.power_additive_fraction,
        power_dropout_prob=args.power_dropout_prob,
        power_sleep_ratio=args.power_sleep_ratio,
        power_shutdown_prob=args.power_shutdown_prob,
        min_power_density=args.min_power_density,
        max_power_density=args.max_power_density,
        tdp_limit=args.tdp_limit,
        tdp_limit_ratio=args.tdp_limit_ratio,
        work_dir=args.work_dir or (args.output_dir / "_thermal"),
        seed=args.seed,
    )

    save_formats = [item.strip() for item in args.save_formats.split(",") if item.strip()]
    output_dir = generator.generate_dataset(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        variation_type=args.variation_type,
        save_formats=save_formats,
    )
    print(f"Saved dataset to {output_dir}")


if __name__ == "__main__":
    main()
