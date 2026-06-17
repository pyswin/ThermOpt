from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from thermopt.data.atplace import load_atplace_cases
from thermopt.data.inputs import CaseInput
from thermopt.data.pointwise import load_pointwise_cases


def summarize_case(case_input: CaseInput) -> dict:
    case = case_input.case
    chiplet_areas = np.array([chiplet.width * chiplet.height for chiplet in case.chiplets], dtype=float)
    powers = np.array([chiplet.power for chiplet in case.chiplets], dtype=float)
    net_degrees = np.array([len(net.chiplets) for net in case.nets], dtype=float) if case.nets else np.array([])

    return {
        "name": case_input.name,
        "source_path": str(case_input.source_path),
        "num_chiplets": len(case.chiplets),
        "num_nets": len(case.nets),
        "outline_width": case.outline_width,
        "outline_height": case.outline_height,
        "total_chiplet_area": case.total_chiplet_area,
        "area_utilization": case.total_chiplet_area / max(case.outline_width * case.outline_height, 1e-9),
        "power_min": float(powers.min()) if powers.size else 0.0,
        "power_max": float(powers.max()) if powers.size else 0.0,
        "power_mean": float(powers.mean()) if powers.size else 0.0,
        "chiplet_area_min": float(chiplet_areas.min()) if chiplet_areas.size else 0.0,
        "chiplet_area_max": float(chiplet_areas.max()) if chiplet_areas.size else 0.0,
        "net_degree_mean": float(net_degrees.mean()) if net_degrees.size else 0.0,
        "has_netlist": bool(case.nets),
    }


def summarize_dataset(cases: list[CaseInput]) -> dict:
    rows = [summarize_case(case) for case in cases]
    return {
        "num_cases": len(rows),
        "cases": rows,
        "ranges": {
            key: [min(row[key] for row in rows), max(row[key] for row in rows)]
            for key in ("num_chiplets", "num_nets", "area_utilization", "power_max")
        }
        if rows
        else {},
    }


def compare(atplace_dir: Path, pointwise_dir: Path, output_path: Path | None) -> dict:
    report = {
        "atplace": summarize_dataset(
            load_atplace_cases(
                {
                    "data_dir": str(atplace_dir),
                    "cases": ["Case1", "Case2", "Case3"],
                    "unit_scale": 0.001,
                    "initial_layout": "random",
                },
                seed=7,
            )
        ),
        "pointwise": summarize_dataset(
            load_pointwise_cases(
                {
                    "data_dir": str(pointwise_dir),
                    "files": ["sample_000000.csv", "sample_000001.csv", "sample_000002.csv"],
                    "outline_width": 100.0,
                    "outline_height": 100.0,
                    "nearest_net_degree": 2,
                }
            )
        ),
        "compatibility_notes": [
            "ATPlace cases provide explicit blocks, power, and high-cardinality Bookshelf netlists; pointwise CSVs provide grid labels and temperature samples but no original netlist.",
            "Both can be converted to FloorplanCase/Layout, so a thermal surrogate that consumes rasterized power maps can reuse ATPlace cases directly after rasterization.",
            "A surrogate that depends on HotSpot labels still needs ATPlace-derived temperature labels or a domain-adaptation check, because pointwise labels are training samples rather than the main placement benchmark.",
        ],
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ATPlace benchmark cases with pointwise thermal-training samples.")
    parser.add_argument("--atplace-dir", type=Path, default=Path("external/ATPlace_pub/cases"))
    parser.add_argument("--pointwise-dir", type=Path, default=Path("pointwise"))
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset_comparison.json"))
    args = parser.parse_args()

    report = compare(args.atplace_dir, args.pointwise_dir, args.output)
    print(json.dumps(report, indent=2))
    print(f"Saved dataset comparison to {args.output}")


if __name__ == "__main__":
    main()
