#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from thermopt.data.pointwise_features import augment_pointwise_sample


def _discover_case_dirs(dataset_dir: Path) -> list[Path]:
    if (dataset_dir / "pointwise").is_dir() and (dataset_dir / "json").is_dir():
        return [dataset_dir]

    case_dirs = [
        child
        for child in sorted(dataset_dir.iterdir())
        if child.is_dir() and (child / "pointwise").is_dir() and (child / "json").is_dir()
    ]
    if not case_dirs:
        raise ValueError(
            f"No dataset directories found under {dataset_dir}. Expected either a case dir with pointwise/json "
            "subdirectories or a parent directory containing case subdirectories."
        )
    return case_dirs


def augment_dataset(dataset_dir: Path, output_dir: Path | None = None, edge_band_mm: float | None = None) -> dict:
    dataset_dir = dataset_dir.resolve()
    case_dirs = _discover_case_dirs(dataset_dir)
    results: dict[str, dict] = {}

    for case_dir in case_dirs:
        pointwise_dir = case_dir / "pointwise"
        json_dir = case_dir / "json"
        if output_dir is None:
            target_root = case_dir / "pointwise_augmented"
        else:
            target_root = output_dir / case_dir.name / "pointwise_augmented" if case_dir != dataset_dir else output_dir
        target_root.mkdir(parents=True, exist_ok=True)

        sample_results: list[dict] = []
        for pointwise_csv in sorted(pointwise_dir.glob("sample_*.csv")):
            sample_json = json_dir / f"{pointwise_csv.stem}.json"
            if not sample_json.is_file():
                raise FileNotFoundError(f"Missing JSON companion for {pointwise_csv}: {sample_json}")
            output_csv = target_root / pointwise_csv.name
            sample_results.append(
                augment_pointwise_sample(
                    pointwise_csv,
                    sample_json,
                    output_csv,
                    edge_band_mm=edge_band_mm,
                )
            )

        summary = {
            "source_case_dir": str(case_dir),
            "output_dir": str(target_root),
            "num_samples": len(sample_results),
            "edge_band_mm": edge_band_mm,
            "feature_columns": sample_results[0]["feature_columns"] if sample_results else [],
        }
        (target_root / "feature_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        results[case_dir.name] = summary

    return {
        "dataset_dir": str(dataset_dir),
        "edge_band_mm": edge_band_mm,
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment pointwise thermal datasets with geometry channels.")
    parser.add_argument("--dataset_dir", type=Path, required=True, help="Dataset root or a single case directory")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Optional output root. If omitted, augmented CSVs are written next to the source pointwise folder.",
    )
    parser.add_argument(
        "--edge_band_mm",
        type=float,
        default=None,
        help="Optional edge-band thickness in mm. Default uses half the grid spacing.",
    )
    args = parser.parse_args()

    report = augment_dataset(args.dataset_dir, args.output_dir, args.edge_band_mm)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
