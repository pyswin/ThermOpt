from __future__ import annotations

import json
from pathlib import Path

from thermopt.data.thermal_dataset import ThermalDatasetGenerator


def write_atplace_case(root: Path, name: str = "Case1") -> Path:
    case_dir = root / name
    case_dir.mkdir(parents=True)
    (case_dir / f"{name}.blocks").write_text(
        "\n".join(
            [
                "NumSoftRectangularBlocks : 0",
                "NumHardRectilinearBlocks : 2",
                "NumTerminals : 0",
                "",
                "CPU hardrectilinear 4 (0, 0) (0, 1000) (2000, 1000) (2000, 0)",
                "GPU hardrectilinear 4 (0, 0) (0, 3000) (3000, 3000) (3000, 0)",
            ]
        ),
        encoding="utf-8",
    )
    (case_dir / f"{name}.nets").write_text(
        "\n".join(
            [
                "NumNets : 1",
                "NumPins : 2",
                "",
                "NetDegree : 2",
                "CPU B : %0 %0",
                "GPU B : %0 %0",
            ]
        ),
        encoding="utf-8",
    )
    (case_dir / f"{name}.power").write_text("CPU 10\nGPU 30\n", encoding="utf-8")
    (case_dir / f"{name}.pl").write_text("CPU 0 0\nGPU 0 0\n", encoding="utf-8")
    return case_dir


def test_generate_thermal_dataset_writes_pointwise_and_json(tmp_path: Path) -> None:
    case_dir = write_atplace_case(tmp_path)
    output_dir = tmp_path / "dataset"

    generator = ThermalDatasetGenerator(
        case_dir,
        {
            "backend": "heuristic",
            "grid_size": [8, 8],
            "ambient": 25.0,
            "scale": 0.1,
            "sigma_factor": 1.0,
        },
        use_case_config=False,
        randomize_position=True,
        randomize_power=True,
        randomize_rotation=True,
        seed=7,
    )

    generator.generate_dataset(num_samples=1, output_dir=output_dir, variation_type="random", save_formats=["pointwise", "json"])

    pointwise = output_dir / "pointwise" / "sample_000000.csv"
    sample_json = output_dir / "json" / "sample_000000.json"
    summary = output_dir / "dataset_summary.json"

    assert pointwise.exists()
    assert sample_json.exists()
    assert summary.exists()

    payload = json.loads(sample_json.read_text(encoding="utf-8"))
    assert payload["temperature_map"]
    assert set(config["rotation"] for config in payload["chiplet_configs"]) <= {0, 90}

    summary_payload = json.loads(summary.read_text(encoding="utf-8"))
    assert summary_payload["successful_samples"] == 1
    assert summary_payload["thermal_backend"]["runtime_mode"] == "heuristic"


def test_pointwise_generation_marks_background_cells(tmp_path: Path) -> None:
    case_dir = write_atplace_case(tmp_path)
    output_dir = tmp_path / "dataset"

    generator = ThermalDatasetGenerator(
        case_dir,
        {
            "backend": "heuristic",
            "grid_size": [4, 4],
            "ambient": 25.0,
            "scale": 0.1,
            "sigma_factor": 1.0,
        },
        use_case_config=False,
        randomize_power=False,
        randomize_rotation=False,
        seed=7,
    )

    generator.generate_dataset(num_samples=1, output_dir=output_dir, variation_type="random", save_formats=["pointwise"])

    pointwise = output_dir / "pointwise" / "sample_000000.csv"
    lines = pointwise.read_text(encoding="utf-8").splitlines()
    assert any(line.split(",")[2] == "background" and line.split(",")[3] == "0.000000" for line in lines[1:])
