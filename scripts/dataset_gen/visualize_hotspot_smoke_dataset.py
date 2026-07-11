#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from thermopt.data.atplace import load_atplace_case
from thermopt.layout.objects import Layout, Placement
from thermopt.layout.visualization import draw_layout


SMOKE_ROOT = ROOT / "outputs/thermopt_hotspot_smoke_10"
CASES_ROOT = ROOT / "external/ATPlace_pub/cases"


@dataclass
class SampleSummary:
    sample_file: str
    sample_id: int
    layout_mode: str
    tmin_c: float
    tmean_c: float
    tmax_c: float


def _case_name_from_dir(case_dir: Path) -> str:
    suffix = case_dir.name.removeprefix("case")
    if not suffix or not suffix.isdigit():
        raise ValueError(f"unexpected case directory name: {case_dir.name}")
    return f"Case{suffix}"


def _load_sample(path: Path) -> tuple[int, Layout, np.ndarray, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    placements = []
    for item in data["chiplet_configs"]:
        x = float(item.get("cx", float(item["x"]) + float(item["width"]) * 0.5))
        y = float(item.get("cy", float(item["y"]) + float(item["height"]) * 0.5))
        rotation = int(item.get("rotation", 0)) % 360
        placements.append(
            Placement(
                chiplet_id=item["name"],
                x=x,
                y=y,
                rotation=rotation,
            )
        )
    layout = Layout(tuple(placements))
    temperature = np.asarray(data["temperature_map"], dtype=np.float32)
    return int(data["sample_id"]), layout, temperature, data


def _collect_case_samples(case_json_dir: Path, limit: int | None) -> list[Path]:
    sample_files = sorted(case_json_dir.glob("sample_*.json"))
    if limit is not None:
        sample_files = sample_files[: max(0, int(limit))]
    return sample_files


def _draw_case_overview(case_name: str, case, samples: list[dict], out_path: Path) -> None:
    rows = len(samples)
    fig, axes = plt.subplots(rows, 2, figsize=(12.0, max(3.0 * rows, 4.0)), constrained_layout=False)
    axes = np.atleast_2d(axes)

    vmin = min(float(np.min(item["temperature"])) for item in samples)
    vmax = max(float(np.max(item["temperature"])) for item in samples)

    temp_im = None
    for row, item in enumerate(samples):
        layout_ax = axes[row, 0]
        temp_ax = axes[row, 1]

        sample_label = f"sample_{item['sample_id']:06d}"
        draw_layout(
            layout_ax,
            case,
            item["layout"],
            f"{sample_label} layout",
            center_based=True,
        )
        layout_ax.tick_params(labelsize=7)

        temp_im = temp_ax.imshow(
            item["temperature"],
            origin="lower",
            cmap="hot",
            vmin=vmin,
            vmax=vmax,
            extent=[0.0, case.outline_width, 0.0, case.outline_height],
            aspect="equal",
            interpolation="nearest",
        )
        temp_ax.set_title(
            f"{sample_label} temperature | Tmax={item['temperature'].max():.2f} °C",
            fontsize=9,
        )
        temp_ax.set_xlabel("x (mm)")
        temp_ax.set_ylabel("y (mm)")
        temp_ax.tick_params(labelsize=7)

    if temp_im is not None:
        fig.colorbar(temp_im, ax=axes[:, 1].tolist(), fraction=0.02, pad=0.02, label="temperature (°C)")

    fig.suptitle(f"{case_name} smoke dataset: layout vs HotSpot temperature", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 0.97, 0.985])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize smoke dataset layouts and HotSpot temperature maps.")
    parser.add_argument(
        "--smoke_root",
        type=Path,
        default=SMOKE_ROOT,
        help="Root directory of thermopt_hotspot_smoke_10.",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Case directories to process, for example: case1 case5 case10. Default: all cases under smoke_root.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional limit of samples per case.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=ROOT / "outputs/thermopt_hotspot_smoke_10_viz",
        help="Directory for generated figures and summaries.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    smoke_root = args.smoke_root if args.smoke_root.is_absolute() else (ROOT / args.smoke_root)
    out_dir = args.out_dir if args.out_dir.is_absolute() else (ROOT / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = []
    if args.cases:
        for case_name in args.cases:
            case_dir = smoke_root / case_name.lower()
            if not case_dir.is_dir():
                raise FileNotFoundError(f"missing case directory: {case_dir}")
            case_dirs.append(case_dir)
    else:
        case_dirs = sorted(
            [path for path in smoke_root.iterdir() if path.is_dir() and re.fullmatch(r"case\d+", path.name)],
            key=lambda path: int(path.name.removeprefix("case")),
        )

    if not case_dirs:
        raise FileNotFoundError(f"no case directories found under {smoke_root}")

    summary: dict[str, object] = {
        "smoke_root": str(smoke_root),
        "out_dir": str(out_dir),
        "cases": {},
    }

    for case_dir in case_dirs:
        case_name = _case_name_from_dir(case_dir)
        case = load_atplace_case(CASES_ROOT / case_name, {"unit_scale": 0.001, "initial_layout": "pl"}, 42).case
        sample_files = _collect_case_samples(case_dir / "json", args.max_samples)
        if not sample_files:
            raise FileNotFoundError(f"no sample json files found in {case_dir / 'json'}")

        sample_rows: list[dict] = []
        case_summaries: list[SampleSummary] = []
        for sample_path in sample_files:
            sample_id, layout, temperature, data = _load_sample(sample_path)
            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "layout": layout,
                    "temperature": temperature,
                    "layout_mode": data.get("layout_mode", "unknown"),
                }
            )
            case_summaries.append(
                SampleSummary(
                    sample_file=sample_path.name,
                    sample_id=sample_id,
                    layout_mode=str(data.get("layout_mode", "unknown")),
                    tmin_c=float(np.min(temperature)),
                    tmean_c=float(np.mean(temperature)),
                    tmax_c=float(np.max(temperature)),
                )
            )

        fig_path = out_dir / case_dir.name / "layout_vs_temperature.png"
        _draw_case_overview(case_name, case, sample_rows, fig_path)

        (out_dir / case_dir.name / "summary.json").write_text(
            json.dumps(
                {
                    "case": case_name,
                    "source_case_dir": str(case_dir),
                    "sample_count": len(case_summaries),
                    "samples": [asdict(item) for item in case_summaries],
                    "figure": str(fig_path),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        summary["cases"][case_name] = {
            "case_dir": str(case_dir),
            "sample_count": len(case_summaries),
            "figure": str(fig_path),
        }

    (out_dir / "index.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
