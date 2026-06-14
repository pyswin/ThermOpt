from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from thermopt.layout.geometry import bounds
from thermopt.layout.objects import FloorplanCase, Layout


def draw_layout(ax: plt.Axes, case: FloorplanCase, layout: Layout, title: str) -> None:
    chiplets = case.chiplet_by_id
    powers = [chiplets[p.chiplet_id].power for p in layout.placements]
    pmin, pmax = min(powers), max(powers)
    cmap = plt.get_cmap("inferno")
    for placement in layout.placements:
        chiplet = chiplets[placement.chiplet_id]
        x0, y0, x1, y1 = bounds(case, placement)
        frac = (chiplet.power - pmin) / max(pmax - pmin, 1e-9)
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=cmap(frac), edgecolor="white", lw=1.0)
        ax.add_patch(rect)
        ax.text((x0 + x1) * 0.5, (y0 + y1) * 0.5, chiplet.id, color="white", ha="center", va="center", fontsize=8)
    ax.set_xlim(0, case.outline_width)
    ax.set_ylim(0, case.outline_height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def save_layout_figure(case: FloorplanCase, layout: Layout, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_layout(ax, case, layout, title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def draw_temperature(ax: plt.Axes, temperature: np.ndarray, title: str):
    image = ax.imshow(temperature, origin="lower", cmap="hot", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    return image


def save_temperature_figure(temperature: np.ndarray, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    image = draw_temperature(ax, temperature, title)
    fig.colorbar(image, ax=ax, label="temperature")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_cost_curve(costs: list[float], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(costs)
    ax.set_title(title)
    ax.set_xlabel("recorded step")
    ax.set_ylabel("best total cost")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_final_summary(
    case: FloorplanCase,
    results: list[dict],
    path: Path,
    title: str = "ThermOpt V0 final summary",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [item["name"] for item in results]
    fig = plt.figure(figsize=(14, 10))
    grid = fig.add_gridspec(4, len(results), height_ratios=[1.2, 1.0, 0.8, 0.8])

    for col, item in enumerate(results):
        ax_layout = fig.add_subplot(grid[0, col])
        draw_layout(ax_layout, case, item["layout"], f"{item['name']} layout")
        ax_temp = fig.add_subplot(grid[1, col])
        image = draw_temperature(ax_temp, item["temperature"], f"{item['name']} temperature")
        fig.colorbar(image, ax=ax_temp, fraction=0.046, pad=0.04)

    metric_specs = [
        ("wirelength", "Wirelength"),
        ("tmax", "Tmax"),
        ("top5", "Top-5% temp"),
        ("total_cost", "Total cost"),
    ]
    for i, (key, label) in enumerate(metric_specs):
        row = 2 + i // 2
        col_start = (i % 2) * max(1, len(results) // 2)
        col_end = len(results) if i % 2 else max(1, len(results) // 2)
        ax = fig.add_subplot(grid[row, col_start:col_end])
        values = [item["metrics"][key] for item in results]
        bars = ax.bar(names, values, color="#4c78a8")
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=15)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
