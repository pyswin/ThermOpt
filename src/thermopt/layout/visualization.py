from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from thermopt.layout.geometry import bounds
from thermopt.layout.objects import FloorplanCase, Layout


def save_layout_figure(case: FloorplanCase, layout: Layout, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
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
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_temperature_figure(temperature: np.ndarray, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(temperature, origin="lower", cmap="hot", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
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
