from __future__ import annotations

import numpy as np

from thermopt.layout.objects import Chiplet, FloorplanCase, Layout, Net, Placement


def generate_random_case(config: dict, seed: int) -> FloorplanCase:
    rng = np.random.default_rng(seed)
    n = int(config["num_chiplets"])
    widths = rng.uniform(config["min_size"], config["max_size"], size=n)
    heights = rng.uniform(config["min_size"], config["max_size"], size=n)
    powers = rng.uniform(config["min_power"], config["max_power"], size=n)

    hot_count = max(1, int(round(n * config.get("hot_chiplet_fraction", 0.2))))
    hot_idx = rng.choice(n, size=hot_count, replace=False)
    powers[hot_idx] *= float(config.get("hot_power_multiplier", 2.0))

    chiplets = tuple(
        Chiplet(id=f"C{i}", width=float(widths[i]), height=float(heights[i]), power=float(powers[i]))
        for i in range(n)
    )
    chiplet_ids = [chiplet.id for chiplet in chiplets]

    nets: list[Net] = []
    for i in range(int(config.get("num_nets", 0))):
        degree = int(rng.integers(config.get("net_min_degree", 2), config.get("net_max_degree", 4) + 1))
        degree = min(degree, n)
        pins = tuple(rng.choice(chiplet_ids, size=degree, replace=False).tolist())
        nets.append(Net(id=f"N{i}", chiplets=pins))

    return FloorplanCase(
        chiplets=chiplets,
        nets=tuple(nets),
        outline_width=float(config["outline_width"]),
        outline_height=float(config["outline_height"]),
    )


def random_initial_layout(case: FloorplanCase, seed: int) -> Layout:
    rng = np.random.default_rng(seed)
    placements: list[Placement] = []
    for chiplet in case.chiplets:
        rotation = int(rng.choice([0, 90, 180, 270]))
        width, height = (chiplet.height, chiplet.width) if rotation % 180 == 90 else (chiplet.width, chiplet.height)
        max_x = max(0.0, case.outline_width - width)
        max_y = max(0.0, case.outline_height - height)
        placements.append(
            Placement(
                chiplet_id=chiplet.id,
                x=float(rng.uniform(0.0, max_x)),
                y=float(rng.uniform(0.0, max_y)),
                rotation=rotation,
            )
        )
    return Layout(tuple(placements))
