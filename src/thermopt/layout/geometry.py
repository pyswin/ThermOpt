from __future__ import annotations

from thermopt.layout.objects import FloorplanCase, Layout, Placement


def bounds(case: FloorplanCase, placement: Placement) -> tuple[float, float, float, float]:
    chiplet = case.chiplet_by_id[placement.chiplet_id]
    width, height = placement.rotated_size(chiplet)
    return placement.x, placement.y, placement.x + width, placement.y + height


def center(case: FloorplanCase, placement: Placement) -> tuple[float, float]:
    x0, y0, x1, y1 = bounds(case, placement)
    return (x0 + x1) * 0.5, (y0 + y1) * 0.5


def overlap_area(case: FloorplanCase, a: Placement, b: Placement) -> float:
    ax0, ay0, ax1, ay1 = bounds(case, a)
    bx0, by0, bx1, by1 = bounds(case, b)
    width = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    height = max(0.0, min(ay1, by1) - max(ay0, by0))
    return width * height


def outline_violation(case: FloorplanCase, placement: Placement) -> float:
    x0, y0, x1, y1 = bounds(case, placement)
    dx = max(0.0, -x0) + max(0.0, x1 - case.outline_width)
    dy = max(0.0, -y0) + max(0.0, y1 - case.outline_height)
    return dx * dx + dy * dy


def total_overlap_penalty(case: FloorplanCase, layout: Layout) -> float:
    placements = layout.placements
    overlap = 0.0
    for i, left in enumerate(placements):
        for right in placements[i + 1 :]:
            overlap += overlap_area(case, left, right)
    return overlap / max(case.total_chiplet_area, 1e-9)


def total_outline_penalty(case: FloorplanCase, layout: Layout) -> float:
    norm = max(case.outline_width * case.outline_height, 1e-9)
    return sum(outline_violation(case, placement) for placement in layout.placements) / norm


def hpwl(case: FloorplanCase, layout: Layout) -> float:
    placement_by_id = layout.by_id
    total = 0.0
    for net in case.nets:
        xs: list[float] = []
        ys: list[float] = []
        for chiplet_id in net.chiplets:
            cx, cy = center(case, placement_by_id[chiplet_id])
            xs.append(cx)
            ys.append(cy)
        if xs:
            total += (max(xs) - min(xs)) + (max(ys) - min(ys))
    return total
