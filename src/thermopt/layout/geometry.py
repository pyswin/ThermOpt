from __future__ import annotations

from thermopt.layout.objects import FloorplanCase, Layout, Placement


def bounds(case: FloorplanCase, placement: Placement) -> tuple[float, float, float, float]:
    chiplet = case.chiplet_by_id[placement.chiplet_id]
    width, height = placement.rotated_size(chiplet)
    return placement.x, placement.y, placement.x + width, placement.y + height


def center(case: FloorplanCase, placement: Placement) -> tuple[float, float]:
    x0, y0, x1, y1 = bounds(case, placement)
    return (x0 + x1) * 0.5, (y0 + y1) * 0.5


def rotate_offset(offset_x: float, offset_y: float, rotation: int) -> tuple[float, float]:
    rotation = rotation % 360
    if rotation == 0:
        return offset_x, offset_y
    if rotation == 90:
        return -offset_y, offset_x
    if rotation == 180:
        return -offset_x, -offset_y
    if rotation == 270:
        return offset_y, -offset_x
    raise ValueError(f"unsupported rotation: {rotation}")


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
        if len(set(net.chiplets)) < 2:
            continue
        offsets = net.pin_offsets or tuple((0.0, 0.0) for _ in net.chiplets)
        for chiplet_id, (offset_x, offset_y) in zip(net.chiplets, offsets):
            placement = placement_by_id[chiplet_id]
            cx, cy = center(case, placement)
            offset_x, offset_y = rotate_offset(offset_x, offset_y, placement.rotation)
            xs.append(cx)
            ys.append(cy)
            xs[-1] += offset_x
            ys[-1] += offset_y
        if xs:
            total += (max(xs) - min(xs)) + (max(ys) - min(ys))
    return total
