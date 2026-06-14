from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable


@dataclass(frozen=True)
class Chiplet:
    id: str
    width: float
    height: float
    power: float


@dataclass(frozen=True)
class Net:
    id: str
    chiplets: tuple[str, ...]


@dataclass(frozen=True)
class FloorplanCase:
    chiplets: tuple[Chiplet, ...]
    nets: tuple[Net, ...]
    outline_width: float
    outline_height: float

    @property
    def chiplet_ids(self) -> tuple[str, ...]:
        return tuple(chiplet.id for chiplet in self.chiplets)

    @property
    def chiplet_by_id(self) -> dict[str, Chiplet]:
        return {chiplet.id: chiplet for chiplet in self.chiplets}

    @property
    def total_chiplet_area(self) -> float:
        return sum(chiplet.width * chiplet.height for chiplet in self.chiplets)


@dataclass(frozen=True)
class Placement:
    chiplet_id: str
    x: float
    y: float
    rotation: int = 0

    def rotated_size(self, chiplet: Chiplet) -> tuple[float, float]:
        if self.rotation % 180 == 90:
            return chiplet.height, chiplet.width
        return chiplet.width, chiplet.height

    def moved(self, x: float | None = None, y: float | None = None) -> "Placement":
        return replace(self, x=self.x if x is None else x, y=self.y if y is None else y)

    def rotated(self) -> "Placement":
        return replace(self, rotation=90 if self.rotation % 180 == 0 else 0)


@dataclass(frozen=True)
class Layout:
    placements: tuple[Placement, ...]

    @property
    def by_id(self) -> dict[str, Placement]:
        return {placement.chiplet_id: placement for placement in self.placements}

    def replace_placement(self, new_placement: Placement) -> "Layout":
        return Layout(
            tuple(
                new_placement if p.chiplet_id == new_placement.chiplet_id else p
                for p in self.placements
            )
        )

    def replace_many(self, placements: Iterable[Placement]) -> "Layout":
        updates = {placement.chiplet_id: placement for placement in placements}
        return Layout(tuple(updates.get(p.chiplet_id, p) for p in self.placements))
