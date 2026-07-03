from __future__ import annotations

"""
Loader for the milp_wl_dataset snapshot/perturbation JSON records produced by
scripts/run_milp_dataset.py. Converts the flat per-chiplet dicts back into a
thermopt Layout so a record can be fed to any objective or thermal backend
(e.g. thermopt.thermal.hotspot.HotSpotBackend) exactly like any other Layout
-- the dataset JSON itself is not a type either backend understands.
"""

from thermopt.layout.objects import Layout, Placement


def layout_from_chiplets_json(chiplets: list[dict]) -> Layout:
    """Convert a record's "chiplets" list (see snapshots.json / perturbations.json)
    into a Layout. Only x_mm/y_mm/rotation are used to reconstruct placements --
    cx_mm/cy_mm/width_mm/height_mm/power are derived fields kept in the JSON for
    convenience and are not needed here."""
    return Layout(tuple(Placement(c["name"], c["x_mm"], c["y_mm"], c["rotation"]) for c in chiplets))
