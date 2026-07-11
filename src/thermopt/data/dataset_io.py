from __future__ import annotations

"""
Loader for the milp_wl_dataset snapshot/perturbation JSON records produced by
scripts/dataset_gen/run_milp_dataset.py. Converts the flat per-chiplet dicts back into a
thermopt Layout so a record can be fed to any objective or thermal backend
(e.g. thermopt.thermal.hotspot.HotSpotBackend) exactly like any other Layout
-- the dataset JSON itself is not a type either backend understands.
"""

from thermopt.layout.objects import Layout, Placement


def layout_from_chiplets_json(chiplets: list[dict]) -> Layout:
    """Convert a record's "chiplets" list (see snapshots.json / perturbations.json)
    into a Layout. Placement.x/y is center-based, so this uses cx_mm/cy_mm directly
    (x_mm/y_mm in the JSON are the lower-left corner, kept for readability/tooling
    that expects corner coordinates -- not used here)."""
    return Layout(tuple(Placement(c["name"], c["cx_mm"], c["cy_mm"], c["rotation"]) for c in chiplets))
