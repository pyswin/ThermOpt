from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from thermopt.layout.objects import FloorplanCase, Layout


@dataclass(frozen=True)
class CaseInput:
    name: str
    case: FloorplanCase
    layout: Layout
    source_path: Path
