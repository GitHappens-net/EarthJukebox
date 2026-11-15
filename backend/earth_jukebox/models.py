"""Shared immutable data contracts for Earth Jukebox."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class TouchPoint:
    """Normalized pointer/touchpad coordinate."""

    x: int
    y: int
    x_ratio: float
    y_ratio: float
    touching: bool


@dataclass(frozen=True)
class AnalysisResult:
    """Computer-vision metrics extracted from the cursor sensor box."""

    dominant_color: str
    intensity: float
    density: float
    shape: str = "uniform"
    shape_score: float = 0.0
    spottiness: float = 0.0
    swirliness: float = 0.0
    sensor_size: int = 40
    sensor_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    cursor_xy: tuple[int, int] = (0, 0)
    touching: bool = True


@dataclass(frozen=True)
class DatasetEntry:
    """Metadata and scientific context for a NASA Earth visualization."""

    name: str
    path: Path
    category: str = "NASA Earth Observation"
    unit: str = ""
    summary: str = ""
    color_meanings: tuple[tuple[str, str], ...] = ()

    def meaning_for(self, color: str) -> Optional[str]:
        """Look up dataset-specific scientific meaning for a dominant color."""
        for key, description in self.color_meanings:
            if key == color:
                return description
        return None
