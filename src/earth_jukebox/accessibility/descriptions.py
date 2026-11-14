"""Screen-reader descriptive text helpers for Earth Jukebox (no spoken TTS)."""

from __future__ import annotations

from typing import Optional

from earth_jukebox.models import AnalysisResult, DatasetEntry

AUDIO_LEGEND_TEXT = (
    "Earth Jukebox: 75 BPM C natural minor sonification. "
    "Cool blues and greens play lower warm octaves; warm reds and oranges play higher registers. "
    "Low density sustains long chords, medium density plays eighth-note pulses, and high density "
    "plays sixteenth-note arpeggios. Intensity controls volume and brightness. "
    "Swirliness adds harmonic arpeggio motion, spottiness adds octave bell accents, and shape "
    "selects timbre. Shortcuts: Space for legend, R to read current zone, I for dataset info, "
    "P for auto-scan, D for drums, plus or minus to resize sensor box, Tab or 1-8 for dataset."
)


def format_zone_description(
    analysis: AnalysisResult,
    dataset: Optional[DatasetEntry] = None,
) -> str:
    """Return a rich, screen-reader-friendly description of the current sensor zone."""
    if not analysis.touching:
        return (
            "Cursor is outside the visualization canvas. "
            "Move the pointer over the NASA map or press P to start auto-scan."
        )

    dataset_prefix = f"Dataset: {dataset.name}. " if dataset is not None else ""
    density_pct = int(round(analysis.density * 100))
    intensity_pct = int(round(analysis.intensity * 100))
    spot_pct = int(round(analysis.spottiness * 100))
    swirl_pct = int(round(analysis.swirliness * 100))
    x, y = analysis.cursor_xy
    return (
        f"{dataset_prefix}"
        f"Position X {x}, Y {y}, sensor box {analysis.sensor_size} pixels. "
        f"Dominant color {analysis.dominant_color}, intensity {intensity_pct} percent, "
        f"edge density {density_pct} percent. "
        f"Shape {analysis.shape}, spottiness {spot_pct} percent, swirliness {swirl_pct} percent."
    )


def format_dataset_description(entry: DatasetEntry, index: int, total: int) -> str:
    """Return a screen-reader-friendly description of the active dataset."""
    unit_text = f" Measured in {entry.unit}." if entry.unit else ""
    summary = entry.summary or f"Viewing {entry.name}."
    return f"Dataset {index} of {total}: {entry.name} ({entry.category}).{unit_text} {summary}"
