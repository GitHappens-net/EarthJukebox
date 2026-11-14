"""Screen-reader-friendly descriptive text for Earth Jukebox."""

from __future__ import annotations

from earth_jukebox.accessibility.descriptions import (
    AUDIO_LEGEND_TEXT,
    format_dataset_description,
    format_zone_description,
)
from earth_jukebox.accessibility.tts import AccessibilityTTS

__all__ = [
    "AUDIO_LEGEND_TEXT",
    "AccessibilityTTS",
    "format_dataset_description",
    "format_zone_description",
]
