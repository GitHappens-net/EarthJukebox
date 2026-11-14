"""Unit tests for screen-reader descriptive text accessibility (no spoken TTS)."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk

from earth_jukebox.accessibility import (
    AUDIO_LEGEND_TEXT,
    format_dataset_description,
    format_zone_description,
)
from earth_jukebox.audio import CMinorAudioEngine
from earth_jukebox.datasets import DatasetManager
from earth_jukebox.models import AnalysisResult, DatasetEntry
from earth_jukebox.vision import CVAnalyzer


def test_format_zone_description_touching_and_idle() -> None:
    idle = AnalysisResult(
        dominant_color="Blue",
        intensity=0.0,
        density=0.0,
        shape="uniform",
        shape_score=0.0,
        spottiness=0.0,
        swirliness=0.0,
        sensor_size=40,
        sensor_bbox=(0, 0, 0, 0),
        cursor_xy=(0, 0),
        touching=False,
    )
    idle_text = format_zone_description(idle)
    assert "outside the visualization canvas" in idle_text

    entry = DatasetEntry(
        name="Global Sea Surface Temperature",
        path=Path("datasets/sst.png"),
        category="Ocean",
        summary="Warm equatorial currents.",
        unit="deg C",
    )
    active = AnalysisResult(
        dominant_color="Red",
        intensity=0.82,
        density=0.75,
        shape="circular",
        shape_score=0.88,
        spottiness=0.45,
        swirliness=0.65,
        sensor_size=64,
        sensor_bbox=(10, 10, 74, 74),
        cursor_xy=(42, 42),
        touching=True,
    )
    text = format_zone_description(active, dataset=entry)
    assert "Global Sea Surface Temperature" in text
    assert "Red" in text
    assert "64 pixels" in text
    assert "circular" in text
    assert "swirliness 65 percent" in text


def test_format_dataset_description() -> None:
    entry = DatasetEntry(
        name="Carbon Monoxide Concentration",
        path=Path("datasets/co.png"),
        category="Atmosphere",
        summary="Tracks regional biomass burning plumes.",
        unit="ppbv",
    )
    text = format_dataset_description(entry, 2, 8)
    assert "Dataset 2 of 8" in text
    assert "Carbon Monoxide Concentration" in text
    assert "ppbv" in text
    assert "biomass burning plumes" in text


def test_ui_screen_reader_description_shortcuts_and_no_wobble() -> None:
    from earth_jukebox.ui import EarthJukeboxApp

    root = tk.Tk()
    root.withdraw()
    app = EarthJukeboxApp(
        root=root,
        dataset_manager=DatasetManager(),
        cv_analyzer=CVAnalyzer(),
        audio_engine=CMinorAudioEngine(enable_audio=False),
        input_mode="mouse",
    )
    try:
        root.update_idletasks()
        assert app.WINDOW_WIDTH == 992
        assert app.WINDOW_HEIGHT == 806
        assert app.accessible_description == AUDIO_LEGEND_TEXT

        app._on_toggle_auto_scan(None)
        assert "Auto-scan jukebox play started" in app.accessible_description

        app._on_toggle_drums(None)
        assert "Ambient drums" in app.accessible_description

        app.adjust_sensor_size(8)
        assert "48 pixels" in app.accessible_description

        app._on_dataset_info(None)
        assert "Dataset 1 of" in app.accessible_description

        app._on_space_legend(None)
        assert app.accessible_description == AUDIO_LEGEND_TEXT

        assert app.screen_reader_entry.cget("takefocus")
        assert app.canvas.cget("takefocus")
    finally:
        app.close()
