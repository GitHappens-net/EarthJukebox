"""Earth Jukebox — Accessible NASA Imagery Sonification Engine (Windows Proxy & Core).

2026 NASA Space Apps Challenge: "Earth Jukebox" (NASA Earth Information Center).

Architecture Overview
---------------------
This module strictly separates the four system layers using an object-oriented
design so the Windows mouse proxy can be swapped out cleanly for the Linux
``evdev`` hardware trackpad reader (``backend/tests/trackpad.py``):

1. **Module 1 — Input Proxy & NASA EIC Dataset Manager**
   - ``TouchPoint``: Shared immutable coordinate contract matching
     ``backend/tests/trackpad.py``.
   - ``CoordinateInputSource`` (ABC): Common interface implemented by both
     ``WindowsMouseInputProxy`` (hover/mouse on Windows) and
     ``EvdevTrackpadInputProxy`` (3D-printed tactile map over Linux trackpad).
   - ``extract_sensor_box``: Edge-safe 40x40 pixel bounding-box extractor.
   - ``DatasetManager``: Discovers, normalizes, caches, and curates metadata for
     NASA Earth Information Center (EIC) visualizations in ``images/``.

2. **Module 2 — Computer Vision Analysis (``CVAnalyzer``)**
   - Inspects the 40x40 sensor patch in real time to produce:
     * Dominant Color category (``Red``, ``Orange``, ``Green``, ``Blue``,
       ``Black``, ``White``) with chromatic streamline salience and deep-crimson
       Fire Weather core detection
     * Intensity (``0.0`` to ``1.0``)
     * Edge Density (``0.0`` to ``1.0``) via Gaussian-blurred Canny edge detection

3. **Module 3 — Beat-Synchronized C-Minor Audio Engine (``CMinorAudioEngine``)**
   - Runs on a dedicated daemon thread driven by a **locked 90 BPM 16th-note
     master clock** (``166.67 ms`` per 16th step, 16 steps per 4/4 bar).
   - Strictly quantizes all melody, harmony, and bass pitches to **C Natural
     Minor** (``C, D, Eb, F, G, Ab, Bb``).
   - Multi-layered musical arrangement:
     * **Ambient Drum & Percussion Groove (MIDI Channel 9)** adapting to data
       density and intensity (toggleable with ``D``)
     * **4-Bar C-Minor Harmonic Pad & Bass Bed** (``Cm9 -> Abmaj7 -> Fm7 -> Gm7``)
     * **Beat-Quantized Data Sonification Voice** falling strictly on 90 BPM
       subdivisions (Whole / Half / 8th / 16th Arpeggio) with reverb/chorus CC
       and downbeat harmonic dyads.

4. **Module 4 — Accessibility & Keyboard-First Design (``AccessibilityTTS`` & UI)**
   - Runs TTS asynchronously on its own daemon thread (``pyttsx3`` preferred,
     with native Windows ``System.Speech`` / Linux fallback).
   - Keyboard shortcuts: ``Space`` (Audio Legend), ``R`` (Read Zone Data),
     ``I`` (Dataset Scientific Briefing), ``Tab`` / ``1–8`` (Switch Dataset),
     ``P`` (Auto-Scan Jukebox Mode), ``D`` (Toggle Ambient Drums), ``Esc`` (Exit).
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import math
import platform
import queue
import shutil
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import tkinter as tk
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover - optional only in headless test runs
    tk = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]
    ImageTk = None  # type: ignore[assignment]

try:
    import pygame
    import pygame.midi
except ImportError:
    pygame = None  # type: ignore[assignment]

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None  # type: ignore[assignment]

try:
    import pythoncom
except ImportError:
    pythoncom = None  # type: ignore[assignment]


# =============================================================================
# SHARED DATA CONTRACTS (Windows Mouse Proxy <-> Linux evdev Trackpad)
# =============================================================================
# === [LINUX EVDEV INTEGRATION POINT] ===
# Keep `TouchPoint` field-for-field identical to `backend/tests/trackpad.py`:
#   x, y, x_ratio, y_ratio, touching
# Because both WindowsMouseInputProxy and `trackpad.read_touchpad(...)` emit
# this exact dataclass into a bounded `queue.Queue[TouchPoint]`, the CV, Audio,
# and TTS modules remain 100% platform-agnostic.
@dataclass(frozen=True)
class TouchPoint:
    """Normalized pointer/touchpad coordinate matching ``trackpad.py``."""

    x: int
    y: int
    x_ratio: float
    y_ratio: float
    touching: bool


@dataclass(frozen=True)
class AnalysisResult:
    """Computer-vision metrics extracted from the 40x40 cursor sensor box."""

    dominant_color: str
    intensity: float
    density: float
    sensor_bbox: tuple[int, int, int, int]
    cursor_xy: tuple[int, int]
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


def push_latest_point(points_queue: queue.Queue[TouchPoint], point: TouchPoint) -> None:
    """Push a ``TouchPoint`` into a bounded queue, dropping stale frames first."""
    if points_queue.full():
        try:
            points_queue.get_nowait()
        except queue.Empty:
            pass
    try:
        points_queue.put_nowait(point)
    except queue.Full:
        pass


# =============================================================================
# MODULE 1: INPUT PROXY & NASA EIC DATASET MANAGEMENT
# =============================================================================
def extract_sensor_box(
    image: np.ndarray,
    img_x: int,
    img_y: int,
    box_size: int = 40,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Extract an edge-safe ``box_size x box_size`` patch around ``(img_x, img_y)``.

    Near the borders or corners of the image (including negative or
    out-of-bounds coordinates), the cursor is clamped to valid image bounds and
    the bounding box is shifted inward so a full ``40x40`` patch is returned
    whenever the image is at least ``40x40`` pixels, never crashing or slicing
    an empty array.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot extract sensor box from an empty image.")

    height, width = image.shape[:2]
    size = max(1, int(box_size))

    clamped_x = max(0, min(int(img_x), width - 1))
    clamped_y = max(0, min(int(img_y), height - 1))

    half = size // 2
    max_x1 = max(0, width - size)
    max_y1 = max(0, height - size)

    x1 = max(0, min(clamped_x - half, max_x1))
    y1 = max(0, min(clamped_y - half, max_y1))
    x2 = min(width, x1 + size)
    y2 = min(height, y1 + size)

    patch = image[y1:y2, x1:x2]
    return patch, (x1, y1, x2, y2)


class CoordinateInputSource(ABC):
    """Abstract coordinate stream feeding ``TouchPoint`` items into ``points_queue``."""

    def __init__(
        self,
        points_queue: queue.Queue[TouchPoint],
        stop_event: threading.Event,
    ) -> None:
        self.points_queue = points_queue
        self.stop_event = stop_event

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Human-readable label describing the active hardware/proxy source."""

    @abstractmethod
    def start(self) -> None:
        """Begin emitting ``TouchPoint`` events into ``self.points_queue``."""

    @abstractmethod
    def stop(self) -> None:
        """Stop emitting events and release input resources."""


class WindowsMouseInputProxy(CoordinateInputSource):
    """Windows stand-in input driver tracking Tkinter canvas hover coordinates.

    === [LINUX EVDEV INTEGRATION POINT] ===
    On Linux with the 3D-printed tactile overlay mounted over the trackpad,
    replace this class instantiation with ``EvdevTrackpadInputProxy`` (or pass
    ``--input evdev`` on the CLI). Downstream ``CVAnalyzer`` and
    ``CMinorAudioEngine`` read only from ``points_queue`` and are completely
    unaware of whether coordinates came from a Windows mouse or Linux ``evdev``.
    """

    def __init__(
        self,
        canvas: "tk.Canvas",
        points_queue: queue.Queue[TouchPoint],
        stop_event: threading.Event,
        on_user_activity: Optional[callable] = None,
    ) -> None:
        super().__init__(points_queue, stop_event)
        self.canvas = canvas
        self.on_user_activity = on_user_activity
        self._binding_ids: list[tuple[str, str]] = []
        self._last_x = 0
        self._last_y = 0

    @property
    def source_name(self) -> str:
        return "Windows Mouse Proxy"

    def start(self) -> None:
        """Bind canvas pointer motion and leave events."""
        events = [
            ("<Motion>", self._on_motion),
            ("<Enter>", self._on_motion),
            ("<ButtonPress-1>", self._on_motion),
            ("<B1-Motion>", self._on_motion),
            ("<Leave>", self._on_leave),
        ]
        for sequence, handler in events:
            bind_id = self.canvas.bind(sequence, handler, add="+")
            if bind_id:
                self._binding_ids.append((sequence, bind_id))

    def stop(self) -> None:
        """Unbind canvas handlers."""
        for sequence, bind_id in self._binding_ids:
            try:
                self.canvas.unbind(sequence, bind_id)
            except Exception:
                pass
        self._binding_ids.clear()

    def _canvas_to_touchpoint(self, raw_x: int, raw_y: int, touching: bool) -> TouchPoint:
        width = max(1, int(self.canvas.winfo_width()))
        height = max(1, int(self.canvas.winfo_height()))
        clamped_x = max(0, min(int(raw_x), width - 1))
        clamped_y = max(0, min(int(raw_y), height - 1))
        x_ratio = clamped_x / max(1, width - 1) if width > 1 else 0.0
        y_ratio = clamped_y / max(1, height - 1) if height > 1 else 0.0
        self._last_x = clamped_x
        self._last_y = clamped_y
        return TouchPoint(
            x=clamped_x,
            y=clamped_y,
            x_ratio=float(np.clip(x_ratio, 0.0, 1.0)),
            y_ratio=float(np.clip(y_ratio, 0.0, 1.0)),
            touching=touching,
        )

    def _on_motion(self, event: "tk.Event") -> None:
        if self.stop_event.is_set():
            return
        if self.on_user_activity is not None:
            self.on_user_activity()
        point = self._canvas_to_touchpoint(event.x, event.y, touching=True)
        push_latest_point(self.points_queue, point)

    def _on_leave(self, _event: "tk.Event") -> None:
        if self.stop_event.is_set():
            return
        width = max(1, int(self.canvas.winfo_width()))
        height = max(1, int(self.canvas.winfo_height()))
        x_ratio = self._last_x / max(1, width - 1) if width > 1 else 0.0
        y_ratio = self._last_y / max(1, height - 1) if height > 1 else 0.0
        push_latest_point(
            self.points_queue,
            TouchPoint(
                x=self._last_x,
                y=self._last_y,
                x_ratio=float(np.clip(x_ratio, 0.0, 1.0)),
                y_ratio=float(np.clip(y_ratio, 0.0, 1.0)),
                touching=False,
            ),
        )


class EvdevTrackpadInputProxy(CoordinateInputSource):
    """Linux hardware adapter wrapping ``backend/tests/trackpad.py``.

    === [LINUX EVDEV INTEGRATION POINT] ===
    Your Linux teammate does not need to rewrite any CV, Audio, or TTS code.
    This adapter imports ``find_touchpad`` and ``read_touchpad`` directly from
    ``backend/tests/trackpad.py`` and runs ``read_touchpad`` on a background
    thread pushing ``TouchPoint`` structs into ``self.points_queue``.
    """

    def __init__(
        self,
        points_queue: queue.Queue[TouchPoint],
        stop_event: threading.Event,
        device_path: Optional[str] = None,
    ) -> None:
        super().__init__(points_queue, stop_event)
        self.device_path = device_path
        self._thread: Optional[threading.Thread] = None
        self._device_label = "Linux evdev Trackpad"

    @property
    def source_name(self) -> str:
        return self._device_label

    def start(self) -> None:
        trackpad_file = Path(__file__).resolve().parent / "tests" / "trackpad.py"
        spec = importlib.util.spec_from_file_location("earth_jukebox_trackpad", trackpad_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not locate Linux trackpad driver at {trackpad_file}")
        trackpad_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trackpad_mod)

        device = trackpad_mod.find_touchpad(self.device_path)
        self._device_label = f"evdev: {device.name} [{device.path}]"
        self._thread = threading.Thread(
            target=trackpad_mod.read_touchpad,
            args=(device, self.points_queue, self.stop_event),
            name="EvdevTrackpadReader",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)


# Curated scientific metadata for the NASA EIC imagery in `images/` & tactile map
NASA_EIC_CATALOG: dict[str, dict[str, object]] = {
    "aerosol optical depth.png": {
        "name": "Global PM2.5 Aerosol Optical Depth (2019)",
        "category": "Air Quality & Aerosols (MODIS / MISR / SeaWiFS)",
        "unit": "µg/m³",
        "summary": (
            "Global fine particulate matter PM2.5 concentrations. Dark blue marks clean air "
            "under 5 micrograms per cubic meter, yellow and orange indicate elevated dust and "
            "pollution from 10 to 80, and dark red warns of severe hotspots up to 139."
        ),
        "color_meanings": (
            ("Red", "severe fine particulate hotspot, 40 to 139 micrograms per cubic meter"),
            ("Orange", "elevated dust or smoke particulate load, 10 to 80 micrograms per cubic meter"),
            ("Blue", "clean air corridor or ocean zone under 10 micrograms per cubic meter"),
            ("White", "polar land mask or chart margin"),
        ),
    },
    "aerosols_2025-08-19t20_00_00_1920x1080.png": {
        "name": "GEOS Global Aerosol Transport (Aug 2025)",
        "category": "Atmospheric Transport (NASA GEOS)",
        "unit": "Aerosol Plumes",
        "summary": (
            "NASA GEOS global aerosol transport simulation. Swirling green currents trace marine "
            "sea salt over storm oceans, violet plumes trace Saharan mineral dust crossing the "
            "Atlantic, and bright orange-red filaments trace wildfire smoke and sulfates."
        ),
        "color_meanings": (
            ("Red", "intense wildfire smoke or concentrated biomass burning emission"),
            ("Orange", "Saharan mineral dust plume or warm smoke transport corridor"),
            ("Green", "marine sea-salt aerosol lifted by ocean storm winds"),
            ("Blue", "cyclonic sea-spray swirl or temperate marine aerosol"),
            ("Black", "calm background atmosphere with minimal aerosol load"),
        ),
    },
    "ecco2_and_grid.jpg": {
        "name": "ECCO2 Perpetual Ocean Currents & Temperature",
        "category": "Ocean Circulation & Climate (NASA ECCO2)",
        "unit": "°C / Flow",
        "summary": (
            "NASA ECCO2 high-resolution global ocean circulation visualization. Textured streams "
            "trace surface currents transitioning from cold polar blues and temperate greens to "
            "warm equatorial oranges and reds."
        ),
        "color_meanings": (
            ("Red", "warmest tropical ocean current or intense boundary eddy"),
            ("Orange", "warm equatorial ocean current or subtropical gyre"),
            ("Green", "temperate mid-latitude current stream and upwelling belt"),
            ("Blue", "cold polar and subpolar ocean current"),
            ("Black", "continental landmass or calm abyssal basin"),
        ),
    },
    "fireweatherindex_conus_080126_1920x1080.png": {
        "name": "North America Fire Weather Index (Aug 2026)",
        "category": "Wildfire Extremes (NASA GFWED)",
        "unit": "FWI Risk",
        "summary": (
            "North America Fire Weather Index for August 1, 2026. Green marks very low fire "
            "danger, yellow and orange mark moderate danger, red marks very high danger, and "
            "deep crimson marks extreme wildfire weather conditions across the western interior."
        ),
        "color_meanings": (
            ("Red", "very high to extreme fire weather danger"),
            ("Orange", "low to moderate fire weather danger"),
            ("Green", "very low fire weather danger with moist conditions"),
            ("Black", "ocean water or unvegetated high-altitude mask"),
            ("Blue", "legend box or coastal water boundary"),
        ),
    },
    "fwi_forecast_latest.png": {
        "name": "Global Fire Weather Index Forecast (Nov 2026)",
        "category": "Global Wildfire Forecast (NASA GFWED)",
        "unit": "FWI Risk",
        "summary": (
            "Global Fire Weather Index forecast for November 14, 2026. Grey masks unvegetated "
            "desert or snow, green marks very low fire risk, orange marks moderate risk, and "
            "deep red highlights extreme fire weather belts across Africa and Australia."
        ),
        "color_meanings": (
            ("Red", "very high to extreme global fire weather risk belt"),
            ("Orange", "low to moderate fire weather risk"),
            ("Green", "very low fire weather risk"),
            ("White", "unvegetated desert, polar snow, or ocean background mask"),
            ("Black", "extreme wildfire core or polar shoreline"),
        ),
    },
    "global_co2_airs_60south_720x480.00253.png": {
        "name": "AIRS Global Tropospheric CO2 (365–435 ppm)",
        "category": "Greenhouse Gases (NASA Aqua / AIRS)",
        "unit": "ppm CO2",
        "summary": (
            "Global tropospheric carbon dioxide measured by NASA's AIRS instrument. Warm oranges "
            "track elevated CO2 around 395 to 415 parts per million, while dark red hatches mark "
            "regional concentration peaks approaching 435 parts per million."
        ),
        "color_meanings": (
            ("Red", "peak regional CO2 concentration approaching 435 parts per million"),
            ("Orange", "elevated tropospheric CO2 around 395 to 415 parts per million"),
            ("Black", "polar frame border or legend banner"),
            ("White", "baseline 365 parts per million scale floor"),
        ),
    },
    "screenshot 2026-11-14 161145.png": {
        "name": "NASA SVS 3D Carbon Dioxide Column (May 2021)",
        "category": "Atmospheric Carbon Plumes (NASA SVS)",
        "unit": "ppm CO2",
        "summary": (
            "NASA Scientific Visualization Studio 3D carbon dioxide column with a global mean of "
            "416.8 parts per million. Swirling orange-red clouds depict dense northern hemisphere "
            "CO2 plumes contrasting with lower southern levels and bright Antarctic ice."
        ),
        "color_meanings": (
            ("Red", "dense CO2 filament approaching 425 parts per million"),
            ("Orange", "elevated northern hemisphere CO2 plume above 405 parts per million"),
            ("White", "Antarctic ice sheet or baseline scale band"),
            ("Black", "lower southern hemisphere CO2 corridor over dark ocean"),
        ),
    },
    "world map.jpg": {
        "name": "World Topography and Bathymetry (3D Tactile Reference)",
        "category": "3D Tactile Overlay Map",
        "unit": "Topography",
        "summary": (
            "Reference world map matching the 3D-printed tactile trackpad overlay for "
            "geographic orientation across continents and oceans."
        ),
        "color_meanings": (),
    },
}


class DatasetManager:
    """Manages loading, normalizing, caching, and cycling NASA imagery datasets."""

    SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
    MAX_WORKING_WIDTH = 960
    MAX_WORKING_HEIGHT = 540

    def __init__(
        self,
        datasets_dir: Optional[Path] = None,
        world_map_path: Optional[Path] = None,
        include_tactile_reference: bool = True,
    ) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        self._explicit_dir = datasets_dir is not None
        self.world_map_path = world_map_path or (repo_root / "3D files" / "World Map.jpg")
        self.include_tactile_reference = include_tactile_reference

        if datasets_dir is not None:
            self.datasets_dir = datasets_dir
        else:
            images_dir = repo_root / "images"
            has_user_images = images_dir.is_dir() and any(
                p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTENSIONS
                for p in images_dir.iterdir()
            )
            self.datasets_dir = images_dir if has_user_images else (repo_root / "backend" / "datasets")

        self._entries: list[DatasetEntry] = []
        self._index = 0
        self._image_cache: dict[Path, np.ndarray] = {}

    def ensure_default_datasets(self) -> None:
        """Populate fallback sample datasets when pointing at an empty datasets directory."""
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        existing = [
            p
            for p in self.datasets_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]
        if existing and not self._explicit_dir:
            return

        world_dest = self.datasets_dir / "01_World_Topography_and_Bathymetry.jpg"
        if self.world_map_path.exists() and not world_dest.exists():
            shutil.copy2(self.world_map_path, world_dest)

        sst_path = self.datasets_dir / "02_NASA_Sea_Surface_Temperature_Anomaly.png"
        if not sst_path.exists():
            cv2.imwrite(str(sst_path), self._generate_sst_dataset())

        aerosol_path = self.datasets_dir / "03_NASA_Wildfire_and_Storm_Aerosol_Index.png"
        if not aerosol_path.exists():
            cv2.imwrite(str(aerosol_path), self._generate_wildfire_storm_dataset())

    def load_datasets(self) -> list[DatasetEntry]:
        """Discover all valid NASA dataset images and bind scientific metadata."""
        if not self.datasets_dir.exists() or self._explicit_dir:
            self.ensure_default_datasets()

        candidate_paths = [
            path
            for path in sorted(self.datasets_dir.iterdir(), key=lambda p: p.name.lower())
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

        if not candidate_paths:
            self.ensure_default_datasets()
            candidate_paths = [
                path
                for path in sorted(self.datasets_dir.iterdir(), key=lambda p: p.name.lower())
                if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ]

        if (
            self.include_tactile_reference
            and self.world_map_path.exists()
            and all(p.resolve() != self.world_map_path.resolve() for p in candidate_paths)
            and not self._explicit_dir
        ):
            candidate_paths.append(self.world_map_path)

        entries = [self._build_entry(path) for path in candidate_paths]
        if not entries:
            raise FileNotFoundError(f"No NASA dataset images found in {self.datasets_dir}")

        self._entries = entries
        self._index %= len(self._entries)
        return list(self._entries)

    def current_dataset(self) -> DatasetEntry:
        """Return the active dataset entry."""
        if not self._entries:
            self.load_datasets()
        return self._entries[self._index]

    def current_index(self) -> tuple[int, int]:
        """Return ``(1_based_index, total_count)`` for HUD display."""
        if not self._entries:
            self.load_datasets()
        return self._index + 1, len(self._entries)

    def select_dataset(self, index: int) -> DatasetEntry:
        """Select a dataset by 0-based index (wrapping safely)."""
        if not self._entries:
            self.load_datasets()
        self._index = index % len(self._entries)
        return self._entries[self._index]

    def next_dataset(self) -> DatasetEntry:
        """Advance cyclically to the next NASA dataset."""
        if not self._entries:
            self.load_datasets()
        self._index = (self._index + 1) % len(self._entries)
        return self._entries[self._index]

    def load_current_image(self) -> np.ndarray:
        """Read, normalize oversized frames to working resolution, and cache BGR image."""
        entry = self.current_dataset()
        cached = self._image_cache.get(entry.path)
        if cached is not None:
            return cached

        image = cv2.imread(str(entry.path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(f"Failed to decode dataset image: {entry.path}")

        height, width = image.shape[:2]
        if width > self.MAX_WORKING_WIDTH or height > self.MAX_WORKING_HEIGHT:
            scale = min(self.MAX_WORKING_WIDTH / width, self.MAX_WORKING_HEIGHT / height)
            target_w = max(40, int(round(width * scale)))
            target_h = max(40, int(round(height * scale)))
            image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)

        self._image_cache[entry.path] = image
        return image

    def _build_entry(self, path: Path) -> DatasetEntry:
        meta = NASA_EIC_CATALOG.get(path.name.lower())
        if meta is not None:
            return DatasetEntry(
                name=str(meta["name"]),
                path=path,
                category=str(meta["category"]),
                unit=str(meta["unit"]),
                summary=str(meta["summary"]),
                color_meanings=tuple(meta["color_meanings"]),  # type: ignore[arg-type]
            )
        return DatasetEntry(
            name=self._format_name(path.stem),
            path=path,
            category="NASA Earth Visualization",
            unit="",
            summary=f"Exploring NASA dataset {self._format_name(path.stem)}.",
        )

    @staticmethod
    def _format_name(stem: str) -> str:
        parts = stem.replace("-", " ").split("_")
        if parts and parts[0].isdigit():
            parts = parts[1:]
        cleaned = " ".join(part for part in parts if part).strip()
        return cleaned if cleaned else stem

    @staticmethod
    def _generate_sst_dataset(width: int = 900, height: int = 500) -> np.ndarray:
        """Synthesize a smooth-to-turbulent Sea Surface Temperature fallback map."""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        y_grid, x_grid = np.mgrid[0:height, 0:width]
        ocean_wave = (np.sin(x_grid / 95.0) * 14 + np.cos(y_grid / 70.0) * 10).astype(np.int16)
        img[:, :, 0] = np.clip(190 + ocean_wave, 130, 235).astype(np.uint8)
        img[:, :, 1] = np.clip(85 + ocean_wave // 2, 50, 135).astype(np.uint8)
        img[:, :, 2] = 20

        cv2.rectangle(img, (0, 0), (width, 46), (242, 245, 248), -1)
        cv2.rectangle(img, (0, height - 48), (width, height), (238, 242, 246), -1)

        continents = [
            np.array([[120, 90], [270, 80], [310, 175], [220, 225], [140, 195]]),
            np.array([[220, 265], [305, 290], [285, 420], [215, 405], [190, 320]]),
            np.array([[470, 85], [760, 75], [810, 200], [670, 250], [540, 210], [450, 155]]),
            np.array([[485, 215], [595, 225], [610, 385], [520, 410], [465, 315]]),
        ]
        for poly in continents:
            cv2.fillPoly(img, [poly], (45, 165, 55))
            cv2.polylines(img, [poly], True, (215, 235, 205), 3)

        cv2.ellipse(img, (375, 260), (145, 42), 0, 0, 360, (20, 140, 245), -1)
        cv2.ellipse(img, (375, 260), (75, 22), 0, 0, 360, (25, 35, 240), -1)
        rng = np.random.default_rng(2026)
        for _ in range(110):
            cx = int(rng.integers(305, 445))
            cy = int(rng.integers(240, 280))
            cv2.circle(img, (cx, cy), int(rng.integers(2, 6)), (0, 210, 255), 1)
        return img

    @staticmethod
    def _generate_wildfire_storm_dataset(width: int = 900, height: int = 500) -> np.ndarray:
        """Synthesize a night-side Earth aerosol, storm swirl, and wildfire fallback map."""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :] = (24, 18, 16)
        cv2.rectangle(img, (540, 110), (880, 430), (155, 70, 18), -1)

        center = (690, 265)
        for angle_deg in range(0, 1080, 6):
            rad = np.deg2rad(angle_deg)
            radius = 8.0 + angle_deg * 0.11
            px = int(center[0] + radius * np.cos(rad))
            py = int(center[1] + radius * np.sin(rad) * 0.8)
            color = (245, 248, 252) if angle_deg % 12 == 0 else (225, 190, 120)
            cv2.circle(img, (px, py), 4, color, 2)

        cv2.ellipse(img, (265, 285), (175, 130), -12, 0, 360, (38, 135, 42), -1)
        rng = np.random.default_rng(9090)
        for _ in range(280):
            fx = int(rng.integers(145, 360))
            fy = int(rng.integers(180, 360))
            radius = int(rng.integers(2, 8))
            color = (20, 30, 245) if rng.random() > 0.38 else (15, 135, 250)
            cv2.circle(img, (fx, fy), radius, color, -1)
            cv2.circle(img, (fx + 3, fy - 2), max(1, radius - 1), (40, 220, 255), 1)
        return img


# =============================================================================
# MODULE 2: COMPUTER VISION ANALYSIS
# =============================================================================
class CVAnalyzer:
    """Extracts dominant color, intensity, and edge density from a sensor patch."""

    COLOR_CATEGORIES = ("Black", "White", "Red", "Orange", "Green", "Blue")
    CHROMATIC_CATEGORIES = ("Red", "Orange", "Green", "Blue")
    DENSITY_LABELS = (
        (0.30, "Low density"),
        (0.70, "Medium density"),
        (1.01, "High density"),
    )
    ZONE_INTERPRETATIONS = {
        ("Red", "High density"): "likely active wildfire hotspot or intense thermal concentration",
        ("Red", "Medium density"): "likely warm thermal boundary or elevated heat plume",
        ("Red", "Low density"): "likely uniform high-temperature or extreme risk zone",
        ("Orange", "High density"): "likely turbulent aerosol plume or fragmented heat front",
        ("Orange", "Medium density"): "likely warm continental margin or moderate anomaly",
        ("Orange", "Low density"): "likely calm warm water, elevated CO2, or arid plateau",
        ("Green", "High density"): "likely complex terrain, mountain ridge, or sea-salt stream",
        ("Green", "Medium density"): "likely coastline or vegetated landmass",
        ("Green", "Low density"): "likely uniform plains or very low fire-risk basin",
        ("Blue", "High density"): "likely churning ocean current or storm front over water",
        ("Blue", "Medium density"): "likely continental shelf or coastal current",
        ("Blue", "Low density"): "likely calm open ocean or clean air zone",
        ("White", "High density"): "likely turbulent storm vortex or fractured sea ice",
        ("White", "Medium density"): "likely broken cloud band or snow-capped ridge",
        ("White", "Low density"): "likely solid polar ice sheet or smooth cloud deck",
        ("Black", "High density"): "likely sharp shadow boundary or night-side coastline",
        ("Black", "Medium density"): "likely dark bathymetric trench or dusk terminator",
        ("Black", "Low density"): "likely deep night-side ocean or unlit background",
    }

    def __init__(
        self,
        edge_ratio_ceiling: float = 0.22,
        chromatic_salience_ratio: float = 0.18,
    ) -> None:
        self.edge_ratio_ceiling = max(0.05, float(edge_ratio_ceiling))
        self.chromatic_salience_ratio = float(np.clip(chromatic_salience_ratio, 0.05, 0.50))

    def analyze_patch(
        self,
        roi: np.ndarray,
        bbox: tuple[int, int, int, int],
        cursor_xy: tuple[int, int],
        touching: bool = True,
    ) -> AnalysisResult:
        """Compute Dominant Color, Intensity (0..1), and Canny Density (0..1)."""
        if roi is None or roi.size == 0:
            return AnalysisResult(
                dominant_color="Black",
                intensity=0.0,
                density=0.0,
                sensor_bbox=bbox,
                cursor_xy=cursor_xy,
                touching=touching,
            )

        bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR) if roi.ndim == 2 else roi
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h_chan = hsv[:, :, 0]
        s_chan = hsv[:, :, 1]
        v_chan = hsv[:, :, 2]

        dominant_color = self._classify_dominant_color(h_chan, s_chan, v_chan)

        mean_v = float(np.mean(v_chan)) / 255.0
        mean_s = float(np.mean(s_chan)) / 255.0
        intensity = float(np.clip(0.65 * mean_v + 0.35 * mean_s, 0.0, 1.0))

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)
        edge_ratio = float(np.count_nonzero(edges)) / float(edges.size)
        density = float(np.clip(edge_ratio / self.edge_ratio_ceiling, 0.0, 1.0))

        return AnalysisResult(
            dominant_color=dominant_color,
            intensity=intensity,
            density=density,
            sensor_bbox=bbox,
            cursor_xy=cursor_xy,
            touching=touching,
        )

    def _classify_dominant_color(
        self,
        h_chan: np.ndarray,
        s_chan: np.ndarray,
        v_chan: np.ndarray,
    ) -> str:
        """Partition HSV pixels into canonical categories with EIC color sensitivity.

        * High-saturation deep crimson (e.g. NASA FWI 'Extreme' dark maroon) is
          captured as ``Red`` even when luminance is low.
        * Purple/violet mineral dust plumes (``130 <= H < 145``, ``S >= 50``)
          map to ``Orange`` (elevated warm aerosol alert).
        * When at least ``chromatic_salience_ratio`` (default 18%) of the patch
          contains chromatic signal, glowing filaments over dark ocean/land
          backgrounds (e.g. ECCO2 currents, GEOS aerosols) take precedence.
        """
        deep_crimson = (v_chan >= 16) & (s_chan >= 100) & ((h_chan < 12) | (h_chan >= 145))
        black_mask = (~deep_crimson) & ((v_chan < 45) | ((s_chan < 35) & (v_chan < 100)))
        white_mask = (~deep_crimson) & (~black_mask) & (s_chan < 35) & (v_chan >= 100)
        chromatic = (~black_mask) & (~white_mask)

        violet_dust = (h_chan >= 130) & (h_chan < 145) & (s_chan >= 50) & (v_chan >= 60)
        red_mask = deep_crimson | (chromatic & ((h_chan < 10) | (h_chan >= 155)))
        orange_mask = (~red_mask) & chromatic & (((h_chan >= 10) & (h_chan < 33)) | violet_dust)
        green_mask = (~red_mask) & (~orange_mask) & chromatic & (h_chan >= 33) & (h_chan < 88)
        blue_mask = (~red_mask) & (~orange_mask) & (~green_mask) & chromatic

        counts = {
            "Black": int(np.count_nonzero(black_mask)),
            "White": int(np.count_nonzero(white_mask)),
            "Red": int(np.count_nonzero(red_mask)),
            "Orange": int(np.count_nonzero(orange_mask)),
            "Green": int(np.count_nonzero(green_mask)),
            "Blue": int(np.count_nonzero(blue_mask)),
        }

        total_pixels = int(h_chan.size)
        chromatic_total = sum(counts[name] for name in self.CHROMATIC_CATEGORIES)
        if total_pixels > 0 and (chromatic_total / total_pixels) >= self.chromatic_salience_ratio:
            return max(self.CHROMATIC_CATEGORIES, key=lambda name: counts[name])
        return max(self.COLOR_CATEGORIES, key=lambda name: counts[name])

    def density_tier(self, density: float) -> str:
        """Return a spoken/human-readable density band."""
        clamped = float(np.clip(density, 0.0, 1.0))
        for upper_bound, label in self.DENSITY_LABELS:
            if clamped < upper_bound:
                return label
        return "High density"

    def format_speech_summary(
        self,
        result: AnalysisResult,
        dataset: Optional[DatasetEntry] = None,
    ) -> str:
        """Synthesize an accessible spoken description for the current sensor box."""
        tier = self.density_tier(result.density)
        color = result.dominant_color
        intensity_pct = int(round(result.intensity * 100))

        dataset_meaning = dataset.meaning_for(color) if dataset is not None else None
        interpretation = dataset_meaning or self.ZONE_INTERPRETATIONS.get(
            (color, tier),
            "distinct Earth surface feature",
        )
        return (
            f"{tier} {color.lower()} zone, {intensity_pct} percent intensity, "
            f"{interpretation}."
        )


# =============================================================================
# MODULE 3: BEAT-SYNCHRONIZED C-MINOR AUDIO ENGINE (90 BPM MASTER CLOCK)
# =============================================================================
class AudioBackend(ABC):
    """Abstract MIDI/note synthesizer backend."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Name of the active audio output driver."""

    @abstractmethod
    def start(self, channel_programs: dict[int, int]) -> None:
        """Open audio output and configure initial MIDI channel instruments."""

    @abstractmethod
    def note_on(self, channel: int, midi_note: int, velocity: int) -> None:
        """Trigger a quantized MIDI note."""

    @abstractmethod
    def note_off(self, channel: int, midi_note: int) -> None:
        """Release a quantized MIDI note."""

    def control_change(self, channel: int, control: int, value: int) -> None:
        """Send an optional MIDI CC message (Reverb, Chorus, Pan, etc.)."""

    @abstractmethod
    def close(self) -> None:
        """Silence active voices and release hardware handles."""


class NullAudioBackend(AudioBackend):
    """Silent backend used in headless environments or automated self-tests."""

    @property
    def backend_name(self) -> str:
        return "Null Audio (Headless)"

    def start(self, channel_programs: dict[int, int]) -> None:
        pass

    def note_on(self, channel: int, midi_note: int, velocity: int) -> None:
        pass

    def note_off(self, channel: int, midi_note: int) -> None:
        pass

    def close(self) -> None:
        pass


class PygameMidiBackend(AudioBackend):
    """Cross-platform MIDI playback via ``pygame.midi`` when installed."""

    def __init__(self) -> None:
        self._out = None

    @property
    def backend_name(self) -> str:
        return "pygame.midi"

    def start(self, channel_programs: dict[int, int]) -> None:
        if pygame is None:
            raise RuntimeError("pygame is not installed")
        pygame.midi.init()
        device_id = pygame.midi.get_default_output_id()
        if device_id == -1:
            pygame.midi.quit()
            raise RuntimeError("No default pygame.midi output device available")
        self._out = pygame.midi.Output(device_id)
        for channel, program in channel_programs.items():
            if channel != 9:
                self._out.set_instrument(program, channel=channel)

    def note_on(self, channel: int, midi_note: int, velocity: int) -> None:
        if self._out is not None:
            self._out.note_on(midi_note, velocity, channel=channel)

    def note_off(self, channel: int, midi_note: int) -> None:
        if self._out is not None:
            self._out.note_off(midi_note, 0, channel=channel)

    def control_change(self, channel: int, control: int, value: int) -> None:
        if self._out is not None:
            self._out.write_short(0xB0 | (channel & 0x0F), control & 0x7F, value & 0x7F)

    def close(self) -> None:
        if self._out is not None:
            try:
                for ch in range(16):
                    self.control_change(ch, 123, 0)
                self._out.close()
            finally:
                self._out = None
                pygame.midi.quit()


class WinmmMidiBackend(AudioBackend):
    """Native Windows General MIDI output via ``winmm.dll`` (zero pip dependencies)."""

    def __init__(self) -> None:
        self._winmm = ctypes.windll.winmm
        self._handle = ctypes.c_void_p()
        self._opened = False
        self._configure_ctypes()

    @property
    def backend_name(self) -> str:
        return "Windows winmm MIDI Synth"

    def _configure_ctypes(self) -> None:
        self._winmm.midiOutGetNumDevs.restype = wintypes.UINT
        self._winmm.midiOutOpen.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            wintypes.UINT,
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        self._winmm.midiOutOpen.restype = wintypes.UINT
        self._winmm.midiOutShortMsg.argtypes = [ctypes.c_void_p, wintypes.DWORD]
        self._winmm.midiOutShortMsg.restype = wintypes.UINT
        self._winmm.midiOutClose.argtypes = [ctypes.c_void_p]
        self._winmm.midiOutClose.restype = wintypes.UINT

    def start(self, channel_programs: dict[int, int]) -> None:
        if self._winmm.midiOutGetNumDevs() == 0:
            raise RuntimeError("No Windows MIDI output devices detected")
        result = self._winmm.midiOutOpen(ctypes.byref(self._handle), 0, 0, 0, 0)
        if result != 0:
            raise RuntimeError(f"midiOutOpen failed with error code {result}")
        self._opened = True
        for channel, program in channel_programs.items():
            if channel != 9:
                msg = (0xC0 | (channel & 0x0F)) | ((program & 0x7F) << 8)
                self._winmm.midiOutShortMsg(self._handle, msg)

    def note_on(self, channel: int, midi_note: int, velocity: int) -> None:
        if not self._opened:
            return
        status = 0x90 | (channel & 0x0F)
        msg = status | ((midi_note & 0x7F) << 8) | ((velocity & 0x7F) << 16)
        self._winmm.midiOutShortMsg(self._handle, msg)

    def note_off(self, channel: int, midi_note: int) -> None:
        if not self._opened:
            return
        status = 0x80 | (channel & 0x0F)
        msg = status | ((midi_note & 0x7F) << 8)
        self._winmm.midiOutShortMsg(self._handle, msg)

    def control_change(self, channel: int, control: int, value: int) -> None:
        if not self._opened:
            return
        msg = (0xB0 | (channel & 0x0F)) | ((control & 0x7F) << 8) | ((value & 0x7F) << 16)
        self._winmm.midiOutShortMsg(self._handle, msg)

    def close(self) -> None:
        if self._opened:
            try:
                for channel in range(16):
                    self.control_change(channel, 123, 0)
                self._winmm.midiOutClose(self._handle)
            finally:
                self._opened = False
                self._handle = ctypes.c_void_p()


def create_best_audio_backend(enable_audio: bool = True) -> AudioBackend:
    """Select the highest-fidelity available MIDI backend on the current OS."""
    if not enable_audio:
        return NullAudioBackend()

    candidates: list[type[AudioBackend]] = []
    if pygame is not None:
        candidates.append(PygameMidiBackend)
    if platform.system() == "Windows":
        candidates.append(WinmmMidiBackend)

    for backend_cls in candidates:
        backend = backend_cls()
        try:
            backend.start({})
            backend.close()
            return backend_cls()
        except Exception:
            continue
    return NullAudioBackend()


class CMinorAudioEngine:
    """Locked 90 BPM 16th-note sequencer combining C-minor pads, ambient drums, and data melody."""

    BPM: float = 90.0
    BEAT_DURATION_SEC: float = 60.0 / BPM            # 0.6667s per quarter beat
    SIXTEENTH_DURATION_SEC: float = BEAT_DURATION_SEC / 4.0  # 0.1667s per 16th step

    # Strict C Natural Minor pitch classes: C, D, Eb, F, G, Ab, Bb
    C_MINOR_PITCH_CLASSES: frozenset[int] = frozenset({0, 2, 3, 5, 7, 8, 10})
    NOTE_NAMES: tuple[str, ...] = (
        "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"
    )

    # Color -> strictly quantized C-Minor MIDI pitch palette
    COLOR_PITCH_MAP: dict[str, tuple[int, ...]] = {
        "Black": (36, 43, 48, 51),          # C2, G2, C3, Eb3
        "Blue": (48, 51, 55, 60),           # C3, Eb3, G3, C4
        "Green": (48, 53, 55, 58, 63),      # C3, F3, G3, Bb3, Eb4
        "White": (60, 67, 70, 72, 75),      # C4, G4, Bb4, C5, Eb5
        "Orange": (62, 63, 68, 70, 74),     # D4, Eb4, Ab4, Bb4, D5
        "Red": (74, 75, 79, 80, 82, 84),    # D5, Eb5, G5, Ab5, Bb5, C6
    }

    # Color -> (MIDI Channel, General MIDI Program)
    COLOR_CHANNEL_PROGRAM: dict[str, tuple[int, int]] = {
        "Black": (1, 89),   # Warm Pad
        "Blue": (2, 88),    # New Age Pad
        "Green": (3, 12),   # Marimba
        "White": (4, 11),   # Vibraphone
        "Orange": (5, 4),   # Electric Piano 1
        "Red": (7, 81),     # Lead 2 (Sawtooth)
    }

    # 4-Bar Ambient C-Minor Progression (all tones strictly in C Natural Minor)
    # Bar 0: Cm9 | Bar 1: Abmaj7 | Bar 2: Fm7 | Bar 3: Gm7
    HARMONIC_PROGRESSION: tuple[tuple[str, int, tuple[int, ...]], ...] = (
        ("Cm9", 36, (48, 51, 55, 62)),
        ("Abmaj7", 44, (44, 48, 51, 55)),
        ("Fm7", 41, (41, 48, 51, 56)),
        ("Gm7", 43, (43, 50, 53, 58)),
    )
    PAD_CHANNEL = 0
    BASS_CHANNEL = 6
    DRUM_CHANNEL = 9

    # General MIDI Percussion Note Numbers (Channel 9)
    DRUM_KICK = 36
    DRUM_RIM = 37
    DRUM_HAT_CLOSED = 42
    DRUM_LOW_TOM = 45
    DRUM_HAT_OPEN = 46
    DRUM_CONGA_HI = 63
    DRUM_SHAKER = 70

    def __init__(
        self,
        backend: Optional[AudioBackend] = None,
        enable_audio: bool = True,
        ambient_drums: bool = True,
    ) -> None:
        self.backend = backend or create_best_audio_backend(enable_audio=enable_audio)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._ambient_drums_enabled = ambient_drums
        self._latest_analysis = AnalysisResult(
            dominant_color="Blue",
            intensity=0.0,
            density=0.0,
            sensor_bbox=(0, 0, 0, 0),
            cursor_xy=(0, 0),
            touching=False,
        )
        self._active_data_notes: list[tuple[int, int, int]] = []  # (ch, note, off_step)
        self._active_pad_notes: list[tuple[int, int]] = []
        self._active_drum_notes: list[int] = []

        self._data_step_index = 0
        self._beat_info = (0, 0)  # (bar_0..3, beat_0..3)
        self._current_telemetry = "90 BPM | Standby"

    @property
    def ambient_drums_enabled(self) -> bool:
        with self._lock:
            return self._ambient_drums_enabled

    def toggle_drums(self) -> bool:
        """Toggle the Channel 9 ambient drum & percussion groove."""
        with self._lock:
            self._ambient_drums_enabled = not self._ambient_drums_enabled
            return self._ambient_drums_enabled

    def current_beat_info(self) -> tuple[int, int]:
        """Return ``(bar_index 0..3, beat_in_bar 0..3)`` for live UI beat indicator."""
        with self._lock:
            return self._beat_info

    @classmethod
    def snap_to_c_minor(cls, midi_note: int) -> int:
        """Quantize any MIDI note strictly onto the nearest C Natural Minor tone."""
        clamped = max(0, min(int(midi_note), 127))
        if (clamped % 12) in cls.C_MINOR_PITCH_CLASSES:
            return clamped
        for offset in range(1, 12):
            down = clamped - offset
            if down >= 0 and (down % 12) in cls.C_MINOR_PITCH_CLASSES:
                return down
            up = clamped + offset
            if up <= 127 and (up % 12) in cls.C_MINOR_PITCH_CLASSES:
                return up
        return 60

    @classmethod
    def midi_to_note_name(cls, midi_note: int) -> str:
        """Convert a MIDI pitch integer into note + octave notation (e.g. ``Eb4``)."""
        octave = (midi_note // 12) - 1
        return f"{cls.NOTE_NAMES[midi_note % 12]}{octave}"

    @staticmethod
    def midi_to_freq(midi_note: int) -> float:
        """Convert a quantized MIDI pitch to frequency in Hz (A4 = 440 Hz)."""
        return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

    @classmethod
    def select_rhythm(cls, density: float) -> tuple[str, float]:
        """Map visual edge density ``[0.0, 1.0]`` to rhythmic subdivision in beats."""
        d = float(np.clip(density, 0.0, 1.0))
        if d < 0.12:
            return "Whole Note", 4.0
        if d < 0.30:
            return "Half Note", 2.0
        if d <= 0.70:
            return "8th Note", 0.5
        return "16th Arpeggio", 0.25

    @staticmethod
    def should_trigger_data_step(density: float, step_16: int) -> bool:
        """Return True when the data melody falls cleanly on the 90 BPM beat grid."""
        step = step_16 % 16
        d = float(np.clip(density, 0.0, 1.0))
        if d < 0.12:
            return step == 0
        if d < 0.30:
            return step in (0, 8)
        if d <= 0.70:
            return step % 2 == 0
        return True

    @staticmethod
    def intensity_to_velocity(intensity: float) -> int:
        """Map brightness/saturation intensity ``[0.0, 1.0]`` to MIDI velocity."""
        clamped = float(np.clip(intensity, 0.0, 1.0))
        return int(round(30 + clamped * 97))

    @staticmethod
    def intensity_to_volume(intensity: float) -> float:
        """Map intensity ``[0.0, 1.0]`` to normalized scalar amplitude."""
        clamped = float(np.clip(intensity, 0.0, 1.0))
        return 0.15 + clamped * 0.80

    @classmethod
    def plan_musical_step(
        cls,
        analysis: AnalysisResult,
        step_index: int,
    ) -> tuple[int, int, int, float, str, float]:
        """Determine channel, quantized pitch, velocity, volume, motif, and duration."""
        palette = cls.COLOR_PITCH_MAP.get(analysis.dominant_color, cls.COLOR_PITCH_MAP["Blue"])
        motif_name, beats = cls.select_rhythm(analysis.density)
        step_duration_sec = beats * cls.BEAT_DURATION_SEC

        if analysis.density < 0.30:
            raw_pitch = palette[(step_index // 2) % min(2, len(palette))]
        elif analysis.density <= 0.70:
            melodic_walk = (0, 1, 2, 1, 3, 2, 1, 0)
            raw_pitch = palette[melodic_walk[step_index % len(melodic_walk)] % len(palette)]
        else:
            arp_pattern = (0, 2, 1, 3, 2, 4, 3, 5)
            idx = arp_pattern[step_index % len(arp_pattern)] % len(palette)
            raw_pitch = palette[idx]

        quantized_pitch = cls.snap_to_c_minor(raw_pitch)
        channel, _ = cls.COLOR_CHANNEL_PROGRAM.get(analysis.dominant_color, (2, 88))
        velocity = cls.intensity_to_velocity(analysis.intensity)
        volume = cls.intensity_to_volume(analysis.intensity)
        return channel, quantized_pitch, velocity, volume, motif_name, step_duration_sec

    @classmethod
    def plan_ambient_drums(cls, density: float, intensity: float, step_16: int) -> list[tuple[int, int]]:
        """Generate an atmospheric 90 BPM percussion groove reactive to CV density."""
        step = step_16 % 16
        d = float(np.clip(density, 0.0, 1.0))
        i = float(np.clip(intensity, 0.0, 1.0))
        hits: list[tuple[int, int]] = []

        # Deep heartbeat kick on Beat 1, plus Beat 3 / syncopation as density rises
        if step == 0:
            hits.append((cls.DRUM_KICK, int(round(52 + 34 * i))))
        elif step == 10 and d >= 0.25:
            hits.append((cls.DRUM_KICK, int(round(40 + 28 * i))))
        elif step in (6, 14) and d >= 0.75:
            hits.append((cls.DRUM_KICK, int(round(36 + 22 * i))))

        # Warm ambient side-stick backbeat on Beats 2 and 4
        if step in (4, 12):
            hits.append((cls.DRUM_RIM, int(round(34 + 36 * i))))

        # Breathing shaker & hi-hat subdivision scaled by visual chaos
        if d < 0.30:
            if step in (0, 4, 8, 12):
                hits.append((cls.DRUM_SHAKER, int(round(26 + 22 * i))))
        elif d <= 0.70:
            if step == 14:
                hits.append((cls.DRUM_HAT_OPEN, int(round(30 + 24 * i))))
            elif step % 2 == 0:
                hits.append((cls.DRUM_HAT_CLOSED, int(round(28 + 26 * i))))
        else:
            if step == 14:
                hits.append((cls.DRUM_HAT_OPEN, int(round(38 + 30 * i))))
            elif step % 2 == 0:
                hits.append((cls.DRUM_HAT_CLOSED, int(round(34 + 30 * i))))
            else:
                hits.append((cls.DRUM_SHAKER, int(round(26 + 24 * i))))
            if step in (6, 15):
                hits.append((cls.DRUM_CONGA_HI, int(round(38 + 32 * i))))

        return hits

    def start(self) -> None:
        """Initialize the audio backend, configure spatial reverb, and start clock thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()

        channel_programs = {
            self.PAD_CHANNEL: 89,   # Warm Pad
            self.BASS_CHANNEL: 33,  # Electric Finger Bass
        }
        for channel, program in self.COLOR_CHANNEL_PROGRAM.values():
            channel_programs[channel] = program

        self.backend.start(channel_programs)
        # Apply atmospheric reverb (CC 91) and subtle stereo chorus (CC 93)
        for ch in set(channel_programs) | {self.DRUM_CHANNEL}:
            reverb = 82 if ch == self.PAD_CHANNEL else (45 if ch == self.DRUM_CHANNEL else 68)
            self.backend.control_change(ch, 91, reverb)
            if ch != self.DRUM_CHANNEL:
                self.backend.control_change(ch, 93, 35)

        self._thread = threading.Thread(
            target=self._audio_loop,
            name="CMinorAudioEngine",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the sequencer thread, silence all voices, and close the backend."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._silence_all_active()
        self.backend.close()

    def update_analysis(self, result: AnalysisResult) -> None:
        """Update the target CV metrics consumed on the 90 BPM beat grid."""
        with self._lock:
            self._latest_analysis = result

    def telemetry_text(self) -> str:
        """Return a thread-safe snapshot of the current musical status."""
        with self._lock:
            return self._current_telemetry

    def _silence_data_notes(self) -> None:
        for ch, note, _off_step in self._active_data_notes:
            self.backend.note_off(ch, note)
        self._active_data_notes.clear()

    def _silence_pad_notes(self) -> None:
        for ch, note in self._active_pad_notes:
            self.backend.note_off(ch, note)
        self._active_pad_notes.clear()

    def _silence_drum_notes(self) -> None:
        for drum_note in self._active_drum_notes:
            self.backend.note_off(self.DRUM_CHANNEL, drum_note)
        self._active_drum_notes.clear()

    def _silence_all_active(self) -> None:
        self._silence_data_notes()
        self._silence_pad_notes()
        self._silence_drum_notes()

    def _release_expired_data_notes(self, global_step: int) -> None:
        still_active: list[tuple[int, int, int]] = []
        for ch, note, off_step in self._active_data_notes:
            if global_step >= off_step:
                self.backend.note_off(ch, note)
            else:
                still_active.append((ch, note, off_step))
        self._active_data_notes = still_active

    def _audio_loop(self) -> None:
        clock_origin = time.monotonic()
        global_step = 0
        was_touching = False
        last_rhythm_tier = ""
        last_color = ""
        pending_entry_trigger = False

        while not self._stop_event.is_set():
            target_time = clock_origin + global_step * self.SIXTEENTH_DURATION_SEC
            now = time.monotonic()
            wait_sec = target_time - now
            if wait_sec > 0.002:
                self._stop_event.wait(timeout=wait_sec)
                if self._stop_event.is_set():
                    break
            elif wait_sec < -self.BEAT_DURATION_SEC:
                clock_origin = now - global_step * self.SIXTEENTH_DURATION_SEC

            step_16 = global_step % 16
            bar_index = (global_step // 16) % len(self.HARMONIC_PROGRESSION)
            beat_in_bar = step_16 // 4

            with self._lock:
                analysis = self._latest_analysis
                drums_on = self._ambient_drums_enabled
                self._beat_info = (bar_index, beat_in_bar)

            # Release previous 16th-step percussion gates cleanly
            self._silence_drum_notes()
            self._release_expired_data_notes(global_step)

            if not analysis.touching:
                if was_touching:
                    self._silence_all_active()
                    was_touching = False
                    pending_entry_trigger = False
                with self._lock:
                    self._current_telemetry = f"{self.BPM:.0f} BPM | Standby (hover map or press P)"
                global_step += 1
                continue

            # 1. C-Minor Harmonic Pad & Bass Bed (changes cleanly on Bar downbeats)
            chord_name, bass_note, chord_tones = self.HARMONIC_PROGRESSION[bar_index]
            if not was_touching or step_16 == 0:
                self._silence_pad_notes()
                pad_vel = int(round(30 + analysis.intensity * 26))
                for tone in chord_tones:
                    q_tone = self.snap_to_c_minor(tone)
                    self.backend.note_on(self.PAD_CHANNEL, q_tone, pad_vel)
                    self._active_pad_notes.append((self.PAD_CHANNEL, q_tone))

                bass_vel = int(round(44 + analysis.intensity * 32))
                q_bass = self.snap_to_c_minor(bass_note)
                self.backend.note_on(self.BASS_CHANNEL, q_bass, bass_vel)
                self._active_pad_notes.append((self.BASS_CHANNEL, q_bass))

            # Gentle bass octave pulse on Beat 3 for steady ambient momentum
            if step_16 == 8 and analysis.density >= 0.25:
                q_bass_oct = self.snap_to_c_minor(bass_note + 12)
                self.backend.note_on(self.BASS_CHANNEL, q_bass_oct, int(round(34 + analysis.intensity * 24)))
                self._active_pad_notes.append((self.BASS_CHANNEL, q_bass_oct))

            # 2. Ambient Drum Groove on Channel 9
            if drums_on:
                for drum_note, drum_vel in self.plan_ambient_drums(
                    analysis.density, analysis.intensity, step_16
                ):
                    self.backend.note_on(self.DRUM_CHANNEL, drum_note, drum_vel)
                    self._active_drum_notes.append(drum_note)

            # 3. Beat-Synchronized Data Sonification Voice
            rhythm_name, beats = self.select_rhythm(analysis.density)
            if not was_touching or rhythm_name != last_rhythm_tier or analysis.dominant_color != last_color:
                pending_entry_trigger = True
                last_rhythm_tier = rhythm_name
                last_color = analysis.dominant_color
            was_touching = True

            grid_hit = self.should_trigger_data_step(analysis.density, step_16)
            # When moving into a new zone, align the entry to the nearest quarter beat
            on_beat_entry = pending_entry_trigger and (step_16 % 4 == 0)

            if grid_hit or on_beat_entry:
                pending_entry_trigger = False
                self._silence_data_notes()
                (
                    channel,
                    pitch,
                    velocity,
                    _volume,
                    motif_name,
                    _dur_sec,
                ) = self.plan_musical_step(analysis, self._data_step_index)
                self._data_step_index += 1

                # Slight musical accent on downbeats
                if step_16 % 4 == 0:
                    velocity = min(127, velocity + 8)

                step_span = max(1, int(round(beats * 4)))
                off_step = global_step + max(1, step_span - (1 if step_span > 1 else 0))

                self.backend.note_on(channel, pitch, velocity)
                self._active_data_notes.append((channel, pitch, off_step))

                # Add a C-minor harmonic dyad on quarter beats for richer timbre
                if step_16 % 4 == 0:
                    harmony_pitch = self.snap_to_c_minor(pitch - 5 if pitch >= 65 else pitch + 7)
                    if harmony_pitch != pitch:
                        self.backend.note_on(channel, harmony_pitch, max(24, int(velocity * 0.65)))
                        self._active_data_notes.append((channel, harmony_pitch, off_step))

                note_label = self.midi_to_note_name(pitch)
                drum_tag = "Drums ON" if drums_on else "Drums OFF"
                with self._lock:
                    self._current_telemetry = (
                        f"{self.BPM:.0f} BPM | {chord_name} | {note_label} ({motif_name}) | "
                        f"Vel {velocity} | {drum_tag}"
                    )

            global_step += 1


# =============================================================================
# MODULE 4: ACCESSIBILITY & THREADED TEXT-TO-SPEECH
# =============================================================================
class AccessibilityTTS:
    """Non-blocking, interruptible TTS worker using ``pyttsx3`` with OS fallbacks."""

    AUDIO_LEGEND_TEXT = (
        "Welcome to the Earth Jukebox. A 90 beat per minute C minor soundscape translates "
        "NASA Earth visualizations into sound. Low tones indicate cool zones, high rapid "
        "arpeggios indicate dense or intense regions, and ambient percussion follows spatial "
        "complexity. Press R to read the current zone, I for dataset information, P for auto-scan "
        "jukebox mode, D to toggle drums, and Tab to switch datasets."
    )

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._queue: queue.Queue[Optional[str]] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc_lock = threading.Lock()
        self._active_proc: Optional[subprocess.Popen[bytes]] = None
        self._pyttsx3_engine = None
        self._backend_name = "Disabled"

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def start(self) -> None:
        """Start the dedicated speech worker thread."""
        if not self.enabled:
            self._backend_name = "Disabled"
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._tts_loop,
            name="AccessibilityTTS",
            daemon=True,
        )
        self._thread.start()

    def speak(self, text: str, interrupt: bool = True) -> None:
        """Queue an utterance asynchronously without blocking UI or audio."""
        cleaned = text.strip()
        if not cleaned or not self.enabled:
            return
        if interrupt:
            self._drain_pending()
            self._interrupt_active()
        self._queue.put(cleaned)

    def stop(self) -> None:
        """Stop the speech thread and terminate any active utterance."""
        self._stop_event.set()
        self._drain_pending()
        self._interrupt_active()
        self._queue.put(None)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _drain_pending(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _interrupt_active(self) -> None:
        if self._pyttsx3_engine is not None:
            try:
                self._pyttsx3_engine.stop()
            except Exception:
                pass
        with self._proc_lock:
            if self._active_proc is not None and self._active_proc.poll() is None:
                try:
                    self._active_proc.terminate()
                except Exception:
                    pass
                self._active_proc = None

    def _init_pyttsx3_on_worker(self) -> Optional[object]:
        if pyttsx3 is None:
            return None
        try:
            if pythoncom is not None:
                pythoncom.CoInitialize()
            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            voices = engine.getProperty("voices")
            if not voices:
                return None
            self._backend_name = "pyttsx3"
            return engine
        except Exception:
            return None

    def _tts_loop(self) -> None:
        self._pyttsx3_engine = self._init_pyttsx3_on_worker()
        if self._pyttsx3_engine is None:
            if platform.system() == "Windows":
                self._backend_name = "Windows System.Speech"
            elif shutil.which("spd-say") or shutil.which("espeak"):
                self._backend_name = "Linux speech-dispatcher/espeak"
            else:
                self._backend_name = "Console Fallback"

        try:
            while not self._stop_event.is_set():
                try:
                    text = self._queue.get(timeout=0.25)
                except queue.Empty:
                    continue
                if text is None or self._stop_event.is_set():
                    break
                self._speak_blocking(text)
        finally:
            if self._pyttsx3_engine is not None:
                try:
                    self._pyttsx3_engine.stop()
                except Exception:
                    pass
                self._pyttsx3_engine = None
            if pythoncom is not None:
                try:
                    pythoncom.CoUninitialize()
                except Exception:
                    pass

    def _speak_blocking(self, text: str) -> None:
        if self._pyttsx3_engine is not None:
            try:
                self._pyttsx3_engine.say(text)
                self._pyttsx3_engine.runAndWait()
                return
            except Exception:
                self._pyttsx3_engine = None

        if platform.system() == "Windows":
            safe_text = text.replace("'", "''")
            ps_script = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "$s.Rate = 1; "
                f"$s.Speak('{safe_text}')"
            )
            self._run_tts_subprocess(["powershell", "-NoProfile", "-Command", ps_script])
            return

        if shutil.which("spd-say"):
            self._run_tts_subprocess(["spd-say", "--wait", text])
            return
        if shutil.which("espeak"):
            self._run_tts_subprocess(["espeak", text])
            return

        print(f"[TTS] {text}")

    def _run_tts_subprocess(self, cmd: list[str]) -> None:
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creation_flags,
            )
            with self._proc_lock:
                self._active_proc = proc
            proc.wait()
        except Exception:
            pass
        finally:
            with self._proc_lock:
                self._active_proc = None


# =============================================================================
# APPLICATION CONTROLLER & TKINTER ACCESSIBLE VIEW
# =============================================================================
class EarthJukeboxApp:
    """Coordinates Input Proxy -> CV Analyzer -> C-Minor Audio Engine -> UI/TTS."""

    CANVAS_WIDTH = 960
    CANVAS_HEIGHT = 540
    WINDOW_WIDTH = 992
    WINDOW_HEIGHT = 750
    SENSOR_BOX_SIZE = 40

    COLOR_SWATCH_HEX = {
        "Red": "#ef4444",
        "Orange": "#f97316",
        "Green": "#22c55e",
        "Blue": "#38bdf8",
        "Black": "#334155",
        "White": "#f8fafc",
    }

    def __init__(
        self,
        root: "tk.Tk",
        dataset_manager: DatasetManager,
        cv_analyzer: CVAnalyzer,
        audio_engine: CMinorAudioEngine,
        tts: AccessibilityTTS,
        input_mode: str = "mouse",
        evdev_device: Optional[str] = None,
    ) -> None:
        self.root = root
        self.dataset_manager = dataset_manager
        self.cv_analyzer = cv_analyzer
        self.audio_engine = audio_engine
        self.tts = tts

        self.stop_event = threading.Event()
        self.points_queue: queue.Queue[TouchPoint] = queue.Queue(maxsize=2)
        self._photo_image: Optional["ImageTk.PhotoImage"] = None
        self._current_image_bgr: Optional[np.ndarray] = None
        self._auto_scan_enabled = False
        self._auto_scan_phase = 0.0
        self._last_scan_tick = time.monotonic()

        self._latest_analysis = AnalysisResult(
            dominant_color="Blue",
            intensity=0.0,
            density=0.0,
            sensor_bbox=(0, 0, 0, 0),
            cursor_xy=(0, 0),
            touching=False,
        )

        self._build_ui()
        self._bind_keyboard_shortcuts()

        # === [LINUX EVDEV INTEGRATION POINT] ===
        # Switch between Windows mouse hover and Linux evdev trackpad here.
        if input_mode == "evdev":
            self.input_source: CoordinateInputSource = EvdevTrackpadInputProxy(
                points_queue=self.points_queue,
                stop_event=self.stop_event,
                device_path=evdev_device,
            )
        else:
            self.input_source = WindowsMouseInputProxy(
                canvas=self.canvas,
                points_queue=self.points_queue,
                stop_event=self.stop_event,
                on_user_activity=self._yield_auto_scan_to_user,
            )

        self.dataset_manager.load_datasets()
        self._display_current_dataset(announce=False)

        self.tts.start()
        self.audio_engine.start()
        self.input_source.start()

        self.source_label.configure(
            text=f"Input: {self.input_source.source_name}  |  Audio: {self.audio_engine.backend.backend_name}"
        )
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(16, self._poll_input_and_update)

    def _build_ui(self) -> None:
        self.root.title("Earth Jukebox - NASA Earth Information Center Sonifier")
        self.root.configure(background="#0b1120")
        self.root.geometry(f"{self.WINDOW_WIDTH}x{self.WINDOW_HEIGHT}")
        self.root.minsize(self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
        self.root.maxsize(self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
        self.root.resizable(False, False)

        header = tk.Frame(self.root, bg="#0b1120", width=self.CANVAS_WIDTH, height=32)
        header.pack(fill="x", padx=14, pady=(10, 2))
        header.pack_propagate(False)

        self.dataset_label = tk.Label(
            header,
            text="Dataset: Loading...",
            fg="#f8fafc",
            bg="#0b1120",
            font=("Segoe UI", 12, "bold"),
            anchor="w",
        )
        self.dataset_label.pack(side="left", fill="y")

        self.source_label = tk.Label(
            header,
            text="",
            fg="#94a3b8",
            bg="#0b1120",
            font=("Segoe UI", 9),
            anchor="e",
        )
        self.source_label.pack(side="right", fill="y")

        science_bar = tk.Frame(self.root, bg="#0b1120", width=self.CANVAS_WIDTH, height=44)
        science_bar.pack(fill="x", padx=14, pady=(0, 6))
        science_bar.pack_propagate(False)

        self.science_label = tk.Label(
            science_bar,
            text="",
            fg="#cbd5e1",
            bg="#0b1120",
            font=("Segoe UI", 9),
            anchor="nw",
            justify="left",
            wraplength=self.CANVAS_WIDTH,
        )
        self.science_label.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            self.root,
            width=self.CANVAS_WIDTH,
            height=self.CANVAS_HEIGHT,
            bg="#020617",
            highlightthickness=2,
            highlightbackground="#334155",
            cursor="crosshair",
        )
        self.canvas.pack(padx=14, pady=2)
        self._bg_item = self.canvas.create_image(0, 0, anchor="nw")
        self._box_item = self.canvas.create_rectangle(
            0, 0, 0, 0, outline="#38bdf8", width=2, state="hidden"
        )
        self._dot_item = self.canvas.create_oval(
            0, 0, 0, 0, fill="#facc15", outline="#0f172a", width=1, state="hidden"
        )

        hud = tk.Frame(self.root, bg="#1e293b", padx=12, pady=6, width=self.CANVAS_WIDTH, height=62)
        hud.pack(fill="x", padx=14, pady=(6, 4))
        hud.pack_propagate(False)

        hud_row_top = tk.Frame(hud, bg="#1e293b", height=24)
        hud_row_top.pack(fill="x", expand=True)
        hud_row_top.pack_propagate(False)

        self.color_chip = tk.Label(hud_row_top, text="  ", bg="#38bdf8", width=3)
        self.color_chip.pack(side="left", padx=(0, 8), pady=2)

        self.telemetry_label = tk.Label(
            hud_row_top,
            text="Hover over a NASA map or press [P] for Auto-Scan Jukebox mode.",
            fg="#e2e8f0",
            bg="#1e293b",
            font=("Consolas", 10),
            anchor="w",
        )
        self.telemetry_label.pack(side="left", fill="x", expand=True)

        hud_row_bottom = tk.Frame(hud, bg="#1e293b", height=24)
        hud_row_bottom.pack(fill="x", expand=True)
        hud_row_bottom.pack_propagate(False)

        self.audio_label = tk.Label(
            hud_row_bottom,
            text="90 BPM | Standby",
            fg="#7dd3fc",
            bg="#1e293b",
            font=("Consolas", 10, "bold"),
            anchor="w",
        )
        self.audio_label.pack(side="left", fill="x", expand=True)

        self.beat_label = tk.Label(
            hud_row_bottom,
            text="Beat 1/4  ● ○ ○ ○",
            fg="#38bdf8",
            bg="#1e293b",
            font=("Consolas", 10, "bold"),
            anchor="e",
        )
        self.beat_label.pack(side="right")

        shortcuts_bar = tk.Frame(self.root, bg="#0b1120", width=self.CANVAS_WIDTH, height=28)
        shortcuts_bar.pack(fill="x", padx=14, pady=(0, 8))
        shortcuts_bar.pack_propagate(False)

        shortcuts = tk.Label(
            shortcuts_bar,
            text=(
                "[Space] Audio Legend  |  [R] Read Zone  |  [I] Dataset Info  |  "
                "[P] Auto-Scan  |  [D] Drums  |  [Tab / 1-8] Switch Dataset  |  [Esc] Exit"
            ),
            fg="#cbd5e1",
            bg="#0b1120",
            font=("Segoe UI", 9),
            anchor="center",
        )
        shortcuts.pack(fill="both", expand=True)

    def _bind_keyboard_shortcuts(self) -> None:
        self.root.bind("<space>", self._on_space_legend)
        self.root.bind("<KeyPress-r>", self._on_read_zone)
        self.root.bind("<KeyPress-R>", self._on_read_zone)
        self.root.bind("<KeyPress-i>", self._on_dataset_info)
        self.root.bind("<KeyPress-I>", self._on_dataset_info)
        self.root.bind("<KeyPress-p>", self._on_toggle_auto_scan)
        self.root.bind("<KeyPress-P>", self._on_toggle_auto_scan)
        self.root.bind("<KeyPress-d>", self._on_toggle_drums)
        self.root.bind("<KeyPress-D>", self._on_toggle_drums)
        self.root.bind("<Tab>", self._on_tab_dataset)
        self.root.bind("<Escape>", lambda _event: self.close())
        for num in range(1, 9):
            self.root.bind(
                f"<KeyPress-{num}>",
                lambda _event, idx=num - 1: self._on_select_dataset(idx),
            )

    def _yield_auto_scan_to_user(self) -> None:
        if self._auto_scan_enabled:
            self._auto_scan_enabled = False

    def _on_space_legend(self, _event: Optional["tk.Event"]) -> str:
        self.tts.speak(self.tts.AUDIO_LEGEND_TEXT, interrupt=True)
        return "break"

    def _on_read_zone(self, _event: Optional["tk.Event"]) -> str:
        if not self._latest_analysis.touching:
            self.tts.speak(
                "Cursor is outside the map area. Hover over the visualization or press P for auto-scan.",
                interrupt=True,
            )
        else:
            entry = self.dataset_manager.current_dataset()
            self.tts.speak(
                self.cv_analyzer.format_speech_summary(self._latest_analysis, dataset=entry),
                interrupt=True,
            )
        return "break"

    def _on_dataset_info(self, _event: Optional["tk.Event"]) -> str:
        entry = self.dataset_manager.current_dataset()
        text = entry.summary or f"Viewing {entry.name}."
        self.tts.speak(f"{entry.name}. {text}", interrupt=True)
        return "break"

    def _on_toggle_auto_scan(self, _event: Optional["tk.Event"]) -> str:
        self._auto_scan_enabled = not self._auto_scan_enabled
        self._last_scan_tick = time.monotonic()
        state_msg = "Auto-scan jukebox play started." if self._auto_scan_enabled else "Auto-scan paused."
        self.tts.speak(state_msg, interrupt=True)
        return "break"

    def _on_toggle_drums(self, _event: Optional["tk.Event"]) -> str:
        enabled = self.audio_engine.toggle_drums()
        self.tts.speak("Ambient drums enabled." if enabled else "Ambient drums muted.", interrupt=True)
        return "break"

    def _on_tab_dataset(self, _event: Optional["tk.Event"]) -> str:
        self.dataset_manager.next_dataset()
        self._display_current_dataset(announce=True)
        return "break"

    def _on_select_dataset(self, index: int) -> str:
        self.dataset_manager.select_dataset(index)
        self._display_current_dataset(announce=True)
        return "break"

    def _display_current_dataset(self, announce: bool) -> None:
        entry = self.dataset_manager.current_dataset()
        idx, total = self.dataset_manager.current_index()
        self._current_image_bgr = self.dataset_manager.load_current_image()

        rgb = cv2.cvtColor(self._current_image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).resize(
            (self.CANVAS_WIDTH, self.CANVAS_HEIGHT),
            Image.Resampling.BILINEAR,
        )
        self._photo_image = ImageTk.PhotoImage(pil_img)
        self.canvas.itemconfigure(self._bg_item, image=self._photo_image)

        unit_tag = f"  [{entry.unit}]" if entry.unit else ""
        self.dataset_label.configure(text=f"[{idx}/{total}] {entry.name}{unit_tag}")
        self.science_label.configure(text=f"{entry.category} — {entry.summary}")
        if announce:
            self.tts.speak(f"Switched dataset to {entry.name}.", interrupt=True)

    def _step_auto_scan(self) -> None:
        now = time.monotonic()
        dt = min(0.1, max(0.0, now - self._last_scan_tick))
        self._last_scan_tick = now
        # Sweep gracefully at a musical rate coupled to 90 BPM
        self._auto_scan_phase += dt * (self.audio_engine.BPM / 240.0)
        x_ratio = 0.5 + 0.40 * math.sin(self._auto_scan_phase * 0.85)
        y_ratio = 0.5 + 0.34 * math.sin(self._auto_scan_phase * 1.35)
        cx = int(round(x_ratio * (self.CANVAS_WIDTH - 1)))
        cy = int(round(y_ratio * (self.CANVAS_HEIGHT - 1)))
        push_latest_point(
            self.points_queue,
            TouchPoint(x=cx, y=cy, x_ratio=x_ratio, y_ratio=y_ratio, touching=True),
        )

    def _poll_input_and_update(self) -> None:
        if self._auto_scan_enabled:
            self._step_auto_scan()

        latest_point: Optional[TouchPoint] = None
        while True:
            try:
                latest_point = self.points_queue.get_nowait()
            except queue.Empty:
                break

        if latest_point is not None and self._current_image_bgr is not None:
            self._process_touch_point(latest_point)

        _bar, beat = self.audio_engine.current_beat_info()
        beat_dots = " ".join("●" if i == beat else "○" for i in range(4))
        self.beat_label.configure(text=f"Beat {beat + 1}/4  {beat_dots}")
        self.audio_label.configure(text=self.audio_engine.telemetry_text())

        if not self.stop_event.is_set():
            self.root.after(16, self._poll_input_and_update)

    def _process_touch_point(self, point: TouchPoint) -> None:
        assert self._current_image_bgr is not None
        img_h, img_w = self._current_image_bgr.shape[:2]

        if not point.touching:
            self.canvas.itemconfigure(self._box_item, state="hidden")
            self.canvas.itemconfigure(self._dot_item, state="hidden")
            idle_result = AnalysisResult(
                dominant_color=self._latest_analysis.dominant_color,
                intensity=0.0,
                density=0.0,
                sensor_bbox=self._latest_analysis.sensor_bbox,
                cursor_xy=self._latest_analysis.cursor_xy,
                touching=False,
            )
            self._latest_analysis = idle_result
            self.audio_engine.update_analysis(idle_result)
            self.telemetry_label.configure(text="Pointer outside dataset canvas.")
            return

        img_x = int(round(point.x_ratio * max(0, img_w - 1)))
        img_y = int(round(point.y_ratio * max(0, img_h - 1)))

        patch, bbox = extract_sensor_box(
            self._current_image_bgr,
            img_x,
            img_y,
            box_size=self.SENSOR_BOX_SIZE,
        )
        analysis = self.cv_analyzer.analyze_patch(
            roi=patch,
            bbox=bbox,
            cursor_xy=(img_x, img_y),
            touching=True,
        )
        self._latest_analysis = analysis
        self.audio_engine.update_analysis(analysis)

        scale_x = self.CANVAS_WIDTH / max(1, img_w)
        scale_y = self.CANVAS_HEIGHT / max(1, img_h)
        bx1, by1, bx2, by2 = bbox
        cx = point.x_ratio * self.CANVAS_WIDTH
        cy = point.y_ratio * self.CANVAS_HEIGHT

        swatch_hex = self.COLOR_SWATCH_HEX.get(analysis.dominant_color, "#38bdf8")
        self.canvas.coords(
            self._box_item,
            bx1 * scale_x,
            by1 * scale_y,
            bx2 * scale_x,
            by2 * scale_y,
        )
        self.canvas.itemconfigure(self._box_item, outline=swatch_hex, state="normal")
        self.canvas.coords(self._dot_item, cx - 4, cy - 4, cx + 4, cy + 4)
        self.canvas.itemconfigure(self._dot_item, state="normal")

        self.color_chip.configure(bg=swatch_hex)
        mode_tag = "AUTO-SCAN" if self._auto_scan_enabled else "MANUAL"
        density_tier = self.cv_analyzer.density_tier(analysis.density)
        self.telemetry_label.configure(
            text=(
                f"[{mode_tag}] XY=({img_x:3d},{img_y:3d}) | "
                f"Color={analysis.dominant_color:<6s} | "
                f"Intensity={analysis.intensity:.2f} | "
                f"Density={analysis.density:.2f} ({density_tier})"
            )
        )

    def close(self) -> None:
        """Gracefully tear down input proxy, audio thread, TTS thread, and window."""
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        self.input_source.stop()
        self.audio_engine.stop()
        self.tts.stop()
        try:
            self.root.destroy()
        except Exception:
            pass


# =============================================================================
# CLI & HEADLESS DIAGNOSTIC ENTRYPOINT
# =============================================================================
def run_self_test() -> int:
    """Run an end-to-end headless verification of NASA datasets, CV, audio, and TTS."""
    manager = DatasetManager()
    datasets = manager.load_datasets()
    analyzer = CVAnalyzer()
    engine = CMinorAudioEngine(enable_audio=False)
    tts = AccessibilityTTS(enabled=False)

    print(f"[Self-Test] Discovered {len(datasets)} datasets:")
    for idx in range(len(datasets)):
        entry = manager.current_dataset()
        img = manager.load_current_image()
        h, w = img.shape[:2]
        corner_patch, corner_bbox = extract_sensor_box(img, 0, 0, box_size=40)
        edge_patch, edge_bbox = extract_sensor_box(img, w, h, box_size=40)
        center_patch, center_bbox = extract_sensor_box(img, w // 2, h // 2, box_size=40)
        center_res = analyzer.analyze_patch(center_patch, center_bbox, (w // 2, h // 2))
        channel, pitch, vel, vol, motif, dur = engine.plan_musical_step(center_res, idx)
        assert (pitch % 12) in engine.C_MINOR_PITCH_CLASSES
        assert corner_patch.shape[:2] == (40, 40) or corner_patch.size > 0
        assert edge_patch.shape[:2] == (40, 40) or edge_patch.size > 0
        print(
            f"  - {entry.name} ({w}x{h}): corner={corner_bbox}, edge={edge_bbox}, "
            f"center={center_res.dominant_color} I={center_res.intensity:.2f} "
            f"D={center_res.density:.2f} -> ch={channel} {engine.midi_to_note_name(pitch)} "
            f"{motif} ({dur:.3f}s) vel={vel} vol={vol:.2f}"
        )
        print(f"    Speech: {analyzer.format_speech_summary(center_res, dataset=entry)}")
        manager.next_dataset()

    engine.start()
    engine.stop()
    tts.start()
    tts.stop()
    print("[Self-Test] PASSED")
    return 0


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Earth Jukebox — NASA EIC Imagery Sonification (Windows Mouse & Linux evdev Ready)"
    )
    parser.add_argument(
        "--input",
        choices=("mouse", "evdev"),
        default="mouse",
        help="Input coordinate source: 'mouse' (Windows default) or 'evdev' (Linux trackpad).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional Linux evdev device path (e.g. /dev/input/event7) when --input evdev.",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable sound output (useful for silent inspection).",
    )
    parser.add_argument(
        "--no-drums",
        action="store_true",
        help="Start with ambient drums muted (toggle anytime with D).",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable spoken TTS announcements.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run headless end-to-end verification and exit.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()

    if tk is None or Image is None or ImageTk is None:
        raise RuntimeError("tkinter and Pillow are required to launch the Earth Jukebox UI.")

    dataset_manager = DatasetManager()
    cv_analyzer = CVAnalyzer()
    audio_engine = CMinorAudioEngine(
        enable_audio=not args.no_audio,
        ambient_drums=not args.no_drums,
    )
    tts = AccessibilityTTS(enabled=not args.no_tts)

    root = tk.Tk()
    EarthJukeboxApp(
        root=root,
        dataset_manager=dataset_manager,
        cv_analyzer=cv_analyzer,
        audio_engine=audio_engine,
        tts=tts,
        input_mode=args.input,
        evdev_device=args.device,
    )
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
