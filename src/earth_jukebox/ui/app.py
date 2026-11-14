"""Accessible Tkinter controller and fixed-layout HUD for Earth Jukebox."""

from __future__ import annotations

import math
import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np

try:
    import tkinter as tk
    from tkinter import font as tkfont
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover - optional only in headless test runs
    tk = None  # type: ignore[assignment]
    tkfont = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]
    ImageTk = None  # type: ignore[assignment]

from earth_jukebox.accessibility import AccessibilityTTS
from earth_jukebox.audio import CMinorAudioEngine
from earth_jukebox.datasets import DatasetManager
from earth_jukebox.input import (
    CoordinateInputSource,
    EvdevTrackpadInputProxy,
    WindowsMouseInputProxy,
    push_latest_point,
)
from earth_jukebox.models import AnalysisResult, TouchPoint
from earth_jukebox.vision import CVAnalyzer, extract_sensor_box


class EarthJukeboxApp:
    """Coordinates Input Proxy -> CV Analyzer -> C-Minor Audio Engine -> UI/TTS."""

    CANVAS_WIDTH = 960
    CANVAS_HEIGHT = 540
    WINDOW_WIDTH = 992
    WINDOW_HEIGHT = 770

    MIN_SENSOR_SIZE = 16
    MAX_SENSOR_SIZE = 160
    SENSOR_SIZE_STEP = 8
    DEFAULT_SENSOR_SIZE = 40
    SENSOR_BOX_SIZE = DEFAULT_SENSOR_SIZE

    COLOR_SWATCH_HEX: dict[str, str] = {
        "Red": "#ef4444",
        "Orange": "#f97316",
        "Green": "#22c55e",
        "Blue": "#38bdf8",
        "Black": "#334155",
        "White": "#f8fafc",
    }

    SANS_CANDIDATES: tuple[str, ...] = (
        "Segoe UI",
        "DejaVu Sans",
        "Liberation Sans",
        "Noto Sans",
        "Helvetica",
    )
    MONO_CANDIDATES: tuple[str, ...] = (
        "Consolas",
        "DejaVu Sans Mono",
        "Liberation Mono",
        "Ubuntu Mono",
        "Courier New",
    )

    def __init__(
        self,
        root: "tk.Tk",
        dataset_manager: DatasetManager,
        cv_analyzer: CVAnalyzer,
        audio_engine: CMinorAudioEngine,
        tts: AccessibilityTTS,
        input_mode: str = "mouse",
        evdev_device: Optional[str] = None,
        sensor_size: int = DEFAULT_SENSOR_SIZE,
    ) -> None:
        if tk is None or Image is None or ImageTk is None:
            raise RuntimeError("tkinter and Pillow are required to launch the Earth Jukebox UI.")

        self.root = root
        self.dataset_manager = dataset_manager
        self.cv_analyzer = cv_analyzer
        self.audio_engine = audio_engine
        self.tts = tts
        self.sensor_size = self._clamp_sensor_size(sensor_size)

        self.stop_event = threading.Event()
        self.points_queue: queue.Queue[TouchPoint] = queue.Queue(maxsize=2)
        self._photo_image: Optional["ImageTk.PhotoImage"] = None
        self._current_image_bgr: Optional[np.ndarray] = None
        self._last_touch_point: Optional[TouchPoint] = None
        self._poll_after_id: Optional[str] = None
        self._auto_scan_enabled = False
        self._auto_scan_phase = 0.0
        self._last_scan_tick = time.monotonic()

        self._latest_analysis = AnalysisResult(
            dominant_color="Blue",
            intensity=0.0,
            density=0.0,
            shape="uniform",
            shape_score=0.0,
            spottiness=0.0,
            swirliness=0.0,
            sensor_size=self.sensor_size,
            sensor_bbox=(0, 0, 0, 0),
            cursor_xy=(0, 0),
            touching=False,
        )

        self._sans_family = self._resolve_font_family(self.SANS_CANDIDATES, "TkDefaultFont")
        self._mono_family = self._resolve_font_family(self.MONO_CANDIDATES, "TkFixedFont")

        self._build_ui()
        self._bind_keyboard_shortcuts()

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
            text=(
                f"Input: {self.input_source.source_name}  |  "
                f"Audio: {self.audio_engine.backend.backend_name}"
            )
        )
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._poll_after_id = self.root.after(16, self._poll_input_and_update)

    def _resolve_font_family(self, candidates: tuple[str, ...], fallback: str) -> str:
        """Pick the first installed font family or fall back to a Tk standard alias."""
        if tkfont is None:
            return fallback
        try:
            available = {name.lower(): name for name in tkfont.families(self.root)}
        except Exception:
            return fallback
        for candidate in candidates:
            match = available.get(candidate.lower())
            if match:
                return match
        return fallback

    @classmethod
    def _clamp_sensor_size(cls, value: int) -> int:
        clamped = max(cls.MIN_SENSOR_SIZE, min(cls.MAX_SENSOR_SIZE, int(value)))
        offset = clamped - cls.MIN_SENSOR_SIZE
        snapped = cls.MIN_SENSOR_SIZE + round(offset / cls.SENSOR_SIZE_STEP) * cls.SENSOR_SIZE_STEP
        return max(cls.MIN_SENSOR_SIZE, min(cls.MAX_SENSOR_SIZE, int(snapped)))

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

        self.source_label = tk.Label(
            header,
            text="",
            fg="#94a3b8",
            bg="#0b1120",
            font=(self._sans_family, 9),
            anchor="e",
        )
        self.source_label.pack(side="right", fill="y", padx=(12, 0))

        self.dataset_label = tk.Label(
            header,
            text="Dataset: Loading...",
            fg="#f8fafc",
            bg="#0b1120",
            font=(self._sans_family, 12, "bold"),
            anchor="w",
        )
        self.dataset_label.pack(side="left", fill="both", expand=True)

        science_bar = tk.Frame(self.root, bg="#0b1120", width=self.CANVAS_WIDTH, height=46)
        science_bar.pack(fill="x", padx=14, pady=(0, 6))
        science_bar.pack_propagate(False)

        self.science_label = tk.Label(
            science_bar,
            text="",
            fg="#cbd5e1",
            bg="#0b1120",
            font=(self._sans_family, 9),
            anchor="nw",
            justify="left",
            wraplength=self.CANVAS_WIDTH - 4,
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
        self._crosshair_h = self.canvas.create_line(
            0, 0, 0, 0, fill="#facc15", width=1, state="hidden"
        )
        self._crosshair_v = self.canvas.create_line(
            0, 0, 0, 0, fill="#facc15", width=1, state="hidden"
        )
        self._dot_item = self.canvas.create_oval(
            0, 0, 0, 0, fill="#facc15", outline="#0f172a", width=1, state="hidden"
        )

        hud = tk.Frame(self.root, bg="#1e293b", padx=12, pady=6, width=self.CANVAS_WIDTH, height=82)
        hud.pack(fill="x", padx=14, pady=(6, 4))
        hud.pack_propagate(False)

        hud_row_top = tk.Frame(hud, bg="#1e293b", height=22)
        hud_row_top.pack(fill="x")
        hud_row_top.pack_propagate(False)

        self.color_chip = tk.Label(hud_row_top, text="  ", bg="#38bdf8", width=3)
        self.color_chip.pack(side="left", padx=(0, 8), pady=1)

        self.telemetry_label = tk.Label(
            hud_row_top,
            text=f"Hover over a NASA map or press [P] for Auto-Scan  |  Box={self.sensor_size}px",
            fg="#e2e8f0",
            bg="#1e293b",
            font=(self._mono_family, 9),
            anchor="w",
        )
        self.telemetry_label.pack(side="left", fill="x", expand=True)

        hud_row_mid = tk.Frame(hud, bg="#1e293b", height=22)
        hud_row_mid.pack(fill="x", pady=(1, 1))
        hud_row_mid.pack_propagate(False)

        self.morphology_label = tk.Label(
            hud_row_mid,
            text=f"Box={self.sensor_size}px  |  Shape=--  |  Spot=--  |  Swirl=--",
            fg="#cbd5e1",
            bg="#1e293b",
            font=(self._mono_family, 9),
            anchor="w",
        )
        self.morphology_label.pack(side="left", fill="x", expand=True)

        hud_row_bottom = tk.Frame(hud, bg="#1e293b", height=24)
        hud_row_bottom.pack(fill="x")
        hud_row_bottom.pack_propagate(False)

        self.beat_label = tk.Label(
            hud_row_bottom,
            text="Beat 1/4  ● ○ ○ ○",
            fg="#38bdf8",
            bg="#1e293b",
            font=(self._mono_family, 10, "bold"),
            anchor="e",
        )
        self.beat_label.pack(side="right", padx=(8, 0))

        self.audio_label = tk.Label(
            hud_row_bottom,
            text="90 BPM | Veridis Quo C-Minor | Standby",
            fg="#7dd3fc",
            bg="#1e293b",
            font=(self._mono_family, 9, "bold"),
            anchor="w",
        )
        self.audio_label.pack(side="left", fill="x", expand=True)

        shortcuts_bar = tk.Frame(self.root, bg="#0b1120", width=self.CANVAS_WIDTH, height=28)
        shortcuts_bar.pack(fill="x", padx=14, pady=(0, 6))
        shortcuts_bar.pack_propagate(False)

        shortcuts = tk.Label(
            shortcuts_bar,
            text=(
                "[Space] Legend  |  [R] Read Zone  |  [I] Info  |  [P] Auto-Scan  |  "
                "[D] Drums  |  [Tab/1-8] Dataset  |  [+/-/Scroll] Box  |  [Esc] Exit"
            ),
            fg="#cbd5e1",
            bg="#0b1120",
            font=(self._sans_family, 9),
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

        # Sensor box resize: keyboard (+, =, KP_Add, ] to grow; -, _, KP_Subtract, [ to shrink)
        grow_keys = (
            "<KeyPress-plus>",
            "<KeyPress-equal>",
            "<KeyPress-KP_Add>",
            "<KeyPress-bracketright>",
        )
        shrink_keys = (
            "<KeyPress-minus>",
            "<KeyPress-underscore>",
            "<KeyPress-KP_Subtract>",
            "<KeyPress-bracketleft>",
        )
        for seq in grow_keys:
            self.root.bind(seq, self._on_grow_sensor_box)
        for seq in shrink_keys:
            self.root.bind(seq, self._on_shrink_sensor_box)

        # Sensor box resize: mouse wheel (Windows/macOS + Linux X11 Button-4/5)
        self.root.bind("<MouseWheel>", self._on_mouse_wheel)
        self.root.bind("<Button-4>", self._on_grow_sensor_box)
        self.root.bind("<Button-5>", self._on_shrink_sensor_box)

    def _yield_auto_scan_to_user(self) -> None:
        if self._auto_scan_enabled:
            self._auto_scan_enabled = False

    def _on_space_legend(self, _event: Optional["tk.Event"]) -> str:
        self.tts.speak(self.tts.AUDIO_LEGEND_TEXT, interrupt=True)
        return "break"

    def _on_read_zone(self, _event: Optional["tk.Event"]) -> str:
        if not self._latest_analysis.touching:
            self.tts.speak(
                "Cursor is outside the map area. "
                "Hover over the visualization or press P for auto-scan.",
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
        state_msg = (
            "Auto-scan jukebox play started."
            if self._auto_scan_enabled
            else "Auto-scan paused."
        )
        self.tts.speak(state_msg, interrupt=True)
        return "break"

    def _on_toggle_drums(self, _event: Optional["tk.Event"]) -> str:
        enabled = self.audio_engine.toggle_drums()
        msg = "Ambient drums enabled." if enabled else "Ambient drums muted."
        self.tts.speak(msg, interrupt=True)
        return "break"

    def _on_tab_dataset(self, _event: Optional["tk.Event"]) -> str:
        self.dataset_manager.next_dataset()
        self._display_current_dataset(announce=True)
        return "break"

    def _on_select_dataset(self, index: int) -> str:
        self.dataset_manager.select_dataset(index)
        self._display_current_dataset(announce=True)
        return "break"

    def _on_grow_sensor_box(self, _event: Optional["tk.Event"] = None) -> str:
        self.adjust_sensor_size(self.SENSOR_SIZE_STEP)
        return "break"

    def _on_shrink_sensor_box(self, _event: Optional["tk.Event"] = None) -> str:
        self.adjust_sensor_size(-self.SENSOR_SIZE_STEP)
        return "break"

    def _on_mouse_wheel(self, event: "tk.Event") -> str:
        delta = getattr(event, "delta", 0)
        if delta > 0:
            self.adjust_sensor_size(self.SENSOR_SIZE_STEP)
        elif delta < 0:
            self.adjust_sensor_size(-self.SENSOR_SIZE_STEP)
        return "break"

    def adjust_sensor_size(self, delta: int) -> int:
        """Resize the CV sensor box within ``[16, 160]`` px and refresh HUD/TTS."""
        new_size = self._clamp_sensor_size(self.sensor_size + int(delta))
        if new_size == self.sensor_size:
            return self.sensor_size

        self.sensor_size = new_size
        self.tts.speak(f"Sensor box {self.sensor_size} pixels.", interrupt=True)

        if (
            self._last_touch_point is not None
            and self._last_touch_point.touching
            and self._current_image_bgr is not None
        ):
            self._process_touch_point(self._last_touch_point)
        else:
            self._latest_analysis = AnalysisResult(
                dominant_color=self._latest_analysis.dominant_color,
                intensity=0.0,
                density=0.0,
                shape=self._latest_analysis.shape,
                shape_score=0.0,
                spottiness=0.0,
                swirliness=0.0,
                sensor_size=self.sensor_size,
                sensor_bbox=self._latest_analysis.sensor_bbox,
                cursor_xy=self._latest_analysis.cursor_xy,
                touching=False,
            )
            self.audio_engine.update_analysis(self._latest_analysis)
            self.telemetry_label.configure(
                text=(
                    "Hover over a NASA map or press [P] for Auto-Scan  "
                    f"|  Box={self.sensor_size}px"
                )
            )
            self.morphology_label.configure(
                text=f"Box={self.sensor_size}px  |  Shape=--  |  Spot=--  |  Swirl=--"
            )
        return self.sensor_size

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
        self.science_label.configure(text=f"{entry.category} -- {entry.summary}")
        if (
            self._last_touch_point is not None
            and self._last_touch_point.touching
            and self._current_image_bgr is not None
        ):
            self._process_touch_point(self._last_touch_point)
        if announce:
            self.tts.speak(f"Switched dataset to {entry.name}.", interrupt=True)

    def _step_auto_scan(self) -> None:
        now = time.monotonic()
        dt = min(0.1, max(0.0, now - self._last_scan_tick))
        self._last_scan_tick = now
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
            self._poll_after_id = self.root.after(16, self._poll_input_and_update)

    def _hide_cursor_overlay(self) -> None:
        for item in (self._box_item, self._crosshair_h, self._crosshair_v, self._dot_item):
            self.canvas.itemconfigure(item, state="hidden")

    def _process_touch_point(self, point: TouchPoint) -> None:
        assert self._current_image_bgr is not None
        self._last_touch_point = point
        img_h, img_w = self._current_image_bgr.shape[:2]

        if not point.touching:
            self._hide_cursor_overlay()
            idle_result = AnalysisResult(
                dominant_color=self._latest_analysis.dominant_color,
                intensity=0.0,
                density=0.0,
                shape=self._latest_analysis.shape,
                shape_score=0.0,
                spottiness=0.0,
                swirliness=0.0,
                sensor_size=self.sensor_size,
                sensor_bbox=self._latest_analysis.sensor_bbox,
                cursor_xy=self._latest_analysis.cursor_xy,
                touching=False,
            )
            self._latest_analysis = idle_result
            self.audio_engine.update_analysis(idle_result)
            self.telemetry_label.configure(
                text=f"Pointer outside dataset canvas.  |  Box={self.sensor_size}px"
            )
            self.morphology_label.configure(
                text=f"Box={self.sensor_size}px  |  Shape=--  |  Spot=--  |  Swirl=--"
            )
            return

        img_x = int(round(point.x_ratio * max(0, img_w - 1)))
        img_y = int(round(point.y_ratio * max(0, img_h - 1)))

        patch, bbox = extract_sensor_box(
            self._current_image_bgr,
            img_x,
            img_y,
            box_size=self.sensor_size,
        )
        analysis = self.cv_analyzer.analyze_patch(
            roi=patch,
            bbox=bbox,
            cursor_xy=(img_x, img_y),
            touching=True,
            sensor_size=self.sensor_size,
        )
        self._latest_analysis = analysis
        self.audio_engine.update_analysis(analysis)

        scale_x = self.CANVAS_WIDTH / max(1, img_w)
        scale_y = self.CANVAS_HEIGHT / max(1, img_h)
        bx1, by1, bx2, by2 = bbox
        x1_c = bx1 * scale_x
        y1_c = by1 * scale_y
        x2_c = bx2 * scale_x
        y2_c = by2 * scale_y
        cx = point.x_ratio * (self.CANVAS_WIDTH - 1)
        cy = point.y_ratio * (self.CANVAS_HEIGHT - 1)

        swatch_hex = self.COLOR_SWATCH_HEX.get(analysis.dominant_color, "#38bdf8")
        self.canvas.coords(self._box_item, x1_c, y1_c, x2_c, y2_c)
        self.canvas.itemconfigure(self._box_item, outline=swatch_hex, state="normal")

        arm = max(6, min(14, int(round(min(x2_c - x1_c, y2_c - y1_c) * 0.25))))
        self.canvas.coords(self._crosshair_h, cx - arm, cy, cx + arm, cy)
        self.canvas.coords(self._crosshair_v, cx, cy - arm, cx, cy + arm)
        self.canvas.itemconfigure(self._crosshair_h, state="normal")
        self.canvas.itemconfigure(self._crosshair_v, state="normal")
        self.canvas.coords(self._dot_item, cx - 3, cy - 3, cx + 3, cy + 3)
        self.canvas.itemconfigure(self._dot_item, state="normal")

        self.color_chip.configure(bg=swatch_hex)
        mode_tag = "AUTO-SCAN" if self._auto_scan_enabled else "MANUAL"
        density_tier = self.cv_analyzer.density_tier(analysis.density)
        self.telemetry_label.configure(
            text=(
                f"[{mode_tag}] XY=({img_x:3d},{img_y:3d}) | "
                f"Box={self.sensor_size:3d}px | "
                f"Color={analysis.dominant_color:<6s} | "
                f"Intensity={analysis.intensity:.2f} | "
                f"Density={analysis.density:.2f} ({density_tier})"
            )
        )
        self.morphology_label.configure(
            text=(
                f"Shape={analysis.shape} ({analysis.shape_score:.2f}) | "
                f"Spot={analysis.spottiness:.2f} | "
                f"Swirl={analysis.swirliness:.2f}"
            )
        )

    def close(self) -> None:
        """Gracefully tear down input proxy, audio thread, TTS thread, and window."""
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        if self._poll_after_id is not None:
            try:
                self.root.after_cancel(self._poll_after_id)
            except Exception:
                pass
            self._poll_after_id = None
        self.input_source.stop()
        self.audio_engine.stop()
        self.tts.stop()
        try:
            self.root.destroy()
        except Exception:
            pass
