from __future__ import annotations
import argparse
import base64
import json
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

if sys.platform.startswith("linux"):
    try:
        from PySide6 import QtWebEngineCore  # noqa: F401
    except ImportError:
        pass
    else:
        os.environ.setdefault("QT_API", "pyside6")

import webview

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from backend.earth_jukebox.datasets import DatasetManager
from backend.earth_jukebox.server.service import SonificationService

try:
    from evdev import InputDevice, ecodes, list_devices
except ImportError:
    InputDevice = None
    ecodes = None
    list_devices = None

@dataclass(frozen=True)
class TouchPoint:
    x: int
    y: int
    x_ratio: float
    y_ratio: float
    touching: bool

PadState = dict[str, float | bool | int]
pad_state: PadState = {"x": 0.5, "y": 0.5, "touching": False, "fingers": 0}

def _absolute_codes(device: InputDevice) -> set[int]:
    if ecodes is None:
        raise RuntimeError("Linux evdev input is unavailable on this platform.")
    return {code for code, _ in device.capabilities().get(ecodes.EV_ABS, [])}

def _has_coordinates(device: InputDevice) -> bool:
    absolute_codes = _absolute_codes(device)
    return (
        {ecodes.ABS_X, ecodes.ABS_Y}.issubset(absolute_codes)
        or {
            ecodes.ABS_MT_POSITION_X,
            ecodes.ABS_MT_POSITION_Y,
        }.issubset(absolute_codes)
    )

def find_touchpad(path: Optional[str]) -> InputDevice:
    if InputDevice is None or ecodes is None or list_devices is None:
        raise RuntimeError(
            "Raw evdev input is only available on Linux. Use --input browser instead."
        )
    if path:
        device = InputDevice(path)
        if not _has_coordinates(device):
            raise RuntimeError(
                f"{path} does not expose absolute touch coordinates. "
                "Choose the raw touchpad event device instead of its mouse "
                "interface, or omit --device to auto-detect it."
            )
        return device

    candidates: list[InputDevice] = []
    for device_path in list_devices():
        device = InputDevice(device_path)
        keys = set(device.capabilities().get(ecodes.EV_KEY, []))
        has_touch_button = ecodes.BTN_TOUCH in keys or ecodes.BTN_TOOL_FINGER in keys
        if _has_coordinates(device) and (
            has_touch_button or "touchpad" in device.name.lower()
        ):
            candidates.append(device)

    if not candidates:
        raise RuntimeError(
            "No touchpad was found. Run with --device /dev/input/eventX "
            "or check access to /dev/input/event*."
        )
    if len(candidates) > 1:
        names = ", ".join(f"{device.path} ({device.name})" for device in candidates)
        raise RuntimeError(
            f"Multiple touchpads found: {names}. Choose one with --device."
        )
    return candidates[0]

def _axis_codes(device: InputDevice) -> tuple[int, int]:
    if ecodes is None:
        raise RuntimeError("Linux evdev input is unavailable on this platform.")
    absolute_codes = _absolute_codes(device)
    if {ecodes.ABS_X, ecodes.ABS_Y}.issubset(absolute_codes):
        return ecodes.ABS_X, ecodes.ABS_Y
    return ecodes.ABS_MT_POSITION_X, ecodes.ABS_MT_POSITION_Y

def read_touchpad(
    device: InputDevice, points: queue.Queue[TouchPoint], stop: threading.Event
) -> None:
    if ecodes is None:
        raise RuntimeError("Linux evdev input is unavailable on this platform.")
    x_code, y_code = _axis_codes(device)
    x_info = device.absinfo(x_code)
    y_info = device.absinfo(y_code)
    if (
        x_info is None
        or y_info is None
        or x_info.max <= x_info.min
        or y_info.max <= y_info.min
    ):
        raise RuntimeError(f"Could not read coordinate range from {device.path}.")

    x = x_info.min
    y = y_info.min
    keys = device.capabilities().get(ecodes.EV_KEY, [])
    has_touch_signal = ecodes.BTN_TOUCH in keys or ecodes.BTN_TOOL_FINGER in keys
    touching = not has_touch_signal

    for event in device.read_loop():
        if stop.is_set():
            return
        if event.type == ecodes.EV_ABS:
            if event.code == x_code:
                x = event.value
            elif event.code == y_code:
                y = event.value
        elif event.type == ecodes.EV_KEY:
            if event.code in (ecodes.BTN_TOUCH, ecodes.BTN_TOOL_FINGER):
                touching = event.value != 0
        elif event.type == ecodes.EV_SYN and event.code == ecodes.SYN_REPORT:
            x = max(x_info.min, min(x, x_info.max))
            y = max(y_info.min, min(y, y_info.max))
            if points.full():
                points.get_nowait()
            points.put_nowait(
                TouchPoint(
                    x=x,
                    y=y,
                    x_ratio=(x - x_info.min) / (x_info.max - x_info.min),
                    y_ratio=(y - y_info.min) / (y_info.max - y_info.min),
                    touching=touching,
                )
            )

def _update_pad_state(
    points: queue.Queue[TouchPoint], stop: threading.Event, api: JukeboxAPI
) -> None:
    while not stop.is_set():
        try:
            point = points.get(timeout=0.1)
        except queue.Empty:
            continue
        api.update_pad(
            x=point.x_ratio,
            y=point.y_ratio,
            touching=point.touching,
            fingers=1 if point.touching else 0,
        )
        pad_state.update(
            x=point.x_ratio,
            y=point.y_ratio,
            touching=point.touching,
            fingers=1 if point.touching else 0,
        )

class JukeboxAPI:
    """Template for methods exposed to the TypeScript frontend."""

    def __init__(self) -> None:
        self._selected_dataset = 0
        self._audio_enabled = True
        image_dir = Path(__file__).resolve().parent / "earth_jukebox" / "datasets" / "images"
        manager = DatasetManager(
            datasets_dir=image_dir,
            include_tactile_reference=False,
        )
        self._service = SonificationService(
            dataset_manager=manager,
            enable_local_audio=True,
        )
        self._service.start()

    def get_datasets(self) -> list[dict[str, object]]:
        return [
            {
                "id": str(item["index"]),
                "title": str(item["name"]),
                "description": str(item["summary"]),
                "image": (
                    "data:image/png;base64,"
                    + base64.b64encode(
                        self._service.get_dataset_image_png(int(item["index"]))
                    ).decode("ascii")
                ),
                "unit": str(item["unit"]),
                "category": str(item["category"]),
                "legend": item["legend"],
                "min": 0.0,
                "max": 0.0,
            }
            for item in self._service.list_datasets()
        ]

    def load_dataset(self, dataset_id: str) -> dict[str, object]:
        self.select_dataset(dataset_id)
        for dataset in self.get_datasets():
            if dataset["id"] == dataset_id:
                return dataset
        raise ValueError(f"Unknown dataset: {dataset_id}")

    def calibrate_pad(self) -> str:
        pad_state.update(x=0.5, y=0.5, touching=False, fingers=0)
        return "Pad calibrated"

    def update_pad(self, x: float, y: float, touching: bool, fingers: int) -> None:
        touching = touching and self._audio_enabled
        pad_state.update(
            x=max(0.0, min(1.0, x)),
            y=max(0.0, min(1.0, y)),
            touching=touching,
            fingers=max(0, fingers),
        )
        self._service.process_coordinates(
            {
                "x_ratio": float(pad_state["x"]),
                "y_ratio": float(pad_state["y"]),
                "touching": bool(pad_state["touching"]),
                "sensor_size": 40,
                "dataset": self._selected_dataset,
            }
        )

    def set_audio_enabled(self, enabled: bool) -> None:
        self._audio_enabled = enabled
        if not enabled:
            self.update_pad(
                float(pad_state["x"]),
                float(pad_state["y"]),
                False,
                0,
            )

    def get_audio_status(self) -> dict[str, object]:
        status = self._service.current_status()
        return {
            "backend": status["backend"],
            "running": status["status"] == "ok",
            "bpm": status["bpm"],
        }

    def close(self) -> None:
        self._service.stop()

    def select_dataset(self, dataset_id: str) -> None:
        try:
            self._selected_dataset = int(dataset_id)
        except ValueError as exc:
            raise ValueError(f"Invalid dataset id: {dataset_id}") from exc

    def get_music_snapshot(self) -> dict[str, object]:
        """Analyze the current browser/evdev position without starting local audio."""
        return self._service.process_coordinates(
            {
                "x_ratio": float(pad_state["x"]),
                "y_ratio": float(pad_state["y"]),
                "touching": bool(pad_state["touching"]),
                "sensor_size": 40,
                "dataset": self._selected_dataset,
            },
            update_live=False,
        )

    def render_audio(self) -> dict[str, object]:
        """Return a browser-playable base64 WAV and its analysis/music snapshot."""
        wav_bytes, snapshot = self._service.render_audio_wav(
            {
                "x_ratio": float(pad_state["x"]),
                "y_ratio": float(pad_state["y"]),
                "touching": True,
                "sensor_size": 40,
                "dataset": self._selected_dataset,
                "bars": 1,
                "sample_rate": 22050,
                "preview_only": True,
            }
        )
        return {
            "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
            "mime_type": "audio/wav",
            "snapshot": snapshot,
        }

def resolve_ui_url() -> str:
    if "--dev" in sys.argv:
        return "http://localhost:5173"
    base_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    packaged_ui = os.path.join(base_dir, "dist", "index.html")
    if os.path.exists(packaged_ui):
        return packaged_ui
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "frontend", "dist", "index.html")
    )

def stream_trackpad_to_ui(
    window: webview.Window, stop: threading.Event
) -> None:
    while not stop.is_set():
        payload = json.dumps(pad_state)
        try:
            window.evaluate_js(
                "window.dispatchEvent(new CustomEvent('pad-update', "
                f"{{ detail: {payload} }}));"
            )
        except Exception:
            if stop.is_set():
                return
        time.sleep(1 / 60)

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show touchpad coordinates in a React UI."
    )
    parser.add_argument(
        "--input",
        choices=("auto", "evdev", "browser"),
        default="auto",
        help="Input source: raw Linux evdev, browser pointer events, or auto-detect.",
    )
    parser.add_argument(
        "--device",
        help="event device path, for example /dev/input/event7; auto-detect by default",
    )
    args = parser.parse_args()

    try:
        points: queue.Queue[TouchPoint] = queue.Queue(maxsize=2)
        stop = threading.Event()
        reader_error: list[Exception] = []
        input_mode = args.input
        if input_mode == "auto":
            input_mode = "evdev" if sys.platform.startswith("linux") else "browser"

        device: Optional[InputDevice] = None
        if input_mode == "evdev":
            device = find_touchpad(args.device)

            def reader() -> None:
                try:
                    read_touchpad(device, points, stop)
                except Exception as exc:
                    reader_error.append(exc)
                    stop.set()

        api = JukeboxAPI()
        if device is not None:
            threading.Thread(target=reader, daemon=True).start()
            threading.Thread(
                target=_update_pad_state, args=(points, stop, api), daemon=True
            ).start()

        window = webview.create_window(
            title="Earth Information Jukebox",
            url=resolve_ui_url(),
            js_api=api,
            width=1200,
            height=800,
        )
        webview.start(
            stream_trackpad_to_ui,
            args=(window, stop),
            http_server=True,
        )
        stop.set()
        api.close()
        if reader_error:
            raise reader_error[0]
        return 0
    except (OSError, RuntimeError) as exc:
        print(f"trackpad monitor: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
