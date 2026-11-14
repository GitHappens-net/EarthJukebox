from __future__ import annotations
import webview
import argparse
import json
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional
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
                "interface."
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

def _update_pad_state(points: queue.Queue[TouchPoint], stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            point = points.get(timeout=0.1)
        except queue.Empty:
            continue
        pad_state.update(
            x=point.x_ratio,
            y=point.y_ratio,
            touching=point.touching,
            fingers=1 if point.touching else 0,
        )

class JukeboxAPI:
    """Template for methods exposed to the TypeScript frontend."""

    def load_dataset(self, dataset_id: str) -> dict[str, str | float]:
        return {
            "id": dataset_id,
            "unit": "ppm" if dataset_id == "co2" else "°C",
            "min": 410.0,
            "max": 435.0,
        }

    def calibrate_pad(self) -> str:
        pad_state.update(x=0.5, y=0.5, touching=False, fingers=0)
        return "Pad calibrated"

    def update_pad(self, x: float, y: float, touching: bool, fingers: int) -> None:
        pad_state.update(
            x=max(0.0, min(1.0, x)),
            y=max(0.0, min(1.0, y)),
            touching=touching,
            fingers=max(0, fingers),
        )

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

        if input_mode == "evdev":
            device = find_touchpad(args.device)

            def reader() -> None:
                try:
                    read_touchpad(device, points, stop)
                except Exception as exc:
                    reader_error.append(exc)
                    stop.set()

            threading.Thread(target=reader, daemon=True).start()
            threading.Thread(
                target=_update_pad_state, args=(points, stop), daemon=True
            ).start()

        window = webview.create_window(
            title="Earth Information Jukebox",
            url=resolve_ui_url(),
            js_api=JukeboxAPI(),
            width=1200,
            height=800,
        )
        webview.start(
            stream_trackpad_to_ui,
            args=(window, stop),
            http_server=True,
        )
        stop.set()
        if reader_error:
            raise reader_error[0]
        return 0
    except (OSError, RuntimeError) as exc:
        print(f"trackpad monitor: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
