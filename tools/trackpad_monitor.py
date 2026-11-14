from __future__ import annotations

import argparse
import queue
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from typing import Optional

try:
    from evdev import InputDevice, ecodes, list_devices
except ImportError as exc:
    raise SystemExit(
        "This script requires python-evdev. Install it with: python3 -m pip install evdev"
    ) from exc


@dataclass(frozen=True)
class TouchPoint:
    x: int
    y: int
    x_ratio: float
    y_ratio: float
    touching: bool


def _absolute_codes(device: InputDevice) -> set[int]:
    return {code for code, _ in device.capabilities().get(ecodes.EV_ABS, [])}


def find_touchpad(path: Optional[str]) -> InputDevice:
    if path:
        device = InputDevice(path)
        absolute_codes = _absolute_codes(device)
        has_coordinates = (
            {ecodes.ABS_X, ecodes.ABS_Y}.issubset(absolute_codes)
            or {
                ecodes.ABS_MT_POSITION_X,
                ecodes.ABS_MT_POSITION_Y,
            }.issubset(absolute_codes)
        )
        if not has_coordinates:
            raise RuntimeError(
                f"{path} does not expose absolute touch coordinates. "
                "Choose the raw touchpad event device instead of its mouse "
                "interface."
            )
        return device

    candidates: list[InputDevice] = []
    for device_path in list_devices():
        device = InputDevice(device_path)
        absolute_codes = _absolute_codes(device)
        keys = set(device.capabilities().get(ecodes.EV_KEY, []))
        has_coordinates = (
            {ecodes.ABS_X, ecodes.ABS_Y}.issubset(absolute_codes)
            or {
                ecodes.ABS_MT_POSITION_X,
                ecodes.ABS_MT_POSITION_Y,
            }.issubset(absolute_codes)
        )
        has_touch_button = ecodes.BTN_TOUCH in keys or ecodes.BTN_TOOL_FINGER in keys
        if has_coordinates and (
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
    absolute_codes = _absolute_codes(device)
    if {ecodes.ABS_X, ecodes.ABS_Y}.issubset(absolute_codes):
        return ecodes.ABS_X, ecodes.ABS_Y
    return ecodes.ABS_MT_POSITION_X, ecodes.ABS_MT_POSITION_Y


def read_touchpad(
    device: InputDevice, points: queue.Queue[TouchPoint], stop: threading.Event
) -> None:
    x_code, y_code = _axis_codes(device)
    x_info = device.absinfo(x_code)
    y_info = device.absinfo(y_code)
    if x_info is None or y_info is None or x_info.max <= x_info.min:
        raise RuntimeError(f"Could not read coordinate range from {device.path}.")
    if y_info.max <= y_info.min:
        raise RuntimeError(f"Could not read coordinate range from {device.path}.")

    x = x_info.min
    y = y_info.min
    has_touch_signal = ecodes.BTN_TOUCH in device.capabilities().get(
        ecodes.EV_KEY, []
    ) or ecodes.BTN_TOOL_FINGER in device.capabilities().get(ecodes.EV_KEY, [])
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


class TouchpadView:
    def __init__(
        self,
        root: tk.Tk,
        device: InputDevice,
        points: queue.Queue[TouchPoint],
        stop: threading.Event,
    ) -> None:
        self.root = root
        self.device = device
        self.points = points
        self.stop = stop
        self.canvas = tk.Canvas(
            root, width=600, height=400, background="#20252b", highlightthickness=0
        )
        self.canvas.pack(padx=16, pady=(16, 8))
        self.dot = self.canvas.create_oval(0, 0, 0, 0, fill="#48d1cc", outline="")
        self.status = tk.Label(root, text=f"{device.name}  [{device.path}]")
        self.status.pack(pady=(0, 12))
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.update()

    def update(self) -> None:
        try:
            while True:
                point = self.points.get_nowait()
                if point.touching:
                    width = self.canvas.winfo_width()
                    height = self.canvas.winfo_height()
                    x = point.x_ratio * width
                    y = point.y_ratio * height
                    radius = 8
                    self.canvas.coords(
                        self.dot, x - radius, y - radius, x + radius, y + radius
                    )
                    self.canvas.itemconfigure(self.dot, state="normal")
                    self.status.configure(text=f"{self.device.name}  x={point.x}  y={point.y}")
                else:
                    self.canvas.itemconfigure(self.dot, state="hidden")
                    self.status.configure(text=f"{self.device.name}  (not touching)")
        except queue.Empty:
            pass
        if not self.stop.is_set():
            self.root.after(16, self.update)

    def close(self) -> None:
        self.stop.set()
        self.root.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show raw coordinates from a Linux touchpad."
    )
    parser.add_argument(
        "--device",
        help="event device path, for example /dev/input/event7; auto-detect by default",
    )
    args = parser.parse_args()

    try:
        device = find_touchpad(args.device)
        points: queue.Queue[TouchPoint] = queue.Queue(maxsize=2)
        stop = threading.Event()
        reader_error: list[Exception] = []

        def reader() -> None:
            try:
                read_touchpad(device, points, stop)
            except Exception as exc:
                reader_error.append(exc)
                stop.set()

        root = tk.Tk()
        root.title("Trackpad monitor")
        TouchpadView(root, device, points, stop)
        threading.Thread(target=reader, daemon=True).start()
        root.mainloop()
        if reader_error:
            raise reader_error[0]
        return 0
    except (OSError, RuntimeError) as exc:
        print(f"trackpad monitor: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
