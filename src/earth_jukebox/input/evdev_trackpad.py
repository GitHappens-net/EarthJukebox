"""Linux evdev touchpad driver and input proxy with lazy evdev imports."""

from __future__ import annotations

import queue
import threading
from typing import Any, Optional

from earth_jukebox.input.base import CoordinateInputSource, push_latest_point
from earth_jukebox.models import TouchPoint


def _import_evdev() -> tuple[Any, Any, Any]:
    """Lazily import ``evdev`` so non-Linux platforms can import this module cleanly."""
    try:
        from evdev import InputDevice, ecodes, list_devices
    except ImportError as exc:
        raise RuntimeError(
            "python-evdev is required for Linux trackpad input. "
            "Install it with: pip install earth-jukebox[linux]"
        ) from exc
    return InputDevice, ecodes, list_devices


def _absolute_codes(device: Any, ecodes: Any) -> set[int]:
    return {code for code, _ in device.capabilities().get(ecodes.EV_ABS, [])}


def _axis_codes(device: Any, ecodes: Any) -> tuple[int, int]:
    absolute_codes = _absolute_codes(device, ecodes)
    if {ecodes.ABS_X, ecodes.ABS_Y}.issubset(absolute_codes):
        return ecodes.ABS_X, ecodes.ABS_Y
    return ecodes.ABS_MT_POSITION_X, ecodes.ABS_MT_POSITION_Y


def find_touchpad(path: Optional[str] = None) -> Any:
    """Locate the raw Linux touchpad ``InputDevice`` exposing absolute coordinates."""
    InputDevice, ecodes, list_devices = _import_evdev()

    if path:
        device = InputDevice(path)
        absolute_codes = _absolute_codes(device, ecodes)
        has_coordinates = (
            {ecodes.ABS_X, ecodes.ABS_Y}.issubset(absolute_codes)
            or {ecodes.ABS_MT_POSITION_X, ecodes.ABS_MT_POSITION_Y}.issubset(absolute_codes)
        )
        if not has_coordinates:
            raise RuntimeError(
                f"{path} does not expose absolute touch coordinates. "
                "Choose the raw touchpad event device instead of its mouse interface."
            )
        return device

    candidates = []
    for device_path in list_devices():
        device = InputDevice(device_path)
        absolute_codes = _absolute_codes(device, ecodes)
        keys = set(device.capabilities().get(ecodes.EV_KEY, []))
        has_coordinates = (
            {ecodes.ABS_X, ecodes.ABS_Y}.issubset(absolute_codes)
            or {ecodes.ABS_MT_POSITION_X, ecodes.ABS_MT_POSITION_Y}.issubset(absolute_codes)
        )
        has_touch_button = ecodes.BTN_TOUCH in keys or ecodes.BTN_TOOL_FINGER in keys
        if has_coordinates and (has_touch_button or "touchpad" in device.name.lower()):
            candidates.append(device)

    if not candidates:
        raise RuntimeError(
            "No touchpad was found. Run with --device /dev/input/eventX "
            "or check access to /dev/input/event*."
        )
    if len(candidates) > 1:
        names = ", ".join(f"{device.path} ({device.name})" for device in candidates)
        raise RuntimeError(f"Multiple touchpads found: {names}. Choose one with --device.")
    return candidates[0]


def read_touchpad(
    device: Any,
    points: queue.Queue[TouchPoint],
    stop: threading.Event,
) -> None:
    """Read absolute touch coordinates from an ``evdev`` device into ``points``."""
    _InputDevice, ecodes, _list_devices = _import_evdev()
    x_code, y_code = _axis_codes(device, ecodes)
    x_info = device.absinfo(x_code)
    y_info = device.absinfo(y_code)
    if x_info is None or y_info is None or x_info.max <= x_info.min or y_info.max <= y_info.min:
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
            push_latest_point(
                points,
                TouchPoint(
                    x=x,
                    y=y,
                    x_ratio=(x - x_info.min) / (x_info.max - x_info.min),
                    y_ratio=(y - y_info.min) / (y_info.max - y_info.min),
                    touching=touching,
                ),
            )


class EvdevTrackpadInputProxy(CoordinateInputSource):
    """Linux hardware adapter streaming raw ``evdev`` touchpad events."""

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
        """Open the touchpad device and start the background reader thread."""
        device = find_touchpad(self.device_path)
        self._device_label = f"evdev: {device.name} [{device.path}]"
        self._thread = threading.Thread(
            target=read_touchpad,
            args=(device, self.points_queue, self.stop_event),
            name="EvdevTrackpadReader",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the reader thread to stop and wait briefly for exit."""
        self.stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)
