"""Input coordinate sources for Earth Jukebox."""

from __future__ import annotations

from earth_jukebox.input.base import CoordinateInputSource, push_latest_point
from earth_jukebox.input.evdev_trackpad import (
    EvdevTrackpadInputProxy,
    find_touchpad,
    read_touchpad,
)
from earth_jukebox.input.mouse import WindowsMouseInputProxy

__all__ = [
    "CoordinateInputSource",
    "EvdevTrackpadInputProxy",
    "WindowsMouseInputProxy",
    "find_touchpad",
    "push_latest_point",
    "read_touchpad",
]
