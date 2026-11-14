"""Mouse hover proxy input source for Tkinter canvases."""

from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING, Callable, Optional

from earth_jukebox.input.base import CoordinateInputSource, push_latest_point
from earth_jukebox.models import TouchPoint

if TYPE_CHECKING:
    import tkinter as tk


class WindowsMouseInputProxy(CoordinateInputSource):
    """Mouse stand-in input driver tracking Tkinter canvas hover coordinates."""

    def __init__(
        self,
        canvas: tk.Canvas,
        points_queue: queue.Queue[TouchPoint],
        stop_event: threading.Event,
        on_user_activity: Optional[Callable[[], None]] = None,
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
            x_ratio=max(0.0, min(1.0, float(x_ratio))),
            y_ratio=max(0.0, min(1.0, float(y_ratio))),
            touching=touching,
        )

    def _on_motion(self, event: tk.Event) -> None:
        if self.stop_event.is_set():
            return
        if self.on_user_activity is not None:
            self.on_user_activity()
        point = self._canvas_to_touchpoint(event.x, event.y, touching=True)
        push_latest_point(self.points_queue, point)

    def _on_leave(self, _event: tk.Event) -> None:
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
                x_ratio=max(0.0, min(1.0, float(x_ratio))),
                y_ratio=max(0.0, min(1.0, float(y_ratio))),
                touching=False,
            ),
        )
