"""Base input abstractions and bounded queue helpers."""

from __future__ import annotations

import queue
import threading
from abc import ABC, abstractmethod

from earth_jukebox.models import TouchPoint


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
        """Human-readable label describing the active hardware or proxy source."""

    @abstractmethod
    def start(self) -> None:
        """Begin emitting ``TouchPoint`` events into ``self.points_queue``."""

    @abstractmethod
    def stop(self) -> None:
        """Stop emitting events and release input resources."""
