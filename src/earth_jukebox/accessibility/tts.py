"""Compatibility shim for the removed voice output path.

Accessibility announcements remain available as visible/live text in the UI.
This class is retained so integrations that construct it do not break, but it
never starts a voice engine or launches a speech subprocess.
"""

from __future__ import annotations


class AccessibilityTTS:
    """No-op compatibility API; screen-reader text is rendered by the UI."""

    AUDIO_LEGEND_TEXT = (
        "Welcome to the Earth Jukebox. A 90 beat per minute C minor soundscape translates "
        "NASA Earth visualizations into sound. Low tones indicate cool zones, high rapid "
        "arpeggios indicate dense or intense regions, swirling patterns modulate motion, "
        "and ambient percussion follows spatial complexity. Press R to read the current zone, "
        "I for dataset information, P for auto-scan jukebox mode, D to toggle drums, "
        "plus or minus to resize the sensor box, and Tab to switch datasets."
    )

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    @property
    def backend_name(self) -> str:
        return "Disabled (screen-reader text only)"

    def start(self) -> None:
        """Retained for callers; no voice output is initialized."""

    def speak(self, text: str, interrupt: bool = True) -> None:
        """Retained for callers; announcements are intentionally not spoken."""

    def stop(self) -> None:
        """Retained for callers; there is no worker to stop."""
