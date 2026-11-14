"""Unit tests for cross-platform AccessibilityTTS."""

from __future__ import annotations

import platform
import shutil
import subprocess
import time

import pytest

from earth_jukebox.accessibility import tts as tts_module
from earth_jukebox.accessibility.tts import AccessibilityTTS


def test_tts_disabled_lifecycle() -> None:
    tts = AccessibilityTTS(enabled=False)
    tts.start()
    assert tts.backend_name == "Disabled"
    tts.speak(AccessibilityTTS.AUDIO_LEGEND_TEXT)
    tts.stop()


def test_tts_linux_spd_say_and_console_fallback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(tts_module, "pyttsx3", None)
    monkeypatch.setattr(platform, "system", lambda: "Linux")

    launched: list[list[str]] = []

    class _FakeProc:
        def __init__(self, cmd: list[str], **_kwargs: object) -> None:
            launched.append(cmd)

        def poll(self) -> int:
            return 0

        def wait(self) -> int:
            return 0

        def terminate(self) -> None:
            pass

    monkeypatch.setattr(subprocess, "Popen", _FakeProc)
    monkeypatch.setattr(
        shutil,
        "which",
        lambda name: "/usr/bin/spd-say" if name == "spd-say" else None,
    )

    tts = AccessibilityTTS(enabled=True)
    tts.start()
    tts.speak("Testing Linux speech dispatcher")
    time.sleep(0.15)
    tts.stop()

    assert tts.backend_name == "Linux speech-dispatcher/espeak"
    assert launched == [["spd-say", "--wait", "Testing Linux speech dispatcher"]]

    # Now verify console fallback when no Linux TTS binary is present
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    fallback_tts = AccessibilityTTS(enabled=True)
    fallback_tts.start()
    fallback_tts.speak("Console utterance")
    time.sleep(0.15)
    fallback_tts.stop()

    captured = capsys.readouterr().out
    assert fallback_tts.backend_name == "Console Fallback"
    assert "[TTS] Console utterance" in captured


def test_tts_windows_fallback_uses_powershell_and_interrupts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tts_module, "pyttsx3", None)
    monkeypatch.setattr(platform, "system", lambda: "Windows")

    launched: list[list[str]] = []
    terminated = False

    class _SlowProc:
        def __init__(self, cmd: list[str], **_kwargs: object) -> None:
            launched.append(cmd)
            self._done = False

        def poll(self) -> None:
            return None if not self._done else 0

        def wait(self) -> int:
            for _ in range(20):
                if self._done:
                    return 0
                time.sleep(0.01)
            return 0

        def terminate(self) -> None:
            nonlocal terminated
            terminated = True
            self._done = True

    monkeypatch.setattr(subprocess, "Popen", _SlowProc)

    tts = AccessibilityTTS(enabled=True)
    tts.start()
    tts.speak("First utterance")
    time.sleep(0.08)
    tts.speak("Interrupting utterance", interrupt=True)
    time.sleep(0.15)
    tts.stop()

    assert tts.backend_name == "Windows System.Speech"
    assert any(cmd[0] == "powershell" for cmd in launched)
    assert terminated is True
