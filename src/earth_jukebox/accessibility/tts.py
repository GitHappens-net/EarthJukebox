"""Threaded, cross-platform text-to-speech worker for accessibility."""

from __future__ import annotations

import platform
import queue
import shutil
import subprocess
import threading
from typing import Any, Optional

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None  # type: ignore[assignment]

try:
    import pythoncom
except ImportError:
    pythoncom = None  # type: ignore[assignment]


class AccessibilityTTS:
    """Non-blocking, interruptible TTS worker using ``pyttsx3`` with OS fallbacks."""

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
        self._queue: queue.Queue[Optional[str]] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc_lock = threading.Lock()
        self._active_proc: Optional[subprocess.Popen[bytes]] = None
        self._pyttsx3_engine: Optional[Any] = None
        self._backend_name = "Disabled"

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def start(self) -> None:
        """Start the dedicated speech worker thread."""
        if not self.enabled:
            self._backend_name = "Disabled"
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._tts_loop,
            name="AccessibilityTTS",
            daemon=True,
        )
        self._thread.start()

    def speak(self, text: str, interrupt: bool = True) -> None:
        """Queue an utterance asynchronously without blocking UI or audio."""
        cleaned = text.strip()
        if not cleaned or not self.enabled:
            return
        if interrupt:
            self._drain_pending()
            self._interrupt_active()
        self._queue.put(cleaned)

    def stop(self) -> None:
        """Stop the speech thread and terminate any active utterance."""
        self._stop_event.set()
        self._drain_pending()
        self._interrupt_active()
        self._queue.put(None)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _drain_pending(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _interrupt_active(self) -> None:
        if self._pyttsx3_engine is not None:
            try:
                self._pyttsx3_engine.stop()
            except Exception:
                pass
        with self._proc_lock:
            if self._active_proc is not None and self._active_proc.poll() is None:
                try:
                    self._active_proc.terminate()
                except Exception:
                    pass
                self._active_proc = None

    def _init_pyttsx3_on_worker(self) -> Optional[Any]:
        if pyttsx3 is None:
            return None
        try:
            if platform.system() == "Windows" and pythoncom is not None:
                pythoncom.CoInitialize()
            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            voices = engine.getProperty("voices")
            if not voices:
                return None
            self._backend_name = "pyttsx3"
            return engine
        except Exception:
            return None

    def _tts_loop(self) -> None:
        self._pyttsx3_engine = self._init_pyttsx3_on_worker()
        if self._pyttsx3_engine is None:
            if platform.system() == "Windows":
                self._backend_name = "Windows System.Speech"
            elif shutil.which("spd-say") or shutil.which("espeak") or shutil.which("espeak-ng"):
                self._backend_name = "Linux speech-dispatcher/espeak"
            else:
                self._backend_name = "Console Fallback"

        try:
            while not self._stop_event.is_set():
                try:
                    text = self._queue.get(timeout=0.25)
                except queue.Empty:
                    continue
                if text is None or self._stop_event.is_set():
                    break
                self._speak_blocking(text)
        finally:
            if self._pyttsx3_engine is not None:
                try:
                    self._pyttsx3_engine.stop()
                except Exception:
                    pass
                self._pyttsx3_engine = None
            if platform.system() == "Windows" and pythoncom is not None:
                try:
                    pythoncom.CoUninitialize()
                except Exception:
                    pass

    def _speak_blocking(self, text: str) -> None:
        if self._pyttsx3_engine is not None:
            try:
                self._pyttsx3_engine.say(text)
                self._pyttsx3_engine.runAndWait()
                return
            except Exception:
                self._pyttsx3_engine = None

        if platform.system() == "Windows":
            safe_text = text.replace("'", "''")
            ps_script = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "$s.Rate = 1; "
                f"$s.Speak('{safe_text}')"
            )
            self._run_tts_subprocess(["powershell", "-NoProfile", "-Command", ps_script])
            return

        if shutil.which("spd-say"):
            self._run_tts_subprocess(["spd-say", "--wait", text])
            return
        if shutil.which("espeak"):
            self._run_tts_subprocess(["espeak", text])
            return
        if shutil.which("espeak-ng"):
            self._run_tts_subprocess(["espeak-ng", text])
            return

        print(f"[TTS] {text}")

    def _run_tts_subprocess(self, cmd: list[str]) -> None:
        popen_kwargs: dict[str, Any] = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if platform.system() == "Windows":
            creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if creation_flags:
                popen_kwargs["creationflags"] = creation_flags

        try:
            proc = subprocess.Popen(cmd, **popen_kwargs)
            with self._proc_lock:
                self._active_proc = proc
            proc.wait()
        except Exception:
            pass
        finally:
            with self._proc_lock:
                self._active_proc = None
