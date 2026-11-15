from __future__ import annotations

import ctypes
import platform
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

try:
    import numpy as np # type: ignore
except ImportError:  # pragma: no cover - numpy is a core dependency
    np = None  # type: ignore[assignment]

# Abstract MIDI / polyphonic synthesizer backend
class AudioBackend(ABC):
    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable name of the active audio output driver."""

    @abstractmethod
    def start(self, channel_programs: dict[int, int]) -> None:
        """Open audio output and configure initial MIDI channel instruments."""

    @abstractmethod
    def note_on(self, channel: int, midi_note: int, velocity: int) -> None:
        """Trigger a quantized MIDI note."""

    @abstractmethod
    def note_off(self, channel: int, midi_note: int) -> None:
        """Release a quantized MIDI note."""

    def control_change(self, channel: int, control: int, value: int) -> None:
        """Send an optional MIDI CC message (Reverb, Chorus, Brightness, etc.)."""

    def pitch_bend(self, channel: int, bend_14bit: int) -> None:
        """Send an optional 14-bit MIDI pitch-bend value (``0..16383``, center ``8192``)."""

    @abstractmethod
    def close(self) -> None:
        """Silence active voices and release audio handles."""

# Silent backend used in headless environments or automated tests
class NullAudioBackend(AudioBackend):
    @property
    def backend_name(self) -> str:
        return "Null Audio (Headless)"

    def start(self, channel_programs: dict[int, int]) -> None:
        pass

    def note_on(self, channel: int, midi_note: int, velocity: int) -> None:
        pass

    def note_off(self, channel: int, midi_note: int) -> None:
        pass

    def close(self) -> None:
        pass

# Cross-platform MIDI playback via ``pygame.midi`` when an output port exists
class PygameMidiBackend(AudioBackend):
    def __init__(self) -> None:
        self._out = None
        self._pygame_midi = None

    @property
    def backend_name(self) -> str:
        return "pygame.midi"

    def start(self, channel_programs: dict[int, int]) -> None:
        try:
            import pygame
            import pygame.midi
        except ImportError as exc:
            raise RuntimeError("pygame.midi is not installed") from exc

        self._pygame_midi = pygame.midi
        pygame.midi.init()
        try:
            device_id = pygame.midi.get_default_output_id()
            if device_id == -1:
                for idx in range(pygame.midi.get_count()):
                    _interf, _name, _is_in, is_out, _opened = pygame.midi.get_device_info(idx)
                    if is_out:
                        device_id = idx
                        break
            if device_id == -1:
                raise RuntimeError("No pygame.midi output device available")
            self._out = pygame.midi.Output(device_id)
        except Exception:
            pygame.midi.quit()
            raise

        for channel, program in channel_programs.items():
            if channel != 9:
                self._out.set_instrument(int(program) & 0x7F, channel=int(channel) & 0x0F)

    def note_on(self, channel: int, midi_note: int, velocity: int) -> None:
        if self._out is not None:
            self._out.note_on(
                int(midi_note) & 0x7F,
                max(0, min(127, int(velocity))),
                channel=int(channel) & 0x0F,
            )

    def note_off(self, channel: int, midi_note: int) -> None:
        if self._out is not None:
            self._out.note_off(int(midi_note) & 0x7F, 0, channel=int(channel) & 0x0F)

    def control_change(self, channel: int, control: int, value: int) -> None:
        if self._out is not None:
            self._out.write_short(
                0xB0 | (int(channel) & 0x0F),
                int(control) & 0x7F,
                max(0, min(127, int(value))),
            )

    def pitch_bend(self, channel: int, bend_14bit: int) -> None:
        if self._out is not None:
            clamped = max(0, min(16383, int(bend_14bit)))
            lsb = clamped & 0x7F
            msb = (clamped >> 7) & 0x7F
            self._out.write_short(0xE0 | (int(channel) & 0x0F), lsb, msb)

    def close(self) -> None:
        if self._out is not None:
            try:
                for ch in range(16):
                    self.control_change(ch, 123, 0)
                self._out.close()
            finally:
                self._out = None
                if self._pygame_midi is not None:
                    try:
                        self._pygame_midi.quit()
                    except Exception:
                        pass

# Native Windows General MIDI output via ``winmm.dll`` (zero pip dependencies)
class WinmmMidiBackend(AudioBackend):
    def __init__(self) -> None:
        if platform.system() != "Windows":
            raise RuntimeError("WinmmMidiBackend is only supported on Windows")

        from ctypes import wintypes

        self._wintypes = wintypes
        self._winmm = ctypes.windll.winmm
        self._handle = ctypes.c_void_p()
        self._opened = False
        self._configure_ctypes()

    @property
    def backend_name(self) -> str:
        return "Windows winmm MIDI Synth"

    def _configure_ctypes(self) -> None:
        wt = self._wintypes
        self._winmm.midiOutGetNumDevs.restype = wt.UINT
        self._winmm.midiOutOpen.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            wt.UINT,
            ctypes.c_void_p,
            ctypes.c_void_p,
            wt.DWORD,
        ]
        self._winmm.midiOutOpen.restype = wt.UINT
        self._winmm.midiOutShortMsg.argtypes = [ctypes.c_void_p, wt.DWORD]
        self._winmm.midiOutShortMsg.restype = wt.UINT
        self._winmm.midiOutClose.argtypes = [ctypes.c_void_p]
        self._winmm.midiOutClose.restype = wt.UINT

    def start(self, channel_programs: dict[int, int]) -> None:
        if self._winmm.midiOutGetNumDevs() == 0:
            raise RuntimeError("No Windows MIDI output devices detected")
        result = self._winmm.midiOutOpen(ctypes.byref(self._handle), 0, 0, 0, 0)
        if result != 0:
            raise RuntimeError(f"midiOutOpen failed with error code {result}")
        self._opened = True
        for channel, program in channel_programs.items():
            if channel != 9:
                msg = (0xC0 | (int(channel) & 0x0F)) | ((int(program) & 0x7F) << 8)
                self._winmm.midiOutShortMsg(self._handle, msg)

    def note_on(self, channel: int, midi_note: int, velocity: int) -> None:
        if not self._opened:
            return
        status = 0x90 | (int(channel) & 0x0F)
        vel = max(0, min(127, int(velocity)))
        msg = status | ((int(midi_note) & 0x7F) << 8) | (vel << 16)
        self._winmm.midiOutShortMsg(self._handle, msg)

    def note_off(self, channel: int, midi_note: int) -> None:
        if not self._opened:
            return
        status = 0x80 | (int(channel) & 0x0F)
        msg = status | ((int(midi_note) & 0x7F) << 8)
        self._winmm.midiOutShortMsg(self._handle, msg)

    def control_change(self, channel: int, control: int, value: int) -> None:
        if not self._opened:
            return
        val = max(0, min(127, int(value)))
        msg = (0xB0 | (int(channel) & 0x0F)) | ((int(control) & 0x7F) << 8) | (val << 16)
        self._winmm.midiOutShortMsg(self._handle, msg)

    def pitch_bend(self, channel: int, bend_14bit: int) -> None:
        if not self._opened:
            return
        clamped = max(0, min(16383, int(bend_14bit)))
        lsb = clamped & 0x7F
        msb = (clamped >> 7) & 0x7F
        msg = (0xE0 | (int(channel) & 0x0F)) | (lsb << 8) | (msb << 16)
        self._winmm.midiOutShortMsg(self._handle, msg)

    def close(self) -> None:
        if self._opened:
            try:
                for channel in range(16):
                    self.control_change(channel, 123, 0)
                self._winmm.midiOutClose(self._handle)
            finally:
                self._opened = False
                self._handle = ctypes.c_void_p()

# Optional Linux MIDI synthesizer via ``fluidsynth`` CLI + system GM SoundFont
class LinuxFluidSynthBackend(AudioBackend):
    DEFAULT_SOUNDFONTS: tuple[str, ...] = (
        "/usr/share/sounds/sf2/FluidR3_GM.sf2",
        "/usr/share/soundfonts/FluidR3_GM.sf2",
        "/usr/share/sounds/sf2/default-GM.sf2",
        "/usr/share/soundfonts/default.sf2",
        "/usr/share/sounds/sf2/TimGM6mb.sf2",
    )

    def __init__(self, soundfont_path: Optional[Path] = None) -> None:
        self._soundfont_path = soundfont_path
        self._proc: Optional[subprocess.Popen[str]] = None

    @property
    def backend_name(self) -> str:
        return "Linux FluidSynth GM"

    @classmethod
    def find_soundfont(cls, explicit_path: Optional[Path] = None) -> Optional[Path]:
        if explicit_path is not None and explicit_path.is_file():
            return explicit_path
        for candidate in cls.DEFAULT_SOUNDFONTS:
            path = Path(candidate)
            if path.is_file():
                return path
        return None

    def start(self, channel_programs: dict[int, int]) -> None:
        if platform.system() != "Linux":
            raise RuntimeError("LinuxFluidSynthBackend targets Linux hosts")
        fluidsynth_bin = shutil.which("fluidsynth")
        if not fluidsynth_bin:
            raise RuntimeError("fluidsynth binary not found")
        sf2 = self.find_soundfont(self._soundfont_path)
        if sf2 is None:
            raise RuntimeError("No GM SoundFont (.sf2) found for FluidSynth")

        self._proc = subprocess.Popen(
            [fluidsynth_bin, "-q", "-g", "0.8", str(sf2)],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if self._proc.poll() is not None:
            self._proc = None
            raise RuntimeError("fluidsynth exited immediately during startup")

        for channel, program in channel_programs.items():
            if channel != 9:
                self._send_cmd(f"prog {int(channel) & 0x0F} {int(program) & 0x7F}")

    def _send_cmd(self, command: str) -> None:
        if self._proc is None or self._proc.stdin is None or self._proc.poll() is not None:
            return
        try:
            self._proc.stdin.write(command + "\n")
            self._proc.stdin.flush()
        except OSError:
            pass

    def note_on(self, channel: int, midi_note: int, velocity: int) -> None:
        vel = max(0, min(127, int(velocity)))
        self._send_cmd(f"noteon {int(channel) & 0x0F} {int(midi_note) & 0x7F} {vel}")

    def note_off(self, channel: int, midi_note: int) -> None:
        self._send_cmd(f"noteoff {int(channel) & 0x0F} {int(midi_note) & 0x7F}")

    def control_change(self, channel: int, control: int, value: int) -> None:
        val = max(0, min(127, int(value)))
        self._send_cmd(f"cc {int(channel) & 0x0F} {int(control) & 0x7F} {val}")

    def pitch_bend(self, channel: int, bend_14bit: int) -> None:
        clamped = max(0, min(16383, int(bend_14bit)))
        self._send_cmd(f"pitch_bend {int(channel) & 0x0F} {clamped}")

    def close(self) -> None:
        if self._proc is not None:
            try:
                for ch in range(16):
                    self._send_cmd(f"cc {ch} 123 0")
                self._send_cmd("quit")
                self._proc.wait(timeout=0.5)
            except Exception:
                try:
                    self._proc.terminate()
                except Exception:
                    pass
            finally:
                self._proc = None

# Cross-platform software wavetable fallback using pygame.mixer + numpy
class PygameMixerSynthBackend(AudioBackend):
    SAMPLE_RATE = 22050

    def __init__(self) -> None:
        self._mixer = None
        self._sndarray = None
        self._active_voices: dict[tuple[int, int], object] = {}
        self._programs: dict[int, int] = {}
        self._sound_cache: dict[tuple[int, int], object] = {}

    @property
    def backend_name(self) -> str:
        return "pygame.mixer Software Synth"

    def start(self, channel_programs: dict[int, int]) -> None:
        if np is None:
            raise RuntimeError("numpy is required for PygameMixerSynthBackend")
        try:
            import pygame
            import pygame.mixer
            import pygame.sndarray
        except ImportError as exc:
            raise RuntimeError("pygame.mixer is not available") from exc

        pygame.mixer.init(frequency=self.SAMPLE_RATE, size=-16, channels=2, buffer=512)
        pygame.mixer.set_num_channels(48)
        self._mixer = pygame.mixer
        self._sndarray = pygame.sndarray
        self._programs = dict(channel_programs)
        self._sound_cache.clear()
        self._active_voices.clear()

    def _build_note_sound(self, channel: int, midi_note: int):
        assert np is not None
        assert self._sndarray is not None

        key = (channel, midi_note)
        cached = self._sound_cache.get(key)
        if cached is not None:
            return cached

        from backend.earth_jukebox.audio.synth import render_channel_note_pcm

        # Rendered at a fixed reference velocity; loudness is applied with set_volume().
        pcm = render_channel_note_pcm(channel, midi_note, sr=self.SAMPLE_RATE, gate_sec=1.2, velocity=100)
        sound = self._sndarray.make_sound(pcm)
        if len(self._sound_cache) > 384:
            self._sound_cache.clear()
        self._sound_cache[key] = sound
        return sound

    def note_on(self, channel: int, midi_note: int, velocity: int) -> None:
        if self._mixer is None:
            return
        try:
            sound = self._build_note_sound(int(channel) & 0x0F, int(midi_note) & 0x7F)
            mixer_ch = sound.play()
            if mixer_ch is not None:
                vol = max(0.05, min(1.0, (int(velocity) / 127.0) ** 1.3)) * 0.55
                mixer_ch.set_volume(vol)
                self._active_voices[(channel, midi_note)] = mixer_ch
        except Exception:
            pass

    def note_off(self, channel: int, midi_note: int) -> None:
        if self._mixer is None:
            return
        mixer_ch = self._active_voices.pop((channel, midi_note), None)
        if mixer_ch is not None:
            try:
                mixer_ch.fadeout(45)
            except Exception:
                pass

    def close(self) -> None:
        if self._mixer is not None:
            try:
                self._mixer.stop()
                self._mixer.quit()
            except Exception:
                pass
            finally:
                self._mixer = None
                self._active_voices.clear()
                self._sound_cache.clear()

# Decorate an audio backend while publishing MIDI events to subscribers
class BroadcastAudioBackend(AudioBackend):
    def __init__(self, inner: AudioBackend, max_queue_size: int = 128) -> None:
        import queue
        import threading
        self.inner = inner
        self._max_queue_size = max(1, int(max_queue_size))
        self._subscribers: set[queue.Queue[dict[str, object]]] = set()
        self._lock = threading.Lock()
        self._programs: dict[int, int] = {}

    @property
    def backend_name(self) -> str:
        return self.inner.backend_name

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    def subscribe(self):
        import queue
        subscriber: queue.Queue[dict[str, object]] = queue.Queue(maxsize=self._max_queue_size)
        with self._lock:
            self._subscribers.add(subscriber)
        return subscriber

    def unsubscribe(self, subscriber) -> None:
        with self._lock:
            self._subscribers.discard(subscriber)

    def _publish(self, event: dict[str, object]) -> None:
        with self._lock:
            subscribers = tuple(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(event)
            except Exception:
                try:
                    subscriber.get_nowait()
                    subscriber.put_nowait(event)
                except Exception:
                    pass

    def publish(self, event: dict[str, object]) -> None:
        self._publish(event)

    def start(self, channel_programs: dict[int, int]) -> None:
        self._programs = dict(channel_programs)
        self.inner.start(channel_programs)
        self._publish({"type": "engine_start", "programs": dict(channel_programs)})

    def note_on(self, channel: int, midi_note: int, velocity: int) -> None:
        self.inner.note_on(channel, midi_note, velocity)
        from backend.earth_jukebox.audio.theory import (
            CHANNEL_PAN,
            CHANNEL_TIMBRE,
            midi_to_freq,
            midi_to_note_name,
        )
        self._publish({"type": "note_on", "channel": channel, "midi_note": midi_note,
                       "note_name": midi_to_note_name(midi_note), "freq_hz": midi_to_freq(midi_note),
                       "velocity": velocity, "program": self._programs.get(channel),
                       "timbre": CHANNEL_TIMBRE.get(channel, "drums" if channel == 9 else None),
                       "pan": CHANNEL_PAN.get(channel, 0.0)})

    def note_off(self, channel: int, midi_note: int) -> None:
        self.inner.note_off(channel, midi_note)
        self._publish({"type": "note_off", "channel": channel, "midi_note": midi_note})

    def control_change(self, channel: int, control: int, value: int) -> None:
        self.inner.control_change(channel, control, value)
        self._publish({"type": "control_change", "channel": channel, "control": control, "value": value})

    def pitch_bend(self, channel: int, bend_14bit: int) -> None:
        self.inner.pitch_bend(channel, bend_14bit)
        self._publish({"type": "pitch_bend", "channel": channel, "value": bend_14bit})

    def close(self) -> None:
        try:
            self.inner.close()
        finally:
            self._publish({"type": "engine_stop"})

# Select the highest-fidelity available MIDI/synth backend on the current OS
def create_best_audio_backend(enable_audio: bool = True) -> AudioBackend:
    if not enable_audio:
        return NullAudioBackend()

    sys_name = platform.system()
    if sys_name == "Linux":
        candidates: list[type[AudioBackend]] = [LinuxFluidSynthBackend, PygameMidiBackend]
    else:
        candidates = [PygameMidiBackend]
    if sys_name == "Windows":
        candidates.append(WinmmMidiBackend)
    candidates.append(PygameMixerSynthBackend)

    for backend_cls in candidates:
        try:
            probe = backend_cls()
            probe.start({})
            probe.close()
            return backend_cls()
        except Exception:
            continue
    return NullAudioBackend()