"""Cross-platform MIDI and software synthesizer audio backends.

Designed for 100% Linux, macOS, and Windows compatibility:
``ctypes.windll`` and ``ctypes.wintypes`` are NEVER imported at module import
time and are accessed lazily inside :class:`WinmmMidiBackend` only on Windows.
"""

from __future__ import annotations

import ctypes
import math
import platform
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a core dependency
    np = None  # type: ignore[assignment]


class AudioBackend(ABC):
    """Abstract MIDI / polyphonic synthesizer backend."""

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


class NullAudioBackend(AudioBackend):
    """Silent backend used in headless environments or automated tests."""

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


class PygameMidiBackend(AudioBackend):
    """Cross-platform MIDI playback via ``pygame.midi`` when an output port exists."""

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


class WinmmMidiBackend(AudioBackend):
    """Native Windows General MIDI output via ``winmm.dll`` (zero pip dependencies)."""

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


class LinuxFluidSynthBackend(AudioBackend):
    """Optional Linux MIDI synthesizer via ``fluidsynth`` CLI + system GM SoundFont."""

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


class PygameMixerSynthBackend(AudioBackend):
    """Cross-platform software wavetable fallback using ``pygame.mixer`` + ``numpy``.

    Useful on Linux desktops without a hardware MIDI daemon (e.g. TiMidity/FluidSynth)
    when ``pygame`` and an audio output device are available.
    """

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
        pygame.mixer.set_num_channels(24)
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

        duration = 0.22 if channel == 9 else 1.25
        n_samples = int(self.SAMPLE_RATE * duration)
        t = np.linspace(0.0, duration, n_samples, endpoint=False, dtype=np.float32)

        if channel == 9:
            # Synthesized soft kick / rim / hat percussion
            if midi_note in (35, 36):
                freq_sweep = 110.0 * np.exp(-t * 18.0) + 42.0
                phase = 2.0 * math.pi * np.cumsum(freq_sweep) / self.SAMPLE_RATE
                wave = np.sin(phase) * np.exp(-t * 14.0)
            elif midi_note in (37, 39, 63):
                wave = (
                    np.sin(2.0 * math.pi * 420.0 * t) * 0.6
                    + np.sin(2.0 * math.pi * 890.0 * t) * 0.4
                )
                wave = wave * np.exp(-t * 28.0)
            else:
                rng = np.random.default_rng(midi_note)
                noise = rng.uniform(-1.0, 1.0, size=n_samples).astype(np.float32)
                decay = 12.0 if midi_note == 46 else 34.0
                wave = noise * np.exp(-t * decay) * 0.35
        else:
            freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
            program = self._programs.get(channel, 80)
            phase = 2.0 * math.pi * freq * t
            if program in (80, 81):
                # Warm Veridis Quo analog square/organ blend
                wave = 0.55 * np.tanh(2.2 * np.sin(phase)) + 0.30 * np.sin(2.0 * phase)
                env = np.minimum(1.0, t / 0.015) * np.exp(-t * 2.2)
            elif program in (33, 38):
                # Analog French-touch synth bass
                wave = 0.70 * np.sin(phase) + 0.30 * np.tanh(1.8 * np.sin(2.0 * phase))
                env = np.minimum(1.0, t / 0.010) * np.exp(-t * 3.0)
            else:
                # Warm Drawbar Organ / Rhodes Pad
                wave = (
                    0.65 * np.sin(phase)
                    + 0.25 * np.sin(2.0 * phase)
                    + 0.10 * np.sin(3.0 * phase)
                )
                env = np.minimum(1.0, t / 0.035) * np.exp(-t * 1.1)
            wave = wave * env

        pcm = np.clip(wave * 11000.0, -32767, 32767).astype(np.int16)
        stereo = np.column_stack((pcm, pcm))
        sound = self._sndarray.make_sound(stereo)
        if len(self._sound_cache) > 256:
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
                vol = max(0.05, min(1.0, int(velocity) / 127.0)) * 0.45
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


def create_best_audio_backend(enable_audio: bool = True) -> AudioBackend:
    """Select the highest-fidelity available MIDI/synth backend on the current OS."""
    if not enable_audio:
        return NullAudioBackend()

    candidates: list[type[AudioBackend]] = [PygameMidiBackend]
    sys_name = platform.system()
    if sys_name == "Windows":
        candidates.append(WinmmMidiBackend)
    elif sys_name == "Linux":
        candidates.append(LinuxFluidSynthBackend)
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
