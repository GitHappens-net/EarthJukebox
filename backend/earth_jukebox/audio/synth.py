"""Pure-NumPy stereo synthesizer for Earth Jukebox.

Two jobs, one set of voices:

1. :func:`render_sonification_wav` / :func:`render_sonification` render a loop of the
   current analysis to a stereo WAV.  They use :class:`StepSequencer`, the *same*
   scheduler as the live engine, so the WAV is the same music as the live sound.
2. :func:`render_channel_note_pcm` renders one note for the real-time
   ``PygameMixerSynthBackend`` so the fallback live synth shares these instruments.

Everything is vectorised NumPy (no SciPy).  Timbres use band-limited wavetables, so
there is no aliasing, and brightness (image intensity) changes the spectral tilt.
"""

from __future__ import annotations

import io
import math
import wave
from functools import lru_cache
from typing import Optional

import numpy as np

from backend.earth_jukebox.audio.theory import (
    ARP_CHANNEL,
    BASS_CHANNEL,
    BPM,
    CHANNEL_PAN,
    CHANNEL_TIMBRE,
    COLOR_BRIGHTNESS_BIAS,
    COLOR_INSTRUMENT_NAME,
    DRUM_CHANNEL,
    DRUM_CLAP,
    DRUM_CONGA_HI,
    DRUM_HAT_CLOSED,
    DRUM_HAT_OPEN,
    DRUM_KICK,
    DRUM_LOW_TOM,
    DRUM_RIM,
    DRUM_SHAKER,
    DRUM_TAMBOURINE,
    PAD_ROLES,
    SIXTEENTH_DURATION_SEC,
    VERIDIS_QUO_PROGRESSION,
    AnalysisResult,
    NoteEvent,
    StepSequencer,
    describe_mapping,
    midi_to_freq,
)

SAMPLE_RATE = 44100
_TABLE_N = 2048
_TWO_PI = 2.0 * math.pi


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# =============================================================================
# OSCILLATORS (band-limited wavetables)
# =============================================================================
@lru_cache(maxsize=512)
def _wavetable(kind: str, harmonics: int, tilt_q: int) -> np.ndarray:
    k = np.arange(1, harmonics + 1, dtype=np.float64)
    if kind == "saw":
        amps = 1.0 / k
    elif kind == "square":
        amps = np.where(k % 2 == 1, 1.0 / k, 0.0)
    elif kind == "triangle":
        sign = np.where(((k - 1) // 2) % 2 == 0, 1.0, -1.0)
        amps = np.where(k % 2 == 1, sign / (k * k), 0.0)
    elif kind == "organ":
        weights = (0.64, 0.24, 0.12, 0.06, 0.03)
        amps = np.zeros(harmonics)
        amps[: min(len(weights), harmonics)] = weights[:harmonics]
    else:  # sine
        amps = np.zeros(harmonics)
        amps[0] = 1.0
    if kind in ("saw", "square", "triangle"):
        amps = amps * k ** (-tilt_q / 4.0)  # spectral tilt = filter brightness
    x = np.arange(_TABLE_N, dtype=np.float64) / _TABLE_N
    table = (amps[:, None] * np.sin(_TWO_PI * k[:, None] * x[None, :])).sum(axis=0)
    table /= max(1e-9, float(np.max(np.abs(table))))
    return np.append(table, table[0]).astype(np.float32)


def _osc(
    kind: str,
    freq: float,
    n: int,
    sr: int,
    brightness: float,
    vib: Optional[np.ndarray] = None,
    max_h: int = 24,
) -> np.ndarray:
    harmonics = max(1, min(max_h, int(0.45 * sr / max(freq, 1.0))))
    table = _wavetable(kind, harmonics, int(round((1.0 - _clamp01(brightness)) * 8)))
    inc = freq / sr
    if vib is None:
        phase = (np.arange(n, dtype=np.float64) * inc) % 1.0
    else:
        phase = np.cumsum(inc * (1.0 + vib)) % 1.0
    idx = phase * _TABLE_N
    i0 = idx.astype(np.int64)
    frac = (idx - i0).astype(np.float32)
    return table[i0] * (1.0 - frac) + table[i0 + 1] * frac


def _env(n: int, sr: int, gate: int, attack: float, decay: float, sustain: float, release: float) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / sr
    env = np.minimum(1.0, t / max(attack, 1e-4)) * (
        sustain + (1.0 - sustain) * np.exp(-np.maximum(0.0, t - attack) * (3.0 / max(decay, 1e-3)))
    )
    if gate < n:
        level = env[max(0, gate - 1)]
        r = np.arange(n - gate, dtype=np.float64) / max(1, n - gate)
        env[gate:] = level * (1.0 - r) ** 2
    return env


def _partials(freq: float, n: int, sr: int, parts, brightness: float) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / sr
    out = np.zeros(n, dtype=np.float64)
    for ratio, amp, decay in parts:
        f = freq * ratio
        if f >= 0.45 * sr:
            continue
        tilt = 1.0 if ratio == parts[0][0] else (0.35 + 0.65 * brightness)
        out += amp * tilt * np.sin(_TWO_PI * f * t) * np.exp(-t * decay)
    return out


# =============================================================================
# INSTRUMENTS
# =============================================================================
# kind "layers": sustained oscillator stack (detuned L/R for width)
#   layer = (waveform, freq_ratio, level, detune_cents, darkness)
# kind "synth":  like layers but with a bright->dark filter-envelope pluck
#   layer = (waveform, freq_ratio, level, detune_cents, filtered?)
# kind "partials": struck/plucked tones built from decaying partials
_VOICES: dict[str, dict] = {
    "organ_pad": dict(kind="layers", gain=0.13, env=(0.09, 0.5, 0.80, 0.35),
                      layers=[("organ", 1, 0.55, 0, 0.0), ("saw", 1, 0.45, 5, 0.10)]),
    "warm_pad": dict(kind="layers", gain=0.12, env=(0.25, 0.6, 0.85, 0.50),
                     layers=[("triangle", 1, 0.60, 6, 0.0), ("saw", 1, 0.40, 8, 0.35)]),
    "church": dict(kind="layers", gain=0.10, env=(0.20, 0.4, 0.85, 0.40),
                   layers=[("organ", 1, 0.70, 3, 0.0), ("organ", 2, 0.30, 0, 0.0)]),
    "sub_pad": dict(kind="layers", gain=0.17, env=(0.15, 0.5, 0.85, 0.40),
                    layers=[("sine", 1, 0.80, 0, 0.0), ("triangle", 2, 0.25, 4, 0.20)]),
    "glass_pad": dict(kind="layers", gain=0.15, env=(0.12, 0.5, 0.80, 0.45), trem=(4.2, 0.12),
                      layers=[("sine", 1, 0.60, 0, 0.0), ("sine", 2, 0.25, 3, 0.0),
                              ("sine", 3, 0.12, 0, 0.0), ("sine", 5, 0.06, 0, 0.0)]),
    "bass": dict(kind="synth", gain=0.26, env=(0.004, 0.25, 0.60, 0.08), fdec=0.14,
                 layers=[("saw", 1, 0.55, 0, True), ("sine", 1, 0.70, 0, False)]),
    "arp": dict(kind="synth", gain=0.15, env=(0.006, 0.30, 0.35, 0.05), fdec=0.09, vib=(5.5, 0.0),
                layers=[("square", 1, 0.80, 0, True), ("saw", 2, 0.15, 0, True)]),
    "saw_lead": dict(kind="synth", gain=0.12, env=(0.02, 0.25, 0.70, 0.12), fdec=0.35, vib=(5.2, 0.004),
                     layers=[("saw", 1, 0.55, 7, True), ("square", 1, 0.35, 0, True)]),
    "rhodes": dict(kind="partials", gain=0.17, release=0.12, trem=(4.5, 0.10),
                   parts=[(1.0, 1.0, 2.2), (2.0, 0.35, 4.0), (7.0, 0.18, 10.0)]),
    "marimba": dict(kind="partials", gain=0.20, release=0.06,
                    parts=[(1.0, 1.0, 6.0), (3.93, 0.38, 14.0), (9.5, 0.12, 22.0)]),
    "bell": dict(kind="partials", gain=0.13, release=0.25, trem=(5.0, 0.08),
                 parts=[(1.0, 1.0, 1.6), (2.76, 0.45, 2.6), (5.40, 0.25, 3.6), (8.93, 0.12, 5.0)]),
    "pluck": dict(kind="partials", gain=0.16, release=0.05,
                  parts=[(1.0, 1.0, 9.0), (2.0, 0.30, 14.0), (6.27, 0.20, 25.0)]),
}


def _balance(left: np.ndarray, right: np.ndarray, pan: float) -> tuple[np.ndarray, np.ndarray]:
    pan = max(-1.0, min(1.0, float(pan)))
    return left * min(1.0, 1.0 - pan), right * min(1.0, 1.0 + pan)


def render_note_stereo(
    timbre: str,
    freq: float,
    gate_sec: float,
    velocity: int,
    brightness: float = 0.6,
    sr: int = SAMPLE_RATE,
    vibrato: float = 0.0,
    pan: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Render one note (gate + release tail) as ``(left, right)`` float32 arrays."""
    spec = _VOICES.get(timbre) or _VOICES["organ_pad"]
    kind = spec["kind"]
    release = spec["env"][3] if "env" in spec else spec.get("release", 0.1)
    gate = max(1, int(gate_sec * sr))
    n = gate + int(release * sr)
    t = np.arange(n, dtype=np.float64) / sr
    amp = (max(1, min(127, int(velocity))) / 127.0) ** 1.35 * spec["gain"]
    b = _clamp01(brightness)

    vib = None
    if "vib" in spec or vibrato > 0.0:
        rate, base_depth = spec.get("vib", (5.5, 0.0))
        depth = base_depth + 0.012 * _clamp01(vibrato)
        if depth > 0.0:
            vib = depth * np.sin(_TWO_PI * rate * t)

    if kind == "partials":
        mono = _partials(freq, n, sr, spec["parts"], b)
        left = right = mono * _env(n, sr, gate, 0.003, 1.0, 1.0, release)
    else:
        left = np.zeros(n, dtype=np.float64)
        right = np.zeros(n, dtype=np.float64)

        def layer_signal(wave_kind: str, f: float, flag) -> np.ndarray:
            if kind == "synth":
                if flag:  # bright -> dark filter-envelope pluck
                    bright = _osc(wave_kind, f, n, sr, min(1.0, b + 0.25), vib)
                    dark = _osc(wave_kind, f, n, sr, max(0.0, b - 0.35), vib)
                    e = np.exp(-t / spec["fdec"])
                    return e * bright + (1.0 - e) * dark
                return _osc(wave_kind, f, n, sr, b, vib)
            return _osc(wave_kind, f, n, sr, max(0.0, b - flag), vib)  # flag = extra darkness

        for wave_kind, ratio, level, cents, flag in spec["layers"]:
            f = freq * ratio
            if cents == 0:
                sig = level * layer_signal(wave_kind, f, flag)
                left += sig
                right += sig
            else:  # detune left/right in opposite directions for stereo width
                left += level * layer_signal(wave_kind, f * 2.0 ** (cents / 1200.0), flag)
                right += level * layer_signal(wave_kind, f * 2.0 ** (-cents / 1200.0), flag)
        att, dec, sus, rel = spec["env"]
        env = _env(n, sr, gate, att, dec, sus, rel)
        left = left * env
        right = right * env
    if "trem" in spec:
        rate, depth = spec["trem"]
        trem = 1.0 - depth * 0.5 * (1.0 - np.sin(_TWO_PI * rate * t))
        left = left * trem
        right = right * trem
    left, right = _balance(left * amp, right * amp, pan)
    return left.astype(np.float32), right.astype(np.float32)


# =============================================================================
# DRUMS
# =============================================================================
def _hp(x: np.ndarray, times: int = 1) -> np.ndarray:
    for _ in range(times):
        x = np.diff(x, prepend=0.0)
    return x


def render_drum_stereo(note: int, velocity: int, sr: int = SAMPLE_RATE, seed: int = 0):
    amp = (max(1, min(127, int(velocity))) / 127.0) ** 1.2
    rng = np.random.default_rng(int(note) * 7919 + int(seed))

    def tt(dur: float) -> np.ndarray:
        return np.arange(int(dur * sr), dtype=np.float64) / sr

    if note == DRUM_KICK or note == 35:
        t = tt(0.32)
        freq = 46.0 + 120.0 * np.exp(-t * 30.0)
        phase = _TWO_PI * np.cumsum(freq) / sr
        sig = (np.sin(phase) * np.exp(-t * 8.5) + rng.standard_normal(len(t)) * 0.12 * np.exp(-t * 400.0)) * 1.05
        pan, gain = 0.0, 0.55
    elif note == DRUM_RIM:
        t = tt(0.08)
        sig = (0.5 * np.sin(_TWO_PI * 1750.0 * t) + 0.7 * np.sin(_TWO_PI * 480.0 * t)) * np.exp(-t * 70.0)
        sig += _hp(rng.standard_normal(len(t))) * 0.25 * np.exp(-t * 140.0)
        pan, gain = -0.15, 0.30
    elif note == DRUM_CLAP:
        t = tt(0.22)
        noise = _hp(rng.standard_normal(len(t)), 2)
        env = np.zeros(len(t))
        for onset in (0.0, 0.009, 0.018):
            env += np.where(t >= onset, np.exp(-(t - onset) * 110.0), 0.0)
        env += np.where(t >= 0.027, 0.5 * np.exp(-(t - 0.027) * 22.0), 0.0)
        sig = noise * env * 0.30
        pan, gain = 0.0, 0.40
    elif note == DRUM_HAT_CLOSED:
        t = tt(0.06)
        sig = _hp(rng.standard_normal(len(t)), 2) * np.exp(-t * 75.0) * 0.45
        pan, gain = 0.30, 0.40
    elif note == DRUM_HAT_OPEN:
        t = tt(0.28)
        sig = _hp(rng.standard_normal(len(t)), 2) * np.exp(-t * 13.0) * 0.40
        pan, gain = 0.30, 0.38
    elif note == DRUM_SHAKER:
        t = tt(0.09)
        sig = _hp(rng.standard_normal(len(t))) * np.minimum(1.0, t / 0.012) * np.exp(-t * 38.0) * 0.55
        pan, gain = -0.30, 0.34
    elif note == DRUM_TAMBOURINE:
        t = tt(0.18)
        sig = _hp(rng.standard_normal(len(t)), 2) * np.exp(-t * 28.0) * 0.35
        sig += (np.sin(_TWO_PI * 6500.0 * t) + np.sin(_TWO_PI * 8200.0 * t)) * np.exp(-t * 30.0) * 0.12
        pan, gain = 0.35, 0.36
    elif note == DRUM_CONGA_HI:
        t = tt(0.20)
        phase = _TWO_PI * np.cumsum(330.0 * (1.0 + 0.25 * np.exp(-t * 40.0))) / sr
        sig = np.sin(phase) * np.exp(-t * 16.0)
        pan, gain = -0.35, 0.40
    elif note == DRUM_LOW_TOM:
        t = tt(0.30)
        phase = _TWO_PI * np.cumsum(110.0 * (1.0 + 0.5 * np.exp(-t * 20.0))) / sr
        sig = np.sin(phase) * np.exp(-t * 10.0)
        pan, gain = -0.20, 0.45
    else:
        t = tt(0.06)
        sig = _hp(rng.standard_normal(len(t)), 2) * np.exp(-t * 70.0) * 0.4
        pan, gain = 0.2, 0.35
    sig = sig * amp * gain
    left, right = _balance(sig, sig, pan)
    return left.astype(np.float32), right.astype(np.float32)


# =============================================================================
# EFFECTS
# =============================================================================
def _comb(x: np.ndarray, d: int, g: float) -> np.ndarray:
    y = x.copy()
    for s in range(d, len(x), d):
        e = min(len(x), s + d)
        y[s:e] += g * y[s - d : e - d]
    return y


def _allpass(x: np.ndarray, d: int, g: float) -> np.ndarray:
    y = np.zeros_like(x)
    for s in range(0, len(x), d):
        e = min(len(x), s + d)
        if s >= d:
            y[s:e] = -g * x[s:e] + x[s - d : e - d] + g * y[s - d : e - d]
        else:
            y[s:e] = -g * x[s:e]
    return y


def _reverb(left: np.ndarray, right: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Small stereo Schroeder room (4 combs + 2 all-passes per side)."""
    mono = 0.5 * (left + right)
    kernel = np.exp(-np.arange(48) / 7.0)
    kernel /= kernel.sum()
    mono = np.convolve(mono, kernel)[: len(mono)]            # damp highs going into the room
    pre = int(0.018 * sr)
    mono = np.concatenate([np.zeros(pre, dtype=mono.dtype), mono])[: len(left)]
    outs = []
    for delays in ((29.7, 37.1, 41.1, 43.7), (30.9, 38.3, 42.5, 45.3)):
        acc = np.zeros_like(mono)
        for ms in delays:
            acc += _comb(mono, int(ms * 0.001 * sr), 0.84)
        acc *= 0.25
        for ms, g in ((5.0, 0.5), (1.7, 0.5)):
            acc = _allpass(acc, int(ms * 0.001 * sr), g)
        outs.append(acc)
    return outs[0], outs[1]


# =============================================================================
# HELPERS SHARED WITH THE LIVE BACKEND
# =============================================================================
def to_int16_stereo(left: np.ndarray, right: np.ndarray, scale: float = 30000.0) -> np.ndarray:
    pcm = np.empty((len(left), 2), dtype="<i2")
    pcm[:, 0] = np.clip(left * scale, -32767, 32767)
    pcm[:, 1] = np.clip(right * scale, -32767, 32767)
    return np.ascontiguousarray(pcm)


def render_channel_note_pcm(
    channel: int,
    midi_note: int,
    sr: int = 22050,
    gate_sec: float = 1.2,
    velocity: int = 100,
    brightness: float = 0.6,
) -> np.ndarray:
    """One note for the real-time mixer backend (``(n, 2)`` int16)."""
    if channel == DRUM_CHANNEL:
        left, right = render_drum_stereo(midi_note, velocity, sr)
    else:
        timbre = CHANNEL_TIMBRE.get(channel, "organ_pad")
        left, right = render_note_stereo(
            timbre, midi_to_freq(midi_note), gate_sec, velocity, brightness, sr,
            pan=CHANNEL_PAN.get(channel, 0.0),
        )
    return to_int16_stereo(left, right, 28000.0)


# =============================================================================
# OFFLINE RENDERER
# =============================================================================
_BUS_OF_ROLE = {"pad": "pad", "warm_pad": "pad", "church": "pad", "bass": "bass", "drum": "drums"}
_BUS_GAIN = {"pad": 1.0, "bass": 1.0, "lead": 1.0, "drums": 1.0}
_REVERB_SEND = {"pad": 0.50, "bass": 0.0, "lead": 0.55, "drums": 0.12}
_ROOM = {"tight": 0.30, "mid": 0.55, "wide": 0.90}


def _silent_wav(sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(np.zeros(int(0.25 * sr) * 2, dtype=np.int16).tobytes())
    return buf.getvalue()


def render_sonification(
    analysis: AnalysisResult,
    bar_index: int = 0,
    bars: int = 1,
    sample_rate: int = 44100,
    ambient_drums: bool = True,
    tail_sec: float = 0.6,
    loop: bool = False,
    sidechain: bool = True,
    include_events: bool = False,
) -> tuple[bytes, dict]:
    """Render ``bars`` bars as stereo WAV bytes plus an explanatory report.

    ``loop=True`` folds the reverb tail back onto the start so the clip repeats seamlessly.
    """
    bars = max(1, min(4, int(bars)))
    sr = max(8000, min(48000, int(sample_rate)))
    total_steps = bars * 16
    main_n = int(round(total_steps * SIXTEENTH_DURATION_SEC * sr))

    if not analysis.touching:
        return _silent_wav(sr), {"touching": False, "explanation": describe_mapping(analysis, ambient_drums)}

    seq = StepSequencer()
    scheduled: list[tuple[int, NoteEvent]] = []
    chords: list[str] = []
    for k in range(total_steps):
        plan = seq.plan(analysis, bar_index * 16 + k, drums_on=ambient_drums)
        if k % 16 == 0:
            chords.append(plan.chord_name)
        for ev in plan.events:
            scheduled.append((k + ev.delay_steps, ev))

    tail = int((max(tail_sec, 1.0) if loop else max(0.0, tail_sec)) * sr)
    n = main_n + tail
    buses = {name: [np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)] for name in _BUS_GAIN}
    kick_starts: list[int] = []
    counts: dict[str, int] = {}
    cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    size = int(getattr(analysis, "sensor_size", 40))
    room = _ROOM["tight" if size < 30 else ("wide" if size > 60 else "mid")]
    inten = _clamp01(analysis.intensity)
    swirl = _clamp01(getattr(analysis, "swirliness", 0.0))
    color_bias = COLOR_BRIGHTNESS_BIAS.get(analysis.dominant_color, 0) / 127.0
    brightness = _clamp01(0.30 + 0.55 * inten + color_bias)

    for step, ev in scheduled:
        start = int(round(step * SIXTEENTH_DURATION_SEC * sr))
        if start >= n:
            continue
        counts[ev.role] = counts.get(ev.role, 0) + 1
        bus = buses[_BUS_OF_ROLE.get(ev.role, "lead")]
        if ev.role == "drum":
            key = ("d", ev.note, ev.velocity, start % 4)
            if key not in cache:
                cache[key] = render_drum_stereo(ev.note, ev.velocity, sr, seed=start % 4)
            left, right = cache[key]
            if ev.note == DRUM_KICK:
                kick_starts.append(start)
        else:
            timbre = CHANNEL_TIMBRE.get(ev.channel, "organ_pad")
            gate = ev.steps * SIXTEENTH_DURATION_SEC * (0.92 if ev.role in ("arp", "pluck", "bass") else 1.0)
            vib = swirl if ev.role in ("arp", "voice", "harmony") else 0.0
            key = (timbre, ev.note, round(gate, 3), ev.velocity, round(brightness, 2), round(vib, 2), ev.pan)
            if key not in cache:
                cache[key] = render_note_stereo(
                    timbre, midi_to_freq(ev.note), max(0.05, gate), ev.velocity, brightness, sr, vib, ev.pan
                )
            left, right = cache[key]
        end = min(n, start + len(left))
        bus[0][start:end] += left[: end - start]
        bus[1][start:end] += right[: end - start]

    # French-house "pump": duck the pad bed on every kick so the downbeat is felt.
    if sidechain and kick_starts:
        duck = np.ones(n, dtype=np.float32)
        seg_n = int(0.30 * sr)
        seg_t = np.arange(seg_n) / sr
        curve = (1.0 - 0.38 * np.exp(-seg_t / 0.11)).astype(np.float32)
        for s in kick_starts:
            e = min(n, s + seg_n)
            duck[s:e] = np.minimum(duck[s:e], curve[: e - s])
        buses["pad"][0] *= duck
        buses["pad"][1] *= duck

    dry_l = sum(buses[k][0] * _BUS_GAIN[k] for k in buses)
    dry_r = sum(buses[k][1] * _BUS_GAIN[k] for k in buses)
    send_l = sum(buses[k][0] * _REVERB_SEND[k] for k in buses)
    send_r = sum(buses[k][1] * _REVERB_SEND[k] for k in buses)
    wet_l, wet_r = _reverb(send_l, send_r, sr)
    left = dry_l + room * wet_l
    right = dry_r + room * wet_r

    if loop:
        fold = min(tail, main_n)
        left[:fold] += left[main_n : main_n + fold]
        right[:fold] += right[main_n : main_n + fold]
        left, right = left[:main_n], right[:main_n]

    # Master: gentle soft-clip (keeps relative loudness; no peak-normalising)
    left = np.tanh(left * 1.05).astype(np.float32) * 0.95
    right = np.tanh(right * 1.05).astype(np.float32) * 0.95
    ramp = min(len(left), int(0.003 * sr))
    if ramp > 0 and not loop:
        left[:ramp] *= np.linspace(0.0, 1.0, ramp, dtype=np.float32)
        right[:ramp] *= np.linspace(0.0, 1.0, ramp, dtype=np.float32)
    fade = min(len(left), int(0.04 * sr))
    if fade > 0 and not loop:
        left[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        right[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(to_int16_stereo(left, right, 32000.0).tobytes())

    peak = float(max(np.max(np.abs(left)), np.max(np.abs(right)), 1e-9))
    rms = float(np.sqrt(np.mean(0.5 * (left**2 + right**2)) + 1e-12))
    info = describe_mapping(analysis, ambient_drums)
    report = {
        "touching": True,
        "bpm": BPM,
        "bars": bars,
        "start_bar": bar_index % len(VERIDIS_QUO_PROGRESSION),
        "chords": chords,
        "sample_rate": sr,
        "channels": 2,
        "duration_sec": round(len(left) / sr, 3),
        "looped_seamlessly": bool(loop),
        "activity_tier": info["tier_name"],
        "layers": info["layers"],
        "color_voice": COLOR_INSTRUMENT_NAME.get(analysis.dominant_color, "pad"),
        "note_counts": counts,
        "peak_dbfs": round(20.0 * math.log10(peak), 1),
        "rms_dbfs": round(20.0 * math.log10(rms), 1),
        "explanation": info,
    }
    if include_events:
        report["events"] = [
            {"step": s, "role": e.role, "channel": e.channel, "note": e.note,
             "velocity": e.velocity, "steps": e.steps}
            for s, e in scheduled
        ]
    return buf.getvalue(), report


def render_sonification_wav(
    analysis: AnalysisResult,
    bar_index: int = 0,
    bars: int = 1,
    sample_rate: int = 44100,
    ambient_drums: bool = True,
    **kwargs,
) -> bytes:
    """Render ``bars`` of quantized C-minor Veridis Quo audio as stereo RIFF WAV bytes."""
    return render_sonification(analysis, bar_index, bars, sample_rate, ambient_drums, **kwargs)[0]


__all__ = [
    "BPM",
    "ARP_CHANNEL",
    "BASS_CHANNEL",
    "DRUM_CHANNEL",
    "render_sonification_wav",
    "render_sonification",
    "render_note_stereo",
    "render_drum_stereo",
    "render_channel_note_pcm",
    "to_int16_stereo",
]