"""Pure-NumPy WAV synthesizer for on-demand coordinate-to-music rendering."""

from __future__ import annotations

import io
import math
import wave

import numpy as np

from earth_jukebox.audio.theory import (
    ARP_CHANNEL,
    BASS_CHANNEL,
    BPM,
    DRUM_CHANNEL,
    DRUM_CLAP,
    DRUM_CONGA_HI,
    DRUM_HAT_OPEN,
    DRUM_KICK,
    DRUM_LOW_TOM,
    DRUM_RIM,
    SIXTEENTH_DURATION_SEC,
    VERIDIS_QUO_PROGRESSION,
    midi_to_freq,
    plan_ambient_drums,
    plan_baroque_arpeggio,
    plan_bass_step,
    plan_chord_voicing,
    plan_musical_step,
    plan_spottiness_accents,
    select_chord_harmony_note,
    should_trigger_data_step,
    snap_to_c_minor,
)
from earth_jukebox.models import AnalysisResult


def _add_tone(
    mix: np.ndarray,
    start_sample: int,
    duration_sec: float,
    freq_hz: float,
    velocity: int,
    timbre: str,
    sample_rate: int,
) -> None:
    """Synthesize a single note with an analog envelope directly into ``mix``."""
    n_samples = max(1, int(round(duration_sec * sample_rate)))
    end_sample = min(len(mix), start_sample + n_samples)
    if end_sample <= start_sample:
        return

    count = end_sample - start_sample
    t = np.arange(count, dtype=np.float32) / float(sample_rate)
    phase = 2.0 * math.pi * float(freq_hz) * t
    amp = max(0.02, min(1.0, velocity / 127.0))

    if timbre == "arp":
        wave_sig = 0.55 * np.tanh(2.1 * np.sin(phase)) + 0.30 * np.sin(2.0 * phase)
        env = np.minimum(1.0, t / 0.012) * np.exp(-t * 2.5)
        gain = 0.16
    elif timbre == "bass":
        wave_sig = 0.72 * np.sin(phase) + 0.28 * np.tanh(1.8 * np.sin(2.0 * phase))
        env = np.minimum(1.0, t / 0.010) * np.exp(-t * 2.6)
        gain = 0.24
    elif timbre == "pluck":
        wave_sig = 0.70 * np.sin(phase) + 0.30 * np.sin(3.0 * phase) * np.exp(-t * 8.0)
        env = np.minimum(1.0, t / 0.005) * np.exp(-t * 5.2)
        gain = 0.15
    else:
        # Warm drawbar organ / pad blend
        wave_sig = (
            0.64 * np.sin(phase)
            + 0.24 * np.sin(2.0 * phase)
            + 0.12 * np.sin(3.0 * phase)
        )
        attack = min(0.045, duration_sec * 0.25)
        tail = np.maximum(0.0, (count - np.arange(count)) / (0.06 * sample_rate))
        release = np.minimum(1.0, tail)
        env = np.minimum(1.0, t / max(0.005, attack)) * release
        gain = 0.14

    mix[start_sample:end_sample] += (wave_sig * env * amp * gain).astype(np.float32)


def _add_drum_hit(
    mix: np.ndarray,
    start_sample: int,
    drum_note: int,
    velocity: int,
    sample_rate: int,
) -> None:
    """Synthesize lightweight ambient percussion into ``mix``."""
    amp = max(0.02, min(1.0, velocity / 127.0)) * 0.18
    if drum_note == DRUM_KICK:
        dur = 0.18
        n = min(len(mix) - start_sample, int(dur * sample_rate))
        if n <= 0:
            return
        t = np.arange(n, dtype=np.float32) / float(sample_rate)
        freq = 52.0 + 70.0 * np.exp(-t * 32.0)
        sig = np.sin(2.0 * math.pi * freq * t) * np.exp(-t * 14.0) * 1.35
    elif drum_note in (DRUM_RIM, DRUM_CLAP, DRUM_CONGA_HI, DRUM_LOW_TOM):
        dur = 0.11
        n = min(len(mix) - start_sample, int(dur * sample_rate))
        if n <= 0:
            return
        t = np.arange(n, dtype=np.float32) / float(sample_rate)
        base = 220.0 if drum_note in (DRUM_RIM, DRUM_CLAP) else 140.0
        rng = np.random.default_rng(seed=drum_note + start_sample)
        noise = rng.standard_normal(n).astype(np.float32) * 0.45
        sig = (0.65 * np.sin(2.0 * math.pi * base * t) + noise) * np.exp(-t * 24.0)
    else:
        # Hi-hats, shaker, tambourine
        dur = 0.09 if drum_note == DRUM_HAT_OPEN else 0.045
        n = min(len(mix) - start_sample, int(dur * sample_rate))
        if n <= 0:
            return
        t = np.arange(n, dtype=np.float32) / float(sample_rate)
        rng = np.random.default_rng(seed=drum_note + start_sample)
        noise = rng.standard_normal(n).astype(np.float32)
        sig = noise * np.exp(-t * (28.0 if drum_note == DRUM_HAT_OPEN else 65.0)) * 0.55

    mix[start_sample : start_sample + n] += (sig * amp).astype(np.float32)


def render_sonification_wav(
    analysis: AnalysisResult,
    bar_index: int = 0,
    bars: int = 1,
    sample_rate: int = 44100,
    ambient_drums: bool = True,
) -> bytes:
    """Render ``bars`` of quantized C-minor Veridis Quo audio as RIFF WAV bytes."""
    bars_clamped = max(1, min(4, int(bars)))
    sr = max(8000, min(48000, int(sample_rate)))
    total_steps = bars_clamped * 16
    total_sec = total_steps * SIXTEENTH_DURATION_SEC + 0.18
    total_samples = int(math.ceil(total_sec * sr))
    mix = np.zeros(total_samples, dtype=np.float32)

    if not analysis.touching:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(np.zeros(int(0.25 * sr), dtype=np.int16).tobytes())
        return buf.getvalue()

    sensor_size = int(getattr(analysis, "sensor_size", 40))
    data_step_idx = 0

    for step_offset in range(total_steps):
        current_bar = (bar_index + (step_offset // 16)) % len(VERIDIS_QUO_PROGRESSION)
        step_16 = step_offset % 16
        start_sample = int(round(step_offset * SIXTEENTH_DURATION_SEC * sr))

        if step_16 == 0:
            _cname, _bass, pad_notes = plan_chord_voicing(
                current_bar,
                analysis.intensity,
                sensor_size=sensor_size,
                dominant_color=analysis.dominant_color,
            )
            pad_vel = int(round(36 + analysis.intensity * 44))
            pad_dur = SIXTEENTH_DURATION_SEC * 15.0
            for note in pad_notes:
                _add_tone(mix, start_sample, pad_dur, midi_to_freq(note), pad_vel, "pad", sr)

        bass_hits = plan_bass_step(
            current_bar,
            analysis.density,
            analysis.intensity,
            sensor_size,
            step_16,
        )
        if bass_hits:
            bass_dur = SIXTEENTH_DURATION_SEC * (6.0 if analysis.density < 0.25 else 2.2)
            for b_note, bass_vel in bass_hits:
                _add_tone(mix, start_sample, bass_dur, midi_to_freq(b_note), bass_vel, "bass", sr)

        if ambient_drums:
            for d_note, d_vel in plan_ambient_drums(
                analysis.density,
                analysis.intensity,
                step_16,
                spottiness=analysis.spottiness,
            ):
                _add_drum_hit(mix, start_sample, d_note, d_vel, sr)

        arp_hit = plan_baroque_arpeggio(analysis, current_bar, step_16, step_offset)
        if arp_hit is not None:
            arp_note, arp_vel, arp_gate = arp_hit
            arp_dur = max(0.06, arp_gate * SIXTEENTH_DURATION_SEC * 0.90)
            _add_tone(
                mix,
                start_sample,
                arp_dur,
                midi_to_freq(snap_to_c_minor(arp_note)),
                arp_vel,
                "arp",
                sr,
            )

        spot_hit = plan_spottiness_accents(analysis, current_bar, step_16)
        if spot_hit is not None:
            spot_note, spot_vel = spot_hit
            _add_tone(
                mix,
                start_sample,
                SIXTEENTH_DURATION_SEC * 0.95,
                midi_to_freq(snap_to_c_minor(spot_note)),
                spot_vel,
                "pluck",
                sr,
            )

        if should_trigger_data_step(analysis.density, step_16) or step_offset == 0:
            _ch, pitch, velocity, _vol, _motif, dur_sec = plan_musical_step(
                analysis, data_step_idx, bar_index=current_bar
            )
            data_step_idx += 1
            q_pitch = snap_to_c_minor(pitch)
            render_dur = min(dur_sec, SIXTEENTH_DURATION_SEC * 14.0)
            _add_tone(mix, start_sample, render_dur, midi_to_freq(q_pitch), velocity, "pad", sr)

            if step_16 % 4 == 0 and sensor_size >= 30:
                harmony_pitch = snap_to_c_minor(select_chord_harmony_note(q_pitch, current_bar))
                if harmony_pitch != q_pitch:
                    _add_tone(
                        mix,
                        start_sample,
                        render_dur,
                        midi_to_freq(harmony_pitch),
                        max(24, int(velocity * 0.65)),
                        "pad",
                        sr,
                    )

    # Soft limiter / normalize to avoid clipping
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.90:
        mix = mix * (0.90 / peak)
    # Gentle tail fade-out over the last 40 ms
    fade_len = min(len(mix), int(0.04 * sr))
    if fade_len > 0:
        mix[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)

    pcm = np.clip(mix * 28000.0, -32767, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


__all__ = ["BPM", "ARP_CHANNEL", "BASS_CHANNEL", "DRUM_CHANNEL", "render_sonification_wav"]
