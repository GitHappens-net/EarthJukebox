"""Veridis Quo-inspired C Natural Minor music theory and CV sonification mapping.

All pitches, chords, arpeggios, basslines, and syncopated counter-melodies are
strictly quantized to **C Natural Minor** (``C, D, Eb, F, G, Ab, Bb`` ->
pitch classes ``{0, 2, 3, 5, 7, 8, 10}``) on a locked **75 BPM** 16th-note grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

try:
    import numpy as np
except ImportError:  # pragma: no cover - fallback only if numpy absent
    np = None  # type: ignore[assignment]

try:
    from backend.earth_jukebox.models import AnalysisResult
except ImportError:  # pragma: no cover - standalone fallback prior to scaffold merge
    @dataclass(frozen=True)
    class AnalysisResult:  # type: ignore[no-redef]
        """Local fallback matching ``earth_jukebox.models.AnalysisResult``."""

        dominant_color: str
        intensity: float
        density: float
        shape: str = "uniform"
        shape_score: float = 0.0
        spottiness: float = 0.0
        swirliness: float = 0.0
        sensor_size: int = 40
        sensor_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
        cursor_xy: tuple[int, int] = (0, 0)
        touching: bool = True


# =============================================================================
# MASTER CLOCK & C NATURAL MINOR SCALE CONSTANTS
# =============================================================================
BPM: float = 75.0
BEAT_DURATION_SEC: float = 60.0 / BPM                 # 0.8s per quarter beat
SIXTEENTH_DURATION_SEC: float = BEAT_DURATION_SEC / 4.0  # 0.2s per 16th step
FIXED_VELOCITY: int = 88
FIXED_VOLUME: float = 0.65

# Strict C Natural Minor pitch classes: C, D, Eb, F, G, Ab, Bb
C_MINOR_PITCH_CLASSES: frozenset[int] = frozenset({0, 2, 3, 5, 7, 8, 10})
C_MINOR_SORTED_PCS: tuple[int, ...] = (0, 2, 3, 5, 7, 8, 10)
NOTE_NAMES: tuple[str, ...] = (
    "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"
)

# MIDI Channel Assignment (0..15, Channel 9 reserved for GM Percussion)
PAD_CHANNEL = 0            # Drawbar Organ (GM 16)
BASS_CHANNEL = 6           # Synth Bass 1 (GM 38)
ARP_CHANNEL = 8            # Lead 1 Square (GM 80) - Veridis Quo baroque lead
DRUM_CHANNEL = 9           # GM Percussion
WARM_PAD_CHANNEL = 10      # Warm Pad (GM 89)
PLUCK_CHANNEL = 11         # Marimba (GM 12) - Spottiness syncopated counter-accents
CHURCH_ORGAN_CHANNEL = 12  # Church Organ (GM 19) - Wide-cursor harmonic swell

# General MIDI Percussion Note Numbers (Channel 9)
DRUM_KICK = 36
DRUM_RIM = 37
DRUM_CLAP = 39
DRUM_HAT_CLOSED = 42
DRUM_LOW_TOM = 45
DRUM_HAT_OPEN = 46
DRUM_TAMBOURINE = 54
DRUM_CONGA_HI = 63
DRUM_SHAKER = 70

# Color -> strictly quantized C-Minor MIDI pitch palette (cool low -> warm high)
COLOR_PITCH_MAP: dict[str, tuple[int, ...]] = {
    "Black": (36, 43, 48, 51, 55),          # C2, G2, C3, Eb3, G3
    "Blue": (48, 51, 55, 58, 60),           # C3, Eb3, G3, Bb3, C4
    "Green": (48, 53, 55, 58, 63, 67),      # C3, F3, G3, Bb3, Eb4, G4
    "White": (60, 67, 70, 72, 75),          # C4, G4, Bb4, C5, Eb5
    "Orange": (62, 63, 68, 70, 74),         # D4, Eb4, Ab4, Bb4, D5
    "Red": (74, 75, 79, 80, 82, 84),        # D5, Eb5, G5, Ab5, Bb5, C6
}

# Color -> octave register anchor relative to C4 (60)
COLOR_OCTAVE_OFFSET: dict[str, int] = {
    "Black": -12,
    "Blue": -12,
    "Green": 0,
    "White": 0,
    "Orange": 0,
    "Red": 12,
}

# Color -> (MIDI Channel, General MIDI Program)
COLOR_CHANNEL_PROGRAM: dict[str, tuple[int, int]] = {
    "Black": (1, 89),   # Warm Pad
    "Blue": (2, 88),    # New Age Pad
    "Green": (3, 12),   # Marimba
    "White": (4, 11),   # Vibraphone
    "Orange": (5, 4),   # Electric Piano 1 (Rhodes)
    "Red": (7, 81),     # Lead 2 (Sawtooth)
}

# Warm colors add a controlled tension tone to the active chord while remaining
# inside the shared C-natural-minor palette.
COLOR_WARNING_INTERVALS: dict[str, int] = {
    "Orange": 10,  # minor 7th
    "Red": 14,     # natural 9th
}


@dataclass(frozen=True)
class ChordVoicing:
    """One bar in the 8-bar Veridis Quo-inspired C Natural Minor progression."""

    name: str
    bass_note: int
    pad_tones: tuple[int, ...]
    arp_tones: tuple[int, ...]


# 8-Bar Baroque Circle-of-Fifths C-Minor Progression (Veridis Quo inspired)
# i -> iv -> bVII -> bIII -> bVI -> iv9 -> iiø7 -> v7 (100% C Natural Minor)
VERIDIS_QUO_PROGRESSION: tuple[ChordVoicing, ...] = (
    ChordVoicing("Cm9", 36, (48, 51, 55, 58, 62), (60, 63, 67, 70, 74, 75)),
    ChordVoicing("Fm7", 41, (48, 53, 56, 60, 63), (65, 68, 72, 75, 77, 72)),
    ChordVoicing("Bb9", 46, (46, 50, 53, 56, 60), (58, 62, 65, 68, 72, 74)),
    ChordVoicing("Ebmaj7", 39, (51, 55, 58, 62, 63), (63, 67, 70, 74, 75, 79)),
    ChordVoicing("Abmaj7", 44, (48, 51, 56, 60, 67), (68, 72, 75, 79, 80, 75)),
    ChordVoicing("Fm9", 41, (48, 53, 56, 62, 63), (65, 68, 72, 74, 77, 75)),
    ChordVoicing("Dm7b5", 38, (50, 53, 56, 60, 65), (62, 65, 68, 72, 74, 68)),
    ChordVoicing("Gm7", 43, (50, 53, 55, 58, 62), (67, 70, 74, 77, 79, 74)),
)

# Compatibility tuple view: (chord_name, bass_note, pad_tones)
HARMONIC_PROGRESSION: tuple[tuple[str, int, tuple[int, ...]], ...] = tuple(
    (chord.name, chord.bass_note, chord.pad_tones) for chord in VERIDIS_QUO_PROGRESSION
)

# Shape-driven melodic contour patterns (indices into 6-note sorted pitch pool)
SHAPE_CONTOUR_PATTERNS: dict[str, tuple[int, ...]] = {
    "uniform": (0, 0, 2, 0, 2, 2, 0, 2),
    "linear": (0, 1, 2, 3, 4, 5, 4, 3),
    "circular": (0, 2, 4, 2, 1, 3, 5, 3),
    "organic": (0, 3, 1, 4, 5, 2, 4, 1),
}


def _clamp01(value: float) -> float:
    if np is not None:
        return float(np.clip(value, 0.0, 1.0))
    return max(0.0, min(1.0, float(value)))


def snap_to_c_minor(midi_note: int) -> int:
    """Quantize any MIDI pitch ``0..127`` strictly onto C Natural Minor."""
    clamped = max(0, min(int(midi_note), 127))
    if (clamped % 12) in C_MINOR_PITCH_CLASSES:
        return clamped
    for offset in range(1, 12):
        down = clamped - offset
        if down >= 0 and (down % 12) in C_MINOR_PITCH_CLASSES:
            return down
        up = clamped + offset
        if up <= 127 and (up % 12) in C_MINOR_PITCH_CLASSES:
            return up
    return 60


def shift_c_minor_scale_degree(midi_note: int, degree_delta: int) -> int:
    """Move a C-minor pitch by ``degree_delta`` scale steps within C Natural Minor."""
    q = snap_to_c_minor(midi_note)
    if degree_delta == 0:
        return q
    octave = q // 12
    pc = q % 12
    idx = C_MINOR_SORTED_PCS.index(pc)
    target_total = octave * 7 + idx + degree_delta
    target_oct, target_idx = divmod(target_total, 7)
    raw = target_oct * 12 + C_MINOR_SORTED_PCS[target_idx]
    return snap_to_c_minor(raw)


def midi_to_note_name(midi_note: int) -> str:
    """Convert a MIDI pitch integer into note + octave notation (e.g. ``Eb4``)."""
    octave = (int(midi_note) // 12) - 1
    return f"{NOTE_NAMES[int(midi_note) % 12]}{octave}"


def midi_to_freq(midi_note: int) -> float:
    """Convert a quantized MIDI pitch to frequency in Hz (A4 = 440 Hz)."""
    return 440.0 * (2.0 ** ((int(midi_note) - 69) / 12.0))


def select_rhythm(density: float) -> tuple[str, float]:
    """Map visual edge density ``[0.0, 1.0]`` to rhythmic subdivision in beats."""
    d = _clamp01(density)
    if d < 0.12:
        return "Whole Note", 4.0
    if d < 0.30:
        return "Half Note", 2.0
    if d <= 0.70:
        return "8th Note", 0.5
    return "16th Arpeggio", 0.25


def should_trigger_data_step(density: float, step_16: int) -> bool:
    """Return True when the data melody falls cleanly on the 75 BPM beat grid."""
    step = step_16 % 16
    d = _clamp01(density)
    if d < 0.12:
        return step == 0
    if d < 0.30:
        return step in (0, 8)
    if d <= 0.70:
        return step % 2 == 0
    return True


def intensity_to_velocity(intensity: float) -> int:
    """Return a stable MIDI velocity; intensity no longer changes loudness."""
    return FIXED_VELOCITY


def intensity_to_volume(intensity: float) -> float:
    """Return a stable normalized amplitude; intensity no longer changes loudness."""
    return FIXED_VOLUME


def _color_warning_tone(dominant_color: str, chord: ChordVoicing) -> int | None:
    """Return the warm-color tension tone above the active chord root."""
    interval = COLOR_WARNING_INTERVALS.get(dominant_color)
    if interval is None:
        return None
    return snap_to_c_minor(chord.bass_note + interval + 24)


def select_chord_harmony_note(midi_note: int, bar_index: int = 0) -> int:
    """Choose a consonant chord tone near ``midi_note`` for the harmony voice."""
    chord = VERIDIS_QUO_PROGRESSION[bar_index % len(VERIDIS_QUO_PROGRESSION)]
    target = snap_to_c_minor(midi_note)
    return min((snap_to_c_minor(note) for note in chord.pad_tones), key=lambda note: abs(note - target))


def plan_chord_voicing(
    bar_index: int,
    intensity: float,
    sensor_size: int = 40,
    dominant_color: str = "Blue",
) -> tuple[str, int, tuple[int, ...]]:
    """Return ``(chord_name, bass_note, active_pad_tones)`` shaped by intensity & cursor size.

    * Small cursor (``sensor_size < 30``) narrows the voicing to an intimate 2-note dyad.
    * Standard cursor (``30 <= sensor_size <= 60``) plays 3–4 chord tones scaled by intensity.
    * Wide cursor (``sensor_size > 60``) opens lush 5–6 note spread voicings with octave shimmer.
    """
    chord = VERIDIS_QUO_PROGRESSION[bar_index % len(VERIDIS_QUO_PROGRESSION)]
    i = _clamp01(intensity)
    size = int(sensor_size)

    if size < 30:
        count = 2
        tones = [snap_to_c_minor(t) for t in chord.pad_tones[:count]]
    elif size <= 60:
        count = 3 if i < 0.40 else (4 if i < 0.75 else 5)
        tones = [snap_to_c_minor(t) for t in chord.pad_tones[:count]]
    else:
        tones = [snap_to_c_minor(t) for t in chord.pad_tones]
        # Wide cursor adds airy upper-octave harmonic extension
        tones.append(snap_to_c_minor(chord.pad_tones[1] + 12))
        if i >= 0.55:
            tones.append(snap_to_c_minor(chord.pad_tones[2] + 12))

    warning_tone = _color_warning_tone(dominant_color, chord)
    if warning_tone is not None:
        tones.append(warning_tone)

    return chord.name, snap_to_c_minor(chord.bass_note), tuple(dict.fromkeys(tones))


def plan_bass_step(
    bar_index: int,
    density: float,
    intensity: float,
    sensor_size: int,
    step_16: int,
) -> list[tuple[int, int]]:
    """Plan pulsing French-touch analog synth bass notes on ``step_16``.

    Returns a list of ``(midi_note, velocity)`` pairs.
    """
    chord = VERIDIS_QUO_PROGRESSION[bar_index % len(VERIDIS_QUO_PROGRESSION)]
    root = snap_to_c_minor(chord.bass_note)
    octave_up = snap_to_c_minor(root + 12)
    fifth = snap_to_c_minor(root + 7)
    step = step_16 % 16
    d = _clamp01(density)
    base_vel = FIXED_VELOCITY

    hits: list[tuple[int, int]] = []
    if d < 0.25:
        if step == 0:
            hits.append((root, base_vel))
        elif step == 8 and sensor_size > 60:
            hits.append((octave_up, max(30, base_vel - 10)))
    elif d <= 0.70:
        # Classic French-touch 8th-note root/octave pulse
        if step % 2 == 0:
            pitch = root if (step % 4 == 0) else octave_up
            vel = base_vel if (step % 4 == 0) else max(30, int(base_vel * 0.78))
            hits.append((pitch, vel))
    else:
        # Energetic 16th-note octave/fifth groove
        if step in (0, 3, 4, 6, 8, 10, 12, 14, 15):
            if step in (0, 4, 8, 12):
                pitch = root
                vel = base_vel
            elif step in (6, 14):
                pitch = fifth
                vel = max(30, int(base_vel * 0.80))
            else:
                pitch = octave_up
                vel = max(30, int(base_vel * 0.72))
            hits.append((pitch, vel))

    # Wide cursor adds deep sub-octave doubling on bar downbeat
    if sensor_size > 60 and step == 0:
        sub = snap_to_c_minor(root - 12)
        if sub != root and sub >= 24:
            hits.append((sub, max(28, int(base_vel * 0.75))))

    return hits


def _build_color_chord_pool(dominant_color: str, bar_index: int) -> tuple[int, ...]:
    """Blend the color register palette with the active Veridis Quo chord tones."""
    chord = VERIDIS_QUO_PROGRESSION[bar_index % len(VERIDIS_QUO_PROGRESSION)]
    color_palette = COLOR_PITCH_MAP.get(dominant_color, COLOR_PITCH_MAP["Blue"])
    oct_shift = COLOR_OCTAVE_OFFSET.get(dominant_color, 0)

    blended = [
        snap_to_c_minor(tone + oct_shift)
        for tone in chord.arp_tones
    ]
    # Anchor the lowest and highest register bounds to the color's pitch range
    merged = sorted({*color_palette[:3], *blended[:4], color_palette[-1]})
    warning_tone = _color_warning_tone(dominant_color, chord)
    if warning_tone is not None:
        merged = sorted({*merged, warning_tone})
    return tuple(merged)


def plan_musical_step(
    analysis: AnalysisResult,
    step_index: int,
    bar_index: int = 0,
) -> tuple[int, int, int, float, str, float]:
    """Determine channel, quantized pitch, velocity, volume, motif, and duration.

    Consumes ``dominant_color``, ``density``, ``intensity``, ``shape``,
    ``swirliness``, ``spottiness``, and ``sensor_size`` while staying strictly
    in C Natural Minor.
    """
    motif_name, beats = select_rhythm(analysis.density)
    step_duration_sec = beats * BEAT_DURATION_SEC
    pool = _build_color_chord_pool(analysis.dominant_color, bar_index)

    shape = getattr(analysis, "shape", "uniform") or "uniform"
    pattern = SHAPE_CONTOUR_PATTERNS.get(shape, SHAPE_CONTOUR_PATTERNS["circular"])

    if shape == "uniform" and analysis.density < 0.30:
        raw_pitch = pool[0] if (step_index % 2 == 0) else pool[min(2, len(pool) - 1)]
    else:
        idx = pattern[step_index % len(pattern)] % len(pool)
        raw_pitch = pool[idx]

    swirliness = _clamp01(getattr(analysis, "swirliness", 0.0))
    if swirliness >= 0.45:
        # Sweeping baroque pendulum turns around the chord tone
        turn_cycle = (0, 1, 0, -1, 1, 2, 0, -1)
        degree_shift = turn_cycle[step_index % len(turn_cycle)]
        if swirliness < 0.72:
            degree_shift = max(-1, min(1, degree_shift))
        raw_pitch = shift_c_minor_scale_degree(raw_pitch, degree_shift)

    quantized_pitch = snap_to_c_minor(raw_pitch)
    channel, _ = COLOR_CHANNEL_PROGRAM.get(analysis.dominant_color, (2, 88))
    velocity = intensity_to_velocity(analysis.intensity)

    # Expressive accent modulation for organic shape & spottiness
    spottiness = _clamp01(getattr(analysis, "spottiness", 0.0))
    if shape == "organic" and step_index % 3 == 0:
        velocity = min(127, velocity + 10)
    if spottiness >= 0.50 and step_index % 2 == 1:
        velocity = max(28, velocity - 12)

    sensor_size = int(getattr(analysis, "sensor_size", 40))
    if sensor_size < 30:
        # Intimate crisp staccato shortening
        step_duration_sec = min(step_duration_sec, BEAT_DURATION_SEC * 0.5)
    elif spottiness >= 0.55:
        step_duration_sec = min(step_duration_sec, SIXTEENTH_DURATION_SEC * 1.5)

    volume = intensity_to_volume(analysis.intensity)
    return channel, quantized_pitch, velocity, volume, motif_name, step_duration_sec


def plan_baroque_arpeggio(
    analysis: AnalysisResult,
    bar_index: int,
    step_16: int,
    step_index: int,
) -> Optional[tuple[int, int, int]]:
    """Plan the signature Veridis Quo square-lead baroque arpeggio note.

    Returns ``(midi_note, velocity, gate_steps)`` or ``None`` on rest steps.
    """
    chord = VERIDIS_QUO_PROGRESSION[bar_index % len(VERIDIS_QUO_PROGRESSION)]
    d = _clamp01(analysis.density)
    i = _clamp01(analysis.intensity)
    shape = getattr(analysis, "shape", "uniform") or "uniform"
    swirl = _clamp01(getattr(analysis, "swirliness", 0.0))
    spot = _clamp01(getattr(analysis, "spottiness", 0.0))
    sensor_size = int(getattr(analysis, "sensor_size", 40))

    # Sparse uniform regions let the warm organ pad breathe without busy lead
    if shape == "uniform" and d < 0.25 and (step_16 % 8 != 0):
        return None
    if d < 0.12 and step_16 not in (0, 8):
        return None
    if d <= 0.65 and (step_16 % 2 != 0) and swirl < 0.60:
        return None

    arp = chord.arp_tones
    if shape == "linear":
        seq = (0, 1, 2, 3, 4, 5, 4, 3)
        note = arp[seq[step_index % len(seq)] % len(arp)]
    elif shape == "organic":
        seq = (0, 3, 2, 5, 1, 4, 2, 3)
        note = arp[seq[step_index % len(seq)] % len(arp)]
    elif shape == "uniform":
        seq = (0, 2, 0, 2, 4, 2, 0, 2)
        note = arp[seq[step_index % len(seq)] % len(arp)]
    else:
        # "circular": classic Veridis Quo cascading baroque turn
        seq = (0, 2, 4, 3, 2, 1, 2, 4)
        note = arp[seq[step_index % len(seq)] % len(arp)]

    if swirl >= 0.45:
        turn = (0, 1, 2, 1, 0, -1, 0, 1)[step_16 % 8]
        note = shift_c_minor_scale_degree(note, turn)

    # Shift upper octave slightly for warm colors
    if analysis.dominant_color in ("Red", "Orange") and note < 72:
        note = snap_to_c_minor(note + 12)
    elif analysis.dominant_color == "Black" and note >= 68:
        note = snap_to_c_minor(note - 12)

    vel = FIXED_VELOCITY
    if step_16 % 4 == 0:
        vel = min(127, vel + 10)

    gate_steps = 1 if (spot >= 0.50 or sensor_size < 30) else (2 if d <= 0.70 else 1)
    return snap_to_c_minor(note), vel, gate_steps


def plan_spottiness_accents(
    analysis: AnalysisResult,
    bar_index: int,
    step_16: int,
) -> Optional[tuple[int, int]]:
    """Emit syncopated staccato Marimba counter-accents when ``spottiness`` is notable."""
    spot = _clamp01(getattr(analysis, "spottiness", 0.0))
    if spot < 0.25:
        return None

    syncopated_slots = (3, 6, 11, 14) if spot < 0.65 else (2, 3, 6, 9, 11, 14, 15)
    step = step_16 % 16
    if step not in syncopated_slots:
        return None

    chord = VERIDIS_QUO_PROGRESSION[bar_index % len(VERIDIS_QUO_PROGRESSION)]
    slot_idx = syncopated_slots.index(step)
    base_tone = chord.arp_tones[slot_idx % len(chord.arp_tones)]
    oct_adjust = 12 if slot_idx % 2 == 1 else 0
    pitch = snap_to_c_minor(base_tone + oct_adjust)
    vel = FIXED_VELOCITY
    return pitch, min(127, vel)


def plan_ambient_drums(
    density: float,
    intensity: float,
    step_16: int,
    spottiness: float = 0.0,
) -> list[tuple[int, int]]:
    """Generate a 75 BPM Veridis Quo-inspired French-house/downtempo drum groove."""
    step = step_16 % 16
    d = _clamp01(density)
    spot = _clamp01(spottiness)
    hits: list[tuple[int, int]] = []

    # Warm muted four-on-the-floor kick (or breathing half-time at very low density)
    if d < 0.25:
        if step == 0:
            hits.append((DRUM_KICK, FIXED_VELOCITY))
    elif d < 0.50:
        if step in (0, 8):
            hits.append((DRUM_KICK, FIXED_VELOCITY))
        elif step == 10:
            hits.append((DRUM_KICK, FIXED_VELOCITY))
    else:
        if step in (0, 4, 8, 12):
            hits.append((DRUM_KICK, FIXED_VELOCITY))
        elif step in (6, 14) and d >= 0.78:
            hits.append((DRUM_KICK, FIXED_VELOCITY))

    # Gentle rimshot + soft clap backbeat on Beats 2 and 4
    if step in (4, 12):
        hits.append((DRUM_RIM, FIXED_VELOCITY))
        if d >= 0.45:
            hits.append((DRUM_CLAP, FIXED_VELOCITY))

    # Crisp hi-hat & shaker subdivision scaled by visual density
    if d < 0.30:
        if step in (0, 4, 8, 12):
            hits.append((DRUM_SHAKER, FIXED_VELOCITY))
    elif d <= 0.70:
        if step in (2, 6, 10, 14):
            hat = DRUM_HAT_OPEN if step == 14 else DRUM_HAT_CLOSED
            hits.append((hat, FIXED_VELOCITY))
        elif step % 2 == 0:
            hits.append((DRUM_HAT_CLOSED, FIXED_VELOCITY))
    else:
        if step == 14:
            hits.append((DRUM_HAT_OPEN, FIXED_VELOCITY))
        elif step % 2 == 0:
            hits.append((DRUM_HAT_CLOSED, FIXED_VELOCITY))
        else:
            hits.append((DRUM_SHAKER, FIXED_VELOCITY))

    # Spottiness injects crisp percussion ghost-notes, conga & tambourine sparkles
    if spot >= 0.35:
        if step in (3, 7, 11, 15):
            hits.append((DRUM_TAMBOURINE, FIXED_VELOCITY))
        if step in (6, 15) and (d > 0.40 or spot >= 0.60):
            hits.append((DRUM_CONGA_HI, FIXED_VELOCITY))
    elif d > 0.70 and step in (6, 15):
        hits.append((DRUM_CONGA_HI, FIXED_VELOCITY))

    return hits


def compute_expression_ccs(analysis: AnalysisResult, step_16: int) -> dict[str, int]:
    """Compute MIDI CC and pitch-bend parameters from CV features & cursor size."""
    i = _clamp01(analysis.intensity)
    swirl = _clamp01(getattr(analysis, "swirliness", 0.0))
    sensor_size = max(16, min(160, int(getattr(analysis, "sensor_size", 40))))

    if sensor_size < 30:
        reverb = 34
    elif sensor_size <= 60:
        reverb = 74
    else:
        reverb = int(round(88 + min(1.0, (sensor_size - 60) / 100.0) * 30))

    chorus = int(round(25 + swirl * 92))
    modulation = int(round(swirl * 72))
    brightness = int(round(42 + i * 82))      # CC 74 Filter Cutoff
    expression = FIXED_VELOCITY                # CC 11 Expression

    if swirl >= 0.15:
        phase = (step_16 % 16) / 16.0 * 2.0 * math.pi
        pitch_bend = int(round(8192 + math.sin(phase) * swirl * 820))
    else:
        pitch_bend = 8192

    return {
        "reverb": max(0, min(127, reverb)),
        "chorus": max(0, min(127, chorus)),
        "modulation": max(0, min(127, modulation)),
        "brightness": max(0, min(127, brightness)),
        "expression": max(0, min(127, expression)),
        "pitch_bend": max(0, min(16383, pitch_bend)),
    }
