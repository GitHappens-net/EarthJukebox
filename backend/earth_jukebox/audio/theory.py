"""Veridis Quo-inspired C Natural Minor music theory and CV sonification mapping.

All pitches, chords, arpeggios, basslines, and syncopated counter-melodies are
strictly quantized to **C Natural Minor** (``C, D, Eb, F, G, Ab, Bb`` ->
pitch classes ``{0, 2, 3, 5, 7, 8, 10}``) on a locked **75 BPM** 16th-note grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

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


BPM: float = 75.0
BEAT_DURATION_SEC: float = 60.0 / BPM                 # 0.8s per quarter beat
SIXTEENTH_DURATION_SEC: float = BEAT_DURATION_SEC / 4.0  # 0.2s per 16th step
FIXED_VELOCITY: int = 88
FIXED_VOLUME: float = 0.65

C_MINOR_PITCH_CLASSES: frozenset[int] = frozenset({0, 2, 3, 5, 7, 8, 10})
C_MINOR_SORTED_PCS: tuple[int, ...] = (0, 2, 3, 5, 7, 8, 10)
NOTE_NAMES: tuple[str, ...] = (
    "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"
)

PAD_CHANNEL = 0            # Drawbar Organ (GM 16)
BASS_CHANNEL = 6           # Synth Bass 1 (GM 38)
ARP_CHANNEL = 8            # Lead 1 Square (GM 80) - Veridis Quo baroque lead
DRUM_CHANNEL = 9           # GM Percussion
WARM_PAD_CHANNEL = 10      # Warm Pad (GM 89)
PLUCK_CHANNEL = 11         # Marimba (GM 12) - Spottiness syncopated counter-accents
CHURCH_ORGAN_CHANNEL = 12  # Church Organ (GM 19) - Wide-cursor harmonic swell

DRUM_KICK = 36
DRUM_RIM = 37
DRUM_CLAP = 39
DRUM_HAT_CLOSED = 42
DRUM_LOW_TOM = 45
DRUM_HAT_OPEN = 46
DRUM_TAMBOURINE = 54
DRUM_CONGA_HI = 63
DRUM_SHAKER = 70

COLOR_PITCH_MAP: dict[str, tuple[int, ...]] = {
    "Black": (36, 43, 48, 51, 55),          # C2, G2, C3, Eb3, G3
    "Blue": (48, 51, 55, 58, 60),           # C3, Eb3, G3, Bb3, C4
    "Green": (48, 53, 55, 58, 63, 67),      # C3, F3, G3, Bb3, Eb4, G4
    "White": (60, 67, 70, 72, 75),          # C4, G4, Bb4, C5, Eb5
    "Orange": (62, 63, 68, 70, 74),         # D4, Eb4, Ab4, Bb4, D5
    "Red": (74, 75, 79, 80, 82, 84),        # D5, Eb5, G5, Ab5, Bb5, C6
}

COLOR_OCTAVE_OFFSET: dict[str, int] = {
    "Black": -12,
    "Blue": -12,
    "Green": 0,
    "White": 0,
    "Orange": 0,
    "Red": 12,
}

COLOR_CHANNEL_PROGRAM: dict[str, tuple[int, int]] = {
    "Black": (1, 89),   # Warm Pad
    "Blue": (2, 88),    # New Age Pad
    "Green": (3, 12),   # Marimba
    "White": (4, 11),   # Vibraphone
    "Orange": (5, 4),   # Electric Piano 1 (Rhodes)
    "Red": (7, 81),     # Lead 2 (Sawtooth)
}

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

HARMONIC_PROGRESSION: tuple[tuple[str, int, tuple[int, ...]], ...] = tuple(
    (chord.name, chord.bass_note, chord.pad_tones) for chord in VERIDIS_QUO_PROGRESSION
)

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


DYNAMICS_ENABLED: bool = True
DYNAMIC_FLOOR: int = 52     # darkest region is still clearly audible
DYNAMIC_CEIL: int = 112     # brightest region


def intensity_to_velocity(intensity: float) -> int:
    """Map visual brightness ``[0, 1]`` to a MIDI velocity (soft -> loud)."""
    if not DYNAMICS_ENABLED:
        return FIXED_VELOCITY
    i = _clamp01(intensity)
    return int(round(DYNAMIC_FLOOR + (DYNAMIC_CEIL - DYNAMIC_FLOOR) * (i ** 0.85)))


def intensity_to_volume(intensity: float) -> float:
    """Map visual brightness to a normalized amplitude ``[0.40, 0.95]``."""
    if not DYNAMICS_ENABLED:
        return FIXED_VOLUME
    return round(0.40 + 0.55 * _clamp01(intensity), 4)


TIER_EDGES: tuple[float, ...] = (0.15, 0.35, 0.65)
TIER_NAMES: tuple[str, ...] = ("Calm", "Gentle", "Busy", "Frantic")
TIER_RHYTHMS: tuple[tuple[str, float], ...] = (
    ("Whole Note", 4.0),
    ("Half Note", 2.0),
    ("8th Note", 0.5),
    ("16th Arpeggio", 0.25),
)
TIER_DESCRIPTIONS: tuple[str, ...] = (
    "spacious: chord pad, one low note per bar, slow whole-note melody",
    "relaxed: half-note melody, quarter-note arpeggio, soft kick and off-beat hats",
    "driving: four-on-the-floor kick, eighth-note bass, arpeggio and melody",
    "dense: sixteenth-note arpeggio and melody, syncopated bass, full percussion",
)
BASS_GATE_STEPS: tuple[float, ...] = (12.0, 6.0, 2.0, 1.0)


def activity_tier(density: float) -> int:
    """Return the activity tier ``0..3`` for a visual edge density."""
    d = _clamp01(density)
    for tier, edge in enumerate(TIER_EDGES):
        if d < edge:
            return tier
    return len(TIER_EDGES)


def select_rhythm(density: float) -> tuple[str, float]:
    """Map visual edge density ``[0.0, 1.0]`` to rhythmic subdivision in beats."""
    return TIER_RHYTHMS[activity_tier(density)]


def should_trigger_data_step(density: float, step_16: int) -> bool:
    """Return True when the data melody falls cleanly on the 75 BPM beat grid."""
    step = step_16 % 16
    tier = activity_tier(density)
    if tier == 0:
        return step == 0
    if tier == 1:
        return step in (0, 8)
    if tier == 2:
        return step % 2 == 0
    return True


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


COLOR_WARMTH_RANK: dict[str, int] = {
    "Black": 0, "Blue": 1, "Green": 2, "White": 3, "Orange": 4, "Red": 5,
}
# Added to the filter-cutoff CC (74): cool colors sound darker, warm colors brighter.
COLOR_BRIGHTNESS_BIAS: dict[str, int] = {
    "Black": -18, "Blue": -6, "Green": 0, "White": 6, "Orange": 8, "Red": 16,
}
COLOR_PAN: dict[str, float] = {
    "Black": -0.25, "Blue": -0.15, "Green": 0.20, "White": 0.30, "Orange": 0.15, "Red": 0.0,
}
COLOR_INSTRUMENT_NAME: dict[str, str] = {
    "Black": "deep sub pad",
    "Blue": "glassy pad",
    "Green": "marimba",
    "White": "bell / vibraphone",
    "Orange": "electric piano",
    "Red": "bright saw lead",
}
COLOR_REGISTER_NAME: dict[str, str] = {
    "Black": "deep register", "Blue": "low-mid register", "Green": "middle register",
    "White": "mid-high register", "Orange": "high register", "Red": "highest register",
}

# Synth timbre per MIDI channel (used by the NumPy synth and the web/broadcast layer).
CHANNEL_TIMBRE: dict[int, str] = {
    PAD_CHANNEL: "organ_pad",
    WARM_PAD_CHANNEL: "warm_pad",
    CHURCH_ORGAN_CHANNEL: "church",
    BASS_CHANNEL: "bass",
    ARP_CHANNEL: "arp",
    PLUCK_CHANNEL: "pluck",
}
_COLOR_TIMBRE: dict[str, str] = {
    "Black": "sub_pad", "Blue": "glass_pad", "Green": "marimba",
    "White": "bell", "Orange": "rhodes", "Red": "saw_lead",
}
CHANNEL_PAN: dict[int, float] = {
    PAD_CHANNEL: 0.0, WARM_PAD_CHANNEL: 0.0, CHURCH_ORGAN_CHANNEL: 0.0,
    BASS_CHANNEL: 0.0, ARP_CHANNEL: -0.30, PLUCK_CHANNEL: 0.40, DRUM_CHANNEL: 0.0,
}
# Mix balance (MIDI CC7) so the data voice stays on top of the bed.
CHANNEL_VOLUME: dict[int, int] = {
    PAD_CHANNEL: 84, WARM_PAD_CHANNEL: 76, CHURCH_ORGAN_CHANNEL: 66, BASS_CHANNEL: 104,
    ARP_CHANNEL: 92, PLUCK_CHANNEL: 88, DRUM_CHANNEL: 90,
}
for _color, (_ch, _prog) in COLOR_CHANNEL_PROGRAM.items():
    CHANNEL_TIMBRE[_ch] = _COLOR_TIMBRE[_color]
    CHANNEL_PAN[_ch] = COLOR_PAN[_color]
    CHANNEL_VOLUME[_ch] = 104


PAD_ROLES: tuple[str, ...] = ("pad", "warm_pad", "church")


@dataclass(frozen=True)
class NoteEvent:
    """One note scheduled by :class:`StepSequencer` (used by live engine AND WAV synth)."""

    role: str            # pad, warm_pad, church, bass, drum, arp, pluck, voice, harmony, cue
    channel: int
    note: int
    velocity: int
    steps: float = 1.0   # length in 16th-note steps
    delay_steps: int = 0
    pan: float = 0.0


@dataclass(frozen=True)
class StepPlan:
    """Everything that should happen on one 16th-note step."""

    step_16: int
    bar_index: int
    chord_name: str
    tier: int
    tier_name: str
    rhythm_name: str
    note_label: str
    touching: bool = True
    retrigger_pad: bool = False
    released: bool = False       # True exactly once when the cursor leaves the map
    events: tuple[NoteEvent, ...] = ()


def snap_all(notes) -> list[int]:
    return [snap_to_c_minor(n) for n in notes]


def plan_bass_step(
    bar_index: int,
    density: float,
    intensity: float,
    sensor_size: int,
    step_16: int,
) -> list[tuple[int, int]]:
    """Plan pulsing French-touch analog synth bass notes on ``step_16``.

    Returns a list of ``(midi_note, velocity)`` pairs.  Pattern complexity follows
    the shared activity ladder.
    """
    chord = VERIDIS_QUO_PROGRESSION[bar_index % len(VERIDIS_QUO_PROGRESSION)]
    root = snap_to_c_minor(chord.bass_note)
    octave_up = snap_to_c_minor(root + 12)
    fifth = snap_to_c_minor(root + 7)
    step = step_16 % 16
    tier = activity_tier(density)
    base = intensity_to_velocity(intensity)

    def v(ratio: float) -> int:
        return max(30, min(127, int(base * ratio)))

    hits: list[tuple[int, int]] = []
    if tier == 0:
        if step == 0:
            hits.append((root, v(1.0)))
        elif step == 8 and sensor_size > 60:
            hits.append((octave_up, v(0.80)))
    elif tier == 1:
        if step == 0:
            hits.append((root, v(1.0)))
        elif step == 8:
            hits.append((root, v(0.85)))
    elif tier == 2:
        if step % 2 == 0:
            if step % 4 == 0:
                hits.append((root, v(1.0)))
            else:
                hits.append((octave_up, v(0.78)))
    else:
        if step in (0, 3, 4, 6, 8, 10, 12, 14, 15):
            if step in (0, 4, 8, 12):
                hits.append((root, v(1.0)))
            elif step in (6, 14):
                hits.append((fifth, v(0.80)))
            else:
                hits.append((octave_up, v(0.72)))

    if sensor_size > 60 and step == 0:
        sub = snap_to_c_minor(root - 12)
        if sub != root and sub >= 24:
            hits.append((sub, v(0.75)))
    return hits


def _build_color_chord_pool(dominant_color: str, bar_index: int) -> tuple[int, ...]:
    """Blend the color register palette with the active Veridis Quo chord tones."""
    chord = VERIDIS_QUO_PROGRESSION[bar_index % len(VERIDIS_QUO_PROGRESSION)]
    color_palette = COLOR_PITCH_MAP.get(dominant_color, COLOR_PITCH_MAP["Blue"])
    oct_shift = COLOR_OCTAVE_OFFSET.get(dominant_color, 0)

    blended = [snap_to_c_minor(tone + oct_shift) for tone in chord.arp_tones]
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

    This is the *color voice*: its instrument and register identify the dominant color.
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
        turn_cycle = (0, 1, 0, -1, 1, 2, 0, -1)
        degree_shift = turn_cycle[step_index % len(turn_cycle)]
        if swirliness < 0.72:
            degree_shift = max(-1, min(1, degree_shift))
        raw_pitch = shift_c_minor_scale_degree(raw_pitch, degree_shift)

    quantized_pitch = snap_to_c_minor(raw_pitch)
    channel, _ = COLOR_CHANNEL_PROGRAM.get(analysis.dominant_color, (2, 88))
    # The color voice sits a little above the bed so the color is always identifiable.
    velocity = min(127, intensity_to_velocity(analysis.intensity) + 6)

    spottiness = _clamp01(getattr(analysis, "spottiness", 0.0))
    if shape == "organic" and step_index % 3 == 0:
        velocity = min(127, velocity + 10)
    if spottiness >= 0.50 and step_index % 2 == 1:
        velocity = max(28, velocity - 12)

    sensor_size = int(getattr(analysis, "sensor_size", 40))
    if sensor_size < 30:
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
    tier = activity_tier(analysis.density)
    shape = getattr(analysis, "shape", "uniform") or "uniform"
    swirl = _clamp01(getattr(analysis, "swirliness", 0.0))
    spot = _clamp01(getattr(analysis, "spottiness", 0.0))
    sensor_size = int(getattr(analysis, "sensor_size", 40))
    step = step_16 % 16

    # Rhythm gate: Calm = every half bar, Gentle = quarters, Busy = eighths, Frantic = 16ths
    if tier == 0 and step % 8 != 0:
        return None
    if tier == 1 and step % 4 != 0:
        return None
    if tier == 2 and step % 2 != 0 and swirl < 0.60:
        return None

    arp = chord.arp_tones
    seqs = {
        "linear": (0, 1, 2, 3, 4, 5, 4, 3),
        "organic": (0, 3, 2, 5, 1, 4, 2, 3),
        "uniform": (0, 2, 0, 2, 4, 2, 0, 2),
        "circular": (0, 2, 4, 3, 2, 1, 2, 4),
    }
    seq = seqs.get(shape, seqs["circular"])
    note = arp[seq[step_index % len(seq)] % len(arp)]

    if swirl >= 0.45:
        turn = (0, 1, 2, 1, 0, -1, 0, 1)[step % 8]
        note = shift_c_minor_scale_degree(note, turn)

    if analysis.dominant_color in ("Red", "Orange") and note < 72:
        note = snap_to_c_minor(note + 12)
    elif analysis.dominant_color == "Black" and note >= 68:
        note = snap_to_c_minor(note - 12)

    vel = max(28, int(intensity_to_velocity(analysis.intensity) * 0.85))
    if step % 4 == 0:
        vel = min(127, vel + 10)

    gate_steps = (4, 3, 2, 1)[tier]
    if spot >= 0.50 or sensor_size < 30:
        gate_steps = 1
    return snap_to_c_minor(note), vel, gate_steps


def plan_spottiness_accents(
    analysis: AnalysisResult,
    bar_index: int,
    step_16: int,
) -> Optional[tuple[int, int]]:
    """Emit syncopated staccato counter-accents when ``spottiness`` is notable."""
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
    vel = max(30, int(intensity_to_velocity(analysis.intensity) * 0.92))
    return pitch, min(127, vel)


def plan_ambient_drums(
    density: float,
    intensity: float,
    step_16: int,
    spottiness: float = 0.0,
) -> list[tuple[int, int]]:
    """75 BPM French-house / downtempo groove, one clear pattern per activity tier."""
    step = step_16 % 16
    tier = activity_tier(density)
    spot = _clamp01(spottiness)
    base = intensity_to_velocity(intensity)

    def v(ratio: float) -> int:
        return max(24, min(127, int(base * ratio)))

    hits: list[tuple[int, int]] = []
    if tier == 0:       # Calm: heartbeat kick + soft shaker
        if step == 0:
            hits.append((DRUM_KICK, v(0.85)))
        if step in (4, 12):
            hits.append((DRUM_SHAKER, v(0.35)))
    elif tier == 1:     # Gentle: half-time kick, rim backbeat, off-beat hats
        if step in (0, 8):
            hits.append((DRUM_KICK, v(0.95)))
        elif step == 10:
            hits.append((DRUM_KICK, v(0.50)))
        if step in (4, 12):
            hits.append((DRUM_RIM, v(0.65)))
        if step in (2, 6, 10, 14):
            hits.append((DRUM_HAT_CLOSED, v(0.42)))
    elif tier == 2:     # Busy: four on the floor, clap, 8th-note hats
        if step in (0, 4, 8, 12):
            hits.append((DRUM_KICK, v(1.0 if step % 8 == 0 else 0.88)))
        if step in (4, 12):
            hits.append((DRUM_RIM, v(0.65)))
            hits.append((DRUM_CLAP, v(0.62)))
        if step == 14:
            hits.append((DRUM_HAT_OPEN, v(0.58)))
        elif step % 2 == 0:
            hits.append((DRUM_HAT_CLOSED, v(0.50 if step % 4 == 2 else 0.38)))
    else:               # Frantic: extra kick pushes, 16th hats, conga
        if step in (0, 4, 8, 12):
            hits.append((DRUM_KICK, v(1.0 if step % 8 == 0 else 0.88)))
        elif step == 14:
            hits.append((DRUM_KICK, v(0.55)))
        if step in (4, 12):
            hits.append((DRUM_CLAP, v(0.70)))
        if step == 14:
            hits.append((DRUM_HAT_OPEN, v(0.60)))
        elif step % 2 == 0:
            hits.append((DRUM_HAT_CLOSED, v(0.46)))
        else:
            hits.append((DRUM_SHAKER, v(0.32)))
        if step in (6, 15):
            hits.append((DRUM_CONGA_HI, v(0.55)))

    if spot >= 0.35:
        if step in (3, 7, 11, 15):
            hits.append((DRUM_TAMBOURINE, v(0.50)))
        if step in (6, 15) and (tier >= 2 or spot >= 0.60) and (DRUM_CONGA_HI, v(0.55)) not in hits:
            hits.append((DRUM_CONGA_HI, v(0.55)))
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
    bias = COLOR_BRIGHTNESS_BIAS.get(analysis.dominant_color, 0)
    brightness = int(round(42 + i * 82 + bias))      # CC 74 Filter Cutoff
    expression = 100                                  # CC 11 Expression

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


def select_harmony_below(midi_note: int, bar_index: int = 0) -> Optional[int]:
    """Pick a consonant chord tone 3-9 semitones *below* the melody (never a tritone)."""
    chord = VERIDIS_QUO_PROGRESSION[bar_index % len(VERIDIS_QUO_PROGRESSION)]
    q = snap_to_c_minor(midi_note)
    pool = {
        snap_to_c_minor(n + off)
        for n in (*chord.pad_tones, *chord.arp_tones)
        for off in (-12, 0, 12)
    }
    choices = [n for n in pool if (q - n) in (3, 4, 5, 7, 8, 9)]
    if choices:
        return max(choices)
    fallback = snap_to_c_minor(q - 5)
    return fallback if fallback != q else None


def plan_transition_cue(
    prev_color: str, new_color: str, bar_index: int, base_velocity: int
) -> list[NoteEvent]:
    """Two-note earcon on the NEW color's instrument: rising = warmer, falling = cooler."""
    new_rank = COLOR_WARMTH_RANK.get(new_color, 2)
    prev_rank = COLOR_WARMTH_RANK.get(prev_color, 2)
    if new_rank == prev_rank:
        return []
    pool = _build_color_chord_pool(new_color, bar_index)
    lo = pool[len(pool) // 3]
    hi = pool[(2 * len(pool)) // 3]
    if hi <= lo:
        hi = shift_c_minor_scale_degree(lo, 2)
    first, second = (lo, hi) if new_rank > prev_rank else (hi, lo)
    channel = COLOR_CHANNEL_PROGRAM.get(new_color, (2, 88))[0]
    vel = min(127, base_velocity + 12)
    return [
        NoteEvent("cue", channel, first, vel, 1.0, 0, COLOR_PAN.get(new_color, 0.0)),
        NoteEvent("cue", channel, second, max(28, vel - 6), 2.0, 1, COLOR_PAN.get(new_color, 0.0)),
    ]


class AnalysisSmoother:
    """Exponentially smooths continuous features, adds tier hysteresis, debounces labels."""

    def __init__(
        self,
        alpha: float = 0.5,
        color_hold: int = 2,
        shape_hold: int = 3,
        tier_margin: float = 0.04,
    ) -> None:
        self.alpha = float(alpha)
        self.color_hold = int(color_hold)
        self.shape_hold = int(shape_hold)
        self.tier_margin = float(tier_margin)
        self.reset()

    def reset(self) -> None:
        self._vals: Optional[dict[str, float]] = None
        self._tier = 0
        self._color: Optional[str] = None
        self._shape: Optional[str] = None
        self._cand: dict[str, tuple[Optional[str], int]] = {"color": (None, 0), "shape": (None, 0)}

    def _debounce(self, key: str, current: str, incoming: str, hold: int) -> str:
        if incoming == current:
            self._cand[key] = (None, 0)
            return current
        cand, count = self._cand[key]
        count = count + 1 if cand == incoming else 1
        if count >= hold:
            self._cand[key] = (None, 0)
            return incoming
        self._cand[key] = (incoming, count)
        return current

    def update(self, a: AnalysisResult) -> AnalysisResult:
        if not a.touching:
            self.reset()
            return a
        raw = {
            "intensity": a.intensity, "density": a.density,
            "spottiness": getattr(a, "spottiness", 0.0), "swirliness": getattr(a, "swirliness", 0.0),
        }
        if self._vals is None:
            self._vals = dict(raw)
            self._tier = activity_tier(raw["density"])
            self._color = a.dominant_color
            self._shape = getattr(a, "shape", "uniform")
        else:
            for k, val in raw.items():
                self._vals[k] += self.alpha * (val - self._vals[k])
            self._color = self._debounce("color", self._color, a.dominant_color, self.color_hold)
            self._shape = self._debounce("shape", self._shape, getattr(a, "shape", "uniform"), self.shape_hold)

        d = self._vals["density"]
        lo = TIER_EDGES[self._tier - 1] if self._tier > 0 else 0.0
        hi = TIER_EDGES[self._tier] if self._tier < len(TIER_EDGES) else 1.0
        if lo - self.tier_margin <= d < lo:
            d = lo
        elif hi <= d < hi + self.tier_margin and self._tier < len(TIER_EDGES):
            d = hi - 1e-6
        self._tier = activity_tier(d)
        return replace(
            a,
            dominant_color=self._color,
            shape=self._shape,
            intensity=self._vals["intensity"],
            density=d,
            spottiness=self._vals["spottiness"],
            swirliness=self._vals["swirliness"],
        )


class StepSequencer:
    """Pure (I/O-free) scheduler shared by the live engine and the offline WAV synth."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._data_idx = 0
        self._arp_idx = 0
        self._pad_sig = None
        self._was_touching = False
        self._last_rhythm = ""
        self._last_color = ""
        self._pending_entry = False
        self._note_label = "C4"

    @property
    def note_label(self) -> str:
        return self._note_label

    def plan(self, analysis: AnalysisResult, global_step: int, drums_on: bool = True) -> StepPlan:
        bar_index = (global_step // 16) % len(VERIDIS_QUO_PROGRESSION)
        step_16 = global_step % 16
        chord = VERIDIS_QUO_PROGRESSION[bar_index]

        if not analysis.touching:
            released = self._was_touching
            self.reset()
            return StepPlan(step_16, bar_index, chord.name, 0, TIER_NAMES[0], "Standby",
                            self._note_label, touching=False, released=released)

        d = _clamp01(analysis.density)
        size = int(getattr(analysis, "sensor_size", 40))
        spot = _clamp01(getattr(analysis, "spottiness", 0.0))
        tier = activity_tier(d)
        rhythm_name, _beats = select_rhythm(d)
        base = intensity_to_velocity(analysis.intensity)
        sensor_tier = "tight" if size < 30 else ("wide" if size > 60 else "mid")
        color = analysis.dominant_color
        events: list[NoteEvent] = []

        # 1. Pad bed: re-voiced on the bar line AND whenever color/brightness/cursor size change
        chord_name, _bass, pad_tones = plan_chord_voicing(bar_index, analysis.intensity, size, color)
        sig = (pad_tones, sensor_tier)
        retrigger = (not self._was_touching) or step_16 == 0 or (sig != self._pad_sig and step_16 % 2 == 0)
        if retrigger:
            self._pad_sig = sig
            pad_vel = max(24, int(base * 0.78))
            hold = float(max(1, 16 - step_16))
            for tone in pad_tones:
                q = snap_to_c_minor(tone)
                events.append(NoteEvent("pad", PAD_CHANNEL, q, pad_vel, hold))
                if size >= 30:
                    events.append(NoteEvent("warm_pad", WARM_PAD_CHANNEL, q, max(24, int(pad_vel * 0.8)), hold))
            if size > 60:
                for tone in pad_tones[:3]:
                    events.append(NoteEvent("church", CHURCH_ORGAN_CHANNEL, snap_to_c_minor(tone + 12),
                                            max(26, int(pad_vel * 0.72)), hold))

        # 2. Bass
        for note, vel in plan_bass_step(bar_index, d, analysis.intensity, size, step_16):
            events.append(NoteEvent("bass", BASS_CHANNEL, snap_to_c_minor(note), vel, BASS_GATE_STEPS[tier]))

        # 3. Drums
        if drums_on:
            for dn, dv in plan_ambient_drums(d, analysis.intensity, step_16, spottiness=spot):
                events.append(NoteEvent("drum", DRUM_CHANNEL, dn, dv, 1.0))

        # 4. Baroque arpeggio (ping-pong panned)
        arp_hit = plan_baroque_arpeggio(analysis, bar_index, step_16, self._arp_idx)
        if arp_hit is not None:
            a_note, a_vel, a_gate = arp_hit
            pan = -0.35 if self._arp_idx % 2 == 0 else 0.35
            self._arp_idx += 1
            events.append(NoteEvent("arp", ARP_CHANNEL, snap_to_c_minor(a_note), a_vel, float(a_gate), pan=pan))

        # 5. Spotty counter-accents
        spot_hit = plan_spottiness_accents(analysis, bar_index, step_16)
        if spot_hit is not None:
            s_note, s_vel = spot_hit
            events.append(NoteEvent("pluck", PLUCK_CHANNEL, snap_to_c_minor(s_note), s_vel, 1.0,
                                    pan=0.45 if step_16 % 2 else -0.45))

        # 6. Color voice + color-change earcon
        prev_color = self._last_color
        color_changed = self._was_touching and color != prev_color
        if (not self._was_touching) or rhythm_name != self._last_rhythm or color != prev_color:
            self._pending_entry = True
        self._last_rhythm = rhythm_name
        self._last_color = color
        if color_changed:
            events.extend(plan_transition_cue(prev_color, color, bar_index, base))

        if should_trigger_data_step(d, step_16) or (self._pending_entry and step_16 % 4 == 0):
            self._pending_entry = False
            channel, pitch, velocity, _vol, _motif, dur_sec = plan_musical_step(
                analysis, self._data_idx, bar_index=bar_index
            )
            self._data_idx += 1
            if step_16 % 4 == 0:
                velocity = min(127, velocity + 8)
            span = max(1, int(round(dur_sec / SIXTEENTH_DURATION_SEC)))
            span = max(1, span - (1 if span > 1 else 0))
            q_pitch = snap_to_c_minor(pitch)
            events.append(NoteEvent("voice", channel, q_pitch, velocity, float(span),
                                    pan=COLOR_PAN.get(color, 0.0)))
            if step_16 % 4 == 0 and size >= 30:
                harmony = select_harmony_below(q_pitch, bar_index)
                if harmony is not None:
                    events.append(NoteEvent("harmony", channel, harmony, max(24, int(velocity * 0.62)),
                                            float(span), pan=COLOR_PAN.get(color, 0.0)))
            self._note_label = midi_to_note_name(q_pitch)

        self._was_touching = True
        return StepPlan(step_16, bar_index, chord_name, tier, TIER_NAMES[tier], rhythm_name,
                        self._note_label, True, retrigger, False, tuple(events))

SHAPE_SOUND: dict[str, str] = {
    "uniform": "simple repeating arpeggio, few note changes",
    "linear": "straight rising/falling scale runs",
    "circular": "cascading baroque turn (classic Veridis Quo)",
    "organic": "wandering, irregular melodic leaps",
}
_LOUDNESS_WORDS = ("very soft", "soft", "medium", "loud", "very loud")

def active_layers(analysis: AnalysisResult, drums_on: bool = True) -> list[str]:
    """Names of the musical layers currently sounding for this analysis."""
    if not analysis.touching:
        return []
    size = int(getattr(analysis, "sensor_size", 40))
    layers = ["Chord pad", "Bass", f"Color voice ({COLOR_INSTRUMENT_NAME.get(analysis.dominant_color, 'pad')})",
              "Arpeggio"]
    if drums_on:
        layers.insert(2, "Drums")
    if size >= 30:
        layers.append("Warm pad + harmony")
    if size > 60:
        layers.append("Church organ shimmer")
    if getattr(analysis, "spottiness", 0.0) >= 0.25:
        layers.append("Marimba sparkles")
    return layers

def describe_mapping(analysis: AnalysisResult, drums_on: bool = True) -> dict:
    """Structured, human-readable explanation of how each image feature shaped the sound."""
    if not analysis.touching:
        return {"headline": "Standby: move over the map to hear it", "tier": 0,
                "tier_name": TIER_NAMES[0], "layers": [], "rows": []}
    color = analysis.dominant_color
    i = _clamp01(analysis.intensity)
    d = _clamp01(analysis.density)
    spot = _clamp01(getattr(analysis, "spottiness", 0.0))
    swirl = _clamp01(getattr(analysis, "swirliness", 0.0))
    shape = getattr(analysis, "shape", "uniform") or "uniform"
    size = int(getattr(analysis, "sensor_size", 40))
    tier = activity_tier(d)
    instrument = COLOR_INSTRUMENT_NAME.get(color, "pad")
    loud = _LOUDNESS_WORDS[min(len(_LOUDNESS_WORDS) - 1, int(i * len(_LOUDNESS_WORDS)))]
    if size < 30:
        size_sound = "intimate: two-note pad, short crisp notes, dry"
    elif size > 60:
        size_sound = "wide: lush 6-7 note pad, church organ shimmer, deep sub bass, more reverb"
    else:
        size_sound = "standard: 3-5 note chord pad with warm pad and harmony"
    rows = [
        {"feature": "Color", "value": color,
         "sound": f"{instrument} in the {COLOR_REGISTER_NAME.get(color, 'middle register')}",
         "hint": "Cool colors are lower and darker, warm colors higher and brighter. "
                 "A color change plays a rising (warmer) or falling (cooler) two-note cue."},
        {"feature": "Brightness", "value": f"{round(i * 100)}%",
         "sound": f"{loud}, {'brighter' if i >= 0.5 else 'darker'} tone",
         "hint": "Brighter pixels play louder and with a more open filter."},
        {"feature": "Detail (edge density)", "value": f"{round(d * 100)}%",
         "sound": f"{TIER_NAMES[tier]}: {TIER_DESCRIPTIONS[tier]}",
         "hint": "More visual detail adds layers in a fixed order: pad, bass, drums, arpeggio, 16th notes."},
        {"feature": "Shape", "value": shape, "sound": SHAPE_SOUND.get(shape, ""),
         "hint": "Shape chooses the contour of the arpeggio and melody."},
        {"feature": "Spottiness", "value": f"{round(spot * 100)}%",
         "sound": "marimba sparkles and tambourine" if spot >= 0.25 else "none",
         "hint": "Scattered small features add syncopated sparkles."},
        {"feature": "Swirl", "value": f"{round(swirl * 100)}%",
         "sound": "vibrato and curling melodic turns" if swirl >= 0.45 else "steady, no vibrato",
         "hint": "Curling flow adds pitch wobble and ornamental turns."},
        {"feature": "Cursor size", "value": f"{size}px", "sound": size_sound,
         "hint": "Bigger cursor = bigger harmony and more space."},
    ]
    return {
        "headline": f"{TIER_NAMES[tier]} {color.lower()} zone: {instrument}, {loud}",
        "tier": tier,
        "tier_name": TIER_NAMES[tier],
        "layers": active_layers(analysis, drums_on),
        "rows": rows,
    }

def format_music_explanation(analysis: AnalysisResult, drums_on: bool = True) -> str:
    """One spoken sentence describing the current sound."""
    info = describe_mapping(analysis, drums_on)
    if not info["rows"]:
        return info["headline"]
    return f"{info['headline']}. Playing: {', '.join(info['layers'])}."