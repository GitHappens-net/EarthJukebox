"""Unit tests for cross-platform audio backends, Veridis Quo C-minor theory, and sequencer."""

from __future__ import annotations

import ast
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from earth_jukebox.models import AnalysisResult
except ImportError:
    from earth_jukebox.audio.theory import AnalysisResult  # type: ignore[assignment]

from earth_jukebox.audio import (  # noqa: E402
    ARP_CHANNEL,
    BASS_CHANNEL,
    CHURCH_ORGAN_CHANNEL,
    PAD_CHANNEL,
    PLUCK_CHANNEL,
    VERIDIS_QUO_PROGRESSION,
    WARM_PAD_CHANNEL,
    AudioBackend,
    CMinorAudioEngine,
    LinuxFluidSynthBackend,
    NullAudioBackend,
    PygameMidiBackend,
    PygameMixerSynthBackend,
    WinmmMidiBackend,
    compute_expression_ccs,
    create_best_audio_backend,
    plan_baroque_arpeggio,
    plan_bass_step,
    plan_chord_voicing,
    plan_spottiness_accents,
)


class RecordingAudioBackend(AudioBackend):
    """Test double capturing MIDI note, CC, and pitch-bend events."""

    def __init__(self) -> None:
        self.started = False
        self.closed = False
        self.programs: dict[int, int] = {}
        self.events: list[tuple[str, int, int, int]] = []
        self.ccs: list[tuple[int, int, int]] = []
        self.bends: list[tuple[int, int]] = []

    @property
    def backend_name(self) -> str:
        return "RecordingTestBackend"

    def start(self, channel_programs: dict[int, int]) -> None:
        self.started = True
        self.programs = dict(channel_programs)

    def note_on(self, channel: int, midi_note: int, velocity: int) -> None:
        self.events.append(("on", channel, midi_note, velocity))

    def note_off(self, channel: int, midi_note: int) -> None:
        self.events.append(("off", channel, midi_note, 0))

    def control_change(self, channel: int, control: int, value: int) -> None:
        self.ccs.append((channel, control, value))

    def pitch_bend(self, channel: int, bend_14bit: int) -> None:
        self.bends.append((channel, bend_14bit))

    def close(self) -> None:
        self.closed = True


class MinimalAudioBackend(AudioBackend):
    """Subclass omitting pitch_bend/control_change to verify ABC default no-ops."""

    @property
    def backend_name(self) -> str:
        return "MinimalBackend"

    def start(self, channel_programs: dict[int, int]) -> None:
        pass

    def note_on(self, channel: int, midi_note: int, velocity: int) -> None:
        pass

    def note_off(self, channel: int, midi_note: int) -> None:
        pass

    def close(self) -> None:
        pass


# =============================================================================
# 1. LINUX & WINDOWS COMPATIBILITY TESTS (backends.py)
# =============================================================================
def test_backends_module_has_no_top_level_windll_or_wintypes_imports() -> None:
    backends_path = SRC_DIR / "earth_jukebox" / "audio" / "backends.py"
    tree = ast.parse(backends_path.read_text(encoding="utf-8"))

    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "ctypes":
            imported = {alias.name for alias in node.names}
            assert "wintypes" not in imported
            assert "windll" not in imported
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in ("ctypes.wintypes", "ctypes.windll")


def test_backends_import_and_fallback_cleanly_on_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Linux")

    # Load a fresh module instance under simulated Linux without mutating shared classes
    backends_path = SRC_DIR / "earth_jukebox" / "audio" / "backends.py"
    spec = importlib.util.spec_from_file_location(
        "earth_jukebox_audio_backends_linux", backends_path
    )
    assert spec is not None and spec.loader is not None
    linux_backends = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(linux_backends)

    with pytest.raises(RuntimeError, match="Windows"):
        linux_backends.WinmmMidiBackend()

    # Audio disabled always returns NullAudioBackend
    assert isinstance(
        linux_backends.create_best_audio_backend(enable_audio=False),
        linux_backends.NullAudioBackend,
    )

    # When no Linux MIDI/synth device is available, gracefully fall back to NullAudioBackend
    def _fail_start(self, _programs):
        raise RuntimeError("unavailable in headless CI")

    monkeypatch.setattr(linux_backends.PygameMidiBackend, "start", _fail_start)
    monkeypatch.setattr(linux_backends.LinuxFluidSynthBackend, "start", _fail_start)
    monkeypatch.setattr(linux_backends.PygameMixerSynthBackend, "start", _fail_start)

    chosen = linux_backends.create_best_audio_backend(enable_audio=True)
    assert isinstance(chosen, linux_backends.NullAudioBackend)


def test_audio_backend_pitch_bend_default_noop_and_null_lifecycle() -> None:
    minimal = MinimalAudioBackend()
    minimal.start({0: 16})
    minimal.control_change(0, 91, 80)
    minimal.pitch_bend(0, 8192)
    minimal.close()

    null_backend = NullAudioBackend()
    null_backend.start({0: 16})
    null_backend.note_on(0, 60, 80)
    null_backend.note_off(0, 60)
    null_backend.pitch_bend(0, 9000)
    null_backend.close()
    assert "Null" in null_backend.backend_name

    # PygameMidiBackend & LinuxFluidSynthBackend construction is safe without hardware
    pg_backend = PygameMidiBackend()
    assert pg_backend.backend_name == "pygame.midi"
    assert issubclass(WinmmMidiBackend, AudioBackend)
    assert issubclass(PygameMixerSynthBackend, AudioBackend)

    fs_backend = LinuxFluidSynthBackend()
    assert "FluidSynth" in fs_backend.backend_name


# =============================================================================
# 2. STRICT C NATURAL MINOR & VERIDIS QUO 8-BAR ARCHITECTURE
# =============================================================================
def test_all_midi_notes_and_8_bar_veridis_quo_chords_strictly_c_minor() -> None:
    for raw_note in range(0, 128):
        snapped = CMinorAudioEngine.snap_to_c_minor(raw_note)
        assert 0 <= snapped <= 127
        assert (snapped % 12) in CMinorAudioEngine.C_MINOR_PITCH_CLASSES

    for color, pitches in CMinorAudioEngine.COLOR_PITCH_MAP.items():
        for pitch in pitches:
            assert (pitch % 12) in CMinorAudioEngine.C_MINOR_PITCH_CLASSES, f"{color} pitch {pitch}"

    assert len(VERIDIS_QUO_PROGRESSION) == 8
    assert len(CMinorAudioEngine.HARMONIC_PROGRESSION) == 8

    # Circle-of-fifths C-minor bass roots: C, F, Bb, Eb, Ab, F, D, G
    expected_root_pcs = (0, 5, 10, 3, 8, 5, 2, 7)
    actual_root_pcs = tuple(chord.bass_note % 12 for chord in VERIDIS_QUO_PROGRESSION)
    assert actual_root_pcs == expected_root_pcs

    for chord in VERIDIS_QUO_PROGRESSION:
        assert (chord.bass_note % 12) in CMinorAudioEngine.C_MINOR_PITCH_CLASSES
        for tone in chord.pad_tones + chord.arp_tones:
            assert (tone % 12) in CMinorAudioEngine.C_MINOR_PITCH_CLASSES, (
                f"{chord.name} tone {tone} is outside C Natural Minor"
            )


def test_color_octave_ordering_and_90_bpm_beat_grid() -> None:
    assert CMinorAudioEngine.BPM == pytest.approx(90.0)
    assert CMinorAudioEngine.BEAT_DURATION_SEC == pytest.approx(60.0 / 90.0)
    assert CMinorAudioEngine.SIXTEENTH_DURATION_SEC == pytest.approx((60.0 / 90.0) / 4.0)

    cool_avg = float(np.mean(CMinorAudioEngine.COLOR_PITCH_MAP["Blue"]))
    warm_avg = float(np.mean(CMinorAudioEngine.COLOR_PITCH_MAP["Red"]))
    assert cool_avg < warm_avg

    whole_steps = [s for s in range(16) if CMinorAudioEngine.should_trigger_data_step(0.05, s)]
    half_steps = [s for s in range(16) if CMinorAudioEngine.should_trigger_data_step(0.20, s)]
    eighth_steps = [s for s in range(16) if CMinorAudioEngine.should_trigger_data_step(0.50, s)]
    sixteenth_steps = [s for s in range(16) if CMinorAudioEngine.should_trigger_data_step(0.90, s)]

    assert whole_steps == [0]
    assert half_steps == [0, 8]
    assert eighth_steps == [0, 2, 4, 6, 8, 10, 12, 14]
    assert sixteenth_steps == list(range(16))


def test_intensity_maps_to_velocity_volume_and_filter_ccs() -> None:
    v_low = CMinorAudioEngine.intensity_to_velocity(0.0)
    v_high = CMinorAudioEngine.intensity_to_velocity(1.0)
    vol_low = CMinorAudioEngine.intensity_to_volume(0.0)
    vol_high = CMinorAudioEngine.intensity_to_volume(1.0)

    assert v_low < v_high
    assert 0 <= v_low <= 127 and 0 <= v_high <= 127
    assert vol_low < vol_high
    assert 0.0 <= vol_low <= 1.0 and 0.0 <= vol_high <= 1.0

    low_res = AnalysisResult(dominant_color="Orange", intensity=0.1, density=0.5)
    high_res = AnalysisResult(dominant_color="Orange", intensity=0.95, density=0.5)
    low_ccs = compute_expression_ccs(low_res, step_16=0)
    high_ccs = compute_expression_ccs(high_res, step_16=0)
    assert high_ccs["brightness"] > low_ccs["brightness"]
    assert high_ccs["expression"] > low_ccs["expression"]


# =============================================================================
# 3. MEASURABLE MUSICAL IMPACT OF SHAPE, SPOTTINESS, SWIRLINESS, SENSOR_SIZE
# =============================================================================
def test_shape_changes_melodic_contour_measurably() -> None:
    contours: dict[str, list[int]] = {}
    for shape in ("uniform", "linear", "circular", "organic"):
        res = AnalysisResult(
            dominant_color="Orange",
            intensity=0.75,
            density=0.80,
            shape=shape,
            shape_score=0.85,
        )
        pitches = [
            CMinorAudioEngine.plan_musical_step(res, step_idx, bar_index=0)[1]
            for step_idx in range(8)
        ]
        for pitch in pitches:
            assert (pitch % 12) in CMinorAudioEngine.C_MINOR_PITCH_CLASSES
        contours[shape] = pitches

    # All four shape modes produce distinct musical contours
    assert len({tuple(seq) for seq in contours.values()}) == 4


def test_swirliness_transforms_arpeggio_chorus_modulation_and_pitch_bend() -> None:
    calm = AnalysisResult(
        dominant_color="Blue",
        intensity=0.70,
        density=0.85,
        shape="circular",
        swirliness=0.0,
    )
    swirly = AnalysisResult(
        dominant_color="Blue",
        intensity=0.70,
        density=0.85,
        shape="circular",
        swirliness=0.90,
    )

    calm_arp = [plan_baroque_arpeggio(calm, 0, s, s) for s in range(8)]
    swirl_arp = [plan_baroque_arpeggio(swirly, 0, s, s) for s in range(8)]
    assert calm_arp != swirl_arp

    for item in swirl_arp:
        assert item is not None
        assert (item[0] % 12) in CMinorAudioEngine.C_MINOR_PITCH_CLASSES

    calm_ccs = compute_expression_ccs(calm, step_16=4)
    swirl_ccs = compute_expression_ccs(swirly, step_16=4)
    assert swirl_ccs["chorus"] > calm_ccs["chorus"]
    assert swirl_ccs["modulation"] > calm_ccs["modulation"]
    assert calm_ccs["pitch_bend"] == 8192
    assert swirl_ccs["pitch_bend"] != 8192


def test_spottiness_adds_syncopated_accents_shortens_gate_and_sparks_drums() -> None:
    smooth = AnalysisResult(
        dominant_color="Green",
        intensity=0.65,
        density=0.55,
        spottiness=0.0,
    )
    spotty = AnalysisResult(
        dominant_color="Green",
        intensity=0.65,
        density=0.55,
        spottiness=0.90,
    )

    smooth_accents = [plan_spottiness_accents(smooth, 0, s) for s in range(16)]
    spotty_accents = [plan_spottiness_accents(spotty, 0, s) for s in range(16)]
    assert all(a is None for a in smooth_accents)
    active_spots = [a for a in spotty_accents if a is not None]
    assert len(active_spots) >= 4
    for pitch, vel in active_spots:
        assert (pitch % 12) in CMinorAudioEngine.C_MINOR_PITCH_CLASSES
        assert 1 <= vel <= 127

    # High spottiness shortens melodic gate duration
    dur_smooth = CMinorAudioEngine.plan_musical_step(smooth, 0)[5]
    dur_spotty = CMinorAudioEngine.plan_musical_step(spotty, 0)[5]
    assert dur_spotty < dur_smooth

    # High spottiness injects tambourine / conga sparkles on Channel 9
    smooth_drums = {
        hit[0]
        for s in range(16)
        for hit in CMinorAudioEngine.plan_ambient_drums(0.55, 0.65, s, spottiness=0.0)
    }
    spotty_drums = {
        hit[0]
        for s in range(16)
        for hit in CMinorAudioEngine.plan_ambient_drums(0.55, 0.65, s, spottiness=0.90)
    }
    assert CMinorAudioEngine.DRUM_TAMBOURINE not in smooth_drums
    assert CMinorAudioEngine.DRUM_TAMBOURINE in spotty_drums
    assert CMinorAudioEngine.DRUM_CONGA_HI in spotty_drums


def test_sensor_size_scales_voicing_spread_bass_depth_and_reverb() -> None:
    tight = AnalysisResult(
        dominant_color="Blue",
        intensity=0.80,
        density=0.50,
        sensor_size=20,
    )
    wide = AnalysisResult(
        dominant_color="Blue",
        intensity=0.80,
        density=0.50,
        sensor_size=120,
    )

    _, _, tight_tones = plan_chord_voicing(0, 0.80, sensor_size=20)
    _, _, wide_tones = plan_chord_voicing(0, 0.80, sensor_size=120)
    assert len(tight_tones) == 2
    assert len(wide_tones) >= 5
    for t in tight_tones + wide_tones:
        assert (t % 12) in CMinorAudioEngine.C_MINOR_PITCH_CLASSES

    tight_bass = plan_bass_step(0, 0.50, 0.80, sensor_size=20, step_16=0)
    wide_bass = plan_bass_step(0, 0.50, 0.80, sensor_size=120, step_16=0)
    assert len(wide_bass) > len(tight_bass)

    tight_ccs = compute_expression_ccs(tight, step_16=0)
    wide_ccs = compute_expression_ccs(wide, step_16=0)
    assert wide_ccs["reverb"] > tight_ccs["reverb"]

    tight_dur = CMinorAudioEngine.plan_musical_step(tight, 0)[5]
    wide_dur = CMinorAudioEngine.plan_musical_step(wide, 0)[5]
    assert tight_dur <= wide_dur


# =============================================================================
# 4. SEQUENCER ENGINE INTEGRATION TESTS (RecordingAudioBackend)
# =============================================================================
def test_audio_engine_emits_veridis_quo_layers_strictly_in_c_minor() -> None:
    backend = RecordingAudioBackend()
    engine = CMinorAudioEngine(backend=backend, enable_audio=True, ambient_drums=True)
    engine.start()
    try:
        engine.update_analysis(
            AnalysisResult(
                dominant_color="Red",
                intensity=0.88,
                density=0.92,
                shape="circular",
                shape_score=0.90,
                spottiness=0.80,
                swirliness=0.85,
                sensor_size=110,
                sensor_bbox=(10, 10, 120, 120),
                cursor_xy=(65, 65),
                touching=True,
            )
        )
        time.sleep(0.65)
    finally:
        engine.stop()

    assert backend.started and backend.closed
    assert backend.programs[PAD_CHANNEL] == 16
    assert backend.programs[BASS_CHANNEL] == 38
    assert backend.programs[ARP_CHANNEL] == 80
    assert backend.programs[WARM_PAD_CHANNEL] == 89
    assert backend.programs[PLUCK_CHANNEL] == 12
    assert backend.programs[CHURCH_ORGAN_CHANNEL] == 19

    assert len(backend.ccs) > 0
    assert any(ch == ARP_CHANNEL and bend != 8192 for ch, bend in backend.bends)

    melodic_ons = [
        evt for evt in backend.events if evt[0] == "on" and evt[1] != CMinorAudioEngine.DRUM_CHANNEL
    ]
    drum_ons = [
        evt for evt in backend.events if evt[0] == "on" and evt[1] == CMinorAudioEngine.DRUM_CHANNEL
    ]
    assert len(melodic_ons) >= 6
    assert len(drum_ons) >= 2

    channels_used = {evt[1] for evt in melodic_ons}
    assert PAD_CHANNEL in channels_used
    assert BASS_CHANNEL in channels_used
    assert ARP_CHANNEL in channels_used
    assert CHURCH_ORGAN_CHANNEL in channels_used

    for _kind, _ch, pitch, _velocity in melodic_ons:
        assert (pitch % 12) in CMinorAudioEngine.C_MINOR_PITCH_CLASSES


def test_toggle_drums_and_standby_silence_behavior() -> None:
    backend = RecordingAudioBackend()
    engine = CMinorAudioEngine(backend=backend, enable_audio=True, ambient_drums=True)
    assert engine.ambient_drums_enabled is True
    assert engine.toggle_drums() is False
    assert engine.ambient_drums_enabled is False

    engine.start()
    try:
        engine.update_analysis(
            AnalysisResult(
                dominant_color="Green",
                intensity=0.60,
                density=0.55,
                shape="linear",
                spottiness=0.70,
                sensor_size=40,
                sensor_bbox=(0, 0, 40, 40),
                cursor_xy=(20, 20),
                touching=True,
            )
        )
        time.sleep(0.30)
        engine.update_analysis(
            AnalysisResult(
                dominant_color="Green",
                intensity=0.0,
                density=0.0,
                touching=False,
            )
        )
        time.sleep(0.25)
    finally:
        engine.stop()

    drum_ons = [
        evt for evt in backend.events if evt[0] == "on" and evt[1] == CMinorAudioEngine.DRUM_CHANNEL
    ]
    assert drum_ons == []
    assert "Standby" in engine.telemetry_text()
