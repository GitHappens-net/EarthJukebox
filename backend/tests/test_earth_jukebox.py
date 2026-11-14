"""Unit tests for the Earth Jukebox modular pipeline and NASA EIC upgrades."""

from __future__ import annotations

import queue
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from earth_jukebox import (  # noqa: E402
    AccessibilityTTS,
    AnalysisResult,
    AudioBackend,
    CMinorAudioEngine,
    CVAnalyzer,
    DatasetManager,
    TouchPoint,
    extract_sensor_box,
    push_latest_point,
)


class RecordingAudioBackend(AudioBackend):
    """Test double capturing MIDI events emitted from CMinorAudioEngine's thread."""

    def __init__(self) -> None:
        self.started = False
        self.closed = False
        self.programs: dict[int, int] = {}
        self.events: list[tuple[str, int, int, int]] = []
        self.ccs: list[tuple[int, int, int]] = []

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

    def close(self) -> None:
        self.closed = True


# =============================================================================
# MODULE 1 TESTS: EDGE-SAFE SENSOR BOX, TOUCHPOINT QUEUE, DATASETS
# =============================================================================
@pytest.mark.parametrize(
    "cursor_xy",
    [
        (0, 0),
        (630, 350),
        (629, 349),
        (-50, -25),
        (9999, 9999),
        (0, 349),
        (629, 0),
        (315, 175),
    ],
)
def test_extract_sensor_box_edge_safety(cursor_xy: tuple[int, int]) -> None:
    image = np.zeros((350, 630, 3), dtype=np.uint8)
    patch, bbox = extract_sensor_box(image, cursor_xy[0], cursor_xy[1], box_size=40)
    x1, y1, x2, y2 = bbox

    assert patch.shape == (40, 40, 3)
    assert 0 <= x1 < x2 <= 630
    assert 0 <= y1 < y2 <= 350


def test_extract_sensor_box_small_image() -> None:
    tiny = np.full((18, 24, 3), 120, dtype=np.uint8)
    patch, bbox = extract_sensor_box(tiny, 0, 0, box_size=40)
    assert patch.shape == (18, 24, 3)
    assert bbox == (0, 0, 24, 18)


def test_push_latest_point_keeps_newest_in_bounded_queue() -> None:
    q: queue.Queue[TouchPoint] = queue.Queue(maxsize=2)
    p1 = TouchPoint(1, 1, 0.1, 0.1, True)
    p2 = TouchPoint(2, 2, 0.2, 0.2, True)
    p3 = TouchPoint(3, 3, 0.3, 0.3, True)

    push_latest_point(q, p1)
    push_latest_point(q, p2)
    push_latest_point(q, p3)

    remaining = []
    while not q.empty():
        remaining.append(q.get_nowait())
    assert remaining[-1] == p3


def test_dataset_manager_synthesizes_and_cycles(tmp_path: Path) -> None:
    world_src = BACKEND_DIR.parent / "3D files" / "World Map.jpg"
    manager = DatasetManager(datasets_dir=tmp_path / "datasets", world_map_path=world_src)
    entries = manager.load_datasets()

    assert len(entries) >= 3
    first = manager.current_dataset()
    second = manager.next_dataset()
    third = manager.next_dataset()
    wrapped = manager.next_dataset()

    assert first != second
    assert second != third
    assert wrapped == first

    img = manager.load_current_image()
    assert img.ndim == 3 and img.size > 0


def test_dataset_manager_loads_real_nasa_eic_images() -> None:
    manager = DatasetManager()
    entries = manager.load_datasets()
    assert len(entries) >= 7
    for idx in range(len(entries)):
        entry = manager.current_dataset()
        img = manager.load_current_image()
        h, w = img.shape[:2]
        assert 40 <= w <= DatasetManager.MAX_WORKING_WIDTH
        assert 40 <= h <= DatasetManager.MAX_WORKING_HEIGHT
        assert entry.summary
        manager.next_dataset()


# =============================================================================
# MODULE 2 TESTS: COMPUTER VISION ANALYSIS
# =============================================================================
@pytest.mark.parametrize(
    ("bgr", "expected_color"),
    [
        ((0, 0, 255), "Red"),
        ((0, 140, 255), "Orange"),
        ((0, 200, 0), "Green"),
        ((235, 60, 10), "Blue"),
        ((10, 10, 10), "Black"),
        ((245, 245, 245), "White"),
    ],
)
def test_cv_analyzer_dominant_color_categories(
    bgr: tuple[int, int, int], expected_color: str
) -> None:
    analyzer = CVAnalyzer()
    patch = np.full((40, 40, 3), bgr, dtype=np.uint8)
    result = analyzer.analyze_patch(patch, (0, 0, 40, 40), (20, 20))

    assert result.dominant_color == expected_color
    assert 0.0 <= result.intensity <= 1.0
    assert 0.0 <= result.density <= 1.0
    assert result.density < 0.3


def test_cv_analyzer_detects_extreme_crimson_and_streamline_salience() -> None:
    analyzer = CVAnalyzer()

    # Deep crimson FWI "Extreme" core (HSV ~ [154, 223, 26])
    crimson_hsv = np.full((40, 40, 3), (154, 223, 26), dtype=np.uint8)
    crimson_bgr = cv2.cvtColor(crimson_hsv, cv2.COLOR_HSV2BGR)
    crimson_res = analyzer.analyze_patch(crimson_bgr, (0, 0, 40, 40), (20, 20))
    assert crimson_res.dominant_color == "Red"

    # Purple/violet Saharan dust plume (HSV ~ [138, 73, 198]) -> elevated Orange aerosol
    dust_hsv = np.full((40, 40, 3), (138, 73, 198), dtype=np.uint8)
    dust_bgr = cv2.cvtColor(dust_hsv, cv2.COLOR_HSV2BGR)
    dust_res = analyzer.analyze_patch(dust_bgr, (0, 0, 40, 40), (20, 20))
    assert dust_res.dominant_color == "Orange"

    # 30% bright green ocean streamlines over 70% dark background -> Green salience
    streamline = np.full((40, 40, 3), (15, 15, 15), dtype=np.uint8)
    streamline[:12, :] = (20, 210, 30)
    stream_res = analyzer.analyze_patch(streamline, (0, 0, 40, 40), (20, 20))
    assert stream_res.dominant_color == "Green"


def test_cv_analyzer_density_distinguishes_smooth_ocean_vs_chaotic_storm() -> None:
    analyzer = CVAnalyzer()

    ocean = np.zeros((40, 40, 3), dtype=np.uint8)
    for x in range(40):
        ocean[:, x] = (200 - x, 80, 15)
    ocean_res = analyzer.analyze_patch(ocean, (0, 0, 40, 40), (20, 20))
    assert ocean_res.dominant_color == "Blue"
    assert ocean_res.density < 0.3

    chaotic = np.full((40, 40, 3), (20, 20, 180), dtype=np.uint8)
    for radius in range(3, 24, 4):
        cv2.circle(chaotic, (20, 20), radius, (240, 240, 240), 1)
    for offset in range(4, 40, 6):
        cv2.line(chaotic, (0, offset), (40, 40 - offset), (0, 220, 255), 1)
    chaotic_res = analyzer.analyze_patch(chaotic, (0, 0, 40, 40), (20, 20))
    assert chaotic_res.density > 0.7


# =============================================================================
# MODULE 3 TESTS: 90 BPM BEAT-GRID C-MINOR AUDIO ENGINE & AMBIENT DRUMS
# =============================================================================
def test_all_midi_notes_and_chords_strictly_snap_to_c_minor() -> None:
    for raw_note in range(0, 128):
        snapped = CMinorAudioEngine.snap_to_c_minor(raw_note)
        assert 0 <= snapped <= 127
        assert (snapped % 12) in CMinorAudioEngine.C_MINOR_PITCH_CLASSES

    for color, pitches in CMinorAudioEngine.COLOR_PITCH_MAP.items():
        for pitch in pitches:
            assert (pitch % 12) in CMinorAudioEngine.C_MINOR_PITCH_CLASSES, (
                f"{color} pitch {pitch} is outside C natural minor"
            )

    for chord_name, bass, tones in CMinorAudioEngine.HARMONIC_PROGRESSION:
        assert (bass % 12) in CMinorAudioEngine.C_MINOR_PITCH_CLASSES, f"{chord_name} bass out of scale"
        for tone in tones:
            assert (tone % 12) in CMinorAudioEngine.C_MINOR_PITCH_CLASSES, f"{chord_name} tone out of scale"


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


def test_intensity_maps_to_velocity_and_volume() -> None:
    v_low = CMinorAudioEngine.intensity_to_velocity(0.0)
    v_high = CMinorAudioEngine.intensity_to_velocity(1.0)
    vol_low = CMinorAudioEngine.intensity_to_volume(0.0)
    vol_high = CMinorAudioEngine.intensity_to_volume(1.0)

    assert v_low < v_high
    assert 0 <= v_low <= 127 and 0 <= v_high <= 127
    assert vol_low < vol_high
    assert 0.0 <= vol_low <= 1.0 and 0.0 <= vol_high <= 1.0


def test_audio_engine_emits_quantized_melody_pad_and_ambient_drums() -> None:
    backend = RecordingAudioBackend()
    engine = CMinorAudioEngine(backend=backend, enable_audio=True, ambient_drums=True)
    engine.start()
    try:
        engine.update_analysis(
            AnalysisResult(
                dominant_color="Red",
                intensity=0.85,
                density=0.92,
                sensor_bbox=(10, 10, 50, 50),
                cursor_xy=(30, 30),
                touching=True,
            )
        )
        time.sleep(0.45)
    finally:
        engine.stop()

    assert backend.started and backend.closed
    assert len(backend.ccs) > 0

    melodic_ons = [
        evt for evt in backend.events if evt[0] == "on" and evt[1] != CMinorAudioEngine.DRUM_CHANNEL
    ]
    drum_ons = [
        evt for evt in backend.events if evt[0] == "on" and evt[1] == CMinorAudioEngine.DRUM_CHANNEL
    ]
    assert len(melodic_ons) >= 3
    assert len(drum_ons) >= 1

    for _kind, _ch, pitch, _velocity in melodic_ons:
        assert (pitch % 12) in CMinorAudioEngine.C_MINOR_PITCH_CLASSES


def test_toggle_drums_disables_channel_9_hits() -> None:
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
                sensor_bbox=(0, 0, 40, 40),
                cursor_xy=(20, 20),
                touching=True,
            )
        )
        time.sleep(0.25)
    finally:
        engine.stop()

    drum_ons = [
        evt for evt in backend.events if evt[0] == "on" and evt[1] == CMinorAudioEngine.DRUM_CHANNEL
    ]
    assert drum_ons == []


# =============================================================================
# MODULE 4 TESTS: ACCESSIBILITY & TTS
# =============================================================================
def test_speech_summary_uses_dataset_context_and_tts_lifecycle() -> None:
    analyzer = CVAnalyzer()
    manager = DatasetManager()
    manager.load_datasets()
    entry = manager.current_dataset()

    res = AnalysisResult(
        dominant_color="Red",
        intensity=0.84,
        density=0.88,
        sensor_bbox=(0, 0, 40, 40),
        cursor_xy=(20, 20),
        touching=True,
    )
    summary = analyzer.format_speech_summary(res, dataset=entry)
    assert "High density" in summary
    assert "red zone" in summary
    assert "84 percent intensity" in summary

    tts = AccessibilityTTS(enabled=False)
    tts.start()
    tts.speak(AccessibilityTTS.AUDIO_LEGEND_TEXT)
    tts.stop()
