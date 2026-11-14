"""Unit tests for Earth Jukebox Computer Vision sensor & analyzer modules."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from earth_jukebox.vision import (  # noqa: E402
    MIN_SENSOR_BOX_SIZE,
    CVAnalyzer,
    extract_sensor_box,
)
from earth_jukebox.vision.analyzer import AnalysisResult, DatasetEntry  # noqa: E402


# =============================================================================
# SENSOR BOX EXTRACTION TESTS
# =============================================================================
@pytest.mark.parametrize("box_size", [16, 40, 120, 999])
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
def test_extract_sensor_box_edge_safety_and_variable_sizes(
    cursor_xy: tuple[int, int],
    box_size: int,
) -> None:
    image = np.zeros((350, 630, 3), dtype=np.uint8)
    patch, bbox = extract_sensor_box(image, cursor_xy[0], cursor_xy[1], box_size=box_size)
    x1, y1, x2, y2 = bbox

    expected_w = min(630, max(MIN_SENSOR_BOX_SIZE, box_size))
    expected_h = min(350, max(MIN_SENSOR_BOX_SIZE, box_size))

    assert patch.shape == (expected_h, expected_w, 3)
    assert 0 <= x1 < x2 <= 630
    assert 0 <= y1 < y2 <= 350
    assert (x2 - x1) == expected_w
    assert (y2 - y1) == expected_h


def test_extract_sensor_box_clamps_negative_and_tiny_box_sizes() -> None:
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    for bad_size in (-20, 0, 4):
        patch, bbox = extract_sensor_box(image, 50, 50, box_size=bad_size)
        assert patch.shape == (MIN_SENSOR_BOX_SIZE, MIN_SENSOR_BOX_SIZE, 3)
        assert bbox[2] - bbox[0] == MIN_SENSOR_BOX_SIZE
        assert bbox[3] - bbox[1] == MIN_SENSOR_BOX_SIZE


def test_extract_sensor_box_small_image_and_empty_error() -> None:
    tiny = np.full((18, 24, 3), 120, dtype=np.uint8)
    patch, bbox = extract_sensor_box(tiny, 0, 0, box_size=40)
    assert patch.shape == (18, 24, 3)
    assert bbox == (0, 0, 24, 18)

    with pytest.raises(ValueError):
        extract_sensor_box(np.empty((0, 0, 3), dtype=np.uint8), 0, 0)


# =============================================================================
# COLOR, SALIENCE, INTENSITY, AND DENSITY TESTS
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
    assert result.shape == "uniform"
    assert result.sensor_size == 40


def test_cv_analyzer_detects_extreme_crimson_and_dust_salience() -> None:
    analyzer = CVAnalyzer()

    # Deep crimson FWI "Extreme" core (HSV ~ [154, 223, 26]) -> Red
    crimson_hsv = np.full((40, 40, 3), (154, 223, 26), dtype=np.uint8)
    crimson_bgr = cv2.cvtColor(crimson_hsv, cv2.COLOR_HSV2BGR)
    crimson_res = analyzer.analyze_patch(crimson_bgr, (0, 0, 40, 40), (20, 20))
    assert crimson_res.dominant_color == "Red"

    # Purple/violet Saharan dust plume (HSV ~ [138, 73, 198]) -> Orange
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
    assert ocean_res.shape == "uniform"

    chaotic = np.full((40, 40, 3), (20, 20, 180), dtype=np.uint8)
    for radius in range(3, 24, 4):
        cv2.circle(chaotic, (20, 20), radius, (240, 240, 240), 1)
    for offset in range(4, 40, 6):
        cv2.line(chaotic, (0, offset), (40, 40 - offset), (0, 220, 255), 1)
    chaotic_res = analyzer.analyze_patch(chaotic, (0, 0, 40, 40), (20, 20))
    assert chaotic_res.density > 0.7


# =============================================================================
# SPATIAL & MORPHOLOGICAL FEATURE TESTS (SHAPE, SPOTTINESS, SWIRLINESS)
# =============================================================================
def test_cv_analyzer_shape_classification_linear_circular_uniform_organic() -> None:
    analyzer = CVAnalyzer()

    # Uniform smooth patch
    flat = np.full((48, 48, 3), (180, 90, 20), dtype=np.uint8)
    flat_res = analyzer.analyze_patch(flat, (0, 0, 48, 48), (24, 24))
    assert flat_res.shape == "uniform"
    assert flat_res.shape_score <= 0.2

    # Strong directional diagonal front -> linear with high shape_score
    linear_patch = np.full((60, 60, 3), (160, 70, 15), dtype=np.uint8)
    cv2.line(linear_patch, (0, 59), (59, 0), (245, 245, 245), 4)
    linear_res = analyzer.analyze_patch(linear_patch, (0, 0, 60, 60), (30, 30))
    assert linear_res.shape == "linear"
    assert linear_res.shape_score > 0.7

    # Enclosed compact circular hotspot -> circular with high circularity
    circle_patch = np.full((60, 60, 3), (20, 20, 20), dtype=np.uint8)
    cv2.circle(circle_patch, (30, 30), 15, (0, 50, 245), -1)
    circle_res = analyzer.analyze_patch(circle_patch, (0, 0, 60, 60), (30, 30))
    assert circle_res.shape == "circular"
    assert circle_res.shape_score > 0.7

    # Irregular multi-spot organic texture -> organic
    rng = np.random.default_rng(123)
    organic_patch = np.full((60, 60, 3), (15, 25, 15), dtype=np.uint8)
    for _ in range(24):
        cx = int(rng.integers(6, 54))
        cy = int(rng.integers(6, 54))
        cv2.circle(organic_patch, (cx, cy), int(rng.integers(2, 5)), (0, 120, 245), -1)
    organic_res = analyzer.analyze_patch(organic_patch, (0, 0, 60, 60), (30, 30))
    assert organic_res.shape == "organic"
    assert 0.2 <= organic_res.shape_score <= 1.0


def test_cv_analyzer_spottiness_high_vs_low() -> None:
    analyzer = CVAnalyzer()

    # Smooth ice / ocean -> near 0.0
    smooth = np.full((60, 60, 3), (230, 235, 240), dtype=np.uint8)
    smooth_res = analyzer.analyze_patch(smooth, (0, 0, 60, 60), (30, 30))
    assert smooth_res.spottiness < 0.1

    # Straight stripes -> low spottiness
    stripes = np.zeros((60, 60, 3), dtype=np.uint8)
    for y in range(0, 60, 10):
        stripes[y : y + 5, :] = (220, 220, 220)
    stripes_res = analyzer.analyze_patch(stripes, (0, 0, 60, 60), (30, 30))
    assert stripes_res.spottiness < 0.2

    # Speckled wildfire / urban lights cluster -> high (> 0.6)
    rng = np.random.default_rng(77)
    speckled = np.full((60, 60, 3), (12, 12, 12), dtype=np.uint8)
    for _ in range(28):
        cx = int(rng.integers(4, 56))
        cy = int(rng.integers(4, 56))
        cv2.circle(speckled, (cx, cy), 2, (10, 70, 250), -1)
    speckled_res = analyzer.analyze_patch(speckled, (0, 0, 60, 60), (30, 30))
    assert speckled_res.spottiness > 0.6


def test_cv_analyzer_swirliness_vortex_vs_stripes_and_flat() -> None:
    analyzer = CVAnalyzer()

    # Flat ocean -> near 0.0
    flat = np.full((80, 80, 3), (180, 70, 15), dtype=np.uint8)
    flat_res = analyzer.analyze_patch(flat, (0, 0, 80, 80), (40, 40))
    assert flat_res.swirliness < 0.1

    # Straight horizontal stripes -> low swirliness
    stripes = np.zeros((80, 80, 3), dtype=np.uint8)
    for y in range(0, 80, 10):
        stripes[y : y + 5, :] = (230, 230, 230)
    stripes_res = analyzer.analyze_patch(stripes, (0, 0, 80, 80), (40, 40))
    assert stripes_res.swirliness < 0.2

    # Spiral hurricane / ocean eddy vortex -> high swirliness (> 0.6)
    vortex = np.full((80, 80, 3), (25, 25, 35), dtype=np.uint8)
    points = []
    for deg in range(0, 960, 5):
        radius = 3.0 + deg * 0.034
        rad = math.radians(deg)
        points.append((int(40 + radius * math.cos(rad)), int(40 + radius * math.sin(rad))))
    for p1, p2 in zip(points, points[1:]):
        cv2.line(vortex, p1, p2, (245, 245, 250), 2)
    vortex_res = analyzer.analyze_patch(vortex, (0, 0, 80, 80), (40, 40))
    assert vortex_res.swirliness > 0.6


def test_cv_analyzer_handles_empty_grayscale_bgra_and_explicit_sensor_size() -> None:
    analyzer = CVAnalyzer()

    empty_res = analyzer.analyze_patch(
        np.empty((0, 0, 3), dtype=np.uint8), (0, 0, 0, 0), (0, 0), sensor_size=64
    )
    assert empty_res.dominant_color == "Black"
    assert empty_res.shape == "uniform"
    assert empty_res.sensor_size == 64

    gray_patch = np.full((24, 24), 240, dtype=np.uint8)
    gray_res = analyzer.analyze_patch(gray_patch, (0, 0, 24, 24), (12, 12), sensor_size=24)
    assert gray_res.dominant_color == "White"
    assert gray_res.sensor_size == 24

    bgra_patch = np.full((32, 32, 4), (0, 0, 255, 255), dtype=np.uint8)
    bgra_res = analyzer.analyze_patch(bgra_patch, (0, 0, 32, 32), (16, 16))
    assert bgra_res.dominant_color == "Red"
    assert bgra_res.sensor_size == 32


# =============================================================================
# SPEECH FORMATTING TESTS
# =============================================================================
def test_format_speech_summary_includes_morphology_and_dataset_meaning() -> None:
    analyzer = CVAnalyzer()
    entry = DatasetEntry(
        name="North America Fire Weather Index",
        path=Path("data/images/fwi-conus-2026-08-01.png"),
        color_meanings=(("Red", "very high to extreme fire weather danger"),),
    )

    swirling_fire = AnalysisResult(
        dominant_color="Red",
        intensity=0.84,
        density=0.88,
        shape="circular",
        shape_score=0.85,
        spottiness=0.72,
        swirliness=0.68,
        sensor_size=40,
        sensor_bbox=(0, 0, 40, 40),
        cursor_xy=(20, 20),
        touching=True,
    )
    summary = analyzer.format_speech_summary(swirling_fire, dataset=entry)
    assert "High density" in summary
    assert "red zone" in summary
    assert "84 percent intensity" in summary
    assert "circular" in summary
    assert "swirling" in summary
    assert "speckled" in summary
    assert "very high to extreme fire weather danger" in summary

    calm_ocean = AnalysisResult(
        dominant_color="Blue",
        intensity=0.40,
        density=0.10,
        shape="uniform",
        shape_score=0.0,
        spottiness=0.0,
        swirliness=0.0,
        sensor_size=40,
        sensor_bbox=(0, 0, 40, 40),
        cursor_xy=(20, 20),
    )
    calm_summary = analyzer.format_speech_summary(calm_ocean)
    assert "Low density blue zone, 40 percent intensity" in calm_summary
    assert "swirling" not in calm_summary
    assert "speckled" not in calm_summary
