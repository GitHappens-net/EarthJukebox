"""Computer vision feature extraction for NASA Earth visualizations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from earth_jukebox.models import AnalysisResult, DatasetEntry
except ImportError:  # pragma: no cover - standalone fallback before models.py merge

    @dataclass(frozen=True)
    class AnalysisResult:
        """Computer-vision metrics extracted from the cursor sensor box."""

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

    @dataclass(frozen=True)
    class DatasetEntry:
        """Metadata and scientific context for a NASA Earth visualization."""

        name: str
        path: Path
        category: str = "NASA Earth Observation"
        unit: str = ""
        summary: str = ""
        color_meanings: tuple[tuple[str, str], ...] = ()

        def meaning_for(self, color: str) -> Optional[str]:
            """Look up dataset-specific scientific meaning for a dominant color."""
            for key, description in self.color_meanings:
                if key == color:
                    return description
            return None


class CVAnalyzer:
    """Extracts color, intensity, edge density, and spatial morphology from a patch."""

    COLOR_CATEGORIES = ("Black", "White", "Red", "Orange", "Green", "Blue")
    CHROMATIC_CATEGORIES = ("Red", "Orange", "Green", "Blue")
    SHAPE_CATEGORIES = ("uniform", "linear", "circular", "organic")
    DENSITY_LABELS = (
        (0.30, "Low density"),
        (0.70, "Medium density"),
        (1.01, "High density"),
    )
    ZONE_INTERPRETATIONS = {
        ("Red", "High density"): "likely active wildfire hotspot or intense thermal concentration",
        ("Red", "Medium density"): "likely warm thermal boundary or elevated heat plume",
        ("Red", "Low density"): "likely uniform high-temperature or extreme risk zone",
        ("Orange", "High density"): "likely turbulent aerosol plume or fragmented heat front",
        ("Orange", "Medium density"): "likely warm continental margin or moderate anomaly",
        ("Orange", "Low density"): "likely calm warm water, elevated CO2, or arid plateau",
        ("Green", "High density"): "likely complex terrain, mountain ridge, or sea-salt stream",
        ("Green", "Medium density"): "likely coastline or vegetated landmass",
        ("Green", "Low density"): "likely uniform plains or very low fire-risk basin",
        ("Blue", "High density"): "likely churning ocean current or storm front over water",
        ("Blue", "Medium density"): "likely continental shelf or coastal current",
        ("Blue", "Low density"): "likely calm open ocean or clean air zone",
        ("White", "High density"): "likely turbulent storm vortex or fractured sea ice",
        ("White", "Medium density"): "likely broken cloud band or snow-capped ridge",
        ("White", "Low density"): "likely solid polar ice sheet or smooth cloud deck",
        ("Black", "High density"): "likely sharp shadow boundary or night-side coastline",
        ("Black", "Medium density"): "likely dark bathymetric trench or dusk terminator",
        ("Black", "Low density"): "likely deep night-side ocean or unlit background",
    }

    _MIN_SHAPE_GRADIENT = 30.0
    _MIN_SWIRL_GRADIENT = 45.0
    _MIN_SPOT_RESPONSE = 22.0

    def __init__(
        self,
        edge_ratio_ceiling: float = 0.22,
        chromatic_salience_ratio: float = 0.18,
    ) -> None:
        self.edge_ratio_ceiling = max(0.05, float(edge_ratio_ceiling))
        self.chromatic_salience_ratio = float(np.clip(chromatic_salience_ratio, 0.05, 0.50))

    def analyze_patch(
        self,
        roi: np.ndarray,
        bbox: tuple[int, int, int, int],
        cursor_xy: tuple[int, int],
        touching: bool = True,
        sensor_size: Optional[int] = None,
    ) -> AnalysisResult:
        """Compute color, intensity, density, shape, spottiness, and swirliness."""
        if roi is None or roi.size == 0:
            resolved_size = int(sensor_size) if sensor_size is not None else 40
            return AnalysisResult(
                dominant_color="Black",
                intensity=0.0,
                density=0.0,
                shape="uniform",
                shape_score=0.0,
                spottiness=0.0,
                swirliness=0.0,
                sensor_size=max(1, resolved_size),
                sensor_bbox=bbox,
                cursor_xy=cursor_xy,
                touching=touching,
            )

        bgr = self._ensure_bgr(roi)
        resolved_size = int(sensor_size) if sensor_size is not None else int(max(bgr.shape[:2]))

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h_chan = hsv[:, :, 0]
        s_chan = hsv[:, :, 1]
        v_chan = hsv[:, :, 2]

        dominant_color = self._classify_dominant_color(h_chan, s_chan, v_chan)

        mean_v = float(np.mean(v_chan)) / 255.0
        mean_s = float(np.mean(s_chan)) / 255.0
        intensity = float(np.clip(0.65 * mean_v + 0.35 * mean_s, 0.0, 1.0))

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)
        edge_ratio = float(np.count_nonzero(edges)) / float(edges.size)
        density = float(np.clip(edge_ratio / self.edge_ratio_ceiling, 0.0, 1.0))

        bgr_f = bgr.astype(np.float32)
        gx, gy, grad_mag = self._dominant_channel_gradients(bgr_f)
        global_coherence, local_coherence_map = self._structure_tensor_coherence(gx, gy)

        spottiness = self._compute_spottiness(bgr_f)
        swirliness = self._compute_swirliness(
            gx, gy, grad_mag, global_coherence, local_coherence_map
        )
        shape, shape_score = self._classify_shape(
            bgr_f, edges, grad_mag, global_coherence, density, spottiness, swirliness
        )

        return AnalysisResult(
            dominant_color=dominant_color,
            intensity=intensity,
            density=density,
            shape=shape,
            shape_score=shape_score,
            spottiness=spottiness,
            swirliness=swirliness,
            sensor_size=max(1, resolved_size),
            sensor_bbox=bbox,
            cursor_xy=cursor_xy,
            touching=touching,
        )

    @staticmethod
    def _ensure_bgr(roi: np.ndarray) -> np.ndarray:
        if roi.ndim == 2:
            return cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        if roi.ndim == 3 and roi.shape[2] == 4:
            return cv2.cvtColor(roi, cv2.COLOR_BGRA2BGR)
        return roi

    def _classify_dominant_color(
        self,
        h_chan: np.ndarray,
        s_chan: np.ndarray,
        v_chan: np.ndarray,
    ) -> str:
        """Partition HSV pixels into canonical categories with EIC color sensitivity.

        * High-saturation deep crimson (e.g. NASA FWI 'Extreme' dark maroon) is
          captured as ``Red`` even when luminance is low.
        * Purple/violet mineral dust plumes (``130 <= H < 145``, ``S >= 50``)
          map to ``Orange`` (elevated warm aerosol alert).
        * When at least ``chromatic_salience_ratio`` (default 18%) of the patch
          contains chromatic signal, glowing filaments over dark ocean/land
          backgrounds (e.g. ECCO2 currents, GEOS aerosols) take precedence.
        """
        deep_crimson = (v_chan >= 16) & (s_chan >= 100) & ((h_chan < 12) | (h_chan >= 145))
        black_mask = (~deep_crimson) & ((v_chan < 45) | ((s_chan < 35) & (v_chan < 100)))
        white_mask = (~deep_crimson) & (~black_mask) & (s_chan < 35) & (v_chan >= 100)
        chromatic = (~black_mask) & (~white_mask)

        violet_dust = (h_chan >= 130) & (h_chan < 145) & (s_chan >= 50) & (v_chan >= 60)
        red_mask = deep_crimson | (chromatic & ((h_chan < 10) | (h_chan >= 155)))
        orange_mask = (~red_mask) & chromatic & (((h_chan >= 10) & (h_chan < 33)) | violet_dust)
        green_mask = (~red_mask) & (~orange_mask) & chromatic & (h_chan >= 33) & (h_chan < 88)
        blue_mask = (~red_mask) & (~orange_mask) & (~green_mask) & chromatic

        counts = {
            "Black": int(np.count_nonzero(black_mask)),
            "White": int(np.count_nonzero(white_mask)),
            "Red": int(np.count_nonzero(red_mask)),
            "Orange": int(np.count_nonzero(orange_mask)),
            "Green": int(np.count_nonzero(green_mask)),
            "Blue": int(np.count_nonzero(blue_mask)),
        }

        total_pixels = int(h_chan.size)
        chromatic_total = sum(counts[name] for name in self.CHROMATIC_CATEGORIES)
        if total_pixels > 0 and (chromatic_total / total_pixels) >= self.chromatic_salience_ratio:
            return max(self.CHROMATIC_CATEGORIES, key=lambda name: counts[name])
        return max(self.COLOR_CATEGORIES, key=lambda name: counts[name])

    @staticmethod
    def _dominant_channel_gradients(
        bgr_f: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Sobel gradients from the strongest-contrast BGR channel per pixel."""
        height, width = bgr_f.shape[:2]
        gx_c = cv2.Sobel(bgr_f, cv2.CV_32F, 1, 0, ksize=3)
        gy_c = cv2.Sobel(bgr_f, cv2.CV_32F, 0, 1, ksize=3)
        mag2_c = gx_c * gx_c + gy_c * gy_c
        dom_idx = np.argmax(mag2_c, axis=2)
        yy, xx = np.ogrid[:height, :width]
        gx = gx_c[yy, xx, dom_idx]
        gy = gy_c[yy, xx, dom_idx]
        grad_mag = np.hypot(gx, gy)
        return gx, gy, grad_mag

    @staticmethod
    def _structure_tensor_coherence(
        gx: np.ndarray,
        gy: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Compute global and 7x7 local structure-tensor anisotropy coherence."""
        gxx = gx * gx
        gyy = gy * gy
        gxy = gx * gy

        jxx = float(np.sum(gxx))
        jyy = float(np.sum(gyy))
        jxy = float(np.sum(gxy))
        global_coherence = math.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-6)
        global_coherence = float(np.clip(global_coherence, 0.0, 1.0))

        lxx = cv2.GaussianBlur(gxx, (7, 7), 1.5)
        lyy = cv2.GaussianBlur(gyy, (7, 7), 1.5)
        lxy = cv2.GaussianBlur(gxy, (7, 7), 1.5)
        local_coherence_map = np.sqrt((lxx - lyy) ** 2 + 4.0 * lxy * lxy) / (lxx + lyy + 1e-6)
        return global_coherence, local_coherence_map

    def _compute_spottiness(self, bgr_f: np.ndarray) -> float:
        """Measure fragmented multi-spot activity via color DoG blobs and 2D peaks."""
        height, width = bgr_f.shape[:2]
        blur_fine = cv2.GaussianBlur(bgr_f, (3, 3), 0)
        blur_coarse = cv2.GaussianBlur(bgr_f, (11, 11), 0)
        spot_response = np.linalg.norm(blur_fine - blur_coarse, axis=2)
        max_spot = float(np.max(spot_response))
        if max_spot <= self._MIN_SPOT_RESPONSE:
            return 0.0

        threshold = max(self._MIN_SPOT_RESPONSE, 0.35 * max_spot)
        spot_mask = (spot_response > threshold).astype(np.uint8)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(spot_mask)

        max_blob_area = max(25.0, 0.08 * float(height * width))
        valid_spots = 0
        compact_mask = np.zeros((height, width), dtype=np.uint8)
        for label_idx in range(1, n_labels):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            bw = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            bh = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            aspect = max(bw, bh) / max(1, min(bw, bh))
            if 3 <= area <= max_blob_area and aspect < 3.5:
                valid_spots += 1
                compact_mask[labels == label_idx] = 1

        if valid_spots >= 2:
            dilated = cv2.dilate(spot_response, np.ones((5, 5), dtype=np.uint8))
            peaks = (
                (spot_response >= dilated - 1e-4)
                & (compact_mask > 0)
                & (spot_response > self._MIN_SPOT_RESPONSE)
            )
            peak_labels, _, _, _ = cv2.connectedComponentsWithStats(peaks.astype(np.uint8))
            peak_count = max(0, peak_labels - 1)
        else:
            peak_count = 0

        area_scale = 40.0 / math.sqrt(float(max(1, height * width)))
        comp_score = min(1.0, max(0.0, (valid_spots * area_scale - 1.5) / 10.0))
        peak_score = min(1.0, max(0.0, (peak_count * area_scale - 1.5) / 10.0))
        return float(np.clip(0.60 * comp_score + 0.40 * peak_score, 0.0, 1.0))

    def _compute_swirliness(
        self,
        gx: np.ndarray,
        gy: np.ndarray,
        grad_mag: np.ndarray,
        global_coherence: float,
        local_coherence_map: np.ndarray,
    ) -> float:
        """Measure curling vortex flow using orientation entropy and local vs global coherence."""
        height, width = grad_mag.shape[:2]
        max_grad = float(np.max(grad_mag))
        min_strong = max(8, min(height, width) // 2)
        strong = grad_mag > max(self._MIN_SWIRL_GRADIENT, 0.22 * max_grad)
        strong_count = int(np.count_nonzero(strong))
        if max_grad < self._MIN_SWIRL_GRADIENT or strong_count < min_strong:
            return 0.0

        angles = np.mod(np.arctan2(gy[strong], gx[strong]), math.pi)
        hist, _ = np.histogram(angles, bins=12, range=(0.0, math.pi), weights=grad_mag[strong])
        hist_sum = float(np.sum(hist))
        if hist_sum <= 0.0:
            return 0.0

        probs = hist / hist_sum
        nonzero = probs[probs > 0]
        entropy = float(-np.sum(nonzero * np.log(nonzero)) / math.log(12))

        local_coherence = float(np.mean(local_coherence_map[strong]))
        curvature = 0.55 * entropy + 0.45 * (1.0 - global_coherence)
        local_factor = float(np.clip(local_coherence / 0.60, 0.20, 1.0))
        coverage = float(np.clip(strong_count / max(1.0, float(min_strong * 2)), 0.0, 1.0))
        return float(np.clip(curvature * local_factor * coverage, 0.0, 1.0))

    def _classify_shape(
        self,
        bgr_f: np.ndarray,
        edges: np.ndarray,
        grad_mag: np.ndarray,
        global_coherence: float,
        density: float,
        spottiness: float,
        swirliness: float,
    ) -> tuple[str, float]:
        """Classify patch morphology into uniform, linear, circular, or organic."""
        height, width = bgr_f.shape[:2]
        max_grad = float(np.max(grad_mag))
        min_strong = max(8, min(height, width) // 2)
        strong_count = int(
            np.count_nonzero(grad_mag > max(self._MIN_SHAPE_GRADIENT, 0.20 * max_grad))
        )

        if max_grad < self._MIN_SHAPE_GRADIENT or strong_count < min_strong:
            uniform_score = float(np.clip(float(np.mean(grad_mag)) / 40.0, 0.0, 0.15))
            return "uniform", uniform_score

        if global_coherence >= 0.58:
            return "linear", float(np.clip(global_coherence, 0.0, 1.0))

        best_circularity = self._max_contour_circularity(bgr_f, edges)
        if best_circularity >= 0.60 and spottiness < 0.55:
            return "circular", float(np.clip(best_circularity, 0.0, 1.0))

        organic_score = float(
            np.clip(0.35 + 0.45 * density + 0.20 * max(spottiness, swirliness), 0.20, 0.95)
        )
        return "organic", organic_score

    @staticmethod
    def _max_contour_circularity(bgr_f: np.ndarray, edges: np.ndarray) -> float:
        """Return the highest ``4*pi*area/perim^2`` among compact enclosed contours."""
        height, width = bgr_f.shape[:2]
        total_area = float(height * width)
        min_area = max(16.0, 0.03 * total_area)
        max_area = 0.85 * total_area

        masks: list[np.ndarray] = []
        median_bgr = np.median(bgr_f.reshape(-1, 3), axis=0)
        dist = np.linalg.norm(bgr_f - median_bgr, axis=2)
        max_dist = float(np.max(dist))
        if max_dist > 20.0:
            dist_u8 = np.clip((dist / max_dist) * 255.0, 0.0, 255.0).astype(np.uint8)
            _, otsu_mask = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            masks.append(otsu_mask)

        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
        masks.append(closed_edges)

        best_circularity = 0.0
        for mask in masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                perim = float(cv2.arcLength(contour, True))
                if min_area <= area <= max_area and perim > 1e-3:
                    circularity = min(1.0, (4.0 * math.pi * area) / (perim * perim))
                    if circularity > best_circularity:
                        best_circularity = circularity
        return best_circularity

    def density_tier(self, density: float) -> str:
        """Return a spoken/human-readable density band."""
        clamped = float(np.clip(density, 0.0, 1.0))
        for upper_bound, label in self.DENSITY_LABELS:
            if clamped < upper_bound:
                return label
        return "High density"

    def format_speech_summary(
        self,
        result: AnalysisResult,
        dataset: Optional[DatasetEntry] = None,
    ) -> str:
        """Synthesize an accessible spoken description for the current sensor box."""
        tier = self.density_tier(result.density)
        color = result.dominant_color
        intensity_pct = int(round(result.intensity * 100))

        details: list[str] = []
        if result.shape == "linear" and result.shape_score >= 0.40:
            details.append("linear boundary structure")
        elif result.shape == "circular" and result.shape_score >= 0.40:
            details.append("circular core structure")
        elif result.shape == "organic" and result.shape_score >= 0.35:
            details.append("organic textured pattern")

        if result.swirliness >= 0.55:
            details.append("strong swirling flow")
        elif result.swirliness >= 0.35:
            details.append("moderate swirling motion")

        if result.spottiness >= 0.55:
            details.append("heavily speckled hotspots")
        elif result.spottiness >= 0.35:
            details.append("scattered spotted features")

        dataset_meaning = dataset.meaning_for(color) if dataset is not None else None
        interpretation = dataset_meaning or self.ZONE_INTERPRETATIONS.get(
            (color, tier),
            "distinct Earth surface feature",
        )

        morphology = f", {', '.join(details)}" if details else ""
        return (
            f"{tier} {color.lower()} zone, {intensity_pct} percent intensity"
            f"{morphology}, {interpretation}."
        )
