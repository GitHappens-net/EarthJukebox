"""Coordinator mapping received frontend image coordinates to music & audio."""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

import cv2
import numpy as np

from backend.earth_jukebox.accessibility import format_zone_description
from backend.earth_jukebox.audio import (
    ARP_CHANNEL,
    BASS_CHANNEL,
    BPM,
    BroadcastAudioBackend,
    CMinorAudioEngine,
    PAD_CHANNEL,
    PLUCK_CHANNEL,
    VERIDIS_QUO_PROGRESSION,
    compute_expression_ccs,
    create_best_audio_backend,
    intensity_to_velocity,
    midi_to_freq,
    midi_to_note_name,
    plan_baroque_arpeggio,
    plan_bass_step,
    plan_chord_voicing,
    plan_musical_step,
    plan_spottiness_accents,
    select_chord_harmony_note,
    snap_to_c_minor,
)
from backend.earth_jukebox.datasets import DatasetManager
from backend.earth_jukebox.models import AnalysisResult, DatasetEntry
from backend.earth_jukebox.vision import CVAnalyzer, extract_sensor_box


class SonificationService:
    """Thread-safe service that receives image coordinates and outputs music."""

    MIN_SENSOR_SIZE = 16
    MAX_SENSOR_SIZE = 160

    def __init__(
        self,
        dataset_manager: Optional[DatasetManager] = None,
        cv_analyzer: Optional[CVAnalyzer] = None,
        enable_local_audio: bool = False,
        ambient_drums: bool = True,
    ) -> None:
        self.dataset_manager = dataset_manager or DatasetManager()
        self.datasets = self.dataset_manager.load_datasets()
        self.cv_analyzer = cv_analyzer or CVAnalyzer()

        inner_backend = create_best_audio_backend(enable_audio=enable_local_audio)
        self.broadcast_backend = BroadcastAudioBackend(inner=inner_backend)
        self.audio_engine = CMinorAudioEngine(
            backend=self.broadcast_backend,
            enable_audio=True,
            ambient_drums=ambient_drums,
        )

        self._lock = threading.Lock()
        self._step_counter = 0
        self._last_snapshot: Optional[dict[str, Any]] = None
        self._png_cache: dict[int, bytes] = {}
        self._started = False

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        self.audio_engine.start()

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return
            self._started = False
        self.audio_engine.stop()

    def list_datasets(self) -> list[dict[str, Any]]:
        """Return dataset catalog metadata and image dimensions."""
        with self._lock:
            current_idx, _total = self.dataset_manager.current_index()
            items: list[dict[str, Any]] = []
            for idx in range(len(self.datasets)):
                self.dataset_manager.select_dataset(idx)
                entry = self.dataset_manager.current_dataset()
                img = self.dataset_manager.load_current_image()
                h, w = img.shape[:2]
                items.append(
                    {
                        "index": idx,
                        "name": entry.name,
                        "category": entry.category,
                        "unit": entry.unit,
                        "summary": entry.summary,
                        "legend": [
                            {"color": color, "meaning": meaning}
                            for color, meaning in entry.color_meanings
                        ],
                        "width": w,
                        "height": h,
                        "image_url": f"/api/datasets/{idx}/image",
                    }
                )
            self.dataset_manager.select_dataset(current_idx - 1)
            return items

    def get_dataset_image_png(self, index: int) -> bytes:
        """Encode the requested normalized NASA dataset image as PNG bytes."""
        with self._lock:
            if index < 0 or index >= len(self.datasets):
                raise IndexError(
                    f"Dataset index {index} out of range (0..{len(self.datasets) - 1})"
                )
            if index not in self._png_cache:
                prev_idx, _ = self.dataset_manager.current_index()
                self.dataset_manager.select_dataset(index)
                img = self.dataset_manager.load_current_image()
                ok, encoded = cv2.imencode(".png", img)
                self.dataset_manager.select_dataset(prev_idx - 1)
                if not ok:
                    raise RuntimeError("Failed to encode dataset image as PNG")
                self._png_cache[index] = encoded.tobytes()
            return self._png_cache[index]

    def current_status(self) -> dict[str, Any]:
        entry = self.dataset_manager.current_dataset()
        idx_1, total = self.dataset_manager.current_index()
        bar_idx, beat_idx = self.audio_engine.current_beat_info()
        chord = VERIDIS_QUO_PROGRESSION[bar_idx % len(VERIDIS_QUO_PROGRESSION)]
        return {
            "status": "ok",
            "bpm": BPM,
            "bar_index": bar_idx,
            "beat": beat_idx,
            "chord": chord.name,
            "drums_enabled": self.audio_engine.ambient_drums_enabled,
            "subscribers": self.broadcast_backend.subscriber_count,
            "backend": self.broadcast_backend.backend_name,
            "dataset": {
                "index": idx_1 - 1,
                "total": total,
                "name": entry.name,
                "category": entry.category,
                "unit": entry.unit,
                "width": int(self.dataset_manager.load_current_image().shape[1]),
                "height": int(self.dataset_manager.load_current_image().shape[0]),
            },
            "last_snapshot": self._last_snapshot,
        }

    def _select_dataset_locked(self, dataset_ref: Any) -> DatasetEntry:
        if isinstance(dataset_ref, int):
            self.dataset_manager.select_dataset(dataset_ref)
        elif isinstance(dataset_ref, str):
            ref_stripped = dataset_ref.strip()
            if ref_stripped.isdigit():
                self.dataset_manager.select_dataset(int(ref_stripped))
            else:
                target = ref_stripped.lower()
                for idx, item in enumerate(self.datasets):
                    if target in item.name.lower() or target in item.path.stem.lower():
                        self.dataset_manager.select_dataset(idx)
                        break
        return self.dataset_manager.current_dataset()

    def _analyze_payload(
        self, payload: dict[str, Any]
    ) -> tuple[AnalysisResult, DatasetEntry, int, int, int, float, float, int]:
        if not isinstance(payload, dict):
            raise ValueError("Coordinate payload must be a JSON object.")

        sensor_raw = payload.get("sensor_size", 40)
        try:
            sensor_size = max(self.MIN_SENSOR_SIZE, min(self.MAX_SENSOR_SIZE, int(sensor_raw)))
        except (TypeError, ValueError) as exc:
            raise ValueError("sensor_size must be an integer between 16 and 160.") from exc

        touching = bool(payload.get("touching", True))

        with self._lock:
            if "dataset" in payload and payload["dataset"] is not None:
                entry = self._select_dataset_locked(payload["dataset"])
            else:
                entry = self.dataset_manager.current_dataset()
            idx_1, _ = self.dataset_manager.current_index()
            dataset_index = idx_1 - 1
            image_bgr = self.dataset_manager.load_current_image()

        img_h, img_w = image_bgr.shape[:2]

        if not touching:
            idle = AnalysisResult(
                dominant_color="Blue",
                intensity=0.0,
                density=0.0,
                shape="uniform",
                shape_score=0.0,
                spottiness=0.0,
                swirliness=0.0,
                sensor_size=sensor_size,
                sensor_bbox=(0, 0, 0, 0),
                cursor_xy=(0, 0),
                touching=False,
            )
            return idle, entry, dataset_index, img_w, img_h, 0.0, 0.0, sensor_size

        has_ratio = "x_ratio" in payload and "y_ratio" in payload
        has_pixels = "x" in payload and "y" in payload
        if not has_ratio and not has_pixels:
            raise ValueError(
                "Provide either normalized ('x_ratio', 'y_ratio') or pixel ('x', 'y') coordinates."
            )

        if has_ratio:
            x_ratio = max(0.0, min(1.0, float(payload["x_ratio"])))
            y_ratio = max(0.0, min(1.0, float(payload["y_ratio"])))
            img_x = int(round(x_ratio * max(0, img_w - 1)))
            img_y = int(round(y_ratio * max(0, img_h - 1)))
        else:
            img_x = max(0, min(max(0, img_w - 1), int(round(float(payload["x"])))))
            img_y = max(0, min(max(0, img_h - 1), int(round(float(payload["y"])))))
            x_ratio = img_x / max(1, img_w - 1)
            y_ratio = img_y / max(1, img_h - 1)

        patch, bbox = extract_sensor_box(image_bgr, img_x, img_y, box_size=sensor_size)
        analysis = self.cv_analyzer.analyze_patch(
            roi=patch,
            bbox=bbox,
            cursor_xy=(img_x, img_y),
            touching=True,
            sensor_size=sensor_size,
        )
        return analysis, entry, dataset_index, img_w, img_h, x_ratio, y_ratio, sensor_size

    def _build_music_snapshot(
        self,
        analysis: AnalysisResult,
        bar_index: int,
        beat: int,
        step_index: int,
    ) -> dict[str, Any]:
        chord = VERIDIS_QUO_PROGRESSION[bar_index % len(VERIDIS_QUO_PROGRESSION)]
        ccs = compute_expression_ccs(analysis, step_16=beat * 4)

        if not analysis.touching:
            return {
                "bpm": BPM,
                "bar_index": bar_index,
                "beat": beat,
                "chord": chord.name,
                "touching": False,
                "drums_enabled": self.audio_engine.ambient_drums_enabled,
                "expression_ccs": ccs,
                "chord_voicing": None,
                "bass_note": None,
                "baroque_arp": None,
                "spottiness_accent": None,
                "primary_voice": None,
            }

        sensor_size = int(getattr(analysis, "sensor_size", 40))
        _cname, _bass, pad_notes = plan_chord_voicing(
            bar_index,
            analysis.intensity,
            sensor_size=sensor_size,
        )
        pad_vel = intensity_to_velocity(analysis.intensity)
        bass_hits = plan_bass_step(
            bar_index,
            analysis.density,
            analysis.intensity,
            sensor_size,
            beat * 4,
        )
        bass_payload = None
        if bass_hits:
            bass_payload = {
                "channel": BASS_CHANNEL,
                "notes": [
                    {
                        "midi_note": n,
                        "note_name": midi_to_note_name(n),
                        "freq_hz": round(midi_to_freq(n), 4),
                        "velocity": vel,
                    }
                    for n, vel in bass_hits
                ],
            }

        arp_hit = plan_baroque_arpeggio(analysis, bar_index, beat * 4, step_index)
        arp_payload = None
        if arp_hit is not None:
            a_note, a_vel, a_gate = arp_hit
            qa = snap_to_c_minor(a_note)
            arp_payload = {
                "channel": ARP_CHANNEL,
                "midi_note": qa,
                "note_name": midi_to_note_name(qa),
                "freq_hz": round(midi_to_freq(qa), 4),
                "velocity": a_vel,
                "gate_steps": a_gate,
            }

        spot_hit = plan_spottiness_accents(analysis, bar_index, 3)
        spot_payload = None
        if spot_hit is not None:
            s_note, s_vel = spot_hit
            qs = snap_to_c_minor(s_note)
            spot_payload = {
                "channel": PLUCK_CHANNEL,
                "midi_note": qs,
                "note_name": midi_to_note_name(qs),
                "freq_hz": round(midi_to_freq(qs), 4),
                "velocity": s_vel,
            }

        channel, pitch, velocity, volume, motif, duration_sec = plan_musical_step(
            analysis, step_index=step_index, bar_index=bar_index
        )
        q_pitch = snap_to_c_minor(pitch)
        harmony_pitch = snap_to_c_minor(select_chord_harmony_note(q_pitch, bar_index))

        return {
            "bpm": BPM,
            "bar_index": bar_index,
            "beat": beat,
            "chord": chord.name,
            "touching": True,
            "drums_enabled": self.audio_engine.ambient_drums_enabled,
            "expression_ccs": ccs,
            "chord_voicing": {
                "channel": PAD_CHANNEL,
                "velocity": pad_vel,
                "notes": [
                    {
                        "midi_note": snap_to_c_minor(n),
                        "note_name": midi_to_note_name(snap_to_c_minor(n)),
                        "freq_hz": round(midi_to_freq(snap_to_c_minor(n)), 4),
                    }
                    for n in pad_notes
                ],
            },
            "bass_note": bass_payload,
            "baroque_arp": arp_payload,
            "spottiness_accent": spot_payload,
            "primary_voice": {
                "channel": channel,
                "midi_note": q_pitch,
                "note_name": midi_to_note_name(q_pitch),
                "freq_hz": round(midi_to_freq(q_pitch), 4),
                "harmony_midi_note": harmony_pitch if harmony_pitch != q_pitch else None,
                "harmony_note_name": midi_to_note_name(harmony_pitch)
                if harmony_pitch != q_pitch
                else None,
                "harmony_freq_hz": round(midi_to_freq(harmony_pitch), 4)
                if harmony_pitch != q_pitch
                else None,
                "velocity": velocity,
                "volume": round(volume, 4),
                "motif": motif,
                "duration_sec": round(duration_sec, 4),
            },
        }

    def process_coordinates(
        self,
        payload: dict[str, Any],
        update_live: bool = True,
    ) -> dict[str, Any]:
        """Analyze received coordinates, drive live audio engine, and return music JSON."""
        (
            analysis,
            entry,
            ds_idx,
            img_w,
            img_h,
            x_ratio,
            y_ratio,
            sensor_size,
        ) = self._analyze_payload(payload)

        if "drums" in payload and payload["drums"] is not None:
            want_drums = bool(payload["drums"])
            if want_drums != self.audio_engine.ambient_drums_enabled:
                self.audio_engine.toggle_drums()

        if update_live:
            self.audio_engine.update_analysis(analysis)

        with self._lock:
            step_idx = self._step_counter
            self._step_counter += 1

        engine_bar, engine_beat = self.audio_engine.current_beat_info()
        bar_index = int(payload.get("bar_index", engine_bar)) % len(VERIDIS_QUO_PROGRESSION)

        music_snapshot = self._build_music_snapshot(
            analysis=analysis,
            bar_index=bar_index,
            beat=engine_beat,
            step_index=step_idx,
        )
        description = format_zone_description(analysis, dataset=entry)

        result = {
            "type": "sonification_state",
            "timestamp_ms": int(time.time() * 1000),
            "dataset": {
                "index": ds_idx,
                "name": entry.name,
                "category": entry.category,
                "unit": entry.unit,
                "width": img_w,
                "height": img_h,
                "meaning": entry.meaning_for(analysis.dominant_color),
            },
            "coordinates": {
                "x": analysis.cursor_xy[0],
                "y": analysis.cursor_xy[1],
                "x_ratio": round(x_ratio, 6),
                "y_ratio": round(y_ratio, 6),
                "sensor_size": sensor_size,
                "sensor_bbox": list(analysis.sensor_bbox),
                "touching": analysis.touching,
            },
            "analysis": {
                "dominant_color": analysis.dominant_color,
                "intensity": round(analysis.intensity, 4),
                "density": round(analysis.density, 4),
                "shape": analysis.shape,
                "shape_score": round(analysis.shape_score, 4),
                "spottiness": round(analysis.spottiness, 4),
                "swirliness": round(analysis.swirliness, 4),
            },
            "description": description,
            "music": music_snapshot,
        }

        if update_live:
            with self._lock:
                self._last_snapshot = result
            self.broadcast_backend.publish(result)

        return result


__all__ = ["SonificationService"]
