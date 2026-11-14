"""CLI and headless self-test entrypoint for Earth Jukebox."""

from __future__ import annotations

import argparse
from typing import Optional

from earth_jukebox.accessibility import AccessibilityTTS
from earth_jukebox.datasets import DatasetManager


def run_self_test() -> int:
    """Run an end-to-end headless verification of NASA datasets, CV, audio, and TTS."""
    from earth_jukebox.audio import CMinorAudioEngine
    from earth_jukebox.vision import CVAnalyzer, extract_sensor_box

    manager = DatasetManager()
    datasets = manager.load_datasets()
    analyzer = CVAnalyzer()
    engine = CMinorAudioEngine(enable_audio=False)
    tts = AccessibilityTTS(enabled=False)

    print(f"[Self-Test] Discovered {len(datasets)} datasets:")
    for idx in range(len(datasets)):
        entry = manager.current_dataset()
        img = manager.load_current_image()
        h, w = img.shape[:2]
        corner_patch, corner_bbox = extract_sensor_box(img, 0, 0, box_size=40)
        edge_patch, edge_bbox = extract_sensor_box(img, w, h, box_size=40)
        center_patch, center_bbox = extract_sensor_box(img, w // 2, h // 2, box_size=40)
        center_res = analyzer.analyze_patch(center_patch, center_bbox, (w // 2, h // 2))
        engine.update_analysis(center_res)
        assert corner_patch.shape[:2] == (40, 40) or corner_patch.size > 0
        assert edge_patch.shape[:2] == (40, 40) or edge_patch.size > 0

        note_desc = ""
        if hasattr(engine, "plan_musical_step"):
            step = engine.plan_musical_step(center_res, idx)
            if isinstance(step, tuple) and len(step) >= 2:
                pitch = int(step[1])
                assert (pitch % 12) in engine.C_MINOR_PITCH_CLASSES
                note_label = (
                    engine.midi_to_note_name(pitch)
                    if hasattr(engine, "midi_to_note_name")
                    else str(pitch)
                )
                note_desc = f" -> {note_label}"

        print(
            f"  - {entry.name} ({w}x{h}): corner={corner_bbox}, edge={edge_bbox}, "
            f"center={center_res.dominant_color} I={center_res.intensity:.2f} "
            f"D={center_res.density:.2f} shape={center_res.shape} "
            f"spot={center_res.spottiness:.2f} swirl={center_res.swirliness:.2f}{note_desc}"
        )
        print(f"    Speech: {analyzer.format_speech_summary(center_res, dataset=entry)}")
        manager.next_dataset()

    engine.start()
    engine.stop()
    tts.start()
    tts.stop()
    print("[Self-Test] PASSED")
    return 0


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI flags for Earth Jukebox."""
    parser = argparse.ArgumentParser(
        description="Earth Jukebox - NASA EIC Imagery Sonification (Mouse & Linux evdev)"
    )
    parser.add_argument(
        "--input",
        choices=("mouse", "evdev"),
        default="mouse",
        help="Input coordinate source: 'mouse' (default) or 'evdev' (Linux trackpad).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional Linux evdev device path (e.g. /dev/input/event7) when --input evdev.",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable sound output (useful for silent inspection).",
    )
    parser.add_argument(
        "--no-drums",
        action="store_true",
        help="Start with ambient drums muted (toggle anytime with D).",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable spoken TTS announcements.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run headless end-to-end verification and exit.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Launch the Earth Jukebox CLI or Tkinter application."""
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()

    try:
        import tkinter as tk
        from PIL import Image, ImageTk  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "tkinter and Pillow are required to launch the Earth Jukebox UI."
        ) from exc

    from earth_jukebox.audio import CMinorAudioEngine
    from earth_jukebox.ui import EarthJukeboxApp
    from earth_jukebox.vision import CVAnalyzer

    dataset_manager = DatasetManager()
    cv_analyzer = CVAnalyzer()
    audio_engine = CMinorAudioEngine(
        enable_audio=not args.no_audio,
        ambient_drums=not args.no_drums,
    )
    tts = AccessibilityTTS(enabled=not args.no_tts)

    root = tk.Tk()
    EarthJukeboxApp(
        root=root,
        dataset_manager=dataset_manager,
        cv_analyzer=cv_analyzer,
        audio_engine=audio_engine,
        tts=tts,
        input_mode=args.input,
        evdev_device=args.device,
    )
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
