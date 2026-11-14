"""Unit tests for models, input proxies, datasets layer, and CLI parsing."""

from __future__ import annotations

import builtins
import importlib
import queue
import sys
import threading
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from earth_jukebox import __version__
from earth_jukebox.cli import parse_args
from earth_jukebox.datasets import NASA_EIC_CATALOG, DatasetManager, load_catalog
from earth_jukebox.input import (
    EvdevTrackpadInputProxy,
    WindowsMouseInputProxy,
    find_touchpad,
    push_latest_point,
    read_touchpad,
)
from earth_jukebox.models import AnalysisResult, DatasetEntry, TouchPoint


class _FakeCanvas:
    def __init__(self, width: int = 960, height: int = 540) -> None:
        self._width = width
        self._height = height
        self.bindings: dict[str, list[object]] = {}
        self.unbound: list[tuple[str, str]] = []

    def winfo_width(self) -> int:
        return self._width

    def winfo_height(self) -> int:
        return self._height

    def bind(self, sequence: str, handler: object, add: str = "") -> str:
        self.bindings.setdefault(sequence, []).append(handler)
        return f"id:{sequence}"

    def unbind(self, sequence: str, bind_id: str) -> None:
        self.unbound.append((sequence, bind_id))


class _FakeEvent:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y


def test_package_version_and_models_contracts() -> None:
    assert __version__ == "0.2.0"

    tp = TouchPoint(x=10, y=20, x_ratio=0.25, y_ratio=0.5, touching=True)
    assert tp.x == 10 and tp.touching is True

    res = AnalysisResult(dominant_color="Red", intensity=0.8, density=0.6)
    assert res.shape == "uniform"
    assert res.shape_score == 0.0
    assert res.spottiness == 0.0
    assert res.swirliness == 0.0
    assert res.sensor_size == 40
    assert res.sensor_bbox == (0, 0, 0, 0)
    assert res.cursor_xy == (0, 0)
    assert res.touching is True

    entry = DatasetEntry(
        name="Test Dataset",
        path=Path("test.png"),
        color_meanings=(("Red", "severe hotspot"), ("Blue", "clean air")),
    )
    assert entry.meaning_for("Red") == "severe hotspot"
    assert entry.meaning_for("Green") is None


def test_push_latest_point_keeps_newest_in_bounded_queue() -> None:
    q: queue.Queue[TouchPoint] = queue.Queue(maxsize=2)
    p1 = TouchPoint(1, 1, 0.1, 0.1, True)
    p2 = TouchPoint(2, 2, 0.2, 0.2, True)
    p3 = TouchPoint(3, 3, 0.3, 0.3, True)

    push_latest_point(q, p1)
    push_latest_point(q, p2)
    push_latest_point(q, p3)

    items = []
    while not q.empty():
        items.append(q.get_nowait())
    assert items[-1] == p3


def test_input_imports_cleanly_without_tkinter(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "tkinter" or name.startswith("tkinter."):
            raise ImportError("No module named 'tkinter'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    for mod_name in list(sys.modules):
        if mod_name == "tkinter" or mod_name.startswith("earth_jukebox.input"):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

    mod = importlib.import_module("earth_jukebox.input")
    assert hasattr(mod, "WindowsMouseInputProxy")
    assert hasattr(mod, "EvdevTrackpadInputProxy")


def test_windows_mouse_proxy_emits_clamped_touchpoints() -> None:
    canvas = _FakeCanvas(width=101, height=51)
    q: queue.Queue[TouchPoint] = queue.Queue(maxsize=2)
    stop = threading.Event()
    activity_count = 0

    def on_activity() -> None:
        nonlocal activity_count
        activity_count += 1

    proxy = WindowsMouseInputProxy(
        canvas,  # type: ignore[arg-type]
        q,
        stop,
        on_user_activity=on_activity,
    )
    assert proxy.source_name == "Windows Mouse Proxy"
    proxy.start()
    assert "<Motion>" in canvas.bindings
    assert "<Leave>" in canvas.bindings

    proxy._on_motion(_FakeEvent(150, -10))  # type: ignore[arg-type]
    pt = q.get_nowait()
    assert pt == TouchPoint(x=100, y=0, x_ratio=1.0, y_ratio=0.0, touching=True)
    assert activity_count == 1

    proxy._on_leave(_FakeEvent(0, 0))  # type: ignore[arg-type]
    leave_pt = q.get_nowait()
    assert leave_pt.touching is False
    assert leave_pt.x == 100 and leave_pt.y == 0

    proxy.stop()
    assert len(canvas.unbound) == 5


def _build_fake_evdev() -> tuple[types.ModuleType, dict[str, object]]:
    ecodes = types.SimpleNamespace(
        EV_ABS=3,
        EV_KEY=1,
        EV_SYN=0,
        SYN_REPORT=0,
        ABS_X=0,
        ABS_Y=1,
        ABS_MT_POSITION_X=53,
        ABS_MT_POSITION_Y=54,
        BTN_TOUCH=330,
        BTN_TOOL_FINGER=325,
    )

    class FakeDevice:
        def __init__(self, path: str) -> None:
            self.path = path
            self.name = "Synaptics TouchPad"

        def capabilities(self) -> dict[int, list[object]]:
            return {
                ecodes.EV_ABS: [(ecodes.ABS_X, None), (ecodes.ABS_Y, None)],
                ecodes.EV_KEY: [ecodes.BTN_TOUCH],
            }

        def absinfo(self, code: int) -> object:
            return types.SimpleNamespace(min=0, max=1000 if code == ecodes.ABS_X else 500)

        def read_loop(self):
            yield types.SimpleNamespace(type=ecodes.EV_KEY, code=ecodes.BTN_TOUCH, value=1)
            yield types.SimpleNamespace(type=ecodes.EV_ABS, code=ecodes.ABS_X, value=500)
            yield types.SimpleNamespace(type=ecodes.EV_ABS, code=ecodes.ABS_Y, value=250)
            yield types.SimpleNamespace(type=ecodes.EV_SYN, code=ecodes.SYN_REPORT, value=0)
            yield types.SimpleNamespace(type=ecodes.EV_KEY, code=ecodes.BTN_TOUCH, value=0)
            yield types.SimpleNamespace(type=ecodes.EV_SYN, code=ecodes.SYN_REPORT, value=0)

    fake_evdev = types.ModuleType("evdev")
    fake_evdev.InputDevice = FakeDevice  # type: ignore[attr-defined]
    fake_evdev.ecodes = ecodes  # type: ignore[attr-defined]
    fake_evdev.list_devices = lambda: ["/dev/input/event7"]  # type: ignore[attr-defined]
    return fake_evdev, {"device_cls": FakeDevice, "ecodes": ecodes}


def test_evdev_find_and_read_touchpad_with_mock_evdev(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_evdev, _meta = _build_fake_evdev()
    monkeypatch.setitem(sys.modules, "evdev", fake_evdev)

    device = find_touchpad(None)
    assert device.path == "/dev/input/event7"

    q: queue.Queue[TouchPoint] = queue.Queue(maxsize=4)
    stop = threading.Event()
    read_touchpad(device, q, stop)

    first = q.get_nowait()
    second = q.get_nowait()
    assert first == TouchPoint(x=500, y=250, x_ratio=0.5, y_ratio=0.5, touching=True)
    assert second.touching is False

    proxy = EvdevTrackpadInputProxy(q, stop, device_path="/dev/input/event7")
    proxy.start()
    assert "Synaptics TouchPad" in proxy.source_name
    proxy.stop()


def test_catalog_contains_all_kebab_case_entries() -> None:
    catalog = load_catalog()
    expected = {
        "aerosol-optical-depth-pm25-2019.png",
        "geos-aerosols-2025-08-19.png",
        "ecco2-ocean-currents-grid.jpg",
        "fwi-conus-2026-08-01.png",
        "fwi-global-forecast-latest.png",
        "airs-global-co2-60s.png",
        "svs-co2-column-2021-05-16.png",
        "world-map.jpg",
    }
    assert expected.issubset(catalog.keys())
    assert expected.issubset(NASA_EIC_CATALOG.keys())


def test_dataset_manager_loads_cycles_and_resizes(sample_datasets_dir: Path) -> None:
    large = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.imwrite(str(sample_datasets_dir / "large_frame.png"), large)

    manager = DatasetManager(datasets_dir=sample_datasets_dir)
    entries = manager.load_datasets()
    assert len(entries) == 3

    first = manager.current_dataset()
    assert first.name == "Global PM2.5 Aerosol Optical Depth (2019)"
    assert manager.current_index() == (1, 3)

    second = manager.next_dataset()
    assert second != first
    third = manager.next_dataset()
    assert third.path.name == "large_frame.png"
    resized = manager.load_current_image()
    h, w = resized.shape[:2]
    assert w <= DatasetManager.MAX_WORKING_WIDTH
    assert h <= DatasetManager.MAX_WORKING_HEIGHT

    wrapped = manager.next_dataset()
    assert wrapped == first
    assert manager.select_dataset(1) == second


def test_dataset_manager_raises_cleanly_when_empty(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty_datasets"
    empty_dir.mkdir()
    manager = DatasetManager(datasets_dir=empty_dir)
    with pytest.raises(FileNotFoundError):
        manager.load_datasets()

    missing_dir = tmp_path / "does_not_exist"
    missing_manager = DatasetManager(datasets_dir=missing_dir)
    with pytest.raises(FileNotFoundError):
        missing_manager.load_datasets()


@pytest.mark.integration
def test_dataset_manager_loads_real_nasa_eic_images() -> None:
    manager = DatasetManager()
    entries = manager.load_datasets()
    assert len(entries) >= 8
    for _ in range(len(entries)):
        entry = manager.current_dataset()
        img = manager.load_current_image()
        h, w = img.shape[:2]
        assert 40 <= w <= DatasetManager.MAX_WORKING_WIDTH
        assert 40 <= h <= DatasetManager.MAX_WORKING_HEIGHT
        assert entry.summary
        manager.next_dataset()


def test_cli_parse_args() -> None:
    defaults = parse_args([])
    assert defaults.input == "mouse"
    assert defaults.device is None
    assert defaults.no_audio is False
    assert defaults.no_drums is False
    assert defaults.no_tts is False
    assert defaults.self_test is False

    parsed = parse_args(
        ["--input", "evdev", "--device", "/dev/input/event7", "--no-audio", "--self-test"]
    )
    assert parsed.input == "evdev"
    assert parsed.device == "/dev/input/event7"
    assert parsed.no_audio is True
    assert parsed.self_test is True
