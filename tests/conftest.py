"""Shared pytest fixtures for Earth Jukebox unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def sample_image_bgr() -> np.ndarray:
    """Return a synthetic 120x160 BGR image for lightweight tests."""
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    img[:, :] = (180, 75, 20)
    cv2.rectangle(img, (20, 20), (80, 90), (20, 140, 240), -1)
    return img


@pytest.fixture
def sample_datasets_dir(tmp_path: Path, sample_image_bgr: np.ndarray) -> Path:
    """Create a temporary directory with two synthetic NASA-named dataset images."""
    ds_dir = tmp_path / "datasets"
    ds_dir.mkdir()
    cv2.imwrite(str(ds_dir / "aerosol-optical-depth-pm25-2019.png"), sample_image_bgr)
    cv2.imwrite(str(ds_dir / "custom_nasa_map.jpg"), sample_image_bgr)
    return ds_dir
