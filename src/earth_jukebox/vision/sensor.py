"""Edge-safe sensor bounding-box extraction for Earth Jukebox."""

from __future__ import annotations

import numpy as np

MIN_SENSOR_BOX_SIZE = 12
DEFAULT_SENSOR_BOX_SIZE = 40


def extract_sensor_box(
    image: np.ndarray,
    img_x: int,
    img_y: int,
    box_size: int = DEFAULT_SENSOR_BOX_SIZE,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Extract an edge-safe sensor patch around ``(img_x, img_y)``.

    Supports variable/resizable ``box_size`` (clamped to at least
    ``MIN_SENSOR_BOX_SIZE`` and bounded by the image dimensions so an oversized
    or negative ``box_size`` never crashes or produces an empty slice).
    Near borders or corners (including negative or out-of-bounds cursor
    coordinates), the bounding box shifts inward so a full square patch is
    returned whenever the image is large enough.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot extract sensor box from an empty image.")

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Cannot extract sensor box from zero-sized image dimensions.")

    requested_size = max(MIN_SENSOR_BOX_SIZE, int(box_size))
    patch_w = min(width, requested_size)
    patch_h = min(height, requested_size)

    clamped_x = max(0, min(int(img_x), width - 1))
    clamped_y = max(0, min(int(img_y), height - 1))

    max_x1 = max(0, width - patch_w)
    max_y1 = max(0, height - patch_h)

    x1 = max(0, min(clamped_x - (patch_w // 2), max_x1))
    y1 = max(0, min(clamped_y - (patch_h // 2), max_y1))
    x2 = min(width, x1 + patch_w)
    y2 = min(height, y1 + patch_h)

    patch = image[y1:y2, x1:x2]
    return patch, (x1, y1, x2, y2)
