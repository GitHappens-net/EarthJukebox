from __future__ import annotations
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from backend.earth_jukebox.datasets.catalog import NASA_EIC_CATALOG
from backend.earth_jukebox.models import DatasetEntry

def _default_repo_root() -> Path:
    package_root = Path(__file__).resolve().parents[3]
    if (package_root / "data" / "images").is_dir():
        return package_root
    cwd = Path.cwd()
    if (cwd / "data" / "images").is_dir():
        return cwd
    return package_root

class DatasetManager:
    """Manages loading, normalizing, caching, and cycling NASA imagery datasets."""

    SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
    MAX_WORKING_WIDTH = 960
    MAX_WORKING_HEIGHT = 540

    def __init__(
        self,
        datasets_dir: Optional[Path] = None,
        world_map_path: Optional[Path] = None,
        include_tactile_reference: bool = True,
    ) -> None:
        repo_root = _default_repo_root()
        self._explicit_dir = datasets_dir is not None
        self.world_map_path = (
            Path(world_map_path)
            if world_map_path is not None
            else repo_root / "hardware" / "tactile-map" / "world-map.jpg"
        )
        self.include_tactile_reference = include_tactile_reference
        self.datasets_dir = (
            Path(datasets_dir) if datasets_dir is not None else repo_root / "data" / "images"
        )

        self._entries: list[DatasetEntry] = []
        self._index = 0
        self._image_cache: dict[Path, np.ndarray] = {}

    def load_datasets(self) -> list[DatasetEntry]:
        """Discover valid NASA dataset images and bind scientific metadata."""
        candidate_paths: list[Path] = []
        if self.datasets_dir.is_dir():
            candidate_paths = [
                path
                for path in sorted(self.datasets_dir.iterdir(), key=lambda p: p.name.lower())
                if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ]

        if (
            self.include_tactile_reference
            and not self._explicit_dir
            and self.world_map_path.is_file()
            and all(p.resolve() != self.world_map_path.resolve() for p in candidate_paths)
        ):
            candidate_paths.append(self.world_map_path)

        if not candidate_paths:
            raise FileNotFoundError(f"No NASA dataset images found in {self.datasets_dir}")

        self._entries = [self._build_entry(path) for path in candidate_paths]
        self._index %= len(self._entries)
        return list(self._entries)

    def current_dataset(self) -> DatasetEntry:
        """Return the active dataset entry."""
        if not self._entries:
            self.load_datasets()
        return self._entries[self._index]

    def current_index(self) -> tuple[int, int]:
        """Return ``(1_based_index, total_count)`` for HUD display."""
        if not self._entries:
            self.load_datasets()
        return self._index + 1, len(self._entries)

    def select_dataset(self, index: int) -> DatasetEntry:
        """Select a dataset by 0-based index (wrapping safely)."""
        if not self._entries:
            self.load_datasets()
        self._index = index % len(self._entries)
        return self._entries[self._index]

    def next_dataset(self) -> DatasetEntry:
        """Advance cyclically to the next NASA dataset."""
        if not self._entries:
            self.load_datasets()
        self._index = (self._index + 1) % len(self._entries)
        return self._entries[self._index]

    def load_current_image(self) -> np.ndarray:
        """Read, normalize oversized frames to working resolution, and cache BGR image."""
        entry = self.current_dataset()
        cached = self._image_cache.get(entry.path)
        if cached is not None:
            return cached

        image = cv2.imread(str(entry.path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(f"Failed to decode dataset image: {entry.path}")

        height, width = image.shape[:2]
        if width > self.MAX_WORKING_WIDTH or height > self.MAX_WORKING_HEIGHT:
            scale = min(self.MAX_WORKING_WIDTH / width, self.MAX_WORKING_HEIGHT / height)
            target_w = max(40, int(round(width * scale)))
            target_h = max(40, int(round(height * scale)))
            image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)

        self._image_cache[entry.path] = image
        return image

    def _build_entry(self, path: Path) -> DatasetEntry:
        meta = NASA_EIC_CATALOG.get(path.name.lower())
        if meta is not None:
            return DatasetEntry(
                name=str(meta["name"]),
                path=path,
                category=str(meta["category"]),
                unit=str(meta["unit"]),
                summary=str(meta["summary"]),
                color_meanings=tuple(meta["color_meanings"]),
            )
        formatted = self._format_name(path.stem)
        return DatasetEntry(
            name=formatted,
            path=path,
            category="NASA Earth Visualization",
            unit="",
            summary=f"Exploring NASA dataset {formatted}.",
        )

    @staticmethod
    def _format_name(stem: str) -> str:
        parts = stem.replace("-", " ").split("_")
        if parts and parts[0].isdigit():
            parts = parts[1:]
        cleaned = " ".join(part for part in parts if part).strip()
        return cleaned if cleaned else stem