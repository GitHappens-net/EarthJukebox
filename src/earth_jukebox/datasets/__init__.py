"""Dataset discovery and metadata catalog for Earth Jukebox."""

from __future__ import annotations

from earth_jukebox.datasets.catalog import NASA_EIC_CATALOG, load_catalog
from earth_jukebox.datasets.manager import DatasetManager

__all__ = [
    "DatasetManager",
    "NASA_EIC_CATALOG",
    "load_catalog",
]
