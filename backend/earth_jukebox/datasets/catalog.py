"""Curated scientific metadata catalog for NASA EIC visualizations."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any


def load_catalog() -> dict[str, dict[str, Any]]:
    """Load the NASA EIC dataset metadata catalog from ``catalog.json``."""
    raw = json.loads(files(__package__).joinpath("catalog.json").read_text(encoding="utf-8"))
    catalog: dict[str, dict[str, Any]] = {}
    for key, entry in raw.items():
        meanings = tuple(tuple(pair) for pair in entry.get("color_meanings", []))
        catalog[key.lower()] = {
            "name": str(entry["name"]),
            "category": str(entry.get("category", "NASA Earth Observation")),
            "unit": str(entry.get("unit", "")),
            "summary": str(entry.get("summary", "")),
            "color_meanings": meanings,
        }
    return catalog


NASA_EIC_CATALOG: dict[str, dict[str, Any]] = load_catalog()
