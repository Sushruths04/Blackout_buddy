from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).resolve().parent / "data" / "emergency_kb.json"

SCENARIOS = {
    "🩹 Severe bleeding": "severe_bleeding",
    "💧 Unsafe water": "unsafe_water",
    "🔥 Building fire": "building_fire",
    "❤️ Not breathing": "unconscious_not_breathing",
    "🌊 Flash flood": "flash_flood",
    "🏥 Possible stroke": "stroke",
    "🌡️ Heatstroke": "heatstroke",
    "🧊 Hypothermia": "hypothermia",
    "🐍 Snakebite": "snakebite",
    "⚡ Long power outage": "extended_power_outage",
}


@lru_cache(maxsize=1)
def load_cards() -> list[dict[str, Any]]:
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        cards = json.load(handle)
    if not isinstance(cards, list) or not cards:
        raise ValueError("Emergency knowledge base is empty or invalid.")
    return cards


@lru_cache(maxsize=1)
def cards_by_id() -> dict[str, dict[str, Any]]:
    return {card["id"]: card for card in load_cards()}
