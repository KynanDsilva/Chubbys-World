"""Random external events and active weather tracking."""

from __future__ import annotations

import json
import random
import sqlite3
from typing import Any

import db
from state import is_dusk


def is_raining(conn: sqlite3.Connection, current_tick: int, cfg: dict[str, Any]) -> bool:
    """Return True if a rain event started recently enough to still be active."""
    duration = int(cfg["events"]["rain_duration_ticks"])
    since_tick = max(1, current_tick - duration + 1)
    rain_events = db.get_recent_events(conn, since_tick=since_tick, event_type="rain")
    return any(ev["tick"] <= current_tick for ev in rain_events)


def roll_random_event(
    conn: sqlite3.Connection,
    tick: int,
    cfg: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any] | None:
    """Roll for at most one random external event on the current tick."""
    ev_cfg = cfg["events"]
    candidates: list[tuple[str, float]] = []

    if not is_raining(conn, tick, cfg):
        candidates.append(("rain", float(ev_cfg["rain_chance"])))

    candidates.extend(
        [
            ("power_flicker", float(ev_cfg["power_flicker_chance"])),
            ("lucky_find", float(ev_cfg["lucky_find_chance"])),
            ("glitch", float(ev_cfg["glitch_chance"])),
        ]
    )

    if is_dusk(tick, cfg):
        candidates.append(("warm_evening", float(ev_cfg["warm_evening_chance"])))

    roll = rng.random()
    cumulative = 0.0
    chosen_type: str | None = None
    for event_type, prob in candidates:
        cumulative += prob
        if roll < cumulative:
            chosen_type = event_type
            break

    if chosen_type is None:
        return None

    effects_map: dict[str, dict[str, Any]] = {
        "rain": {
            "description": "A steady rain begins to fall over the scrapyard.",
            "deltas": {"calm": 0.03},
            "mood_mod_delta": 0.0,
        },
        "power_flicker": {
            "description": "A sudden power flicker surges through the scrapyard grid.",
            "deltas": {"battery": -0.05, "calm": -0.03},
            "mood_mod_delta": 0.0,
        },
        "lucky_find": {
            "description": "Chubby spots a shiny spare coil tucked beside a crate.",
            "deltas": {"joy": 0.06, "curiosity": 0.05},
            "mood_mod_delta": 0.0,
        },
        "glitch": {
            "description": "A brief sensor glitch scrambles Chubby's telemetry for a moment.",
            "deltas": {},
            "mood_mod_delta": -0.05,
        },
        "warm_evening": {
            "description": "A warm golden breeze settles across the scrapyard at dusk.",
            "deltas": {"calm": 0.04, "joy": 0.04},
            "mood_mod_delta": 0.0,
        },
    }

    spec = effects_map[chosen_type]
    payload = {
        "description": spec["description"],
        "deltas": spec["deltas"],
        "mood_mod_delta": spec["mood_mod_delta"],
    }
    db.insert_event(conn, tick, chosen_type, payload)
    return {"type": chosen_type, **payload}


def get_active_event_descriptions(
    conn: sqlite3.Connection, tick: int, cfg: dict[str, Any]
) -> list[str]:
    """Build grounded descriptions of active weather and events for the prompt."""
    descriptions: list[str] = []
    if is_raining(conn, tick, cfg):
        descriptions.append("Rain is falling steadily over the scrapyard.")

    for ev in db.get_events_for_tick(conn, tick):
        if ev["type"] == "rain":
            continue
        try:
            payload = json.loads(ev["payload"])
            desc = payload.get("description") or ev["type"]
        except json.JSONDecodeError:
            desc = str(ev["payload"])
        descriptions.append(f"{ev['type']}: {desc}")

    # Include recent human-triggered events since previous tick (e.g., approvals/rejections)
    if tick >= 1:
        for ev in db.get_events_for_tick(conn, tick - 1):
            if ev["type"] in ("structure_approved", "structure_rejected"):
                try:
                    payload = json.loads(ev["payload"])
                    desc = payload.get("description") or ev["type"]
                except json.JSONDecodeError:
                    desc = str(ev["payload"])
                descriptions.append(desc)

    return descriptions
