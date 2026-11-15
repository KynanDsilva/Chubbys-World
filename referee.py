"""Referee rules, hard caps, noise application, and random-policy resolution."""

from __future__ import annotations

import random
from typing import Any

from llm import RefereeDeltasSchema, RefereeOutputSchema
from state import NEED_NAMES, WorldState, clamp, is_night

_RECHARGE_KEYWORDS = ("recharge", "charge", "solar", "battery", "plug", "power up")
_CHARGER_STRUCTURE_KEYWORDS = ("charg", "solar", "power", "battery")


def has_charging_structure(structures: list[dict[str, Any]]) -> bool:
    """Return True if an approved structure provides charging capabilities."""
    for struct in structures:
        text = f"{struct.get('name', '')} {struct.get('unlocks', '')}".lower()
        if any(kw in text for kw in _CHARGER_STRUCTURE_KEYWORDS):
            return True
    return False


def apply_referee_caps_and_rules(
    raw_output: RefereeOutputSchema,
    category: str,
    intent: str,
    state: WorldState,
    structures: list[dict[str, Any]],
    is_raining_now: bool,
    cfg: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    """Enforce deterministic world rules, caps, weather/night costs, and noise."""
    ref_cfg = cfg["referee"]
    max_delta = float(ref_cfg["max_need_delta"])
    boosted_cap = float(ref_cfg["max_battery_recovery_boosted"])
    solo_belonging_max = float(ref_cfg["solo_belonging_max"])
    noise_min = float(ref_cfg["noise_min"])
    noise_max = float(ref_cfg["noise_max"])

    if raw_output.is_impossible:
        return {
            "outcome": "Chubby pauses and realizes that isn't possible in his scrapyard right now.",
            "is_impossible": True,
            "is_outdoor": raw_output.is_outdoor,
            "deltas": {name: 0.0 for name in NEED_NAMES},
        }

    proposed = raw_output.deltas.model_dump()
    lowered_intent = intent.lower()
    recharge_intent = any(kw in lowered_intent for kw in _RECHARGE_KEYWORDS)
    charger_built = has_charging_structure(structures)
    night_now = is_night(state.tick, cfg)

    # Battery may only rise through rest, recharge-type intent, or a charging structure
    can_gain_battery = (
        category in ("rest", "recharge")
        or recharge_intent
        or (charger_built and recharge_intent)
    )
    if not can_gain_battery and proposed["battery"] > 0.0:
        proposed["battery"] = 0.0

    # Night rest bonus and elevated recovery cap (+0.22 before noise)
    battery_upper = max_delta
    if category == "rest" and night_now:
        proposed["battery"] += float(ref_cfg["night_rest_battery_bonus"])
        battery_upper = boosted_cap
    elif (category == "recharge" or recharge_intent) and charger_built:
        battery_upper = boosted_cap

    # Night & rain outdoor activity battery cost
    if raw_output.is_outdoor and category not in ("rest", "recharge"):
        if night_now:
            proposed["battery"] -= float(ref_cfg["night_activity_battery_cost"])
        if is_raining_now:
            proposed["battery"] -= float(ref_cfg["rain_outdoor_battery_cost"])

    capped: dict[str, float] = {
        "battery": clamp(proposed["battery"], -max_delta, battery_upper),
        "curiosity": clamp(proposed["curiosity"], -max_delta, max_delta),
        "joy": clamp(proposed["joy"], -max_delta, max_delta),
        "calm": clamp(proposed["calm"], -max_delta, max_delta),
        "belonging": clamp(proposed["belonging"], -max_delta, solo_belonging_max),
    }

    noisy_deltas: dict[str, float] = {}
    for name, val in capped.items():
        if val == 0.0:
            noisy_deltas[name] = 0.0
        else:
            factor = rng.uniform(noise_min, noise_max)
            noisy_deltas[name] = round(val * factor, 6)

    return {
        "outcome": raw_output.outcome,
        "is_impossible": False,
        "is_outdoor": raw_output.is_outdoor,
        "deltas": noisy_deltas,
    }


def resolve_random_policy_action(
    category: str,
    intent: str,
    state: WorldState,
    structures: list[dict[str, Any]],
    is_raining_now: bool,
    cfg: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    """Deterministic/random-policy referee proposal used during LLM-free Phase 2 runs."""
    has_charger = has_charging_structure(structures)
    proposals: dict[str, dict[str, float]] = {
        "rest": {"battery": 0.13, "calm": 0.035, "curiosity": -0.015, "joy": 0.0, "belonging": 0.0},
        "recharge": {
            "battery": 0.17 if has_charger else 0.13,
            "calm": 0.0,
            "curiosity": -0.015,
            "joy": -0.01,
            "belonging": 0.0,
        },
        "explore": {"battery": -0.02, "curiosity": 0.09, "joy": 0.0, "calm": -0.02, "belonging": 0.0},
        "make": {"battery": -0.02, "curiosity": 0.03, "joy": 0.04, "calm": 0.0, "belonging": 0.005},
        "play": {"battery": -0.02, "curiosity": 0.01, "joy": 0.08, "calm": -0.02, "belonging": 0.0},
        "social": {"battery": -0.01, "curiosity": 0.0, "joy": 0.02, "calm": 0.01, "belonging": 0.085},
        "reflect": {"battery": 0.0, "curiosity": 0.01, "joy": 0.0, "calm": 0.055, "belonging": 0.02},
        "maintain": {"battery": -0.01, "curiosity": -0.005, "joy": -0.015, "calm": 0.025, "belonging": 0.0},
        "other": {"battery": -0.01, "curiosity": 0.03, "joy": 0.01, "calm": 0.0, "belonging": 0.0},
    }
    raw_out = RefereeOutputSchema(
        outcome=f"Random policy executed {category}: {intent}",
        is_impossible=False,
        is_outdoor=category not in ("rest", "reflect", "maintain"),
        deltas=RefereeDeltasSchema(**proposals.get(category, proposals["other"])),
    )
    return apply_referee_caps_and_rules(
        raw_out, category, intent, state, structures, is_raining_now, cfg, rng
    )

