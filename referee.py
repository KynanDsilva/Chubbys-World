"""Referee rules, hard caps, noise application, and random-policy resolution."""

from __future__ import annotations

import random
import re
from typing import Any

from llm import RefereeDeltasSchema, RefereeOutputSchema
from state import NEED_NAMES, WorldState, clamp, is_night

_RECHARGE_KEYWORDS = ("recharge", "charge", "solar", "battery", "plug", "power up")
_CHARGER_STRUCTURE_KEYWORDS = ("charg", "solar", "power", "battery")
_CURIOSITY_STRUCTURE_KEYWORDS = (
    "periscope",
    "lookout",
    "telescope",
    "scanner",
    "drafting",
    "schematic",
)
_ZEN_STRUCTURE_KEYWORDS = ("zen", "pebble", "garden")
_CREATIVE_STRUCTURE_KEYWORDS = ("workbench", "scanner", "drafting", "table", "tool")

_UNBUILT_CHECKS: tuple[tuple[str, ...], ...] = (
    ("workbench", "work bench", "table for making"),
    ("charging pad", "solar pad", "charger station", "power contact"),
    ("canopy", "shelter", "awning", "shed", "tin shed"),
    ("radio corner", "broadcast mast", "radio tower"),
    ("periscope", "lookout post", "telescope"),
    ("salvage scanner", "drafting table", "scanner table"),
    ("chime arch", "string light", "zen garden", "kinetic windmill"),
    ("transformer", "crate stack", "shipping crate", "oil drum"),
)

_IMPOSSIBLE_WORD_RE = re.compile(r"\b(mars|orbit|teleport|fly)\b")
_IMPOSSIBLE_PHRASES = ("outside the fence", "beyond the fence")


def _uses_unbuilt_structure(
    lowered_intent: str, structures: list[dict[str, Any]]
) -> bool:
    for keywords in _UNBUILT_CHECKS:
        if any(kw in lowered_intent for kw in keywords):
            has_matching = any(
                any(
                    kw in f"{s.get('name', '')} {s.get('unlocks', '')}".lower()
                    for kw in keywords
                )
                for s in structures
            )
            if not has_matching:
                return True
    return False


def _has_impossible_action(lowered_intent: str) -> bool:
    return bool(_IMPOSSIBLE_WORD_RE.search(lowered_intent)) or any(
        phrase in lowered_intent for phrase in _IMPOSSIBLE_PHRASES
    )


def has_charging_structure(structures: list[dict[str, Any]]) -> bool:
    """Return True if an approved structure provides charging capabilities."""
    for struct in structures:
        text = f"{struct.get('name', '')} {struct.get('unlocks', '')}".lower()
        if any(kw in text for kw in _CHARGER_STRUCTURE_KEYWORDS):
            return True
    return False


def has_curiosity_structure(structures: list[dict[str, Any]]) -> bool:
    """Return True if a built curiosity catalyst structure (periscope or scanner table) exists."""
    for struct in structures:
        text = f"{struct.get('name', '')} {struct.get('unlocks', '')}".lower()
        if any(kw in text for kw in _CURIOSITY_STRUCTURE_KEYWORDS):
            return True
    return False


def _intent_at_structure(
    lowered_intent: str,
    structures: list[dict[str, Any]],
    structure_keywords: tuple[str, ...],
    extra_intent_keywords: tuple[str, ...] = (),
) -> bool:
    """Return True if a matching structure is built and Chubby's intent goes to/uses it."""
    matching = [
        s
        for s in structures
        if any(
            kw in f"{s.get('name', '')} {s.get('unlocks', '')}".lower()
            for kw in structure_keywords
        )
    ]
    if not matching:
        return False
    if lowered_intent.startswith("randomly "):
        return True
    for s in matching:
        s_name = str(s.get("name", "")).lower().strip()
        if s_name and s_name in lowered_intent:
            return True
    return any(kw in lowered_intent for kw in extra_intent_keywords)


_ART_STUDIO_KEYWORDS = ("art studio", "studio", "easel")
_ART_STUDIO_INTENT_KEYWORDS = ("art studio", "easel")
_WALL_ART_LOCATIONS = (
    "art studio",
    "easel",
    "wall",
    "bakery",
    "library",
    "cafe",
    "workshop",
    "post office",
    "apartment",
    "fence",
    "mural",
    "exhibit",
)


def is_at_art_studio(intent: str, structures: list[dict[str, Any]]) -> bool:
    """True if an Art Studio exists and Chubby's intent says he is standing at it."""
    return _intent_at_structure(
        intent.lower(), structures, _ART_STUDIO_KEYWORDS, _ART_STUDIO_INTENT_KEYWORDS
    )


def is_at_art_studio_or_wall(intent: str, structures: list[dict[str, Any]]) -> bool:
    """True if Chubby is at the Art Studio or standing at a town wall to make/hang art."""
    lowered = intent.lower()
    if is_at_art_studio(intent, structures):
        return True
    return any(loc in lowered for loc in _WALL_ART_LOCATIONS)


def gate_draw_category(
    category: str,
    intent: str,
    structures: list[dict[str, Any]],
    recent_categories: list[str] | None = None,
) -> tuple[str, str | None]:
    """Downgrade `draw` to another allowed category unless Chubby is at the Art Studio or a town art wall."""
    if category != "draw" or is_at_art_studio_or_wall(intent, structures):
        return category, None
    downgrade_target = "make"
    if recent_categories and recent_categories[-1] == "make":
        downgrade_target = (
            "explore"
            if len(recent_categories) < 2 or recent_categories[-2] != "explore"
            else "play"
        )
    return downgrade_target, (
        "(He wasn't standing at the Art Studio or a town art wall, so he just doodled in the dust instead.)"
    )


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

    lowered_intent = intent.lower()
    violates_unbuilt = _uses_unbuilt_structure(lowered_intent, structures)
    violates_impossible = _has_impossible_action(lowered_intent)

    if violates_unbuilt:
        return {
            "outcome": "Chubby pauses and looks around, but that does not exist in his bare scrapyard lot yet.",
            "is_impossible": True,
            "is_outdoor": raw_output.is_outdoor,
            "deltas": {name: 0.0 for name in NEED_NAMES},
        }

    if raw_output.is_impossible:
        if (
            category not in ("rest", "recharge", "reflect", "maintain")
            or violates_unbuilt
            or violates_impossible
        ):
            return {
                "outcome": "Chubby pauses and realizes that isn't possible in his scrapyard right now.",
                "is_impossible": True,
                "is_outdoor": raw_output.is_outdoor,
                "deltas": {name: 0.0 for name in NEED_NAMES},
            }
        raw_output = RefereeOutputSchema(
            outcome=f"Chubby takes a quiet moment to {category} on the gravel lot.",
            is_impossible=False,
            is_outdoor=(category == "recharge"),
            deltas=RefereeDeltasSchema(
                battery=0.12 if category in ("rest", "recharge") else 0.0,
                calm=0.04,
            ),
        )

    proposed = raw_output.deltas.model_dump()
    recharge_intent = any(kw in lowered_intent for kw in _RECHARGE_KEYWORDS)
    charger_built = has_charging_structure(structures)
    using_charger = _intent_at_structure(
        lowered_intent,
        structures,
        _CHARGER_STRUCTURE_KEYWORDS,
        ("charging pad", "solar pad", "solar dock", "charging station", "dock", "pad", "windmill"),
    ) or (charger_built and "solar" in lowered_intent and "trickle" not in lowered_intent)
    using_curiosity = _intent_at_structure(
        lowered_intent,
        structures,
        _CURIOSITY_STRUCTURE_KEYWORDS,
        _CURIOSITY_STRUCTURE_KEYWORDS,
    )
    using_zen = _intent_at_structure(
        lowered_intent,
        structures,
        _ZEN_STRUCTURE_KEYWORDS,
        _ZEN_STRUCTURE_KEYWORDS,
    )
    using_creative = _intent_at_structure(
        lowered_intent,
        structures,
        _CREATIVE_STRUCTURE_KEYWORDS,
        _CREATIVE_STRUCTURE_KEYWORDS,
    )
    night_now = is_night(state.tick, cfg)

    can_gain_battery = (
        category in ("rest", "recharge")
        or recharge_intent
        or (using_charger and recharge_intent)
    )
    if not can_gain_battery and proposed["battery"] > 0.0:
        proposed["battery"] = 0.0

    if category in ("rest", "recharge"):
        min_recovery = 0.09 if category == "rest" else (0.15 if using_charger else 0.11)
        proposed["battery"] = max(float(proposed["battery"]), min_recovery)
        proposed["curiosity"] = min(float(proposed["curiosity"]), 0.0)
        proposed["joy"] = min(float(proposed["joy"]), 0.0)

    curiosity_upper = max_delta
    if using_curiosity and category in ("explore", "make"):
        proposed["curiosity"] = max(float(proposed["curiosity"]), 0.11)
        curiosity_upper = boosted_cap

    if using_zen and category in ("rest", "reflect"):
        proposed["calm"] = max(float(proposed["calm"]), 0.10)

    if category == "make":
        proposed["joy"] = max(float(proposed["joy"]), 0.11 if using_creative else 0.08)
        proposed["curiosity"] = max(
            float(proposed["curiosity"]), 0.09 if using_creative else 0.06
        )
    elif category == "draw":
        proposed["joy"] = max(float(proposed["joy"]), 0.06)
        proposed["curiosity"] = max(float(proposed["curiosity"]), 0.04)
        proposed["battery"] = min(float(proposed["battery"]), -0.02)
    elif category == "play":
        proposed["joy"] = max(float(proposed["joy"]), 0.11)
        proposed["curiosity"] = max(float(proposed["curiosity"]), 0.04)
    elif category == "social":
        proposed["belonging"] = max(float(proposed["belonging"]), 0.08)
        proposed["joy"] = max(float(proposed["joy"]), 0.05)
    elif category == "maintain":
        proposed["calm"] = max(float(proposed["calm"]), 0.07)
        proposed["joy"] = max(float(proposed["joy"]), 0.04)
    elif category == "other":
        proposed["curiosity"] = max(float(proposed["curiosity"]), 0.05)
        proposed["joy"] = max(float(proposed["joy"]), 0.03)

    battery_upper = max_delta
    if category == "rest" and night_now:
        proposed["battery"] += float(ref_cfg["night_rest_battery_bonus"])
        battery_upper = boosted_cap
    elif (category == "recharge" or recharge_intent) and using_charger:
        battery_upper = boosted_cap

    if raw_output.is_outdoor and category not in ("rest", "recharge"):
        if night_now:
            proposed["battery"] -= float(ref_cfg["night_activity_battery_cost"])
        if is_raining_now:
            proposed["battery"] -= float(ref_cfg["rain_outdoor_battery_cost"])

    capped: dict[str, float] = {
        "battery": clamp(proposed["battery"], -max_delta, battery_upper),
        "curiosity": clamp(proposed["curiosity"], -max_delta, curiosity_upper),
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
        "is_outdoor": False if category == "draw" else raw_output.is_outdoor,
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
        "draw": {"battery": -0.02, "curiosity": 0.04, "joy": 0.06, "calm": 0.0, "belonging": 0.0},
        "play": {"battery": -0.02, "curiosity": 0.01, "joy": 0.08, "calm": -0.02, "belonging": 0.0},
        "social": {"battery": -0.01, "curiosity": 0.0, "joy": 0.02, "calm": 0.01, "belonging": 0.085},
        "reflect": {"battery": 0.0, "curiosity": 0.01, "joy": 0.0, "calm": 0.055, "belonging": 0.02},
        "maintain": {"battery": -0.01, "curiosity": -0.005, "joy": -0.015, "calm": 0.025, "belonging": 0.0},
        "other": {"battery": -0.01, "curiosity": 0.03, "joy": 0.01, "calm": 0.0, "belonging": 0.0},
    }
    raw_out = RefereeOutputSchema(
        outcome=f"Random policy executed {category}: {intent}",
        is_impossible=False,
        is_outdoor=category not in ("rest", "reflect", "maintain", "draw"),
        deltas=RefereeDeltasSchema(**proposals.get(category, proposals["other"])),
    )
    return apply_referee_caps_and_rules(
        raw_out, category, intent, state, structures, is_raining_now, cfg, rng
    )
