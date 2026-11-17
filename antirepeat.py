"""Three-layer anti-repetition logic: category blocking, mood penalties, and Jaccard similarity."""

from __future__ import annotations

import re
from typing import Any

ALL_CATEGORIES: tuple[str, ...] = (
    "rest",
    "recharge",
    "explore",
    "make",
    "draw",
    "play",
    "social",
    "reflect",
    "maintain",
    "other",
)

_WORD_RE = re.compile(r"[a-z0-9]+")


def get_allowed_categories(
    recent_categories: list[str], block_last_n: int = 2
) -> list[str]:
    """Return allowed action categories after removing those used in the last N ticks."""
    if block_last_n <= 0:
        return list(ALL_CATEGORIES)
    blocked = set(recent_categories[-block_last_n:])
    allowed = [cat for cat in ALL_CATEGORIES if cat not in blocked]
    return allowed if allowed else list(ALL_CATEGORIES)


def filter_categories_for_state(
    allowed: list[str], state: Any, recent_categories: list[str]
) -> list[str]:
    """Filter redundant passive categories when needs are already satisfied."""
    filtered = list(allowed)
    recent_3 = set(recent_categories[-3:])
    filtered = [
        c for c in filtered if not (c in recent_3 and c in ("rest", "recharge", "reflect"))
    ]
    if float(state.battery) >= 0.78:
        filtered = [c for c in filtered if c != "recharge"]
    if float(state.battery) >= 0.75 and float(state.calm) >= 0.82:
        filtered = [c for c in filtered if c != "rest"]
    if float(state.calm) >= 0.88:
        filtered = [c for c in filtered if c != "reflect"]
    return filtered if len(filtered) >= 4 else list(allowed)


def tokenize_words(text: str) -> set[str]:
    """Extract lowercase alphanumeric word tokens from text."""
    return set(_WORD_RE.findall(text.lower()))


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity between word sets of two strings."""
    words_a = tokenize_words(text_a)
    words_b = tokenize_words(text_b)
    if not words_a or not words_b:
        return 0.0
    union = words_a | words_b
    if not union:
        return 0.0
    return len(words_a & words_b) / len(union)


def max_intent_similarity(intent: str, recent_intents: list[str]) -> float:
    """Return the highest Jaccard similarity between `intent` and `recent_intents`."""
    if not recent_intents:
        return 0.0
    return max(jaccard_similarity(intent, prev) for prev in recent_intents)


def evaluate_antirepeat(
    chosen_category: str,
    intent: str,
    recent_categories: list[str],
    recent_intents: list[str],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Compute mood_modifier adjustments for category frequency and intent similarity."""
    anti_cfg = cfg["antirepeat"]
    penalty_window = int(anti_cfg["recent_penalty_window"])
    novelty_window = int(anti_cfg["novelty_window"])
    similarity_window = int(anti_cfg["similarity_window"])
    repeat_penalty = float(anti_cfg["repeat_mood_penalty"])
    novelty_bonus = float(anti_cfg["novelty_mood_bonus"])
    sim_threshold = float(anti_cfg["similarity_threshold"])

    window_5 = recent_categories[-penalty_window:] if penalty_window > 0 else []
    window_10 = recent_categories[-novelty_window:] if novelty_window > 0 else []
    intents_window = recent_intents[-similarity_window:] if similarity_window > 0 else []

    occurrences_5 = sum(1 for cat in window_5 if cat == chosen_category)
    is_novel = chosen_category not in window_10
    max_sim = max_intent_similarity(intent, intents_window)
    sim_repeat = max_sim > sim_threshold

    mood_mod_delta = 0.0
    if occurrences_5 > 0:
        mood_mod_delta -= repeat_penalty * occurrences_5
    if is_novel:
        mood_mod_delta += novelty_bonus
    if sim_repeat:
        mood_mod_delta -= repeat_penalty

    return {
        "mood_mod_delta": round(mood_mod_delta, 6),
        "category_occurrences_5": occurrences_5,
        "novelty_bonus": is_novel,
        "max_similarity": round(max_sim, 4),
        "similarity_penalty": sim_repeat,
    }
