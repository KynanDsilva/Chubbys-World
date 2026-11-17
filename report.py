"""Emergent-behavior metrics report matching Iteration 1 Section 2 success criteria."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from typing import Any

import db
from antirepeat import jaccard_similarity


def generate_report(conn: sqlite3.Connection, cfg: dict[str, Any]) -> dict[str, Any]:
    """Compute Section 2 success metrics from the SQLite logs."""
    actions = db.get_all_actions(conn)
    states = db.get_all_states(conn)
    diary = db.get_all_diary(conn)
    events = db.get_all_events(conn)
    requests = db.get_requests(conn)
    structures = db.get_structures(conn)
    memories = db.get_memories(conn)

    total_ticks = len(actions)
    categories = [a["action"] for a in actions]
    intents = [a["detail"] for a in actions]

    overall_distinct_categories = len(set(categories))
    if total_ticks >= 100:
        window_counts = [
            len(set(categories[i : i + 100])) for i in range(total_ticks - 99)
        ]
        min_distinct_100 = min(window_counts)
        max_distinct_100 = max(window_counts)
    else:
        min_distinct_100 = overall_distinct_categories
        max_distinct_100 = overall_distinct_categories

    unique_intents = len({text.strip().lower() for text in intents if text.strip()})
    unique_intent_pct = (
        round((unique_intents / total_ticks) * 100.0, 2) if total_ticks > 0 else 0.0
    )

    longest_streak = 0
    current_streak = 0
    prev_cat: str | None = None
    for cat in categories:
        if cat == prev_cat:
            current_streak += 1
        else:
            current_streak = 1
            prev_cat = cat
        longest_streak = max(longest_streak, current_streak)

    sim_threshold = float(cfg["antirepeat"]["similarity_threshold"])
    high_sim_count = 0
    diary_texts = [d["text"] for d in diary]
    for idx in range(1, len(diary_texts)):
        max_prev_sim = max(
            jaccard_similarity(diary_texts[idx], diary_texts[j])
            for j in range(idx)
        )
        if max_prev_sim > sim_threshold:
            high_sim_count += 1
    diary_high_sim_pct = (
        round((high_sim_count / len(diary_texts)) * 100.0, 2)
        if diary_texts
        else 0.0
    )

    sim_states = [s for s in states if int(s["tick"]) >= 1]
    if not sim_states:
        sim_states = states
    in_range = sum(1 for s in sim_states if 0.2 <= float(s["mood"]) <= 0.9)
    mood_in_range_pct = (
        round((in_range / len(sim_states)) * 100.0, 2) if sim_states else 0.0
    )

    llm_failures = sum(1 for ev in events if ev["type"] == "llm_failure")
    llm_fallbacks = sum(1 for ev in events if ev["type"] == "llm_fallback")
    total_llm_attempts = max(1, total_ticks * 2 + llm_failures)
    invalid_llm_pct = (
        round((llm_failures / total_llm_attempts) * 100.0, 2)
        if total_ticks > 0
        else 0.0
    )

    genesis_ticks = int(cfg["genesis"]["genesis_ticks"])
    genesis_requests = [r for r in requests if int(r["tick"]) <= genesis_ticks]

    category_counts = dict(Counter(categories))
    self_discoveries: list[str] = []
    for mem in memories:
        for line in str(mem["summary"]).splitlines():
            if "self-discovery:" in line.lower() or "learned about myself" in line.lower():
                self_discoveries.append(line.strip())

    latest_state = states[-1] if states else None
    event_counts = dict(Counter(ev["type"] for ev in events))

    return {
        "total_ticks": total_ticks,
        "latest_sim_time": latest_state["sim_time"] if latest_state else "N/A",
        "overall_distinct_categories": overall_distinct_categories,
        "min_distinct_in_100_window": min_distinct_100,
        "max_distinct_in_100_window": max_distinct_100,
        "unique_intents": unique_intents,
        "unique_intent_pct": unique_intent_pct,
        "longest_category_streak": longest_streak,
        "diary_count": len(diary_texts),
        "diary_high_sim_count": high_sim_count,
        "diary_high_sim_pct": diary_high_sim_pct,
        "mood_in_range_pct": mood_in_range_pct,
        "llm_failures": llm_failures,
        "llm_fallbacks": llm_fallbacks,
        "invalid_llm_pct": invalid_llm_pct,
        "genesis_request_count": len(genesis_requests),
        "total_request_count": len(requests),
        "approved_structure_count": len(structures),
        "category_counts": category_counts,
        "event_counts": event_counts,
        "self_discoveries": self_discoveries,
        "sample_diary": diary[-3:] if diary else [],
        "targets_met": {
            "distinct_categories_ge_6": min_distinct_100 >= 6,
            "longest_streak_le_2": longest_streak <= 2,
            "diary_similarity_lt_15pct": diary_high_sim_pct < 15.0,
            "mood_healthy_ge_90pct": mood_in_range_pct >= 90.0,
            "invalid_llm_lt_1pct": invalid_llm_pct < 1.0,
            "genesis_requests_ge_3": len(genesis_requests) >= 3,
        },
    }


def format_report(metrics: dict[str, Any]) -> str:
    """Format computed metrics into a readable terminal report."""
    t = metrics["targets_met"]

    def _mark(flag: bool) -> str:
        return "PASS" if flag else "MISS"

    lines = [
        "=== Chubby Emergent Behavior Report ===",
        f"Total ticks: {metrics['total_ticks']} (Latest: {metrics['latest_sim_time']})",
        "",
        "Success Criteria:",
        (
            f"  [{_mark(t['distinct_categories_ge_6'])}] Distinct categories (100-tick min): "
            f"{metrics['min_distinct_in_100_window']} (overall: {metrics['overall_distinct_categories']}, "
            f"unique intents: {metrics['unique_intent_pct']}%)"
        ),
        (
            f"  [{_mark(t['longest_streak_le_2'])}] Longest same-category streak: "
            f"{metrics['longest_category_streak']} (target <= 2)"
        ),
        (
            f"  [{_mark(t['diary_similarity_lt_15pct'])}] Diary high similarity (>0.6): "
            f"{metrics['diary_high_sim_pct']}% ({metrics['diary_high_sim_count']}/{metrics['diary_count']}, target < 15%)"
        ),
        (
            f"  [{_mark(t['mood_healthy_ge_90pct'])}] Mood in [0.2, 0.9]: "
            f"{metrics['mood_in_range_pct']}% (target >= 90%)"
        ),
        (
            f"  [{_mark(t['invalid_llm_lt_1pct'])}] Invalid LLM outputs: "
            f"{metrics['invalid_llm_pct']}% (failures={metrics['llm_failures']}, fallbacks={metrics['llm_fallbacks']}, target < 1%)"
        ),
        (
            f"  [{_mark(t['genesis_requests_ge_3'])}] Genesis requests: "
            f"{metrics['genesis_request_count']} (total={metrics['total_request_count']}, approved structures={metrics['approved_structure_count']}, target >= 3)"
        ),
        "",
        f"Category counts: {json.dumps(metrics['category_counts'])}",
        f"Event counts: {json.dumps(metrics['event_counts'])}",
    ]

    if metrics["self_discoveries"]:
        lines.append("")
        lines.append("Self-discoveries (from memory summaries):")
        for item in metrics["self_discoveries"]:
            lines.append(f"  - {item}")

    if metrics["sample_diary"]:
        lines.append("")
        lines.append("Recent diary entries:")
        for entry in metrics["sample_diary"]:
            lines.append(f"  - [{entry['sim_time']} | tick {entry['tick']}] {entry['text']}")

    return "\n".join(lines)
