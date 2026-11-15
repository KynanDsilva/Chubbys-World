"""Grounded prompt assembly for Chubby's decision, referee, and memory summarization calls."""

from __future__ import annotations

import json
from typing import Any

from state import WorldState, is_night

PERSONA_SEED = (
    "Chubby is a small, round, slightly clumsy robot who is curious, easily delighted, "
    "and a bit anxious when his battery gets low. He speaks in short, warm sentences. "
    "He knows he is a robot living in a small, quiet scrapyard town, and he has no job "
    "except looking after himself."
)


def _format_actions(actions: list[dict[str, Any]]) -> str:
    if not actions:
        return "None yet."
    lines: list[str] = []
    for row in actions:
        try:
            outcome = json.loads(row["outcome_json"]).get("outcome", "")
        except json.JSONDecodeError:
            outcome = row["outcome_json"]
        lines.append(
            f"- Tick {row['tick']} [{row['action']}]: {row['detail']} -> Outcome: {outcome}"
        )
    return "\n".join(lines)


def _format_diary(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return "No diary entries yet."
    return "\n".join(
        f"- [{row['sim_time']} | tick {row['tick']}]: {row['text']}" for row in entries
    )


def _format_structures(structures: list[dict[str, Any]]) -> str:
    if not structures:
        return "None (just a bare scrapyard lot; basic solar trickle and plain rest still work)."
    return "\n".join(
        f"- {row['name']} (unlocks: {row['unlocks']})" for row in structures
    )


def _format_memories(memories: list[dict[str, Any]]) -> str:
    if not memories:
        return "None yet."
    return "\n".join(
        f"- [ticks {row['tick_range']}]: {row['summary']}" for row in memories
    )


def build_decision_messages(
    state: WorldState,
    allowed_categories: list[str],
    active_events: list[str],
    recent_actions: list[dict[str, Any]],
    recent_diary: list[dict[str, Any]],
    memories: list[dict[str, Any]],
    structures: list[dict[str, Any]],
    unread_count: int,
    newly_read_messages: list[str],
    cfg: dict[str, Any],
) -> list[dict[str, str]]:
    """Assemble grounded messages for Chubby's decision LLM call."""
    system_prompt = (
        f"{PERSONA_SEED}\n\n"
        "Goal: keep your energy and mood in a healthy range.\n"
        "Grounding rule: only refer to facts, structures, events, actions, and messages "
        "listed in the context below; if unsure, say you don't remember.\n"
        "Return a valid JSON object matching the provided schema."
    )

    night_label = "Night" if is_night(state.tick, cfg) else "Daytime"
    events_text = "\n".join(f"- {e}" for e in active_events) if active_events else "None."
    read_msg_block = ""
    if newly_read_messages:
        formatted_msgs = "\n".join(f'- "{m}"' for m in newly_read_messages)
        read_msg_block = (
            f"\nMessages you just opened on the previous tick (react to them):\n{formatted_msgs}\n"
        )

    genesis_block = ""
    if state.tick <= int(cfg["genesis"]["genesis_ticks"]):
        genesis_block = (
            "\nGenesis note: you have just been built; what do you need to live here? "
            "You may include an optional `structure` request if something would help.\n"
        )

    user_prompt = (
        f"Tick: {state.tick} | Time: {state.sim_time} ({night_label})\n"
        f"Needs (0.0-1.0): battery={state.battery:.2f}, curiosity={state.curiosity:.2f}, "
        f"joy={state.joy:.2f}, calm={state.calm:.2f}, belonging={state.belonging:.2f}\n"
        f"Core parameters: energy={state.energy:.2f}, mood={state.mood:.2f}, "
        f"mood_modifier={state.mood_modifier:+.2f}\n\n"
        f"Structures in your lot:\n{_format_structures(structures)}\n\n"
        f"Active events:\n{events_text}\n\n"
        f"Unread inbox messages: {unread_count}\n"
        f"{read_msg_block}"
        f"{genesis_block}\n"
        f"Long-term memories:\n{_format_memories(memories)}\n\n"
        f"Recent actions (last {len(recent_actions)}):\n{_format_actions(recent_actions)}\n\n"
        f"Recent diary entries:\n{_format_diary(recent_diary)}\n\n"
        f"Allowed categories for this tick: {', '.join(allowed_categories)}\n"
        "Write your short thought, your free-text intent (why and what you want to do), "
        "and tag it with one allowed category."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_referee_messages(
    category: str,
    intent: str,
    state: WorldState,
    structures: list[dict[str, Any]],
    is_raining_now: bool,
    cfg: dict[str, Any],
) -> list[dict[str, str]]:
    """Assemble grounded messages for the Referee LLM resolution call."""
    system_prompt = (
        "You are the world Referee for a small scrapyard lot where a robot named Chubby lives. "
        "Resolve Chubby's free-text intent fairly and conservatively based ONLY on what actually exists.\n"
        "Rules:\n"
        "- Describe the outcome in 1 to 2 grounded sentences.\n"
        "- Set is_impossible=true if he tries to meet someone not present or use an unbuilt structure.\n"
        "- Set is_outdoor=true if the intent happens outside in the open yard.\n"
        "- Propose modest need deltas between -0.15 and +0.15 (do not inflate rewards).\n"
        "Return JSON matching the provided schema."
    )
    night_label = "Night" if is_night(state.tick, cfg) else "Daytime"
    weather = "Raining" if is_raining_now else "Clear"
    user_prompt = (
        f"Tick: {state.tick} | Time: {state.sim_time} ({night_label}) | Weather: {weather}\n"
        f"Current needs: battery={state.battery:.2f}, curiosity={state.curiosity:.2f}, "
        f"joy={state.joy:.2f}, calm={state.calm:.2f}, belonging={state.belonging:.2f}\n"
        f"Approved structures:\n{_format_structures(structures)}\n\n"
        f"Chubby's category: {category}\n"
        f"Chubby's intent: {intent}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_summary_messages(
    tick_range: str,
    actions: list[dict[str, Any]],
    diary_entries: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Assemble messages for the 50-tick grounded memory summarization job."""
    system_prompt = (
        "Summarize Chubby's recent logged history into two short grounded fields:\n"
        "1. long_term_memory: one short paragraph compressing only logged facts.\n"
        "2. self_discovery: a brief 'what I have learned about myself' observation derived from the logs.\n"
        "Never invent events or characters not present in the logs."
    )
    user_prompt = (
        f"Tick range: {tick_range}\n\n"
        f"Logged actions:\n{_format_actions(actions)}\n\n"
        f"Logged diary entries:\n{_format_diary(diary_entries)}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

