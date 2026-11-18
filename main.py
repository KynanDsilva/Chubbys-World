"""Main autonomous tick loop for Chubby (Iteration 1)."""

from __future__ import annotations

import argparse
import os
import random
import sqlite3
import time
from pathlib import Path
from typing import Any

import antirepeat
import db
import events
import genesis
import llm
import memory
import prompt
import referee
import state
from svg_utils import sanitize_svg

STARTER_STRUCTURES: tuple[tuple[str, str], ...] = (
    (
        "Art Studio",
        "Draw sketches on an easel and hang them in the gallery",
    ),
)


def ensure_starter_structures(conn: sqlite3.Connection) -> None:
    """Seed the starter Art Studio at tick 0."""
    existing = {s["name"].lower() for s in db.get_structures(conn)}
    for name, unlocks in STARTER_STRUCTURES:
        if name.lower() not in existing:
            db.insert_structure(conn, name, unlocks, created_tick=0)


def ensure_initial_state(conn: sqlite3.Connection, cfg: dict[str, Any]) -> state.WorldState:
    """Return the latest state, initializing tick 0 and starter structures if missing."""
    ensure_starter_structures(conn)
    latest = db.get_latest_state(conn)
    if latest is None:
        initial = state.create_initial_state(cfg)
        db.insert_state(conn, initial.model_dump())
        return initial
    return state.WorldState(**latest)


def run_single_tick(
    conn: sqlite3.Connection,
    cfg: dict[str, Any],
    policy: str,
    llm_client: Any,
    rng: random.Random,
) -> dict[str, Any]:
    """Execute one complete simulation tick and persist all resulting records."""
    prev_state = ensure_initial_state(conn, cfg)
    tick = prev_state.tick + 1
    sim_time = state.format_sim_time(tick, cfg)
    current_state = prev_state.model_copy(update={"tick": tick, "sim_time": sim_time})

    rolled_event = events.roll_random_event(conn, tick, cfg, rng)
    if rolled_event is not None:
        current_state = state.apply_deltas(
            current_state,
            rolled_event.get("deltas", {}),
            mood_mod_delta=float(rolled_event.get("mood_mod_delta", 0.0)),
            cfg=cfg,
        )

    recent_prompt_actions = db.get_recent_actions(
        conn, int(cfg["memory"]["prompt_recent_actions"])
    )
    history_window = max(
        int(cfg["antirepeat"]["novelty_window"]),
        int(cfg["antirepeat"]["similarity_window"]),
        int(cfg["antirepeat"]["recent_penalty_window"]),
        int(cfg["antirepeat"]["block_last_n"]),
    )
    recent_history_actions = db.get_recent_actions(conn, history_window)
    recent_categories = [a["action"] for a in recent_history_actions]
    recent_intents = [a["detail"] for a in recent_history_actions]

    allowed_categories = antirepeat.get_allowed_categories(
        recent_categories, int(cfg["antirepeat"]["block_last_n"])
    )
    if policy != "random":
        allowed_categories = antirepeat.filter_categories_for_state(
            allowed_categories, current_state, recent_categories
        )
    unread_messages = db.get_unread_messages(conn)
    has_unread = len(unread_messages) > 0
    newly_read_messages = memory.get_newly_read_messages_for_prompt(conn, tick)
    allow_structure = genesis.can_request_structure(conn, tick, cfg)
    structures = db.get_structures(conn)
    active_events = events.get_active_event_descriptions(conn, tick, cfg)
    recent_diary = db.get_recent_diary(conn, int(cfg["memory"]["prompt_recent_diary"]))
    memories = db.get_memories(conn, limit=3)

    decision_fallback = False
    if policy == "random":
        category = rng.choice(allowed_categories)
        thought = f"Checking {category} options at {sim_time}."
        intent = f"Randomly explore {category} around the yard on tick {tick}."
        diary_entry = None
        should_read_messages = has_unread
        structure_req = None
    else:
        decision_model = llm.build_decision_model(
            allowed_categories=allowed_categories,
            has_unread_messages=has_unread,
            allow_structure_request=allow_structure,
        )
        decision_messages = prompt.build_decision_messages(
            state=current_state,
            allowed_categories=allowed_categories,
            active_events=active_events,
            recent_actions=recent_prompt_actions,
            recent_diary=recent_diary,
            memories=memories,
            structures=structures,
            unread_count=len(unread_messages),
            newly_read_messages=newly_read_messages,
            cfg=cfg,
            allow_structure_request=allow_structure,
            helicopter_visits=db.count_actions_mentioning(conn, "helicopter"),
        )
        requested_names = list(
            {r["name"] for r in db.get_requests(conn)}
            | {s["name"] for s in structures}
        )
        decision_obj, decision_fallback = llm_client.decide(
            decision_messages,
            decision_model,
            allowed_categories,
            tick=tick,
            context={
                "state": current_state,
                "sim_time": sim_time,
                "recent_categories": recent_categories,
                "structures": structures,
                "newly_read_messages": newly_read_messages,
                "active_events": active_events,
                "requested_names": requested_names,
            },
        )
        category = str(getattr(decision_obj, "category"))
        thought = str(getattr(decision_obj, "thought"))
        intent = str(getattr(decision_obj, "intent"))
        diary_entry = getattr(decision_obj, "diary_entry", None)
        should_read_messages = bool(getattr(decision_obj, "read_messages", False))
        structure_req = getattr(decision_obj, "structure", None)

        if policy == "llm":
            sim_threshold = 0.75
            high_sim = (
                antirepeat.max_intent_similarity(intent, recent_intents[-4:])
                > sim_threshold
            )
            if high_sim:
                best_intent = intent
                best_thought = thought
                best_sim = 2.0
                for attempt in range(2):
                    fresh_client = llm.MockLLMClient(
                        cfg, seed=tick * 31 + attempt * 97 + rng.randint(1, 1000)
                    )
                    alt_obj, _ = fresh_client.decide(
                        decision_messages,
                        decision_model,
                        [category],
                        tick=tick + attempt,
                        context={
                            "state": current_state,
                            "sim_time": sim_time,
                            "recent_categories": recent_categories,
                            "structures": structures,
                            "newly_read_messages": newly_read_messages,
                            "active_events": active_events,
                            "requested_names": requested_names,
                        },
                    )
                    alt_intent = str(getattr(alt_obj, "intent"))
                    alt_sim = antirepeat.max_intent_similarity(
                        alt_intent, recent_intents[-4:]
                    )
                    if alt_sim < best_sim:
                        best_sim = alt_sim
                        best_intent = alt_intent
                        best_thought = str(getattr(alt_obj, "thought"))
                    if alt_sim <= sim_threshold:
                        break
                intent = best_intent
                thought = best_thought

        if allow_structure and structure_req is None and policy == "mock":
            genesis_ticks = int(cfg["genesis"]["genesis_ticks"])
            requested_names_lower = {
                r["name"].lower()
                for r in db.get_requests(conn)
                if r["status"] in ("pending", "approved")
            } | {s["name"].lower() for s in structures}
            if tick <= genesis_ticks or len(db.get_requests(conn, status="pending")) == 0:
                for candidate in llm.MockLLMClient._GENESIS_STRUCTURES:
                    if candidate["name"].lower() not in requested_names_lower:
                        structure_req = llm.StructureRequestSchema(**candidate)
                        break

    read_texts: list[str] = []
    extra_belonging = 0.0
    draw_downgrade_note: str | None = None
    category, draw_downgrade_note = referee.gate_draw_category(
        category, intent, structures, recent_categories=recent_categories
    )
    if should_read_messages and has_unread:
        read_texts = memory.process_read_messages(conn, tick)
        extra_belonging = 0.20

    recorded_request_id: int | None = None
    if structure_req is not None and allow_structure:
        recorded_request_id = genesis.record_structure_request(
            conn,
            tick=tick,
            name=structure_req.name,
            purpose=structure_req.purpose,
            why=structure_req.why,
            priority=structure_req.priority,
        )

    if (
        referee.has_curiosity_structure(structures)
        and category in ("explore", "reflect", "make")
        and (recorded_request_id is not None or tick % 8 == 0)
    ):
        db.insert_event(
            conn,
            tick,
            "curiosity_discovery",
            {
                "description": (
                    "Using his curiosity structure, Chubby spots new details "
                    "and sketches fresh blueprint ideas for the yard."
                )
            },
        )

    is_raining_now = events.is_raining(conn, tick, cfg)
    ref_fallback = False
    if policy == "random":
        outcome_dict = referee.resolve_random_policy_action(
            category, intent, current_state, structures, is_raining_now, cfg, rng
        )
    else:
        ref_messages = prompt.build_referee_messages(
            category=category,
            intent=intent,
            state=current_state,
            structures=structures,
            is_raining_now=is_raining_now,
            cfg=cfg,
        )
        raw_ref_out, ref_fallback = llm_client.referee(
            ref_messages,
            category=category,
            tick=tick,
            context={"intent": intent, "structures": structures},
        )
        outcome_dict = referee.apply_referee_caps_and_rules(
            raw_ref_out,
            category,
            intent,
            current_state,
            structures,
            is_raining_now,
            cfg,
            rng,
        )

    if draw_downgrade_note:
        outcome_dict["outcome"] = f"{outcome_dict['outcome']} {draw_downgrade_note}"

    drawing_info: dict[str, Any] | None = None
    if category == "draw" and not outcome_dict.get("is_impossible"):
        existing = db.get_recent_drawings(conn, 5)
        ctx = {
            "state": current_state,
            "recent_titles": [d["title"] for d in existing],
            "drawing_count": int(
                conn.execute("SELECT COUNT(*) FROM drawings").fetchone()[0]
            ),
        }
        if llm_client is not None and hasattr(llm_client, "generate_drawing"):
            drawing = llm_client.generate_drawing(intent, tick, ctx)
        else:
            drawing = llm.safe_procedural_drawing(
                tick * 1009 + ctx["drawing_count"], intent=intent
            )
        svg_clean = sanitize_svg(drawing.get("svg", ""))
        if svg_clean is None:
            drawing = llm.safe_procedural_drawing(
                tick * 1009 + ctx["drawing_count"], intent=intent
            )
            svg_clean = drawing["svg"]
        title = str(drawing.get("title") or "Untitled Sketch").strip()[:60]
        lowered_intent = intent.lower()
        wall_target: str | None = None
        for kw, loc in [
            ("bakery", "Bakery"),
            ("library", "Library"),
            ("cafe", "Cafe"),
            ("workshop", "Workshop"),
            ("post office", "Post Office"),
            ("apartment", "Apartments"),
            ("fence", "Perimeter Wall"),
            ("studio", "Art Studio"),
            ("easel", "Art Studio"),
            ("gallery", "Art Studio"),
        ]:
            if kw in lowered_intent:
                wall_target = loc
                break
        if not wall_target:
            town_rotation = [
                "Bakery",
                "Library",
                "Cafe",
                "Workshop",
                "Post Office",
                "Apartments",
                "Art Studio",
            ]
            wall_target = town_rotation[ctx["drawing_count"] % len(town_rotation)]

        drawing_id = db.insert_drawing(conn, tick, title, svg_clean, location=wall_target)
        drawing_info = {"id": drawing_id, "title": title, "location": wall_target}
        outcome_dict["outcome"] = (
            f"{outcome_dict['outcome']} He finished a sketch called '{title}' "
            f"and put it up on the {wall_target} wall."
        )
        outcome_dict["drawing"] = drawing_info

    if read_texts:
        joined = " | ".join(read_texts)
        outcome_dict["outcome"] = (
            f"{outcome_dict['outcome']} (Opened message(s): {joined})"
        )
        outcome_dict["read_messages"] = read_texts

    anti_info = antirepeat.evaluate_antirepeat(
        category, intent, recent_categories, recent_intents, cfg
    )
    after_decay = state.apply_passive_decay(current_state, category, cfg)
    combined_deltas = dict(outcome_dict["deltas"])
    combined_deltas["belonging"] = float(combined_deltas.get("belonging", 0.0)) + extra_belonging
    final_state = state.apply_deltas(
        after_decay,
        combined_deltas,
        mood_mod_delta=float(anti_info["mood_mod_delta"]),
        cfg=cfg,
    )

    state_dict = final_state.model_dump()
    db.insert_state(conn, state_dict)
    db.insert_action(conn, tick, category, intent, thought, outcome_dict, state_dict)
    if diary_entry and str(diary_entry).strip():
        db.insert_diary(conn, tick, sim_time, str(diary_entry).strip())

    memory_summary: str | None = None
    if policy != "random":
        memory_summary = memory.maybe_summarize_memory(conn, tick, cfg, llm_client)

    return {
        "tick": tick,
        "sim_time": sim_time,
        "category": category,
        "intent": intent,
        "thought": thought,
        "outcome": outcome_dict["outcome"],
        "event": rolled_event["type"] if rolled_event else None,
        "request_id": recorded_request_id,
        "read_messages_count": len(read_texts),
        "memory_summary": memory_summary,
        "decision_fallback": decision_fallback,
        "ref_fallback": ref_fallback,
        "state": state_dict,
        "drawing": drawing_info,
    }


def build_llm_client(
    policy: str, cfg: dict[str, Any], conn: sqlite3.Connection, rng: random.Random
) -> Any:
    """Instantiate the requested LLM or local policy client."""
    if policy == "groq":
        return llm.GroqClient(cfg, conn=conn)
    if policy == "llm":
        if llm.is_groq_configured(cfg):
            return llm.GroqClient(cfg, conn=conn)
        return llm.OllamaClient(cfg, conn=conn)
    if policy == "mock":
        return llm.MockLLMClient(cfg, seed=rng.randint(0, 1_000_000))
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Chubby's autonomous tick loop")
    parser.add_argument(
        "--ticks",
        type=int,
        default=0,
        help="Number of ticks to run (0 = run until Ctrl+C)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Seconds between ticks (defaults to config.toml tick_interval_seconds)",
    )
    parser.add_argument(
        "--policy",
        choices=("llm", "groq", "random", "mock"),
        default="llm",
        help="Decision policy: llm (auto-Groq/Ollama), groq (cloud LPU), mock, or random",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--db", default=None, help="SQLite database path")
    parser.add_argument("--config", default=None, help="Path to config.toml")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing SQLite database before starting",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-tick terminal summaries"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = db.load_config(args.config)
    db_path = Path(args.db or str(cfg["database"]["path"]))

    if args.reset and db_path.exists():
        os.remove(db_path)

    interval = (
        float(args.interval)
        if args.interval is not None
        else float(cfg["clock"]["tick_interval_seconds"])
    )
    rng = random.Random(args.seed)

    conn = db.get_connection(db_path)
    try:
        db.init_db(conn)
        ensure_initial_state(conn, cfg)
        llm_client = build_llm_client(args.policy, cfg, conn, rng)

        completed = 0
        while args.ticks <= 0 or completed < args.ticks:
            result = run_single_tick(conn, cfg, args.policy, llm_client, rng)
            completed += 1
            if not args.quiet:
                st = result["state"]
                ev_str = f" | event={result['event']}" if result["event"] else ""
                req_str = (
                    f" | request=#{result['request_id']}"
                    if result["request_id"]
                    else ""
                )
                print(
                    f"[Tick {result['tick']} | {result['sim_time']}] "
                    f"{result['category']}: {result['intent'][:72]} | "
                    f"bat={st['battery']:.2f} energy={st['energy']:.2f} "
                    f"mood={st['mood']:.2f}{ev_str}{req_str}"
                )
            if (args.ticks <= 0 or completed < args.ticks) and interval > 0:
                time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped Chubby cleanly.")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
