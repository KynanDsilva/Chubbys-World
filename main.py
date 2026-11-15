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


def ensure_initial_state(conn: sqlite3.Connection, cfg: dict[str, Any]) -> state.WorldState:
    """Return the latest state, initializing tick 0 if the database is empty."""
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

    # 1. Roll at most one random external event
    rolled_event = events.roll_random_event(conn, tick, cfg, rng)
    if rolled_event is not None:
        current_state = state.apply_deltas(
            current_state,
            rolled_event.get("deltas", {}),
            mood_mod_delta=float(rolled_event.get("mood_mod_delta", 0.0)),
            cfg=cfg,
        )

    # 2. Gather grounded history, anti-repeat constraints, unread inbox, and genesis quota
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
    unread_messages = db.get_unread_messages(conn)
    has_unread = len(unread_messages) > 0
    newly_read_messages = memory.get_newly_read_messages_for_prompt(conn, tick)
    allow_structure = genesis.can_request_structure(conn, tick, cfg)
    structures = db.get_structures(conn)
    active_events = events.get_active_event_descriptions(conn, tick, cfg)
    recent_diary = db.get_recent_diary(conn, int(cfg["memory"]["prompt_recent_diary"]))
    memories = db.get_memories(conn, limit=3)

    # 3. Decision call (or random policy)
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
        )
        requested_names = [r["name"] for r in db.get_requests(conn)]
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

    # 4. Step 1 of two-step read_messages flow
    read_texts: list[str] = []
    extra_belonging = 0.0
    if should_read_messages and has_unread:
        read_texts = memory.process_read_messages(conn, tick)
        extra_belonging = 0.20

    # 5. Record structure request if permitted and supplied
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

    # 6. Referee resolution + deterministic caps/noise
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

    if read_texts:
        joined = " | ".join(read_texts)
        outcome_dict["outcome"] = (
            f"{outcome_dict['outcome']} (Opened message(s): {joined})"
        )
        outcome_dict["read_messages"] = read_texts

    # 7. Anti-repetition penalties/bonuses, passive decay, and final need deltas
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

    # 8. Persist state, action, diary, and periodic memory summarization
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
    }


def _build_llm_client(
    policy: str, cfg: dict[str, Any], conn: sqlite3.Connection, rng: random.Random
) -> Any:
    if policy == "llm":
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
        choices=("llm", "random", "mock"),
        default="llm",
        help="Decision policy: llm (Ollama), mock (deterministic local), or random",
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
        llm_client = _build_llm_client(args.policy, cfg, conn, rng)

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

