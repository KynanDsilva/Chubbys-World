"""Two-step message inbox handling and periodic grounded memory summarization."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from typing import Any

import db
import prompt


def process_read_messages(conn: sqlite3.Connection, tick: int) -> list[str]:
    """Mark all unread inbox messages as read on `tick` and log a messages_read event."""
    unread = db.get_unread_messages(conn)
    if not unread:
        return []
    ids = [int(m["id"]) for m in unread]
    texts = [str(m["text"]) for m in unread]
    db.mark_messages_read(conn, ids)
    db.insert_event(
        conn,
        tick,
        "messages_read",
        {
            "message_ids": ids,
            "texts": texts,
            "description": f"Chubby opened {len(texts)} visitor message(s).",
        },
    )
    return texts


def get_newly_read_messages_for_prompt(
    conn: sqlite3.Connection, current_tick: int
) -> list[str]:
    """Return message texts opened on the immediately preceding tick for tick T+1 reaction."""
    if current_tick <= 1:
        return []
    events = db.get_events_for_tick(conn, current_tick - 1)
    texts: list[str] = []
    for ev in events:
        if ev["type"] == "messages_read":
            try:
                payload = json.loads(ev["payload"])
                texts.extend(str(t) for t in payload.get("texts", []))
            except json.JSONDecodeError:
                pass
    return texts


def maybe_summarize_memory(
    conn: sqlite3.Connection,
    tick: int,
    cfg: dict[str, Any],
    llm_client: Any,
) -> str | None:
    """Every `summarize_every_ticks`, compress logged history into a grounded memory note."""
    interval = int(cfg["memory"]["summarize_every_ticks"])
    if tick <= 0 or interval <= 0 or tick % interval != 0:
        return None

    start_tick = tick - interval + 1
    end_tick = tick
    tick_range = f"{start_tick}-{end_tick}"

    action_rows = conn.execute(
        "SELECT * FROM actions WHERE tick >= ? AND tick <= ? ORDER BY tick ASC, id ASC",
        (start_tick, end_tick),
    ).fetchall()
    diary_rows = conn.execute(
        "SELECT * FROM diary WHERE tick >= ? AND tick <= ? ORDER BY tick ASC, id ASC",
        (start_tick, end_tick),
    ).fetchall()

    actions = [dict(r) for r in action_rows]
    diary_entries = [dict(r) for r in diary_rows]
    if not actions and not diary_entries:
        return None

    categories = [a["action"] for a in actions]
    top_category = Counter(categories).most_common(1)[0][0] if categories else "rest"
    messages = prompt.build_summary_messages(tick_range, actions, diary_entries)
    summary_obj = llm_client.summarize(
        messages,
        tick=tick,
        context={"top_category": top_category, "tick_range": tick_range},
    )
    if summary_obj is None:
        return None

    combined = (
        f"{summary_obj.long_term_memory.strip()}\n"
        f"Self-discovery: {summary_obj.self_discovery.strip()}"
    )
    db.insert_memory(conn, tick_range, combined)
    return combined

