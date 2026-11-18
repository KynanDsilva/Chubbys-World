"""SQLite database schema, configuration loading, and query helpers for Chubby."""

from __future__ import annotations

import json
import os
import sqlite3
import tomllib
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.toml"


def load_dotenv(env_path: str | Path | None = None) -> None:
    """Load key-value pairs from a .env file into os.environ if not already set."""
    p = Path(env_path) if env_path is not None else Path(__file__).resolve().parent / ".env"
    if not p.is_file():
        return
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load TOML configuration from disk and load .env if present."""
    load_dotenv()
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with cfg_path.open("rb") as f:
        return tomllib.load(f)


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection configured with row access by column name."""
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create all Iteration 1 tables if they do not already exist."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS state (
            tick INTEGER PRIMARY KEY,
            sim_time TEXT NOT NULL,
            battery REAL NOT NULL,
            curiosity REAL NOT NULL,
            joy REAL NOT NULL,
            calm REAL NOT NULL,
            belonging REAL NOT NULL,
            mood_modifier REAL NOT NULL,
            energy REAL NOT NULL,
            mood REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tick INTEGER NOT NULL,
            action TEXT NOT NULL,
            detail TEXT NOT NULL,
            thought TEXT NOT NULL,
            outcome_json TEXT NOT NULL,
            state_after_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tick INTEGER NOT NULL,
            type TEXT NOT NULL,
            payload TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            received_tick INTEGER NOT NULL,
            read INTEGER NOT NULL DEFAULT 0,
            drawing TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS diary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tick INTEGER NOT NULL,
            sim_time TEXT NOT NULL,
            text TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tick_range TEXT NOT NULL,
            summary TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tick INTEGER NOT NULL,
            name TEXT NOT NULL,
            purpose TEXT NOT NULL,
            why TEXT NOT NULL,
            priority INTEGER NOT NULL,
            status TEXT NOT NULL,
            decision_note TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS structures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            unlocks TEXT NOT NULL,
            created_tick INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS drawings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tick INTEGER,
            title TEXT,
            svg TEXT,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        DROP TRIGGER IF EXISTS keep_only_solar_at_tick0;
        """
    )
    msg_cols = {
        (row[1] if isinstance(row, tuple) else row["name"])
        for row in conn.execute("PRAGMA table_info(messages)").fetchall()
    }
    if "drawing" not in msg_cols:
        conn.execute("ALTER TABLE messages ADD COLUMN drawing TEXT NOT NULL DEFAULT ''")
    drawing_cols = {
        (row[1] if isinstance(row, tuple) else row["name"])
        for row in conn.execute("PRAGMA table_info(drawings)").fetchall()
    }
    if "location" not in drawing_cols:
        conn.execute(
            "ALTER TABLE drawings ADD COLUMN location TEXT NOT NULL DEFAULT 'Art Studio'"
        )
    conn.commit()
    _run_town_revamp_migration(conn)


TOWN_REVAMP_MARKER = "town_revamp_v1"


def _run_town_revamp_migration(conn: sqlite3.Connection) -> None:
    """One-time: drop old built structures and old requests (backing up the file first)."""
    if conn.execute(
        "SELECT 1 FROM meta WHERE key = ?", (TOWN_REVAMP_MARKER,)
    ).fetchone():
        return
    has_old = conn.execute(
        "SELECT (SELECT COUNT(*) FROM structures WHERE created_tick != 0) "
        "+ (SELECT COUNT(*) FROM requests)"
    ).fetchone()[0]
    if has_old:
        db_file = ""
        for row in conn.execute("PRAGMA database_list").fetchall():
            if row[1] == "main":
                db_file = row[2]
        if db_file and Path(db_file).exists():
            bak = Path(db_file + ".bak")
            if not bak.exists():
                dest = sqlite3.connect(str(bak))
                try:
                    conn.backup(dest)
                finally:
                    dest.close()
        conn.execute("DELETE FROM structures WHERE created_tick != 0")
        conn.execute("DELETE FROM requests")
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, '1')",
        (TOWN_REVAMP_MARKER,),
    )
    conn.commit()


def reset_db(conn: sqlite3.Connection) -> None:
    """Clear all simulation data from all tables while preserving schema."""
    conn.executescript(
        """
        DELETE FROM state;
        DELETE FROM actions;
        DELETE FROM events;
        DELETE FROM messages;
        DELETE FROM diary;
        DELETE FROM memories;
        DELETE FROM requests;
        DELETE FROM structures;
        DELETE FROM drawings;
        DELETE FROM sqlite_sequence;
        """
    )
    conn.commit()


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(r) for r in rows]


def get_latest_state(conn: sqlite3.Connection) -> dict[str, Any] | None:
    """Return the most recent state row, or None if the database is empty."""
    row = conn.execute("SELECT * FROM state ORDER BY tick DESC LIMIT 1").fetchone()
    return _row_to_dict(row)


def insert_state(conn: sqlite3.Connection, state_dict: dict[str, Any]) -> None:
    """Insert or replace a state snapshot for a given tick."""
    conn.execute(
        """
        INSERT OR REPLACE INTO state (
            tick, sim_time, battery, curiosity, joy, calm, belonging,
            mood_modifier, energy, mood
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            state_dict["tick"],
            state_dict["sim_time"],
            state_dict["battery"],
            state_dict["curiosity"],
            state_dict["joy"],
            state_dict["calm"],
            state_dict["belonging"],
            state_dict["mood_modifier"],
            state_dict["energy"],
            state_dict["mood"],
        ),
    )
    conn.commit()


def get_all_states(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return all state rows ordered by tick ascending."""
    rows = conn.execute("SELECT * FROM state ORDER BY tick ASC").fetchall()
    return _rows_to_dicts(rows)


def insert_action(
    conn: sqlite3.Connection,
    tick: int,
    action: str,
    detail: str,
    thought: str,
    outcome: dict[str, Any],
    state_after: dict[str, Any],
) -> int:
    """Log an executed action and its outcome/resulting state JSON."""
    cur = conn.execute(
        """
        INSERT INTO actions (tick, action, detail, thought, outcome_json, state_after_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            tick,
            action,
            detail,
            thought,
            json.dumps(outcome, ensure_ascii=False),
            json.dumps(state_after, ensure_ascii=False),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_recent_actions(conn: sqlite3.Connection, limit: int) -> list[dict[str, Any]]:
    """Return the last `limit` actions in chronological order (oldest to newest)."""
    if limit <= 0:
        return []
    rows = conn.execute(
        "SELECT * FROM actions ORDER BY tick DESC, id DESC LIMIT ?", (limit,)
    ).fetchall()
    return _rows_to_dicts(list(reversed(rows)))


def get_all_actions(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return all actions ordered by tick ascending."""
    rows = conn.execute("SELECT * FROM actions ORDER BY tick ASC, id ASC").fetchall()
    return _rows_to_dicts(rows)


def insert_event(
    conn: sqlite3.Connection,
    tick: int,
    event_type: str,
    payload: dict[str, Any] | str,
) -> int:
    """Insert an event record with JSON-encoded payload."""
    payload_str = (
        payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    )
    cur = conn.execute(
        "INSERT INTO events (tick, type, payload) VALUES (?, ?, ?)",
        (tick, event_type, payload_str),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_recent_events(
    conn: sqlite3.Connection,
    since_tick: int,
    event_type: str | None = None,
) -> list[dict[str, Any]]:
    """Return events with tick >= since_tick, optionally filtered by type."""
    if event_type is None:
        rows = conn.execute(
            "SELECT * FROM events WHERE tick >= ? ORDER BY tick ASC, id ASC",
            (since_tick,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM events WHERE tick >= ? AND type = ? ORDER BY tick ASC, id ASC",
            (since_tick, event_type),
        ).fetchall()
    return _rows_to_dicts(rows)


def get_events_for_tick(conn: sqlite3.Connection, tick: int) -> list[dict[str, Any]]:
    """Return all events logged for an exact tick."""
    rows = conn.execute(
        "SELECT * FROM events WHERE tick = ? ORDER BY id ASC", (tick,)
    ).fetchall()
    return _rows_to_dicts(rows)


def get_all_events(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return all events ordered by tick ascending."""
    rows = conn.execute("SELECT * FROM events ORDER BY tick ASC, id ASC").fetchall()
    return _rows_to_dicts(rows)


def insert_message(
    conn: sqlite3.Connection, text: str, received_tick: int, drawing: str = ""
) -> int:
    """Insert a new unread visitor message into the inbox."""
    cur = conn.execute(
        "INSERT INTO messages (text, received_tick, read, drawing) VALUES (?, ?, 0, ?)",
        (text, received_tick, drawing),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_recent_messages(
    conn: sqlite3.Connection, limit: int = 10
) -> list[dict[str, Any]]:
    """Return the most recent visitor messages ordered newest-first."""
    if limit <= 0:
        return []
    rows = conn.execute(
        "SELECT * FROM messages ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return _rows_to_dicts(rows)


def get_unread_messages(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return all unread messages in arrival order."""
    rows = conn.execute(
        "SELECT * FROM messages WHERE read = 0 ORDER BY id ASC"
    ).fetchall()
    return _rows_to_dicts(rows)


def mark_messages_read(conn: sqlite3.Connection, message_ids: list[int]) -> None:
    """Mark the specified message IDs as read."""
    if not message_ids:
        return
    placeholders = ",".join("?" for _ in message_ids)
    conn.execute(
        f"UPDATE messages SET read = 1 WHERE id IN ({placeholders})",
        tuple(message_ids),
    )
    conn.commit()


def insert_diary(
    conn: sqlite3.Connection, tick: int, sim_time: str, text: str
) -> int:
    """Insert a diary entry."""
    cur = conn.execute(
        "INSERT INTO diary (tick, sim_time, text) VALUES (?, ?, ?)",
        (tick, sim_time, text),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_recent_diary(conn: sqlite3.Connection, limit: int) -> list[dict[str, Any]]:
    """Return the last `limit` diary entries in chronological order."""
    if limit <= 0:
        return []
    rows = conn.execute(
        "SELECT * FROM diary ORDER BY tick DESC, id DESC LIMIT ?", (limit,)
    ).fetchall()
    return _rows_to_dicts(list(reversed(rows)))


def get_all_diary(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return all diary entries in chronological order."""
    rows = conn.execute("SELECT * FROM diary ORDER BY tick ASC, id ASC").fetchall()
    return _rows_to_dicts(rows)


def insert_memory(conn: sqlite3.Connection, tick_range: str, summary: str) -> int:
    """Insert a long-term memory summary entry."""
    cur = conn.execute(
        "INSERT INTO memories (tick_range, summary) VALUES (?, ?)",
        (tick_range, summary),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_memories(
    conn: sqlite3.Connection, limit: int | None = None
) -> list[dict[str, Any]]:
    """Return memory summaries in chronological order (or the last `limit` items)."""
    if limit is None:
        rows = conn.execute("SELECT * FROM memories ORDER BY id ASC").fetchall()
        return _rows_to_dicts(rows)
    if limit <= 0:
        return []
    rows = conn.execute(
        "SELECT * FROM memories ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return _rows_to_dicts(list(reversed(rows)))


def insert_request(
    conn: sqlite3.Connection,
    tick: int,
    name: str,
    purpose: str,
    why: str,
    priority: int,
    status: str = "pending",
    decision_note: str = "",
) -> int:
    """Insert a structure request from Chubby."""
    cur = conn.execute(
        """
        INSERT INTO requests (tick, name, purpose, why, priority, status, decision_note)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (tick, name, purpose, why, priority, status, decision_note),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_requests(
    conn: sqlite3.Connection, status: str | None = None
) -> list[dict[str, Any]]:
    """Return structure requests, optionally filtered by status."""
    if status is None:
        rows = conn.execute("SELECT * FROM requests ORDER BY id ASC").fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM requests WHERE status = ? ORDER BY id ASC", (status,)
        ).fetchall()
    return _rows_to_dicts(rows)


def get_request_by_id(
    conn: sqlite3.Connection, request_id: int
) -> dict[str, Any] | None:
    """Return a single structure request by primary key."""
    row = conn.execute(
        "SELECT * FROM requests WHERE id = ?", (request_id,)
    ).fetchone()
    return _row_to_dict(row)


def update_request_status(
    conn: sqlite3.Connection,
    request_id: int,
    status: str,
    decision_note: str = "",
) -> None:
    """Update the review status and decision note for a structure request."""
    conn.execute(
        "UPDATE requests SET status = ?, decision_note = ? WHERE id = ?",
        (status, decision_note, request_id),
    )
    conn.commit()


def get_requests_in_tick_range(
    conn: sqlite3.Connection, start_tick: int, end_tick: int
) -> list[dict[str, Any]]:
    """Return all structure requests submitted within [start_tick, end_tick]."""
    rows = conn.execute(
        "SELECT * FROM requests WHERE tick >= ? AND tick <= ? ORDER BY id ASC",
        (start_tick, end_tick),
    ).fetchall()
    return _rows_to_dicts(rows)


def insert_structure(
    conn: sqlite3.Connection, name: str, unlocks: str, created_tick: int
) -> int:
    """Insert an approved structure into the world."""
    cur = conn.execute(
        "INSERT INTO structures (name, unlocks, created_tick) VALUES (?, ?, ?)",
        (name, unlocks, created_tick),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_structures(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return all built structures."""
    rows = conn.execute("SELECT * FROM structures ORDER BY id ASC").fetchall()
    return _rows_to_dicts(rows)



def count_actions_mentioning(conn: sqlite3.Connection, keyword: str) -> int:
    """Count logged actions whose intent mentions `keyword` (case-insensitive)."""
    row = conn.execute(
        "SELECT COUNT(*) FROM actions WHERE LOWER(detail) LIKE ?",
        (f"%{keyword.lower()}%",),
    ).fetchone()
    return int(row[0]) if row else 0


def insert_drawing(
    conn: sqlite3.Connection,
    tick: int,
    title: str,
    svg: str,
    location: str = "Art Studio",
) -> int:
    """Store one of Chubby's SVG sketches, including the wall location where it is displayed."""
    cur = conn.execute(
        "INSERT INTO drawings (tick, title, svg, location, created_at) "
        "VALUES (?, ?, ?, ?, datetime('now'))",
        (tick, title, svg, location),
    )
    conn.commit()
    return int(cur.lastrowid)


def update_drawing_location(
    conn: sqlite3.Connection, drawing_id: int, location: str
) -> None:
    """Move or hang a drawing on a specific wall."""
    conn.execute(
        "UPDATE drawings SET location = ? WHERE id = ?",
        (location, drawing_id),
    )
    conn.commit()


def get_recent_drawings(
    conn: sqlite3.Connection, limit: int = 15
) -> list[dict[str, Any]]:
    """Return the newest `limit` drawings, newest first."""
    if limit <= 0:
        return []
    rows = conn.execute(
        "SELECT id, tick, title, svg, location, created_at FROM drawings ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return _rows_to_dicts(rows)
