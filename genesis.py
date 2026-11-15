"""Genesis infrastructure requests, daily quota enforcement, and terminal approval/rejection."""

from __future__ import annotations

import sqlite3
from typing import Any

import db
from state import WorldState, apply_deltas


def can_request_structure(
    conn: sqlite3.Connection, tick: int, cfg: dict[str, Any]
) -> bool:
    """Return True if Chubby is permitted to submit a structure request on `tick`.

    Always allowed during Genesis (ticks 1..genesis_ticks). Afterwards, capped at
    requests_per_sim_day_after_genesis (default 1) per 144-tick simulated day.
    """
    genesis_ticks = int(cfg["genesis"]["genesis_ticks"])
    if tick <= genesis_ticks:
        return True

    ticks_per_day = int(cfg["clock"]["ticks_per_day"])
    day_index = (tick - 1) // ticks_per_day
    day_start = day_index * ticks_per_day + 1
    day_end = day_start + ticks_per_day - 1
    daily_limit = int(cfg["genesis"]["requests_per_sim_day_after_genesis"])

    existing = db.get_requests_in_tick_range(conn, day_start, day_end)
    return len(existing) < daily_limit


def record_structure_request(
    conn: sqlite3.Connection,
    tick: int,
    name: str,
    purpose: str,
    why: str,
    priority: int,
) -> int:
    """Persist a pending infrastructure request proposed by Chubby."""
    clean_priority = max(1, min(3, int(priority)))
    return db.insert_request(
        conn,
        tick=tick,
        name=name.strip(),
        purpose=purpose.strip(),
        why=why.strip(),
        priority=clean_priority,
        status="pending",
        decision_note="",
    )


def approve_request(
    conn: sqlite3.Connection,
    request_id: int,
    cfg: dict[str, Any],
    note: str = "Approved",
) -> dict[str, Any]:
    """Approve a pending structure request, add it to `structures`, and boost mood needs."""
    req = db.get_request_by_id(conn, request_id)
    if req is None:
        raise ValueError(f"Request #{request_id} not found.")
    if req["status"] != "pending":
        raise ValueError(
            f"Request #{request_id} is already '{req['status']}', not 'pending'."
        )

    latest = db.get_latest_state(conn)
    current_tick = int(latest["tick"]) if latest is not None else 0

    db.update_request_status(conn, request_id, "approved", note)
    structure_id = db.insert_structure(
        conn,
        name=req["name"],
        unlocks=req["purpose"],
        created_tick=current_tick,
    )

    joy_bonus = float(cfg["genesis"]["approval_joy_bonus"])
    belonging_bonus = float(cfg["genesis"]["approval_belonging_bonus"])
    if latest is not None:
        ws = WorldState(**latest)
        updated = apply_deltas(
            ws,
            {"joy": joy_bonus, "belonging": belonging_bonus},
            mood_mod_delta=0.0,
            cfg=cfg,
        )
        db.insert_state(conn, updated.model_dump())

    db.insert_event(
        conn,
        current_tick,
        "structure_approved",
        {
            "request_id": request_id,
            "structure_id": structure_id,
            "name": req["name"],
            "note": note,
            "description": f"Your request for '{req['name']}' was approved! ({note})",
        },
    )
    return {
        "request_id": request_id,
        "structure_id": structure_id,
        "name": req["name"],
        "status": "approved",
        "note": note,
    }


def reject_request(
    conn: sqlite3.Connection,
    request_id: int,
    cfg: dict[str, Any],
    reason: str = "Not feasible right now",
) -> dict[str, Any]:
    """Reject a pending structure request with a gentle mood cost and event log."""
    req = db.get_request_by_id(conn, request_id)
    if req is None:
        raise ValueError(f"Request #{request_id} not found.")
    if req["status"] != "pending":
        raise ValueError(
            f"Request #{request_id} is already '{req['status']}', not 'pending'."
        )

    latest = db.get_latest_state(conn)
    current_tick = int(latest["tick"]) if latest is not None else 0

    db.update_request_status(conn, request_id, "rejected", reason)
    mod_cost = float(cfg["genesis"]["rejection_mood_modifier_cost"])
    if latest is not None:
        ws = WorldState(**latest)
        updated = apply_deltas(ws, {}, mood_mod_delta=mod_cost, cfg=cfg)
        db.insert_state(conn, updated.model_dump())

    db.insert_event(
        conn,
        current_tick,
        "structure_rejected",
        {
            "request_id": request_id,
            "name": req["name"],
            "reason": reason,
            "description": f"Your request for '{req['name']}' was gently declined. Reason: {reason}",
        },
    )
    return {
        "request_id": request_id,
        "name": req["name"],
        "status": "rejected",
        "reason": reason,
    }

