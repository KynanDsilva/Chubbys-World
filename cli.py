"""Terminal CLI for checking status, leaving messages, reviewing requests, and reading reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import db
import genesis
import report


def _open_db(db_path: str | None, cfg_path: str | None):
    cfg = db.load_config(cfg_path)
    resolved_db = db_path or str(cfg["database"]["path"])
    conn = db.get_connection(resolved_db)
    db.init_db(conn)
    return cfg, conn


def cmd_status(args: argparse.Namespace) -> int:
    cfg, conn = _open_db(args.db, args.config)
    try:
        latest = db.get_latest_state(conn)
        if latest is None:
            print("Chubby has not started yet (no state rows found).")
            return 0

        unread = db.get_unread_messages(conn)
        pending_reqs = db.get_requests(conn, status="pending")
        structures = db.get_structures(conn)
        recent_actions = db.get_recent_actions(conn, 1)
        recent_diary = db.get_recent_diary(conn, 1)

        print(f"Tick: {latest['tick']} | Sim time: {latest['sim_time']}")
        print(
            f"Needs -> battery={latest['battery']:.3f}, curiosity={latest['curiosity']:.3f}, "
            f"joy={latest['joy']:.3f}, calm={latest['calm']:.3f}, belonging={latest['belonging']:.3f}"
        )
        print(
            f"Core  -> energy={latest['energy']:.3f}, mood={latest['mood']:.3f} "
            f"(mood_modifier={latest['mood_modifier']:+.3f})"
        )
        print(
            f"World -> structures={len(structures)}, unread_messages={len(unread)}, "
            f"pending_requests={len(pending_reqs)}"
        )
        if recent_actions:
            act = recent_actions[-1]
            print(f"Last action [{act['action']}]: {act['detail']}")
        if recent_diary:
            entry = recent_diary[-1]
            print(f"Last diary [{entry['sim_time']}]: {entry['text']}")
        return 0
    finally:
        conn.close()


def cmd_leave(args: argparse.Namespace) -> int:
    _, conn = _open_db(args.db, args.config)
    try:
        text = " ".join(args.text).strip()
        if not text:
            print("Message cannot be empty.", file=sys.stderr)
            return 1
        latest = db.get_latest_state(conn)
        current_tick = int(latest["tick"]) if latest is not None else 0
        msg_id = db.insert_message(conn, text, current_tick)
        db.insert_event(
            conn,
            current_tick,
            "visitor_message",
            {"message_id": msg_id, "description": f"A visitor left a message (#{msg_id})."},
        )
        print(f"Queued message #{msg_id} at tick {current_tick}: {text}")
        return 0
    finally:
        conn.close()


def cmd_requests(args: argparse.Namespace) -> int:
    _, conn = _open_db(args.db, args.config)
    try:
        rows = db.get_requests(conn, status=None if args.all else "pending")
        if not rows:
            print("No matching structure requests.")
            return 0
        for row in rows:
            print(
                f"#{row['id']} [tick {row['tick']} | priority {row['priority']} | {row['status']}] "
                f"{row['name']}"
            )
            print(f"   purpose: {row['purpose']}")
            print(f"   why:     {row['why']}")
            if row["decision_note"]:
                print(f"   note:    {row['decision_note']}")
        return 0
    finally:
        conn.close()


def cmd_approve(args: argparse.Namespace) -> int:
    cfg, conn = _open_db(args.db, args.config)
    try:
        note = " ".join(args.note).strip() if args.note else "Approved"
        result = genesis.approve_request(conn, args.id, cfg, note=note)
        print(
            f"Approved request #{result['request_id']}: added structure "
            f"'{result['name']}' (#{result['structure_id']})."
        )
        return 0
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        conn.close()


def cmd_reject(args: argparse.Namespace) -> int:
    cfg, conn = _open_db(args.db, args.config)
    try:
        reason = " ".join(args.reason).strip() if args.reason else "Not needed right now"
        result = genesis.reject_request(conn, args.id, cfg, reason=reason)
        print(
            f"Rejected request #{result['request_id']} ('{result['name']}'): {result['reason']}"
        )
        return 0
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        conn.close()


def cmd_diary(args: argparse.Namespace) -> int:
    _, conn = _open_db(args.db, args.config)
    try:
        entries = (
            db.get_all_diary(conn)
            if args.limit <= 0
            else db.get_recent_diary(conn, args.limit)
        )
        if not entries:
            print("Diary is empty.")
            return 0
        for entry in entries:
            print(f"[{entry['sim_time']} | tick {entry['tick']}] {entry['text']}")
        return 0
    finally:
        conn.close()


def cmd_report(args: argparse.Namespace) -> int:
    cfg, conn = _open_db(args.db, args.config)
    try:
        metrics = report.generate_report(conn, cfg)
        print(report.format_report(metrics))
        return 0
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chubby CLI")
    parser.add_argument("--db", default=None, help="Path to SQLite database")
    parser.add_argument("--config", default=None, help="Path to config.toml")

    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="Show Chubby's current state")
    p_status.set_defaults(func=cmd_status)

    p_leave = sub.add_parser("leave", help="Leave a visitor message for Chubby")
    p_leave.add_argument("text", nargs="+", help="Message text")
    p_leave.set_defaults(func=cmd_leave)

    p_reqs = sub.add_parser("requests", help="List infrastructure requests")
    p_reqs.add_argument("--all", action="store_true", help="Include approved/rejected requests")
    p_reqs.set_defaults(func=cmd_requests)

    p_approve = sub.add_parser("approve", help="Approve a structure request")
    p_approve.add_argument("id", type=int, help="Request ID")
    p_approve.add_argument("note", nargs="*", help="Optional approval note")
    p_approve.set_defaults(func=cmd_approve)

    p_reject = sub.add_parser("reject", help="Reject a structure request")
    p_reject.add_argument("id", type=int, help="Request ID")
    p_reject.add_argument("reason", nargs="+", help="Reason for rejection")
    p_reject.set_defaults(func=cmd_reject)

    p_diary = sub.add_parser("diary", help="Read Chubby's diary")
    p_diary.add_argument("--limit", type=int, default=10, help="Number of entries (0 for all)")
    p_diary.set_defaults(func=cmd_diary)

    p_report = sub.add_parser("report", help="Print emergent-behavior metrics report")
    p_report.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

