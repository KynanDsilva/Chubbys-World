"""Local HTTP API and static web server for Chubby's Three.js virtual world."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import random
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import db
import events
import genesis
import llm
import main as sim_main
import state

DEFAULT_WEB_DIR = Path(__file__).resolve().parent / "web"

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("model/gltf+json", ".gltf")

_APPROVE_RE = re.compile(r"^/api/requests/(\d+)/approve$")
_REJECT_RE = re.compile(r"^/api/requests/(\d+)/reject$")


def _is_ollama_reachable(base_url: str, timeout: float = 0.8) -> bool:
    """Return True if the local Ollama HTTP server responds on /api/tags."""
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= getattr(resp, "status", 200) < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _public_drawings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": r["id"],
            "tick": r["tick"],
            "title": r["title"],
            "svg": r["svg"],
            "location": r.get("location", "Art Studio"),
        }
        for r in rows
    ]


def _parse_action_row(row: dict[str, Any]) -> dict[str, Any]:
    parsed = dict(row)
    try:
        parsed["outcome"] = json.loads(row["outcome_json"])
    except (json.JSONDecodeError, TypeError):
        parsed["outcome"] = {"outcome": str(row.get("outcome_json", "")), "deltas": {}}
    return parsed


class WorldController:
    """Thread-safe controller coordinating Chubby ticks, DB queries, and the auto loop."""

    def __init__(
        self,
        cfg: dict[str, Any],
        db_path: str | Path,
        policy: str = "llm",
        interval: float | None = None,
        seed: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.db_path = Path(db_path)
        self.policy = policy
        self.interval = (
            float(interval)
            if interval is not None
            else float(cfg["clock"]["tick_interval_seconds"])
        )
        self.rng = random.Random(seed)
        self._tick_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._loop_running = False
        self._tick_in_progress = False
        self._effective_policy = self.policy
        self._logged_offline_fallback = False
        self._stop_event = threading.Event()
        self._loop_thread: threading.Thread | None = None
        self._clients: dict[str, Any] = {}

        self.conn = db.get_connection(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        db.init_db(self.conn)
        sim_main.ensure_initial_state(self.conn, self.cfg)

    def close(self) -> None:
        """Stop any active background loop and close the database connection."""
        self.set_loop(running=False)
        self.conn.close()

    def _get_client(self, policy: str) -> Any:
        if policy == "random":
            return None
        if policy not in self._clients:
            self._clients[policy] = sim_main.build_llm_client(
                policy, self.cfg, self.conn, self.rng
            )
        return self._clients[policy]

    def _read_snapshot(self, conn: sqlite3.Connection) -> dict[str, Any]:
        latest = db.get_latest_state(conn)
        if latest is None:
            latest = state.create_initial_state(self.cfg).model_dump()
        tick = int(latest["tick"])

        recent_actions_raw = db.get_recent_actions(conn, 10)
        recent_actions = [_parse_action_row(a) for a in recent_actions_raw]
        latest_action = recent_actions[-1] if recent_actions else None

        with self._state_lock:
            loop_info = {
                "running": self._loop_running,
                "policy": self.policy,
                "effective_policy": self._effective_policy,
                "interval": self.interval,
                "tick_in_progress": self._tick_in_progress,
            }

        return {
            "state": latest,
            "sim_hour": state.get_sim_hour(tick, self.cfg),
            "sim_day": state.get_sim_day(tick, self.cfg),
            "is_night": state.is_night(tick, self.cfg),
            "is_dusk": state.is_dusk(tick, self.cfg),
            "is_raining": events.is_raining(conn, tick, self.cfg),
            "active_events": events.get_active_event_descriptions(conn, tick, self.cfg),
            "latest_action": latest_action,
            "recent_actions": recent_actions,
            "structures": db.get_structures(conn),
            "pending_requests": db.get_requests(conn, status="pending"),
            "all_requests": db.get_requests(conn),
            "unread_messages": db.get_unread_messages(conn),
            "recent_messages": db.get_recent_messages(conn, 10),
            "recent_diary": db.get_recent_diary(conn, 15),
            "memories": db.get_memories(conn, limit=5),
            "drawings": _public_drawings(db.get_recent_drawings(conn, 15)),
            "loop": loop_info,
        }

    def get_world_snapshot(self) -> dict[str, Any]:
        """Return a non-blocking read snapshot of Chubby's world and simulation status."""
        read_conn = db.get_connection(self.db_path)
        try:
            return self._read_snapshot(read_conn)
        finally:
            read_conn.close()

    def step_tick(self, policy: str | None = None) -> dict[str, Any]:
        """Run exactly one simulation tick safely and return the tick result plus snapshot."""
        with self._tick_lock:
            chosen_policy = policy or self.policy
            if chosen_policy not in ("llm", "groq", "mock", "random"):
                raise ValueError(f"Unsupported policy: {chosen_policy}")

            active_policy = chosen_policy
            effective_label = chosen_policy
            if chosen_policy in ("llm", "groq"):
                if llm.is_groq_configured(self.cfg) or chosen_policy == "groq":
                    active_policy = "llm" if chosen_policy == "llm" else "groq"
                    model_name = (
                        os.environ.get("GROQ_MODEL")
                        or self.cfg.get("llm", {}).get("groq_model", "qwen/qwen3.8-27b")
                    )
                    effective_label = f"groq ({model_name})"
                else:
                    base_url = str(self.cfg["llm"]["base_url"])
                    if not _is_ollama_reachable(base_url):
                        active_policy = "mock"
                        effective_label = "mock (ollama offline)"
                        if not self._logged_offline_fallback:
                            print(
                                f"[server] Ollama not reachable at {base_url}; "
                                "falling back to 'mock' policy until Ollama comes online (or add GROQ_API_KEY to .env)."
                            )
                            self._logged_offline_fallback = True
                    else:
                        if self._logged_offline_fallback:
                            print(f"[server] Ollama detected at {base_url}; resuming 'llm' policy.")
                        self._logged_offline_fallback = False

            with self._state_lock:
                self._tick_in_progress = True
                self._effective_policy = effective_label
            try:
                client = self._get_client(active_policy)
                result = sim_main.run_single_tick(
                    self.conn, self.cfg, active_policy, client, self.rng
                )
            finally:
                with self._state_lock:
                    self._tick_in_progress = False
            world = self._read_snapshot(self.conn)
            return {"ok": True, "result": result, "world": world}

    def _loop_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.step_tick()
            except Exception as exc:  # noqa: BLE001
                print(f"[server loop] tick error: {exc}")
            with self._state_lock:
                wait_seconds = max(0.5, float(self.interval))
            if self._stop_event.wait(wait_seconds):
                break

    def set_loop(
        self,
        running: bool | None = None,
        policy: str | None = None,
        interval: float | None = None,
    ) -> dict[str, Any]:
        """Start, stop, or reconfigure the background simulation loop."""
        thread_to_join: threading.Thread | None = None
        with self._state_lock:
            if policy is not None:
                if policy not in ("llm", "groq", "mock", "random"):
                    raise ValueError(f"Unsupported policy: {policy}")
                self.policy = policy
                self._effective_policy = policy
                self._logged_offline_fallback = False
            if interval is not None:
                self.interval = max(0.5, float(interval))

            if running is True and not self._loop_running:
                self._stop_event.clear()
                self._loop_running = True
                self._loop_thread = threading.Thread(
                    target=self._loop_worker, daemon=True
                )
                self._loop_thread.start()
            elif running is False and self._loop_running:
                self._loop_running = False
                self._stop_event.set()
                thread_to_join = self._loop_thread
                self._loop_thread = None

        if thread_to_join is not None and thread_to_join is not threading.current_thread():
            thread_to_join.join(timeout=2.0)

        return {"ok": True, "world": self.get_world_snapshot()}

    def hang_drawing(self, drawing_id: int, location: str) -> dict[str, Any]:
        """Hang or relocate a drawing to a specific wall in town."""
        clean_loc = str(location or "Art Studio").strip()[:64]
        with self._tick_lock:
            db.update_drawing_location(self.conn, int(drawing_id), clean_loc)
        return {"ok": True, "id": drawing_id, "location": clean_loc}

    def leave_message(self, text: str, drawing: str = "") -> dict[str, Any]:
        """Add an unread visitor message (and optional pixel drawing) into Chubby's inbox."""
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Message text cannot be empty.")
        clean_drawing = str(drawing or "").strip()[:2048]
        with self._tick_lock:
            latest = db.get_latest_state(self.conn)
            current_tick = int(latest["tick"]) if latest is not None else 0
            msg_id = db.insert_message(
                self.conn, cleaned, current_tick, drawing=clean_drawing
            )
            db.insert_event(
                self.conn,
                current_tick,
                "visitor_message",
                {
                    "message_id": msg_id,
                    "description": f"A visitor left a message (#{msg_id}).",
                },
            )
            return {
                "ok": True,
                "message_id": msg_id,
                "world": self._read_snapshot(self.conn),
            }

    def approve_request(self, request_id: int, note: str = "Approved") -> dict[str, Any]:
        """Approve a pending structure request and add it to the empty lot."""
        with self._tick_lock:
            decision = genesis.approve_request(
                self.conn, request_id, self.cfg, note=note.strip() or "Approved"
            )
            return {
                "ok": True,
                "decision": decision,
                "world": self._read_snapshot(self.conn),
            }

    def reject_request(
        self, request_id: int, reason: str = "Not feasible right now"
    ) -> dict[str, Any]:
        """Reject a pending structure request with a reason."""
        with self._tick_lock:
            decision = genesis.reject_request(
                self.conn,
                request_id,
                self.cfg,
                reason=reason.strip() or "Not feasible right now",
            )
            return {
                "ok": True,
                "decision": decision,
                "world": self._read_snapshot(self.conn),
            }

    def reset_world(self) -> dict[str, Any]:
        """Reset all tables and return Chubby to tick 0 on an empty scrapyard lot."""
        self.set_loop(running=False)
        with self._tick_lock:
            db.reset_db(self.conn)
            sim_main.ensure_initial_state(self.conn, self.cfg)
            self._clients.clear()
            with self._state_lock:
                self._effective_policy = self.policy
                self._logged_offline_fallback = False
            return {"ok": True, "world": self._read_snapshot(self.conn)}


def make_handler(
    controller: WorldController, web_dir: Path | None = None
) -> type[BaseHTTPRequestHandler]:
    """Create an API request handler bound to `controller` (`web_dir` is ignored)."""
    class ChubbyRequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            # Keep server stdout readable without noisy per-poll logs
            return

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8")
            if not raw.strip():
                return {}
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("JSON body must be an object.")
            return data

        def _serve_static(self, rel_path: str) -> None:
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"ok": False, "error": "API only. Open the Next.js app at http://localhost:3000"},
            )

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/world":
                try:
                    self._send_json(HTTPStatus.OK, controller.get_world_snapshot())
                except Exception as exc:  # noqa: BLE001
                    self._send_json(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {"ok": False, "error": str(exc)},
                    )
                return
            if parsed.path == "/api/drawings":
                try:
                    read_conn = db.get_connection(controller.db_path)
                    try:
                        rows = db.get_recent_drawings(read_conn, 30)
                    finally:
                        read_conn.close()
                    self._send_json(HTTPStatus.OK, {"drawings": _public_drawings(rows)})
                except Exception as exc:  # noqa: BLE001
                    self._send_json(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {"ok": False, "error": str(exc)},
                    )
                return
            self._serve_static(parsed.path)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            try:
                body = self._read_json_body()
                if path == "/api/tick":
                    resp = controller.step_tick(policy=body.get("policy"))
                    self._send_json(HTTPStatus.OK, resp)
                elif path == "/api/loop":
                    resp = controller.set_loop(
                        running=body.get("running"),
                        policy=body.get("policy"),
                        interval=body.get("interval"),
                    )
                    self._send_json(HTTPStatus.OK, resp)
                elif path == "/api/message":
                    resp = controller.leave_message(
                        str(body.get("text", "")),
                        drawing=str(body.get("drawing", "")),
                    )
                    self._send_json(HTTPStatus.OK, resp)
                elif path == "/api/reset":
                    resp = controller.reset_world()
                    self._send_json(HTTPStatus.OK, resp)
                elif path in ("/api/drawings/hang", "/api/drawings/location"):
                    drawing_id = int(body.get("id", 0))
                    location = str(body.get("location", "Art Studio"))
                    resp = controller.hang_drawing(drawing_id, location)
                    self._send_json(HTTPStatus.OK, resp)
                else:
                    m_app = _APPROVE_RE.match(path)
                    m_rej = _REJECT_RE.match(path)
                    if m_app:
                        req_id = int(m_app.group(1))
                        note = str(body.get("note", "Approved"))
                        self._send_json(
                            HTTPStatus.OK, controller.approve_request(req_id, note=note)
                        )
                    elif m_rej:
                        req_id = int(m_rej.group(1))
                        reason = str(body.get("reason", "Not feasible right now"))
                        self._send_json(
                            HTTPStatus.OK, controller.reject_request(req_id, reason=reason)
                        )
                    else:
                        self._send_json(
                            HTTPStatus.NOT_FOUND, {"ok": False, "error": "Unknown API route"}
                        )
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            except json.JSONDecodeError as exc:
                self._send_json(
                    HTTPStatus.BAD_REQUEST, {"ok": False, "error": f"Invalid JSON: {exc}"}
                )
            except Exception as exc:  # noqa: BLE001
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(exc)}
                )

    return ChubbyRequestHandler


def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    db_path: str | Path | None = None,
    config_path: str | Path | None = None,
    policy: str = "llm",
    interval: float | None = None,
    seed: int | None = None,
    web_dir: Path | None = None,
) -> tuple[ThreadingHTTPServer, WorldController]:
    """Instantiate the HTTP server and WorldController."""
    cfg = db.load_config(config_path)
    resolved_db = Path(db_path) if db_path is not None else Path(str(cfg["database"]["path"]))
    controller = WorldController(
        cfg=cfg,
        db_path=resolved_db,
        policy=policy,
        interval=interval,
        seed=seed,
    )
    handler_cls = make_handler(controller, web_dir=web_dir)
    server = ThreadingHTTPServer((host, port), handler_cls)
    return server, controller


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Chubby's 3D Web World Server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port")
    parser.add_argument("--db", default=None, help="SQLite database path")
    parser.add_argument("--config", default=None, help="Path to config.toml")
    parser.add_argument(
        "--policy",
        choices=("llm", "groq", "mock", "random"),
        default="llm",
        help="Default tick policy",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Tick interval seconds for auto loop",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--auto-start",
        action="store_true",
        default=True,
        help="Start the autonomous tick loop immediately on boot (default)",
    )
    parser.add_argument(
        "--no-auto-start",
        action="store_false",
        dest="auto_start",
        help="Do not start the autonomous tick loop automatically on boot",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    server, controller = create_server(
        host=args.host,
        port=args.port,
        db_path=args.db,
        config_path=args.config,
        policy=args.policy,
        interval=args.interval,
        seed=args.seed,
    )
    if args.auto_start:
        controller.set_loop(running=True)

    url = f"http://{args.host}:{server.server_address[1]}"
    print(f"Chubby 3D World running at {url} (policy={args.policy}, db={controller.db_path})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Chubby web server...")
    finally:
        controller.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
