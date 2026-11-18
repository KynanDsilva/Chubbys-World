"""Unit and integration tests for Chubby Iteration 1 + 3D World Server."""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import antirepeat
import cli
import db
import events
import genesis
import llm
import main as sim_main
import memory
import referee
import report
import server
import state


class TestChubbyCore(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = db.load_config(PROJECT_ROOT / "config.toml")
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _new_db(self, name: str = "test.sqlite") -> tuple[Path, Any]:
        db_path = self.temp_path / name
        conn = db.get_connection(db_path)
        db.init_db(conn)
        return db_path, conn

    def _http_opener(self) -> urllib.request.OpenerDirector:
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _http_json(
        self,
        opener: urllib.request.OpenerDirector,
        url: str,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with opener.open(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8")
            return int(resp.status), json.loads(raw)

    def test_decay_skips_battery_on_rest_and_recharge(self) -> None:
        ws = state.create_initial_state(self.cfg)

        decayed_explore = state.apply_passive_decay(ws, "explore", self.cfg)
        self.assertAlmostEqual(
            decayed_explore.battery,
            ws.battery - float(self.cfg["needs"]["decay_battery"]),
            places=6,
        )
        self.assertAlmostEqual(
            decayed_explore.curiosity,
            ws.curiosity - float(self.cfg["needs"]["decay_curiosity"]),
            places=6,
        )

        decayed_rest = state.apply_passive_decay(ws, "rest", self.cfg)
        self.assertAlmostEqual(decayed_rest.battery, ws.battery, places=6)
        self.assertLess(decayed_rest.joy, ws.joy)

        decayed_recharge = state.apply_passive_decay(ws, "recharge", self.cfg)
        self.assertAlmostEqual(decayed_recharge.battery, ws.battery, places=6)
        self.assertLess(decayed_recharge.calm, ws.calm)

    def test_mood_modifier_decay_and_clamping(self) -> None:
        ws = state.create_initial_state(self.cfg).model_copy(
            update={"mood_modifier": 0.25}
        )
        decayed = state.apply_passive_decay(ws, "explore", self.cfg)
        expected_mod = 0.25 * (1.0 - float(self.cfg["needs"]["mood_modifier_decay"]))
        self.assertAlmostEqual(decayed.mood_modifier, expected_mod, places=6)

        over_max = state.apply_deltas(
            ws,
            {"battery": 5.0, "joy": 5.0, "curiosity": -5.0},
            mood_mod_delta=1.0,
            cfg=self.cfg,
        )
        self.assertEqual(over_max.battery, 1.0)
        self.assertEqual(over_max.joy, 1.0)
        self.assertEqual(over_max.curiosity, 0.0)
        self.assertEqual(
            over_max.mood_modifier, float(self.cfg["needs"]["mood_modifier_max"])
        )

        under_min = state.apply_deltas(
            ws,
            {"battery": -5.0},
            mood_mod_delta=-2.0,
            cfg=self.cfg,
        )
        self.assertEqual(under_min.battery, 0.0)
        self.assertEqual(
            under_min.mood_modifier, float(self.cfg["needs"]["mood_modifier_min"])
        )

    def test_referee_caps_and_impossible_outcomes(self) -> None:
        ws = state.create_initial_state(self.cfg)
        rng = random.Random(7)

        impossible_raw = llm.RefereeOutputSchema(
            outcome="Chubby tries to launch into orbit.",
            is_impossible=True,
            is_outdoor=True,
            deltas=llm.RefereeDeltasSchema(
                battery=0.5, curiosity=0.5, joy=0.5, calm=0.5, belonging=0.5
            ),
        )
        res_imp = referee.apply_referee_caps_and_rules(
            impossible_raw,
            "explore",
            "fly to Mars",
            ws,
            structures=[],
            is_raining_now=False,
            cfg=self.cfg,
            rng=rng,
        )
        self.assertTrue(res_imp["is_impossible"])
        self.assertTrue(all(v == 0.0 for v in res_imp["deltas"].values()))

        large_raw = llm.RefereeOutputSchema(
            outcome="Big discovery",
            is_impossible=False,
            is_outdoor=True,
            deltas=llm.RefereeDeltasSchema(
                battery=0.9, curiosity=0.9, joy=0.9, calm=0.9, belonging=0.9
            ),
        )
        max_delta = float(self.cfg["referee"]["max_need_delta"])
        solo_belonging = float(self.cfg["referee"]["solo_belonging_max"])
        noise_max = float(self.cfg["referee"]["noise_max"])

        res_explore = referee.apply_referee_caps_and_rules(
            large_raw,
            "explore",
            "inspect fence",
            ws,
            structures=[],
            is_raining_now=False,
            cfg=self.cfg,
            rng=rng,
        )
        self.assertEqual(res_explore["deltas"]["battery"], 0.0)
        self.assertLessEqual(res_explore["deltas"]["curiosity"], max_delta * noise_max + 1e-6)
        self.assertLessEqual(
            res_explore["deltas"]["belonging"], solo_belonging * noise_max + 1e-6
        )

        res_recharge_boosted = referee.apply_referee_caps_and_rules(
            large_raw,
            "recharge",
            "charge at solar dock",
            ws,
            structures=[{"name": "Recharge Station", "unlocks": "fast solar recharge"}],
            is_raining_now=False,
            cfg=self.cfg,
            rng=rng,
        )
        boosted_cap = float(self.cfg["referee"]["max_battery_recovery_boosted"])
        self.assertGreater(res_recharge_boosted["deltas"]["battery"], 0.0)
        self.assertLessEqual(
            res_recharge_boosted["deltas"]["battery"], boosted_cap * noise_max + 1e-6
        )

        # Night (tick 90 -> 23:00 when start_hour=8, 10 min/tick) + rain outdoor cost
        night_ws = ws.model_copy(update={"tick": 90})
        zero_raw = llm.RefereeOutputSchema(
            outcome="Walked in night rain",
            is_impossible=False,
            is_outdoor=True,
            deltas=llm.RefereeDeltasSchema(),
        )
        res_night_rain = referee.apply_referee_caps_and_rules(
            zero_raw,
            "explore",
            "walk in rain",
            night_ws,
            structures=[],
            is_raining_now=True,
            cfg=self.cfg,
            rng=rng,
        )
        self.assertLess(res_night_rain["deltas"]["battery"], 0.0)

    def test_antirepeat_blocking_penalty_and_similarity(self) -> None:
        allowed = antirepeat.get_allowed_categories(["explore", "play", "make"], block_last_n=2)
        self.assertNotIn("play", allowed)
        self.assertNotIn("make", allowed)
        self.assertIn("explore", allowed)

        sim = antirepeat.jaccard_similarity(
            "inspect the rusty copper coil by the north fence",
            "inspect the copper coil near the north fence again",
        )
        self.assertGreater(sim, 0.5)

        rep = antirepeat.evaluate_antirepeat(
            chosen_category="explore",
            intent="inspect the rusty copper coil by the north fence",
            recent_categories=["explore", "play", "explore", "make"],
            recent_intents=["inspect the rusty copper coil by the north fence"],
            cfg=self.cfg,
        )
        self.assertGreaterEqual(rep["category_occurrences_5"], 2)
        self.assertTrue(rep["similarity_penalty"])
        self.assertLess(rep["mood_mod_delta"], 0.0)

        nov = antirepeat.evaluate_antirepeat(
            chosen_category="reflect",
            intent="listen quietly to crickets under dusk light",
            recent_categories=["explore", "play", "make"],
            recent_intents=["sort bolts"],
            cfg=self.cfg,
        )
        self.assertTrue(nov["novelty_bonus"])
        self.assertGreater(nov["mood_mod_delta"], 0.0)

    def test_dynamic_schemas_and_genesis_quota(self) -> None:
        m_full = llm.build_decision_model(
            allowed_categories=["explore", "play"],
            has_unread_messages=True,
            allow_structure_request=True,
        )
        self.assertIn("read_messages", m_full.model_fields)
        self.assertIn("structure", m_full.model_fields)

        m_min = llm.build_decision_model(
            allowed_categories=["explore", "play"],
            has_unread_messages=False,
            allow_structure_request=False,
        )
        self.assertNotIn("read_messages", m_min.model_fields)
        self.assertNotIn("structure", m_min.model_fields)

        with self.assertRaises(Exception):
            m_min(
                thought="hi",
                intent="rest now",
                category="rest",
            )

        _, conn = self._new_db("quota.sqlite")
        try:
            self.assertTrue(genesis.can_request_structure(conn, 1, self.cfg))
            genesis.record_structure_request(conn, 1, "Shed", "shelter", "need roof", 1)
            genesis.record_structure_request(conn, 2, "Bench", "work", "need tools", 2)
            # Genesis period (<= 5) stays open
            self.assertTrue(genesis.can_request_structure(conn, 3, self.cfg))

            # Post-genesis (tick 10 is in day 1: ticks 1..144 already has 2 requests)
            self.assertFalse(genesis.can_request_structure(conn, 10, self.cfg))

            # Next sim day (tick 145+) allows 1 request again
            self.assertTrue(genesis.can_request_structure(conn, 145, self.cfg))
            genesis.record_structure_request(conn, 145, "Beacon", "signal", "visibility", 3)
            self.assertFalse(genesis.can_request_structure(conn, 146, self.cfg))
        finally:
            conn.close()

    def test_two_step_read_messages_and_genesis_cli_flow(self) -> None:
        db_path, conn = self._new_db("cli_flow.sqlite")
        try:
            sim_main.ensure_initial_state(conn, self.cfg)
            conn.close()

            rc = cli.main(
                ["--db", str(db_path), "--config", str(PROJECT_ROOT / "config.toml"), "leave", "Hello", "little", "robot!"]
            )
            self.assertEqual(rc, 0)

            conn = db.get_connection(db_path)
            self.assertEqual(len(db.get_unread_messages(conn)), 1)

            rng = random.Random(11)
            test_structs = (
                {"name": "Test Shed", "purpose": "shelter", "why": "need roof", "priority": 1},
                {"name": "Test Bench", "purpose": "work", "why": "need tools", "priority": 2},
            )
            client = llm.MockLLMClient(self.cfg, seed=11, structures=test_structs)
            tick1 = sim_main.run_single_tick(conn, self.cfg, "mock", client, rng)
            self.assertEqual(tick1["read_messages_count"], 1)
            self.assertEqual(len(db.get_unread_messages(conn)), 0)
            newly_read_t2 = memory.get_newly_read_messages_for_prompt(conn, 2)
            self.assertEqual(newly_read_t2, ["Hello little robot!"])

            tick2 = sim_main.run_single_tick(conn, self.cfg, "mock", client, rng)
            self.assertEqual(tick2["tick"], 2)

            # Ensure at least two pending requests exist via ticks 1..5
            while len(db.get_requests(conn, status="pending")) < 2:
                sim_main.run_single_tick(conn, self.cfg, "mock", client, rng)
            pending = db.get_requests(conn, status="pending")
            req_approve_id = int(pending[0]["id"])
            req_reject_id = int(pending[1]["id"])
            before_mood_mod = float(db.get_latest_state(conn)["mood_modifier"])
            conn.close()

            self.assertEqual(
                cli.main(
                    ["--db", str(db_path), "--config", str(PROJECT_ROOT / "config.toml"), "requests"]
                ),
                0,
            )
            self.assertEqual(
                cli.main(
                    ["--db", str(db_path), "--config", str(PROJECT_ROOT / "config.toml"), "approve", str(req_approve_id), "Built it"]
                ),
                0,
            )
            self.assertEqual(
                cli.main(
                    ["--db", str(db_path), "--config", str(PROJECT_ROOT / "config.toml"), "reject", str(req_reject_id), "Too big"]
                ),
                0,
            )

            conn = db.get_connection(db_path)
            self.assertEqual(len(db.get_structures(conn)), 2)
            self.assertEqual(db.get_request_by_id(conn, req_approve_id)["status"], "approved")
            self.assertEqual(db.get_request_by_id(conn, req_reject_id)["status"], "rejected")
            after_mood_mod = float(db.get_latest_state(conn)["mood_modifier"])
            self.assertLess(after_mood_mod, before_mood_mod)
        finally:
            conn.close()

    def test_deterministic_replay_with_fixed_seed(self) -> None:
        def _run_sequence(db_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            _, conn = self._new_db(db_name)
            try:
                rng = random.Random(123)
                client = llm.MockLLMClient(self.cfg, seed=123)
                for _ in range(15):
                    sim_main.run_single_tick(conn, self.cfg, "mock", client, rng)
                return db.get_all_states(conn), db.get_all_actions(conn)
            finally:
                conn.close()

        states_a, actions_a = _run_sequence("replay_a.sqlite")
        states_b, actions_b = _run_sequence("replay_b.sqlite")
        self.assertEqual(states_a, states_b)
        self.assertEqual(actions_a, actions_b)

    def test_phase2_random_policy_2000_ticks_bounded_and_unstuck(self) -> None:
        _, conn = self._new_db("phase2_random.sqlite")
        try:
            rng = random.Random(2026)
            for _ in range(2000):
                sim_main.run_single_tick(conn, self.cfg, "random", None, rng)

            states = db.get_all_states(conn)
            self.assertEqual(len(states), 2001)
            for row in states:
                for key in ("battery", "curiosity", "joy", "calm", "belonging", "energy", "mood"):
                    val = float(row[key])
                    self.assertGreaterEqual(val, 0.0)
                    self.assertLessEqual(val, 1.0)
                self.assertGreaterEqual(float(row["mood_modifier"]), -0.30)
                self.assertLessEqual(float(row["mood_modifier"]), 0.30)

            metrics = report.generate_report(conn, self.cfg)
            self.assertEqual(metrics["total_ticks"], 2000)
            self.assertLessEqual(metrics["longest_category_streak"], 2)
            self.assertEqual(metrics["overall_distinct_categories"], 10)
        finally:
            conn.close()

    def test_mock_200_ticks_meets_emergent_metrics(self) -> None:
        _, conn = self._new_db("mock_200.sqlite")
        try:
            rng = random.Random(42)
            client = llm.MockLLMClient(self.cfg, seed=42)
            for _ in range(200):
                sim_main.run_single_tick(conn, self.cfg, "mock", client, rng)

            metrics = report.generate_report(conn, self.cfg)
            targets = metrics["targets_met"]
            self.assertTrue(targets["distinct_categories_ge_6"], metrics)
            self.assertTrue(targets["longest_streak_le_2"], metrics)
            self.assertTrue(targets["diary_similarity_lt_15pct"], metrics)
            self.assertTrue(targets["mood_healthy_ge_90pct"], metrics)
            self.assertTrue(targets["invalid_llm_lt_1pct"], metrics)
            self.assertTrue(targets["genesis_requests_ge_3"], metrics)

            memories = db.get_memories(conn)
            self.assertEqual(len(memories), 4)
        finally:
            conn.close()

    def test_web_server_world_and_api_endpoints(self) -> None:
        db_path = self.temp_path / "web_world.sqlite"
        httpd, controller = server.create_server(
            host="127.0.0.1",
            port=0,
            db_path=db_path,
            config_path=PROJECT_ROOT / "config.toml",
            policy="mock",
            interval=30.0,
            seed=99,
        )
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            port = httpd.server_address[1]
            base = f"http://127.0.0.1:{port}"
            opener = self._http_opener()

            code, world0 = self._http_json(opener, f"{base}/api/world")
            self.assertEqual(code, 200)
            self.assertEqual(world0["state"]["tick"], 0)
            self.assertEqual(len(world0["structures"]), 1)
            self.assertEqual(
                [s["name"] for s in world0["structures"]],
                ["Art Studio"],
            )
            self.assertEqual(world0["drawings"], [])
            self.assertFalse(world0["is_raining"])

            code, msg_resp = self._http_json(
                opener,
                f"{base}/api/message",
                method="POST",
                payload={"text": "Welcome to your empty fenced lot, Chubby!"},
            )
            self.assertEqual(code, 200)
            self.assertEqual(len(msg_resp["world"]["unread_messages"]), 1)

            # Insert two test requests directly
            req1 = genesis.record_structure_request(controller.conn, 1, "Test Lookout", "lookout", "why", 1)
            req2 = genesis.record_structure_request(controller.conn, 2, "Test Pavilion", "shelter", "why", 2)
            pending_reqs = [{"id": req1}, {"id": req2}]

            approve_id = int(pending_reqs[0]["id"])
            reject_id = int(pending_reqs[1]["id"])

            code, app_resp = self._http_json(
                opener,
                f"{base}/api/requests/{approve_id}/approve",
                method="POST",
                payload={"note": "Looks useful"},
            )
            self.assertEqual(code, 200)
            self.assertEqual(len(app_resp["world"]["structures"]), 2)

            code, rej_resp = self._http_json(
                opener,
                f"{base}/api/requests/{reject_id}/reject",
                method="POST",
                payload={"reason": "Wait for later"},
            )
            self.assertEqual(code, 200)
            self.assertEqual(len(rej_resp["world"]["structures"]), 2)

            code, reset_resp = self._http_json(
                opener, f"{base}/api/reset", method="POST", payload={}
            )
            self.assertEqual(code, 200)
            self.assertEqual(reset_resp["world"]["state"]["tick"], 0)
            self.assertEqual(len(reset_resp["world"]["structures"]), 1)
        finally:
            httpd.shutdown()
            controller.close()
            httpd.server_close()
            thread.join(timeout=2.0)


class _DrawClient(llm.MockLLMClient):
    """Mock client that always decides to draw with a fixed intent."""

    def __init__(self, cfg: dict[str, Any], intent: str) -> None:
        super().__init__(cfg, seed=7)
        self.fixed_intent = intent

    def decide(self, messages, decision_model, allowed_categories, tick, context=None):
        payload = {
            "thought": "I want to make art.",
            "intent": self.fixed_intent,
            "category": "draw",
            "diary_entry": None,
        }
        if "read_messages" in decision_model.model_fields:
            payload["read_messages"] = False
        return decision_model.model_validate(payload), False


class TestDrawing(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = db.load_config(PROJECT_ROOT / "config.toml")
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_sanitize_svg_strips_dangerous_content(self) -> None:
        from svg_utils import sanitize_svg

        dirty = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10" onload="alert(1)">'
            "<script>alert(1)</script>"
            '<a href="http://evil"><circle cx="5" cy="5" r="3" fill="red" onclick="x()" '
            'style="fill:url(#a)"/></a>'
            '<rect x="1" y="1" width="5" height="5" fill="url(#g)" href="javascript:1"/>'
            '<foreignObject><div>hi</div></foreignObject></svg>'
        )
        out = sanitize_svg(dirty)
        self.assertIsNotNone(out)
        for bad in ("script", "onload", "onclick", "href", "url(", "style", "foreignObject", "<a"):
            self.assertNotIn(bad, out)
        self.assertIn('viewBox="0 0 100 100"', out)
        self.assertIn("xmlns=", out)
        self.assertIsNone(sanitize_svg("<script>alert(1)</script>"))
        self.assertIsNone(sanitize_svg("not svg"))
        self.assertIsNone(sanitize_svg('<svg><script>x</script></svg>'))

    def test_sanitize_svg_accepts_good_svg(self) -> None:
        from svg_utils import sanitize_svg

        good = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
            '<rect x="0" y="0" width="100" height="100" fill="#87ceeb"/>'
            '<g><circle cx="50" cy="50" r="20" fill="gold" stroke="orange" stroke-width="2"/>'
            '<path d="M 10 80 Q 50 40 90 80" fill="none" stroke="#333"/></g></svg>'
        )
        out = sanitize_svg(good)
        self.assertIsNotNone(out)
        self.assertIn("<circle", out)
        self.assertIn("<path", out)

    def test_procedural_drawings_vary_and_are_valid(self) -> None:
        from svg_utils import sanitize_svg

        titles_svgs = {llm.safe_procedural_drawing(i)["svg"] for i in range(30)}
        self.assertGreater(len(titles_svgs), 10)
        for svg in titles_svgs:
            self.assertEqual(sanitize_svg(svg), svg)

    def test_draw_downgraded_when_not_at_studio(self) -> None:
        db_path = self.temp_path / "d.sqlite"
        conn = db.get_connection(db_path)
        db.init_db(conn)
        rng = random.Random(1)
        client = _DrawClient(self.cfg, "Draw something in the open gravel yard.")
        res = sim_main.run_single_tick(conn, self.cfg, "mock", client, rng)
        self.assertNotEqual(res["category"], "draw")
        self.assertEqual(db.get_recent_drawings(conn, 5), [])
        self.assertIn("Art Studio", res["outcome"])
        conn.close()

    def test_draw_at_studio_stores_drawing_and_shows_in_api(self) -> None:
        db_path = self.temp_path / "d2.sqlite"
        httpd, controller = server.create_server(
            host="127.0.0.1",
            port=0,
            db_path=db_path,
            config_path=PROJECT_ROOT / "config.toml",
            policy="mock",
            interval=30.0,
            seed=5,
        )
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            client = _DrawClient(
                self.cfg, "Roll over to and stand at my Art Studio to sketch a robot."
            )
            res = sim_main.run_single_tick(
                controller.conn, self.cfg, "mock", client, random.Random(3)
            )
            self.assertEqual(res["category"], "draw")
            self.assertIsNotNone(res["drawing"])
            self.assertIn(res["drawing"]["title"], res["outcome"])
            base = f"http://127.0.0.1:{httpd.server_address[1]}"
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(f"{base}/api/world", timeout=10) as r:
                world = json.loads(r.read().decode("utf-8"))
            self.assertEqual(len(world["drawings"]), 1)
            d = world["drawings"][0]
            self.assertEqual(set(d), {"id", "tick", "title", "svg", "location"})
            self.assertTrue(d["svg"].startswith("<svg"))
            with opener.open(f"{base}/api/drawings", timeout=10) as r:
                listing = json.loads(r.read().decode("utf-8"))
            self.assertEqual(len(listing["drawings"]), 1)
        finally:
            httpd.shutdown()
            controller.close()
            httpd.server_close()
            thread.join(timeout=2.0)

    def test_chubby_draws_on_town_walls_and_relocates_art(self) -> None:
        db_path = self.temp_path / "d_walls.sqlite"
        httpd, controller = server.create_server(
            host="127.0.0.1",
            port=0,
            db_path=db_path,
            config_path=PROJECT_ROOT / "config.toml",
            policy="mock",
            interval=30.0,
            seed=7,
        )
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            client = _DrawClient(
                self.cfg, "Roll over to the Bakery wall to sketch a fresh morning sun."
            )
            res = sim_main.run_single_tick(
                controller.conn, self.cfg, "mock", client, random.Random(7)
            )
            self.assertEqual(res["category"], "draw")
            self.assertIn("Bakery wall", res["outcome"])
            self.assertEqual(res["drawing"]["location"], "Bakery")

            base = f"http://127.0.0.1:{httpd.server_address[1]}"
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            req = urllib.request.Request(
                f"{base}/api/drawings/hang",
                data=json.dumps({"id": res["drawing"]["id"], "location": "Library"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with opener.open(req, timeout=10) as r:
                resp = json.loads(r.read().decode("utf-8"))
            self.assertTrue(resp["ok"])
            self.assertEqual(resp["location"], "Library")

            with opener.open(f"{base}/api/drawings", timeout=10) as r:
                listing = json.loads(r.read().decode("utf-8"))
            self.assertEqual(listing["drawings"][0]["location"], "Library")
        finally:
            httpd.shutdown()
            controller.close()
            httpd.server_close()
            thread.join(timeout=2.0)

    def test_town_revamp_migration_is_one_time(self) -> None:
        db_path = self.temp_path / "old.sqlite"
        conn = db.get_connection(db_path)
        db.init_db(conn)
        conn.execute("DELETE FROM meta")
        db.insert_structure(conn, "Scrap Workbench", "tinker", 12)
        db.insert_request(conn, 3, "Thing", "p", "w", 1)
        conn.commit()
        db.init_db(conn)
        self.assertEqual(db.get_structures(conn), [])
        self.assertEqual(db.get_requests(conn), [])
        self.assertTrue((self.temp_path / "old.sqlite.bak").exists())
        db.insert_structure(conn, "Later Thing", "x", 20)
        db.init_db(conn)
        self.assertEqual(len(db.get_structures(conn)), 1)
        conn.close()

    def test_groq_client_configuration_and_mock_response(self) -> None:
        cfg = dict(self.cfg)
        cfg["llm"] = dict(self.cfg.get("llm", {}))

        cfg["llm"]["groq_api_key"] = "your_groq_api_key_here"
        old_env = os.environ.get("GROQ_API_KEY")
        try:
            if "GROQ_API_KEY" in os.environ:
                del os.environ["GROQ_API_KEY"]
            self.assertFalse(llm.is_groq_configured(cfg))
            client = llm.GroqClient(cfg)
            with self.assertRaises(ValueError):
                client._post_chat([], {})

            cfg["llm"]["groq_api_key"] = "gsk_test_key_12345"
            self.assertTrue(llm.is_groq_configured(cfg))
            client = llm.GroqClient(cfg)

            class _FakeResp:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

                def read(self):
                    return json.dumps({
                        "choices": [{
                            "message": {
                                "content": json.dumps({
                                    "thought": "Thinking fast on Groq!",
                                    "intent": "Wander over to inspect the cobblestones.",
                                    "category": "explore",
                                    "diary_entry": None,
                                })
                            }
                        }]
                    }).encode("utf-8")

            class _FakeOpener:
                def open(self, req, timeout=30):
                    return _FakeResp()

            client._opener = _FakeOpener()
            m = llm.build_decision_model(["explore"], False, False)
            dec, fb = client.decide([], m, ["explore"], tick=1)
            self.assertFalse(fb)
            self.assertEqual(dec.category, "explore")
            self.assertIn("Groq", dec.thought)
        finally:
            if old_env is not None:
                os.environ["GROQ_API_KEY"] = old_env


if __name__ == "__main__":    unittest.main()

