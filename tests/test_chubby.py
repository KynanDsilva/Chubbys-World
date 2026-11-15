"""Unit, schema, replay, and milestone verification tests for Chubby Iteration 1."""

from __future__ import annotations

import random
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import antirepeat
import cli
import db
import genesis
import llm
import main
import memory
import prompt
import referee
import report
import state


class TestChubbyCore(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = db.load_config(ROOT_DIR / "config.toml")
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "test_chubby.sqlite"
        self.conn = db.get_connection(self.db_path)
        db.init_db(self.conn)

    def tearDown(self) -> None:
        self.conn.close()
        self.tmp_dir.cleanup()

    def test_decay_skips_battery_on_rest_and_recharge(self) -> None:
        ws = state.create_initial_state(self.cfg)
        after_rest = state.apply_passive_decay(ws, "rest", self.cfg)
        after_recharge = state.apply_passive_decay(ws, "recharge", self.cfg)
        after_explore = state.apply_passive_decay(ws, "explore", self.cfg)

        self.assertAlmostEqual(after_rest.battery, ws.battery, places=5)
        self.assertAlmostEqual(after_recharge.battery, ws.battery, places=5)
        self.assertAlmostEqual(
            after_explore.battery,
            ws.battery - float(self.cfg["needs"]["decay_battery"]),
            places=5,
        )
        # Other needs always decay
        self.assertLess(after_rest.curiosity, ws.curiosity)

    def test_mood_modifier_decay_and_clamping(self) -> None:
        ws = state.create_initial_state(self.cfg).model_copy(
            update={"mood_modifier": 0.25}
        )
        decayed = state.apply_passive_decay(ws, "play", self.cfg)
        self.assertAlmostEqual(decayed.mood_modifier, 0.225, places=5)

        boosted = state.apply_deltas(
            decayed,
            {"battery": 5.0, "joy": -5.0},
            mood_mod_delta=1.0,
            cfg=self.cfg,
        )
        self.assertEqual(boosted.battery, 1.0)
        self.assertEqual(boosted.joy, 0.0)
        self.assertEqual(boosted.mood_modifier, 0.3)
        self.assertGreaterEqual(boosted.mood, 0.0)
        self.assertLessEqual(boosted.mood, 1.0)

    def test_referee_caps_and_impossible_outcomes(self) -> None:
        ws = state.create_initial_state(self.cfg).model_copy(update={"tick": 1})
        rng = random.Random(10)

        # Impossible claim zeroes all deltas and replaces with neutral outcome
        impossible_raw = llm.RefereeOutputSchema(
            outcome="Chubby meets a talking dragon.",
            is_impossible=True,
            is_outdoor=True,
            deltas=llm.RefereeDeltasSchema(
                battery=0.1, curiosity=0.15, joy=0.15, calm=0.1, belonging=0.15
            ),
        )
        impossible_res = referee.apply_referee_caps_and_rules(
            impossible_raw, "social", "Talk to dragon", ws, [], False, self.cfg, rng
        )
        self.assertTrue(impossible_res["is_impossible"])
        self.assertEqual(sum(abs(v) for v in impossible_res["deltas"].values()), 0.0)

        # Battery cannot rise during 'play' without recharge intent
        play_raw = llm.RefereeOutputSchema(
            outcome="Chubby spins around happily.",
            is_impossible=False,
            is_outdoor=True,
            deltas=llm.RefereeDeltasSchema(battery=0.12, joy=0.25, belonging=0.25),
        )
        play_res = referee.apply_referee_caps_and_rules(
            play_raw, "play", "Spin around the gravel yard", ws, [], False, self.cfg, rng
        )
        self.assertEqual(play_res["deltas"]["battery"], 0.0)
        self.assertLessEqual(play_res["deltas"]["joy"], 0.15 * 1.3 + 1e-6)
        self.assertLessEqual(play_res["deltas"]["belonging"], 0.10 * 1.3 + 1e-6)

        # Elevated recovery cap (+0.22 before noise) when approved charging structure exists
        charger = [{"id": 1, "name": "Solar Charging Pad", "unlocks": "fast charging", "created_tick": 1}]
        recharge_raw = llm.RefereeOutputSchema(
            outcome="Chubby plugs into the Solar Charging Pad.",
            is_impossible=False,
            is_outdoor=True,
            deltas=llm.RefereeDeltasSchema(battery=0.30),
        )
        recharge_res = referee.apply_referee_caps_and_rules(
            recharge_raw,
            "recharge",
            "Plug into Solar Charging Pad",
            ws,
            charger,
            False,
            self.cfg,
            random.Random(1),
        )
        self.assertGreaterEqual(recharge_res["deltas"]["battery"], 0.22 * 0.7 - 1e-6)
        self.assertLessEqual(recharge_res["deltas"]["battery"], 0.22 * 1.3 + 1e-6)

    def test_antirepeat_blocking_penalty_and_similarity(self) -> None:
        allowed = antirepeat.get_allowed_categories(["explore", "make", "play"], block_last_n=2)
        self.assertNotIn("make", allowed)
        self.assertNotIn("play", allowed)
        self.assertIn("explore", allowed)

        self.assertGreater(
            antirepeat.jaccard_similarity(
                "Roll along the east fence looking for gears",
                "Roll along the east fence searching for gears",
            ),
            0.6,
        )
        self.assertEqual(antirepeat.jaccard_similarity("", "anything"), 0.0)

        # Repeat penalty (-0.07 * 2) + similarity penalty (-0.07)
        res_repeat = antirepeat.evaluate_antirepeat(
            "explore",
            "Roll along the east fence looking for shiny gears",
            ["explore", "play", "explore", "make", "rest"],
            ["Roll along the east fence looking for shiny gears"],
            self.cfg,
        )
        self.assertAlmostEqual(res_repeat["mood_mod_delta"], -0.21, places=5)

        # Novelty bonus (+0.01) when absent from last 10 ticks
        res_novel = antirepeat.evaluate_antirepeat(
            "reflect",
            "Watch clouds drift silently over the roof",
            ["explore", "play", "make"],
            ["Stack pebbles in a tidy tower"],
            self.cfg,
        )
        self.assertAlmostEqual(res_novel["mood_mod_delta"], 0.01, places=5)

    def test_dynamic_schemas_and_genesis_quota(self) -> None:
        allowed = ["rest", "explore", "play"]
        model_plain = llm.build_decision_model(
            allowed, has_unread_messages=False, allow_structure_request=False
        )
        schema_plain = model_plain.model_json_schema()
        self.assertEqual(schema_plain["properties"]["category"]["enum"], allowed)
        self.assertNotIn("read_messages", schema_plain["properties"])
        self.assertNotIn("structure", schema_plain["properties"])

        model_full = llm.build_decision_model(
            allowed, has_unread_messages=True, allow_structure_request=True
        )
        schema_full = model_full.model_json_schema()
        self.assertIn("read_messages", schema_full["properties"])
        self.assertIn("structure", schema_full["properties"])

        # Genesis ticks 1..5 always allow structure requests
        for t in range(1, 6):
            self.assertTrue(genesis.can_request_structure(self.conn, t, self.cfg))
        genesis.record_structure_request(
            self.conn, tick=2, name="Workbench", purpose="Tinkering", why="Need tools", priority=1
        )
        # Same simulated day (tick 6..144) is now capped because Day 1 already has a request
        self.assertFalse(genesis.can_request_structure(self.conn, 6, self.cfg))
        # Day 2 (tick 145) opens quota again until 1 request is made
        self.assertTrue(genesis.can_request_structure(self.conn, 145, self.cfg))
        genesis.record_structure_request(
            self.conn, tick=145, name="Music Corner", purpose="Play chimes", why="Joy", priority=2
        )
        self.assertFalse(genesis.can_request_structure(self.conn, 146, self.cfg))

    def test_two_step_read_messages_and_genesis_cli_flow(self) -> None:
        main.ensure_initial_state(self.conn, self.cfg)
        mock_client = llm.MockLLMClient(self.cfg, seed=11)
        rng = random.Random(11)

        # Leave a visitor message via CLI before Tick 1
        rc = cli.main(["--db", str(self.db_path), "leave", "Hi Chubby, keep shining!"])
        self.assertEqual(rc, 0)
        self.assertEqual(len(db.get_unread_messages(self.conn)), 1)

        # Tick 1: Chubby sees unread count and sets read_messages=True
        tick1 = main.run_single_tick(self.conn, self.cfg, "mock", mock_client, rng)
        self.assertEqual(tick1["read_messages_count"], 1)
        self.assertEqual(len(db.get_unread_messages(self.conn)), 0)
        self.assertIsNotNone(tick1["request_id"])

        # Tick 2: newly read message is injected into prompt and Chubby reacts in diary
        tick2 = main.run_single_tick(self.conn, self.cfg, "mock", mock_client, rng)
        diary_rows = db.get_all_diary(self.conn)
        self.assertTrue(
            any("Hi Chubby, keep shining!" in row["text"] for row in diary_rows),
            f"Expected reaction to message in diary, got: {diary_rows}",
        )
        self.assertEqual(tick2["tick"], 2)

        # Approve request #1 and reject request #2 via CLI
        rc_app = cli.main(
            ["--db", str(self.db_path), "approve", str(tick1["request_id"]), "Great idea"]
        )
        self.assertEqual(rc_app, 0)
        self.assertEqual(len(db.get_structures(self.conn)), 1)

        if tick2["request_id"] is not None:
            rc_rej = cli.main(
                ["--db", str(self.db_path), "reject", str(tick2["request_id"]), "Too bulky"]
            )
            self.assertEqual(rc_rej, 0)
            req2 = db.get_request_by_id(self.conn, tick2["request_id"])
            self.assertEqual(req2["status"], "rejected")

        # Tick 3 sees the approved/rejected events in active event context
        tick3 = main.run_single_tick(self.conn, self.cfg, "mock", mock_client, rng)
        self.assertEqual(tick3["tick"], 3)

    def test_deterministic_replay_with_fixed_seed(self) -> None:
        def run_sequence(db_file: Path) -> tuple[list[dict], list[dict]]:
            c = db.get_connection(db_file)
            try:
                db.init_db(c)
                r = random.Random(12345)
                client = llm.MockLLMClient(self.cfg, seed=12345)
                for _ in range(30):
                    main.run_single_tick(c, self.cfg, "mock", client, r)
                return db.get_all_actions(c), db.get_all_states(c)
            finally:
                c.close()

        db_a = Path(self.tmp_dir.name) / "replay_a.sqlite"
        db_b = Path(self.tmp_dir.name) / "replay_b.sqlite"
        actions_a, states_a = run_sequence(db_a)
        actions_b, states_b = run_sequence(db_b)

        self.assertEqual(actions_a, actions_b)
        self.assertEqual(states_a, states_b)

    def test_phase2_random_policy_2000_ticks_bounded_and_unstuck(self) -> None:
        rng = random.Random(77)
        for _ in range(2000):
            main.run_single_tick(self.conn, self.cfg, "random", None, rng)

        states = db.get_all_states(self.conn)
        self.assertEqual(len(states), 2001)  # tick 0..2000

        # Every state remains bounded in [0.0, 1.0]
        for s in states:
            for field in (*state.NEED_NAMES, "energy", "mood"):
                self.assertGreaterEqual(s[field], 0.0)
                self.assertLessEqual(s[field], 1.0)

        # In the final 500 ticks, no need is permanently stuck at 0.0 or 1.0
        tail = states[-500:]
        for field in state.NEED_NAMES:
            values = [float(s[field]) for s in tail]
            has_interior = any(0.05 < v < 0.95 for v in values)
            self.assertTrue(
                has_interior,
                f"Need '{field}' got stuck near boundary in tail: min={min(values)}, max={max(values)}",
            )

    def test_mock_200_ticks_meets_emergent_metrics(self) -> None:
        rng = random.Random(99)
        mock_client = llm.MockLLMClient(self.cfg, seed=99)
        for _ in range(200):
            main.run_single_tick(self.conn, self.cfg, "mock", mock_client, rng)

        metrics = report.generate_report(self.conn, self.cfg)
        self.assertLessEqual(metrics["longest_category_streak"], 2)
        self.assertGreaterEqual(metrics["min_distinct_in_100_window"], 6)
        self.assertGreaterEqual(metrics["mood_in_range_pct"], 90.0)
        self.assertLess(metrics["diary_high_sim_pct"], 15.0)
        self.assertGreaterEqual(metrics["genesis_request_count"], 3)
        self.assertEqual(len(db.get_memories(self.conn)), 4)  # ticks 50, 100, 150, 200
        self.assertTrue(all(metrics["targets_met"].values()), f"Targets failed: {metrics}")


if __name__ == "__main__":
    unittest.main()

