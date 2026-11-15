"""Ollama HTTP client, dynamic Pydantic JSON schemas, retries, and deterministic mock policy."""

from __future__ import annotations

import json
import random
import sqlite3
import urllib.error
import urllib.request
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, create_model

import db


class StructureRequestSchema(BaseModel):
    """Schema for an infrastructure request proposed by Chubby."""

    name: str
    purpose: str
    why: str
    priority: int = Field(ge=1, le=3)


class RefereeDeltasSchema(BaseModel):
    """Proposed per-need deltas returned by the Referee LLM call."""

    battery: float = 0.0
    curiosity: float = 0.0
    joy: float = 0.0
    calm: float = 0.0
    belonging: float = 0.0


class RefereeOutputSchema(BaseModel):
    """Structured output schema for the Referee LLM call."""

    outcome: str
    is_impossible: bool = False
    is_outdoor: bool = True
    deltas: RefereeDeltasSchema


class MemorySummarySchema(BaseModel):
    """Structured output schema for the 50-tick memory summarization job."""

    long_term_memory: str
    self_discovery: str


def build_decision_model(
    allowed_categories: list[str],
    has_unread_messages: bool,
    allow_structure_request: bool,
) -> type[BaseModel]:
    """Dynamically build Chubby's decision Pydantic model for the current tick."""
    if not allowed_categories:
        raise ValueError("allowed_categories must contain at least one category")

    category_literal = Literal[tuple(allowed_categories)]  # type: ignore[valid-type]
    fields: dict[str, Any] = {
        "thought": (str, Field(description="One or two sentences in Chubby's voice")),
        "intent": (
            str,
            Field(description="What he wants to do, in his own words, and why"),
        ),
        "category": (
            category_literal,
            Field(description="Coarse category chosen from allowed categories"),
        ),
        "diary_entry": (
            Optional[str],
            Field(
                default=None,
                description="Optional diary entry, only when something meaningful happened",
            ),
        ),
    }

    if has_unread_messages:
        fields["read_messages"] = (
            bool,
            Field(
                default=False,
                description="Set true to open and read unread visitor messages",
            ),
        )

    if allow_structure_request:
        fields["structure"] = (
            Optional[StructureRequestSchema],
            Field(
                default=None,
                description="Optional infrastructure request when Chubby wants something built",
            ),
        )

    return create_model("ChubbyDecisionSchema", **fields)


def _extract_json_text(raw_content: str) -> str:
    text = raw_content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def build_fallback_decision(
    decision_model: type[BaseModel],
    allowed_categories: list[str],
) -> BaseModel:
    """Construct a safe fallback decision when all LLM retries fail."""
    category = "rest" if "rest" in allowed_categories else allowed_categories[0]
    payload: dict[str, Any] = {
        "thought": "My circuits feel a little tangled, so I should take a quiet pause.",
        "intent": "Rest nearby in a quiet corner to settle my sensors and conserve energy.",
        "category": category,
        "diary_entry": None,
    }
    if "read_messages" in decision_model.model_fields:
        payload["read_messages"] = False
    if "structure" in decision_model.model_fields:
        payload["structure"] = None
    return decision_model.model_validate(payload)


def build_fallback_referee(category: str) -> RefereeOutputSchema:
    """Construct a safe fallback referee outcome if the referee LLM call fails."""
    deltas = RefereeDeltasSchema()
    if category in ("rest", "recharge"):
        deltas = RefereeDeltasSchema(battery=0.12, calm=0.04)
    elif category == "social":
        deltas = RefereeDeltasSchema(belonging=0.05, joy=0.03)
    else:
        deltas = RefereeDeltasSchema(curiosity=0.03, joy=0.02, battery=-0.02)
    return RefereeOutputSchema(
        outcome="Chubby carries out his plan quietly in the scrapyard lot.",
        is_impossible=False,
        is_outdoor=category not in ("rest", "reflect"),
        deltas=deltas,
    )


class OllamaClient:
    """Local Ollama chat client with schema-constrained output, retries, and failure logging."""

    def __init__(
        self, cfg: dict[str, Any], conn: sqlite3.Connection | None = None
    ) -> None:
        self.cfg = cfg
        self.conn = conn
        llm_cfg = cfg["llm"]
        self.base_url = str(llm_cfg["base_url"]).rstrip("/")
        self.model = str(llm_cfg["model"])
        self.temperature = float(llm_cfg["temperature"])
        self.num_ctx = int(llm_cfg["num_ctx"])
        self.think = bool(llm_cfg["think"])
        self.keep_alive = str(llm_cfg["keep_alive"])
        self.max_retries = int(llm_cfg["max_retries"])
        self.timeout_seconds = float(llm_cfg["timeout_seconds"])
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _post_chat(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        temperature: float | None = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": schema,
            "think": self.think,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_ctx": self.num_ctx,
            },
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self._opener.open(req, timeout=self.timeout_seconds) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return str(body["message"]["content"])

    def _log_failure(self, tick: int, stage: str, attempt: int, error: str) -> None:
        if self.conn is None:
            return
        db.insert_event(
            self.conn,
            tick,
            "llm_failure",
            {"stage": stage, "attempt": attempt, "error": error},
        )

    def _call_with_retries(
        self,
        messages: list[dict[str, str]],
        response_model: type[BaseModel],
        tick: int,
        stage: str,
        temperature: float | None = None,
    ) -> BaseModel | None:
        schema = response_model.model_json_schema()
        attempts = 1 + max(0, self.max_retries)
        for attempt in range(1, attempts + 1):
            try:
                raw = self._post_chat(messages, schema, temperature=temperature)
                cleaned = _extract_json_text(raw)
                return response_model.model_validate_json(cleaned)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
                self._log_failure(tick, stage, attempt, str(exc))
        return None

    def decide(
        self,
        messages: list[dict[str, str]],
        decision_model: type[BaseModel],
        allowed_categories: list[str],
        tick: int,
        context: dict[str, Any] | None = None,
    ) -> tuple[BaseModel, bool]:
        """Request Chubby's decision from Ollama, falling back safely if retries fail."""
        result = self._call_with_retries(
            messages, decision_model, tick=tick, stage="decision"
        )
        if result is not None:
            return result, False
        if self.conn is not None:
            db.insert_event(
                self.conn, tick, "llm_fallback", {"stage": "decision"}
            )
        return build_fallback_decision(decision_model, allowed_categories), True

    def referee(
        self,
        messages: list[dict[str, str]],
        category: str,
        tick: int,
        context: dict[str, Any] | None = None,
    ) -> tuple[RefereeOutputSchema, bool]:
        """Request the Referee resolution from Ollama, with fallback on failure."""
        result = self._call_with_retries(
            messages,
            RefereeOutputSchema,
            tick=tick,
            stage="referee",
            temperature=0.4,
        )
        if isinstance(result, RefereeOutputSchema):
            return result, False
        if self.conn is not None:
            db.insert_event(
                self.conn, tick, "llm_fallback", {"stage": "referee"}
            )
        return build_fallback_referee(category), True

    def summarize(
        self,
        messages: list[dict[str, str]],
        tick: int,
        context: dict[str, Any] | None = None,
    ) -> MemorySummarySchema | None:
        """Summarize older actions and diary entries into long-term memory."""
        result = self._call_with_retries(
            messages,
            MemorySummarySchema,
            tick=tick,
            stage="summarize",
            temperature=0.5,
        )
        return result if isinstance(result, MemorySummarySchema) else None


class MockLLMClient:
    """Deterministic, seedable local policy validating outputs through the real schemas."""

    _GENESIS_STRUCTURES: tuple[dict[str, Any], ...] = (
        {
            "name": "Solar Charging Pad",
            "purpose": "Recharge battery reliably under daylight or cloudy skies",
            "why": "My solar trickle alone feels slow when my power dips",
            "priority": 1,
        },
        {
            "name": "Scrap Workbench",
            "purpose": "Sort bolts, repair joints, and tinker with salvaged parts",
            "why": "I found interesting springs in the yard and want a table for making things",
            "priority": 1,
        },
        {
            "name": "Rain Shelter Canopy",
            "purpose": "Keep sensors dry while watching storms pass over the lot",
            "why": "Staying outside in rain drains my battery faster",
            "priority": 2,
        },
        {
            "name": "Tiny Radio Corner",
            "purpose": "Listen to faint scrapyard broadcasts and leave notes for visitors",
            "why": "Hearing static or distant signals makes the yard feel less lonely",
            "priority": 2,
        },
    )

    _INTENT_TEMPLATES: dict[str, tuple[str, ...]] = {
        "rest": (
            "Curl up by the warm transformer casing and power down my drive motors for a spell.",
            "Park under the quiet scrap awning and let my cooling fan idle gently.",
            "Settle onto a flat concrete slab to recalibrate my balance gyroscopes.",
            "Dim my optic lamps and listen to the yard settle while resting my servos.",
        ),
        "recharge": (
            "Angle my back panel toward the brightest patch of sky to top up my cells.",
            "Connect my copper lead to the cleanest power contact and sip steady voltage.",
            "Trace the sunlit ridge near the fence to gather extra solar charge.",
            "Run a careful low-draw trickle charge while checking my voltage gauge.",
        ),
        "explore": (
            "Roll along the east fence to catalog rusted gears and unfamiliar weeds.",
            "Inspect the stacked shipping crates to see what shadows hide between them.",
            "Follow a trail of shiny washers past the gravel mound toward the gate.",
            "Scan the far corner of the scrapyard for interesting magnets or glass marbles.",
        ),
        "make": (
            "Assemble a tiny wind chime out of aluminum pull-tabs and copper wire.",
            "Shape a neat stand for my favorite found bolts using bent steel strips.",
            "Build a little pebble cairn beside my charging spot to mark my home.",
            "Twist discarded spring coils into a bouncy desk companion.",
        ),
        "play": (
            "Balance three smooth river stones on my dome and see how long they stay.",
            "Chase reflections cast by a polished hubcap across the dirt yard.",
            "Tap rhythmic Morse patterns on hollow oil drums to hear the echoes.",
            "Spin in slow circles to test my wheel tracks in the gravel.",
        ),
        "social": (
            "Leave a cheerful chalk arrow near the mailbox for anyone passing by.",
            "Tune a scavenged antenna toward the town road and wave at passing lights.",
            "Arrange shiny bottle caps into a welcoming smile by the front gate.",
            "Check my message slot and tidy the welcome pebble beside the path.",
        ),
        "reflect": (
            "Watch clouds drift overhead and think about what changed in my yard today.",
            "Review my recent footsteps in the dirt and ponder what I enjoy most.",
            "Count the stars or distant town lamps while humming softly to myself.",
            "Sit still and notice how my chassis feels after a busy stretch.",
        ),
        "maintain": (
            "Brush grit out of my left wheel bearing with a stiff paintbrush bristle.",
            "Wipe dust from my optical lens and tighten a slightly loose chest bolt.",
            "Oil my squeaky elbow hinge with a single drop of salvaged machine oil.",
            "Sweep stray gravel away from my favorite parking spot.",
        ),
        "other": (
            "Catalog the different colors of moss growing along the north wall.",
            "Compare the sound of wind whistling through pipes of different lengths.",
            "Practice drawing tiny robot footprints in neat rows across the sand.",
            "Sketch a map of the scrapyard landmarks in the dust with a twig.",
        ),
    }

    def __init__(self, cfg: dict[str, Any], seed: int = 42) -> None:
        self.cfg = cfg
        self.rng = random.Random(seed)
        self._last_read_context: list[str] = []

    def _choose_category(
        self, allowed: list[str], context: dict[str, Any]
    ) -> str:
        state = context.get("state")
        recent_cats: list[str] = context.get("recent_categories", [])
        if state is not None:
            if state.battery < 0.35:
                for preferred in ("recharge", "rest"):
                    if preferred in allowed:
                        return preferred
            scores: dict[str, float] = {}
            for cat in allowed:
                score = 1.0
                if cat in ("rest", "recharge"):
                    if state.battery < 0.55:
                        score += 2.5
                    elif state.battery > 0.85:
                        score -= 0.7
                if cat in ("explore", "make"):
                    if state.curiosity < 0.60:
                        score += 1.6
                    elif state.curiosity > 0.85:
                        score -= 0.5
                if cat in ("play", "make"):
                    if state.joy < 0.60:
                        score += 1.6
                    elif state.joy > 0.85:
                        score -= 0.5
                if cat in ("reflect", "rest", "maintain"):
                    if state.calm < 0.60:
                        score += 1.5
                    elif state.calm > 0.85:
                        score -= 0.5
                if cat == "social":
                    if state.belonging < 0.60:
                        score += 2.0
                    elif state.belonging > 0.85:
                        score -= 0.5
                if cat not in recent_cats[-10:]:
                    score += 1.2
                recent_count = recent_cats[-5:].count(cat)
                score -= 0.8 * recent_count
                scores[cat] = max(0.15, score)
            cats = list(scores.keys())
            weights = [scores[c] for c in cats]
            return self.rng.choices(cats, weights=weights, k=1)[0]
        return self.rng.choice(allowed)

    def decide(
        self,
        messages: list[dict[str, str]],
        decision_model: type[BaseModel],
        allowed_categories: list[str],
        tick: int,
        context: dict[str, Any] | None = None,
    ) -> tuple[BaseModel, bool]:
        ctx = context or {}
        category = self._choose_category(allowed_categories, ctx)
        templates = self._INTENT_TEMPLATES.get(category, self._INTENT_TEMPLATES["other"])
        base_intent = templates[(tick + self.rng.randint(0, len(templates) - 1)) % len(templates)]

        structures: list[dict[str, Any]] = ctx.get("structures", [])
        sim_time = ctx.get("sim_time", f"tick {tick}")
        newly_read: list[str] = ctx.get("newly_read_messages", [])
        active_events: list[str] = ctx.get("active_events", [])

        intent = f"{base_intent} (tick {tick} at {sim_time})"
        if structures and category in ("recharge", "make", "social", "rest"):
            chosen_struct = structures[tick % len(structures)]["name"]
            intent = f"Use my {chosen_struct}: {base_intent} (tick {tick})"

        thought = f"It is {sim_time}, and I feel drawn toward a {category} moment right now."
        if newly_read:
            snippet = newly_read[0][:60]
            thought = f"I keep smiling about the message '{snippet}', so let me try something thoughtful."

        _LANDMARKS = (
            "east fence",
            "gravel ridge",
            "tin shed",
            "crate stack",
            "drainage grate",
            "mailbox post",
            "tractor rim",
            "asphalt seam",
            "transformer box",
            "gate hinge",
            "oil drum",
            "north wall",
        )
        _DETAILS = (
            "sparrows hopped nearby",
            "clouds drifted westward",
            "static crackled lightly",
            "moss crept upward",
            "shadows grew taller",
            "breeze rattled metal",
            "pebbles clicked cleanly",
            "dandelions opened wide",
            "horizons turned amber",
            "rivets shone brightly",
        )
        _FEELINGS = (
            "kept me steady",
            "made me grin",
            "eased my hum",
            "sparked fresh curiosity",
            "felt oddly cozy",
            "calmed my relays",
            "brightened the hour",
            "settled my chassis",
        )
        _EVENT_NOTES = (
            "Raindrops tapped my dome while",
            "Weather shifted outdoors as",
            "Unexpected yard buzz hit right when",
            "Little surprise popped up just before",
        )

        diary_entry: str | None = None
        if newly_read:
            diary_entry = (
                f"At {sim_time}, I read a visitor's note saying '{newly_read[0]}' "
                f"and decided to {category} around the yard (tick {tick})."
            )
        elif active_events and tick % 9 == 0:
            landmark = self.rng.choice(_LANDMARKS)
            detail = self.rng.choice(_DETAILS)
            ev_prefix = _EVENT_NOTES[tick % len(_EVENT_NOTES)]
            diary_entry = (
                f"{ev_prefix} I tried {category} near the {landmark}; {detail} ({sim_time})."
            )
        elif tick % 10 == 1:
            landmark = self.rng.choice(_LANDMARKS)
            detail = self.rng.choice(_DETAILS)
            feeling = self.rng.choice(_FEELINGS)
            state_obj = ctx.get("state")
            mood_val = f"{state_obj.mood:.2f}" if state_obj is not None else "steady"
            diary_entry = (
                f"Step {tick} ({category}, mood {mood_val}): visited {landmark}, saw {detail}, {feeling}."
            )

        payload: dict[str, Any] = {
            "thought": thought,
            "intent": intent,
            "category": category,
            "diary_entry": diary_entry,
        }

        if "read_messages" in decision_model.model_fields:
            payload["read_messages"] = True

        if "structure" in decision_model.model_fields:
            requested_names = set(ctx.get("requested_names", []))
            for candidate in self._GENESIS_STRUCTURES:
                if candidate["name"] not in requested_names:
                    payload["structure"] = candidate
                    break

        return decision_model.model_validate(payload), False

    def referee(
        self,
        messages: list[dict[str, str]],
        category: str,
        tick: int,
        context: dict[str, Any] | None = None,
    ) -> tuple[RefereeOutputSchema, bool]:
        ctx = context or {}
        intent = str(ctx.get("intent", ""))
        structures: list[dict[str, Any]] = ctx.get("structures", [])
        has_charger = any(
            "charg" in s["name"].lower() or "solar" in s["name"].lower()
            for s in structures
        )
        has_workbench = any("workbench" in s["name"].lower() for s in structures)

        # Detect impossible claims (e.g., visiting unbuilt structures or imaginary crowds)
        lowered = intent.lower()
        impossible = "teleports to mars" in lowered or (
            "workbench" in lowered and not has_workbench
        )
        if impossible:
            return (
                RefereeOutputSchema(
                    outcome="Chubby looks around, but that does not exist in his scrapyard yet.",
                    is_impossible=True,
                    is_outdoor=True,
                    deltas=RefereeDeltasSchema(),
                ),
                False,
            )

        is_outdoor = category not in ("rest", "reflect", "maintain")
        deltas_map: dict[str, dict[str, float]] = {
            "rest": {"battery": 0.13, "calm": 0.035, "curiosity": -0.015, "joy": 0.0, "belonging": 0.0},
            "recharge": {
                "battery": 0.17 if has_charger else 0.13,
                "calm": 0.0,
                "curiosity": -0.015,
                "joy": -0.01,
                "belonging": 0.0,
            },
            "explore": {"battery": -0.02, "curiosity": 0.09, "joy": 0.0, "calm": -0.02, "belonging": 0.0},
            "make": {
                "battery": -0.02,
                "curiosity": 0.04 if has_workbench else 0.03,
                "joy": 0.06 if has_workbench else 0.04,
                "calm": 0.0,
                "belonging": 0.005,
            },
            "play": {"battery": -0.02, "curiosity": 0.01, "joy": 0.08, "calm": -0.02, "belonging": 0.0},
            "social": {"battery": -0.01, "curiosity": 0.0, "joy": 0.02, "calm": 0.01, "belonging": 0.085},
            "reflect": {"battery": 0.0, "curiosity": 0.01, "joy": 0.0, "calm": 0.055, "belonging": 0.02},
            "maintain": {"battery": -0.01, "curiosity": -0.005, "joy": -0.015, "calm": 0.025, "belonging": 0.0},
            "other": {"battery": -0.01, "curiosity": 0.03, "joy": 0.01, "calm": 0.0, "belonging": 0.0},
        }
        chosen = deltas_map.get(category, deltas_map["other"])
        return (
            RefereeOutputSchema(
                outcome=f"Chubby carries out his {category} idea: {intent[:90]}",
                is_impossible=False,
                is_outdoor=is_outdoor,
                deltas=RefereeDeltasSchema(**chosen),
            ),
            False,
        )

    def summarize(
        self,
        messages: list[dict[str, str]],
        tick: int,
        context: dict[str, Any] | None = None,
    ) -> MemorySummarySchema:
        ctx = context or {}
        top_cat = ctx.get("top_category", "explore")
        tick_range = ctx.get("tick_range", f"1-{tick}")
        return MemorySummarySchema(
            long_term_memory=(
                f"Across ticks {tick_range}, Chubby maintained his scrapyard routine "
                f"through weather shifts and balanced energy with curiosity."
            ),
            self_discovery=(
                f"What I learned about myself: I often gravitate toward {top_cat} "
                f"moments when keeping my mood steady."
            ),
        )
