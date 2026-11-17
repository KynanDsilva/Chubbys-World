"""Ollama HTTP client, dynamic Pydantic JSON schemas, retries, and deterministic mock policy."""

from __future__ import annotations

import json
import math
import random
import sqlite3
import urllib.error
import urllib.request
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, create_model

import db
from svg_utils import sanitize_svg


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


class DrawingSchema(BaseModel):
    """Structured output schema for Chubby's SVG sketch."""

    title: str
    svg: str


def _star_points(cx: float, cy: float, outer: float, inner: float, pts: int = 4) -> str:
    coords = []
    for i in range(pts * 2):
        r = outer if i % 2 == 0 else inner
        ang = i * math.pi / pts - math.pi / 2
        coords.append(f"{cx + r * math.cos(ang):.1f},{cy + r * math.sin(ang):.1f}")
    return " ".join(coords)


def _motif_helicopter(rng: random.Random) -> dict[str, str]:
    sky = rng.choice(["#bae6fd", "#fed7aa", "#e0f2fe"])
    dark = "#1e293b"
    heli_red = "#ef4444"
    parts = [
        f'<rect x="0" y="0" width="100" height="100" fill="{sky}"/>',
        '<circle cx="88" cy="14" r="7" fill="#facc15" stroke="#eab308" stroke-width="1.5"/>',
        f'<line x1="15" y1="22" x2="75" y2="22" stroke="{dark}" stroke-width="3" stroke-linecap="round"/>',
        '<circle cx="15" cy="22" r="2" fill="#facc15"/>',
        '<circle cx="75" cy="22" r="2" fill="#facc15"/>',
        f'<rect x="43" y="22" width="4" height="8" fill="{dark}"/>',
        f'<path d="M 25 35 Q 22 56 42 58 L 58 58 Q 66 58 68 46 Q 69 35 55 35 Z" fill="{heli_red}" stroke="{dark}" stroke-width="2.5" stroke-linejoin="round"/>',
        f'<path d="M 26 40 Q 25 50 36 53 L 45 53 Q 48 40 45 40 Z" fill="#38bdf8" stroke="{dark}" stroke-width="1.5"/>',
        '<path d="M 30 42 Q 29 47 34 49" fill="none" stroke="#ffffff" stroke-width="1.5" stroke-linecap="round"/>',
        '<path d="M 42 47 L 68 47 L 67 52 L 42 52 Z" fill="#f8fafc"/>',
        f'<polygon points="58,42 85,38 85,44 58,48" fill="{heli_red}" stroke="{dark}" stroke-width="2" stroke-linejoin="round"/>',
        f'<polygon points="82,34 88,34 86,48 80,48" fill="{heli_red}" stroke="{dark}" stroke-width="1.5"/>',
        f'<line x1="84" y1="32" x2="84" y2="42" stroke="{dark}" stroke-width="2" stroke-linecap="round"/>',
        '<circle cx="84" cy="37" r="2" fill="#facc15"/>',
        f'<line x1="34" y1="58" x2="30" y2="68" stroke="{dark}" stroke-width="2" stroke-linecap="round"/>',
        f'<line x1="52" y1="58" x2="48" y2="68" stroke="{dark}" stroke-width="2" stroke-linecap="round"/>',
        f'<path d="M 22 68 L 60 68 Q 65 68 66 65" fill="none" stroke="#475569" stroke-width="3" stroke-linecap="round"/>',
        '<rect x="0" y="80" width="100" height="20" fill="#22c55e"/>',
        '<rect x="0" y="78" width="100" height="3" fill="#16a34a"/>',
        f'<ellipse cx="50" cy="86" rx="24" ry="8" fill="#e2e8f0" stroke="{dark}" stroke-width="2"/>',
        '<line x1="50" y1="81" x2="50" y2="91" stroke="#ef4444" stroke-width="2" stroke-linecap="round"/>',
        '<line x1="42" y1="86" x2="58" y2="86" stroke="#ef4444" stroke-width="2" stroke-linecap="round"/>',
    ]
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">' + "".join(parts) + "</svg>"
    titles = ["The Red Rescue Chopper", "Helicopter on the Pad", "Sky Whirlybird", "Red Rotors in Flight"]
    return {"title": rng.choice(titles), "svg": svg}


def _motif_self_portrait(rng: random.Random) -> dict[str, str]:
    bg = rng.choice(["#fef3c7", "#f1f5f9", "#e2e8f0", "#ccfbf1"])
    dark = "#1e293b"
    body_cyan = "#38bdf8"
    parts = [
        f'<rect x="0" y="0" width="100" height="100" fill="{bg}"/>',
        '<rect x="0" y="76" width="100" height="24" fill="#94a3b8"/>',
        '<line x1="0" y1="76" x2="100" y2="76" stroke="#64748b" stroke-width="2"/>',
        f'<line x1="50" y1="12" x2="50" y2="24" stroke="{dark}" stroke-width="3" stroke-linecap="round"/>',
        '<circle cx="50" cy="11" r="5" fill="#ef4444" stroke="#b91c1c" stroke-width="1.5"/>',
        '<circle cx="48" cy="9" r="1.5" fill="#ffffff"/>',
        f'<path d="M 25 48 Q 14 42 12 34" fill="none" stroke="{dark}" stroke-width="3.5" stroke-linecap="round"/>',
        '<circle cx="12" cy="33" r="3.5" fill="#facc15" stroke="#1e293b" stroke-width="1.5"/>',
        f'<path d="M 75 48 Q 86 52 88 42" fill="none" stroke="{dark}" stroke-width="3.5" stroke-linecap="round"/>',
        '<circle cx="88" cy="41" r="3.5" fill="#facc15" stroke="#1e293b" stroke-width="1.5"/>',
        f'<rect x="24" y="24" width="52" height="46" rx="18" fill="{body_cyan}" stroke="{dark}" stroke-width="3"/>',
        f'<rect x="30" y="32" width="40" height="18" rx="8" fill="#0f172a" stroke="{dark}" stroke-width="2"/>',
        '<circle cx="41" cy="41" r="5" fill="#22d3ee"/>',
        '<circle cx="59" cy="41" r="5" fill="#22d3ee"/>',
        '<circle cx="39.5" cy="39.5" r="1.5" fill="#ffffff"/>',
        '<circle cx="57.5" cy="39.5" r="1.5" fill="#ffffff"/>',
        '<circle cx="50" cy="58" r="4.5" fill="#facc15" stroke="#ca8a04" stroke-width="1.5"/>',
        f'<path d="M 44 63 Q 50 67 56 63" fill="none" stroke="{dark}" stroke-width="2" stroke-linecap="round"/>',
        f'<rect x="28" y="70" width="44" height="10" rx="5" fill="#475569" stroke="{dark}" stroke-width="2.5"/>',
        '<circle cx="36" cy="75" r="3" fill="#cbd5e1"/>',
        '<circle cx="50" cy="75" r="3" fill="#cbd5e1"/>',
        '<circle cx="64" cy="75" r="3" fill="#cbd5e1"/>',
    ]
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">' + "".join(parts) + "</svg>"
    titles = ["Self-Portrait in Mirror", "Me, Chubby!", "Robot Optic Check", "My Shiny Antenna"]
    return {"title": rng.choice(titles), "svg": svg}


def _motif_town_building(rng: random.Random) -> dict[str, str]:
    sky = rng.choice(["#bae6fd", "#fed7aa", "#fbcfe8"])
    facade = rng.choice(["#fef08a", "#fed7aa", "#e2e8f0", "#bbf7d0"])
    roof = "#ea580c"
    dark = "#1e293b"
    parts = [
        f'<rect x="0" y="0" width="100" height="100" fill="{sky}"/>',
        f'<rect x="68" y="16" width="8" height="16" fill="#9a3412" stroke="{dark}" stroke-width="1.5"/>',
        '<circle cx="72" cy="12" r="3" fill="#e2e8f0"/>',
        '<circle cx="75" cy="8" r="4" fill="#f1f5f9"/>',
        '<circle cx="79" cy="4" r="5" fill="#ffffff"/>',
        f'<polygon points="12,38 50,14 88,38" fill="{roof}" stroke="{dark}" stroke-width="2.5" stroke-linejoin="round"/>',
        f'<rect x="20" y="38" width="60" height="46" fill="{facade}" stroke="{dark}" stroke-width="2.5"/>',
        f'<rect x="42" y="42" width="16" height="14" rx="2" fill="#fef9c3" stroke="{dark}" stroke-width="1.5"/>',
        f'<line x1="50" y1="42" x2="50" y2="56" stroke="{dark}" stroke-width="1.2"/>',
        f'<line x1="42" y1="49" x2="58" y2="49" stroke="{dark}" stroke-width="1.2"/>',
        '<polygon points="24,60 76,60 80,66 20,66" fill="#ef4444" stroke="#1e293b" stroke-width="1.5"/>',
        '<polygon points="28,60 38,60 36,66 26,66" fill="#ffffff"/>',
        '<polygon points="46,60 54,60 52,66 44,66" fill="#ffffff"/>',
        '<polygon points="62,60 72,60 70,66 60,66" fill="#ffffff"/>',
        f'<rect x="44" y="67" width="13" height="17" rx="2" fill="#78350f" stroke="{dark}" stroke-width="1.5"/>',
        '<circle cx="54" cy="76" r="1.5" fill="#facc15"/>',
        f'<rect x="23" y="70" width="14" height="6" fill="#92400e" stroke="{dark}" stroke-width="1"/>',
        '<circle cx="26" cy="69" r="2.5" fill="#ef4444"/>',
        '<circle cx="30" cy="68" r="2.5" fill="#ec4899"/>',
        '<circle cx="34" cy="69" r="2.5" fill="#facc15"/>',
        '<rect x="0" y="84" width="100" height="16" fill="#94a3b8"/>',
        '<line x1="0" y1="84" x2="100" y2="84" stroke="#475569" stroke-width="2"/>',
        '<line x1="30" y1="84" x2="25" y2="100" stroke="#64748b" stroke-width="1.5"/>',
        '<line x1="65" y1="84" x2="60" y2="100" stroke="#64748b" stroke-width="1.5"/>',
    ]
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">' + "".join(parts) + "</svg>"
    titles = ["Cozy Town Bakery", "The Cobblestone Cafe", "Corner Brick Shop", "Sunny Street Building"]
    return {"title": rng.choice(titles), "svg": svg}


def _motif_stargazing(rng: random.Random) -> dict[str, str]:
    dark = "#0f172a"
    parts = [
        f'<rect x="0" y="0" width="100" height="100" fill="{dark}"/>',
        '<circle cx="78" cy="22" r="12" fill="#fef08a"/>',
        f'<circle cx="83" cy="19" r="11" fill="{dark}"/>',
    ]
    star_positions = [(15, 18), (35, 12), (54, 25), (24, 38), (85, 45), (42, 48), (68, 36), (12, 55)]
    for sx, sy in star_positions:
        r_out = rng.choice([3.5, 4.5, 5.5])
        col = rng.choice(["#ffffff", "#fef08a", "#67e8f9"])
        parts.append(f'<polygon points="{_star_points(sx, sy, r_out, r_out * 0.35)}" fill="{col}"/>')
    parts += [
        '<ellipse cx="50" cy="102" rx="65" ry="32" fill="#1e293b"/>',
        '<ellipse cx="50" cy="100" rx="60" ry="28" fill="#1e3a5f"/>',
        '<circle cx="50" cy="62" r="7" fill="#38bdf8" stroke="#0284c7" stroke-width="1.5"/>',
        '<line x1="50" y1="55" x2="50" y2="51" stroke="#0284c7" stroke-width="1.5"/>',
        '<circle cx="50" cy="50" r="1.5" fill="#facc15"/>',
        '<circle cx="48" cy="61" r="1.5" fill="#22d3ee"/>',
        '<circle cx="52" cy="61" r="1.5" fill="#22d3ee"/>',
        '<rect x="45" y="68" width="10" height="6" rx="2" fill="#475569"/>',
    ]
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">' + "".join(parts) + "</svg>"
    titles = ["Stargazing on the Hill", "Cosmic Night Sky", "Watching the Crescent Moon", "Little Robot Under Stars"]
    return {"title": rng.choice(titles), "svg": svg}


def _motif_flower_and_sun(rng: random.Random) -> dict[str, str]:
    sky = rng.choice(["#bae6fd", "#e0f2fe"])
    dark = "#1e293b"
    parts = [
        f'<rect x="0" y="0" width="100" height="100" fill="{sky}"/>',
        '<circle cx="82" cy="20" r="11" fill="#facc15" stroke="#eab308" stroke-width="2"/>',
    ]
    for i in range(8):
        ang = i * math.pi / 4
        parts.append(
            f'<line x1="{82 + 13 * math.cos(ang):.1f}" y1="{20 + 13 * math.sin(ang):.1f}" '
            f'x2="{82 + 19 * math.cos(ang):.1f}" y2="{20 + 19 * math.sin(ang):.1f}" '
            f'stroke="#facc15" stroke-width="2.5" stroke-linecap="round"/>'
        )
    parts += [
        '<ellipse cx="25" cy="98" rx="45" ry="24" fill="#22c55e"/>',
        '<ellipse cx="80" cy="100" rx="50" ry="22" fill="#16a34a"/>',
        '<path d="M 44 88 Q 42 66 48 48" fill="none" stroke="#15803d" stroke-width="4" stroke-linecap="round"/>',
        '<path d="M 45 68 Q 30 64 26 58 Q 34 54 44 64" fill="#22c55e" stroke="#15803d" stroke-width="1.5"/>',
        '<circle cx="32" cy="60" r="2.5" fill="#ef4444"/>',
        '<circle cx="31.5" cy="59.5" r="0.7" fill="#000000"/>',
        '<path d="M 47 60 Q 62 58 66 52 Q 58 48 46 56" fill="#22c55e" stroke="#15803d" stroke-width="1.5"/>',
    ]
    fcx, fcy = 48, 44
    petals = 10
    for i in range(petals):
        ang = 2 * math.pi * i / petals
        px = fcx + 13 * math.cos(ang)
        py = fcy + 13 * math.sin(ang)
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5.5" fill="#facc15" stroke="#eab308" stroke-width="1"/>')
    parts += [
        f'<circle cx="{fcx}" cy="{fcy}" r="8.5" fill="#78350f" stroke="{dark}" stroke-width="1.5"/>',
        f'<circle cx="{fcx - 2}" cy="{fcy - 2}" r="1.5" fill="#a16207"/>',
        f'<circle cx="{fcx + 2}" cy="{fcy + 1}" r="1.5" fill="#a16207"/>',
    ]
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">' + "".join(parts) + "</svg>"
    titles = ["Scrapyard Sunflower", "Bright Sunny Meadow", "Petals and Sunbeams", "Curious Little Ladybug"]
    return {"title": rng.choice(titles), "svg": svg}


def _motif_art_studio(rng: random.Random) -> dict[str, str]:
    dark = "#1e293b"
    wall = rng.choice(["#fef3c7", "#f1f5f9", "#e0e7ff"])
    parts = [
        f'<rect x="0" y="0" width="100" height="100" fill="{wall}"/>',
        '<rect x="0" y="78" width="100" height="22" fill="#d97706"/>',
        '<line x1="0" y1="78" x2="100" y2="78" stroke="#92400e" stroke-width="2"/>',
        '<line x1="25" y1="78" x2="20" y2="100" stroke="#b45309" stroke-width="1.5"/>',
        '<line x1="55" y1="78" x2="52" y2="100" stroke="#b45309" stroke-width="1.5"/>',
        '<line x1="85" y1="78" x2="82" y2="100" stroke="#b45309" stroke-width="1.5"/>',
        '<line x1="50" y1="12" x2="30" y2="86" stroke="#78350f" stroke-width="3" stroke-linecap="round"/>',
        '<line x1="50" y1="12" x2="70" y2="86" stroke="#78350f" stroke-width="3" stroke-linecap="round"/>',
        '<line x1="50" y1="12" x2="50" y2="86" stroke="#92400e" stroke-width="2.5" stroke-linecap="round"/>',
        '<line x1="24" y1="62" x2="76" y2="62" stroke="#78350f" stroke-width="4" stroke-linecap="round"/>',
        f'<rect x="30" y="24" width="40" height="36" rx="2" fill="#f8fafc" stroke="{dark}" stroke-width="2"/>',
        '<rect x="32" y="26" width="36" height="32" fill="#bae6fd"/>',
        '<circle cx="50" cy="38" r="6" fill="#f97316"/>',
        '<ellipse cx="50" cy="55" rx="17" ry="9" fill="#22c55e"/>',
        '<ellipse cx="76" cy="66" rx="10" ry="7" fill="#d97706" stroke="#78350f" stroke-width="1.5"/>',
        '<circle cx="73" cy="65" r="1.5" fill="#ef4444"/>',
        '<circle cx="77" cy="63" r="1.5" fill="#38bdf8"/>',
        '<circle cx="80" cy="66" r="1.5" fill="#facc15"/>',
        '<circle cx="76" cy="68" r="1.5" fill="#22c55e"/>',
        '<line x1="72" y1="72" x2="88" y2="58" stroke="#451a03" stroke-width="1.8" stroke-linecap="round"/>',
        '<circle cx="88" cy="58" r="1.2" fill="#ef4444"/>',
    ]
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">' + "".join(parts) + "</svg>"
    titles = ["My Painting Easel", "Art Studio Corner", "Fresh Paint Palette", "Sunset on Mini Canvas"]
    return {"title": rng.choice(titles), "svg": svg}


def procedural_drawing(seed: int, intent: str = "") -> dict[str, str]:
    """Generate a deterministic, seed-varied signature SVG sketch in Chubby's art style."""
    rng = random.Random(seed)
    lowered = intent.lower() if intent else ""
    if any(k in lowered for k in ("chopper", "copter", "helicopter", "fly", "rotor", "propeller", "air")):
        return _motif_helicopter(rng)
    if any(k in lowered for k in ("portrait", "myself", "chubby", "mirror", "face", "chassis", "antenna")):
        return _motif_self_portrait(rng)
    if any(k in lowered for k in ("town", "building", "house", "shop", "bakery", "cafe", "store", "cobblestone", "street", "library", "workshop")):
        return _motif_town_building(rng)
    if any(k in lowered for k in ("star", "night", "moon", "galaxy", "cosmic", "space", "sky", "stargaz")):
        return _motif_stargazing(rng)
    if any(k in lowered for k in ("flower", "sunflower", "daisy", "plant", "meadow", "grass", "sunbeam", "ladybug")):
        return _motif_flower_and_sun(rng)
    if any(k in lowered for k in ("studio", "easel", "palette", "canvas", "brush", "paint", "art")):
        return _motif_art_studio(rng)

    motifs = [
        _motif_helicopter,
        _motif_self_portrait,
        _motif_town_building,
        _motif_stargazing,
        _motif_flower_and_sun,
        _motif_art_studio,
    ]
    fn = rng.choice(motifs)
    return fn(rng)


def safe_procedural_drawing(seed: int, intent: str = "") -> dict[str, str]:
    """Procedural drawing in Chubby's signature art style, guaranteed to pass sanitizer."""
    for offset in range(5):
        d = procedural_drawing(seed + offset, intent=intent)
        cleaned = sanitize_svg(d["svg"])
        if cleaned:
            return {"title": d["title"], "svg": cleaned}
    return {
        "title": "Little Robot Beacon",
        "svg": (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
            '<rect x="0" y="0" width="100" height="100" fill="#bae6fd"/>'
            '<circle cx="50" cy="50" r="22" fill="#38bdf8" stroke="#1e293b" stroke-width="3"/>'
            '<circle cx="50" cy="50" r="10" fill="#facc15"/>'
            "</svg>"
        ),
    }


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
    if category in ("rest", "recharge"):
        deltas = RefereeDeltasSchema(battery=0.12, calm=0.04)
    elif category == "social":
        deltas = RefereeDeltasSchema(belonging=0.05, joy=0.03)
    elif category == "draw":
        deltas = RefereeDeltasSchema(curiosity=0.05, joy=0.09, battery=-0.015)
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
            except (
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
                KeyError,
                ValueError,
            ) as exc:
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

    def generate_drawing(
        self,
        intent: str,
        tick: int,
        context: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """Ask the model for a compact SVG sketch; fall back to a procedural one."""
        ctx = context or {}
        seed = tick * 1009 + int(ctx.get("drawing_count", 0))
        import prompt as _prompt  # local import avoids a circular dependency

        messages = _prompt.build_drawing_messages(
            intent, ctx.get("state"), ctx.get("recent_titles", [])
        )
        result = self._call_with_retries(
            messages, DrawingSchema, tick=tick, stage="drawing", temperature=0.9
        )
        if isinstance(result, DrawingSchema):
            cleaned = sanitize_svg(result.svg)
            if cleaned:
                title = " ".join(result.title.split())[:60] or "Untitled Sketch"
                return {"title": title, "svg": cleaned}
        if self.conn is not None:
            db.insert_event(self.conn, tick, "llm_fallback", {"stage": "drawing"})
        return safe_procedural_drawing(seed, intent=intent)


class MockLLMClient:
    """Deterministic, seedable local policy validating outputs through the real schemas."""

    _GENESIS_STRUCTURES: tuple[dict[str, Any], ...] = (
        {
            "name": "Town Square Gazebo",
            "purpose": "A shaded gazebo near the town edge to rest and watch visitors",
            "why": "It gives a nice view of the cobblestone paths and shelters from rain",
            "priority": 1,
        },
        {
            "name": "Cobblestone Workshop",
            "purpose": "A small craft table to inspect salvaged gadgets and repair parts",
            "why": "Having a dry workbench helps me sort gears and fix loose bolts",
            "priority": 2,
        },
        {
            "name": "Community Tool Shed",
            "purpose": "Store scrap metal, copper wire, and extra canvas supplies",
            "why": "Keeping parts organized makes it easier to build new things",
            "priority": 2,
        },
    )

    _INTENT_TEMPLATES: dict[str, tuple[str, ...]] = {
        "rest": (
            "Power down my drive motors and rest quietly on the warm gravel ground.",
            "Park near the quiet fence corner and let my cooling fan idle gently.",
            "Settle onto a flat patch of dirt to recalibrate my balance gyroscopes.",
            "Dim my optic lamps and listen to the yard settle while resting my servos.",
        ),
        "recharge": (
            "Angle my back panel toward the brightest patch of sky to top up my cells.",
            "Follow the warmest sunlit patch of dirt across the yard to gather solar charge.",
            "Trace the sunlit ridge near the fence line to gather extra solar charge.",
            "Run a careful low-draw solar trickle charge while checking my voltage gauge.",
        ),
        "explore": (
            "Roll along the east fence to catalog small pebbles and dried weeds in the dirt.",
            "Inspect the perimeter fence posts to see how the gravel has shifted.",
            "Follow a trail of shiny washers across the bare ground toward the gate.",
            "Scan the far corner of the bare lot for interesting magnets or glass marbles.",
            "Check sunlight shadows along the north corner to map cool afternoon spots.",
            "Trace cloud shapes slowly from horizon to horizon while scanning the outer berm.",
        ),
        "make": (
            "Assemble a tiny wind chime out of aluminum pull-tabs and copper wire.",
            "Shape a neat little stand for my favorite found bolts in the dirt.",
            "Build a little pebble cairn on the bare ground to mark my home spot.",
            "Twist discarded spring coils into a bouncy companion on the gravel.",
            "Wire a tiny blinking LED pebble beacon from salvage resistor leads.",
            "Carve interlocking wooden gear teeth out of scrap crate splinters.",
            "Design a miniature sundial using a brass pin and smooth flat stones.",
        ),
        "play": (
            "Balance three smooth river stones on my dome and see how long they stay.",
            "Chase reflections cast by a polished hubcap across the dirt yard.",
            "Roll back and forth between two fence posts to test my stopping distance.",
            "Spin in slow circles to test my wheel tracks in the gravel.",
            "Play hopscotch between sunlit gravel patches without touching boundary stones.",
            "Bowl a rusty lug nut gently toward three tin-can pins across the yard.",
        ),
        "social": (
            "Leave a cheerful chalk arrow on the dirt path for anyone passing by.",
            "Look out through the perimeter fence toward the town road and wave at distant lights.",
            "Arrange shiny bottle caps into a welcoming smile by the front gate.",
            "Check my message slot and tidy the welcome pebble beside the path.",
        ),
        "reflect": (
            "Watch clouds drift overhead and notice how peaceful the open lot feels today.",
            "Review my recent wheel tracks in the dirt and ponder what I enjoy most.",
            "Count the stars or distant town lamps while humming softly to myself.",
            "Sit still on the soil and notice how my chassis feels after a busy stretch.",
        ),
        "maintain": (
            "Brush grit out of my left wheel bearing with a stiff paintbrush bristle.",
            "Wipe dust from my optical lens and tighten a slightly loose chest bolt.",
            "Oil my squeaky elbow hinge with a single drop of salvaged machine oil.",
            "Sweep stray gravel away from my favorite resting spot.",
        ),
        "other": (
            "Catalog the subtle shades of dust and soil across the yard.",
            "Listen to the sound of wind whistling through the fence wires.",
            "Practice drawing tiny robot footprints in neat rows across the sand.",
            "Sketch a map of the fence corners in the dust with a twig.",
        ),
        "draw": (
            "Sketch the view from my window on a fresh sheet and pin it to the gallery wall.",
            "Doodle a cheerful picture of something I saw today with bright crayons.",
            "Paint a quick, wobbly portrait of my own round face on the easel.",
            "Draw a little scene of the sky and hang it up for visitors to see.",
            "Try a bold new sketch with fresh colors and see what appears.",
        ),
    }

    def __init__(
        self,
        cfg: dict[str, Any],
        seed: int = 42,
        structures: tuple[dict[str, Any], ...] | None = None,
    ) -> None:
        self.cfg = cfg
        self.seed = seed
        self.rng = random.Random(seed)
        self.structures_pool = structures if structures is not None else self._GENESIS_STRUCTURES

    def generate_drawing(
        self,
        intent: str,
        tick: int,
        context: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """Deterministic procedural sketch, varied by seed, tick and drawing count."""
        ctx = context or {}
        seed = self.seed * 100003 + tick * 1009 + int(ctx.get("drawing_count", 0))
        intent = str(ctx.get("intent", ""))
        return safe_procedural_drawing(seed, intent=intent)

    def _choose_category(self, allowed: list[str], context: dict[str, Any]) -> str:
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
                if cat in ("explore", "make", "draw"):
                    if state.curiosity < 0.60:
                        score += 1.6
                    elif state.curiosity > 0.85:
                        score -= 0.5
                if cat in ("play", "make", "draw"):
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
        base_intent = templates[
            (tick + self.rng.randint(0, len(templates) - 1)) % len(templates)
        ]

        structures: list[dict[str, Any]] = ctx.get("structures", [])
        sim_time = ctx.get("sim_time", f"tick {tick}")
        newly_read: list[str] = ctx.get("newly_read_messages", [])
        active_events: list[str] = ctx.get("active_events", [])

        _CATEGORY_STRUCTURE_KEYWORDS: dict[str, tuple[str, ...]] = {
            "explore": ("periscope", "scanner", "drafting", "lookout"),
            "reflect": ("zen", "light"),
            "make": ("workbench", "scanner", "drafting"),
            "draw": ("studio", "easel"),
            "recharge": ("solar", "charg", "windmill"),
            "rest": ("canopy", "shelter", "zen", "light"),
            "social": ("radio", "chime", "light"),
        }
        intent = f"{base_intent} (tick {tick} at {sim_time})"
        cat_keywords = _CATEGORY_STRUCTURE_KEYWORDS.get(category)
        if structures and cat_keywords:
            matching_structs = [
                s
                for s in structures
                if any(
                    kw in f"{s.get('name', '')} {s.get('unlocks', '')}".lower()
                    for kw in cat_keywords
                )
            ]
            if matching_structs:
                chosen_struct = matching_structs[tick % len(matching_structs)]["name"]
                intent = f"Roll over to and stand at my {chosen_struct} to use it: {base_intent} (tick {tick})"

        thought = f"It is {sim_time}, and I feel drawn toward a {category} moment right now."
        if newly_read:
            snippet = newly_read[0][:60]
            thought = (
                f"I keep smiling about the message '{snippet}', so let me try something thoughtful."
            )

        # Base landmarks only refer to actual features of the empty fenced lot
        base_landmarks = [
            "east fence",
            "gravel patch",
            "north fence line",
            "south perimeter",
            "drainage grate",
            "gate hinge",
            "center clearing",
            "west fence corner",
        ]
        # Include names of approved structures if any have been built
        for st in structures:
            base_landmarks.append(st.get("name", "structure"))
        _LANDMARKS = tuple(base_landmarks)
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
            "A faint grid hum echoed while",
            "Daylight shifted across the fence as",
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

        if "structure" in decision_model.model_fields and self.structures_pool:
            requested_names = set(ctx.get("requested_names", []))
            for candidate in self.structures_pool:
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

        is_outdoor = category not in ("rest", "reflect", "maintain", "draw")
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
            "draw": {"battery": -0.02, "curiosity": 0.04, "joy": 0.06, "calm": 0.0, "belonging": 0.0},
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
                f"through each day and balanced energy with curiosity."
            ),
            self_discovery=(
                f"What I learned about myself: I often gravitate toward {top_cat} "
                f"moments when keeping my mood steady."
            ),
        )
