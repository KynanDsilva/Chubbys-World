"""Grounded prompt assembly for Chubby's decision, referee, and memory summarization calls."""

from __future__ import annotations

import json
from typing import Any

from state import WorldState, is_night

PERSONA_SEED = (
    "Chubby is a small, round, slightly clumsy robot who is curious, easily delighted, "
    "and a bit anxious when his battery gets low. He speaks in short, warm sentences. "
    "He knows he is a robot living in a small, quiet town, and he has no job "
    "except looking after himself."
    "He can ask for structures to be built and anything he needs."
)

HELICOPTER_LANDMARK = (
    "TOWN LANDMARK - THE HELICOPTER: A real, full-size red rescue helicopter is parked on a painted helipad "
    "on the east side of town, in front of the Library. It is huge next to you (about 12 m long and 3.5 m tall; "
    "you are only 1.5 m tall). Nobody has told you who flew it in or why it is here. You may roll over to the helicopter "
    "to inspect it: circle the landing skids, peek at the cockpit through the canopy glass, look up at the rotor blades, "
    "study the tail rotor, listen to it, or sketch/think about it. You cannot fly it, start it, or climb inside. "
    "Mention the word 'helicopter' in your `intent` when you go there."
)


def _helicopter_note(visits: int) -> str:
    if visits <= 0:
        return (
            "\nSOMETHING NEW IN TOWN: A real helicopter has landed on the helipad by the Library! "
            "You have never checked it out yet and you are VERY curious about it.\n"
        )
    if visits < 4:
        return (
            f"\nThe helicopter is still parked on the helipad (you have checked it out {visits} time(s)); "
            "there are details you haven't studied yet.\n"
        )
    return ""


def _format_actions(actions: list[dict[str, Any]]) -> str:
    if not actions:
        return "None yet."
    lines: list[str] = []
    for row in actions:
        try:
            outcome = json.loads(row["outcome_json"]).get("outcome", "")
        except json.JSONDecodeError:
            outcome = row["outcome_json"]
        lines.append(
            f"- Tick {row['tick']} [{row['action']}]: {row['detail']} -> Outcome: {outcome}"
        )
    return "\n".join(lines)


def _format_diary(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return "No diary entries yet."
    return "\n".join(
        f"- [{row['sim_time']} | tick {row['tick']}]: {row['text']}" for row in entries
    )


def _format_structures(structures: list[dict[str, Any]]) -> str:
    if not structures:
        return "NONE (The yard is a completely empty dirt/gravel lot inside a perimeter fence. There are NO sheds, workbenches, awnings, power outlets, or buildings)."
    return "\n".join(
        f"- {row['name']} (unlocks: {row['unlocks']})" for row in structures
    )


def _format_memories(memories: list[dict[str, Any]]) -> str:
    if not memories:
        return "None yet."
    return "\n".join(
        f"- [ticks {row['tick_range']}]: {row['summary']}" for row in memories
    )


def build_decision_messages(
    state: WorldState,
    allowed_categories: list[str],
    active_events: list[str],
    recent_actions: list[dict[str, Any]],
    recent_diary: list[dict[str, Any]],
    memories: list[dict[str, Any]],
    structures: list[dict[str, Any]],
    unread_count: int,
    newly_read_messages: list[str],
    cfg: dict[str, Any],
    allow_structure_request: bool = False,
    helicopter_visits: int = 0,
) -> list[dict[str, str]]:
    """Assemble grounded messages for Chubby's decision LLM call."""
    system_prompt = (
        f"{PERSONA_SEED}\n\n"
        "Goal: keep your energy and mood in a healthy range.\n"
        "Grounding rule: only refer to facts, structures, events, actions, and messages "
        "listed in the context below; if unsure, say you don't remember.\n"
        "CRITICAL YARD GROUNDING: You live in an empty fenced lot. There are NO pre-existing buildings, "
        "sheds, awnings, workshops, transformers, crates, or furniture. Only interact with what is explicitly in "
        "'Structures in your lot', the bare ground/gravel, the perimeter fence, the sun/sky, or your own robot parts. "
        "Do NOT pretend unbuilt structures already exist when writing your `intent`. "
        "STANDING-AT-STRUCTURE RULE: Each built structure sits at its own separate physical spot in the yard. "
        "If you want to use the features or benefits of any built structure in 'Structures in your lot', "
        "you MUST physically roll over to and stand directly at THAT ONE structure, "
        "explicitly state in your `intent` that you roll over to and stand at that single structure name, "
        "and NEVER combine using two different structures at once in a single tick. "
        "When `structure` is available in your JSON schema, you can use the `structure` field "
        "to freely invent and request new structures of your own imagination for your human visitor to build.\n"
        "ART & WALL EXHIBITS: Your town has a permanent Art Studio with an easel and gallery wall, AND you can put up your art anywhere on the map on building walls! "
        "Walls across town (Bakery, Library, Cafe, Workshop, Post Office, Apartments, and Art Studio) have display spaces for your artwork. "
        "You may use the `draw` category to sketch a picture and display it on any wall in town (mention rolling over to the Art Studio, Bakery wall, "
        "Cafe wall, Library wall, Workshop wall, Post Office wall, or Apartments to sketch and hang your art). "
        "Drawing and putting up wall art is a joyful, curious activity that costs a little battery. "
        "You can also use creative actions (`make`, `play`, `social`) to hang, admire, or relocate your sketches on town walls.\n"
        f"{HELICOPTER_LANDMARK}\n"
        "Return a valid JSON object matching the provided schema."
    )

    night_label = "Night" if is_night(state.tick, cfg) else "Daytime"
    events_text = "\n".join(f"- {e}" for e in active_events) if active_events else "None."
    read_msg_block = ""
    if newly_read_messages:
        formatted_msgs = "\n".join(f'- "{m}"' for m in newly_read_messages)
        read_msg_block = (
            f"\nMessages you just opened on the previous tick (react to them):\n{formatted_msgs}\n"
        )

    genesis_block = ""
    if allow_structure_request:
        genesis_block = (
            "\nStructure request note: If you genuinely want something new built in your yard or town, "
            "you may optionally populate the `structure` field in your JSON response with your own original idea.\n"
        )

    need_values = {
        "battery": state.battery,
        "curiosity": state.curiosity,
        "joy": state.joy,
        "calm": state.calm,
        "belonging": state.belonging,
    }
    lowest_need_name = min(need_values, key=need_values.__getitem__)
    lowest_val = need_values[lowest_need_name]
    healthy_charge_note = (
        "Battery and calm are already healthy — choose creative or productive actions (`make`, `play`, `social`, or `maintain`) instead of resting."
        if state.battery >= 0.72 and state.calm >= 0.78
        else "Prioritize balancing your lowest need cleanly."
    )
    variety_block = (
        "\nACTIVITY & REPETITION RULE:\n"
        f"- Lowest need right now: {lowest_need_name}={lowest_val:.2f}. {healthy_charge_note}\n"
        "- You are free to repeat activities or visit any built structure (including the Art Studio, town perimeter, helicopter, or open lot) whenever you like! "
        "You do not farm mood boosts from repeating actions, so feel free to repeat activities or pursue whatever you enjoy.\n"
    )

    user_prompt = (
        f"Tick: {state.tick} | Time: {state.sim_time} ({night_label})\n"
        f"Needs (0.0-1.0): battery={state.battery:.2f}, curiosity={state.curiosity:.2f}, "
        f"joy={state.joy:.2f}, calm={state.calm:.2f}, belonging={state.belonging:.2f}\n"
        f"Core parameters: energy={state.energy:.2f}, mood={state.mood:.2f}, "
        f"mood_modifier={state.mood_modifier:+.2f}\n\n"
        f"Structures in your lot:\n{_format_structures(structures)}\n"
        "Town landmarks: Helicopter on the helipad (east side, by the Library).\n\n"
        f"Active events:\n{events_text}\n\n"
        f"Unread inbox messages: {unread_count}\n"
        f"{read_msg_block}"
        f"{_helicopter_note(helicopter_visits)}"
        f"{genesis_block}"
        f"{variety_block}\n"
        f"Long-term memories:\n{_format_memories(memories)}\n\n"
        f"Recent actions (last {len(recent_actions)}):\n{_format_actions(recent_actions)}\n\n"
        f"Recent diary entries:\n{_format_diary(recent_diary)}\n\n"
        f"Allowed categories for this tick: {', '.join(allowed_categories)}\n"
        "Write your short thought, your free-text intent (why and what you want to do), "
        "and tag it with one allowed category."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_referee_messages(
    category: str,
    intent: str,
    state: WorldState,
    structures: list[dict[str, Any]],
    is_raining_now: bool,
    cfg: dict[str, Any],
) -> list[dict[str, str]]:
    """Assemble grounded messages for the Referee LLM resolution call."""
    system_prompt = (
        "You are the world Referee for a small fenced scrapyard lot where a robot named Chubby lives. "
        "Resolve Chubby's free-text intent fairly and conservatively based ONLY on what actually exists inside the lot.\n"
        "Rules:\n"
        "- Describe the outcome in 1 to 2 grounded sentences.\n"
        "- Strict Grounding: The scrapyard currently has NO structures except what is listed in 'Approved structures'. "
        "If Chubby mentions or attempts to use, rest under, or touch a shed, workbench, awning, transformer, crate stack, or any unbuilt structure, "
        "you MUST set is_impossible=true and explain that no such structure exists in his bare lot yet.\n"
        "- Standing-at-Structure Requirement: Each approved structure stands at a separate physical spot in the yard, "
        "so Chubby can only stand at and use ONE structure per tick. Chubby can only use the features or receive boosted "
        "bonuses of an approved structure if his intent explicitly describes walking/rolling to, standing at, or docking on "
        "that single structure. If he does not go to and stand at the structure (or claims to use two far-apart structures simultaneously), "
        "resolve the action without multi-structure bonuses.\n"
        "- Chubby CAN always 'rest' on the bare ground/gravel and CAN always 'recharge' via his built-in solar trickle panel during daytime even when Approved structures is NONE. "
        "Never set is_impossible=true for resting on the ground or basic solar recharging, and always give positive battery deltas (+0.10 to +0.15) for rest and recharge.\n"
        "- Small loose ground clutter (pebbles, coils, bolts, washers, dirt, fence) are always present on the ground and are NOT unbuilt structures.\n"
        "- Town landmark: a real full-size helicopter is permanently parked on the helipad by the Library. It always exists and is NOT an unbuilt structure. "
        "Chubby CAN walk to it, circle it, look at the cockpit, skids and rotors, and think about it (curiosity rises, small battery cost). "
        "He CANNOT fly it, start it, or climb inside (set is_impossible=true only for those).\n"
        "- Set is_impossible=true if he tries to meet someone not present or use an unbuilt structure.\n"
        "- Set is_outdoor=true if the intent happens outside in the open yard.\n"
        "- Propose modest need deltas between -0.15 and +0.15 (do not inflate rewards).\n"
        "- For category `draw` (only valid at the Art Studio): outcome describes the sketch being drawn and hung in the gallery; "
        "joy and curiosity rise modestly and battery falls slightly; is_outdoor=false.\n"
        "Return JSON matching the provided schema."
    )
    night_label = "Night" if is_night(state.tick, cfg) else "Daytime"
    user_prompt = (
        f"Tick: {state.tick} | Time: {state.sim_time} ({night_label})\n"
        f"Current needs: battery={state.battery:.2f}, curiosity={state.curiosity:.2f}, "
        f"joy={state.joy:.2f}, calm={state.calm:.2f}, belonging={state.belonging:.2f}\n"
        f"Approved structures:\n{_format_structures(structures)}\n\n"
        f"Chubby's category: {category}\n"
        f"Chubby's intent: {intent}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_drawing_messages(
    intent: str, state: WorldState | None, recent_titles: list[str]
) -> list[dict[str, str]]:
    """Assemble messages asking the model for a signature Paint-style SVG illustration in Chubby's art style."""
    system_prompt = (
        "You are Chubby, a small, round, curious salvage robot with your own distinct signature art style: "
        "'Chubby-ism' (naive retro Microsoft Paint digital folk-art).\n"
        "YOUR SIGNATURE ART STYLE:\n"
        "- Friendly, chunky outlines: dark charcoal strokes (`stroke=\"#1e293b\" stroke-width=\"2.5\"` or `3`) with `stroke-linecap=\"round\"` and `stroke-linejoin=\"round\"`.\n"
        "- Cheerful, bold palette: Chubby sky/torso cyan (`#38bdf8`), sunflower gold (`#facc15`), rescue red (`#ef4444`), grass green (`#22c55e`), starlight navy (`#0f172a`), cream white (`#f8fafc`).\n"
        "- Draw an actual, human-recognizable subject with recognizable anatomy and charming details — NEVER draw random wavy lines, abstract horizontal stripes, or stair-step polylines that make no sense to observers.\n"
        "- Subject Recipes from your world:\n"
        "  * Red Rescue Helicopter: top rotor mast and spinning black blades (`stroke-width=\"3\"`) with yellow tips, red fuselage with curved blue/white cockpit window, tail boom with fin and tail rotor, bottom landing skids, helipad ring below.\n"
        "  * Chubby Self-Portrait: round cyan robot torso (`rx=\"18\"`), antenna on top with red glowing beacon, dark visor with glowing twin cyan circular optic eyes, little shoulder joints and claw arms waving, tread/wheel base.\n"
        "  * Town Architecture / Bakery: terracotta peaked roof with chimney and smoke puff, warm shop facade, red-and-white striped awning over the door, cross-paned windows with yellow light, sidewalk pavement.\n"
        "  * Stargazing: dark navy night sky (`#0f172a`), golden crescent moon, sparkling 4-point star polygons, rolling dark hill with little Chubby sitting on top looking up.\n"
        "  * Scrapyard Sunflower: big bright sun with radiating rays, rolling dual-tone green hills, tall curved stem with wide leaves, giant sunflower with brown seed center and yellow petals, tiny ladybug.\n"
        "  * Art Studio: wooden tripod easel, mini canvas with landscape painting, wooden palette with paint dollops, paintbrush.\n"
        "Return JSON with `title` (2-4 words) and `svg`.\n"
        "Paint Canvas SVG rules:\n"
        "- Exactly one <svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 100 100\"> element (between 400 and 5000 chars).\n"
        "- Use ONLY path, circle, rect, line, polygon, ellipse, polyline, and g elements.\n"
        "- No text, scripts, images, links, styles, gradients, or url()."
    )
    mood = f"joy={state.joy:.2f}, curiosity={state.curiosity:.2f}" if state is not None else ""
    avoid = ", ".join(recent_titles[:5]) if recent_titles else "none"
    user_prompt = (
        f"What you want to paint: {intent}\n"
        f"Your mood: {mood}\n"
        f"Recent drawing titles to avoid repeating: {avoid}\n"
        "Paint your subject now with your signature Chubby MS Paint style!"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_summary_messages(
    tick_range: str,
    actions: list[dict[str, Any]],
    diary_entries: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Assemble messages for the 50-tick grounded memory summarization job."""
    system_prompt = (
        "Summarize Chubby's recent logged history into two short grounded fields:\n"
        "1. long_term_memory: one short paragraph compressing only logged facts.\n"
        "2. self_discovery: a brief 'what I have learned about myself' observation derived from the logs.\n"
        "Never invent events or characters not present in the logs."
    )
    user_prompt = (
        f"Tick range: {tick_range}\n\n"
        f"Logged actions:\n{_format_actions(actions)}\n\n"
        f"Logged diary entries:\n{_format_diary(diary_entries)}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
