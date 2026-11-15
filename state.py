"""Core state model, simulated clock, decay rules, and bounded need math."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

NEED_NAMES: tuple[str, ...] = ("battery", "curiosity", "joy", "calm", "belonging")


class WorldState(BaseModel):
    """Snapshot of Chubby's internal state and simulated time at a given tick."""

    tick: int = Field(ge=0)
    sim_time: str
    battery: float = Field(ge=0.0, le=1.0)
    curiosity: float = Field(ge=0.0, le=1.0)
    joy: float = Field(ge=0.0, le=1.0)
    calm: float = Field(ge=0.0, le=1.0)
    belonging: float = Field(ge=0.0, le=1.0)
    mood_modifier: float = Field(ge=-0.3, le=0.3)
    energy: float = Field(ge=0.0, le=1.0)
    mood: float = Field(ge=0.0, le=1.0)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a floating-point value into [low, high]."""
    return max(low, min(high, float(value)))


def compute_energy(battery: float, calm: float) -> float:
    """Compute energy from battery and calm, bounded to [0.0, 1.0]."""
    return clamp(0.7 * battery + 0.3 * calm, 0.0, 1.0)


def compute_mood(
    joy: float,
    curiosity: float,
    belonging: float,
    calm: float,
    mood_modifier: float,
) -> float:
    """Compute mood from weighted needs plus mood_modifier, bounded to [0.0, 1.0]."""
    raw = (
        0.30 * joy
        + 0.25 * curiosity
        + 0.25 * belonging
        + 0.20 * calm
        + mood_modifier
    )
    return clamp(raw, 0.0, 1.0)


def _total_sim_minutes(tick: int, cfg: dict[str, Any]) -> int:
    clock_cfg = cfg["clock"]
    return int(clock_cfg["start_hour"]) * 60 + tick * int(clock_cfg["sim_minutes_per_tick"])


def get_sim_hour(tick: int, cfg: dict[str, Any]) -> int:
    """Return the simulated hour of day (0-23) for the given tick."""
    minute_of_day = _total_sim_minutes(tick, cfg) % (24 * 60)
    return minute_of_day // 60


def get_sim_day(tick: int, cfg: dict[str, Any]) -> int:
    """Return the 1-indexed simulated day count based on ticks_per_day."""
    ticks_per_day = int(cfg["clock"]["ticks_per_day"])
    if tick <= 0:
        return 1
    return ((tick - 1) // ticks_per_day) + 1


def format_sim_time(tick: int, cfg: dict[str, Any]) -> str:
    """Format simulated clock as 'Day N, HH:MM'."""
    total_minutes = _total_sim_minutes(tick, cfg)
    day_num = (total_minutes // (24 * 60)) + 1
    minute_of_day = total_minutes % (24 * 60)
    hour = minute_of_day // 60
    minute = minute_of_day % 60
    return f"Day {day_num}, {hour:02d}:{minute:02d}"


def is_night(tick: int, cfg: dict[str, Any]) -> bool:
    """Return True if the simulated clock is within night hours."""
    hour = get_sim_hour(tick, cfg)
    night_start = int(cfg["clock"]["night_start_hour"])
    night_end = int(cfg["clock"]["night_end_hour"])
    return hour >= night_start or hour < night_end


def is_dusk(tick: int, cfg: dict[str, Any]) -> bool:
    """Return True if the simulated clock is within dusk hours."""
    hour = get_sim_hour(tick, cfg)
    dusk_start = int(cfg["clock"]["dusk_start_hour"])
    dusk_end = int(cfg["clock"]["dusk_end_hour"])
    return dusk_start <= hour <= dusk_end


def create_initial_state(cfg: dict[str, Any]) -> WorldState:
    """Build Chubby's starting WorldState at tick 0."""
    needs_cfg = cfg["needs"]
    battery = float(needs_cfg["initial_battery"])
    curiosity = float(needs_cfg["initial_curiosity"])
    joy = float(needs_cfg["initial_joy"])
    calm = float(needs_cfg["initial_calm"])
    belonging = float(needs_cfg["initial_belonging"])
    mood_modifier = float(needs_cfg["initial_mood_modifier"])
    return WorldState(
        tick=0,
        sim_time=format_sim_time(0, cfg),
        battery=battery,
        curiosity=curiosity,
        joy=joy,
        calm=calm,
        belonging=belonging,
        mood_modifier=mood_modifier,
        energy=compute_energy(battery, calm),
        mood=compute_mood(joy, curiosity, belonging, calm, mood_modifier),
    )


def _with_recomputed(
    state: WorldState,
    values: dict[str, float],
    mood_modifier: float,
    cfg: dict[str, Any],
) -> WorldState:
    needs_cfg = cfg["needs"]
    mod_min = float(needs_cfg["mood_modifier_min"])
    mod_max = float(needs_cfg["mood_modifier_max"])
    clamped = {name: round(clamp(values[name], 0.0, 1.0), 6) for name in NEED_NAMES}
    clamped_mod = round(clamp(mood_modifier, mod_min, mod_max), 6)
    energy = round(compute_energy(clamped["battery"], clamped["calm"]), 6)
    mood = round(
        compute_mood(
            clamped["joy"],
            clamped["curiosity"],
            clamped["belonging"],
            clamped["calm"],
            clamped_mod,
        ),
        6,
    )
    return state.model_copy(
        update={
            **clamped,
            "mood_modifier": clamped_mod,
            "energy": energy,
            "mood": mood,
        }
    )


def apply_passive_decay(
    state: WorldState, chosen_category: str, cfg: dict[str, Any]
) -> WorldState:
    """Apply per-tick passive decay to needs and mood_modifier.

    Battery passive decay is skipped when Chubby chooses 'rest' or 'recharge'.
    """
    needs_cfg = cfg["needs"]
    battery_decay = (
        0.0
        if chosen_category in ("rest", "recharge")
        else float(needs_cfg["decay_battery"])
    )
    decayed = {
        "battery": state.battery - battery_decay,
        "curiosity": state.curiosity - float(needs_cfg["decay_curiosity"]),
        "joy": state.joy - float(needs_cfg["decay_joy"]),
        "calm": state.calm - float(needs_cfg["decay_calm"]),
        "belonging": state.belonging - float(needs_cfg["decay_belonging"]),
    }
    mod_decay = float(needs_cfg["mood_modifier_decay"])
    new_modifier = state.mood_modifier * (1.0 - mod_decay)
    return _with_recomputed(state, decayed, new_modifier, cfg)


def apply_deltas(
    state: WorldState,
    deltas: dict[str, float],
    mood_mod_delta: float = 0.0,
    cfg: dict[str, Any] | None = None,
) -> WorldState:
    """Apply need deltas and a mood_modifier adjustment with clamping."""
    if cfg is None:
        cfg = {
            "needs": {
                "mood_modifier_min": -0.30,
                "mood_modifier_max": 0.30,
            }
        }
    updated = {
        name: getattr(state, name) + float(deltas.get(name, 0.0))
        for name in NEED_NAMES
    }
    new_modifier = state.mood_modifier + float(mood_mod_delta)
    return _with_recomputed(state, updated, new_modifier, cfg)

