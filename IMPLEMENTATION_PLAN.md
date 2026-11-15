# Chubby: Iteration 1 Implementation Plan

*A headless, local-LLM autonomous robot character. Version 1.2 (open-ended actions with a referee, refined math & schemas), drafted November 15, 2026.*

---

## 1. Goal and scope

**Goal of Iteration 1:** prove that Chubby, a robot with needs, mood and energy, produces *emergent, non-repetitive behavior* on his own, running entirely on your machine with a downloaded Ollama model.

This iteration is **headless**: no 3D world, no website, no art. Everything runs in the terminal and a SQLite database. If Chubby does not feel alive in the logs, no amount of visuals will fix it, so we validate the mind first.

**Direction (updated):** Chubby has no fixed action menu. He decides in his own words what to do, and a referee layer decides what happens. The 3D world is postponed until the mind works.

### In scope

* Core state engine: five needs, two core parameters (energy and mood), simulated clock
* Open-ended decision loop: Chubby states his own intent in free text through Ollama with schema-constrained JSON output; a referee layer (rules plus a second LLM call) resolves what happens within hard caps
* Anti-repetition system (mood penalty, blocked categories, similarity check)
* Random external events
* Diary and grounded memory
* Genesis phase: Chubby requests infrastructure; you approve or reject from the terminal
* Visitor messages through a command-line inbox
* A metrics report that measures emergent behavior

### Out of scope (later iterations)

* 3D world, character model, animation, UI panels, minimap, loading screen
* Public website, visitor accounts, images, moderation
* Per-visitor memory, promises, vector search
* Cloud hosting

---

## 2. Success criteria

The design goals come from the reference video you shared: **emergent behavior** and **organic, non-repetitive output**. We turn them into numbers.

| Metric | Target after a 24-hour soak test |
| --- | --- |
| Distinct action categories used in any 100-tick window | 6 or more, with intent text mostly unique |
| Longest streak of the same category | 2 or fewer |
| Diary entries with high similarity to a previous one (Jaccard above 0.6) | under 15% |
| Mood stays between 0.2 and 0.9 without manual fixes | at least 90% of ticks |
| Invalid or rejected LLM outputs | under 1% |
| Chubby makes at least 3 sensible infrastructure requests in genesis | yes |
| Behavior you did not script that you can point to in the diary (a habit, a preference, a recurring theme) | at least 1 |

---

## 3. Tech stack

| Piece | Choice | Why |
| --- | --- | --- |
| Language | Python 3.11+ | Fast to build, strong libraries |
| LLM runtime | Ollama on localhost | Free per call, simple API, supports JSON-schema output |
| Default model | `qwen3:4b` (configurable) | Small enough for a 4 GB GPU at 4-bit with good structured output; try other 3 to 4B models if the voice feels bland |
| Storage | SQLite (stdlib `sqlite3`) | Zero setup, easy to inspect |
| Validation | `pydantic` | Typed schemas for state, actions and LLM output |
| Scheduler | A simple loop with sleep (`APScheduler` later) | Keeps v1 debuggable |
| Config | `config.toml` | Model name, tick interval, constants |

> [!NOTE]
> **Hardware & runtime note:** Your Lenovo IdeaPad Gaming (RTX 2050 with 4 GB VRAM, 8 GB RAM) fits a 3 to 4B model at 4-bit. Keep the context window small (2048 to 4096 tokens), close other heavy apps while it runs, and make sure Ollama uses the NVIDIA GPU, not the integrated AMD one. Pass `"think": false` (disabling hidden reasoning tokens on routine ticks for `qwen3:4b`) and `"keep_alive": "15m"` (keeping the model warm in VRAM across 30-second ticks) in Ollama requests. Check `ollama.com/library` for current model names before pulling.

---

## 4. Project structure

```text
chubby/
  config.toml
  main.py        # tick loop
  cli.py         # leave message, approve requests, show status
  state.py       # needs, energy, mood, clock
  referee.py     # resolves free-text intents into capped effects
  events.py      # random events
  antirepeat.py  # repetition penalty, blocking, similarity
  llm.py         # Ollama client, schema building, retries
  prompt.py      # prompt assembly
  memory.py      # diary, event log, summaries
  genesis.py     # infrastructure requests and structures
  report.py      # emergent-behavior metrics
  db.py          # schema and queries
  chubby.sqlite
```

---

## 5. Design

### 5.1 Needs and core parameters

Five needs, each from `0.0` to `1.0`:

* `battery` (hunger analogue)
* `curiosity` (wonder)
* `joy`
* `calm`
* `belonging` (rises when messages arrive or he interacts; decays when alone)

Two core parameters that Chubby is trying to regulate:

* `energy = 0.7 * battery + 0.3 * calm`
* `mood = 0.30 * joy + 0.25 * curiosity + 0.25 * belonging + 0.20 * calm + mood_modifier`

`mood_modifier` is a separate value in the range `-0.3` to `+0.3`. It collects anti-repetition penalties, novelty rewards and event effects, and it decays back toward `0` by 10% each tick. Chubby's goal in the prompt is simply: keep energy and mood in a healthy range.

**Per-tick passive decay (starting values, tune later):** `battery -0.020`, `curiosity -0.015`, `joy -0.015`, `calm -0.010`, `belonging -0.012`.

**Refined decay and recovery balance:**

* Passive `battery` decay (`-0.020`) is skipped on ticks where Chubby chooses `rest` or `recharge` so recovery is not eroded during rest ticks.
* Solo social or connection intents (e.g., writing a note for future visitors, tending his scrapyard space, listening to radio static) can grant modest `belonging` recovery (`+0.04` to `+0.10`), while CLI visitor messages grant a major boost (`+0.20`), preventing `belonging` from pinning at `0.0` during unattended soak runs.

### 5.2 Simulated time

Each tick advances the simulated clock by 10 minutes (144 ticks per simulated day). The clock drives a day and night cycle: night raises the battery cost of outdoor/active actions and boosts the recovery effect of rest.

**Real tick interval:** **30 seconds** default in `config.toml`, with CLI flags (`--interval` and `--ticks`) on `main.py` for fast test runs or custom soak intervals.

### 5.3 Open-ended actions and the referee

Chubby has no menu of actions. Each tick he writes, in his own words, what he wants to do and why, and tags it with one coarse category so the system can enforce variety. A **referee** then decides what happens.

* **Rules first.** Code checks hard limits: time of day, the battery cost of the category, which structures actually exist (only approved ones), and what he has already tried.
* **Then a second LLM call.** The model describes the outcome in one or two sentences, grounded in the world state, and proposes changes to the five needs using the explicit **Referee JSON schema**:

```json
{
  "outcome": "1-2 sentences grounded in current structures, weather, and time of day",
  "is_impossible": false,
  "is_outdoor": true,
  "deltas": {
    "battery": 0.0,
    "curiosity": 0.0,
    "joy": 0.0,
    "calm": 0.0,
    "belonging": 0.0
  }
}
```

* **Caps.** Standard proposed changes are clamped to `[-0.15, +0.15]` in one tick. `battery` rises only through `rest`, a `recharge`-type intent, or an approved charging structure (with an elevated recovery cap of `+0.22` before noise when using an approved charging structure or resting at night). If `is_impossible` is `true` (such as meeting a person who does not exist or using an unbuilt structure), all deltas are zeroed out and replaced with a neutral outcome. If `is_outdoor` is `true` during rain or night, additional battery cost is applied.
* **Noise.** Every applied effect is multiplied by a random factor between `0.7` and `1.3`, so the same intent never gives the same result twice.

**Categories** are only buckets used for variety control, never instructions: `rest`, `recharge`, `explore`, `make`, `play`, `social`, `reflect`, `maintain`, `other`. They say nothing about what Chubby should do.

The referee is where Chubby's imagination meets the rules of the world: he can try anything, but the world decides how much it changes him.

### 5.4 Anti-repetition (three layers)

This follows the reference design. Repetition is discouraged by code, not by prompt wording.

1. **Mood penalty.** If the chosen category appeared in the last 5 ticks, `mood_modifier` decreases by `0.07` per occurrence. If it did not appear in the last 10 ticks, `mood_modifier` increases by `0.01`. Negative is seven times stronger than positive, as in the reference.
2. **Blocked categories.** The categories used in the last 2 ticks are removed from the allowed list for the next tick. The JSON schema's `category` field is an enum built from the allowed list, so the model cannot choose them. Inside the remaining categories, his intent is free text.
3. **Intent similarity.** The free-text `intent` is compared with the last 10 intents using Jaccard similarity on word sets (a small local embedding model can replace this later). Above `0.6` counts as a repeat and takes the same penalty (`-0.07`) as layer 1.

### 5.5 Random external events

Each tick, roll for at most one event. Chubby cannot control them and must adapt.

| Event | Chance per tick | Effect |
| --- | --- | --- |
| Rain | 5% (lasts 6 ticks) | `calm` up slightly, outdoor actions cost more `battery` |
| Power flicker | 3% | `battery` down, `calm` down |
| Lucky find | 3% | `joy` up, `curiosity` up |
| Glitch | 2% | `mood_modifier` down |
| Warm evening | 4% at dusk | `calm` up, `joy` up |
| Visitor message | from the CLI inbox | `belonging` changes depending on content |

Later, real weather from a free weather API can replace the random weather.

### 5.6 Memory and grounding

LLMs invent things, so Chubby may only reference what is logged.

* `actions` table: every tick, with action, detail, thought and resulting state
* `events` table: every random event and message
* `diary` table: entries Chubby writes, with the simulated timestamp
* The prompt includes the last 8 actions with their *real* outcomes and the last 3 diary entries
* The system prompt says: only refer to things listed in the context; if unsure, say you don't remember
* Every 50 ticks, a summarization job (using the same model) compresses older diary and action history into a short "long-term memory" paragraph saved in a `memories` table

Habits and preferences come only from this log. Each summarization job also writes a short "what I have learned about myself" note, so preferences are discovered by Chubby, not scripted by us.

### 5.7 Genesis: Chubby asks for infrastructure

Chubby starts in an almost empty world with only a bare lot and no structures. For the first 5 ticks, the prompt adds: *you have just been built; what do you need to live here?*

1. During genesis Chubby may include a `structure` request in his output. It holds `name`, `purpose`, `why`, and a `priority` from `1` to `3`.
2. Requests are stored in a `requests` table with status `pending`.
3. You review them from the terminal: `python cli.py requests`, then `approve <id>` or `reject <id> <reason>`.
4. Approved requests become rows in `structures` and join the world state the referee sees, so intents that use them (a workbench, a charging station, a music corner) get stronger effects and new possibilities. A basic solar trickle and plain rest always work, so Chubby can never be soft-locked.
5. **Approval and rejection effects:** Approval raises `joy` (`+0.10`) and `belonging` (`+0.10`), inserts the structure into `structures`, and logs an event so it is written into his next diary entry. Rejection applies a gentle cost (`-0.03` to `mood_modifier`) and logs an event with the rejection reason so Chubby reflects on it in his next diary entry.
6. **Schema-enforced rate limit:** After genesis (tick > 5), Chubby may request more structures, limited to 1 request per simulated day (144 ticks) to stop spam. The optional `"structure"` field is dynamically included in the decision JSON schema **only** during genesis (ticks 1–5) or when Chubby has not yet submitted a request in the current simulated day.

### 5.8 The decision call

Each tick builds a prompt with: Chubby's short persona, current needs, energy and mood, the time of day, active event, recent actions and their outcomes, unread message count (plus any newly read message content from the previous tick), structures he has, and the allowed categories. The model must return JSON matching this schema (built fresh each tick):

```json
{
  "thought": "one or two sentences in Chubby's voice",
  "intent": "what he wants to do, in his own words, and why",
  "category": "<enum of currently allowed categories>",
  "read_messages": false,
  "diary_entry": "optional, only when something meaningful happened",
  "structure": { "name": "", "purpose": "", "why": "", "priority": 1 }
}
```

* `structure` is optional and only offered in the schema during genesis (ticks 1–5) or when the per-simulated-day quota (1 request per 144 ticks) permits.
* `read_messages` is only offered in the schema when unread messages exist (`read = 0`).
* **Two-step `read_messages` flow:** When Chubby sets `"read_messages": true` on tick `T`, the engine marks those unread messages as `read = 1`, includes their text in tick `T`'s outcome, and pins the read message text into tick `T+1`'s prompt context so he reacts directly to it in his next thought and diary entry.

The referee then resolves the intent in a second call (see 5.3). Settings: `temperature` around `0.9`, `"think": false` for routine ticks, `"keep_alive": "15m"` to keep the model warm in VRAM across 30-second ticks, up to 2 retries on invalid output, and a safe fallback intent (`rest` nearby) if all fail. Every failure is logged.

### 5.9 Persona seed

Chubby is a small, round, slightly clumsy robot who is curious, easily delighted, and a bit anxious when his battery gets low. He speaks in short, warm sentences. He knows he is a robot living in a small, quiet scrapyard town, and he has no job except looking after himself. This is deliberately light: the behavior should come from the needs system, not from a long prompt.

---

## 6. Database tables

* `state`: `tick, sim_time, battery, curiosity, joy, calm, belonging, mood_modifier, energy, mood`
* `actions`: `id, tick, action, detail, thought, outcome_json, state_after_json`
* `events`: `id, tick, type, payload`
* `messages`: `id, text, received_tick, read` (`0` or `1`)
* `diary`: `id, tick, sim_time, text`
* `memories`: `id, tick_range, summary`
* `requests`: `id, tick, name, purpose, why, priority, status, decision_note`
* `structures`: `id, name, unlocks, created_tick`

---

## 7. Build roadmap (milestone-based phases)

### Phase 1: Environment & Ollama Verification

* Install/verify Python, Ollama, and `qwen3:4b`; initialize project and git repo.
* Verify Ollama returns valid schema-constrained JSON with `"think": false` for a toy prompt.
* **Done when:** a script prints a valid JSON object from the model.

### Phase 2: Core State Engine & Referee Rules (Without LLM)

* Build `db.py`, `state.py`, and deterministic rules/caps in `referee.py`, including decay, day/night clock, and `rest`/`recharge` decay exemption.
* Run 2,000 ticks with a random policy and print/inspect needs and mood.
* **Done when:** needs stay bounded in `[0.0, 1.0]` and no value gets permanently stuck at `0` or `1`.

### Phase 3: LLM Decision Loop & Referee Resolution

* Build `llm.py` and `prompt.py` with dynamic Pydantic JSON schemas, retries, and fallback, plus the second LLM call in `referee.py`.
* Run 100 ticks and inspect logs.
* **Done when:** at least 99 of 100 outputs are valid and sensible.

### Phase 4: Anti-Repetition & External Events

* Build `antirepeat.py` and `events.py`; wire in category blocking, `mood_modifier` penalties/bonuses, Jaccard similarity, and per-tick event rolls.
* **Done when:** longest same-category streak is 2 or fewer in a 200-tick run.

### Phase 5: Genesis & Infrastructure Workflow

* Build `genesis.py` and the CLI `requests` / `approve` / `reject` commands in `cli.py`; enforce the schema-level 1-request-per-simulated-day limit after tick 5.
* **Done when:** Chubby requests structures during Genesis, approvals add rows to `structures`, and approved structures influence Referee outcomes.

### Phase 6: Diary, Two-Step Message Inbox & Grounded Memory

* Build `memory.py`, `cli.py leave "text"`, the two-step `read_messages` context injection, and the 50-tick summarization job.
* **Done when:** Chubby sees unread message count, sets `read_messages: true`, reacts to the message content in the next diary entry, and references only logged facts.

### Phase 7: Soak Test, Metrics Report & Iteration Review

* Build `report.py`; run the 24-hour soak test at the 30-second tick interval and evaluate against the Section 2 success-criteria table.
* Read the full diary like a reader would, tune constants in `config.toml` if needed, and record findings for Iteration 2.

---

## 8. Testing approach

* **Unit tests** for the effects math, clamping, decay (including `rest`/`recharge` decay exemption), penalty and blocking logic, and similarity function
* **Replay tests:** save a fixed seed and a mocked LLM so the same run reproduces for debugging
* **Schema tests:** confirm the enum always excludes blocked actions, includes `read_messages` only when messages are unread, and includes `structure` only during Genesis or when daily request quota permits
* **Soak test:** the 24-hour run at the 30-second default tick interval in Phase 7, followed by `report.py`
* **Human read-through:** the real test of whether it feels alive

---

## 9. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| The model finds a favorite loop anyway | Tune penalty strength, widen the block window, add more events and world structures |
| Invalid JSON or off-schema output | Schema-constrained decoding, retries, fallback action |
| Model invents events or memories | Grounded context only, explicit instruction, log-based facts |
| Mood spirals to the bottom or sticks at the top | Clamp the modifier, decay it, add gentle recovery effects, and allow solo belonging recovery |
| Voice becomes bland | Try a larger or different model, vary the diary prompt, keep the persona short |
| Machine too slow | Smaller model, longer tick interval via `--interval`, disable thinking mode (`"think": false`), keep model warm (`"keep_alive": "15m"`) |
| Over-engineering | Keep to the out-of-scope list until the success criteria are met |
| Referee inflates rewards (every intent is wonderful) | Hard per-tick caps, noise, a stricter referee prompt, and checking each outcome against logged world state |

---

## 10. Definition of done

Iteration 1 is complete when:

1. The 24-hour soak test meets the success-criteria table
2. Chubby has requested, and you have reviewed, at least three structures
3. You can point to at least one behavior or habit that was never scripted
4. The code runs from a single command, with settings in `config.toml` (30-second default tick interval, overridable via `--interval` / `--ticks`)
5. You have a short notes file listing what surprised you and what to change

---

## 11. What comes next (Iteration 2 preview)

* A simple web viewer: a status panel, a "right now" card and a diary page
* Per-visitor memory and promises
* A basic 3D or 2D scene with a procedural Chubby and a day and night cycle
* A "leave him something" form with moderation

