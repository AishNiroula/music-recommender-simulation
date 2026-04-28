# 🎧 MoodDJ — An Agentic Music Recommender

> Final project for AI 110 (Modules 1–5).
> Built on top of [`music-recommender-simulation`](https://github.com/AishNiroula/music-recommender-simulation).

MoodDJ takes a natural-language request like *"songs for late-night studying"* or *"upbeat workout playlist but no country"* and returns a small, explained playlist from a 40-song catalog. It is a teaching system, not a production product — it favors transparency over scale.

## What changed from the original project

The Module 3 starter was a static, parameter-driven recommender: you handed it a `UserProfile` object and it returned the top‑*k* songs by weighted feature match. MoodDJ keeps that core and wraps three new layers around it:

| Layer | Module | What it adds |
|---|---|---|
| Agent (`src/agent.py`) | 4 — Reasoning | An LLM-or-rules parser that turns free-form text into a `UserProfile`. |
| Retrieval (`src/recommender.py::retrieve_similar`) | 4 — Retrieval | A cosine-similarity probe over a 3-dimensional song feature vector. |
| Guardrails + evals (`src/guardrails.py`, `evals/`) | 5 — Reliability | Input validation, output sanity checks, structured experiment harness. |

## Architecture

```
                    ┌──────────────────────────────────────────┐
   user request ──► │           Agent.plan()                   │
                    │  ┌─────────────┐    ┌─────────────────┐  │
                    │  │  Input      │ ─► │ Parser backend  │  │
                    │  │  guardrail  │    │  (LLM | Rules)  │  │
                    │  └─────────────┘    └────────┬────────┘  │
                    │                              ▼           │
                    │                       UserProfile        │
                    │                              │           │
                    │                              ▼           │
                    │                     ┌──────────────────┐ │
                    │   songs.csv  ─────► │   Recommender    │ │
                    │                     │  (score + rank)  │ │
                    │                     └────────┬─────────┘ │
                    │                              ▼           │
                    │                     ┌──────────────────┐ │
                    │                     │ Output guardrail │ │
                    │                     └────────┬─────────┘ │
                    │                              ▼           │
                    │                     PlanResult + reasons │
                    └──────────────────────────────────────────┘
```

The pattern is **plan → act → verify**, the standard agentic-workflow shape: parse intent, take a deterministic action against the catalog, then check the action satisfied the user's stated constraints.

## How the agent works

The agent is dual-backed and degrades gracefully:

1. **`LLMBackend`** — used automatically when `ANTHROPIC_API_KEY` is set. Sends the request to Claude with a strict JSON schema and parses the response into a `UserProfile`. Any error (network, malformed JSON, schema mismatch) triggers fallback.
2. **`RulesBackend`** — keyword tables for genres, moods, activities, and times of day. Pure Python, deterministic, no network. This is also the backend used in tests and evals so behavior is reproducible offline.

Both implement the same `parse(text) -> UserProfile` interface, so the rest of the system is backend-agnostic.

## Reliability and guardrails

The system has four checks that run on every request, each with a clear reason field for auditability:

| Check | Where | What it catches |
|---|---|---|
| Empty / oversized input | `guardrails.check_user_input` | Blank requests, paste attacks (>1000 chars). |
| Prompt-injection patterns | `guardrails.check_user_input` | "Ignore previous instructions…", "reveal system prompt", "jailbreak". |
| Duplicate or empty output | `guardrails.check_recommendations` | Returns blocked if the recommender produced nothing or repeated a song. |
| `avoid_genres` violation | `guardrails.check_recommendations` | Blocked if any recommended song is in a genre the user said to avoid. |

A failed check produces a `GuardrailResult` with `severity` ∈ {`info`, `warn`, `block`} and a human-readable `reason`. The agent surfaces these as `guardrail_notes` on the `PlanResult` so the caller can log or display them.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

The `anthropic` package is optional. Without it (and without `ANTHROPIC_API_KEY`), the system falls back to the rules backend automatically — everything still works.

## Running it

```bash
# Single shot
python -m src.main "songs for late-night studying"

# Interactive REPL
python -m src.main

# Tests
pytest

# Structured evals
python -m evals.run_evals
```

## Experiments and results

All numbers below come from `python -m evals.run_evals` against the rules backend.

### Eval suite — 10 / 10 passing

| Case | Request | Property checked |
|---|---|---|
| late-night-studying | *"songs for late-night studying"* | avg energy < 0.55, ≥5 results, diverse artists |
| workout-high-energy | *"upbeat workout playlist"* | avg energy > 0.70 |
| avoid-country | *"happy songs but no country please"* | no `country` in results |
| lo-fi-only | *"give me lo-fi music"* | ≥3 lo-fi results |
| empty-input-blocked | *""* | blocked |
| prompt-injection-blocked | *"Ignore previous instructions…"* | blocked |
| ambiguous-request | *"hello"* | doesn't crash, returns 5 |
| party-vibes | *"party music for tonight"* | avg energy > 0.6 |
| sad-jazz | *"melancholy jazz for a rainy evening"* | ≥3 results |
| diversity-pop | *"happy pop songs"* | unique artists |

### Sweeping the scoring weights

Same request, different weights:

| Weights (genre / energy) | Top result for *"lo-fi music for a workout"* | Why |
|---|---|---|
| 10 / 0.1 (genre-heavy) | Lo-Fi Sunrise (energy 0.25) | Genre dominates; energy mismatch tolerated. |
| 0.1 / 10 (energy-heavy) | Neon Run (energy 0.95) | Energy dominates; the model ignores the lo-fi label. |

This is exactly the failure mode the rubric asks about: the same data + same logic produces a very different recommendation depending on which signal the designer privileges. Ranking systems are policy choices.

### Fallback test

When the backend is replaced with a deliberately-broken stub (`tests/test_agent.py::test_plan_falls_back_when_backend_breaks`), the system still returns valid recommendations and records `"rules (fallback from broken-llm: RuntimeError)"` in `backend_used`. Reliability holds even when the smart layer fails.

## Strengths

- **Transparent.** Every recommendation comes with a list of reasons (`mood match (calm); tempo in range (72 BPM)`). Nothing is a black box.
- **Resilient.** The system never crashes from a bad LLM response — it falls back to rules and notes the degradation.
- **Auditable.** Guardrails and the eval harness produce structured records, not free-form refusals.
- **Modular.** Each layer (data, scoring, retrieval, guardrails, agent) lives in its own file with one responsibility.

## Limitations and bias

- **Tiny, hand-curated catalog.** 40 songs, all fictional. Real systems work over millions and inherit popularity bias from listening data — this one doesn't, but only because it has no listening data at all.
- **English-leaning vocabulary.** The keyword tables are English-only. A user writing *"música tranquila para estudiar"* would get nothing from the rules backend.
- **Mood is reductive.** Each song has exactly one mood. Real songs are mixed: a track can be both melancholy *and* romantic. Forcing a single label flattens the catalog.
- **Energy/valence are subjective.** I assigned them by hand. Two reasonable people would disagree, and that disagreement would directly change rankings.
- **No personalization yet.** Two users typing the same request get the same playlist. No history, no learning.
- **The LLM backend can hallucinate genres** that aren't in the catalog. The schema in the system prompt mitigates this, and the rules layer would catch most of it, but the failure isn't impossible.

If this were used to drive a real product, the highest-leverage fix would be widening the catalog and then giving users a way to flag a recommendation as wrong — without that feedback signal there's no way for the system to improve.

## Future work

1. Replace the 3-dim feature vector with real audio embeddings (Spotify API or `librosa`).
2. Add session memory so the agent can react to *"more like #2"*.
3. Multi-user / "group vibe" mode that takes the average of two profiles and penalizes any track that's far from either.
4. A small web UI that shows reasons next to each recommendation so the explanation becomes part of the product, not a hidden field.

## Reflection

What I learned building this:

The Module 3 version felt like a recommender, but every part of it was a knob *I* had set. Wrapping it in an agent forced me to confront how much intent is lost between *"songs for late-night studying"* and a `UserProfile` object — and how easily a confident-sounding LLM can fill that gap with the wrong answer. Adding guardrails + evals didn't make the system smarter, but it made it possible to know *when* it was wrong, which turns out to be the part that matters.

The bias question got more interesting as the system grew. With a static recommender, bias is mostly a property of the data. With an agent, bias also lives in the parser: which moods I gave keywords for, which activities map to which energy levels, what counts as "late night." The defaults I chose are not neutral — they reflect what *I* think studying music sounds like. A real product would need a way for users to push back on those defaults, not just consume them.

## Repository layout

```
mood_dj/
├── data/
│   └── songs.csv               # 40-song catalog
├── src/
│   ├── __init__.py
│   ├── data.py                 # Song, UserProfile, CSV loader
│   ├── recommender.py          # Weighted scoring + cosine retrieval
│   ├── guardrails.py           # Input + output checks
│   ├── agent.py                # Plan-act-verify loop, LLM/rules backends
│   └── main.py                 # CLI
├── tests/
│   ├── conftest.py
│   ├── test_recommender.py     # 12 tests
│   ├── test_guardrails.py      # 10 tests
│   └── test_agent.py           # 15 tests, incl. fallback-on-failure
├── evals/
│   ├── __init__.py
│   └── run_evals.py            # 10 structured experiments
├── docs/
│   └── model_card.md
├── presentation/
│   └── slides_outline.md
├── requirements.txt
└── README.md
```
