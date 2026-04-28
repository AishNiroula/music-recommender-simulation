"""
Agent: parses a natural-language request into a UserProfile.

Two backends are supported, selected automatically:

  1. LLMBackend   - uses the Anthropic API if ANTHROPIC_API_KEY is set
  2. RulesBackend - keyword-matching fallback that runs offline

Both implement the same interface: parse(text) -> UserProfile.

The wrapper (Agent.plan) adds:
  - input guardrails
  - JSON validation when the LLM is used
  - graceful fallback from LLM -> rules if anything goes wrong
  - retrieval + scoring orchestration
  - output guardrails
  - a final natural-language explanation

This 'plan -> act -> verify' shape is the agentic-workflow piece of the project.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Protocol

from .data import Song, UserProfile
from .guardrails import GuardrailResult, check_recommendations, check_user_input
from .recommender import (
    ScoredSong,
    ScoringWeights,
    retrieve_similar,
    score_songs,
)


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------
class ParserBackend(Protocol):
    name: str
    def parse(self, text: str) -> UserProfile: ...


# ---------------------------------------------------------------------------
# Rules backend (always available, runs offline)
# ---------------------------------------------------------------------------
# Vocabulary tables. Keep these explicit so behavior is easy to audit.
_GENRE_KEYWORDS = {
    "lo-fi":      ["lo-fi", "lofi", "lo fi", "chillhop"],
    "ambient":    ["ambient", "atmospheric", "soundscape"],
    "jazz":       ["jazz", "bebop", "swing"],
    "indie":      ["indie", "indie rock", "indie pop"],
    "pop":        ["pop", "top 40"],
    "hip-hop":    ["hip hop", "hip-hop", "rap"],
    "electronic": ["electronic", "edm", "techno", "house", "synth"],
    "rock":       ["rock", "alt rock", "alternative"],
    "country":    ["country", "americana"],
    "folk":       ["folk", "acoustic"],
    "r&b":        ["r&b", "rnb", "soul"],
    "classical":  ["classical", "orchestra", "symphony", "string quartet"],
}

_MOOD_KEYWORDS = {
    "calm":       ["calm", "chill", "relax", "relaxing", "mellow", "peaceful", "soothing"],
    "happy":      ["happy", "upbeat", "cheerful", "feel-good", "feel good", "joyful", "sunny"],
    "energetic":  ["energetic", "hype", "pumped", "intense", "high energy", "workout"],
    "melancholy": ["sad", "melancholy", "blue", "moody", "somber", "heartbroken"],
    "romantic":   ["romantic", "love", "date night"],
    "focused":    ["focus", "focused", "concentration", "deep work"],
    "chill":      ["chill"],
}

# Activities suggest energy/valence/tempo defaults.
_ACTIVITY_HINTS = {
    "studying":  {"target_energy": 0.25, "target_valence": 0.55, "tempo_range": (60, 100), "moods": ["focused", "calm"]},
    "study":     {"target_energy": 0.25, "target_valence": 0.55, "tempo_range": (60, 100), "moods": ["focused", "calm"]},
    "working":   {"target_energy": 0.30, "target_valence": 0.55, "tempo_range": (60, 110), "moods": ["focused"]},
    "workout":   {"target_energy": 0.90, "target_valence": 0.75, "tempo_range": (120, 160), "moods": ["energetic"]},
    "running":   {"target_energy": 0.85, "target_valence": 0.75, "tempo_range": (140, 170), "moods": ["energetic"]},
    "sleeping":  {"target_energy": 0.10, "target_valence": 0.50, "tempo_range": (50, 75),  "moods": ["calm"]},
    "driving":   {"target_energy": 0.65, "target_valence": 0.70, "tempo_range": (100, 130), "moods": ["happy"]},
    "party":     {"target_energy": 0.90, "target_valence": 0.85, "tempo_range": (120, 140), "moods": ["energetic", "happy"]},
    "cooking":   {"target_energy": 0.55, "target_valence": 0.75, "tempo_range": (90, 120),  "moods": ["happy"]},
}

# Time-of-day hints. Composes with activities.
_TIME_HINTS = {
    "late night":   {"target_energy": 0.25, "target_valence": 0.50},
    "late-night":   {"target_energy": 0.25, "target_valence": 0.50},
    "midnight":     {"target_energy": 0.20, "target_valence": 0.45},
    "morning":      {"target_energy": 0.55, "target_valence": 0.70},
    "afternoon":    {"target_energy": 0.50, "target_valence": 0.65},
    "evening":      {"target_energy": 0.45, "target_valence": 0.60},
}


@dataclass
class RulesBackend:
    name: str = "rules"

    def parse(self, text: str) -> UserProfile:
        t = text.lower()
        profile = UserProfile()

        # Genres
        for genre, kws in _GENRE_KEYWORDS.items():
            if any(kw in t for kw in kws):
                profile.genres.append(genre)

        # Moods
        for mood, kws in _MOOD_KEYWORDS.items():
            if any(kw in t for kw in kws):
                profile.moods.append(mood)

        # "no X" / "not X" / "without X" -> avoid_genres
        for genre, kws in _GENRE_KEYWORDS.items():
            for kw in kws:
                if any(neg in t for neg in [f"no {kw}", f"not {kw}", f"without {kw}", f"avoid {kw}"]):
                    profile.avoid_genres.append(genre)
                    if genre in profile.genres:
                        profile.genres.remove(genre)

        # Activities
        for activity, hints in _ACTIVITY_HINTS.items():
            if activity in t:
                profile.activity = activity
                profile.target_energy = hints["target_energy"]
                profile.target_valence = hints["target_valence"]
                profile.tempo_range = hints["tempo_range"]
                for m in hints["moods"]:
                    if m not in profile.moods:
                        profile.moods.append(m)
                break  # first activity wins

        # Time of day adjusts energy/valence (overrides activity defaults if more specific)
        for phrase, hints in _TIME_HINTS.items():
            if phrase in t:
                profile.target_energy = hints["target_energy"]
                profile.target_valence = hints["target_valence"]

        return profile


# ---------------------------------------------------------------------------
# LLM backend (used when ANTHROPIC_API_KEY is set)
# ---------------------------------------------------------------------------
_LLM_SYSTEM_PROMPT = """You convert a user's natural-language music request
into a JSON taste profile. Respond with ONLY a JSON object, no prose.

Schema:
{
  "genres":         array of strings from {lo-fi, ambient, jazz, indie, pop, hip-hop, electronic, rock, country, folk, r&b, classical},
  "moods":          array of strings from {calm, happy, energetic, melancholy, romantic, focused, chill},
  "target_energy":  number 0.0-1.0 or null,
  "target_valence": number 0.0-1.0 or null,
  "tempo_range":    [low_bpm, high_bpm] or null,
  "avoid_genres":   array of genre strings,
  "activity":       short string or null
}

Rules:
- Output JSON only.
- Leave a field null/empty if the user didn't imply it.
- "studying" -> low energy, focused mood, 60-100 BPM.
- "workout" -> high energy, 120-160 BPM.
- "late night" -> low energy."""


@dataclass
class LLMBackend:
    name: str = "llm"
    model: str = "claude-sonnet-4-5"
    _client: object = None

    def __post_init__(self):
        try:
            import anthropic  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "LLMBackend requires the `anthropic` package. "
                "Install it with `pip install anthropic`, or unset "
                "ANTHROPIC_API_KEY to use the rules backend."
            ) from e
        self._client = anthropic.Anthropic()

    def parse(self, text: str) -> UserProfile:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=400,
            system=_LLM_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text}],
        )
        # Pull text out of the response
        raw = "".join(
            block.text for block in msg.content if getattr(block, "type", "") == "text"
        ).strip()

        # Strip code fences if the model added any
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        data = json.loads(raw)  # raises -> Agent.plan catches and falls back
        return UserProfile(
            genres=data.get("genres") or [],
            moods=data.get("moods") or [],
            target_energy=data.get("target_energy"),
            target_valence=data.get("target_valence"),
            tempo_range=tuple(data["tempo_range"]) if data.get("tempo_range") else None,
            avoid_genres=data.get("avoid_genres") or [],
            activity=data.get("activity"),
        )


# ---------------------------------------------------------------------------
# Top-level Agent
# ---------------------------------------------------------------------------
@dataclass
class PlanResult:
    """Everything the caller needs to render a response or audit a run."""
    ok: bool
    request: str
    backend_used: str
    profile: UserProfile
    recommendations: List[ScoredSong] = field(default_factory=list)
    explanation: str = ""
    guardrail_notes: List[str] = field(default_factory=list)
    error: Optional[str] = None


class Agent:
    def __init__(
        self,
        songs: List[Song],
        weights: Optional[ScoringWeights] = None,
        backend: Optional[ParserBackend] = None,
    ):
        self.songs = songs
        self.weights = weights or ScoringWeights()
        self.backend = backend or self._auto_backend()

    @staticmethod
    def _auto_backend() -> ParserBackend:
        """Pick LLM if a key is present and the SDK is installed; else rules."""
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                return LLMBackend()
            except Exception:
                # SDK missing or init failed - fall back silently
                pass
        return RulesBackend()

    # -----------------------------------------------------------------------
    # The agentic plan->act->verify loop
    # -----------------------------------------------------------------------
    def plan(self, request: str, top_k: int = 5) -> PlanResult:
        # Step 1: input guardrail
        in_check = check_user_input(request)
        if not in_check.ok:
            return PlanResult(
                ok=False, request=request, backend_used="none",
                profile=UserProfile(),
                guardrail_notes=[f"input: {in_check.reason}"],
                error=in_check.reason,
            )

        # Step 2: parse with current backend, fall back on failure
        backend_used = self.backend.name
        try:
            profile = self.backend.parse(request)
        except Exception as e:
            # LLM hiccup, malformed JSON, network failure - degrade to rules
            profile = RulesBackend().parse(request)
            backend_used = f"rules (fallback from {self.backend.name}: {type(e).__name__})"

        # Step 3: score the catalog
        recs = score_songs(self.songs, profile, self.weights, top_k=top_k)

        # Step 4: output guardrail
        out_check = check_recommendations(recs, profile)
        notes: List[str] = []
        if out_check.reason:
            notes.append(f"output: {out_check.reason}")
        if not out_check.ok:
            return PlanResult(
                ok=False, request=request, backend_used=backend_used,
                profile=profile, recommendations=recs,
                guardrail_notes=notes, error=out_check.reason,
            )

        # Step 5: explain
        explanation = self._explain(profile, recs, backend_used)

        return PlanResult(
            ok=True, request=request, backend_used=backend_used,
            profile=profile, recommendations=recs,
            explanation=explanation, guardrail_notes=notes,
        )

    # Convenience: retrieval-only mode (the 'Module 4' surface)
    def retrieve(self, request: str, top_k: int = 10) -> PlanResult:
        in_check = check_user_input(request)
        if not in_check.ok:
            return PlanResult(
                ok=False, request=request, backend_used="none",
                profile=UserProfile(), guardrail_notes=[f"input: {in_check.reason}"],
                error=in_check.reason,
            )
        try:
            profile = self.backend.parse(request)
            backend_used = self.backend.name
        except Exception as e:
            profile = RulesBackend().parse(request)
            backend_used = f"rules (fallback: {type(e).__name__})"

        recs = retrieve_similar(self.songs, profile, top_k=top_k)
        return PlanResult(
            ok=True, request=request, backend_used=backend_used,
            profile=profile, recommendations=recs,
            explanation=f"Top {top_k} by cosine similarity to parsed profile.",
        )

    # -----------------------------------------------------------------------
    @staticmethod
    def _explain(profile: UserProfile, recs: List[ScoredSong], backend: str) -> str:
        bits: List[str] = []
        bits.append(f"Parsed via [{backend}].")
        if profile.activity:
            bits.append(f"Activity: {profile.activity}.")
        if profile.genres:
            bits.append(f"Genres: {', '.join(profile.genres)}.")
        if profile.moods:
            bits.append(f"Moods: {', '.join(profile.moods)}.")
        if profile.target_energy is not None:
            bits.append(f"Target energy ~{profile.target_energy:.2f}.")
        if profile.tempo_range:
            bits.append(f"Tempo {profile.tempo_range[0]}-{profile.tempo_range[1]} BPM.")
        if profile.avoid_genres:
            bits.append(f"Avoiding: {', '.join(profile.avoid_genres)}.")
        bits.append(f"Returning {len(recs)} song(s) ranked by weighted feature match.")
        return " ".join(bits)
