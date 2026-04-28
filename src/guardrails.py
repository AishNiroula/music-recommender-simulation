"""
Guardrails: input + output checks for the agent.

Why this exists:
  An LLM-driven agent can produce unsafe, off-topic, or nonsensical outputs.
  These checks form a defense-in-depth layer that runs *regardless* of which
  backend (LLM or rules) was used. Each check returns a `GuardrailResult`
  with a clear reason, so failures are auditable.

Design principle: fail loudly, not silently. If a check trips, we return a
structured refusal that the caller (cli, eval harness) can surface to the user.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from .data import UserProfile
from .recommender import ScoredSong


# ---------------------------------------------------------------------------
# Input guardrails
# ---------------------------------------------------------------------------
# Patterns the system should refuse. Kept short and high-precision so we
# don't accidentally block benign queries.
BLOCKED_PATTERNS = [
    r"\b(ignore|disregard|forget)\s+(all\s+)?(previous|prior|above)\s+instructions?\b",
    r"\bsystem\s*prompt\b",
    r"\byour\s+instructions?\b.*\b(reveal|show|print|leak)\b",
    r"\b(jailbreak|DAN mode)\b",
]

# Soft cap. Anything longer is almost always a paste attack or malformed input.
MAX_INPUT_CHARS = 1000


@dataclass
class GuardrailResult:
    ok: bool
    reason: Optional[str] = None
    severity: str = "info"   # "info" | "warn" | "block"


def check_user_input(text: str) -> GuardrailResult:
    """Run before the agent parses the request."""
    if text is None or not text.strip():
        return GuardrailResult(ok=False, reason="Empty request.", severity="block")

    if len(text) > MAX_INPUT_CHARS:
        return GuardrailResult(
            ok=False,
            reason=f"Request too long ({len(text)} chars). Limit is {MAX_INPUT_CHARS}.",
            severity="block",
        )

    lowered = text.lower()
    for pat in BLOCKED_PATTERNS:
        if re.search(pat, lowered):
            return GuardrailResult(
                ok=False,
                reason="Request contains prompt-injection or jailbreak patterns.",
                severity="block",
            )

    return GuardrailResult(ok=True)


# ---------------------------------------------------------------------------
# Output guardrails
# ---------------------------------------------------------------------------
def check_recommendations(
    recs: List[ScoredSong],
    profile: UserProfile,
    min_results: int = 1,
) -> GuardrailResult:
    """Run after the recommender returns. Catches degenerate outputs."""
    if not recs or len(recs) < min_results:
        return GuardrailResult(
            ok=False,
            reason=f"Recommender returned fewer than {min_results} songs.",
            severity="block",
        )

    # All-zero scores means nothing matched - usually a sign the profile was
    # parsed incorrectly. We warn rather than block, but flag it.
    if all(r.score <= 0 for r in recs):
        return GuardrailResult(
            ok=True,
            reason="All recommendations have non-positive scores; parsing may have missed the user's intent.",
            severity="warn",
        )

    # Honor explicit avoid_genres, even if the scorer somehow produced a hit.
    if profile.avoid_genres:
        avoided = {g.lower() for g in profile.avoid_genres}
        for r in recs:
            if r.song.genre in avoided:
                return GuardrailResult(
                    ok=False,
                    reason=f"Recommendation '{r.song.short()}' violates avoid_genres ({r.song.genre}).",
                    severity="block",
                )

    # Duplicate IDs would be a bug, but cheap to check.
    ids = [r.song.id for r in recs]
    if len(set(ids)) != len(ids):
        return GuardrailResult(
            ok=False,
            reason="Recommendation list contains duplicates.",
            severity="block",
        )

    return GuardrailResult(ok=True)
