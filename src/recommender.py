"""
Recommender: scores songs against a UserProfile.

This is the Module 1-3 core, evolved. It now supports two modes:

  1. score_songs():       weighted feature-match scoring (transparent, rule-based)
  2. retrieve_similar():  cosine-similarity retrieval over a feature vector
                          (the lightweight Module 4 'retrieval' component)

The agent layer (agent.py) decides which mode to call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from .data import Song, UserProfile


# ---------------------------------------------------------------------------
# Configurable weights. Exposed as a dataclass so experiments can sweep them.
# ---------------------------------------------------------------------------
@dataclass
class ScoringWeights:
    genre: float = 2.0
    mood: float = 2.0
    energy: float = 1.5
    valence: float = 1.0
    tempo: float = 1.0
    avoid_penalty: float = 5.0   # subtracted if song matches an avoid_genre


@dataclass
class ScoredSong:
    song: Song
    score: float
    reasons: List[str]


# ---------------------------------------------------------------------------
# Mode 1: Transparent weighted scoring
# ---------------------------------------------------------------------------
def score_song(song: Song, profile: UserProfile, w: ScoringWeights) -> ScoredSong:
    """Score one song. Each rule contributes a small explanation string,
    which the agent can later surface to the user."""
    score = 0.0
    reasons: List[str] = []

    if profile.genres and song.genre in {g.lower() for g in profile.genres}:
        score += w.genre
        reasons.append(f"genre match ({song.genre})")

    if profile.moods and song.mood in {m.lower() for m in profile.moods}:
        score += w.mood
        reasons.append(f"mood match ({song.mood})")

    if profile.target_energy is not None:
        # Closeness on [0,1]. Perfect match -> +w.energy, opposite -> 0.
        diff = abs(song.energy - profile.target_energy)
        score += w.energy * max(0.0, 1.0 - diff)
        if diff < 0.15:
            reasons.append(f"energy near target ({song.energy:.2f})")

    if profile.target_valence is not None:
        diff = abs(song.valence - profile.target_valence)
        score += w.valence * max(0.0, 1.0 - diff)
        if diff < 0.15:
            reasons.append(f"valence near target ({song.valence:.2f})")

    if profile.tempo_range is not None:
        lo, hi = profile.tempo_range
        if lo <= song.tempo <= hi:
            score += w.tempo
            reasons.append(f"tempo in range ({song.tempo} BPM)")

    if profile.avoid_genres and song.genre in {g.lower() for g in profile.avoid_genres}:
        score -= w.avoid_penalty
        reasons.append(f"penalty: avoided genre ({song.genre})")

    return ScoredSong(song=song, score=round(score, 3), reasons=reasons)


def score_songs(
    songs: List[Song],
    profile: UserProfile,
    weights: Optional[ScoringWeights] = None,
    top_k: int = 5,
    diversify: bool = True,
) -> List[ScoredSong]:
    """Score the full catalog, sort, and optionally enforce artist diversity
    so the same artist doesn't dominate the top results."""
    weights = weights or ScoringWeights()
    scored = [score_song(s, profile, weights) for s in songs]
    scored.sort(key=lambda x: x.score, reverse=True)

    if not diversify:
        return scored[:top_k]

    seen_artists: set = set()
    out: List[ScoredSong] = []
    for s in scored:
        if s.song.artist in seen_artists:
            continue
        out.append(s)
        seen_artists.add(s.song.artist)
        if len(out) >= top_k:
            break
    # Backfill if diversity filter left us short
    if len(out) < top_k:
        for s in scored:
            if s not in out:
                out.append(s)
            if len(out) >= top_k:
                break
    return out


# ---------------------------------------------------------------------------
# Mode 2: Vector retrieval (the 'Module 4' retrieval layer)
# ---------------------------------------------------------------------------
# Each song is mapped to a small numeric vector in a shared space.
# We use cosine similarity rather than Euclidean so magnitude doesn't dominate.

def _song_vector(song: Song) -> List[float]:
    return [song.energy, song.valence, song.tempo / 200.0]


def _profile_vector(profile: UserProfile) -> Optional[List[float]]:
    """Build a probe vector from the profile. Returns None if too sparse."""
    if profile.target_energy is None and profile.target_valence is None and profile.tempo_range is None:
        return None
    energy = profile.target_energy if profile.target_energy is not None else 0.5
    valence = profile.target_valence if profile.target_valence is not None else 0.5
    if profile.tempo_range is not None:
        lo, hi = profile.tempo_range
        tempo = ((lo + hi) / 2.0) / 200.0
    else:
        tempo = 0.5
    return [energy, valence, tempo]


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def retrieve_similar(
    songs: List[Song],
    profile: UserProfile,
    top_k: int = 10,
) -> List[ScoredSong]:
    """Cosine-similarity retrieval over the song catalog.

    This is intentionally simple (3-dim feature vector) so behavior is easy
    to inspect. In a production system this would be replaced with
    embeddings + a vector DB, but the interface stays the same.
    """
    probe = _profile_vector(profile)
    if probe is None:
        # Nothing to retrieve against -> fall back to a neutral preference
        probe = [0.5, 0.5, 0.5]

    scored: List[ScoredSong] = []
    for song in songs:
        sim = _cosine(_song_vector(song), probe)
        reasons = [f"cosine similarity = {sim:.3f}"]
        scored.append(ScoredSong(song=song, score=round(sim, 4), reasons=reasons))
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_k]
