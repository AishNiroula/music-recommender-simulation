"""
Data layer: Song and UserProfile.

Carries forward the Module 1-2 foundation. Both objects are intentionally small
and inspectable so the recommender's behavior stays explainable end-to-end.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class Song:
    """A single track in the catalog.

    Numeric features (energy, valence) are normalized to [0, 1].
    Tempo is in BPM. Year and language are kept for diversity-aware ranking.
    """
    id: str
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo: int
    valence: float
    year: int
    language: str

    def short(self) -> str:
        return f"{self.title} - {self.artist}"


@dataclass
class UserProfile:
    """A structured taste profile, either explicit or parsed by the agent.

    Any field can be left blank (None / empty list) and the recommender
    will simply skip it during scoring.
    """
    genres: List[str] = field(default_factory=list)
    moods: List[str] = field(default_factory=list)
    target_energy: Optional[float] = None      # 0.0 - 1.0
    target_valence: Optional[float] = None     # 0.0 - 1.0
    tempo_range: Optional[tuple] = None        # (low_bpm, high_bpm)
    avoid_genres: List[str] = field(default_factory=list)
    activity: Optional[str] = None             # free-form, e.g. "studying"

    def is_empty(self) -> bool:
        return not any([
            self.genres, self.moods, self.avoid_genres,
            self.target_energy is not None,
            self.target_valence is not None,
            self.tempo_range is not None,
            self.activity,
        ])


def load_songs(path: str | Path) -> List[Song]:
    """Load the catalog from a CSV. Type-coerces numeric columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Song catalog not found: {path}")

    songs: List[Song] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append(Song(
                id=row["id"],
                title=row["title"],
                artist=row["artist"],
                genre=row["genre"].lower(),
                mood=row["mood"].lower(),
                energy=float(row["energy"]),
                tempo=int(row["tempo"]),
                valence=float(row["valence"]),
                year=int(row["year"]),
                language=row["language"].lower(),
            ))
    return songs
