"""
Tests for the recommender and scoring layer.

These cover the Module 1-3 invariants: data loads, scoring is monotonic in
its weights, diversity dedupes by artist, retrieval ranks by similarity.
"""

from pathlib import Path

import pytest

from src.data import Song, UserProfile, load_songs
from src.recommender import (
    ScoringWeights,
    retrieve_similar,
    score_song,
    score_songs,
)


CATALOG = Path(__file__).resolve().parent.parent / "data" / "songs.csv"


@pytest.fixture(scope="module")
def songs():
    return load_songs(CATALOG)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def test_catalog_loads_and_is_nonempty(songs):
    assert len(songs) >= 30
    assert all(isinstance(s, Song) for s in songs)


def test_features_are_in_valid_ranges(songs):
    for s in songs:
        assert 0.0 <= s.energy <= 1.0, f"{s.id} energy out of range"
        assert 0.0 <= s.valence <= 1.0, f"{s.id} valence out of range"
        assert 30 <= s.tempo <= 220, f"{s.id} tempo out of range"
        assert 1900 <= s.year <= 2030


def test_song_ids_are_unique(songs):
    ids = [s.id for s in songs]
    assert len(set(ids)) == len(ids)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def test_genre_match_increases_score(songs):
    profile = UserProfile(genres=["lo-fi"])
    w = ScoringWeights()
    lofi = next(s for s in songs if s.genre == "lo-fi")
    pop = next(s for s in songs if s.genre == "pop")
    assert score_song(lofi, profile, w).score > score_song(pop, profile, w).score


def test_avoid_penalty_applies(songs):
    profile = UserProfile(avoid_genres=["pop"])
    pop = next(s for s in songs if s.genre == "pop")
    res = score_song(pop, profile, ScoringWeights())
    assert res.score < 0
    assert any("penalty" in r for r in res.reasons)


def test_top_k_returns_correct_count(songs):
    profile = UserProfile(genres=["lo-fi"], moods=["calm"])
    recs = score_songs(songs, profile, top_k=3)
    assert len(recs) == 3


def test_top_k_results_are_sorted(songs):
    profile = UserProfile(genres=["pop"], target_energy=0.85)
    recs = score_songs(songs, profile, top_k=5)
    scores = [r.score for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_diversify_dedupes_artists(songs):
    profile = UserProfile(genres=["pop"])
    recs = score_songs(songs, profile, top_k=5, diversify=True)
    artists = [r.song.artist for r in recs]
    assert len(set(artists)) == len(artists)


def test_empty_profile_does_not_crash(songs):
    profile = UserProfile()
    recs = score_songs(songs, profile, top_k=3)
    assert len(recs) == 3
    assert all(r.score == 0.0 for r in recs)


def test_weight_change_changes_ranking(songs):
    profile = UserProfile(genres=["lo-fi"], target_energy=0.9)
    w_genre_heavy = ScoringWeights(genre=10.0, energy=0.1)
    w_energy_heavy = ScoringWeights(genre=0.1, energy=10.0)
    top_genre = score_songs(songs, profile, w_genre_heavy, top_k=1)[0]
    top_energy = score_songs(songs, profile, w_energy_heavy, top_k=1)[0]
    assert top_genre.song.genre == "lo-fi"
    # With energy-heavy weights, the top song should have high energy
    assert top_energy.song.energy >= 0.7


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def test_retrieval_returns_sorted_by_similarity(songs):
    profile = UserProfile(target_energy=0.9, target_valence=0.85, tempo_range=(125, 145))
    recs = retrieve_similar(songs, profile, top_k=10)
    sims = [r.score for r in recs]
    assert sims == sorted(sims, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in sims)


def test_retrieval_top_match_is_high_energy(songs):
    profile = UserProfile(target_energy=0.95, target_valence=0.85, tempo_range=(130, 140))
    top = retrieve_similar(songs, profile, top_k=1)[0]
    assert top.song.energy >= 0.7
