"""
Command-line entrypoint.

  Single shot:   python -m src.main "songs for late-night studying"
  Interactive:   python -m src.main
"""

from __future__ import annotations

import sys
from pathlib import Path

from .agent import Agent
from .data import load_songs


CATALOG = Path(__file__).resolve().parent.parent / "data" / "songs.csv"


def _print_result(result) -> None:
    print()
    print("=" * 72)
    print(f"Request: {result.request}")
    print(f"Backend: {result.backend_used}")
    print()
    if not result.ok:
        print(f"[blocked] {result.error}")
        for note in result.guardrail_notes:
            print(f"  - {note}")
        return

    print(result.explanation)
    if result.guardrail_notes:
        print()
        for note in result.guardrail_notes:
            print(f"  ! {note}")

    print()
    print("Recommendations:")
    for i, r in enumerate(result.recommendations, 1):
        reasons = "; ".join(r.reasons) if r.reasons else "—"
        print(f"  {i}. {r.song.short():<40}  score={r.score:>6.2f}  ({reasons})")


def main() -> int:
    songs = load_songs(CATALOG)
    agent = Agent(songs)

    if len(sys.argv) > 1:
        request = " ".join(sys.argv[1:])
        _print_result(agent.plan(request))
        return 0

    print(f"MoodDJ ready. Catalog: {len(songs)} songs. Backend: {agent.backend.name}.")
    print("Type a request (or 'quit' to exit).")
    while True:
        try:
            request = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if request.lower() in {"quit", "exit", "q"}:
            return 0
        if not request:
            continue
        _print_result(agent.plan(request))


if __name__ == "__main__":
    sys.exit(main())
