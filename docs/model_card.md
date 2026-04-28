# 🎧 Model Card — MoodDJ

## 1. Model Name

**MoodDJ 1.0** — an agentic music recommender built on top of the Module 3 `music-recommender-simulation` starter.

## 2. Intended Use

MoodDJ takes a natural-language request (for example, *"chill jazz for a rainy evening"* or *"upbeat workout playlist but no country"*) and returns 3–5 songs from a small, hand-curated catalog, along with a short explanation of why each song was picked.

It is intended for **classroom and portfolio use only**. It is not connected to any music service, has no access to a real song catalog, and should not be used to drive recommendations to real listeners.

## 3. How It Works (Short Explanation)

The system has three stages. First, the user's request is read by an *agent* whose job is to translate the free-form text into a structured taste profile — what genres, moods, energy level, and tempo they probably want, and what they want to avoid. The agent uses a large language model when one is available and falls back to a rules-based parser when one is not, so the same code path works online and offline.

Second, that taste profile is handed to a *recommender* that walks the catalog and gives each song a numeric score. The score rewards genre and mood matches, rewards songs whose energy and tempo are close to the target, and applies a penalty if the song is in a genre the user said to avoid. The five highest scorers are returned, with one rule on top: no two recommendations can be by the same artist.

Third, before anything is shown to the user, a *guardrail* layer checks the result. It rejects empty or oversized requests, blocks prompt-injection attempts, and refuses to return recommendations that violate the user's stated avoid list. If the recommender returns nothing or returns duplicates, that's caught here too.

## 4. Data

The catalog lives in `data/songs.csv` and contains **40 songs** I wrote by hand for this project. They are not real recordings; the artist names are fictional, so the system cannot accidentally recommend something that exists.

Each song carries: a title, an artist name, a single genre label, a single mood label, an energy score (0–1), a tempo in BPM, a valence score (0–1, roughly "musical positivity"), a release year, and a language tag.

The catalog is balanced across **12 genres** (lo-fi, ambient, jazz, indie, pop, hip-hop, electronic, rock, country, folk, R&B, classical) and **7 moods** (calm, happy, energetic, melancholy, romantic, focused, chill). Years span 2017–2024. Most songs are tagged English; instrumental tracks are tagged as such.

Because I assigned every feature myself, the catalog reflects *my* sense of what each genre, mood, and energy level sound like. That is a real source of bias and the model card flags it explicitly below.

## 5. Strengths

- **Recommendations are explained.** Every song comes back with a list of reasons (e.g. *"mood match (calm); tempo in range (72 BPM)"*). The user can see exactly why a song was picked, which is rare in real recommenders.
- **The system degrades gracefully.** If the LLM is unavailable, returns malformed JSON, or hallucinates a genre, the agent falls back to the rules backend and records the fallback in the result. It never crashes from a bad model response.
- **It honors hard constraints.** When the user says "no country," the guardrail layer enforces that as a post-condition. Even if the scorer somehow ranked a country song highly, it would be filtered out before the user saw it.
- **The whole system is auditable.** Every check has a reason field; every recommendation has a score and explanations; every eval case has a clear pass/fail. There is no hidden state.

## 6. Limitations and Bias

- **Catalog is tiny and synthetic.** 40 hand-written songs cannot represent any real listening population. A real system would inherit popularity bias from listening data; this one inherits *my* taste, which is arguably worse.
- **Single-label features flatten reality.** Each song has exactly one mood and one genre. A real song can be melancholy *and* romantic, indie *and* electronic; collapsing that into one label biases the recommender toward the cleanest examples and away from interesting hybrids.
- **English-only parser.** The rules backend's keyword tables are entirely English. A request in Spanish, Hindi, or Mandarin would parse as empty even though the user's intent is perfectly clear. This is an obvious fairness gap.
- **My energy and valence judgments are not neutral.** I scored these by ear. Two reasonable people would disagree on whether a given indie track is "0.55 energy" or "0.65," and that disagreement directly changes which songs come up. The model presents these as numbers, which can mask the fact that they are opinions.
- **The "studying = low energy, slow tempo" default is cultural.** Some listeners study to fast electronic music; the system would ignore that profile and push them toward lo-fi. The activity-to-feature mapping is a strong opinion baked into code.
- **The LLM backend can hallucinate.** Even with a constrained schema, a language model can return a genre name that isn't in the catalog or pick wildly off-target energy values. The rules layer catches most of these but not all.

If this were a real product, two of these would be unacceptable: the English-only parser (an accessibility failure) and the activity defaults (a personalization failure that would make the system feel wrong to anyone whose habits don't match mine).

## 7. Evaluation

Evaluation is done two ways:

**Unit tests (`pytest`, 37 cases).** These cover the data invariants (no duplicate IDs, features in valid ranges), scoring properties (changing a weight changes the ranking, avoid-genre penalty is applied, top-*k* is sorted), guardrail behavior (empty/oversized/injection inputs are blocked), and the agent's plan-act-verify loop, including a deliberately-broken backend to prove the fallback works.

**Eval harness (`evals/run_evals.py`, 10 cases).** Each case is a (request, assertion-set) pair. The assertions are composable predicates over the agent's `PlanResult` — for example, *"average energy of the returned songs must be below 0.55"* for the late-night studying request, or *"no song in the result has genre `country`"* for an avoid-country request. The harness prints a pass/fail table and exits non-zero on any failure, so it can be wired into CI.

Both currently pass at 100%. The eval harness is intentionally easy to extend: adding a case is two lines plus a list of assertion factories.

I also did one *qualitative* check: I tried each of the eval requests by hand in the CLI and read the explanations out loud. For all 10 the explanations made sense and matched the request — there were no cases where the system returned a song I would have called clearly wrong.

## 8. Future Work

- Use real audio features (via the Spotify Web API or `librosa`) instead of hand-assigned numbers, so the catalog can be much larger and the energy/valence numbers stop being subjective.
- Add session memory so the agent can handle follow-ups like *"more like #2"* or *"too slow, give me something with more energy."*
- Translate the keyword tables (or use the LLM as a translator) so non-English requests work with the rules backend too.
- Multi-user / "group vibe" mode that combines two profiles and penalizes songs that are far from either.
- A simple web UI that shows the reasons next to each recommendation, so the explanation becomes a feature instead of a hidden field.
- Replace the keyword-based prompt-injection check with a small classifier and add adversarial test cases to the eval harness.

## 9. Personal Reflection

The biggest surprise was how much of the "intelligence" in a recommender lives in defaults that I chose without thinking about them. Mapping *"studying"* to *"low energy, 60–100 BPM, focused/calm mood"* feels like common sense until you remember that plenty of people study to drum-and-bass. Once I noticed that, I started seeing similar choices everywhere in the code — which moods got keywords, which activities were even on the list, what counted as "late night." None of those are technical decisions; they're product decisions that I made silently while writing tables.

Building the agent layer on top of the Module 3 starter made me think about reliability differently than I had before. With just a scoring function, "wrong" mostly meant "the weights need tuning." With an LLM in the loop, "wrong" can mean the model invented a genre, ignored a constraint, or confidently produced a profile that has nothing to do with what the user asked. The guardrails and evals don't fix any of that — they just make it possible to *notice* when it happens, and that turns out to be the part that lets you trust the system at all.

Where human judgment still matters, even with a smarter model: deciding what the system should refuse, deciding what counts as a fair distribution of recommendations across genres and artists, and deciding what to do when the user's request is ambiguous. Those are not problems the model can solve by being better; they are problems someone has to take responsibility for.
