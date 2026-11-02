"""Checks your Gemini key and pre-generates the AI notes for the sample team on every
match (English + Hinglish), saving them to data/ai_cache.json.

Why: the free Gemini tier is rate-limited. With the cache committed to GitHub, the
default demo is instant for every reviewer and never waits on the API.

Run:  .venv\\Scripts\\python pregenerate.py
"""
import sys
import time
import tomllib
from pathlib import Path

from matchlens import ai
from matchlens.contest import Contest
from matchlens.engine import list_matches, load_match
from matchlens.facts import catch_up_facts, preset_team, step_facts

CHUNK = 10  # must match app.py

secrets = tomllib.loads((Path(__file__).parent / ".streamlit" / "secrets.toml").read_text(encoding="utf-8"))
ai.configure(secrets.get("GEMINI_API_KEY"))
if not ai.available():
    sys.exit("No key found. Paste your key into .streamlit/secrets.toml first.")

print("Testing your Gemini key...")
if ai.catch_up({"from": "test", "to": "test", "note": "reply with: ok"}, "en") is None:
    sys.exit(f"Gemini call failed: {ai.last_error}")
print(f"OK - using model {ai._working_model}\n")

for m in list_matches():
    contest = Contest.build(load_match(m["id"]))
    team = preset_team(contest)
    mm = contest.my_mult(team["picks"], team["captain"], team["vice"])
    tl = contest.timeline(mm)
    last = len(contest.match.steps) - 1
    for lang in ("en", "hinglish"):
        for start in range(1, last + 1, CHUNK):
            facts = [step_facts(contest, mm, tl, x) for x in range(start, min(start + CHUNK - 1, last) + 1)]
            for attempt in range(3):
                got = ai.over_notes(facts, lang)
                if got:
                    break
                print(f"   retrying in 20s ({ai.last_error})")
                time.sleep(20)
            status = "ok" if got else "FAILED (app will use template notes here)"
            print(f"{m['label'][:40]:40s} {lang:8s} overs {start:2d}-{facts[-1]['step']:2d}: {status}")
            if got:
                print("   e.g.", next(iter(got.values()))["note"])
            time.sleep(4)
        # one sample catch-up for the end of the match
        ai.catch_up(catch_up_facts(contest, mm, tl, max(1, last - 5), last), lang)

print("\nDone. Cache saved to data/ai_cache.json")
