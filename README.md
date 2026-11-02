# Live match companion: a concept feature for the Dream11 app

Your fantasy team's view of a live match, designed to live inside the Dream11 app rather than as a separate product.

**The fan problem.** Fantasy cricket players don't watch a match like normal fans; they watch *their 11 versus everyone else's*.
During a live match they see points and a rank that jumps every over, but not **why** it moved or **who to root for next**.

**What it does** (replaying real ball-by-ball data, over by over):

- **Rank timeline + "What just happened to you":** every rank move explained in one line, in English or Hinglish
  (e.g. *"Down 1,636: Bhuvneshwar took a wicket. He's in 41% of rival teams, not yours."*)
- **Root for this · next over:** for the two batters and the bowler, whether their success helps or hurts *you*
- **Catch me up:** missed 6 overs? A 3-line summary of only what mattered to your team
- **Your edge vs the field:** the players where you differ most from the contest
- **What if…:** pick a delivery (dot / 1 / 4 / 6 / wicket bowled, LBW or caught) and stack up to an over of them.
  Every player is recomputed from the full updated stats, **all 10,001 teams are re-scored**, and you see points and
  rank before → after, teams you pass, and the exact rules behind every point
  (e.g. *runs +6 + six bonus +6 = +12 → ×2 captain = +24 for you; 94% of rivals own him, field avg +15*). **Reset** restores the exact state.
- **Next ball:** step through the replay ball by ball, not just over by over
- **Simulated leaderboard:** top teams + the teams just above and below you (updates with your what-if)

## How it works

| Layer | What | AI? |
|---|---|---|
| `data/t20_scoring_rules.json` | Versioned T20 ruleset (`dream11-style-t20 v1.0`), incl. economy/strike-rate bands and tie rule | - |
| `companion/engine.py` | Ball-by-ball → itemised fantasy points after every ball and over | No (exact) |
| `companion/whatif.py` | What-if simulator: apply hypothetical deliveries, re-score all lineups, rule breakdown | No (exact) |
| `companion/contest.py` | 10,000 simulated rival teams → rank, "picked by %", effective ownership | No (exact) |
| `companion/facts.py` | Builds the fact packets the AI is allowed to talk about | No |
| `companion/ai.py` | Gemini turns facts into short fan-friendly notes; template fallback + cache | **Yes** |
| `app.py` | Streamlit UI | - |

**The AI only explains and never calculates**, so it cannot make up a point or a rank. The `companion` package has no UI
dependency, so it can feed an in-app widget, push notifications or a chatbot directly.

**Core insight:** a player helps you only if you own him *more than the field does*
(`your multiplier − field's average multiplier`). Your differential captain scoring big is worth cheering. A 90%-owned player you skipped scoring is not.

## Tests

```bash
.venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest -q
```

The 16 tests cover: innings totals match the real scorecard; captain 2x and vice-captain 1.5x applied exactly once;
a six = runs + six bonus (from the ruleset, not hard-coded); milestone bonus applied once; a popular shared player lifts
every rival who owns him; wicket + catch updates batter, bowler and fielder; Reset restores the exact state; impossible
scenarios rejected; tie convention; and the full app working with no LLM key.

**Rank convention:** rank = 1 + number of teams with strictly more points (tied teams share the better rank).

## Run locally

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
# put your key in .streamlit/secrets.toml:  GEMINI_API_KEY = "..."
.venv\Scripts\python pregenerate.py   # optional: caches AI notes for the sample team
.venv\Scripts\streamlit run app.py
```

## Honest limitations

- Rival teams and popularity % are **simulated** from pre-match priors (Dream11 doesn't publish contest data). With real contest data the maths is unchanged.
- Scoring is an approximation of Dream11's T20 points system.
- Live mode is a replay of a completed match. A live feed would call the same engine every ball.

Match data: [Cricsheet](https://cricsheet.org) (Open Data Commons Attribution License). Concept prototype, not an official Dream11 product.
