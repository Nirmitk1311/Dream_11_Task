# MatchLens: your fantasy team's-eye view of a live match

**The fan problem.** Fantasy cricket players don't watch a match like normal fans; they watch *their 11 versus everyone else's*.
During a live match they see points and a rank that jumps every over, but not **why** it moved or **who to root for next**.

**What MatchLens does** (replaying real ball-by-ball data, over by over):

- **Rank timeline + "What just happened to you":** every rank move explained in one line, in English or Hinglish
  (e.g. *"Down 1,636: Bhuvneshwar took a wicket. He's in 41% of rival teams, not yours."*)
- **Root for this · next over:** for the two batters and the bowler, whether their success helps or hurts *you*
- **Catch me up:** missed 6 overs? A 3-line summary of only what mattered to your team
- **Your edge vs the field:** the players where you differ most from the contest

## How it works

| Layer | What | AI? |
|---|---|---|
| `matchlens/engine.py` | Ball-by-ball → Dream11-style T20 fantasy points after every over | No (exact) |
| `matchlens/contest.py` | 10,000 simulated rival teams → rank, "picked by %", effective ownership | No (exact) |
| `matchlens/facts.py` | Builds the fact packets the AI is allowed to talk about | No |
| `matchlens/ai.py` | Gemini turns facts into short fan-friendly notes; template fallback + cache | **Yes** |
| `app.py` | Streamlit UI | - |

**The AI only explains and never calculates**, so it cannot make up a point or a rank. The `matchlens` package has no UI
dependency, so it can feed an in-app widget, push notifications or a chatbot directly.

**Core insight:** a player helps you only if you own him *more than the field does*
(`your multiplier − field's average multiplier`). Your differential captain scoring big is worth cheering. A 90%-owned player you skipped scoring is not.

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

Match data: [Cricsheet](https://cricsheet.org) (Open Data Commons Attribution License). Not affiliated with Dream11.
