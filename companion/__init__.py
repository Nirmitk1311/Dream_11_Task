"""Live match companion: a fantasy-aware view of a live match (concept feature for the Dream11 app).

engine.py  - ball-by-ball -> fantasy points (exact, no AI)
contest.py - simulated contest: rank, effective ownership, what moved you
facts.py   - builds the fact packets the AI is allowed to talk about
ai.py      - Gemini turns facts into fan language (with template fallbacks)

Everything here is UI-independent, so it can be dropped into any app,
push-notification service or chatbot.
"""
