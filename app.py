"""MatchLens - your team's-eye view of a live cricket match.

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import time

import altair as alt
import pandas as pd
import streamlit as st

from matchlens import ai
from matchlens.contest import Contest, validate_team
from matchlens.engine import SCORING, list_matches, load_match
from matchlens.facts import catch_up_facts, preset_team, step_facts

st.set_page_config(page_title="MatchLens", page_icon="🏏", layout="wide")

# ------------------------------------------------------------------ setup
try:
    ai.configure(st.secrets.get("GEMINI_API_KEY"))
except Exception:
    ai.configure(None)

st.markdown(
    """
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
.ml-tag {font-size: .78rem; color: #6b7280; margin-top:-.6rem;}
.ml-card {border-radius: 12px; padding: .75rem .9rem; margin-bottom: .5rem; border: 1px solid #e5e7eb; background:#fff;}
.ml-cheer {border-left: 6px solid #16a34a; background: #f0fdf4;}
.ml-fear  {border-left: 6px solid #dc2626; background: #fef2f2;}
.ml-neutral {border-left: 6px solid #9ca3af; background: #f9fafb;}
.ml-big {font-weight: 700; font-size: 1.02rem;}
.ml-small {font-size: .82rem; color: #4b5563;}
.ml-note {border-left: 3px solid #4f46e5; padding: .35rem .7rem; margin: .35rem 0; background: #f5f5ff; border-radius: 6px;}
.ml-note b {color:#3730a3;}
.ml-up {color:#16a34a; font-weight:700;} .ml-down {color:#dc2626; font-weight:700;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Simulating a 10,000-team contest…")
def get_contest(match_id: str) -> Contest:
    return Contest.build(load_match(match_id))


matches = list_matches()
match_labels = {m["id"]: m["label"] for m in matches}

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## 🏏 MatchLens")
    match_id = st.selectbox("Match (real ball-by-ball replay)", list(match_labels),
                            format_func=lambda k: match_labels[k])
    lang = st.radio("Language", ["en", "hinglish"], horizontal=True,
                    format_func=lambda k: {"en": "English", "hinglish": "Hinglish"}[k])

contest = get_contest(match_id)
match = contest.match
last_step = len(match.steps) - 1

if st.session_state.get("match_id") != match_id:
    st.session_state.match_id = match_id
    st.session_state.step = 0
    st.session_state.team = preset_team(contest)
    st.session_state.autoplay = False


def label(p):
    pl = match.players[p]
    return f"{pl['name']} · {pl['role']} · {match.short_team(p)}"


with st.sidebar:
    st.markdown("#### Your fantasy team")
    mode = st.radio("Team", ["Sample fan team", "Build my own"], horizontal=True, label_visibility="collapsed")
    if mode == "Sample fan team":
        if st.session_state.team != preset_team(contest):
            st.session_state.team = preset_team(contest)
    else:
        with st.form("build"):
            cur = st.session_state.team
            picks = st.multiselect("Pick 11 players", list(match.players), default=cur["picks"],
                                   format_func=label, max_selections=11)
            opts = picks or list(match.players)
            c = st.selectbox("Captain (2x)", opts, format_func=lambda p: match.name(p),
                             index=opts.index(cur["captain"]) if cur["captain"] in opts else 0)
            v = st.selectbox("Vice-captain (1.5x)", opts, format_func=lambda p: match.name(p),
                             index=opts.index(cur["vice"]) if cur["vice"] in opts else min(1, len(opts) - 1))
            if st.form_submit_button("Lock team", type="primary", width="stretch"):
                errs = validate_team(match, picks, c, v)
                if errs:
                    for e in errs:
                        st.error(e)
                else:
                    st.session_state.team = {"picks": picks, "captain": c, "vice": v}
                    st.session_state.step = 0
                    st.success("Team locked. Replay restarted.")

team = st.session_state.team
mm = contest.my_mult(team["picks"], team["captain"], team["vice"])
tl = contest.timeline(mm)
team_key = (match_id, tuple(sorted(team["picks"])), team["captain"], team["vice"])

# ------------------------------------------------------------------ replay controls
def _set(v):
    st.session_state.step = max(0, min(last_step, v))
    st.session_state.autoplay = False


AUTOPLAY_SECS = 2.5
if st.session_state.get("autoplay"):
    _now = time.time()
    if _now - st.session_state.get("_last_tick", 0) >= AUTOPLAY_SECS:
        st.session_state._last_tick = _now
        if st.session_state.step < last_step:
            st.session_state.step += 1
        else:
            st.session_state.autoplay = False

st.markdown("# MatchLens")
st.markdown("<div class='ml-tag'>Your fantasy team's-eye view of the live match: why your rank moves, "
            "and who to root for next.</div>", unsafe_allow_html=True)
st.caption(f"{match.title} · {match.subtitle}")

cb = st.columns([1, 1, 1.4, 1.2, 5.4], vertical_alignment="center")
cb[0].button("⏮", help="Restart", on_click=_set, args=(0,), width="stretch")
cb[1].button("◀", help="Previous over", on_click=lambda: _set(st.session_state.step - 1),
             width="stretch")
cb[2].button("Next over ▶", type="primary", on_click=lambda: _set(st.session_state.step + 1),
             width="stretch", disabled=st.session_state.step >= last_step)
cb[3].toggle("Autoplay", key="autoplay")
cb[4].slider("Replay position", 0, last_step, key="step", label_visibility="collapsed",
             format="%d")


@st.fragment(run_every=1 if st.session_state.get("autoplay") else None)
def _ticker():
    # only triggers a full rerun; the step is advanced at the top of the script
    if st.session_state.get("autoplay") and \
            time.time() - st.session_state.get("_last_tick", 0) >= AUTOPLAY_SECS:
        st.rerun(scope="app")


_ticker()

s = st.session_state.step
step = match.steps[s]

# ------------------------------------------------------------------ AI notes (lazy, chunked, cached)
CHUNK = 10


def notes_for(steps_needed: list[int], generate: bool = True) -> dict[int, dict]:
    store = st.session_state.setdefault("notes", {})
    key = (team_key, lang)
    have = store.setdefault(key, {})
    if not generate:  # use what exists, template for the rest (no API calls)
        return {x: have.get(x) or {"note": ai.template_note(step_facts(contest, mm, tl, x)), "ai": False}
                for x in steps_needed if x >= 1}
    chunks = sorted({(x - 1) // CHUNK for x in steps_needed if x >= 1 and x not in have})
    for ch in chunks:
        rng = range(ch * CHUNK + 1, min((ch + 1) * CHUNK, last_step) + 1)
        facts = [step_facts(contest, mm, tl, x) for x in rng]
        got = None
        if ai.available():
            with st.spinner("MatchLens is reading the overs…"):
                got = ai.over_notes(facts, lang)
        for f in facts:
            g = (got or {}).get(f["step"])
            have[f["step"]] = {
                "note": (g or {}).get("note") or ai.template_note(f),
                "root": (g or {}).get("root") or ai.template_root(f),
                "ai": bool(g and g.get("note")),
            }
    return have


visible = list(range(s, max(0, s - 6), -1))
notes = notes_for(visible) if s >= 1 else {}

# ------------------------------------------------------------------ metrics
m = st.columns(4)
m[0].metric("Match", step.score if s else "Yet to start", border=True)
pts_now = float(tl["points"][s])
pts_delta = float(tl["points"][s] - tl["points"][s - 1]) if s else None
m[1].metric("Your points", f"{pts_now:.0f}", f"{pts_delta:+.0f} this over" if pts_delta is not None else None,
            border=True)
if s == 0:
    m[2].metric("Your rank", "All tied", f"{tl['n']:,} teams in contest", delta_color="off", border=True)
else:
    r_now = int(tl["rank"][s])
    r_prev = int(tl["rank"][s - 1]) if s > 1 else None
    m[2].metric("Your rank", f"#{r_now:,}",
                f"{r_prev - r_now:+,} places" if r_prev else f"of {tl['n']:,}", border=True,
                delta_color="normal" if r_prev else "off")
m[3].metric("Field average", f"{float(tl['field_avg'][s]):.0f} pts",
            f"top 10% at {float(tl['top10pct'][s]):.0f}", delta_color="off", border=True)

if s == 0:
    st.info("**Press “Next over ▶”** (or switch on Autoplay) to replay the match over by over. "
            "MatchLens explains every rank move for *your* team and tells you who to root for next.")

left, right = st.columns([3, 2], gap="large")

# ------------------------------------------------------------------ left: rank chart + feed
with left:
    st.subheader("Your rank, over by over")
    inn_break = next((i for i, x in enumerate(match.steps)
                      if i > 0 and not x.label.startswith(match.steps[1].label.split(" ")[0])), None)
    df = pd.DataFrame({
        "over": list(range(1, s + 1)),
        "rank": [int(r) for r in tl["rank"][1:s + 1]],
        "points": [round(float(p)) for p in tl["points"][1:s + 1]],
        "label": [x.label for x in match.steps[1:s + 1]],
        "score": [x.score for x in match.steps[1:s + 1]],
    })
    if len(df):
        base = alt.Chart(df).encode(
            x=alt.X("over:Q", scale=alt.Scale(domain=[1, last_step]),
                    axis=alt.Axis(title="Over (1st innings → chase)", tickMinStep=1, values=list(range(0, last_step + 1, 5)))),
            y=alt.Y("rank:Q", scale=alt.Scale(reverse=True, zero=False), axis=alt.Axis(title="Rank (higher is better)", format=",")),
            tooltip=[alt.Tooltip("label:N", title="Over"), alt.Tooltip("score:N", title="Score"),
                     alt.Tooltip("rank:Q", title="Your rank", format=","), alt.Tooltip("points:Q", title="Your points")],
        )
        chart = base.mark_line(color="#4f46e5", strokeWidth=3) + base.mark_circle(color="#4f46e5", size=45)
        if inn_break and inn_break <= s:
            rule = alt.Chart(pd.DataFrame({"x": [inn_break - 0.5]})).mark_rule(strokeDash=[4, 4], color="#9ca3af").encode(x="x:Q")
            txt = alt.Chart(pd.DataFrame({"x": [inn_break - 0.5], "t": ["innings break"]})).mark_text(
                align="left", dx=4, dy=-120, color="#6b7280", fontSize=11).encode(x="x:Q", text="t:N")
            chart = chart + rule + txt
        st.altair_chart(chart.properties(height=280), width="stretch")
    else:
        st.caption("The chart fills in as the match is replayed.")

    st.subheader("What just happened to you")
    if s == 0:
        st.caption("Over-by-over explanations appear here.")
    for x in visible:
        if x < 1 or x not in notes:
            continue
        st.markdown(f"<div class='ml-note'><b>{match.steps[x].label}</b> · {match.steps[x].score}<br>"
                    f"{notes[x]['note']}</div>", unsafe_allow_html=True)
    older = list(range(s - 6, 0, -1))
    if older:
        with st.expander("Earlier overs"):
            have = notes_for(older, generate=False)
            for x in older:
                st.markdown(f"**{match.steps[x].label}** · {match.steps[x].score}  \n{have[x]['note']}")

# ------------------------------------------------------------------ right: root-for, catch-up, edge
with right:
    st.subheader("Root for this · next over")
    if s >= last_step:
        st.success(f"**Full time.** {match.result}. You finished **#{int(tl['rank'][s]):,}** of {tl['n']:,} "
                   f"with {pts_now:.0f} points.")
    else:
        if s >= 1 and s in notes:
            st.markdown(f"**{notes[s]['root']}**")
        for r in contest.root_for(mm, s):
            icon = {"cheer": "✅ Cheer", "fear": "⚠️ Fear", "neutral": "➖ Neutral"}[r["verdict"]]
            want = ("runs" if r["job"] == "batting" else "wickets") if r["verdict"] == "cheer" else \
                   ("him getting out" if r["job"] == "batting" else "him going for runs") if r["verdict"] == "fear" else \
                   "little impact on your rank"
            st.markdown(
                f"<div class='ml-card ml-{r['verdict']}'><span class='ml-big'>{icon}: {r['player']}</span> "
                f"<span class='ml-small'>({r['team']}, {r['job']})</span><br>"
                f"<span class='ml-small'>You want: <b>{want}</b> · You: {r['you']} · "
                f"picked by {r['picked_by_pct']}% of rivals</span></div>",
                unsafe_allow_html=True)

    st.subheader("Catch me up")
    if s < 2:
        st.caption("Missed a few overs? After over 2 you can get a 3-line catch-up on just what mattered to your team.")
    else:
        options = list(range(1, s))
        a = st.selectbox("I last checked at…", options, index=max(0, len(options) - 6),
                         format_func=lambda i: f"{match.steps[i].label} ({match.steps[i].score})")
        if st.button("Catch me up ⚡", width="stretch"):
            f = catch_up_facts(contest, mm, tl, a, s)
            text = None
            if ai.available():
                with st.spinner("Summarising what you missed…"):
                    text = ai.catch_up(f, lang)
            st.session_state.catchup = (a, s, lang, text or ai.template_catch_up(f), bool(text))
        cu = st.session_state.get("catchup")
        if cu and cu[1] == s and cu[0] == a and cu[2] == lang:
            st.markdown(f"<div class='ml-card'>{cu[3].replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)

    st.subheader("Your edge vs the field")
    st.caption("Players where your team differs most from the 10,000 rival teams. "
               "▲ = their points push you up, ▼ = their points push you down.")
    rows = contest.differentials(mm)[:9]
    cur_pts = match.steps[s].points
    st.dataframe(pd.DataFrame([{
        "Player": r["player"],
        "You": {"your Captain (2x)": "C (2x)", "your Vice-captain (1.5x)": "VC (1.5x)",
                "in your team": "✓", "NOT in your team": "—"}[r["you"]],
        "Picked by": f"{r['picked_by_pct']}%",
        "C by": f"{r['captained_by_pct']}%",
        "Effect": "▲ helps you" if r["leverage"] > 0 else "▼ hurts you",
        "Pts so far": cur_pts[r["pid"]],
    } for r in rows]), hide_index=True, width="stretch")

# ------------------------------------------------------------------ sidebar: team list + status
with st.sidebar:
    pts = match.steps[s].points
    tdf = pd.DataFrame([{
        "Player": match.name(p) + (" (C)" if p == team["captain"] else " (VC)" if p == team["vice"] else ""),
        "Role": match.players[p]["role"],
        "Pts": pts[p] * (2 if p == team["captain"] else 1.5 if p == team["vice"] else 1),
    } for p in sorted(team["picks"], key=lambda p: ["WK", "BAT", "AR", "BOWL"].index(match.players[p]["role"]))])
    st.dataframe(tdf, hide_index=True, width="stretch", height=422)
    st.divider()
    any_ai = any(v.get("ai") for v in notes.values()) if notes else None
    if not ai.available():
        st.warning("AI offline (no Gemini key). Showing basic template notes.")
    elif ai.last_error and not any_ai:
        st.warning("Gemini is busy right now, so you're seeing basic notes. Try again in a minute.")
    else:
        st.success("AI: Gemini connected")

# ------------------------------------------------------------------ how it works
with st.expander("How MatchLens works (and what's simulated)"):
    st.markdown(f"""
**The problem.** Fantasy players don't watch a match like normal fans. They watch *their 11 players versus everyone else's*.
The app shows them a points number and a rank that jumps every over, but not **why** it moved or **what to hope for next**.

**How it works**
1. **Scoring engine (exact, no AI):** replays real ball-by-ball data and computes Dream11-style T20 fantasy points after every over.
2. **Contest (simulated):** {tl['n'] - 1:,} rival teams are generated from *pre-match* popularity, so rank and
   "picked by %" can be computed. With real contest data the same maths works unchanged.
3. **AI layer (Gemini):** gets only the exact computed facts and turns them into short notes in English or Hinglish.
   It explains and never calculates, so it can't make up a point or a rank. If the AI is unavailable, the tool falls back to template text.

**The insight behind “Root for this”:** a player only helps you if you own him *more than the field does*.
Your captain whom only 4% of rivals captained is worth cheering. A player in 90% of teams that you skipped hurts you every time he scores.

**Scoring used (approximation):** +{SCORING['run']}/run, +{SCORING['four_bonus']} four bonus, +{SCORING['six_bonus']} six bonus,
25/50/75/100 run bonuses, duck −2, wicket +{SCORING['wicket']} (+8 bowled/LBW), 3/4/5-wicket hauls, maiden +12, dot +1,
catch +8, stumping +12, run-out +12/+6, economy & strike-rate bonuses, C 2x, VC 1.5x, +4 for playing.

*Match data: [Cricsheet](https://cricsheet.org) (ODC-By). Rival teams and popularity % are simulated. Not affiliated with Dream11.*
""")
