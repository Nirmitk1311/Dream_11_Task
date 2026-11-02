"""MatchLens - your team's view of a live cricket match.

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import html
import re
import time

import altair as alt
import pandas as pd
import streamlit as st

from matchlens import ai
from matchlens.contest import Contest, validate_team
from matchlens.engine import SCORING, list_matches, load_match
from matchlens.facts import catch_up_facts, preset_team, step_facts

st.set_page_config(page_title="MatchLens", page_icon=":material/sports_cricket:", layout="wide",
                   initial_sidebar_state="auto")

INK, MUTED, RULE, ACCENT, UP, DOWN = "#17171B", "#6E6A60", "#D8D2C4", "#C2410C", "#1F7A4D", "#B42318"

try:
    ai.configure(st.secrets.get("GEMINI_API_KEY"))
except Exception:
    ai.configure(None)

st.markdown("""<style>
header[data-testid="stHeader"]{background:transparent;}
footer{visibility:hidden;}
.block-container{padding-top:1.4rem;max-width:1280px;}
.bug{display:flex;align-items:stretch;background:#17171B;color:#EDE9E0;border-radius:3px;overflow:hidden;margin-bottom:14px;}
.bug .brand{background:#C2410C;color:#fff;padding:0 16px;font-family:'Barlow Condensed',sans-serif;font-weight:700;letter-spacing:.08em;font-size:21px;display:flex;align-items:center;}
.bug .inn{padding:8px 18px;border-right:1px solid #34343B;display:flex;flex-direction:column;justify-content:center;min-width:150px;}
.bug .t{font-family:'IBM Plex Mono',monospace;font-size:10.5px;color:#9B968A;letter-spacing:.09em;text-transform:uppercase;}
.bug .s{font-family:'Barlow Condensed',sans-serif;font-size:25px;font-weight:600;line-height:1.1;}
.bug .s small{font-size:16px;color:#9B968A;font-weight:500;margin-left:4px;}
.bug .inn.dim .s{color:#6E6A60;font-size:18px;}
.bug .inn.on{box-shadow:inset 0 -3px 0 #F97316;}
.bug .ctx{margin-left:auto;padding:8px 18px;text-align:right;display:flex;flex-direction:column;justify-content:center;}
.bug .ctx .r{font-size:13px;color:#EDE9E0;margin-top:2px;}
.dot{display:inline-block;width:7px;height:7px;background:#F97316;border-radius:50%;margin-right:7px;vertical-align:1px;}
.hero{display:grid;grid-template-columns:1.6fr 1fr 1fr 1fr;border-top:2px solid #17171B;border-bottom:1px solid #D8D2C4;margin:2px 0 6px;}
.hero .c{padding:12px 16px 14px;border-right:1px solid #D8D2C4;}
.hero .c:last-child{border-right:none;}
.k{font-family:'IBM Plex Mono',monospace;font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;color:#6E6A60;}
.v{font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:42px;line-height:1;margin-top:6px;color:#17171B;}
.hero .c.main .v{font-size:60px;}
.v.sm{font-size:32px;}
.d{font-size:13px;margin-top:6px;color:#6E6A60;}
.up{color:#1F7A4D;font-weight:600;} .down{color:#B42318;font-weight:600;}
.pbar{height:4px;background:#E3DED2;margin-top:10px;position:relative;}
.pbar i{position:absolute;left:0;top:0;bottom:0;background:#17171B;}
.sh{display:flex;align-items:baseline;justify-content:space-between;border-bottom:2px solid #17171B;padding-bottom:3px;margin:20px 0 12px;}
.sh .h{font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:21px;text-transform:uppercase;letter-spacing:.03em;margin:0;padding:0;}
.sh span{font-family:'IBM Plex Mono',monospace;font-size:10.5px;color:#6E6A60;text-transform:uppercase;letter-spacing:.07em;}
.feed{border-left:1px solid #D8D2C4;margin-left:5px;}
.fi{position:relative;padding:0 0 15px 18px;}
.fi:before{content:"";position:absolute;left:-5px;top:5px;width:9px;height:9px;background:#F4F1EA;border:1.5px solid #17171B;border-radius:50%;box-sizing:border-box;}
.fi.latest:before{background:#C2410C;border-color:#C2410C;}
.fi .m{font-family:'IBM Plex Mono',monospace;font-size:11.5px;color:#6E6A60;}
.fi .m b{color:#17171B;font-weight:500;}
.fi .tx{font-size:15px;line-height:1.5;margin-top:3px;}
.fi.latest .tx{font-size:16.5px;}
.chip{font-family:'IBM Plex Mono',monospace;font-size:11px;padding:1px 6px;border-radius:2px;margin-left:8px;}
.chip.up{background:#E1EFE6;color:#1F7A4D;} .chip.down{background:#F6E1DE;color:#B42318;} .chip.flat{background:#E3DED2;color:#6E6A60;}
.lede{font-size:15px;line-height:1.5;border-left:3px solid #C2410C;padding:1px 0 1px 11px;margin:0 0 12px;}
.nx{border:1px solid #D8D2C4;background:#FBFAF6;}
.nx .row{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:10px 12px;border-bottom:1px solid #E6E1D6;}
.nx .row:last-child{border-bottom:none;}
.nx .nm{font-weight:600;font-size:15px;}
.nx .sub{font-size:12.5px;color:#6E6A60;margin-top:2px;}
.tag{font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:14px;letter-spacing:.05em;text-transform:uppercase;padding:3px 8px;border-radius:2px;white-space:nowrap;}
.tag.cheer{background:#1F7A4D;color:#fff;} .tag.fear{background:#B42318;color:#fff;} .tag.neutral{background:#E3DED2;color:#6E6A60;}
.ft{border:2px solid #17171B;padding:12px 14px;background:#FBFAF6;}
.ft .v{font-size:40px;}
.edge{width:100%;border-collapse:collapse;font-size:13.5px;border:none!important;margin:0;}
.edge tr{background:transparent!important;border:none!important;}
.edge th{font-family:'IBM Plex Mono',monospace;font-weight:400;font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:#6E6A60;text-align:left;border-bottom:1px solid #17171B!important;padding:4px 6px;border-top:none!important;border-left:none!important;border-right:none!important;background:transparent!important;}
.edge td{padding:7px 6px;border:none!important;border-bottom:1px solid #E6E1D6!important;vertical-align:middle;}
.edge .who{font-size:11.5px;color:#6E6A60;}
.edge .b{height:5px;margin:3px 0;background:#E3DED2;position:relative;width:110px;}
.edge .b i{position:absolute;left:0;top:0;bottom:0;}
.edge .b.you i{background:#17171B;} .edge .b.fld i{background:#A39E92;}
.num{font-family:'IBM Plex Mono',monospace;font-size:12.5px;text-align:right;white-space:nowrap;}
.legend{font-family:'IBM Plex Mono',monospace;font-size:10.5px;color:#6E6A60;margin-top:6px;}
.legend i{display:inline-block;width:10px;height:5px;margin:0 4px 1px 10px;}
.catch{background:#FBFAF6;border:1px solid #D8D2C4;padding:10px 12px;font-size:14.5px;line-height:1.55;margin-top:8px;}
.wm{font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:26px;letter-spacing:.08em;line-height:1;}
.wm b{color:#F97316;}
.tagline{font-size:12.5px;color:#9B968A;margin:4px 0 6px;}
.lu .grp{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:.12em;color:#9B968A;margin:12px 0 3px;text-transform:uppercase;}
.lu .p{display:flex;justify-content:space-between;font-size:13.5px;padding:4px 0;border-bottom:1px solid #2E2E35;}
.lu .cv{font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:12px;background:#F97316;color:#17171B;padding:0 4px;margin-left:6px;border-radius:2px;}
.lu .pts{font-family:'IBM Plex Mono',monospace;font-size:12.5px;}
.lu .tot{display:flex;justify-content:space-between;align-items:baseline;font-family:'Barlow Condensed',sans-serif;font-size:22px;font-weight:700;margin-top:10px;}
.status{font-family:'IBM Plex Mono',monospace;font-size:10.5px;color:#9B968A;margin-top:14px;letter-spacing:.04em;}
.status i{display:inline-block;width:6px;height:6px;border-radius:50%;margin-right:7px;vertical-align:1px;}
@media (max-width:760px){.hero{grid-template-columns:1fr 1fr;}.hero .c{border-bottom:1px solid #D8D2C4;}
.bug{flex-wrap:wrap;}.bug .ctx{margin-left:0;text-align:left;}.hero .c.main .v{font-size:46px;}}
</style>""", unsafe_allow_html=True)


def H(s: str):
    """Render compact HTML (strip newlines/indent so markdown never treats it as code)."""
    st.markdown(re.sub(r"\n\s*", "", s), unsafe_allow_html=True)


def sh(title: str, right: str = ""):
    H(f"<div class='sh'><div class='h'>{title}</div><span>{right}</span></div>")


EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]")


def clean(text: str) -> str:
    return html.escape(re.sub(r"\s{2,}", " ", EMOJI.sub("", text or "")).strip())


@st.cache_resource(show_spinner="Building a 10,000-team contest…")
def get_contest(match_id: str) -> Contest:
    return Contest.build(load_match(match_id))


matches = list_matches()
match_labels = {m["id"]: m["label"] for m in matches}

# ------------------------------------------------------------------ sidebar: match + language
with st.sidebar:
    H("<div class='wm'>MATCH<b>LENS</b></div><div class='tagline'>Your fantasy team's view of the match</div>")
    match_id = st.selectbox("Match", list(match_labels), format_func=lambda k: match_labels[k])
    lang = st.segmented_control("Notes language", ["en", "hinglish"], default="en",
                                format_func=lambda k: {"en": "English", "hinglish": "Hinglish"}[k]) or "en"

contest = get_contest(match_id)
match = contest.match
last_step = len(match.steps) - 1
first_short = match.steps[1].label.split(" ")[0]
inn_break = next((i for i, x in enumerate(match.steps) if i > 0 and not x.label.startswith(first_short)),
                 last_step + 1)
n1 = inn_break - 1  # overs in first innings
second_short = match.steps[inn_break].label.split(" ")[0] if inn_break <= last_step else ""

if st.session_state.get("match_id") != match_id:
    st.session_state.match_id = match_id
    st.session_state.step = 0
    if "over" in st.query_params and "_deeplinked" not in st.session_state:  # e.g. ?over=27
        st.session_state._deeplinked = True
        try:
            st.session_state.step = max(0, min(last_step, int(st.query_params["over"])))
        except ValueError:
            pass
    st.session_state.team = preset_team(contest)
    st.session_state.autoplay = False


def plabel(p):
    pl = match.players[p]
    return f"{pl['name']} · {pl['role']} · {match.short_team(p)}"


with st.sidebar:
    with st.expander("Change team", expanded=False):
        mode = st.radio("Team", ["Sample fan team", "Build my own"], label_visibility="collapsed")
        if mode == "Sample fan team":
            if st.session_state.team != preset_team(contest):
                st.session_state.team = preset_team(contest)
        else:
            with st.form("build", border=False):
                cur = st.session_state.team
                picks = st.multiselect("Pick 11 players", list(match.players), default=cur["picks"],
                                       format_func=plabel, max_selections=11)
                opts = picks or list(match.players)
                c = st.selectbox("Captain (2x)", opts, format_func=match.name,
                                 index=opts.index(cur["captain"]) if cur["captain"] in opts else 0)
                v = st.selectbox("Vice-captain (1.5x)", opts, format_func=match.name,
                                 index=opts.index(cur["vice"]) if cur["vice"] in opts else min(1, len(opts) - 1))
                if st.form_submit_button("Lock team", type="primary", width="stretch"):
                    errs = validate_team(match, picks, c, v)
                    for e in errs:
                        st.error(e)
                    if not errs:
                        st.session_state.team = {"picks": picks, "captain": c, "vice": v}
                        st.session_state.step = 0
                        st.success("Team locked. Replay restarted.")

team = st.session_state.team
mm = contest.my_mult(team["picks"], team["captain"], team["vice"])
tl = contest.timeline(mm)
team_key = (match_id, tuple(sorted(team["picks"])), team["captain"], team["vice"])

# ------------------------------------------------------------------ replay state
AUTOPLAY_SECS = 2.5
if st.session_state.get("autoplay"):
    _now = time.time()
    if _now - st.session_state.get("_last_tick", 0) >= AUTOPLAY_SECS:
        st.session_state._last_tick = _now
        if st.session_state.step < last_step:
            st.session_state.step += 1
        else:
            st.session_state.autoplay = False


def _set(v):
    st.session_state.step = max(0, min(last_step, v))
    st.session_state.autoplay = False


s = st.session_state.get("step", 0)
step = match.steps[s]
SCORE_RE = re.compile(r"^(\S+) (\d+/\d+) \(([\d.]+)\)$")

# ------------------------------------------------------------------ scorebug
def inn_cell(short: str, idx_range: range) -> str:
    if s == 0 or s < idx_range.start:
        return f"<div class='inn dim'><div class='t'>{short}</div><div class='s'>Yet to bat</div></div>"
    at = min(s, idx_range.stop - 1)
    mt = SCORE_RE.match(match.steps[at].score)
    on = " on" if s in idx_range else ""
    sc, ov = (mt.group(2), mt.group(3)) if mt else (match.steps[at].score, "")
    return f"<div class='inn{on}'><div class='t'>{short}</div><div class='s'>{sc}<small>({ov})</small></div></div>"


pos = "Before the first ball" if s == 0 else ("Full time" if s == last_step else
                                               f"Over {s - (n1 if s > n1 else 0)} · {'1st innings' if s <= n1 else 'chase'}")
H(f"""<div class='bug'><div class='brand'>{first_short} v {second_short}</div>
{inn_cell(first_short, range(1, inn_break))}{inn_cell(second_short, range(inn_break, last_step + 1))}
<div class='ctx'><div class='t'><span class='dot'></span>Replay</div><div class='r'>{pos}</div></div></div>""")

# ------------------------------------------------------------------ controls
cb = st.columns([0.9, 0.9, 1.4, 1.1, 5.2], vertical_alignment="center")
cb[0].button("Restart", on_click=_set, args=(0,), width="stretch")
cb[1].button("Back", on_click=lambda: _set(st.session_state.step - 1), width="stretch", disabled=s == 0)
cb[2].button("Next over  →", type="primary", on_click=lambda: _set(st.session_state.step + 1),
             width="stretch", disabled=s >= last_step)
cb[3].toggle("Autoplay", key="autoplay")
cb[4].slider("Replay position", 0, last_step, key="step", label_visibility="collapsed")


@st.fragment(run_every=1 if st.session_state.get("autoplay") else None)
def _ticker():
    if st.session_state.get("autoplay") and \
            time.time() - st.session_state.get("_last_tick", 0) >= AUTOPLAY_SECS:
        st.rerun(scope="app")


_ticker()

# ------------------------------------------------------------------ AI notes (lazy, chunked, cached)
CHUNK = 10


def notes_for(steps_needed: list[int], generate: bool = True) -> dict[int, dict]:
    store = st.session_state.setdefault("notes", {})
    have = store.setdefault((team_key, lang), {})
    if not generate:
        return {x: have.get(x) or {"note": ai.template_note(step_facts(contest, mm, tl, x)), "ai": False}
                for x in steps_needed if x >= 1}
    chunks = sorted({(x - 1) // CHUNK for x in steps_needed if x >= 1 and x not in have})
    for ch in chunks:
        rng = range(ch * CHUNK + 1, min((ch + 1) * CHUNK, last_step) + 1)
        facts = [step_facts(contest, mm, tl, x) for x in rng]
        got = None
        if ai.available():
            with st.spinner("Reading the overs…"):
                got = ai.over_notes(facts, lang)
        for f in facts:
            g = (got or {}).get(f["step"])
            have[f["step"]] = {
                "note": (g or {}).get("note") or ai.template_note(f),
                "root": (g or {}).get("root") or ai.template_root(f),
                "ai": bool(g and g.get("note")),
            }
    return have


visible = list(range(s, max(0, s - 5), -1))
notes = notes_for(visible) if s >= 1 else {}


def move(x: int) -> int | None:
    return int(tl["rank"][x - 1] - tl["rank"][x]) if x >= 2 else None


def chip(mv: int | None) -> str:
    if mv is None:
        return "<span class='chip flat'>start</span>"
    if mv > 0:
        return f"<span class='chip up'>▲ {mv:,}</span>"
    if mv < 0:
        return f"<span class='chip down'>▼ {-mv:,}</span>"
    return "<span class='chip flat'>no change</span>"


# ------------------------------------------------------------------ hero numbers
n = tl["n"]
pts_now = float(tl["points"][s])
if s == 0:
    rank_html = f"<div class='v'>—</div><div class='d'>All {n:,} teams start level</div>"
else:
    r = int(tl["rank"][s])
    mv = move(s)
    top = max(1, round(r / n * 100))
    mv_html = ("<span class='up'>▲ {:,} this over</span>".format(mv) if mv and mv > 0 else
               "<span class='down'>▼ {:,} this over</span>".format(-mv) if mv and mv < 0 else
               "no change this over" if mv == 0 else "after the first over")
    rank_html = (f"<div class='v'>#{r:,}</div><div class='d'>{mv_html} · of {n:,} · top {top}%</div>"
                 f"<div class='pbar'><i style='width:{100 - r / n * 100:.1f}%'></i></div>")
pd_ = float(tl["points"][s] - tl["points"][s - 1]) if s else 0
fa = float(tl["field_avg"][s])
lead = pts_now - fa
mt = SCORE_RE.match(step.score)
need = re.search(r"Need (\d+) more", step.summary)
match_cell = (f"<div class='v sm'>{mt.group(1)} {mt.group(2)}</div><div class='d'>{mt.group(3)} ov"
              f"{' · need ' + need.group(1) if need else ''}</div>") if (s and mt) else \
    "<div class='v sm'>0/0</div><div class='d'>Toss done, first ball soon</div>"
H(f"""<div class='hero'>
<div class='c main'><div class='k'>Your rank</div>{rank_html}</div>
<div class='c'><div class='k'>Your points</div><div class='v'>{pts_now:.0f}</div>
<div class='d'>{f'+{pd_:.0f} this over' if s else '+4 per player for playing'}</div></div>
<div class='c'><div class='k'>Field average</div><div class='v'>{fa:.0f}</div>
<div class='d'>you're <span class='{'up' if lead >= 0 else 'down'}'>{lead:+.0f}</span> · top 10% cut {float(tl['top10pct'][s]):.0f}</div></div>
<div class='c'><div class='k'>Match</div>{match_cell}</div>
</div>""")

left, right = st.columns([1.55, 1], gap="large")

# ------------------------------------------------------------------ left: chart + feed
with left:
    sh("Rank, over by over", f"{n - 1:,} simulated rivals")
    df = pd.DataFrame({
        "over": list(range(1, s + 1)),
        "rank": [int(x) for x in tl["rank"][1:s + 1]],
        "points": [round(float(p)) for p in tl["points"][1:s + 1]],
        "label": [x.label for x in match.steps[1:s + 1]],
        "score": [x.score for x in match.steps[1:s + 1]],
    })
    label_expr = (f"datum.value <= {n1} ? '{first_short} ' + datum.value : "
                  f"'{second_short} ' + (datum.value - {n1})")
    x_enc = alt.X("over:Q", scale=alt.Scale(domain=[1, last_step], nice=False),
                  axis=alt.Axis(title=None, values=list(range(5, last_step, 5)), labelExpr=label_expr))
    y_max = int(max(tl["rank"][1:].max(), 10))
    y_enc = alt.Y("rank:Q", scale=alt.Scale(reverse=True, domain=[1, y_max * 1.05], nice=False),
                  axis=alt.Axis(title="RANK", format="~s", tickCount=5))
    layers = []
    if n1 < last_step:
        band = pd.DataFrame({"a": [n1 + 0.5], "b": [last_step], "t": ["CHASE"]})
        layers.append(alt.Chart(band).mark_rect(color="#EAE5DA", opacity=0.6).encode(x="a:Q", x2="b:Q"))
        layers.append(alt.Chart(band).mark_text(align="left", dx=6, dy=0, font="IBM Plex Mono", fontSize=10,
                                                color=MUTED).encode(x="a:Q", y=alt.value(10), text="t:N"))
    if len(df):
        tip = [alt.Tooltip("label:N", title="Over"), alt.Tooltip("score:N", title="Score"),
               alt.Tooltip("rank:Q", title="Your rank", format=","), alt.Tooltip("points:Q", title="Your points")]
        base = alt.Chart(df).encode(x=x_enc, y=y_enc, tooltip=tip)
        layers.append(base.mark_line(color=INK, strokeWidth=2.2, interpolate="monotone"))
        layers.append(base.mark_circle(color=INK, size=22, opacity=1))
        cur = df.tail(1).assign(txt=lambda d: d["rank"].map(lambda v: f"#{v:,}"))
        layers.append(alt.Chart(cur).mark_circle(color=ACCENT, size=110, opacity=1).encode(x=x_enc, y=y_enc))
        layers.append(alt.Chart(cur).mark_text(align="left", dx=9, dy=-10, font="Barlow Condensed", fontSize=17,
                                               fontWeight=700, color=ACCENT).encode(x=x_enc, y=y_enc, text="txt:N"))
    else:
        layers.append(alt.Chart(pd.DataFrame({"over": [1], "rank": [1]})).mark_point(opacity=0).encode(x=x_enc, y=y_enc))
    chart = alt.layer(*layers).properties(height=260, background="transparent").configure_axis(
        labelFont="IBM Plex Mono", titleFont="IBM Plex Mono", labelColor=MUTED, titleColor=MUTED,
        labelFontSize=10.5, titleFontSize=10, titleFontWeight="normal", gridColor="#E3DED2",
        domainColor=RULE, tickColor=RULE, titleAngle=0, titleAlign="left", titleY=-10, titleX=-38,
    ).configure_view(stroke=None)
    st.altair_chart(chart, theme=None, width="stretch")

    sh("What moved you", "latest first")
    if s == 0:
        H("<div class='lede'>Press <b>Next over</b> or turn on <b>Autoplay</b>. After every over you get one line "
          "on what happened to <i>your</i> rank and why. Players you own more than your rivals do push you up; "
          "popular players you skipped push you down.</div>")
    else:
        items = []
        for i, x in enumerate(visible):
            if x < 1 or x not in notes:
                continue
            st_ = match.steps[x]
            items.append(f"<div class='fi{' latest' if i == 0 else ''}'><div class='m'><b>{st_.label}</b> · "
                         f"{st_.score}{chip(move(x))}</div><div class='tx'>{clean(notes[x]['note'])}</div></div>")
        H("<div class='feed'>" + "".join(items) + "</div>")
        older = list(range(s - 5, 0, -1))
        if older:
            with st.expander(f"Earlier overs ({len(older)})"):
                have = notes_for(older, generate=False)
                H("<div class='feed'>" + "".join(
                    f"<div class='fi'><div class='m'><b>{match.steps[x].label}</b> · {match.steps[x].score}"
                    f"{chip(move(x))}</div><div class='tx'>{clean(have[x]['note'])}</div></div>" for x in older) + "</div>")

# ------------------------------------------------------------------ right: next over, catch-up, edge
WANT = {("batting", "cheer"): "Want runs", ("batting", "fear"): "Want him out",
        ("bowling", "cheer"): "Want wickets", ("bowling", "fear"): "Want him hit",
        ("batting", "neutral"): "Low stake", ("bowling", "neutral"): "Low stake"}
YOU = {"your Captain (2x)": "your captain", "your Vice-captain (1.5x)": "your vice-captain",
       "in your team": "in your team", "NOT in your team": "not in your team"}

with right:
    if s >= last_step:
        sh("Full time")
        H(f"<div class='ft'><div class='k'>{html.escape(match.result)}</div>"
          f"<div class='v'>#{int(tl['rank'][s]):,}</div>"
          f"<div class='d'>Your final rank of {n:,}, with {pts_now:.0f} points "
          f"(field average {fa:.0f}).</div></div>")
    else:
        nxt = match.steps[s].next_over or {}
        bowler = match.name(nxt["bowler"]) if nxt else ""
        sh("Next over", f"{bowler} to bowl" if bowler else "")
        if s >= 1 and s in notes and notes[s].get("root"):
            H(f"<div class='lede'>{clean(notes[s]['root'])}</div>")
        rows = []
        for r_ in contest.root_for(mm, s):
            rows.append(
                f"<div class='row'><div><div class='nm'>{html.escape(r_['player'])}</div>"
                f"<div class='sub'>{r_['team']} · {r_['job']} · {YOU[r_['you']]} · "
                f"in {r_['picked_by_pct']}% of rival teams</div></div>"
                f"<span class='tag {r_['verdict']}'>{WANT[(r_['job'], r_['verdict'])]}</span></div>")
        H("<div class='nx'>" + "".join(rows) + "</div>")

    sh("Catch me up", "missed a few overs?")
    if s < 2:
        H("<div class='d' style='margin-top:0'>Available from over 2: a three-line summary of only what "
          "mattered to your team since you last looked.</div>")
    else:
        options = list(range(1, s))
        cc = st.columns([2.2, 1], vertical_alignment="bottom")
        a = cc[0].selectbox("I last checked at", options, index=max(0, len(options) - 6),
                            format_func=lambda i: f"{match.steps[i].label}  ·  {match.steps[i].score}")
        if cc[1].button("Catch up", width="stretch"):
            f = catch_up_facts(contest, mm, tl, a, s)
            text = None
            if ai.available():
                with st.spinner("Summarising…"):
                    text = ai.catch_up(f, lang)
            st.session_state.catchup = (a, s, lang, text or ai.template_catch_up(f))
        cu = st.session_state.get("catchup")
        if cu and cu[0] == a and cu[1] == s and cu[2] == lang:
            lines = [clean(x.lstrip("-•* ").strip()) for x in cu[3].splitlines() if x.strip()]
            H("<div class='catch'>" + "<br>".join(lines) + "</div>")

    sh("Where you differ", "you vs the field")
    cur_pts = match.steps[s].points
    trs = []
    for r_ in contest.differentials(mm)[:8]:
        i = contest.pids.index(r_["pid"])
        you_m, fld_m = float(mm[i]), float(contest.eo[i])
        eff = "<span class='up'>helps</span>" if r_["leverage"] > 0 else "<span class='down'>hurts</span>"
        trs.append(
            f"<tr><td><div>{html.escape(r_['player'])}</div><div class='who'>{r_['team']} · "
            f"{YOU[r_['you']]} · in {r_['picked_by_pct']}%</div></td>"
            f"<td><div class='b you'><i style='width:{you_m / 2 * 100:.0f}%'></i></div>"
            f"<div class='b fld'><i style='width:{min(fld_m, 2) / 2 * 100:.0f}%'></i></div></td>"
            f"<td>{eff}</td><td class='num'>{cur_pts[r_['pid']]}</td></tr>")
    H("<table class='edge'><tr><th>Player</th><th>Stake</th><th>If he scores</th><th class='num'>Pts</th></tr>"
      + "".join(trs) + "</table>"
      "<div class='legend'>Stake:<i style='background:#17171B'></i>you<i style='background:#A39E92'></i>"
      "rivals' average (captain = 2x)</div>")

# ------------------------------------------------------------------ sidebar: lineup + status
with st.sidebar:
    pts = match.steps[s].points
    groups = []
    for role, title in [("WK", "Wicket-keeper"), ("BAT", "Batters"), ("AR", "All-rounders"), ("BOWL", "Bowlers")]:
        ps = [p for p in team["picks"] if match.players[p]["role"] == role]
        if not ps:
            continue
        rows = []
        for p in ps:
            mult = 2 if p == team["captain"] else 1.5 if p == team["vice"] else 1
            badge = "<span class='cv'>C</span>" if mult == 2 else "<span class='cv'>VC</span>" if mult == 1.5 else ""
            rows.append(f"<div class='p'><span>{html.escape(match.name(p))}{badge}</span>"
                        f"<span class='pts'>{pts[p] * mult:g}</span></div>")
        groups.append(f"<div class='grp'>{title}</div>" + "".join(rows))
    H("<div class='lu'>" + "".join(groups) +
      f"<div class='tot'><span>Total</span><span>{pts_now:.0f}</span></div></div>")
    any_ai = any(v.get("ai") for v in notes.values()) if notes else False
    if not ai.available():
        dot, msg = "#6E6A60", "Notes: basic mode (no AI key)"
    elif ai.last_error and not any_ai:
        dot, msg = "#F59E0B", "Notes: Gemini busy, basic mode"
    else:
        dot, msg = "#22C55E", "Notes: written by Gemini"
    H(f"<div class='status'><i style='background:{dot}'></i>{msg}</div>")

# ------------------------------------------------------------------ how it works
st.markdown("")
with st.expander("How MatchLens works, and what is simulated"):
    st.markdown(f"""
**The problem.** A fantasy player watches *their 11 against everyone else's*. The app gives them a points total and
a rank that moves every over, but not **why** it moved or **what to hope for next**.

**How it works**
1. **Scoring engine (exact, no AI).** Replays real ball-by-ball data and computes Dream11-style T20 points after every over.
2. **Contest (simulated).** {n - 1:,} rival teams are generated from *pre-match* popularity, which gives a rank and
   "picked by %" for every player. With real contest data the maths is the same.
3. **Gemini.** Receives only the computed facts and writes the short notes in English or Hinglish. It explains and never
   calculates. If it's unavailable, plain template notes are shown instead.

**Why "Next over" works:** a player helps you only when you own him *more than the field does*. A captain that few
rivals captained is worth cheering. A player in 90% of teams that you left out hurts you every time he scores.

**Scoring (approximation):** +{SCORING['run']} per run, +{SCORING['four_bonus']} per four, +{SCORING['six_bonus']} per six,
25/50/75/100-run bonuses, duck −2, wicket +{SCORING['wicket']} (+8 bowled/LBW), 3/4/5-wicket bonuses, maiden +12, dot ball +1,
catch +8, stumping +12, run-out +12/+6, economy and strike-rate bonuses, captain 2x, vice-captain 1.5x, +4 for playing.

Match data from [Cricsheet](https://cricsheet.org) (ODC-By). Rival teams and popularity are simulated. Not affiliated with Dream11.
""")
