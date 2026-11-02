"""MatchLens - your team's view of a live cricket match.

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import html
import re
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from matchlens import ai, whatif
from matchlens.contest import Contest, validate_team
from matchlens.engine import RULESET, SCORING, list_matches, load_match
from matchlens.facts import catch_up_facts, preset_team, step_facts

st.set_page_config(page_title="MatchLens", page_icon=":material/sports_cricket:", layout="wide",
                   initial_sidebar_state="auto")

INK, MUTED, RULE, ACCENT, UP, DOWN = "#17171B", "#6E6A60", "#D8D2C4", "#C2410C", "#1F7A4D", "#B42318"

try:
    ai.configure(st.secrets.get("GEMINI_API_KEY"))
except Exception:
    ai.configure(None)

st.markdown("""<style>
:root{--u:clamp(1.05px, calc(0.62px + 0.03vw), 1.3px);}
header[data-testid="stHeader"]{background:transparent;}
footer{visibility:hidden;}
.block-container{padding:1rem 1.6rem 0;max-width:none;}
[data-testid="stMain"] .stMarkdown:has(.hero){margin-bottom:6px;}
[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"]{gap:.55rem;}
.st-key-main [data-testid="stHorizontalBlock"]{flex-wrap:wrap;row-gap:1rem;}
@media (min-width:900px){.st-key-main [data-testid="stColumn"]{min-width:360px;}}
.st-key-main [data-testid="stColumn"] [data-testid="stColumn"]{min-width:0!important;}
.st-key-main [data-testid="stColumn"] [data-testid="stHorizontalBlock"]{flex-wrap:nowrap!important;}
.st-key-feed,.st-key-edge,.st-key-wi,.st-key-cu,.st-key-lb{overflow-y:auto;overflow-x:hidden;padding-right:8px;padding-bottom:1.4rem;}
@media (min-width:1300px) and (min-height:680px){.st-key-feed{max-height:max(260px,calc(100vh - 298px - 4.7vw));}.st-key-edge,.st-key-wi,.st-key-cu{max-height:max(320px,calc(100vh - 305px - 4.7vw));}}
.bug{display:flex;align-items:stretch;background:#17171B;color:#EDE9E0;border-radius:3px;overflow:hidden;margin-bottom:14px;}
.bug .brand{background:#C2410C;color:#fff;padding:0 16px;font-family:'Barlow Condensed',sans-serif;font-weight:700;letter-spacing:.08em;font-size:calc(23.5 * var(--u));display:flex;align-items:center;}
.bug .inn{padding:8px 18px;border-right:1px solid #34343B;display:flex;flex-direction:column;justify-content:center;min-width:150px;}
.bug .t{font-family:'IBM Plex Mono',monospace;font-size:calc(12 * var(--u));color:#9B968A;letter-spacing:.09em;text-transform:uppercase;}
.bug .s{font-family:'Barlow Condensed',sans-serif;font-size:calc(25 * var(--u));font-weight:600;line-height:1.1;}
.bug .s small{font-size:calc(18 * var(--u));color:#9B968A;font-weight:500;margin-left:4px;}
.bug .inn.dim .s{color:#6E6A60;font-size:calc(20 * var(--u));}
.bug .inn.on{box-shadow:inset 0 -3px 0 #F97316;}
.bug .ctx{margin-left:auto;padding:8px 18px;text-align:right;display:flex;flex-direction:column;justify-content:center;}
.bug .ctx .r{font-size:calc(14.5 * var(--u));color:#EDE9E0;margin-top:2px;}
.dot{display:inline-block;width:7px;height:7px;background:#F97316;border-radius:50%;margin-right:7px;vertical-align:1px;}
.hero{display:grid;grid-template-columns:1.6fr 1fr 1fr 1fr;border-top:2px solid #17171B;border-bottom:1px solid #D8D2C4;margin:0;}
.hero .c{padding:9px 16px 11px;border-right:1px solid #D8D2C4;}
.hero .c:last-child{border-right:none;}
.k{font-family:'IBM Plex Mono',monospace;font-size:calc(12 * var(--u));letter-spacing:.09em;text-transform:uppercase;color:#6E6A60;}
.v{font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:calc(38 * var(--u));line-height:1;margin-top:5px;color:#17171B;}
.hero .c.main .v{font-size:calc(52 * var(--u));}
.v.sm{font-size:calc(32 * var(--u));}
.d{font-size:calc(14.5 * var(--u));margin-top:6px;color:#6E6A60;}
.up{color:#1F7A4D;font-weight:600;} .down{color:#B42318;font-weight:600;}
.pbar{height:4px;background:#E3DED2;margin-top:10px;position:relative;}
.pbar i{position:absolute;left:0;top:0;bottom:0;background:#17171B;}
.sh{display:flex;align-items:baseline;justify-content:space-between;border-bottom:2px solid #17171B;padding-bottom:3px;margin:4px 0 8px;}
.sh .h{white-space:nowrap;font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:calc(23.5 * var(--u));text-transform:uppercase;letter-spacing:.03em;margin:0;padding:0;}
.sh span{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-left:12px;font-family:'IBM Plex Mono',monospace;font-size:calc(12 * var(--u));color:#6E6A60;text-transform:uppercase;letter-spacing:.07em;}
.feed{border-left:1px solid #D8D2C4;margin-left:5px;}
.fi{position:relative;padding:0 0 15px 18px;}
.fi:before{content:"";position:absolute;left:-5px;top:5px;width:9px;height:9px;background:#F4F1EA;border:1.5px solid #17171B;border-radius:50%;box-sizing:border-box;}
.fi.latest:before{background:#C2410C;border-color:#C2410C;}
.fi .m{font-family:'IBM Plex Mono',monospace;font-size:calc(13 * var(--u));color:#6E6A60;}
.fi .m b{color:#17171B;font-weight:500;}
.fi .tx{font-size:calc(17 * var(--u));line-height:1.5;margin-top:3px;}
.fi.latest .tx{font-size:calc(18.5 * var(--u));}
.chip{font-family:'IBM Plex Mono',monospace;font-size:calc(12.5 * var(--u));padding:1px 6px;border-radius:2px;margin-left:8px;}
.chip.up{background:#E1EFE6;color:#1F7A4D;} .chip.down{background:#F6E1DE;color:#B42318;} .chip.flat{background:#E3DED2;color:#6E6A60;}
.lede{font-size:calc(17 * var(--u));line-height:1.5;border-left:3px solid #C2410C;padding:1px 0 1px 11px;margin:0 0 12px;}
.nx{border:1px solid #D8D2C4;background:#FBFAF6;}
.nx .row{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:10px 12px;border-bottom:1px solid #E6E1D6;}
.nx .row:last-child{border-bottom:none;}
.nx .nm{font-weight:600;font-size:calc(17 * var(--u));}
.nx .sub{font-size:calc(14 * var(--u));color:#6E6A60;margin-top:2px;}
.tag{font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:calc(15.5 * var(--u));letter-spacing:.05em;text-transform:uppercase;padding:3px 8px;border-radius:2px;white-space:nowrap;}
.tag.cheer{background:#1F7A4D;color:#fff;} .tag.fear{background:#B42318;color:#fff;} .tag.neutral{background:#E3DED2;color:#6E6A60;}
.ft{border:2px solid #17171B;padding:12px 14px;background:#FBFAF6;}
.ft .v{font-size:calc(40 * var(--u));}
.edge{width:100%;border-collapse:collapse;font-size:calc(15 * var(--u));border:none!important;margin:0;}
.edge tr{background:transparent!important;border:none!important;}
.edge th{font-family:'IBM Plex Mono',monospace;font-weight:400;font-size:calc(11 * var(--u));text-transform:uppercase;letter-spacing:.07em;color:#6E6A60;text-align:left;border-bottom:1px solid #17171B!important;padding:4px 6px;border-top:none!important;border-left:none!important;border-right:none!important;background:transparent!important;}
.edge td{padding:7px 6px;border:none!important;border-bottom:1px solid #E6E1D6!important;vertical-align:middle;}
.edge .who{font-size:calc(13 * var(--u));color:#6E6A60;}
.edge .b{height:calc(5 * var(--u));margin:3px 0;background:#E3DED2;position:relative;width:calc(90 * var(--u));}
.edge .b i{position:absolute;left:0;top:0;bottom:0;}
.edge .b.you i{background:#17171B;} .edge .b.fld i{background:#A39E92;}
.num{font-family:'IBM Plex Mono',monospace;font-size:calc(14 * var(--u));text-align:right;white-space:nowrap;}
.legend{font-family:'IBM Plex Mono',monospace;font-size:calc(12 * var(--u));color:#6E6A60;margin-top:6px;}
.legend i{display:inline-block;width:10px;height:5px;margin:0 4px 1px 10px;}
.catch{background:#FBFAF6;border:1px solid #D8D2C4;padding:10px 12px;font-size:calc(16 * var(--u));line-height:1.55;margin-top:8px;}
.wm{font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:calc(26 * var(--u));letter-spacing:.08em;line-height:1;}
.wm b{color:#F97316;}
.tagline{font-size:calc(14 * var(--u));color:#9B968A;margin:4px 0 6px;}
.lu .grp{font-family:'IBM Plex Mono',monospace;font-size:calc(11 * var(--u));letter-spacing:.12em;color:#9B968A;margin:12px 0 3px;text-transform:uppercase;}
.lu .p{display:flex;justify-content:space-between;font-size:calc(15 * var(--u));padding:4px 0;border-bottom:1px solid #2E2E35;}
.lu .cv{font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:calc(13.5 * var(--u));background:#F97316;color:#17171B;padding:0 4px;margin-left:6px;border-radius:2px;}
.lu .pts{font-family:'IBM Plex Mono',monospace;font-size:calc(14 * var(--u));}
.lu .tot{display:flex;justify-content:space-between;align-items:baseline;font-family:'Barlow Condensed',sans-serif;font-size:calc(24.5 * var(--u));font-weight:700;margin-top:10px;}
.status{font-family:'IBM Plex Mono',monospace;font-size:calc(12 * var(--u));color:#9B968A;margin-top:14px;letter-spacing:.04em;}
.status i{display:inline-block;width:6px;height:6px;border-radius:50%;margin-right:7px;vertical-align:1px;}
.ba{display:grid;grid-template-columns:1fr 1fr;border:2px solid #17171B;margin:8px 0;}
.ba>div{padding:8px 12px;} .ba>div+div{border-left:1px solid #D8D2C4;}
.sim{font-family:'IBM Plex Mono',monospace;font-size:calc(10 * var(--u));letter-spacing:.08em;background:#17171B;color:#F4F1EA;padding:1px 6px;border-radius:2px;vertical-align:2px;}
.rb{font-size:calc(13.5 * var(--u));line-height:1.5;padding:7px 0;border-bottom:1px solid #E6E1D6;}
.rb .rl{font-family:'IBM Plex Mono',monospace;font-size:calc(12 * var(--u));color:#6E6A60;}
.scn{font-family:'IBM Plex Mono',monospace;font-size:calc(12.5 * var(--u));margin:4px 0 0;padding-left:20px;}
.lb{width:100%;border-collapse:collapse;font-size:calc(14 * var(--u));border:none!important;}
.lb td{border:none!important;border-bottom:1px solid #E6E1D6!important;padding:6px 6px;}
.lb tr{background:transparent!important;border:none!important;}
.lb tr.me td{background:#17171B!important;color:#F4F1EA;font-weight:600;}
.lb .gap td{text-align:center;color:#A39E92;padding:2px;}
.wq2{font-size:calc(13.5 * var(--u));color:#6E6A60;margin:6px 0 2px;}
.wq{font-size:calc(14.5 * var(--u));line-height:1.45;margin:0 0 4px;}
.hyp{font-family:'IBM Plex Mono',monospace;font-size:calc(10 * var(--u));letter-spacing:.08em;border:1px solid #C2410C;color:#C2410C;padding:0 5px;border-radius:2px;vertical-align:2px;margin-left:4px;}
.verdict{font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:calc(22 * var(--u));text-transform:uppercase;letter-spacing:.02em;padding:8px 12px;margin:6px 0 0;color:#fff;}
.verdict.good{background:#1F7A4D;} .verdict.bad{background:#B42318;} .verdict.flat{background:#6E6A60;}
.story{border:1px solid #D8D2C4;border-top:none;background:#FBFAF6;}
.sr{display:grid;grid-template-columns:calc(92 * var(--u)) auto;column-gap:10px;padding:9px 12px;border-top:1px solid #E6E1D6;}
.sr:first-child{border-top:none;}
.sl{font-family:'IBM Plex Mono',monospace;font-size:calc(11 * var(--u));text-transform:uppercase;letter-spacing:.06em;color:#6E6A60;padding-top:4px;}
.sv{font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:calc(22 * var(--u));line-height:1.1;}
.sx{grid-column:2;font-size:calc(13.5 * var(--u));color:#3F3D38;line-height:1.4;margin-top:2px;}
@media (max-width:760px){.block-container{padding:2.8rem 0.8rem 0;}.hero{grid-template-columns:1fr 1fr;}.hero .c{border-bottom:1px solid #D8D2C4;}
.bug{flex-wrap:wrap;}.bug .ctx{margin-left:0;text-align:left;}.hero .c.main .v{font-size:calc(46 * var(--u));}}
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
    st.session_state.ball = 0


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
                        st.session_state.ball = 0
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
            st.session_state.ball = 0
        else:
            st.session_state.autoplay = False


def _set(v):
    st.session_state.step = max(0, min(last_step, v))
    st.session_state.ball = 0
    st.session_state.autoplay = False


def _next_ball():
    s0, b0 = st.session_state.step, st.session_state.get("ball", 0)
    st.session_state.autoplay = False
    if s0 >= last_step:
        return
    if b0 + 1 >= len(match.steps[s0 + 1].balls):  # last delivery of the over -> over-end state
        st.session_state.step, st.session_state.ball = s0 + 1, 0
    else:
        st.session_state.ball = b0 + 1


s = st.session_state.get("step", 0)
step = match.steps[s]
SCORE_RE = re.compile(r"^(\S+) (\d+/\d+) \(([\d.]+)\)$")
ball = st.session_state.get("ball", 0)
bsnap = match.steps[s + 1].balls[ball - 1] if (ball and s < last_step and ball <= len(match.steps[s + 1].balls)) else None
if not bsnap:
    ball = 0
cur_points = bsnap.points if bsnap else step.points
cur_vec = np.array([cur_points[p] for p in contest.pids], dtype=np.float32)
riv_now = contest.rival_mult @ cur_vec
cur_idx = s + 1 if bsnap else s


def rank_of(vec) -> int:
    return int(1 + (contest.rival_mult @ vec > float(vec @ mm)).sum())

# ------------------------------------------------------------------ scorebug
def inn_cell(short: str, idx_range: range) -> str:
    if cur_idx == 0 or cur_idx < idx_range.start:
        return f"<div class='inn dim'><div class='t'>{short}</div><div class='s'>Yet to bat</div></div>"
    at = min(cur_idx, idx_range.stop - 1)
    txt = bsnap.score if (bsnap and cur_idx in idx_range) else match.steps[at].score
    mt = SCORE_RE.match(txt)
    on = " on" if cur_idx in idx_range else ""
    sc, ov = (mt.group(2), mt.group(3)) if mt else (txt, "")
    return f"<div class='inn{on}'><div class='t'>{short}</div><div class='s'>{sc}<small>({ov})</small></div></div>"


pos = (f"Ball {bsnap.label} · {bsnap.text}" if bsnap else "Before the first ball") if (s == 0 or bsnap) else ("Full time" if s == last_step else
                                               f"Over {s - (n1 if s > n1 else 0)} · {'1st innings' if s <= n1 else 'chase'}")
H(f"""<div class='bug'><div class='brand'>{first_short} v {second_short}</div>
{inn_cell(first_short, range(1, inn_break))}{inn_cell(second_short, range(inn_break, last_step + 1))}
<div class='ctx'><div class='t'><span class='dot'></span>Replay</div><div class='r'>{pos}</div></div></div>""")

# ------------------------------------------------------------------ controls
cb = st.columns([0.9, 0.9, 1.1, 1.3, 1.1, 4.6], vertical_alignment="center")
cb[0].button("Restart", on_click=_set, args=(0,), width="stretch")
cb[1].button("Back", on_click=lambda: _set(st.session_state.step - 1), width="stretch", disabled=s == 0)
cb[2].button("Next ball", on_click=_next_ball, width="stretch", disabled=s >= last_step)
cb[3].button("Next over  →", type="primary", on_click=lambda: _set(st.session_state.step + 1),
             width="stretch", disabled=s >= last_step)
cb[4].toggle("Autoplay", key="autoplay")
cb[5].slider("Replay position", 0, last_step, key="step", label_visibility="collapsed",
             on_change=lambda: st.session_state.update(ball=0))


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
pts_now = float(cur_vec @ mm)
fa = float(riv_now.mean())
top10 = float(np.percentile(riv_now, 90))
SIM = "<span class='sim'>SIMULATED</span>"
if cur_idx == 0:
    rank_html = f"<div class='v'>—</div><div class='d'>All {n:,} teams start level {SIM}</div>"
else:
    r = rank_of(cur_vec)
    if bsnap:
        mv = int(tl["rank"][s]) - r if s >= 1 else None
        unit = "since over start"
    else:
        mv = move(s)
        unit = "this over"
    top = max(1, round(r / n * 100))
    mv_html = (f"<span class='up'>▲ {mv:,} {unit}</span>" if mv and mv > 0 else
               f"<span class='down'>▼ {-mv:,} {unit}</span>" if mv and mv < 0 else
               f"no change {unit}" if mv == 0 else "after the first over")
    rank_html = (f"<div class='v'>#{r:,}</div><div class='d'>{mv_html} · of {n:,} · top {top}% {SIM}</div>"
                 f"<div class='pbar'><i style='width:{100 - r / n * 100:.1f}%'></i></div>")
pd_ = pts_now - float(tl["points"][s]) if bsnap else (float(tl["points"][s] - tl["points"][s - 1]) if s else 0)
lead = pts_now - fa
score_now = bsnap.score if bsnap else step.score
mt = SCORE_RE.match(score_now)
need = None if bsnap else re.search(r"Need (\d+) more", step.summary)
match_cell = (f"<div class='v sm'>{mt.group(1)} {mt.group(2)}</div><div class='d'>{mt.group(3)} ov"
              f"{' · need ' + need.group(1) if need else ''}</div>") if (cur_idx and mt) else \
    "<div class='v sm'>0/0</div><div class='d'>Toss done, first ball soon</div>"
pts_note = (f"{pd_:+.0f} this over so far" if bsnap else f"+{pd_:.0f} this over") if cur_idx else "+4 per player for playing"
H(f"""<div class='hero'>
<div class='c main'><div class='k'>Your rank</div>{rank_html}</div>
<div class='c'><div class='k'>Your points</div><div class='v'>{pts_now:.0f}</div>
<div class='d'>{pts_note}</div></div>
<div class='c'><div class='k'>Field average</div><div class='v'>{fa:.0f}</div>
<div class='d'>you're <span class='{'up' if lead >= 0 else 'down'}'>{lead:+.0f}</span> · top 10% cut {top10:.0f}</div></div>
<div class='c'><div class='k'>Match</div>{match_cell}</div>
</div>""")

with st.container(key="main"):
    feedcol, mid, right = st.columns([1.15, 1, 1], gap="medium")

# ------------------------------------------------------------------ left: chart + feed
with mid:
    sh("Rank, over by over", f"{n - 1:,} rivals")
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
                  axis=alt.Axis(title=None, values=list(range(5, last_step, 5)), labelExpr=label_expr, labelOverlap="greedy"))
    y_max = int(max(tl["rank"][1:].max(), 10))
    y_enc = alt.Y("rank:Q", scale=alt.Scale(reverse=True, domain=[1, y_max * 1.05], nice=False),
                  axis=alt.Axis(title="RANK", format="~s", tickCount=5))
    layers = []
    if n1 < last_step:
        band = pd.DataFrame({"a": [n1 + 0.5], "b": [last_step], "t": ["CHASE"]})
        layers.append(alt.Chart(band).mark_rect(color="#EAE5DA", opacity=0.6).encode(x="a:Q", x2="b:Q"))
        layers.append(alt.Chart(band).mark_text(align="left", dx=6, dy=0, font="IBM Plex Mono", fontSize=12,
                                                color=MUTED).encode(x="a:Q", y=alt.value(10), text="t:N"))
    if len(df):
        tip = [alt.Tooltip("label:N", title="Over"), alt.Tooltip("score:N", title="Score"),
               alt.Tooltip("rank:Q", title="Your rank", format=","), alt.Tooltip("points:Q", title="Your points")]
        base = alt.Chart(df).encode(x=x_enc, y=y_enc, tooltip=tip)
        layers.append(base.mark_line(color=INK, strokeWidth=2.2, interpolate="monotone"))
        layers.append(base.mark_circle(color=INK, size=22, opacity=1))
        cur = df.tail(1).assign(txt=lambda d: d["rank"].map(lambda v: f"#{v:,}"))
        layers.append(alt.Chart(cur).mark_circle(color=ACCENT, size=110, opacity=1).encode(x=x_enc, y=y_enc))
        layers.append(alt.Chart(cur).mark_text(align="left", dx=9, dy=-10, font="Barlow Condensed", fontSize=22,
                                               fontWeight=700, color=ACCENT).encode(x=x_enc, y=y_enc, text="txt:N"))
    else:
        layers.append(alt.Chart(pd.DataFrame({"over": [1], "rank": [1]})).mark_point(opacity=0).encode(x=x_enc, y=y_enc))
    chart = alt.layer(*layers).properties(height=175, background="transparent").configure_axis(
        labelFont="IBM Plex Mono", titleFont="IBM Plex Mono", labelColor=MUTED, titleColor=MUTED,
        labelFontSize=13, titleFontSize=12, titleFontWeight="normal", gridColor="#E3DED2",
        domainColor=RULE, tickColor=RULE, titleAngle=0, titleAlign="left", titleY=-10, titleX=-38,
    ).configure_view(stroke=None)
    st.altair_chart(chart, theme=None, width="stretch")

with feedcol:
    sh("What moved you", "latest first")
    if s == 0 and not bsnap:
        H("<div class='lede'>Press <b>Next over</b> or turn on <b>Autoplay</b>. After every over you get one line "
          "on what happened to <i>your</i> rank and why. Players you own more than your rivals do push you up; "
          "popular players you skipped push you down.</div>")
    else:
        older = list(range(s - 5, 0, -1))
        have_old = notes_for(older, generate=False) if older else {}
        items = []
        if bsnap:
            prev = rank_of(np.array([match.steps[s].points[p] for p in contest.pids], dtype=np.float32)) if s else None
            ball_rows = []
            for bl in match.steps[s + 1].balls[:ball]:
                rk = rank_of(np.array([bl.points[p] for p in contest.pids], dtype=np.float32))
                ball_rows.append((bl, rk, (prev - rk) if prev else None))
                prev = rk
            for k, (bl, rk, mvb) in enumerate(reversed(ball_rows)):
                items.append(f"<div class='fi{' latest' if k == 0 else ''}'><div class='m'><b>Ball {bl.label}</b> · "
                             f"{bl.score}{chip(mvb)}</div><div class='tx'>{html.escape(bl.text)}. You're #{rk:,}</div></div>")
        for i, x in enumerate(range(s, 0, -1)):
            nt = notes.get(x) or have_old.get(x)
            if not nt:
                continue
            st_ = match.steps[x]
            items.append(f"<div class='fi{' latest' if (i == 0 and not bsnap) else ''}'><div class='m'><b>{st_.label}</b> · "
                         f"{st_.score}{chip(move(x))}</div><div class='tx'>{clean(nt['note'])}</div></div>")
        with st.container(key="feed"):
            H("<div class='feed'>" + "".join(items) + "</div>")

# ------------------------------------------------------------------ right side: tabs
WANT = {("batting", "cheer"): "Want runs", ("batting", "fear"): "Want him out",
        ("bowling", "cheer"): "Want wickets", ("bowling", "fear"): "Want him hit",
        ("batting", "neutral"): "Low stake", ("bowling", "neutral"): "Low stake"}
YOU = {"your Captain (2x)": "your C", "your Vice-captain (1.5x)": "your VC",
       "in your team": "yours", "NOT in your team": "not yours"}
MULT_TXT = {2.0: "your captain (2x)", 1.5: "your vice-captain (1.5x)", 1.0: "in your team (1x)", 0.0: "not in your team"}

with mid:
    t_next, t_lb = st.tabs(["Next ball" if bsnap else "Next over", "Leaderboard"])
with right:
    t_wi, t_edge, t_cu = st.tabs(["What if…", "Where you differ", "Catch me up"])

# ---------------- next over / next ball
with t_next:
    if s >= last_step:
        H(f"<div class='ft'><div class='k'>{html.escape(match.result)}</div>"
          f"<div class='v'>#{rank_of(cur_vec):,}</div>"
          f"<div class='d'>Your final rank of {n:,}, with {pts_now:.0f} points "
          f"(field average {fa:.0f}). {SIM}</div></div>")
    else:
        nxt = (bsnap.next if bsnap else match.steps[s].next_over) or {}
        bowler_name = match.name(nxt["bowler"]) if nxt else ""
        H(f"<div class='d' style='margin:0 0 8px'>{bowler_name} to bowl. Whose success helps you?</div>")
        if not bsnap and s >= 1 and s in notes and notes[s].get("root"):
            H(f"<div class='lede'>{clean(notes[s]['root'])}</div>")
        rows = []
        for r_ in contest.root_for(mm, s, nxt):
            rows.append(
                f"<div class='row'><div><div class='nm'>{html.escape(r_['player'])}</div>"
                f"<div class='sub'>{r_['team']} · {r_['job']} · {YOU[r_['you']]} · "
                f"{r_['picked_by_pct']}% own</div></div>"
                f"<span class='tag {r_['verdict']}'>{WANT[(r_['job'], r_['verdict'])]}</span></div>")
        H("<div class='nx'>" + "".join(rows) + "</div>")

# ---------------- what-if simulator
wi_vec = None
with t_wi:
    with st.container(key="wi"):
        stt = whatif.state_at(match, s, ball)
        base_key = (team_key, s, ball)
        if st.session_state.get("wi_base") != base_key:
            st.session_state.wi_base = base_key
            st.session_state.wi_events = []
        events = st.session_state.wi_events
        if not stt.next:
            H("<div class='d'>The match is over. Go back to any over to try a what-if.</div>")
        else:
            bowler = stt.next["bowler"]
            out_now = whatif.outs_in(events)
            if not events:
                H("<div class='wq'>Test a ball that <b>hasn't happened yet</b> and see how it would move your rank. "
                  f"<span class='hyp'>HYPOTHETICAL</span></div>")
            nth = "What if, on the next ball" if not events else "Then, on the ball after that"
            batters = [b for b in stt.next["batters"] if b not in out_now and not stt.stats[b].out]
            H(f"<div class='wq2'>{nth} from {html.escape(match.name(bowler))}…</div>")
            c1, c2 = st.columns([1, 1])
            batter = c1.selectbox("Batter", batters, format_func=match.name, key="wi_bat",
                                  label_visibility="collapsed") if batters else None
            ev_type = c2.selectbox("Event", list(whatif.EVENT_TYPES), index=3, key="wi_type2",
                                   format_func=lambda k: whatif.VERB[k], label_visibility="collapsed")
            fielder = None
            if ev_type == "caught":
                fielder = st.selectbox("Caught by", whatif.fielding_side(match, stt), format_func=match.name,
                                       key="wi_f")
            b1, b2 = st.columns([1.4, 1])
            if b1.button("Show me" if not events else "+ Add another ball", type="primary", width="stretch",
                         disabled=not batter):
                ev = {"type": ev_type, "batter": batter, "bowler": bowler, "fielder": fielder}
                errs = whatif.validate(match, stt, events, ev)
                if errs:
                    for e in errs:
                        st.error(e)
                else:
                    events.append(ev)
                    st.rerun()
            if b2.button("Start over", width="stretch", disabled=not events):
                events.clear()
                st.rerun()
            if not events:
                H("<div class='d' style='margin-top:8px'>Try these: a six by <b>your captain</b>, then a six by a "
                  "batter <b>you don't own</b>. Notice how the second one can push you down.</div>")
            else:
                res = whatif.run(contest, mm, stt, events)
                wi_vec = res["_after_vec"]
                story = whatif.plain_story(res)
                rb, ra = res["rank_before"], res["rank_after"]
                scen = " → ".join(
                    f"{match.name(e['batter'])} {whatif.VERB[e['type']]}" for e in events)
                H(f"<div class='d' style='margin-top:8px'>If <b>{html.escape(scen)}</b>:</div>")
                H(f"<div class='verdict {story['tone']}'>{html.escape(story['verdict'])}</div>")
                H(f"""<div class='story'>
<div class='sr'><div class='sl'>You</div><div class='sv'>{story['you_gain']:+g} pts</div>
<div class='sx'>{html.escape(story['you_reason'])}</div></div>
<div class='sr'><div class='sl'>Your rivals</div><div class='sv'>{story['rivals_gain']:+g} pts</div>
<div class='sx'>on average. {html.escape(story['rivals_reason'])}</div></div>
<div class='sr'><div class='sl'>Your rank</div><div class='sv'>#{rb:,} → #{ra:,}</div>
<div class='sx'>{html.escape(story['rank_reason'])}</div></div>
</div>""")
                H(f"<div class='lede' style='margin-top:10px'><b>Why:</b> {html.escape(story['takeaway'])}</div>")
                with st.expander("See the exact points maths"):
                    for p_ in res["players"]:
                        rules = " + ".join(f"{lbl} {v:+g}" for lbl, v in p_["rules"])
                        H(f"<div class='rb'><b>{html.escape(p_['player'])}</b>: {html.escape(rules)} = "
                          f"<b>{p_['delta']:+g} pts</b><br><span class='rl'>"
                          f"{MULT_TXT.get(p_['your_multiplier'], '')} → {p_['your_gain']:+g} for you · "
                          f"{p_['picked_by_pct']}% of rivals own him → rivals {p_['field_avg_gain']:+g} on average"
                          f"</span></div>")
                    H("<div class='legend'>Rivals are 10,000 simulated teams. "
                      + " ".join(html.escape(w) for w in res["warnings"]) + "</div>")

# ---------------- simulated leaderboard
with t_lb:
    with st.container(key="lb"):
        vec = wi_vec if wi_vec is not None else cur_vec
        lb_rows = contest.leaderboard(mm, vec)
        H(f"<div class='d' style='margin:0 0 6px'>{SIM} league of {n:,} teams"
          f"{' · <b>with your what-if applied</b>' if wi_vec is not None else ''}. "
          f"Rank = 1 + teams with more points (ties share a rank).</div>")
        trs = []
        for r_ in lb_rows:
            if r_.get("gap"):
                trs.append("<tr class='gap'><td colspan='4'>⋯</td></tr>")
                continue
            trs.append(f"<tr class='{'me' if r_['you'] else ''}'><td class='num'>#{r_['rank']:,}</td>"
                       f"<td>{r_['name']}</td><td class='who'>C: {html.escape(r_['captain'])}</td>"
                       f"<td class='num'>{r_['points']:g}</td></tr>")
        H("<table class='lb'>" + "".join(trs) + "</table>")

# ---------------- catch me up
with t_cu:
    with st.container(key="cu"):
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

# ---------------- where you differ
with t_edge:
    trs = []
    for r_ in contest.differentials(mm)[:8]:
        i = contest.pids.index(r_["pid"])
        you_m, fld_m = float(mm[i]), float(contest.eo[i])
        eff = "<span class='up'>helps</span>" if r_["leverage"] > 0 else "<span class='down'>hurts</span>"
        trs.append(
            f"<tr><td><div>{html.escape(r_['player'])}</div><div class='who'>{r_['team']} · "
            f"{YOU[r_['you']]} · {r_['picked_by_pct']}% own</div></td>"
            f"<td><div class='b you'><i style='width:{you_m / 2 * 100:.0f}%'></i></div>"
            f"<div class='b fld'><i style='width:{min(fld_m, 2) / 2 * 100:.0f}%'></i></div></td>"
            f"<td>{eff}</td><td class='num'>{cur_points[r_['pid']]}</td></tr>")
    with st.container(key="edge"):
        H("<table class='edge'><tr><th>Player</th><th>Stake</th><th>His runs</th><th class='num'>Pts</th></tr>"
          + "".join(trs) + "</table>"
          "<div class='legend'>Stake:<i style='background:#17171B'></i>you<i style='background:#A39E92'></i>"
          "rivals' average (captain = 2x)</div>")

# ------------------------------------------------------------------ sidebar: lineup + status
with st.sidebar:
    pts = cur_points
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
with st.sidebar.expander("How it works"):
    st.markdown(f"""
**The problem.** A fantasy player watches *their 11 against everyone else's*. The app gives them a points total and
a rank that moves every over, but not **why** it moved or **what to hope for next**.

**How it works**
1. **Scoring engine (exact, no AI).** Replays real ball-by-ball data and computes Dream11-style T20 points after every over.
2. **Contest (simulated).** {n - 1:,} rival teams are generated from *pre-match* popularity, which gives a rank and
   "picked by %" for every player. With real contest data the maths is the same.
3. **Gemini.** Receives only the computed facts and writes the short notes in English or Hinglish. It explains and never
   calculates. If it's unavailable, plain template notes are shown instead.

**What if… (try it).** Pick a delivery (dot, 1, 4, 6, wicket). The engine applies it to the current match state,
recomputes every player from the full updated stats (strike-rate, economy and milestone bands included), re-scores
all {n:,} teams and re-ranks them. Each result shows the exact rules that produced the points.

**Why "Next over" works:** a player helps you only when you own him *more than the field does*. A captain that few
rivals captained is worth cheering. A player in 90% of teams that you left out hurts you every time he scores.

**Scoring ruleset `{RULESET}` (approximation):** +{SCORING['run']} per run, +{SCORING['four_bonus']} per four, +{SCORING['six_bonus']} per six,
25/50/75/100-run bonuses, duck −2, wicket +{SCORING['wicket']} (+8 bowled/LBW), 3/4/5-wicket bonuses, maiden +12, dot ball +1,
catch +8, stumping +12, run-out +12/+6, economy and strike-rate bonuses, captain 2x, vice-captain 1.5x, +4 for playing.

Match data from [Cricsheet](https://cricsheet.org) (ODC-By). Rival teams and popularity are simulated. Not affiliated with Dream11.
""")
