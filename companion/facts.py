"""Builds the exact 'fact packets' the AI is allowed to talk about, plus preset teams."""
from __future__ import annotations

from .contest import Contest, ROLES

# A typical fan team for the default match: popular captain (Abhishek), but a
# differential vice-captain (Tilak Varma) -- picked BEFORE the match on reputation.
PRESETS = {
    "1539433": {
        "picks": ["SV Samson", "Abhishek Sharma", "V Suryavanshi", "Tilak Varma", "FH Allen",
                  "GD Phillips", "AR Patel", "Nithish Kumar Reddy", "Arshdeep Singh", "MJ Henry",
                  "LH Ferguson"],
        "captain": "Abhishek Sharma",
        "vice": "Tilak Varma",
    },
}


def preset_team(contest: Contest) -> dict:
    m = contest.match
    if m.id in PRESETS:
        return PRESETS[m.id]
    # Generic: most-picked valid XI, with the 2nd-most-captained player as C for a little spice.
    order = sorted(m.players, key=lambda p: -m.players[p]["pick"])
    picks = []
    for r in ROLES:  # one of each role first
        picks.append(next(p for p in order if m.players[p]["role"] == r and p not in picks))
    for p in order:
        if len(picks) == 11:
            break
        if p not in picks:
            picks.append(p)
    caps = sorted(picks, key=lambda p: -m.players[p]["cap"])
    return {"picks": picks, "captain": caps[1], "vice": caps[0]}


def _root_rows(contest: Contest, mm, step: int) -> list[dict]:
    return [{k: r[k] for k in ("player", "team", "job", "you", "picked_by_pct", "verdict")}
            for r in contest.root_for(mm, step)]


def step_facts(contest: Contest, mm, tl: dict, s: int) -> dict:
    st = contest.match.steps[s]
    return {
        "step": s,
        "over": st.label,
        "score_after_over": st.score,
        "what_happened": st.summary,
        "rank_before": int(tl["rank"][s - 1]) if s > 1 else None,
        "rank_after": int(tl["rank"][s]),
        "contest_size": tl["n"],
        "your_points_this_over": round(float(tl["points"][s] - tl["points"][s - 1]), 1),
        "field_avg_points_this_over": round(float(tl["field_avg"][s] - tl["field_avg"][s - 1]), 1),
        "drivers": contest.drivers(mm, s - 1, s, top=3),
        "next_over": _root_rows(contest, mm, s),
    } | _friends(contest, tl, s - 1, s)


def _friends(contest: Contest, tl: dict, a: int, b: int) -> dict:
    if not contest.names:
        return {}
    out = contest.movers(float(tl["points"][a]), float(tl["points"][b]),
                         contest.rival_totals[:, a], contest.rival_totals[:, b])
    out["league"] = f"private league with friends: {', '.join(contest.names)}"
    return out


def catch_up_facts(contest: Contest, mm, tl: dict, a: int, b: int) -> dict:
    steps = contest.match.steps
    moments = [steps[i].summary for i in range(a + 1, b + 1)
               if "WICKET" in steps[i].summary or "six" in steps[i].summary or "maiden" in steps[i].summary]
    return {
        "from": steps[a].label,
        "to": steps[b].label,
        "rank_from": int(tl["rank"][a]),
        "rank_to": int(tl["rank"][b]),
        "contest_size": tl["n"],
        "your_points_gained": round(float(tl["points"][b] - tl["points"][a]), 1),
        "field_avg_points_gained": round(float(tl["field_avg"][b] - tl["field_avg"][a]), 1),
        "drivers": contest.drivers(mm, a, b, top=3),
        "key_moments": moments[-6:],
        "score_now": steps[b].score,
        "next_over": _root_rows(contest, mm, b),
    } | _friends(contest, tl, a, b)
