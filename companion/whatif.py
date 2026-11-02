"""What-if scenario simulator.

Take the CURRENT match state (any over or ball), apply hypothetical deliveries
(dot / 1 / 4 / 6 / wicket), recompute every player's points from the full updated
stats (so strike-rate, economy and milestone bands are re-derived, never double
counted), then re-score the user AND all 10,000 rival lineups and re-rank.

The base state is never mutated: each run works on a copy (so "Reset" is exact).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .contest import Contest
from .engine import RULE_LABELS, Match

EVENT_TYPES = {
    "dot": "Dot ball", "1": "1 run", "4": "Four", "6": "Six",
    "bowled": "Wicket (bowled)", "lbw": "Wicket (LBW)", "caught": "Wicket (caught)",
}
BAT_EVENTS = ("dot", "1", "4", "6")
WICKETS = ("bowled", "lbw", "caught")
MAX_BALLS = 6


@dataclass
class State:
    stats: dict          # pid -> Stats (never mutated here)
    label: str
    score: str
    next: dict | None    # {"batters": [striker, non_striker], "bowler": pid}


def state_at(match: Match, step: int, ball: int = 0) -> State:
    """ball = 0 -> end of over `step`; ball = k -> after k-th delivery of the NEXT over."""
    if ball == 0:
        st = match.steps[step]
        return State(st.stats, st.label, st.score, st.next_over)
    b = match.steps[step + 1].balls[ball - 1]
    return State(b.stats, f"over {b.label}", b.score, b.next)


def fielding_side(match: Match, state: State) -> list[str]:
    if not state.next:
        return []
    team = match.players[state.next["bowler"]]["team"]
    return [p for p, v in match.players.items() if v["team"] == team]


def outs_in(scenario: list[dict]) -> set:
    return {e["batter"] for e in scenario if e["type"] in WICKETS}


def validate(match: Match, state: State, scenario: list[dict], event: dict) -> list[str]:
    errs = []
    if not state.next:
        return ["The match is over: there is no next delivery to simulate."]
    if len(scenario) >= MAX_BALLS:
        errs.append(f"A scenario covers at most one over ({MAX_BALLS} balls). Press Reset to start again.")
    if event["type"] not in EVENT_TYPES:
        errs.append("Unknown event.")
    bat = event.get("batter")
    if bat not in state.next["batters"]:
        errs.append("Choose one of the two batters at the crease.")
    if bat in outs_in(scenario) or (bat in state.stats and state.stats[bat].out):
        errs.append(f"{match.name(bat)} is already out in this scenario.")
    if event["type"] == "caught":
        f = event.get("fielder")
        if f not in fielding_side(match, state):
            errs.append("The catch must be taken by a player from the fielding side.")
    return errs


def apply(stats: dict, ev: dict) -> None:
    """Apply one legal delivery to a stats dict (mutates the given copy)."""
    bat, bwl = stats[ev["batter"]], stats[ev["bowler"]]
    bat.balls += 1
    bwl.legal_balls += 1
    t = ev["type"]
    if t in BAT_EVENTS:
        r = 0 if t == "dot" else int(t)
        bat.runs += r
        bwl.conceded += r
        if r == 4:
            bat.fours += 1
        elif r == 6:
            bat.sixes += 1
        if r == 0:
            bwl.dots += 1
    else:
        bat.out = True
        bwl.dots += 1
        bwl.wkts += 1
        if t in ("bowled", "lbw"):
            bwl.bowled_lbw += 1
        if t == "caught":
            stats[ev["fielder"]].catches += 1  # fielder may be the bowler (caught & bowled)


def describe(match: Match, ev: dict) -> str:
    b, w = match.name(ev["batter"]), match.name(ev["bowler"])
    t = ev["type"]
    if t == "caught":
        f = ev["fielder"]
        return f"{b} caught by {match.name(f)} off {w}" if f != ev["bowler"] else f"{b} caught & bowled by {w}"
    if t in WICKETS:
        return f"{b} {'bowled' if t == 'bowled' else 'LBW'} by {w}"
    return f"{EVENT_TYPES[t]}: {b} off {w}"


def _vec(contest: Contest, stats: dict) -> np.ndarray:
    m = contest.match
    return np.array([stats[p].points(m.players[p]["role"]) for p in contest.pids], dtype=np.float32)


def run(contest: Contest, mm: np.ndarray, state: State, scenario: list[dict]) -> dict:
    match = contest.match
    new = {p: replace(s) for p, s in state.stats.items()}
    for ev in scenario:
        apply(new, ev)

    before, after = _vec(contest, state.stats), _vec(contest, new)
    my_b, my_a = float(before @ mm), float(after @ mm)
    riv_b, riv_a = contest.rival_mult @ before, contest.rival_mult @ after
    ahead_b, ahead_a = riv_b > my_b, riv_a > my_a
    rank_b, rank_a = int(1 + ahead_b.sum()), int(1 + ahead_a.sum())

    players = []
    for i, p in enumerate(contest.pids):
        d = float(after[i] - before[i])
        if d == 0:
            continue
        role = match.players[p]["role"]
        bb, ab = state.stats[p].breakdown(role), new[p].breakdown(role)
        items = [(RULE_LABELS.get(k, k), ab.get(k, 0) - bb.get(k, 0))
                 for k in sorted(set(bb) | set(ab)) if ab.get(k, 0) != bb.get(k, 0)]
        mult = float(mm[i])
        players.append({
            "pid": p, "player": match.name(p), "team": match.short_team(p),
            "points_before": float(before[i]), "points_after": float(after[i]), "delta": d,
            "rules": items,
            "your_multiplier": mult, "your_gain": d * mult,
            "picked_by_pct": round(float(contest.pick_pct[i])),
            "captained_by_pct": round(float(contest.cap_pct[i])),
            "field_avg_multiplier": round(float(contest.eo[i]), 2),
            "field_avg_gain": round(d * float(contest.eo[i]), 1),
        })
    players.sort(key=lambda r: -abs(r["your_gain"]) - abs(r["field_avg_gain"]))

    # who overtakes you, and how much of the main player they own
    main_i = int(np.argmax(np.abs(after - before)))
    passers = ~ahead_b & ahead_a
    pm = contest.rival_mult[passers, main_i]
    passer_mults = {"captain": int((pm == 2.0).sum()), "vice": int((pm == 1.5).sum()),
                    "owner": int((pm == 1.0).sum()), "none": int((pm == 0).sum())}

    warnings = []
    if any(e["type"] in WICKETS for e in scenario):
        warnings.append("The incoming batter after a wicket is unknown, so only the other batter can face next.")
    warnings.append("Maiden bonuses and extras (wides/no-balls) are not simulated.")
    return {
        "scenario": [describe(match, e) for e in scenario],
        "points_before": my_b, "points_after": my_a,
        "rank_before": rank_b, "rank_after": rank_a, "contest_size": len(riv_b) + 1,
        "teams_you_passed": int((ahead_b & ~ahead_a).sum()),
        "teams_that_passed_you": int((~ahead_b & ahead_a).sum()),
        "field_avg_gain": round(float(riv_a.mean() - riv_b.mean()), 1),
        "all_level_before": bool(np.all(riv_b == my_b)),
        "teams_behind_after": int((riv_a < my_a).sum()),
        "teams_ahead_after": int(ahead_a.sum()),
        "main_player": match.name(contest.pids[main_i]),
        "main_player_delta": float(after[main_i] - before[main_i]),
        "passer_mults": passer_mults,
        "players": players,
        "warnings": warnings,
        "_after_vec": after,
    }


VERB = {"dot": "plays a dot ball", "1": "takes 1 run", "4": "hits a four", "6": "hits a six",
        "bowled": "is bowled", "lbw": "is out LBW", "caught": "is caught"}


def plain_story(r: dict) -> dict:
    """Plain-language version of a result: a verdict + three short 'why' rows + a takeaway
    that is always consistent with the actual rank movement."""
    n = r["contest_size"]
    move = r["rank_before"] - r["rank_after"]
    you_gain = r["points_after"] - r["points_before"]
    name, d = r["main_player"], r["main_player_delta"]
    pm = r["passer_mults"]

    if r["all_level_before"]:
        behind, ahead = r["teams_behind_after"], r["teams_ahead_after"]
        if ahead < (n - 1) / 2:
            verdict, tone = f"Good for you: only {ahead:,} of {n - 1:,} teams would be ahead of you", "good"
        elif ahead > behind:
            verdict, tone = f"Bad for you: {ahead:,} teams would be ahead of you", "bad"
        else:
            verdict, tone = "No change: you'd stay level with most teams", "flat"
    elif move > 0:
        verdict, tone = f"Good for you: you'd climb {move:,} places", "good"
    elif move < 0:
        verdict, tone = f"Bad for you: you'd drop {-move:,} places", "bad"
    else:
        verdict, tone = "No change to your rank", "flat"

    bits = []
    for p in r["players"]:
        if not p["delta"]:
            continue
        m = p["your_multiplier"]
        if m == 2:
            bits.append(f"{p['player']} is your captain, so his {p['delta']:+g} counts double ({p['your_gain']:+g})")
        elif m == 1.5:
            bits.append(f"{p['player']} is your vice-captain, so his {p['delta']:+g} counts 1.5x ({p['your_gain']:+g})")
        elif m == 1:
            bits.append(f"{p['player']} is in your team, so you get his {p['delta']:+g}")
        else:
            bits.append(f"{p['player']} is not in your team, so his {p['delta']:+g} doesn't count for you")
    you_reason = "; ".join(bits[:2]) + "." if bits else "Nobody in your team is affected."

    main = max(r["players"], key=lambda p: abs(p["field_avg_gain"]), default=None)
    if main and main["field_avg_gain"]:
        rivals_reason = (f"{main['picked_by_pct']}% of rival teams have {main['player']} "
                         f"({main['captained_by_pct']}% as captain), so they "
                         f"{'gain' if main['field_avg_gain'] > 0 else 'lose'} too.")
    else:
        rivals_reason = "Rivals' points barely change."

    higher = pm["captain"] + pm["vice"]
    if r["all_level_before"]:
        level = n - 1 - r["teams_ahead_after"] - r["teams_behind_after"]
        rank_reason = (f"Before the first ball every team is level. After this, {r['teams_ahead_after']:,} teams "
                       f"would be ahead of you, {level:,} still level with you, and {r['teams_behind_after']:,} behind.")
        if r["teams_ahead_after"] and higher >= 0.5 * r["teams_ahead_after"]:
            takeaway = (f"The only teams that beat you are the ones that made {name} captain or vice-captain: "
                        f"his {d:+g} counts 2x or 1.5x for them. Everyone else gains less than you.")
        elif tone == "good":
            takeaway = "You gain more than most teams, so you jump ahead of them."
        else:
            takeaway = "Most teams gain more than you from this ball, so they move ahead."
    elif move < 0:
        rank_reason = f"{r['teams_that_passed_you']:,} teams that were behind you would move ahead of you."
        if you_gain > 0 and higher >= 0.5 * max(1, r["teams_that_passed_you"]):
            takeaway = (f"You gain {you_gain:+g}, but the teams just behind you made {name} captain or "
                        f"vice-captain, so they gain more ({d * 2:+g} or {d * 1.5:+g}) and overtake you.")
        elif you_gain > 0:
            takeaway = f"You gain {you_gain:+g}, but the teams just behind you gain even more, so they overtake you."
        else:
            takeaway = (f"Teams just behind you own {name} and you don't, so they gain and overtake you, "
                        f"even though your own points don't fall.")
    elif move > 0:
        rank_reason = f"You would overtake {r['teams_you_passed']:,} teams."
        takeaway = "You gain more than the teams just ahead of you, so you move past them."
    else:
        rank_reason = "Nobody overtakes you and you overtake nobody."
        takeaway = "The teams around you gain about the same as you, so your rank doesn't move."
    rank_txt = (f"Level → #{r['rank_after']:,}" if r["all_level_before"]
                else f"#{r['rank_before']:,} → #{r['rank_after']:,}")
    return {"verdict": verdict, "tone": tone, "you_gain": you_gain, "you_reason": you_reason,
            "rivals_gain": r["field_avg_gain"], "rivals_reason": rivals_reason, "rank_reason": rank_reason,
            "rank_txt": rank_txt, "takeaway": takeaway}


def template_explanation(r: dict) -> str:
    move = r["rank_before"] - r["rank_after"]
    lines = []
    for p in r["players"][:3]:
        own = ("your captain (2x)" if p["your_multiplier"] == 2 else
               "your vice-captain (1.5x)" if p["your_multiplier"] == 1.5 else
               "in your team" if p["your_multiplier"] == 1 else "not in your team")
        lines.append(f"{p['player']} {p['delta']:+g} pts ({own}) is worth {p['your_gain']:+g} to you; "
                     f"{p['picked_by_pct']}% of rivals own him, so the field gains {p['field_avg_gain']:+g} on average.")
    verdict = (f"Net: you move up {move:,} places." if move > 0 else
               f"Net: you drop {-move:,} places." if move < 0 else
               "Net: no change in rank, because your rivals gain about as much as you.")
    return " ".join(lines + [verdict])
