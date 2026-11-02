"""Scoring, what-if and rank tests (mirrors the brief's acceptance criteria).

Run:  .venv\\Scripts\\python -m pytest -q
"""
from dataclasses import replace

import numpy as np
import pytest

from matchlens import ai, whatif
from matchlens.contest import Contest
from matchlens.engine import SCORING, Stats, load_match
from matchlens.facts import preset_team, step_facts

MATCH_ID = "1539433"  # NZ vs IND, 2nd T20I, 24 Oct 2026


@pytest.fixture(scope="module")
def contest():
    return Contest.build(load_match(MATCH_ID))


@pytest.fixture(scope="module")
def team(contest):
    t = preset_team(contest)
    return t, contest.my_mult(t["picks"], t["captain"], t["vice"])


def mid_state(contest, step=26):
    return whatif.state_at(contest.match, step, 0)


# ------------------------------------------------------------------ engine vs real data
def test_innings_totals_match_scorecard(contest):
    steps = contest.match.steps
    assert steps[20].score == "NZ 180/5 (20)"
    assert steps[-1].score == "IND 181/6 (19.5)"


def test_known_final_points_and_rank_regression(contest, team):
    pts = contest.match.steps[-1].points
    assert pts["MJ Santner"] == 159 and pts["GD Phillips"] == 142 and pts["Tilak Varma"] == 115
    _, mm = team
    assert int(contest.timeline(mm)["rank"][-1]) == 3556


def test_ball_snapshots_end_at_over_state(contest):
    for st in contest.match.steps[1:]:
        assert st.balls[-1].points == st.points


# ------------------------------------------------------------------ rules
def test_six_is_runs_plus_bonus_not_hardcoded():
    s = Stats()
    before = s.breakdown("BAT")
    s.runs, s.balls, s.sixes = 6, 1, 1
    after = s.breakdown("BAT")
    assert after["runs"] - before.get("runs", 0) == 6 * SCORING["run"]
    assert after["six_bonus"] == SCORING["six_bonus"]


def test_milestone_bonus_applied_once_highest_only():
    s = Stats(runs=48, balls=30)
    assert s.breakdown("BAT").get("milestone_25") == 4
    s.runs, s.balls, s.sixes = 54, 31, 1
    b = s.breakdown("BAT")
    assert b.get("milestone_50") == 8 and "milestone_25" not in b


def test_duck_not_for_bowlers():
    s = Stats(out=True, balls=2)
    assert s.breakdown("BAT")["duck"] == -2
    assert "duck" not in s.breakdown("BOWL")


# ------------------------------------------------------------------ what-if: multipliers
def _six_by(contest, mm, state, pid):
    return whatif.run(contest, mm, state, [{"type": "6", "batter": pid, "bowler": state.next["bowler"]}])


def test_captain_multiplier_applied_exactly_once(contest):
    st = mid_state(contest)
    bat = st.next["batters"][0]
    others = [p for p in contest.pids if p != bat]
    picks = [bat] + [p for p in others if contest.match.players[p]["role"] != "BOWL"][:10]
    mm = contest.my_mult(picks, bat, picks[1])
    r = _six_by(contest, mm, st, bat)
    batter_delta = next(p["delta"] for p in r["players"] if p["pid"] == bat)
    bowler_delta = sum(p["delta"] * p["your_multiplier"] for p in r["players"] if p["pid"] != bat)
    assert r["points_after"] - r["points_before"] == pytest.approx(2.0 * batter_delta + bowler_delta)


def test_vice_captain_multiplier_applied_exactly_once(contest):
    st = mid_state(contest)
    bat = st.next["batters"][0]
    others = [p for p in contest.pids if p != bat]
    picks = [bat] + others[:10]
    mm = contest.my_mult(picks, picks[1], bat)
    r = _six_by(contest, mm, st, bat)
    batter = next(p for p in r["players"] if p["pid"] == bat)
    assert batter["your_gain"] == pytest.approx(1.5 * batter["delta"])


def test_shared_player_lifts_every_owner(contest, team):
    _, mm = team
    st = mid_state(contest)
    bat = st.next["batters"][0]
    r = _six_by(contest, mm, st, bat)
    i = contest.pids.index(bat)
    after, before = r["_after_vec"], np.array([st.stats[p].points(contest.match.players[p]["role"])
                                               for p in contest.pids], dtype=np.float32)
    gain = contest.rival_mult @ after - contest.rival_mult @ before
    owners = contest.rival_mult[:, i] > 0
    assert (gain[owners] > gain[~owners].max() - 1e-6).all() or owners.all()


def test_wicket_with_fielder_updates_batter_bowler_fielder(contest, team):
    _, mm = team
    st = mid_state(contest)
    bat, bowler = st.next["batters"][0], st.next["bowler"]
    fielder = next(p for p in whatif.fielding_side(contest.match, st) if p != bowler)
    r = whatif.run(contest, mm, st, [{"type": "caught", "batter": bat, "bowler": bowler, "fielder": fielder}])
    by = {p["pid"]: dict(p["rules"]) for p in r["players"]}
    assert by[bowler]["wicket"] == SCORING["wicket"]
    assert by[fielder]["catch"] == SCORING["catch"]
    assert bat in by or st.stats[bat].runs > 0  # batter changes only if duck/SR band moves


def test_bowled_gets_bonus(contest, team):
    _, mm = team
    st = mid_state(contest)
    r = whatif.run(contest, mm, st, [{"type": "bowled", "batter": st.next["batters"][0],
                                      "bowler": st.next["bowler"], "fielder": None}])
    rules = dict(next(p for p in r["players"] if p["pid"] == st.next["bowler"])["rules"])
    assert rules["bowled/LBW bonus"] == SCORING["bowled_lbw_bonus"]


def test_reset_restores_exact_state(contest, team):
    _, mm = team
    st = mid_state(contest)
    snapshot = {p: replace(s) for p, s in st.stats.items()}
    _six_by(contest, mm, st, st.next["batters"][0])
    assert st.stats == snapshot


def test_validation_rejects_impossible(contest):
    st = mid_state(contest)
    bat, bowler = st.next["batters"][0], st.next["bowler"]
    out = [{"type": "bowled", "batter": bat, "bowler": bowler}]
    assert whatif.validate(contest.match, st, out, {"type": "6", "batter": bat, "bowler": bowler})
    assert whatif.validate(contest.match, st, [], {"type": "6", "batter": "SV Samson", "bowler": bowler}) \
        if "SV Samson" not in st.next["batters"] else True
    six = [{"type": "dot", "batter": bat, "bowler": bowler}] * whatif.MAX_BALLS
    assert whatif.validate(contest.match, st, six, {"type": "dot", "batter": bat, "bowler": bowler})


def test_rank_tie_convention(contest, team):
    _, mm = team
    lb = contest.leaderboard(mm, np.array([4] * len(contest.pids), dtype=np.float32))
    me = next(r for r in lb if r.get("you"))
    assert me["rank"] == 1  # everyone level before the first ball -> everyone shares rank 1


# ------------------------------------------------------------------ AI fallback
def test_llm_unavailable_uses_templates(contest, team, monkeypatch):
    monkeypatch.setattr(ai, "available", lambda: False)
    _, mm = team
    tl = contest.timeline(mm)
    f = step_facts(contest, mm, tl, 5)
    assert ai.over_notes([f], "en") is None
    assert "#" in ai.template_note(f)
    st = mid_state(contest)
    r = _six_by(contest, mm, st, st.next["batters"][0])
    assert ai.whatif_note({k: v for k, v in r.items() if not k.startswith("_")}, "en") is None
    assert "Net:" in whatif.template_explanation(r)


def test_leaderboard_mid_match_has_you_once_in_order(contest, team):
    _, mm = team
    vec = np.array([contest.match.steps[27].points[p] for p in contest.pids], dtype=np.float32)
    lb = [r for r in contest.leaderboard(mm, vec) if not r.get("gap")]
    assert sum(r["you"] for r in lb) == 1
    ranks = [r["rank"] for r in lb]
    assert ranks == sorted(ranks)
    assert next(r for r in lb if r["you"])["rank"] == int(contest.timeline(mm)["rank"][27])
