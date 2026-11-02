"""Fantasy scoring engine: turns Cricsheet ball-by-ball JSON into over-by-over
fantasy points for every player. Pure Python, no AI -- every number is exact.

Scoring follows Dream11-style T20 rules (an approximation, see SCORING below).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SHORT = {"India": "IND", "New Zealand": "NZ"}

# Dream11-style T20 points (approximation, documented in the app).
SCORING = {
    "playing_xi": 4,
    "run": 1,
    "four_bonus": 4,
    "six_bonus": 6,
    "milestones": [(100, 16), (75, 12), (50, 8), (25, 4)],  # highest only
    "duck": -2,  # not for pure bowlers
    "wicket": 30,
    "bowled_lbw_bonus": 8,
    "hauls": [(5, 12), (4, 8), (3, 4)],  # highest only
    "maiden": 12,
    "dot_ball": 1,
    "catch": 8,
    "three_catch_bonus": 4,
    "stumping": 12,
    "run_out_direct": 12,
    "run_out_indirect": 6,
}

BOWLER_WICKETS = {"bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"}


def economy_points(runs: int, legal_balls: int) -> int:
    if legal_balls < 12:  # min 2 overs
        return 0
    eco = runs / (legal_balls / 6)
    if eco < 5:
        return 6
    if eco < 6:
        return 4
    if eco <= 7:
        return 2
    if eco < 10:
        return 0
    if eco <= 11:
        return -2
    if eco <= 12:
        return -4
    return -6


def strike_rate_points(runs: int, balls: int, role: str) -> int:
    if role == "BOWL" or balls < 10:  # min 10 balls, not for bowlers
        return 0
    sr = runs / balls * 100
    if sr > 170:
        return 6
    if sr > 150:
        return 4
    if sr >= 130:
        return 2
    if sr >= 70:
        return 0
    if sr >= 60:
        return -2
    if sr >= 50:
        return -4
    return -6


@dataclass
class Stats:
    runs: int = 0
    balls: int = 0
    fours: int = 0
    sixes: int = 0
    out: bool = False
    legal_balls: int = 0
    conceded: int = 0
    wkts: int = 0
    bowled_lbw: int = 0
    dots: int = 0
    maidens: int = 0
    catches: int = 0
    stumpings: int = 0
    ro_direct: int = 0
    ro_indirect: int = 0

    def points(self, role: str) -> int:
        s = SCORING
        p = s["playing_xi"]
        p += self.runs * s["run"] + self.fours * s["four_bonus"] + self.sixes * s["six_bonus"]
        for threshold, bonus in s["milestones"]:
            if self.runs >= threshold:
                p += bonus
                break
        if self.out and self.runs == 0 and role != "BOWL":
            p += s["duck"]
        p += strike_rate_points(self.runs, self.balls, role)
        p += self.wkts * s["wicket"] + self.bowled_lbw * s["bowled_lbw_bonus"]
        for threshold, bonus in s["hauls"]:
            if self.wkts >= threshold:
                p += bonus
                break
        p += self.maidens * s["maiden"] + self.dots * s["dot_ball"]
        p += economy_points(self.conceded, self.legal_balls)
        p += self.catches * s["catch"] + (s["three_catch_bonus"] if self.catches >= 3 else 0)
        p += self.stumpings * s["stumping"]
        p += self.ro_direct * s["run_out_direct"] + self.ro_indirect * s["run_out_indirect"]
        return p


@dataclass
class Step:
    """State of the match at the end of one over (step 0 = before the first ball)."""
    idx: int
    label: str            # e.g. "NZ · over 7"
    score: str            # e.g. "NZ 62/1 (7)"
    points: dict          # pid -> cumulative fantasy points
    events: dict          # pid -> what that player did THIS over
    summary: str          # one-line description of the over
    next_over: dict | None = None  # {"batters": [...], "bowler": pid} for the following over


@dataclass
class Match:
    id: str
    title: str
    subtitle: str
    result: str
    teams: list
    players: dict          # pid -> {"name", "role", "team", "pick", "cap"}
    steps: list = field(default_factory=list)

    def name(self, pid: str) -> str:
        return self.players[pid]["name"]

    def short_team(self, pid: str) -> str:
        return SHORT.get(self.players[pid]["team"], self.players[pid]["team"])


def _event_text(match_players, ev: dict) -> str:
    bits = []
    if ev.get("runs") or ev.get("balls"):
        b = f"{ev['runs']} off {ev['balls']}"
        extra = []
        if ev.get("fours"):
            extra.append(f"{ev['fours']}x4")
        if ev.get("sixes"):
            extra.append(f"{ev['sixes']}x6")
        if extra:
            b += f" ({', '.join(extra)})"
        bits.append(b)
    if ev.get("out"):
        bits.append("OUT")
    if ev.get("wkts"):
        bits.append(f"{ev['wkts']} wkt")
    if ev.get("bowled_conceded") is not None:
        bits.append(f"bowled the over for {ev['bowled_conceded']}")
    if ev.get("catches"):
        bits.append(f"{ev['catches']} catch")
    if ev.get("runouts"):
        bits.append("run-out")
    return ", ".join(bits)


def load_players_meta() -> dict:
    return json.loads((DATA_DIR / "players.json").read_text(encoding="utf-8"))["players"]


def list_matches() -> list[dict]:
    out = []
    for f in sorted((DATA_DIR / "matches").glob("*.json")):
        info = json.loads(f.read_text(encoding="utf-8"))["info"]
        out.append({"id": f.stem, "date": info["dates"][0], "label": _title(info)})
    return sorted(out, key=lambda m: m["date"], reverse=True)


def _title(info: dict) -> str:
    ev = info.get("event", {})
    stage = ev.get("stage") or (f"T20I {ev['match_number']}" if ev.get("match_number") else "")
    a, b = info["teams"]
    return f"{SHORT.get(a, a)} vs {SHORT.get(b, b)} · {ev.get('name', '')} {stage} · {info['dates'][0]}".strip()


def load_match(match_id: str) -> Match:
    raw = json.loads((DATA_DIR / "matches" / f"{match_id}.json").read_text(encoding="utf-8"))
    info = raw["info"]
    meta = load_players_meta()

    players = {}
    for team, xi in info["players"].items():
        for pid in xi:
            m = meta.get(pid, {"name": pid, "role": "BAT", "pick": 30, "cap": 2})
            players[pid] = {**m, "team": team}

    out = info.get("outcome", {})
    by = out.get("by", {})
    margin = f"{by['runs']} runs" if "runs" in by else f"{by.get('wickets', '?')} wickets"
    result = f"{out.get('winner', '')} won by {margin}" if out.get("winner") else "No result"

    match = Match(
        id=match_id,
        title=_title(info),
        subtitle=f"{info.get('venue', '')}",
        result=result,
        teams=info["teams"],
        players=players,
    )

    stats = {pid: Stats() for pid in players}

    def snapshot():
        return {pid: stats[pid].points(players[pid]["role"]) for pid in players}

    match.steps.append(Step(0, "Before first ball", "Match about to start",
                            snapshot(), {}, "Everyone gets +4 for being in the playing XI."))

    for inn_no, inn in enumerate(raw["innings"]):
        bat_team = inn["team"]
        bat_short = SHORT.get(bat_team, bat_team)
        total = wkts = 0
        for over in inn["overs"]:
            ov_no = over["over"]
            ev: dict = {}
            over_runs = over_wkts = 0
            bowler_runs = legal = 0
            bowler = over["deliveries"][0]["bowler"]
            highlights = []

            def e(pid):
                return ev.setdefault(pid, {"runs": 0, "balls": 0, "fours": 0, "sixes": 0,
                                           "wkts": 0, "catches": 0, "runouts": 0, "out": False})

            for d in over["deliveries"]:
                extras = d.get("extras", {})
                br = d["runs"]["batter"]
                is_wide = "wides" in extras
                is_nb = "noballs" in extras
                bat = d["batter"]
                bwl = d["bowler"]
                s_bat, s_bwl = stats[bat], stats[bwl]

                if not is_wide:
                    s_bat.balls += 1
                    e(bat)["balls"] += 1
                s_bat.runs += br
                e(bat)["runs"] += br
                if br == 4 and not d["runs"].get("non_boundary"):
                    s_bat.fours += 1
                    e(bat)["fours"] += 1
                elif br == 6:
                    s_bat.sixes += 1
                    e(bat)["sixes"] += 1
                    highlights.append(f"six by {players[bat]['name']}")

                conceded = br + extras.get("wides", 0) + extras.get("noballs", 0)
                s_bwl.conceded += conceded
                bowler_runs += conceded
                if not is_wide and not is_nb:
                    s_bwl.legal_balls += 1
                    legal += 1
                    if d["runs"]["total"] == 0:
                        s_bwl.dots += 1

                total += d["runs"]["total"]
                over_runs += d["runs"]["total"]

                for w in d.get("wickets", []):
                    kind = w["kind"]
                    out_pid = w["player_out"]
                    wkts += 1
                    over_wkts += 1
                    if out_pid in stats:
                        stats[out_pid].out = True
                        e(out_pid)["out"] = True
                    if kind in BOWLER_WICKETS:
                        s_bwl.wkts += 1
                        e(bwl)["wkts"] += 1
                        if kind in ("bowled", "lbw"):
                            s_bwl.bowled_lbw += 1
                    fielders = [f["name"] for f in w.get("fielders", [])
                                if not f.get("substitute") and f.get("name") in stats]
                    if kind == "caught and bowled":
                        stats[bwl].catches += 1
                        e(bwl)["catches"] += 1
                    elif kind == "caught":
                        for f in fielders:
                            stats[f].catches += 1
                            e(f)["catches"] += 1
                    elif kind == "stumped":
                        for f in fielders:
                            stats[f].stumpings += 1
                            e(f)["catches"] += 1
                    elif kind == "run out":
                        for f in fielders:
                            if len(fielders) == 1:
                                stats[f].ro_direct += 1
                            else:
                                stats[f].ro_indirect += 1
                            e(f)["runouts"] += 1
                    how = kind if kind != "caught" or not fielders else f"c {players[fielders[0]]['name']}"
                    if kind in BOWLER_WICKETS:
                        how += f" b {players[bwl]['name']}" if kind != "bowled" else f" by {players[bwl]['name']}"
                    highlights.insert(0, f"WICKET {players.get(out_pid, {'name': out_pid})['name']} ({how})")

            if legal == 6 and bowler_runs == 0:
                stats[bowler].maidens += 1
                highlights.append(f"maiden by {players[bowler]['name']}")
            e(bowler)["bowled_conceded"] = bowler_runs

            balls_txt = f"{ov_no + 1}" if legal >= 6 else f"{ov_no}.{legal}"
            score = f"{bat_short} {total}/{wkts} ({balls_txt})"
            summ = f"{bat_short} over {ov_no + 1}: {over_runs} runs, {over_wkts} wkt"
            if highlights:
                summ += " - " + "; ".join(highlights[:4])
            if inn_no == 1 and "target" in inn:
                need = inn["target"]["runs"] - total
                if need > 0:
                    summ += f". Need {need} more."
            for pid, x in ev.items():
                x["text"] = _event_text(players, x)
            match.steps.append(Step(len(match.steps), f"{bat_short} · over {ov_no + 1}", score,
                                    snapshot(), ev, summ))

    # who is involved in the NEXT over (shown on the "root for this" card)
    flat = []
    for inn in raw["innings"]:
        for over in inn["overs"]:
            flat.append(over["deliveries"][0])
    for i, st in enumerate(match.steps):
        if i < len(flat):
            d = flat[i]
            st.next_over = {"batters": [d["batter"], d["non_striker"]], "bowler": d["bowler"]}
    return match
