"""Fantasy scoring engine: turns Cricsheet ball-by-ball JSON into fantasy points for
every player, after every ball and every over. Pure Python, no AI -- every number is exact.

Rules come from a versioned config (data/t20_scoring_rules.json). Every point is
itemised (Stats.breakdown) so explanations can cite the exact rule that produced it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SHORT = {"India": "IND", "New Zealand": "NZ"}

SCORING = json.loads((DATA_DIR / "t20_scoring_rules.json").read_text(encoding="utf-8"))
RULESET = f"{SCORING['id']} v{SCORING['version']}"

BOWLER_WICKETS = {"bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"}

_OPS = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b, "else": lambda a, b: True}


def _band(value: float, bands: list) -> int:
    for op, threshold, pts in bands:
        if _OPS[op](value, threshold):
            return pts
    return 0


def economy_points(runs: int, legal_balls: int) -> int:
    cfg = SCORING["economy"]
    if legal_balls < cfg["min_legal_balls"]:
        return 0
    return _band(runs / (legal_balls / 6), cfg["bands"])


def strike_rate_points(runs: int, balls: int, role: str) -> int:
    cfg = SCORING["strike_rate"]
    if role in cfg["exempt_roles"] or balls < cfg["min_balls"]:
        return 0
    return _band(runs / balls * 100, cfg["bands"])


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

    def breakdown(self, role: str) -> dict[str, int]:
        """Itemised points: rule id -> points (only rules that apply)."""
        s = SCORING
        b: dict[str, int] = {"playing_xi": s["playing_xi"]}

        def add(k, v):
            if v:
                b[k] = b.get(k, 0) + v

        add("runs", self.runs * s["run"])
        add("four_bonus", self.fours * s["four_bonus"])
        add("six_bonus", self.sixes * s["six_bonus"])
        for threshold, bonus in s["milestones"]:
            if self.runs >= threshold:
                add(f"milestone_{threshold}", bonus)
                break
        if self.out and self.runs == 0 and role not in s["duck_exempt_roles"]:
            add("duck", s["duck"])
        add("strike_rate", strike_rate_points(self.runs, self.balls, role))
        add("wickets", self.wkts * s["wicket"])
        add("bowled_lbw_bonus", self.bowled_lbw * s["bowled_lbw_bonus"])
        for threshold, bonus in s["hauls"]:
            if self.wkts >= threshold:
                add(f"haul_{threshold}", bonus)
                break
        add("maidens", self.maidens * s["maiden"])
        add("dot_balls", self.dots * s["dot_ball"])
        add("economy", economy_points(self.conceded, self.legal_balls))
        add("catches", self.catches * s["catch"])
        if self.catches >= 3:
            add("three_catch_bonus", s["three_catch_bonus"])
        add("stumpings", self.stumpings * s["stumping"])
        add("run_out_direct", self.ro_direct * s["run_out_direct"])
        add("run_out_indirect", self.ro_indirect * s["run_out_indirect"])
        return b

    def points(self, role: str) -> int:
        return sum(self.breakdown(role).values())


RULE_LABELS = {
    "playing_xi": "playing XI", "runs": "runs", "four_bonus": "four bonus", "six_bonus": "six bonus",
    "milestone_25": "25-run bonus", "milestone_50": "half-century bonus", "milestone_75": "75-run bonus",
    "milestone_100": "century bonus", "duck": "duck penalty", "strike_rate": "strike-rate band",
    "wickets": "wicket", "bowled_lbw_bonus": "bowled/LBW bonus", "haul_3": "3-wicket bonus",
    "haul_4": "4-wicket bonus", "haul_5": "5-wicket bonus", "maidens": "maiden", "dot_balls": "dot ball",
    "economy": "economy band", "catches": "catch", "three_catch_bonus": "3-catch bonus",
    "stumpings": "stumping", "run_out_direct": "direct run-out", "run_out_indirect": "run-out assist",
}


@dataclass
class Ball:
    """State right after one delivery."""
    label: str            # e.g. "7.3"
    score: str            # e.g. "IND 47/2 (7.3)"
    text: str             # e.g. "Bracewell to Iyer, FOUR"
    stats: dict           # pid -> Stats (copy)
    points: dict          # pid -> points
    next: dict | None = None  # who faces / bowls the following delivery


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
    stats: dict = field(default_factory=dict)   # pid -> Stats at end of this over
    balls: list = field(default_factory=list)   # Ball snapshots within this over


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


def _copy_stats(stats: dict) -> dict:
    return {pid: replace(s) for pid, s in stats.items()}


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
                            snapshot(), {}, "Everyone gets +4 for being in the playing XI.",
                            stats=_copy_stats(stats)))
    all_balls: list[tuple[Ball, dict]] = []  # (snapshot, delivery) in match order

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
            balls: list[Ball] = []

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
                ball_txt = f"{players[bwl]['name']} to {players[bat]['name']}: "
                if is_wide:
                    ball_txt += "wide"
                elif br == 4:
                    ball_txt += "FOUR"
                elif br == 6:
                    ball_txt += "SIX"
                else:
                    ball_txt += f"{br} run{'s' if br != 1 else ''}" if br else "dot ball"
                if is_nb:
                    ball_txt += " (no-ball)"

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
                    ball_txt += f", WICKET {players.get(out_pid, {'name': out_pid})['name']} ({how})"

                # maiden is credited when the over completes, so the snapshot of the last ball includes it
                is_last = d is over["deliveries"][-1]
                if is_last and legal == 6 and bowler_runs == 0:
                    stats[bowler].maidens += 1
                    highlights.append(f"maiden by {players[bowler]['name']}")
                b = Ball(label=f"{ov_no}.{legal}", score=f"{bat_short} {total}/{wkts} ({ov_no}.{legal})",
                         text=ball_txt, stats=_copy_stats(stats), points=snapshot())
                balls.append(b)
                all_balls.append((b, d))

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
                                    snapshot(), ev, summ, stats=_copy_stats(stats), balls=balls))

    # who is on strike / bowling for the delivery that follows each ball
    for i, (b, _) in enumerate(all_balls):
        if i + 1 < len(all_balls):
            nd = all_balls[i + 1][1]
            b.next = {"batters": [nd["batter"], nd["non_striker"]], "bowler": nd["bowler"]}

    # who is involved in the NEXT over (shown on the "next over" card)
    flat = []
    for inn in raw["innings"]:
        for over in inn["overs"]:
            flat.append(over["deliveries"][0])
    for i, st in enumerate(match.steps):
        if i < len(flat):
            d = flat[i]
            st.next_over = {"batters": [d["batter"], d["non_striker"]], "bowler": d["bowler"]}
    return match
