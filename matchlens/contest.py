"""Simulated contest: 10,000 rival teams built from pre-match popularity priors.

Dream11 does not publish contest data, so rivals are SIMULATED. With real contest
data (every team's 11 + C/VC) the same maths works unchanged.

Key idea -- "effective ownership" (EO): the average multiplier the field has on a
player (C counts 2x, VC 1.5x). If you own a player at 2x and the field's EO is 0.4,
every point he scores pushes you UP the leaderboard. If he's in 90% of teams and
not in yours, every point he scores pushes you DOWN.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .engine import Match

ROLES = ("WK", "BAT", "AR", "BOWL")
N_RIVALS = 10_000


def validate_team(match: Match, picks: list, captain: str | None, vice: str | None) -> list[str]:
    errs = []
    if len(picks) != 11:
        errs.append(f"Pick exactly 11 players (you have {len(picks)}).")
    for r in ROLES:
        n = sum(match.players[p]["role"] == r for p in picks)
        if n < 1:
            errs.append(f"Need at least 1 {r}.")
        if n > 8:
            errs.append(f"Max 8 {r}.")
    for t in match.teams:
        if sum(match.players[p]["team"] == t for p in picks) > 10:
            errs.append(f"Max 10 players from {t}.")
    if not captain or not vice:
        errs.append("Choose a Captain (2x) and Vice-captain (1.5x).")
    elif captain == vice:
        errs.append("Captain and Vice-captain must be different.")
    elif captain not in picks or vice not in picks:
        errs.append("Captain and Vice-captain must be in your 11.")
    return errs


@dataclass
class Contest:
    match: Match
    pids: list            # column order
    pts: np.ndarray       # steps x players (cumulative points)
    rival_mult: np.ndarray  # rivals x players
    rival_totals: np.ndarray  # rivals x steps
    pick_pct: np.ndarray
    cap_pct: np.ndarray
    vc_pct: np.ndarray
    eo: np.ndarray

    @classmethod
    def build(cls, match: Match, n: int = N_RIVALS, seed: int = 11) -> "Contest":
        rng = np.random.default_rng(seed)
        pids = list(match.players)
        P = len(pids)
        role = np.array([ROLES.index(match.players[p]["role"]) for p in pids])
        team = np.array([match.teams.index(match.players[p]["team"]) for p in pids])
        pick = np.array([match.players[p]["pick"] for p in pids], dtype=float) / 100
        cap = np.array([match.players[p]["cap"] for p in pids], dtype=float) + 0.5

        # Gumbel top-k: sample 11 players per team, weighted by popularity.
        logits = np.log(pick / (1 - pick + 1e-6))
        mult = np.zeros((0, P), dtype=np.float32)
        while len(mult) < n:
            k = int((n - len(mult)) * 1.4) + 50
            g = logits + rng.gumbel(size=(k, P))
            top = np.argsort(-g, axis=1)[:, :11]
            sel = np.zeros((k, P), dtype=bool)
            np.put_along_axis(sel, top, True, axis=1)
            ok = np.ones(k, dtype=bool)
            for r in range(4):
                c = sel[:, role == r].sum(1)
                ok &= (c >= 1) & (c <= 8)
            for t in range(2):
                ok &= sel[:, team == t].sum(1) <= 10
            sel = sel[ok]
            m = sel.astype(np.float32)
            # captain then vice-captain, weighted by captaincy popularity
            w = sel * cap
            cg = np.log(w + 1e-12) + rng.gumbel(size=w.shape)
            c_idx = cg.argmax(1)
            cg[np.arange(len(cg)), c_idx] = -np.inf
            v_idx = cg.argmax(1)
            m[np.arange(len(m)), c_idx] = 2.0
            m[np.arange(len(m)), v_idx] = 1.5
            mult = np.vstack([mult, m])
        mult = mult[:n]

        pts = np.array([[s.points[p] for p in pids] for s in match.steps], dtype=np.float32)
        totals = mult @ pts.T
        return cls(
            match=match, pids=pids, pts=pts, rival_mult=mult, rival_totals=totals,
            pick_pct=(mult > 0).mean(0) * 100,
            cap_pct=(mult == 2.0).mean(0) * 100,
            vc_pct=(mult == 1.5).mean(0) * 100,
            eo=mult.mean(0),
        )

    # ---------- your team ----------
    def my_mult(self, picks, captain, vice) -> np.ndarray:
        m = np.zeros(len(self.pids), dtype=np.float32)
        for p in picks:
            m[self.pids.index(p)] = 1.0
        m[self.pids.index(captain)] = 2.0
        m[self.pids.index(vice)] = 1.5
        return m

    def timeline(self, mm: np.ndarray) -> dict:
        mine = self.pts @ mm
        ranks = 1 + (self.rival_totals > mine[None, :]).sum(0)
        return {
            "points": mine,
            "rank": ranks,
            "field_avg": self.rival_totals.mean(0),
            "top10pct": np.percentile(self.rival_totals, 90, axis=0),
            "n": len(self.rival_totals) + 1,
        }

    def drivers(self, mm: np.ndarray, a: int, b: int, top: int = 4) -> list[dict]:
        """Players who moved you most vs the field between steps a -> b."""
        delta = self.pts[b] - self.pts[a]
        swing = (mm - self.eo) * delta
        order = np.argsort(-np.abs(swing))
        out = []
        for i in order[:top]:
            if abs(swing[i]) < 2:
                continue
            p = self.pids[i]
            out.append(self._player_row(p, mm[i]) | {
                "points_scored": round(float(delta[i])),
                "net_effect_on_you": round(float(swing[i])),
                "what_he_did": "; ".join(
                    s.events[p]["text"] for s in self.match.steps[a + 1:b + 1]
                    if p in s.events and s.events[p]["text"]),
            })
        return out

    def _player_row(self, p, m) -> dict:
        i = self.pids.index(p)
        mine = {2.0: "your Captain (2x)", 1.5: "your Vice-captain (1.5x)", 1.0: "in your team"}.get(
            float(m), "NOT in your team")
        return {
            "player": self.match.name(p),
            "team": self.match.short_team(p),
            "you": mine,
            "picked_by_pct": round(float(self.pick_pct[i])),
            "captained_by_pct": round(float(self.cap_pct[i])),
        }

    def leverage(self, mm: np.ndarray, p: str) -> float:
        i = self.pids.index(p)
        return float(mm[i] - self.eo[i])

    def root_for(self, mm: np.ndarray, step: int) -> list[dict]:
        """For the next over: who is involved and whether their success helps you."""
        nxt = self.match.steps[step].next_over
        if not nxt:
            return []
        out = []
        for p, kind in [(nxt["batters"][0], "bat"), (nxt["batters"][1], "bat"), (nxt["bowler"], "bowl")]:
            lev = self.leverage(mm, p)
            out.append(self._player_row(p, mm[self.pids.index(p)]) | {
                "pid": p,
                "job": "batting" if kind == "bat" else "bowling",
                "leverage": round(lev, 2),
                "verdict": "cheer" if lev > 0.15 else ("fear" if lev < -0.15 else "neutral"),
            })
        return out

    def differentials(self, mm: np.ndarray) -> list[dict]:
        rows = []
        for i, p in enumerate(self.pids):
            lev = float(mm[i] - self.eo[i])
            rows.append(self._player_row(p, mm[i]) | {"pid": p, "leverage": round(lev, 2),
                                                       "role": self.match.players[p]["role"]})
        return sorted(rows, key=lambda r: -abs(r["leverage"]))
