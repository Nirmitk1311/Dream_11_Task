"""AI layer (Gemini). The model NEVER calculates -- it receives exact facts from
the engine and turns them into short, fan-friendly language. Every call has a
deterministic template fallback, so the tool keeps working if the API is down
or rate-limited. Responses are cached to data/ai_cache.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

CACHE_FILE = Path(__file__).resolve().parent.parent / "data" / "ai_cache.json"
_lock = threading.Lock()
_cache: dict = {}
try:
    _cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
except Exception:
    _cache = {}

# Tried in order; on overload/timeout/not-found we fall through to the next one.
MODEL_CANDIDATES = ["gemini-3-flash-preview", "gemini-3.1-flash-lite", "gemini-flash-lite-latest",
                    "gemini-3.8-flash", "gemini-flash-latest"]
TIMEOUT_MS = 45_000
_working_model: str | None = None
_client = None
_api_key: str | None = None
last_error: str | None = None

SYSTEM = """You are MatchLens, a live fantasy-cricket companion for Indian fans.
A fan has a fantasy team in a contest of 10,001 teams and is watching the match live.
You receive EXACT facts computed by a scoring engine. Rules:
- Use ONLY numbers present in the facts. Never invent or estimate numbers, players or events.
- Explain from the fan's point of view: what happened TO THEIR TEAM and RANK, and why.
- The key insight is relative: a player scoring big only helps the fan if the fan owns him
  MORE than the field does (e.g. he is their captain but few rivals captained him).
  A popular player the fan does NOT own scoring big pushes the fan DOWN.
- Sound like a sharp friend texting during the match: short, punchy, no hashtags, no emoji.
- Never mention "facts", "data", "JSON", "leverage" or "effective ownership".
- Write ranks with a # and commas (e.g. #7,983) and points as whole numbers (e.g. 26 pts).
"""

LANG = {
    "en": "Write in simple English.",
    "hinglish": "Write in natural Hinglish (Hindi-English mix in Roman script, the way Indian fans text, "
                "e.g. 'Tilak ne game palat diya'). Keep player names and numbers as-is.",
}


def configure(api_key: str | None):
    global _api_key, _client
    if api_key and "paste-your-key" in api_key:  # placeholder not filled in yet
        api_key = None
    if api_key and api_key != _api_key:
        _api_key = api_key
        _client = None


def _resolve_key() -> str | None:
    k = _api_key or os.environ.get("GEMINI_API_KEY")
    if not k or "paste-your-key" in k:
        return None
    return k


def available() -> bool:
    return bool(_resolve_key())


def _get_client():
    global _client
    if _client is None:
        from google import genai
        from google.genai import types
        _client = genai.Client(api_key=_resolve_key(),
                               http_options=types.HttpOptions(timeout=TIMEOUT_MS))
    return _client


def _key(kind: str, payload) -> str:
    raw = json.dumps([kind, payload], sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _save():
    try:
        with _lock:
            CACHE_FILE.write_text(json.dumps(_cache, ensure_ascii=False, indent=0), encoding="utf-8")
    except Exception:
        pass  # read-only filesystem on some hosts: in-memory cache still works


def _call(prompt: str, json_mode: bool) -> str:
    global _working_model, last_error
    from google.genai import types

    client = _get_client()
    preferred = [m for m in [_working_model, os.environ.get("GEMINI_MODEL")] if m]
    models = list(dict.fromkeys(preferred + MODEL_CANDIDATES))
    err = None
    for model in models:
        try:
            cfg = types.GenerateContentConfig(
                system_instruction=SYSTEM,
                temperature=0.7,
                response_mime_type="application/json" if json_mode else "text/plain",
            )
            resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
            _working_model = model
            last_error = None
            return resp.text or ""
        except Exception as e:
            err = e
            msg = str(e).lower()
            if any(s in msg for s in ("api key", "permission_denied", "401", "403")):
                break  # bad key: no point trying other models
            continue  # overloaded / timeout / not found / rate-limited -> next model
    last_error = f"{type(err).__name__}: {str(err)[:200]}"
    raise RuntimeError(last_error)


# ---------------------------------------------------------------- over notes
def over_notes(steps_facts: list[dict], lang: str) -> dict[int, dict] | None:
    """Batch: one short note + one 'root for' line per over. Returns {step: {...}} or None."""
    if not available() or not steps_facts:
        return None
    k = _key("notes-v2", [steps_facts, lang])
    if k in _cache:
        return {int(s): v for s, v in _cache[k].items()}
    prompt = (
        f"{LANG[lang]}\n\nFor EACH item below write:\n"
        '- "note": max 28 words. Why the fan\'s rank moved this over (or held). Name the player(s) '
        "who caused it and why it mattered for THIS fan (owned/not owned, captain, how popular).\n"
        '- "root": max 18 words. Who to cheer / fear in the NEXT over, from "next_over" '
        '(verdict cheer = his success helps the fan, fear = it hurts). Empty string if next_over is empty.\n'
        "Treat every item independently; never refer to later overs.\n"
        'Return JSON: {"items": [{"step": <int>, "note": "...", "root": "..."}]}\n\n'
        f"ITEMS:\n{json.dumps(steps_facts, ensure_ascii=False)}"
    )
    try:
        data = json.loads(_call(prompt, json_mode=True))
        items = data["items"] if isinstance(data, dict) else data
        out = {int(it["step"]): {"note": str(it.get("note", "")).strip(),
                                 "root": str(it.get("root", "")).strip()} for it in items}
    except Exception:
        return None
    _cache[k] = {str(s): v for s, v in out.items()}
    _save()
    return out


# ---------------------------------------------------------------- catch me up
def catch_up(facts: dict, lang: str) -> str | None:
    if not available():
        return None
    k = _key("catchup-v2", [facts, lang])
    if k in _cache:
        return _cache[k]
    prompt = (
        f"{LANG[lang]}\n\nThe fan just opened the app. They last checked at '{facts['from']}' and it is now "
        f"'{facts['to']}'. Catch them up in exactly 3 short lines (each starting with '- '):\n"
        "1) what happened to THEIR rank and points, 2) the main reason (players), "
        "3) what to watch for next. Max 70 words total.\n\n"
        f"FACTS:\n{json.dumps(facts, ensure_ascii=False)}"
    )
    try:
        text = _call(prompt, json_mode=False).strip()
    except Exception:
        return None
    if not text:
        return None
    _cache[k] = text
    _save()
    return text


# ---------------------------------------------------------------- fallbacks
def template_note(f: dict) -> str:
    move = (f["rank_before"] or f["rank_after"]) - f["rank_after"]
    if f["rank_before"] is None:
        head = f"You're #{f['rank_after']:,} after the first over."
    elif move > 0:
        head = f"Up {move:,} to #{f['rank_after']:,}."
    elif move < 0:
        head = f"Down {-move:,} to #{f['rank_after']:,}."
    else:
        head = f"Holding at #{f['rank_after']:,}."
    if f["drivers"]:
        d = f["drivers"][0]
        sign = "helped" if d["net_effect_on_you"] > 0 else "hurt"
        head += (f" {d['player']} {sign} you ({d['what_he_did'] or 'quiet over'}) - "
                 f"{d['you']}, picked by {d['picked_by_pct']}%.")
    return head


def template_root(f: dict) -> str:
    bits = []
    for r in f.get("next_over", []):
        if r["verdict"] == "cheer":
            bits.append(f"cheer {r['player']} ({r['job']})")
        elif r["verdict"] == "fear":
            bits.append(f"fear {r['player']} ({r['job']})")
    return ("Next over: " + ", ".join(bits) + ".") if bits else "Next over: nothing big riding on it for you."


def template_catch_up(f: dict) -> str:
    move = f["rank_from"] - f["rank_to"]
    lines = [f"- Rank {'up' if move >= 0 else 'down'} {abs(move):,}: #{f['rank_from']:,} → #{f['rank_to']:,} "
             f"(you +{f['your_points_gained']:.0f} pts vs field avg +{f['field_avg_points_gained']:.0f})."]
    if f["drivers"]:
        d = f["drivers"][0]
        lines.append(f"- Biggest reason: {d['player']} ({d['you']}, picked by {d['picked_by_pct']}%): "
                     f"{d['points_scored']:.0f} pts.")
    lines.append(f"- Now: {f['score_now']}.")
    return "\n".join(lines)
