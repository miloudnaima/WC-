import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Iterable

import requests
from flask import Flask, jsonify, request

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("smart-football-bot")
app = Flask(__name__)


@dataclass(slots=True)
class Settings:
    telegram_token: str = os.getenv("TELEGRAM_TOKEN", "").strip()
    webhook_secret: str = os.getenv("WEBHOOK_SECRET", "").strip()
    odds_api_key: str = os.getenv("ODDS_API_KEY", "").strip()
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "").strip()
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", os.getenv("ROUTER_API_KEY", "")).strip()

    def validate(self) -> None:
        if not self.telegram_token:
            raise RuntimeError("TELEGRAM_TOKEN is required")
        if not self.webhook_secret:
            raise RuntimeError("WEBHOOK_SECRET is required")


settings = Settings()
settings.validate()

TELEGRAM_API = f"https://api.telegram.org/bot{settings.telegram_token}"
ODDS_SPORTS_URL = "https://api.the-odds-api.com/v4/sports"
ODDS_UPCOMING_URL = "https://api.the-odds-api.com/v4/sports/upcoming/odds"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT = 20
CACHE_TTL_SECONDS = 300
AI_CACHE_TTL_SECONDS = 180
COOLDOWN_SECONDS = 8
SPAM_WINDOW_SECONDS = 20
SPAM_MAX_REQUESTS = 6
MATCH_WINDOW_HOURS = 24
PAST_GRACE_HOURS = 3
MAX_PAGE_SIZE = 3
ODDS_SPORT_FILTER = {"soccer"}


class TTLCache:
    def __init__(self) -> None:
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        item = self._store.get(key)
        if not item:
            return None
        expires_at, value = item
        if time.time() >= expires_at:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        self._store[key] = (time.time() + ttl_seconds, value)


class RateLimiter:
    def __init__(self) -> None:
        self._events: dict[str, list[float]] = {}
        self._cooldowns: dict[str, float] = {}

    def is_spamming(self, key: str) -> bool:
        now = time.time()
        events = [ts for ts in self._events.get(key, []) if now - ts <= SPAM_WINDOW_SECONDS]
        events.append(now)
        self._events[key] = events
        return len(events) > SPAM_MAX_REQUESTS

    def cooldown_remaining(self, key: str) -> int:
        now = time.time()
        allowed_at = self._cooldowns.get(key, 0)
        if allowed_at > now:
            return max(1, int(round(allowed_at - now)))
        self._cooldowns[key] = now + COOLDOWN_SECONDS
        return 0


cache = TTLCache()
ai_cache = TTLCache()
rate_limiter = RateLimiter()
session = requests.Session()
session.headers.update({"User-Agent": "smart-football-bot/2.1"})


@dataclass(slots=True)
class MatchInsight:
    match_id: str
    competition: str
    kickoff_utc: str
    home_team: str
    away_team: str
    bookmaker: str
    odds_home: float | None
    odds_draw: float | None
    odds_away: float | None
    totals_over_25: float | None
    totals_under_25: float | None
    btts_yes: float | None
    btts_no: float | None
    totals_over_15: float | None
    totals_under_15: float | None
    score_map: dict[str, float]
    support_stats: dict[str, float]


COMMAND_DESCRIPTIONS: list[dict[str, str]] = [
    {"command": "start", "description": "Open the welcome screen"},
    {"command": "help", "description": "Show all commands"},
    {"command": "menu", "description": "Open the main menu"},
    {"command": "today", "description": "Best football matches today"},
    {"command": "live", "description": "Upcoming watchlist"},
    {"command": "now", "description": "Alias of live"},
    {"command": "safe", "description": "Safer picks"},
    {"command": "value", "description": "Value picks"},
    {"command": "vip", "description": "Alias of value"},
    {"command": "banker", "description": "Alias of safe"},
    {"command": "acca", "description": "Accumulator picks"},
    {"command": "corners", "description": "Corners angles"},
    {"command": "cards", "description": "Cards angles"},
    {"command": "goals", "description": "Goals-focused picks"},
    {"command": "btts", "description": "BTTS picks"},
    {"command": "over25", "description": "Over 2.5 picks"},
    {"command": "under25", "description": "Under 2.5 picks"},
    {"command": "firsthalf", "description": "First-half picks"},
    {"command": "secondhalf", "description": "Second-half picks"},
    {"command": "halftime", "description": "Alias of firsthalf"},
    {"command": "predictions", "description": "Alias of acca"},
    {"command": "top5", "description": "Alias of acca"},
    {"command": "highconfidence", "description": "Alias of safe"},
    {"command": "matches", "description": "Alias of today"},
    {"command": "stats", "description": "Show loaded market stats"},
    {"command": "analyze", "description": "Analyze a match or board"},
    {"command": "team", "description": "Search a team in upcoming fixtures"},
    {"command": "match", "description": "Analyze TEAM1 vs TEAM2"},
    {"command": "search", "description": "Search fixtures by text"},
]

ALIAS_MAP = {
    "matches": "today",
    "now": "live",
    "vip": "value",
    "banker": "safe",
    "highconfidence": "safe",
    "predictions": "acca",
    "top5": "acca",
    "halftime": "firsthalf",
}


def parse_iso_time(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except Exception:
        return None


def format_kickoff(value: str) -> str:
    dt = parse_iso_time(value)
    return dt.strftime("%d %b %H:%M UTC") if dt else value


def retry_request(method: str, url: str, *, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None, json_body: dict[str, Any] | None = None, timeout: int = DEFAULT_TIMEOUT, attempts: int = 3) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = session.request(method, url, params=params, headers=headers, json=json_body, timeout=timeout)
            if response.status_code < 500:
                return response
        except requests.RequestException as exc:
            last_error = exc
            log.warning("Request failed for %s on attempt %s: %s", url, attempt, exc)
        time.sleep(0.25 * attempt)
    if last_error:
        raise last_error
    raise RuntimeError(f"Request failed: {url}")


class TelegramClient:
    def send_message(self, chat_id: int, text: str, reply_markup: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text[:4096],
            "disable_web_page_preview": True,
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup
        retry_request("POST", f"{TELEGRAM_API}/sendMessage", json_body=payload, timeout=15)

    def answer_callback(self, callback_id: str, text: str) -> None:
        retry_request(
            "POST",
            f"{TELEGRAM_API}/answerCallbackQuery",
            json_body={"callback_query_id": callback_id, "text": text[:180]},
            timeout=10,
        )

    def set_commands(self) -> None:
        try:
            retry_request(
                "POST",
                f"{TELEGRAM_API}/setMyCommands",
                json_body={"commands": COMMAND_DESCRIPTIONS},
                timeout=15,
            )
        except Exception as exc:
            log.warning("setMyCommands failed: %s", exc)


telegram = TelegramClient()


def get_sport_keys() -> list[str]:
    cached = cache.get("odds:sport_keys")
    if cached is not None:
        return cached
    if not settings.odds_api_key:
        return []
    response = retry_request(
        "GET",
        ODDS_SPORTS_URL,
        params={"apiKey": settings.odds_api_key},
        timeout=20,
    )
    if response.status_code != 200:
        log.error("Odds sports status=%s body=%s", response.status_code, response.text[:1000])
        return []
    payload = response.json()
    sports = payload if isinstance(payload, list) else []
    keys = [str(item.get("key", "")) for item in sports if str(item.get("key", "")).startswith("soccer_") and not item.get("has_outrights", False)]
    cache.set("odds:sport_keys", keys, CACHE_TTL_SECONDS)
    return keys


def fetch_events_for_sport(sport_key: str) -> list[dict[str, Any]]:
    cache_key = f"odds:sport:{sport_key}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    params = {
        "apiKey": settings.odds_api_key,
        "regions": "eu,uk,us",
        "markets": "h2h,totals,spreads",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    response = retry_request("GET", f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds", params=params, timeout=20)
    if response.status_code != 200:
        log.warning("Odds fetch failed for %s status=%s body=%s", sport_key, response.status_code, response.text[:600])
        return []
    payload = response.json()
    events = payload if isinstance(payload, list) else []
    cache.set(cache_key, events, CACHE_TTL_SECONDS)
    return events


def get_upcoming_odds() -> list[dict[str, Any]]:
    cached = cache.get("odds:upcoming")
    if cached is not None:
        return cached
    if not settings.odds_api_key:
        return []

    all_events: list[dict[str, Any]] = []

    params = {
        "apiKey": settings.odds_api_key,
        "regions": "eu,uk,us",
        "markets": "h2h,totals,spreads",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    response = retry_request("GET", ODDS_UPCOMING_URL, params=params, timeout=20)
    if response.status_code == 200:
        payload = response.json()
        events = payload if isinstance(payload, list) else []
        soccer_from_upcoming = [event for event in events if str(event.get("sport_key", "")).startswith("soccer_")]
        log_odds_debug("upcoming_soccer_events", soccer_from_upcoming)
        all_events.extend(soccer_from_upcoming)
    else:
        log.warning("Upcoming odds status=%s body=%s", response.status_code, response.text[:600])

    if not all_events:
        sport_keys = get_sport_keys()[:12]
        log.info("soccer sport keys=%s", sport_keys)
        for sport_key in sport_keys:
            sport_events = fetch_events_for_sport(sport_key)
            log_odds_debug(f"sport_feed:{sport_key}", sport_events)
            all_events.extend(sport_events)

    deduped: dict[str, dict[str, Any]] = {}
    for event in all_events:
        event_id = str(event.get("id") or "")
        deduped[event_id or f"{event.get('home_team','')}-{event.get('away_team','')}-{event.get('commence_time','')}"] = event

    soccer_events = list(deduped.values())
    cache.set("odds:upcoming", soccer_events, CACHE_TTL_SECONDS)
    return soccer_events




def log_odds_debug(label: str, events: list[dict[str, Any]]) -> None:
    preview = []
    for event in events[:5]:
        preview.append({
            "sport_key": event.get("sport_key"),
            "home_team": event.get("home_team"),
            "away_team": event.get("away_team"),
            "commence_time": event.get("commence_time"),
            "bookmakers": len(event.get("bookmakers", []) or []),
        })
    log.info("%s count=%s preview=%s", label, len(events), json.dumps(preview)[:1500])

def select_bookmaker(event: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    bookmakers = event.get("bookmakers", []) or []
    if not bookmakers:
        return "Unavailable", {}
    bookmaker = bookmakers[0]
    return str(bookmaker.get("title", "Unavailable")), {market.get("key"): market for market in bookmaker.get("markets", [])}


def find_price(market: dict[str, Any] | None, names: Iterable[str], point: float | None = None) -> float | None:
    if not market:
        return None
    name_set = {name.lower() for name in names}
    for outcome in market.get("outcomes", []):
        if str(outcome.get("name", "")).lower() in name_set:
            if point is None or outcome.get("point") == point:
                price = outcome.get("price")
                if isinstance(price, (int, float)):
                    return float(price)
    return None


def implied_probability(price: float | None) -> float:
    if not price or price <= 1:
        return 0.0
    return 1.0 / price


def build_support_stats(markets: dict[str, Any], home_team: str, away_team: str) -> dict[str, float]:
    home_price = find_price(markets.get("h2h"), [home_team])
    draw_price = find_price(markets.get("h2h"), ["draw"])
    away_price = find_price(markets.get("h2h"), [away_team])
    over25 = find_price(markets.get("totals"), ["over"], 2.5)
    under25 = find_price(markets.get("totals"), ["under"], 2.5)
    over15 = find_price(markets.get("totals"), ["over"], 1.5)
    under15 = find_price(markets.get("totals"), ["under"], 1.5)
    btts_yes = find_price(markets.get("btts"), ["yes"])
    btts_no = find_price(markets.get("btts"), ["no"])
    if not btts_yes and not btts_no and over25 and under25:
        over25_prob = implied_probability(over25)
        under25_prob = implied_probability(under25)
        btts_yes = round(max(1.55, min(2.6, 1.15 + (1.65 - over25_prob) + (under25_prob * 0.55))), 2)
        btts_no = round(max(1.55, min(2.6, 1.2 + (1.55 - under25_prob) + (over25_prob * 0.45))), 2)

    home_prob = implied_probability(home_price)
    draw_prob = implied_probability(draw_price)
    away_prob = implied_probability(away_price)
    total = home_prob + draw_prob + away_prob
    if total > 0:
        home_prob /= total
        draw_prob /= total
        away_prob /= total

    return {
        "home_price": home_price or 0.0,
        "draw_price": draw_price or 0.0,
        "away_price": away_price or 0.0,
        "over25": over25 or 0.0,
        "under25": under25 or 0.0,
        "over15": over15 or 0.0,
        "under15": under15 or 0.0,
        "btts_yes": btts_yes or 0.0,
        "btts_no": btts_no or 0.0,
        "home_prob": round(home_prob, 4),
        "draw_prob": round(draw_prob, 4),
        "away_prob": round(away_prob, 4),
        "over25_prob": round(implied_probability(over25), 4),
        "under25_prob": round(implied_probability(under25), 4),
        "over15_prob": round(implied_probability(over15), 4),
        "under15_prob": round(implied_probability(under15), 4),
        "btts_yes_prob": round(implied_probability(btts_yes), 4),
        "btts_no_prob": round(implied_probability(btts_no), 4),
    }


def build_score_map(stats: dict[str, float]) -> dict[str, float]:
    home = stats["home_prob"]
    away = stats["away_prob"]
    draw = stats["draw_prob"]
    over25 = stats["over25_prob"]
    under25 = stats["under25_prob"]
    over15 = stats["over15_prob"]
    btts_yes = stats["btts_yes_prob"]
    btts_no = stats["btts_no_prob"]
    corners_proxy = min(0.82, (over25 * 0.55) + ((home + away) * 0.25) + 0.08)
    cards_proxy = min(0.80, (draw * 0.45) + ((home + away) * 0.25) + 0.10)
    first_half = min(0.84, (over15 * 0.7) + (over25 * 0.15) + 0.05)
    second_half = min(0.86, (over25 * 0.40) + (btts_yes * 0.20) + 0.18)
    return {
        "home_win": home,
        "away_win": away,
        "double_chance_home": min(0.95, home + draw),
        "double_chance_away": min(0.95, away + draw),
        "over25": over25,
        "under25": under25,
        "over15": over15,
        "btts_yes": btts_yes,
        "btts_no": btts_no,
        "corners": corners_proxy,
        "cards": cards_proxy,
        "first_half": first_half,
        "second_half": second_half,
        "safe": max(home, away, over15),
        "value": max(stats["home_price"] * home, stats["away_price"] * away, stats["over25"] * over25, stats["btts_yes"] * btts_yes),
    }


def upcoming_match_insights() -> list[MatchInsight]:
    now = datetime.now(UTC)
    cutoff = now + timedelta(hours=MATCH_WINDOW_HOURS)
    past_cutoff = now - timedelta(hours=PAST_GRACE_HOURS)
    insights: list[MatchInsight] = []
    raw_events = get_upcoming_odds()
    log_odds_debug("raw_soccer_events_before_time_filter", raw_events)
    for event in raw_events:
        kickoff = parse_iso_time(str(event.get("commence_time", "")))
        if not kickoff or kickoff < past_cutoff or kickoff > cutoff:
            continue
        bookmaker, markets = select_bookmaker(event)
        if "h2h" not in markets:
            continue
        home_team = str(event.get("home_team", "Unknown"))
        away_team = str(event.get("away_team", "Unknown"))
        support = build_support_stats(markets, home_team, away_team)
        insights.append(
            MatchInsight(
                match_id=str(event.get("id") or f"{home_team}-{away_team}-{event.get('commence_time', '')}"),
                competition=str(event.get("sport_title", "Football")),
                kickoff_utc=str(event.get("commence_time", "")),
                home_team=home_team,
                away_team=away_team,
                bookmaker=bookmaker,
                odds_home=support["home_price"] or None,
                odds_draw=support["draw_price"] or None,
                odds_away=support["away_price"] or None,
                totals_over_25=support["over25"] or None,
                totals_under_25=support["under25"] or None,
                btts_yes=support["btts_yes"] or None,
                btts_no=support["btts_no"] or None,
                totals_over_15=support["over15"] or None,
                totals_under_15=support["under15"] or None,
                score_map=build_score_map(support),
                support_stats=support,
            )
        )
    insights.sort(key=lambda item: item.kickoff_utc)
    log.info("match insights count=%s", len(insights))
    return insights


def confidence(prob: float, low: int = 48, high: int = 85) -> int:
    prob = max(0.0, min(1.0, prob))
    return max(low, min(high, round(low + (prob * (high - low)))))


def support_summary(insight: MatchInsight) -> str:
    s = insight.support_stats
    chunks = [f"1X2 {round(s['home_prob']*100)}-{round(s['draw_prob']*100)}-{round(s['away_prob']*100)}"]
    if insight.totals_over_25 and insight.totals_under_25:
        chunks.append(f"O2.5 {insight.totals_over_25:.2f} / U2.5 {insight.totals_under_25:.2f}")
    if insight.btts_yes and insight.btts_no:
        chunks.append(f"BTTS {insight.btts_yes:.2f} / {insight.btts_no:.2f}")
    if insight.totals_over_15:
        chunks.append(f"O1.5 {insight.totals_over_15:.2f}")
    return " | ".join(chunks)


def pick_for_mode(insight: MatchInsight, mode: str) -> dict[str, Any]:
    s = insight.score_map
    choices = {
        "today": ("Best Available", insight.home_team if s["home_win"] >= s["away_win"] else insight.away_team, confidence(max(s["safe"], s["over25"])), "Low-Medium", "B+", insight.odds_home if s["home_win"] >= s["away_win"] else insight.odds_away, max(s["safe"], s["over25"])),
        "live": ("Watchlist", "Wait for in-play entry on the stronger side", confidence(max(s["value"] / 2.5, 0.48), 45, 74), "Medium", "B", insight.odds_home if s["home_win"] >= s["away_win"] else insight.odds_away, s["value"]),
        "safe": ("Double Chance", f"{insight.home_team} or Draw" if s["double_chance_home"] >= s["double_chance_away"] else f"{insight.away_team} or Draw", confidence(max(s["double_chance_home"], s["double_chance_away"]), 56, 87), "Low", "A-", insight.odds_home if s["double_chance_home"] >= s["double_chance_away"] else insight.odds_away, max(s["double_chance_home"], s["double_chance_away"])),
        "value": ("Value Bet", "Over 2.5 Goals" if insight.totals_over_25 and s["over25"] >= max(s["home_win"], s["away_win"]) else (insight.home_team if s["home_win"] >= s["away_win"] else insight.away_team), confidence(max(s["over25"], s["home_win"], s["away_win"]), 50, 82), "Medium", "A", insight.totals_over_25 if insight.totals_over_25 and s["over25"] >= max(s["home_win"], s["away_win"]) else (insight.odds_home if s["home_win"] >= s["away_win"] else insight.odds_away), s["value"]),
        "acca": ("Accumulator Leg", "Over 1.5 Goals" if insight.totals_over_15 else (insight.home_team if s["home_win"] >= s["away_win"] else insight.away_team), confidence(max(s["over15"], s["safe"]), 52, 84), "Medium", "B+", insight.totals_over_15 if insight.totals_over_15 else (insight.odds_home if s["home_win"] >= s["away_win"] else insight.odds_away), max(s["over15"], s["safe"])),
        "corners": ("Corners Angle", "Over 8.5 Corners", confidence(s["corners"], 47, 76), "Medium", "B", None, s["corners"]),
        "cards": ("Cards Angle", "Over 3.5 Cards", confidence(s["cards"], 46, 75), "Medium-High", "B", None, s["cards"]),
        "goals": ("Goals", "Over 2.5 Goals" if s["over25"] >= s["under25"] else "Under 2.5 Goals", confidence(max(s["over25"], s["under25"]), 48, 81), "Medium", "B+", insight.totals_over_25 if s["over25"] >= s["under25"] else insight.totals_under_25, max(s["over25"], s["under25"])),
        "btts": ("BTTS", "BTTS Yes" if s["btts_yes"] >= s["btts_no"] else "BTTS No", confidence(max(s["btts_yes"], s["btts_no"]), 47, 78), "Medium", "B", insight.btts_yes if s["btts_yes"] >= s["btts_no"] else insight.btts_no, max(s["btts_yes"], s["btts_no"])),
        "over25": ("Goals", "Over 2.5 Goals", confidence(s["over25"], 48, 82), "Medium", "B+", insight.totals_over_25, s["over25"]),
        "under25": ("Goals", "Under 2.5 Goals", confidence(s["under25"], 46, 77), "Medium", "B", insight.totals_under_25, s["under25"]),
        "firsthalf": ("First Half", "Over 0.5 First Half Goals", confidence(s["first_half"], 46, 76), "Medium", "B", insight.totals_over_15, s["first_half"]),
        "secondhalf": ("Second Half", "Over 0.5 Second Half Goals", confidence(s["second_half"], 47, 78), "Medium", "B", insight.totals_over_15, s["second_half"]),
    }
    market, bet, conf, risk, value_rating, odds, key_score = choices.get(mode, choices["today"])
    return {
        "market": market,
        "bet": bet,
        "confidence": conf,
        "risk": risk,
        "value": value_rating,
        "odds": odds,
        "key": key_score,
        "stake": max(1, min(10, round((conf - 40) / 6))),
        "support": support_summary(insight),
    }


def ai_reasoning(mode: str, picks: list[tuple[MatchInsight, dict[str, Any]]]) -> dict[str, str]:
    cache_key = f"ai:{mode}:{'|'.join(item[0].match_id for item in picks)}"
    cached = ai_cache.get(cache_key)
    if cached is not None:
        return cached

    prompt_items = []
    for insight, pick in picks:
        prompt_items.append(
            {
                "match_id": insight.match_id,
                "match": f"{insight.home_team} vs {insight.away_team}",
                "market": pick["market"],
                "recommended_bet": pick["bet"],
                "odds": pick["odds"],
                "support": insight.support_stats,
            }
        )

    instruction = (
        "Use only the supplied market data. Do not invent xG, injuries, possession, form, head-to-head, or live stats. "
        "Return pure JSON mapping match_id to one short reason sentence."
    )

    if settings.gemini_api_key:
        try:
            response = retry_request(
                "POST",
                GEMINI_URL,
                headers={"Content-Type": "application/json", "x-goog-api-key": settings.gemini_api_key},
                json_body={
                    "contents": [{"parts": [{"text": instruction + "\n\n" + json.dumps(prompt_items, ensure_ascii=False)}]}],
                    "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
                },
                timeout=25,
            )
            if response.status_code == 200:
                data = response.json()
                text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    ai_cache.set(cache_key, parsed, AI_CACHE_TTL_SECONDS)
                    return parsed
        except Exception as exc:
            log.warning("Gemini reasoning failed: %s", exc)

    fallback = {item[0].match_id: "This angle follows the current bookmaker pricing and market balance only." for item in picks}
    ai_cache.set(cache_key, fallback, AI_CACHE_TTL_SECONDS)
    return fallback


def sort_insights(insights: list[MatchInsight], mode: str) -> list[MatchInsight]:
    key_map = {
        "today": "safe",
        "live": "value",
        "safe": "safe",
        "value": "value",
        "acca": "over15",
        "corners": "corners",
        "cards": "cards",
        "goals": "over25",
        "btts": "btts_yes",
        "over25": "over25",
        "under25": "under25",
        "firsthalf": "first_half",
        "secondhalf": "second_half",
    }
    key = key_map.get(mode, "safe")
    return sorted(insights, key=lambda item: item.score_map.get(key, 0.0), reverse=True)


def build_prediction_message(mode: str, insights: list[MatchInsight], page: int = 0) -> tuple[str, dict[str, Any]]:
    ordered = sort_insights(insights, mode)
    if not ordered:
        return (
            f"No football matches were available between the last {PAST_GRACE_HOURS} hours and the next {MATCH_WINDOW_HOURS} hours from The Odds API.",
            main_menu_keyboard(),
        )
    total_pages = max(1, (len(ordered) + MAX_PAGE_SIZE - 1) // MAX_PAGE_SIZE)
    page = max(0, min(page, total_pages - 1))
    selected = ordered[page * MAX_PAGE_SIZE : (page + 1) * MAX_PAGE_SIZE]
    picks = [(insight, pick_for_mode(insight, mode)) for insight in selected]
    ai_notes = ai_reasoning(mode, picks)
    title_map = {
        "today": "Today board",
        "live": "Live watchlist",
        "safe": "Safe board",
        "value": "Value board",
        "acca": "Acca board",
        "corners": "Corners board",
        "cards": "Cards board",
        "goals": "Goals board",
        "btts": "BTTS board",
        "over25": "Over 2.5 board",
        "under25": "Under 2.5 board",
        "firsthalf": "First half board",
        "secondhalf": "Second half board",
    }
    lines = [f"{title_map.get(mode, 'Betting board')} ({page + 1}/{total_pages})"]
    for insight, pick in picks:
        odds_text = f"{pick['odds']:.2f}" if isinstance(pick['odds'], (int, float)) and pick['odds'] > 0 else "N/A"
        lines.extend(
            [
                "",
                f"âš½ {insight.home_team} vs {insight.away_team}",
                f"ðŸ† {insight.competition}",
                f"ðŸ•’ {format_kickoff(insight.kickoff_utc)}",
                f"ðŸŽ¯ Market: {pick['market']}",
                f"âœ… Bet: {pick['bet']}",
                f"ðŸ”¥ Confidence: {pick['confidence']}%",
                f"ðŸ“Š Stats: {pick['support']}",
                f"ðŸ’° Odds: {odds_text} via {insight.bookmaker}",
                f"âš ï¸ Risk: {pick['risk']}",
                f"â­ Value: {pick['value']}",
                f"ðŸŽšï¸ Stake: {pick['stake']}/10",
                f"ðŸ§  Why: {ai_notes.get(insight.match_id, 'This angle follows the current bookmaker pricing and market balance only.')}",
            ]
        )
    return "\n".join(lines), prediction_keyboard(mode, page, total_pages)


def main_menu_keyboard() -> dict[str, Any]:
    return {
        "inline_keyboard": [
            [
                {"text": "Today", "callback_data": "menu:today:0"},
                {"text": "Safe", "callback_data": "menu:safe:0"},
                {"text": "Value", "callback_data": "menu:value:0"},
            ],
            [
                {"text": "Goals", "callback_data": "menu:goals:0"},
                {"text": "Corners", "callback_data": "menu:corners:0"},
                {"text": "Cards", "callback_data": "menu:cards:0"},
            ],
            [
                {"text": "Acca", "callback_data": "menu:acca:0"},
                {"text": "Stats", "callback_data": "menu:stats:0"},
            ],
        ]
    }


def prediction_keyboard(mode: str, page: int, total_pages: int) -> dict[str, Any]:
    rows: list[list[dict[str, str]]] = []
    nav: list[dict[str, str]] = []
    if page > 0:
        nav.append({"text": "Prev", "callback_data": f"menu:{mode}:{page - 1}"})
    if page + 1 < total_pages:
        nav.append({"text": "Next", "callback_data": f"menu:{mode}:{page + 1}"})
    if nav:
        rows.append(nav)
    rows.append([
        {"text": "Menu", "callback_data": "menu:menu:0"},
        {"text": "Stats", "callback_data": "menu:stats:0"},
    ])
    return {"inline_keyboard": rows}


def help_text() -> str:
    rows = ["Football Intelligence Pro", "", "Commands:"]
    for item in COMMAND_DESCRIPTIONS:
        rows.append(f"/{item['command']} - {item['description']}")
    rows.append("")
    rows.append("Examples: /team arsenal | /match arsenal vs chelsea | /analyze arsenal vs chelsea")
    return "\n".join(rows)


def parse_command(text: str) -> tuple[str, str]:
    if not text.startswith("/"):
        return "help", text
    body = text[1:]
    command_part, _, args = body.partition(" ")
    command = command_part.split("@", 1)[0].strip().lower()
    return ALIAS_MAP.get(command, command), args.strip()


def filter_by_team(insights: list[MatchInsight], query: str) -> list[MatchInsight]:
    q = query.lower().strip()
    return [item for item in insights if q and (q in item.home_team.lower() or q in item.away_team.lower())]


def find_match(insights: list[MatchInsight], left: str, right: str) -> MatchInsight | None:
    left_q = left.lower().strip()
    right_q = right.lower().strip()
    for item in insights:
        home = item.home_team.lower()
        away = item.away_team.lower()
        if left_q in home and right_q in away:
            return item
        if left_q in away and right_q in home:
            return item
    return None


def single_match_text(insight: MatchInsight, mode: str) -> str:
    pick = pick_for_mode(insight, mode)
    reason = ai_reasoning(mode, [(insight, pick)]).get(insight.match_id, "This angle follows the current bookmaker pricing and market balance only.")
    odds_text = f"{pick['odds']:.2f}" if isinstance(pick['odds'], (int, float)) and pick['odds'] > 0 else "N/A"
    return "\n".join([
        f"âš½ {insight.home_team} vs {insight.away_team}",
        f"ðŸ† {insight.competition}",
        f"ðŸ•’ {format_kickoff(insight.kickoff_utc)}",
        f"ðŸŽ¯ Market: {pick['market']}",
        f"âœ… Bet: {pick['bet']}",
        f"ðŸ”¥ Confidence: {pick['confidence']}%",
        f"ðŸ“Š Stats: {pick['support']}",
        f"ðŸ’° Odds: {odds_text} via {insight.bookmaker}",
        f"âš ï¸ Risk: {pick['risk']}",
        f"â­ Value: {pick['value']}",
        f"ðŸŽšï¸ Stake: {pick['stake']}/10",
        f"ðŸ§  Why: {reason}",
    ])


def stats_text(insights: list[MatchInsight]) -> str:
    if not insights:
        return "No odds data loaded right now."
    avg_home = round(sum(item.support_stats['home_prob'] for item in insights) / len(insights) * 100)
    over25_values = [item.support_stats['over25_prob'] for item in insights if item.support_stats['over25_prob'] > 0]
    btts_values = [item.support_stats['btts_yes_prob'] for item in insights if item.support_stats['btts_yes_prob'] > 0]
    avg_over25 = round(sum(over25_values) / len(over25_values) * 100) if over25_values else 0
    avg_btts = round(sum(btts_values) / len(btts_values) * 100) if btts_values else 0
    return "\n".join([
        "Loaded market stats",
        f"Matches in next {MATCH_WINDOW_HOURS}h: {len(insights)}",
        f"Average favorite implied win rate: {avg_home}%",
        f"Average over 2.5 implied rate: {avg_over25}%",
        f"Average BTTS implied rate: {avg_btts}%",
        "Data source: The Odds API h2h/totals/spreads markets, with BTTS estimated only when the API does not provide it directly, and AI used only for wording.",
    ])


def handle_text_command(command: str, args: str, insights: list[MatchInsight]) -> tuple[str, dict[str, Any] | None]:
    if command == "start":
        return (
            "Welcome to Football Intelligence Pro\n\nPremium betting assistant using Telegram, Odds API, Gemini, OpenRouter, and webhook only.",
            main_menu_keyboard(),
        )
    if command in {"help", "menu"}:
        return help_text(), main_menu_keyboard()
    if command in {"today", "live", "safe", "value", "acca", "corners", "cards", "goals", "btts", "over25", "under25", "firsthalf", "secondhalf"}:
        return build_prediction_message(command, insights, 0)
    if command == "stats":
        return stats_text(insights), {"inline_keyboard": [[{"text": "Menu", "callback_data": "menu:menu:0"}]]}
    if command == "team":
        if not args:
            return "Usage: /team arsenal", None
        matches = filter_by_team(insights, args)
        if not matches:
            return f"No upcoming fixtures found for {args}.", None
        lines = [f"Fixtures for {args}:"]
        for item in matches[:8]:
            lines.append(f"- {item.home_team} vs {item.away_team} | {format_kickoff(item.kickoff_utc)}")
        return "\n".join(lines), None
    if command in {"match", "analyze"}:
        if args and " vs " in args.lower():
            left, right = re.split(r"\s+vs\s+", args, maxsplit=1, flags=re.IGNORECASE)
            insight = find_match(insights, left, right)
            if not insight:
                return "Match not found in the current window.", None
            return single_match_text(insight, "value" if command == "analyze" else "today"), None
        if args:
            filtered = filter_by_team(insights, args)
            if filtered:
                return build_prediction_message("today", filtered, 0)
        return build_prediction_message("value" if command == "analyze" else "today", insights, 0)
    if command == "search":
        if not args:
            return "Usage: /search arsenal", None
        filtered = filter_by_team(insights, args)
        if not filtered:
            return f"No upcoming fixtures found for {args}.", None
        return build_prediction_message("today", filtered, 0)
    return "Unknown command. Use /help.", main_menu_keyboard()


@app.get("/")
def home() -> tuple[str, int]:
    return "Bot is alive.", 200


@app.get("/healthz")
def healthz() -> tuple[Any, int]:
    return jsonify({"ok": True, "time": datetime.now(UTC).isoformat()}), 200


@app.post(f"/webhook/{settings.webhook_secret}")
def webhook() -> tuple[Any, int]:
    try:
        update = request.get_json(force=True, silent=True) or {}
        log.info("Incoming update: %s", json.dumps(update)[:2000])

        if "callback_query" in update:
            callback = update["callback_query"]
            callback_id = str(callback.get("id") or "")
            chat_id = int(callback.get("message", {}).get("chat", {}).get("id", 0) or 0)
            data = str(callback.get("data") or "")
            parts = data.split(":")
            if len(parts) == 3 and parts[0] == "menu":
                mode = parts[1]
                page = int(parts[2]) if parts[2].isdigit() else 0
                insights = upcoming_match_insights()
                telegram.answer_callback(callback_id, f"Opened {mode}")
                if mode == "menu":
                    telegram.send_message(chat_id, help_text(), main_menu_keyboard())
                elif mode == "stats":
                    telegram.send_message(chat_id, stats_text(insights), {"inline_keyboard": [[{"text": "Menu", "callback_data": "menu:menu:0"}]]})
                else:
                    text, keyboard = build_prediction_message(mode, insights, page)
                    telegram.send_message(chat_id, text, keyboard)
            return jsonify({"ok": True}), 200

        message = update.get("message") or update.get("edited_message") or {}
        chat_id = int(message.get("chat", {}).get("id", 0) or 0)
        if not chat_id:
            return jsonify({"ok": True}), 200

        user_id = str(message.get("from", {}).get("id", "0"))
        text = str(message.get("text") or "").strip()
        command, args = parse_command(text)

        if rate_limiter.is_spamming(user_id):
            telegram.send_message(chat_id, "Slow down. Too many requests in a short time.")
            return jsonify({"ok": True}), 200

        if command not in {"start", "help", "menu"}:
            remaining = rate_limiter.cooldown_remaining(f"{user_id}:{command}")
            if remaining > 0:
                telegram.send_message(chat_id, f"Cooldown active. Please wait {remaining}s before /{command} again.")
                return jsonify({"ok": True}), 200

        insights = upcoming_match_insights()
        text_out, keyboard = handle_text_command(command, args, insights)
        telegram.send_message(chat_id, text_out, keyboard)
        return jsonify({"ok": True}), 200
    except Exception as exc:
        log.exception("Webhook error: %s", exc)
        return jsonify({"ok": False, "error": "internal_error"}), 500


telegram.set_commands()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)