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
    football_api_key: str = os.getenv("FOOTBALL_API_KEY", "").strip()
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
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT = 20
CACHE_TTL_SECONDS = 300
AI_CACHE_TTL_SECONDS = 180
COOLDOWN_SECONDS = 6
SPAM_WINDOW_SECONDS = 20
SPAM_MAX_REQUESTS = 8
MATCH_WINDOW_HOURS = 72
PAST_GRACE_HOURS = 12
MAX_PAGE_SIZE = 5
BOOKMAKER_SCAN_LIMIT = 10
FORM_FIXTURES = 6


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
session.headers.update({"User-Agent": "smart-football-bot/7.0"})


@dataclass(slots=True)
class MatchInsight:
    match_id: str
    fixture_id: int | None
    league_id: int | None
    season: int | None
    competition: str
    kickoff_utc: str
    home_team: str
    away_team: str
    home_team_id: int | None
    away_team_id: int | None
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
    data_notes: list[str]


COMMAND_DESCRIPTIONS: list[dict[str, str]] = [
    {"command": "start", "description": "Open the welcome screen"},
    {"command": "help", "description": "Show all commands"},
    {"command": "menu", "description": "Open the main menu"},
    {"command": "today", "description": "Best football matches"},
    {"command": "live", "description": "Upcoming watchlist"},
    {"command": "now", "description": "Alias of live"},
    {"command": "safe", "description": "Safer picks"},
    {"command": "value", "description": "Value picks"},
    {"command": "vip", "description": "Alias of value"},
    {"command": "banker", "description": "Alias of safe"},
    {"command": "acca", "description": "Accumulator picks"},
    {"command": "corners", "description": "Corners angles"},
    {"command": "cards", "description": "Cards angles"},
    {"command": "goals", "description": "Goals picks"},
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
    {"command": "team", "description": "Search a team in fixtures"},
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


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


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


def football_headers() -> dict[str, str]:
    return {"x-apisports-key": settings.football_api_key} if settings.football_api_key else {}


def football_get(path: str, params: dict[str, Any], cache_suffix: str, ttl: int = CACHE_TTL_SECONDS) -> dict[str, Any] | None:
    if not settings.football_api_key:
        return None
    cache_key = f"football:{cache_suffix}:{json.dumps(params, sort_keys=True)}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    response = retry_request("GET", f"{API_FOOTBALL_BASE}{path}", params=params, headers=football_headers(), timeout=25)
    if response.status_code != 200:
        log.warning("API-Football failed path=%s status=%s body=%s", path, response.status_code, response.text[:600])
        return None
    payload = response.json()
    cache.set(cache_key, payload, ttl)
    return payload


class TelegramClient:
    def send_message(self, chat_id: int, text: str, reply_markup: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"chat_id": chat_id, "text": text[:4096], "disable_web_page_preview": True}
        if reply_markup:
            payload["reply_markup"] = reply_markup
        retry_request("POST", f"{TELEGRAM_API}/sendMessage", json_body=payload, timeout=15)

    def answer_callback(self, callback_id: str, text: str) -> None:
        retry_request("POST", f"{TELEGRAM_API}/answerCallbackQuery", json_body={"callback_query_id": callback_id, "text": text[:180]}, timeout=10)

    def set_commands(self) -> None:
        try:
            retry_request("POST", f"{TELEGRAM_API}/setMyCommands", json_body={"commands": COMMAND_DESCRIPTIONS}, timeout=15)
        except Exception as exc:
            log.warning("setMyCommands failed: %s", exc)


telegram = TelegramClient()


def get_odds_sport_keys() -> list[str]:
    cached = cache.get("odds:sport_keys")
    if cached is not None:
        return cached
    if not settings.odds_api_key:
        return []
    response = retry_request("GET", ODDS_SPORTS_URL, params={"apiKey": settings.odds_api_key}, timeout=20)
    if response.status_code != 200:
        return []
    payload = response.json()
    sports = payload if isinstance(payload, list) else []
    keys = [str(item.get("key", "")) for item in sports if str(item.get("key", "")).startswith("soccer_") and not item.get("has_outrights", False)]
    cache.set("odds:sport_keys", keys, CACHE_TTL_SECONDS)
    return keys


def fetch_odds_events_for_sport(sport_key: str) -> list[dict[str, Any]]:
    cache_key = f"odds:sport:{sport_key}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    response = retry_request(
        "GET",
        f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds",
        params={
            "apiKey": settings.odds_api_key,
            "regions": "eu,uk,us,au",
            "markets": "h2h,totals,spreads",
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        },
        timeout=20,
    )
    if response.status_code != 200:
        return []
    payload = response.json()
    events = payload if isinstance(payload, list) else []
    cache.set(cache_key, events, CACHE_TTL_SECONDS)
    return events


def get_odds_events() -> list[dict[str, Any]]:
    cached = cache.get("odds:upcoming")
    if cached is not None:
        return cached
    if not settings.odds_api_key:
        return []
    all_events: list[dict[str, Any]] = []
    for sport_key in get_odds_sport_keys()[:25]:
        all_events.extend(fetch_odds_events_for_sport(sport_key))
    deduped: dict[str, dict[str, Any]] = {}
    for event in all_events:
        event_id = str(event.get("id") or f"{event.get('home_team', '')}-{event.get('away_team', '')}-{event.get('commence_time', '')}")
        deduped[event_id] = event
    result = list(deduped.values())
    cache.set("odds:upcoming", result, CACHE_TTL_SECONDS)
    return result


def get_fixture_events() -> list[dict[str, Any]]:
    cached = cache.get("fixtures:upcoming")
    if cached is not None:
        return cached
    if not settings.football_api_key:
        return []
    now = datetime.now(UTC)
    day_values = sorted({(now + timedelta(days=offset)).strftime("%Y-%m-%d") for offset in range(-1, 5)})
    events: list[dict[str, Any]] = []
    for day_value in day_values:
        payload = football_get("/fixtures", {"date": day_value, "timezone": "UTC"}, f"fixtures:{day_value}")
        if not payload:
            continue
        day_events = payload.get("response", []) if isinstance(payload, dict) else []
        events.extend(day_events)
    deduped: dict[str, dict[str, Any]] = {}
    for item in events:
        fixture = item.get("fixture", {})
        teams = item.get("teams", {})
        fixture_id = str(fixture.get("id") or "")
        key = fixture_id or f"{teams.get('home', {}).get('name', '')}-{teams.get('away', {}).get('name', '')}-{fixture.get('date', '')}"
        deduped[key] = item
    merged = list(deduped.values())
    cache.set("fixtures:upcoming", merged, CACHE_TTL_SECONDS)
    return merged


def choose_best_bookmaker(event: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    bookmakers = (event.get("bookmakers", []) or [])[:BOOKMAKER_SCAN_LIMIT]
    best_title = "No odds"
    best_markets: dict[str, Any] = {}
    best_score = -1
    for bookmaker in bookmakers:
        market_map = {market.get("key"): market for market in bookmaker.get("markets", [])}
        score = int("h2h" in market_map) + int("totals" in market_map) + int("spreads" in market_map)
        if score > best_score:
            best_score = score
            best_title = str(bookmaker.get("title", "No odds"))
            best_markets = market_map
    return best_title, best_markets


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


def estimate_btts_from_totals(over25: float | None, under25: float | None) -> tuple[float | None, float | None]:
    if not over25 or not under25:
        return None, None
    over25_prob = implied_probability(over25)
    under25_prob = implied_probability(under25)
    btts_yes = round(max(1.55, min(2.60, 1.15 + (1.65 - over25_prob) + (under25_prob * 0.55))), 2)
    btts_no = round(max(1.55, min(2.60, 1.20 + (1.55 - under25_prob) + (over25_prob * 0.45))), 2)
    return btts_yes, btts_no


def to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(str(value).replace("%", "").strip())
    except Exception:
        return None


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def team_stats_lookup(team_id: int, league_id: int, season: int) -> dict[str, float]:
    payload = football_get("/teams/statistics", {"team": team_id, "league": league_id, "season": season}, f"teamstats:{team_id}:{league_id}:{season}", ttl=1800)
    stats = payload.get("response", {}) if isinstance(payload, dict) else {}
    fixtures = stats.get("fixtures", {}) if isinstance(stats, dict) else {}
    goals = stats.get("goals", {}) if isinstance(stats, dict) else {}
    cards = stats.get("cards", {}) if isinstance(stats, dict) else {}
    played_total = to_float(fixtures.get("played", {}).get("total")) or 0.0
    wins_total = to_float(fixtures.get("wins", {}).get("total")) or 0.0
    draws_total = to_float(fixtures.get("draws", {}).get("total")) or 0.0
    loses_total = to_float(fixtures.get("loses", {}).get("total")) or 0.0
    gf_total = to_float(goals.get("for", {}).get("total", {}).get("total")) or 0.0
    ga_total = to_float(goals.get("against", {}).get("total", {}).get("total")) or 0.0
    avg_gf = to_float(goals.get("for", {}).get("average", {}).get("total")) or (gf_total / played_total if played_total else 0.0)
    avg_ga = to_float(goals.get("against", {}).get("average", {}).get("total")) or (ga_total / played_total if played_total else 0.0)
    over25_pct = to_float(goals.get("for", {}).get("minute", {}).get("76-90", {}).get("percentage"))
    yellow_total = 0.0
    for minute_bucket in cards.get("yellow", {}).values() if isinstance(cards.get("yellow"), dict) else []:
        total = to_float(minute_bucket.get("total"))
        if total:
            yellow_total += total
    return {
        "played_total": played_total,
        "wins_total": wins_total,
        "draws_total": draws_total,
        "loses_total": loses_total,
        "avg_gf": round(avg_gf, 3),
        "avg_ga": round(avg_ga, 3),
        "yellow_total": yellow_total,
        "late_goal_pct": (over25_pct or 0.0),
    }


def fixture_stats_lookup(fixture_id: int) -> dict[str, dict[str, float]]:
    payload = football_get("/fixtures/statistics", {"fixture": fixture_id}, f"fixturestats:{fixture_id}", ttl=900)
    response = payload.get("response", []) if isinstance(payload, dict) else []
    parsed: dict[str, dict[str, float]] = {}
    for item in response:
        team = item.get("team", {})
        team_id = int(team.get("id", 0) or 0)
        stats = item.get("statistics", [])
        row: dict[str, float] = {}
        for stat in stats:
            stat_type = str(stat.get("type", ""))
            value = to_float(stat.get("value"))
            if value is not None:
                row[stat_type] = value
        if team_id:
            parsed[str(team_id)] = row
    return parsed


def recent_form_lookup(team_id: int, season: int) -> dict[str, float]:
    payload = football_get("/fixtures", {"team": team_id, "last": FORM_FIXTURES, "season": season, "timezone": "UTC"}, f"recent:{team_id}:{season}", ttl=900)
    fixtures = payload.get("response", []) if isinstance(payload, dict) else []
    goals_for = 0.0
    goals_against = 0.0
    wins = 0.0
    draws = 0.0
    losses = 0.0
    count = 0.0
    for item in fixtures:
        teams = item.get("teams", {})
        goals = item.get("goals", {})
        home_id = int(teams.get("home", {}).get("id", 0) or 0)
        away_id = int(teams.get("away", {}).get("id", 0) or 0)
        home_goals = to_float(goals.get("home"))
        away_goals = to_float(goals.get("away"))
        if home_goals is None or away_goals is None:
            continue
        if team_id == home_id:
            gf, ga = home_goals, away_goals
        elif team_id == away_id:
            gf, ga = away_goals, home_goals
        else:
            continue
        goals_for += gf
        goals_against += ga
        count += 1
        if gf > ga:
            wins += 1
        elif gf == ga:
            draws += 1
        else:
            losses += 1
    if count == 0:
        return {"form_gf": 0.0, "form_ga": 0.0, "form_points": 0.0, "form_matches": 0.0}
    return {
        "form_gf": round(goals_for / count, 3),
        "form_ga": round(goals_against / count, 3),
        "form_points": round(((wins * 3) + draws) / (count * 3), 3),
        "form_matches": count,
    }


def enrich_stats(home_team_id: int | None, away_team_id: int | None, league_id: int | None, season: int | None, fixture_id: int | None) -> tuple[dict[str, float], list[str]]:
    notes: list[str] = []
    if not home_team_id or not away_team_id or not league_id or not season:
        return {}, notes
    home_team_stats = team_stats_lookup(home_team_id, league_id, season)
    away_team_stats = team_stats_lookup(away_team_id, league_id, season)
    home_form = recent_form_lookup(home_team_id, season)
    away_form = recent_form_lookup(away_team_id, season)
    fixture_stats = fixture_stats_lookup(fixture_id) if fixture_id else {}
    home_fixture = fixture_stats.get(str(home_team_id), {})
    away_fixture = fixture_stats.get(str(away_team_id), {})
    enriched = {
        "home_avg_gf": home_team_stats.get("avg_gf", 0.0),
        "home_avg_ga": home_team_stats.get("avg_ga", 0.0),
        "away_avg_gf": away_team_stats.get("avg_gf", 0.0),
        "away_avg_ga": away_team_stats.get("avg_ga", 0.0),
        "home_form_points": home_form.get("form_points", 0.0),
        "away_form_points": away_form.get("form_points", 0.0),
        "home_form_gf": home_form.get("form_gf", 0.0),
        "home_form_ga": home_form.get("form_ga", 0.0),
        "away_form_gf": away_form.get("form_gf", 0.0),
        "away_form_ga": away_form.get("form_ga", 0.0),
        "home_possession": home_fixture.get("Ball Possession", 0.0),
        "away_possession": away_fixture.get("Ball Possession", 0.0),
        "home_shots_on": home_fixture.get("Shots on Goal", 0.0),
        "away_shots_on": away_fixture.get("Shots on Goal", 0.0),
        "home_corners": home_fixture.get("Corner Kicks", 0.0),
        "away_corners": away_fixture.get("Corner Kicks", 0.0),
        "home_yellows": home_fixture.get("Yellow Cards", 0.0),
        "away_yellows": away_fixture.get("Yellow Cards", 0.0),
    }
    if home_team_stats.get("avg_gf") or away_team_stats.get("avg_gf"):
        notes.append("team-stats")
    if home_form.get("form_matches") or away_form.get("form_matches"):
        notes.append("recent-form")
    if home_fixture or away_fixture:
        notes.append("fixture-stats")
    return enriched, notes


def build_support_stats(markets: dict[str, Any], home_team: str, away_team: str) -> dict[str, float]:
    home_price = find_price(markets.get("h2h"), [home_team])
    draw_price = find_price(markets.get("h2h"), ["draw"])
    away_price = find_price(markets.get("h2h"), [away_team])
    over25 = find_price(markets.get("totals"), ["over"], 2.5)
    under25 = find_price(markets.get("totals"), ["under"], 2.5)
    over15 = find_price(markets.get("totals"), ["over"], 1.5)
    under15 = find_price(markets.get("totals"), ["under"], 1.5)
    btts_yes, btts_no = estimate_btts_from_totals(over25, under25)
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
    home = stats.get("home_prob", 0.0)
    away = stats.get("away_prob", 0.0)
    draw = stats.get("draw_prob", 0.0)
    over25 = stats.get("over25_prob", 0.0)
    under25 = stats.get("under25_prob", 0.0)
    over15 = stats.get("over15_prob", 0.0)
    btts_yes = stats.get("btts_yes_prob", 0.0)
    home_attack = stats.get("home_avg_gf", 0.0) + stats.get("home_form_gf", 0.0)
    away_attack = stats.get("away_avg_gf", 0.0) + stats.get("away_form_gf", 0.0)
    home_def_weak = stats.get("home_avg_ga", 0.0) + stats.get("home_form_ga", 0.0)
    away_def_weak = stats.get("away_avg_ga", 0.0) + stats.get("away_form_ga", 0.0)
    form_edge_home = stats.get("home_form_points", 0.0)
    form_edge_away = stats.get("away_form_points", 0.0)
    home_shots = stats.get("home_shots_on", 0.0)
    away_shots = stats.get("away_shots_on", 0.0)
    corners_proxy = clamp(0.35 + ((stats.get("home_corners", 0.0) + stats.get("away_corners", 0.0)) / 20.0) + (over25 * 0.20), 0.18, 0.86)
    cards_proxy = clamp(0.30 + ((stats.get("home_yellows", 0.0) + stats.get("away_yellows", 0.0)) / 12.0) + (draw * 0.15), 0.18, 0.84)
    attack_total = home_attack + away_attack
    defense_total = home_def_weak + away_def_weak
    stat_goals = clamp(0.20 + (attack_total / 6.0) + (defense_total / 8.0), 0.18, 0.88)
    stat_under = clamp(1.0 - (stat_goals * 0.82), 0.12, 0.82)
    stat_btts = clamp(0.18 + ((home_attack + away_attack) / 7.0) + ((home_def_weak + away_def_weak) / 10.0), 0.16, 0.86)
    home_strength = clamp((home * 0.45) + (form_edge_home * 0.20) + ((home_attack - away_def_weak + 2.0) / 5.0 * 0.20) + ((home_shots - away_shots + 6.0) / 12.0 * 0.15), 0.05, 0.90)
    away_strength = clamp((away * 0.45) + (form_edge_away * 0.20) + ((away_attack - home_def_weak + 2.0) / 5.0 * 0.20) + ((away_shots - home_shots + 6.0) / 12.0 * 0.15), 0.05, 0.90)
    return {
        "home_win": home_strength,
        "away_win": away_strength,
        "double_chance_home": clamp(home_strength + draw, 0.12, 0.95),
        "double_chance_away": clamp(away_strength + draw, 0.12, 0.95),
        "over25": clamp((over25 * 0.55) + (stat_goals * 0.45), 0.12, 0.92),
        "under25": clamp((under25 * 0.55) + (stat_under * 0.45), 0.08, 0.88),
        "over15": clamp((over15 * 0.55) + (stat_goals * 0.45) + 0.10, 0.20, 0.95),
        "btts_yes": clamp((btts_yes * 0.55) + (stat_btts * 0.45), 0.10, 0.88),
        "btts_no": clamp(1.0 - ((btts_yes * 0.55) + (stat_btts * 0.45)), 0.10, 0.88),
        "corners": corners_proxy,
        "cards": cards_proxy,
        "first_half": clamp((over15 * 0.30) + (stat_goals * 0.45) + 0.10, 0.10, 0.84),
        "second_half": clamp((over25 * 0.30) + (stat_goals * 0.40) + 0.16, 0.10, 0.88),
        "safe": clamp(max(home_strength, away_strength, over15), 0.10, 0.95),
        "value": clamp(max(stats.get("home_price", 0.0) * home_strength, stats.get("away_price", 0.0) * away_strength, stats.get("over25", 0.0) * ((over25 * 0.55) + (stat_goals * 0.45)), stats.get("btts_yes", 0.0) * ((btts_yes * 0.55) + (stat_btts * 0.45))), 0.0, 3.2),
    }


def fixture_to_insight(fixture: dict[str, Any], odds_event: dict[str, Any] | None) -> MatchInsight:
    fixture_info = fixture.get("fixture", {})
    teams = fixture.get("teams", {})
    league = fixture.get("league", {})
    home_team = str(teams.get("home", {}).get("name", "Unknown"))
    away_team = str(teams.get("away", {}).get("name", "Unknown"))
    home_team_id = int(teams.get("home", {}).get("id", 0) or 0) or None
    away_team_id = int(teams.get("away", {}).get("id", 0) or 0) or None
    league_id = int(league.get("id", 0) or 0) or None
    season = int(league.get("season", 0) or 0) or None
    fixture_id = int(fixture_info.get("id", 0) or 0) or None
    bookmaker = "No odds"
    support = {
        "home_price": 0.0,
        "draw_price": 0.0,
        "away_price": 0.0,
        "over25": 0.0,
        "under25": 0.0,
        "over15": 0.0,
        "under15": 0.0,
        "btts_yes": 0.0,
        "btts_no": 0.0,
        "home_prob": 0.38,
        "draw_prob": 0.27,
        "away_prob": 0.35,
        "over25_prob": 0.52,
        "under25_prob": 0.48,
        "over15_prob": 0.68,
        "under15_prob": 0.32,
        "btts_yes_prob": 0.53,
        "btts_no_prob": 0.47,
    }
    if odds_event:
        bookmaker, markets = choose_best_bookmaker(odds_event)
        if markets:
            support = build_support_stats(markets, home_team, away_team)
    enriched, data_notes = enrich_stats(home_team_id, away_team_id, league_id, season, fixture_id)
    support.update(enriched)
    return MatchInsight(
        match_id=str(fixture_info.get("id") or f"{home_team}-{away_team}-{fixture_info.get('date', '')}"),
        fixture_id=fixture_id,
        league_id=league_id,
        season=season,
        competition=str(league.get("name", "Football")),
        kickoff_utc=str(fixture_info.get("date", "")),
        home_team=home_team,
        away_team=away_team,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
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
        data_notes=data_notes,
    )


def merge_fixture_with_odds(fixtures: list[dict[str, Any]], odds_events: list[dict[str, Any]]) -> list[MatchInsight]:
    odds_index: dict[tuple[str, str], dict[str, Any]] = {}
    for event in odds_events:
        home = normalize_name(str(event.get("home_team", "")).strip())
        away = normalize_name(str(event.get("away_team", "")).strip())
        if home and away:
            odds_index[(home, away)] = event
            odds_index[(away, home)] = event
    insights: list[MatchInsight] = []
    now = datetime.now(UTC)
    cutoff = now + timedelta(hours=MATCH_WINDOW_HOURS)
    past_cutoff = now - timedelta(hours=PAST_GRACE_HOURS)
    seen: set[str] = set()
    for fixture in fixtures:
        fixture_info = fixture.get("fixture", {})
        teams = fixture.get("teams", {})
        home_team = str(teams.get("home", {}).get("name", "Unknown"))
        away_team = str(teams.get("away", {}).get("name", "Unknown"))
        key = f"{normalize_name(home_team)}::{normalize_name(away_team)}::{str(fixture_info.get('date', ''))[:16]}"
        if key in seen:
            continue
        seen.add(key)
        kickoff_utc = str(fixture_info.get("date", ""))
        kickoff = parse_iso_time(kickoff_utc)
        if not kickoff or kickoff < past_cutoff or kickoff > cutoff:
            continue
        odds_event = odds_index.get((normalize_name(home_team), normalize_name(away_team)))
        insights.append(fixture_to_insight(fixture, odds_event))
    insights.sort(key=lambda item: item.kickoff_utc)
    return insights


def odds_only_insights(odds_events: list[dict[str, Any]]) -> list[MatchInsight]:
    insights: list[MatchInsight] = []
    now = datetime.now(UTC)
    cutoff = now + timedelta(hours=MATCH_WINDOW_HOURS)
    past_cutoff = now - timedelta(hours=PAST_GRACE_HOURS)
    seen: set[str] = set()
    for event in odds_events:
        home_team = str(event.get("home_team", "Unknown"))
        away_team = str(event.get("away_team", "Unknown"))
        kickoff_utc = str(event.get("commence_time", ""))
        key = f"{normalize_name(home_team)}::{normalize_name(away_team)}::{kickoff_utc[:16]}"
        if key in seen:
            continue
        seen.add(key)
        kickoff = parse_iso_time(kickoff_utc)
        if not kickoff or kickoff < past_cutoff or kickoff > cutoff:
            continue
        bookmaker, markets = choose_best_bookmaker(event)
        support = build_support_stats(markets, home_team, away_team) if markets else {
            "home_price": 0.0,
            "draw_price": 0.0,
            "away_price": 0.0,
            "over25": 0.0,
            "under25": 0.0,
            "over15": 0.0,
            "under15": 0.0,
            "btts_yes": 0.0,
            "btts_no": 0.0,
            "home_prob": 0.38,
            "draw_prob": 0.27,
            "away_prob": 0.35,
            "over25_prob": 0.52,
            "under25_prob": 0.48,
            "over15_prob": 0.68,
            "under15_prob": 0.32,
            "btts_yes_prob": 0.53,
            "btts_no_prob": 0.47,
        }
        insights.append(MatchInsight(
            match_id=str(event.get("id") or key),
            fixture_id=None,
            league_id=None,
            season=None,
            competition=str(event.get("sport_title", "Football")),
            kickoff_utc=kickoff_utc,
            home_team=home_team,
            away_team=away_team,
            home_team_id=None,
            away_team_id=None,
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
            data_notes=["odds-only"],
        ))
    insights.sort(key=lambda item: item.kickoff_utc)
    return insights


def upcoming_match_insights() -> list[MatchInsight]:
    fixtures = get_fixture_events()
    odds_events = get_odds_events()
    merged = merge_fixture_with_odds(fixtures, odds_events) if fixtures else []
    odds_only = odds_only_insights(odds_events) if len(merged) < 10 else []
    combined: list[MatchInsight] = []
    seen: set[str] = set()
    for item in merged + odds_only:
        key = f"{normalize_name(item.home_team)}::{normalize_name(item.away_team)}::{item.kickoff_utc[:16]}"
        if key in seen:
            continue
        seen.add(key)
        combined.append(item)
    return combined


def confidence(prob: float, low: int = 48, high: int = 89) -> int:
    prob = max(0.0, min(1.0, prob))
    return max(low, min(high, round(low + (prob * (high - low)))))


def support_summary(insight: MatchInsight) -> str:
    s = insight.support_stats
    parts = [f"1X2 {round(s.get('home_prob', 0) * 100)}-{round(s.get('draw_prob', 0) * 100)}-{round(s.get('away_prob', 0) * 100)}"]
    if s.get("home_avg_gf") or s.get("away_avg_gf"):
        parts.append(f"GF {s.get('home_avg_gf', 0):.2f}-{s.get('away_avg_gf', 0):.2f}")
    if s.get("home_form_points") or s.get("away_form_points"):
        parts.append(f"Form {round(s.get('home_form_points', 0) * 100)}-{round(s.get('away_form_points', 0) * 100)}")
    if insight.totals_over_25 and insight.totals_under_25:
        parts.append(f"O2.5 {insight.totals_over_25:.2f} / U2.5 {insight.totals_under_25:.2f}")
    if s.get("home_corners") or s.get("away_corners"):
        parts.append(f"Corners {s.get('home_corners', 0):.0f}-{s.get('away_corners', 0):.0f}")
    if s.get("home_yellows") or s.get("away_yellows"):
        parts.append(f"Cards {s.get('home_yellows', 0):.0f}-{s.get('away_yellows', 0):.0f}")
    return " | ".join(parts)


def build_reason(insight: MatchInsight, bet: str) -> str:
    s = insight.support_stats
    reasons: list[str] = []
    if "Over 2.5" in bet or "Over 1.5" in bet:
        if s.get("home_avg_gf", 0) + s.get("away_avg_gf", 0) >= 2.4:
            reasons.append("both teams bring a solid scoring average")
        if s.get("home_form_gf", 0) + s.get("away_form_gf", 0) >= 2.2:
            reasons.append("recent form supports goals")
        if s.get("home_shots_on", 0) + s.get("away_shots_on", 0) >= 7:
            reasons.append("shot volume supports a goals angle")
    if "BTTS" in bet:
        if s.get("home_avg_gf", 0) >= 1.0 and s.get("away_avg_gf", 0) >= 1.0:
            reasons.append("both sides usually find a goal")
        if s.get("home_avg_ga", 0) >= 1.0 and s.get("away_avg_ga", 0) >= 1.0:
            reasons.append("both defenses allow chances")
    if "Corners" in bet:
        if s.get("home_corners", 0) + s.get("away_corners", 0) >= 8:
            reasons.append("recent fixture stats show healthy corner volume")
    if "Cards" in bet:
        if s.get("home_yellows", 0) + s.get("away_yellows", 0) >= 4:
            reasons.append("recent fixture stats point to card activity")
    if "Draw" in bet or insight.home_team in bet or insight.away_team in bet:
        if s.get("home_form_points", 0) > s.get("away_form_points", 0) + 0.15:
            reasons.append(f"{insight.home_team} have the stronger recent form")
        elif s.get("away_form_points", 0) > s.get("home_form_points", 0) + 0.15:
            reasons.append(f"{insight.away_team} have the stronger recent form")
        if s.get("home_avg_gf", 0) - s.get("away_avg_ga", 0) >= 0.35:
            reasons.append(f"{insight.home_team} attack well against this defensive profile")
        if s.get("away_avg_gf", 0) - s.get("home_avg_ga", 0) >= 0.35:
            reasons.append(f"{insight.away_team} attack well against this defensive profile")
    if not reasons:
        reasons.append("the available team statistics and market prices point in the same direction")
    return "; ".join(reasons[:3]).capitalize() + "."


def pick_for_mode(insight: MatchInsight, mode: str, rank_index: int = 0) -> dict[str, Any]:
    s = insight.score_map
    default_bet = insight.home_team if s["home_win"] >= s["away_win"] else insight.away_team
    mapping = {
        "safe": ("Double Chance", f"{insight.home_team} or Draw" if s["double_chance_home"] >= s["double_chance_away"] else f"{insight.away_team} or Draw", confidence(max(s["double_chance_home"], s["double_chance_away"]), 58, 89), "Low", "A-", insight.odds_home if s["double_chance_home"] >= s["double_chance_away"] else insight.odds_away, max(s["double_chance_home"], s["double_chance_away"])),
        "value": ("Value Bet", "Over 2.5 Goals" if s["over25"] >= max(s["home_win"], s["away_win"]) else default_bet, confidence(max(s["over25"], s["home_win"], s["away_win"]), 52, 86), "Medium", "A", insight.totals_over_25 if s["over25"] >= max(s["home_win"], s["away_win"]) else (insight.odds_home if default_bet == insight.home_team else insight.odds_away), s["value"]),
        "acca": ("Accumulator Leg", "Over 1.5 Goals" if s["over15"] >= max(s["home_win"], s["away_win"]) else default_bet, confidence(max(s["over15"], s["safe"]), 54, 87), "Medium", "B+", insight.totals_over_15 if s["over15"] >= max(s["home_win"], s["away_win"]) else (insight.odds_home if default_bet == insight.home_team else insight.odds_away), max(s["over15"], s["safe"])),
        "corners": ("Corners", "Over 8.5 Corners", confidence(s["corners"], 50, 82), "Medium", "B+", None, s["corners"]),
        "cards": ("Cards", "Over 3.5 Cards", confidence(s["cards"], 50, 81), "Medium", "B+", None, s["cards"]),
        "goals": ("Goals", "Over 2.5 Goals" if s["over25"] >= s["under25"] else "Under 2.5 Goals", confidence(max(s["over25"], s["under25"]), 50, 84), "Medium", "B+", insight.totals_over_25 if s["over25"] >= s["under25"] else insight.totals_under_25, max(s["over25"], s["under25"])),
        "btts": ("BTTS", "BTTS Yes" if s["btts_yes"] >= s["btts_no"] else "BTTS No", confidence(max(s["btts_yes"], s["btts_no"]), 50, 82), "Medium", "B", insight.btts_yes if s["btts_yes"] >= s["btts_no"] else insight.btts_no, max(s["btts_yes"], s["btts_no"])),
        "over25": ("Goals", "Over 2.5 Goals", confidence(s["over25"], 50, 85), "Medium", "B+", insight.totals_over_25, s["over25"]),
        "under25": ("Goals", "Under 2.5 Goals", confidence(s["under25"], 48, 82), "Medium", "B", insight.totals_under_25, s["under25"]),
        "firsthalf": ("First Half", "Over 0.5 First Half Goals", confidence(s["first_half"], 49, 80), "Medium", "B", insight.totals_over_15, s["first_half"]),
        "secondhalf": ("Second Half", "Over 0.5 Second Half Goals", confidence(s["second_half"], 50, 82), "Medium", "B", insight.totals_over_15, s["second_half"]),
        "live": ("Live Opportunity", f"Back {default_bet} live if the price improves after 10-20 minutes", confidence(max(s["value"] / 2.5, 0.48), 52, 80), "Medium", "B+", insight.odds_home if s["home_win"] >= s["away_win"] else insight.odds_away, s["value"]),
    }
    if mode == "today":
        cycle = rank_index % 4
        if cycle == 0:
            market, bet, conf, risk, value_rating, odds, key = mapping["safe"]
        elif cycle == 1:
            market, bet, conf, risk, value_rating, odds, key = mapping["goals"]
        elif cycle == 2:
            market, bet, conf, risk, value_rating, odds, key = mapping["value"]
        else:
            market = "Match Winner"
            bet = default_bet
            odds = insight.odds_home if default_bet == insight.home_team else insight.odds_away
            key = max(s["home_win"], s["away_win"])
            conf = confidence(key, 52, 86)
            risk = "Medium"
            value_rating = "B"
    else:
        market, bet, conf, risk, value_rating, odds, key = mapping.get(mode, mapping["safe"])
    return {
        "market": market,
        "bet": bet,
        "confidence": conf,
        "risk": risk,
        "value": value_rating,
        "odds": odds,
        "key": key,
        "stake": max(1, min(10, round((conf - 40) / 6))),
        "support": support_summary(insight),
        "why": build_reason(insight, bet),
    }


def sort_insights(insights: list[MatchInsight], mode: str) -> list[MatchInsight]:
    key_map = {"today": "safe", "live": "value", "safe": "safe", "value": "value", "acca": "over15", "corners": "corners", "cards": "cards", "goals": "over25", "btts": "btts_yes", "over25": "over25", "under25": "under25", "firsthalf": "first_half", "secondhalf": "second_half"}
    key = key_map.get(mode, "safe")
    ordered = sorted(insights, key=lambda item: (item.score_map.get(key, 0.0), item.kickoff_utc), reverse=True)
    if mode == "live":
        return [item for item in ordered if (item.odds_home or 0) > 0 or (item.odds_away or 0) > 0][:25]
    return ordered


def build_prediction_message(mode: str, insights: list[MatchInsight], page: int = 0) -> tuple[str, dict[str, Any]]:
    ordered = sort_insights(insights, mode)
    if not ordered:
        return ("No football matches were available in the current window.", main_menu_keyboard())
    total_pages = max(1, (len(ordered) + MAX_PAGE_SIZE - 1) // MAX_PAGE_SIZE)
    page = max(0, min(page, total_pages - 1))
    selected = ordered[page * MAX_PAGE_SIZE:(page + 1) * MAX_PAGE_SIZE]
    title_map = {"today": "Today board", "live": "Live shortlist", "safe": "Safe board", "value": "Value board", "acca": "Acca board", "corners": "Corners board", "cards": "Cards board", "goals": "Goals board", "btts": "BTTS board", "over25": "Over 2.5 board", "under25": "Under 2.5 board", "firsthalf": "First half board", "secondhalf": "Second half board"}
    lines = [f"{title_map.get(mode, 'Betting board')} ({page + 1}/{total_pages})"]
    for idx, insight in enumerate(selected):
        pick = pick_for_mode(insight, mode, page * MAX_PAGE_SIZE + idx)
        odds_text = f"{pick['odds']:.2f}" if isinstance(pick['odds'], (int, float)) and pick['odds'] > 0 else "N/A"
        lines.extend([
            "",
            f"Match: {insight.home_team} vs {insight.away_team}",
            f"Competition: {insight.competition}",
            f"Kickoff: {format_kickoff(insight.kickoff_utc)}",
            f"Market: {pick['market']}",
            f"Bet: {pick['bet']}",
            f"Confidence: {pick['confidence']}%",
            f"Stats: {pick['support']}",
            f"Odds: {odds_text} via {insight.bookmaker}",
            f"Risk: {pick['risk']}",
            f"Value: {pick['value']}",
            f"Stake: {pick['stake']}/10",
            f"Why: {pick['why']}",
        ])
    return ("\n".join(lines), prediction_keyboard(mode, page, total_pages))


def main_menu_keyboard() -> dict[str, Any]:
    return {"inline_keyboard": [[{"text": "Today", "callback_data": "menu:today:0"}, {"text": "Safe", "callback_data": "menu:safe:0"}, {"text": "Value", "callback_data": "menu:value:0"}], [{"text": "Goals", "callback_data": "menu:goals:0"}, {"text": "Corners", "callback_data": "menu:corners:0"}, {"text": "Cards", "callback_data": "menu:cards:0"}], [{"text": "Acca", "callback_data": "menu:acca:0"}, {"text": "Stats", "callback_data": "menu:stats:0"}]]}


def prediction_keyboard(mode: str, page: int, total_pages: int) -> dict[str, Any]:
    rows: list[list[dict[str, str]]] = []
    nav: list[dict[str, str]] = []
    if page > 0:
        nav.append({"text": "Prev", "callback_data": f"menu:{mode}:{page - 1}"})
    if page + 1 < total_pages:
        nav.append({"text": "Next", "callback_data": f"menu:{mode}:{page + 1}"})
    if nav:
        rows.append(nav)
    rows.append([{"text": "Menu", "callback_data": "menu:menu:0"}, {"text": "Stats", "callback_data": "menu:stats:0"}])
    return {"inline_keyboard": rows}


def help_text() -> str:
    lines = ["Football Intelligence Pro", "", "Commands:"]
    for item in COMMAND_DESCRIPTIONS:
        lines.append(f"/{item['command']} - {item['description']}")
    lines.append("")
    lines.append("Examples: /team arsenal | /match arsenal vs chelsea | /analyze arsenal vs chelsea")
    return "\n".join(lines)


def parse_command(text: str) -> tuple[str, str]:
    if not text.startswith("/"):
        return ("help", text)
    body = text[1:]
    command_part, _, args = body.partition(" ")
    command = command_part.split("@", 1)[0].strip().lower()
    return (ALIAS_MAP.get(command, command), args.strip())


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
    pick = pick_for_mode(insight, mode, 0)
    odds_text = f"{pick['odds']:.2f}" if isinstance(pick['odds'], (int, float)) and pick['odds'] > 0 else "N/A"
    return "\n".join([
        f"Match: {insight.home_team} vs {insight.away_team}",
        f"Competition: {insight.competition}",
        f"Kickoff: {format_kickoff(insight.kickoff_utc)}",
        f"Market: {pick['market']}",
        f"Bet: {pick['bet']}",
        f"Confidence: {pick['confidence']}%",
        f"Stats: {pick['support']}",
        f"Odds: {odds_text} via {insight.bookmaker}",
        f"Risk: {pick['risk']}",
        f"Value: {pick['value']}",
        f"Stake: {pick['stake']}/10",
        f"Why: {pick['why']}",
    ])


def stats_text(insights: list[MatchInsight]) -> str:
    if not insights:
        return "No match data loaded right now."
    avg_home = round(sum(item.support_stats.get("home_prob", 0.0) for item in insights) / len(insights) * 100)
    over25_values = [item.score_map.get("over25", 0.0) for item in insights if item.score_map.get("over25", 0.0) > 0]
    btts_values = [item.score_map.get("btts_yes", 0.0) for item in insights if item.score_map.get("btts_yes", 0.0) > 0]
    avg_over25 = round(sum(over25_values) / len(over25_values) * 100) if over25_values else 0
    avg_btts = round(sum(btts_values) / len(btts_values) * 100) if btts_values else 0
    team_stats_count = sum(1 for item in insights if "team-stats" in item.data_notes)
    form_count = sum(1 for item in insights if "recent-form" in item.data_notes)
    fixture_stats_count = sum(1 for item in insights if "fixture-stats" in item.data_notes)
    return "\n".join([
        "Loaded market stats",
        f"Matches in last {PAST_GRACE_HOURS}h to next {MATCH_WINDOW_HOURS}h: {len(insights)}",
        f"Average favorite win signal: {avg_home}%",
        f"Average over 2.5 signal: {avg_over25}%",
        f"Average BTTS signal: {avg_btts}%",
        f"Matches with team stats: {team_stats_count}",
        f"Matches with recent form: {form_count}",
        f"Matches with fixture stats: {fixture_stats_count}",
        "Model source: API-Football fixtures, team statistics, fixture statistics, recent form, and The Odds API pricing.",
    ])


def handle_text_command(command: str, args: str, insights: list[MatchInsight]) -> tuple[str, dict[str, Any] | None]:
    if command == "start":
        return ("Welcome to Football Intelligence Pro\n\nThis version uses fixture lists, team statistics, recent form, fixture statistics, and bookmaker pricing.", main_menu_keyboard())
    if command in {"help", "menu"}:
        return (help_text(), main_menu_keyboard())
    if command in {"today", "live", "safe", "value", "acca", "corners", "cards", "goals", "btts", "over25", "under25", "firsthalf", "secondhalf"}:
        return build_prediction_message(command, insights, 0)
    if command == "stats":
        return (stats_text(insights), {"inline_keyboard": [[{"text": "Menu", "callback_data": "menu:menu:0"}]]})
    if command == "team":
        if not args:
            return ("Usage: /team arsenal", None)
        matches = filter_by_team(insights, args)
        if not matches:
            return (f"No upcoming fixtures found for {args}.", None)
        lines = [f"Fixtures for {args}:"]
        for item in matches[:10]:
            lines.append(f"- {item.home_team} vs {item.away_team} | {format_kickoff(item.kickoff_utc)}")
        return ("\n".join(lines), None)
    if command in {"match", "analyze"}:
        if args and " vs " in args.lower():
            left, right = re.split(r"\s+vs\s+", args, maxsplit=1, flags=re.IGNORECASE)
            insight = find_match(insights, left, right)
            if not insight:
                return ("Match not found in the current window.", None)
            return (single_match_text(insight, "value" if command == "analyze" else "today"), None)
        if args:
            filtered = filter_by_team(insights, args)
            if filtered:
                return build_prediction_message("today", filtered, 0)
        return build_prediction_message("value" if command == "analyze" else "today", insights, 0)
    if command == "search":
        if not args:
            return ("Usage: /search arsenal", None)
        filtered = filter_by_team(insights, args)
        if not filtered:
            return (f"No upcoming fixtures found for {args}.", None)
        return build_prediction_message("today", filtered, 0)
    return ("Unknown command. Use /help.", main_menu_keyboard())


@app.get("/")
def home() -> tuple[str, int]:
    return ("Bot is alive.", 200)


@app.get("/healthz")
def healthz() -> tuple[Any, int]:
    return (jsonify({"ok": True, "time": datetime.now(UTC).isoformat()}), 200)


@app.post(f"/webhook/{settings.webhook_secret}")
def webhook() -> tuple[Any, int]:
    try:
        update = request.get_json(force=True, silent=True) or {}
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
            return (jsonify({"ok": True}), 200)
        message = update.get("message") or update.get("edited_message") or {}
        chat_id = int(message.get("chat", {}).get("id", 0) or 0)
        if not chat_id:
            return (jsonify({"ok": True}), 200)
        user_id = str(message.get("from", {}).get("id", "0"))
        text = str(message.get("text") or "").strip()
        command, args = parse_command(text)
        if rate_limiter.is_spamming(user_id):
            telegram.send_message(chat_id, "Slow down. Too many requests in a short time.")
            return (jsonify({"ok": True}), 200)
        if command not in {"start", "help", "menu"}:
            remaining = rate_limiter.cooldown_remaining(f"{user_id}:{command}")
            if remaining > 0:
                telegram.send_message(chat_id, f"Cooldown active. Please wait {remaining}s before /{command} again.")
                return (jsonify({"ok": True}), 200)
        insights = upcoming_match_insights()
        text_out, keyboard = handle_text_command(command, args, insights)
        telegram.send_message(chat_id, text_out, keyboard)
        return (jsonify({"ok": True}), 200)
    except Exception as exc:
        log.exception("Webhook error: %s", exc)
        return (jsonify({"ok": False, "error": "internal_error"}), 500)


telegram.set_commands()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)