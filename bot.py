import asyncio
import html
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
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/soccer/odds"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT = 20
CACHE_TTL_SECONDS = 300
AI_CACHE_TTL_SECONDS = 180
COOLDOWN_SECONDS = 8
SPAM_WINDOW_SECONDS = 20
SPAM_MAX_REQUESTS = 6
MATCH_WINDOW_HOURS = 24
MAX_PAGE_SIZE = 3


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
session.headers.update({"User-Agent": "smart-football-bot/2.0"})


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
    totals_over_35: float | None
    totals_under_35: float | None
    spread_home: float | None
    spread_away: float | None
    score_map: dict[str, float]
    support_stats: dict[str, float]
    raw_event: dict[str, Any]


COMMAND_DESCRIPTIONS: list[dict[str, str]] = [
    {"command": "start", "description": "Open the premium welcome screen"},
    {"command": "help", "description": "Show all commands and shortcuts"},
    {"command": "menu", "description": "Open the main inline menu"},
    {"command": "today", "description": "Today's best fixtures"},
    {"command": "live", "description": "Live-style watchlist from upcoming odds"},
    {"command": "now", "description": "Alias of /live"},
    {"command": "safe", "description": "Safer betting angles"},
    {"command": "value", "description": "Value-focused selections"},
    {"command": "vip", "description": "Premium shortlist"},
    {"command": "banker", "description": "Banker-style picks"},
    {"command": "acca", "description": "Accumulator shortlist"},
    {"command": "corners", "description": "Corners opportunities from odds context"},
    {"command": "cards", "description": "Cards opportunities from odds context"},
    {"command": "goals", "description": "Goals-focused analysis"},
    {"command": "btts", "description": "Both teams to score angles"},
    {"command": "over25", "description": "Over 2.5 goals focus"},
    {"command": "under25", "description": "Under 2.5 goals focus"},
    {"command": "firsthalf", "description": "First-half goal angles"},
    {"command": "secondhalf", "description": "Second-half goal angles"},
    {"command": "halftime", "description": "Alias of /firsthalf"},
    {"command": "predictions", "description": "Premium all-market predictions"},
    {"command": "top5", "description": "Top 5 shortlist"},
    {"command": "highconfidence", "description": "High-confidence view"},
    {"command": "matches", "description": "Alias of /today"},
    {"command": "stats", "description": "Show model inputs from odds"},
    {"command": "analyze", "description": "Analyze a specific match or today board"},
    {"command": "team", "description": "Filter fixtures by team name"},
    {"command": "match", "description": "Analyze TEAM1 vs TEAM2"},
    {"command": "search", "description": "Search fixtures by team text"},
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


def markdown_escape(value: str) -> str:
    escaped = html.escape(value, quote=False)
    return re.sub(r"([_\*\[\]()~`>#+\-=|{}.!])", r"\\\1", escaped)


def parse_iso_time(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except Exception:
        return None


def format_kickoff(value: str) -> str:
    dt = parse_iso_time(value)
    return dt.strftime("%d %b %Y %H:%M UTC") if dt else value


def retry_request(method: str, url: str, *, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None, json_body: dict[str, Any] | None = None, timeout: int = DEFAULT_TIMEOUT, attempts: int = 3) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = session.request(method, url, params=params, headers=headers, json=json_body, timeout=timeout)
            if response.status_code < 500:
                return response
            log.warning("Request to %s returned status %s on attempt %s", url, response.status_code, attempt)
        except requests.RequestException as exc:
            last_error = exc
            log.warning("Request to %s failed on attempt %s: %s", url, attempt, exc)
        time.sleep(0.25 * attempt)
    if last_error:
        raise last_error
    raise RuntimeError(f"Failed request to {url}")


class TelegramClient:
    def send_message(self, chat_id: int, text: str, reply_markup: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text[:4096],
            "parse_mode": "MarkdownV2",
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
            log.warning("Telegram setMyCommands failed: %s", exc)


telegram = TelegramClient()


def get_upcoming_odds() -> list[dict[str, Any]]:
    cached = cache.get("odds:upcoming")
    if cached is not None:
        return cached
    if not settings.odds_api_key:
        return []
    params = {
        "apiKey": settings.odds_api_key,
        "regions": "eu",
        "markets": "h2h,totals,spreads,btts",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    response = retry_request("GET", ODDS_API_URL, params=params, timeout=20)
    if response.status_code != 200:
        log.error("Odds API status=%s body=%s", response.status_code, response.text[:1000])
        return []
    payload = response.json()
    events = payload if isinstance(payload, list) else []
    cache.set("odds:upcoming", events, CACHE_TTL_SECONDS)
    return events


def select_bookmaker(event: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    bookmakers = event.get("bookmakers", []) or []
    if not bookmakers:
        return "Unavailable", {}
    bookmaker = bookmakers[0]
    markets = {market.get("key"): market for market in bookmaker.get("markets", [])}
    return str(bookmaker.get("title", "Unavailable")), markets


def find_price(market: dict[str, Any] | None, names: Iterable[str], point: float | None = None) -> float | None:
    if not market:
        return None
    names_lower = {name.lower() for name in names}
    for outcome in market.get("outcomes", []):
        outcome_name = str(outcome.get("name", "")).lower()
        if outcome_name in names_lower:
            if point is None or outcome.get("point") == point:
                price = outcome.get("price")
                if isinstance(price, (int, float)):
                    return float(price)
    return None


def implied_probability(price: float | None) -> float:
    if not price or price <= 1:
        return 0.0
    return 1 / price


def build_support_stats(markets: dict[str, Any], home_team: str, away_team: str) -> dict[str, float]:
    home_price = find_price(markets.get("h2h"), [home_team])
    draw_price = find_price(markets.get("h2h"), ["draw"])
    away_price = find_price(markets.get("h2h"), [away_team])
    over25 = find_price(markets.get("totals"), ["over"], 2.5)
    under25 = find_price(markets.get("totals"), ["under"], 2.5)
    over15 = find_price(markets.get("totals"), ["over"], 1.5)
    under15 = find_price(markets.get("totals"), ["under"], 1.5)
    over35 = find_price(markets.get("totals"), ["over"], 3.5)
    under35 = find_price(markets.get("totals"), ["under"], 3.5)
    btts_yes = find_price(markets.get("btts"), ["yes"])
    btts_no = find_price(markets.get("btts"), ["no"])
    spread_home = find_price(markets.get("spreads"), [home_team])
    spread_away = find_price(markets.get("spreads"), [away_team])
    home_prob = implied_probability(home_price)
    draw_prob = implied_probability(draw_price)
    away_prob = implied_probability(away_price)
    total_prob = home_prob + draw_prob + away_prob
    if total_prob > 0:
        home_prob /= total_prob
        draw_prob /= total_prob
        away_prob /= total_prob
    over25_prob = implied_probability(over25)
    under25_prob = implied_probability(under25)
    btts_yes_prob = implied_probability(btts_yes)
    btts_no_prob = implied_probability(btts_no)
    return {
        "home_price": home_price or 0.0,
        "draw_price": draw_price or 0.0,
        "away_price": away_price or 0.0,
        "over25": over25 or 0.0,
        "under25": under25 or 0.0,
        "over15": over15 or 0.0,
        "under15": under15 or 0.0,
        "over35": over35 or 0.0,
        "under35": under35 or 0.0,
        "btts_yes": btts_yes or 0.0,
        "btts_no": btts_no or 0.0,
        "spread_home": spread_home or 0.0,
        "spread_away": spread_away or 0.0,
        "home_prob": round(home_prob, 4),
        "draw_prob": round(draw_prob, 4),
        "away_prob": round(away_prob, 4),
        "over25_prob": round(over25_prob, 4),
        "under25_prob": round(under25_prob, 4),
        "btts_yes_prob": round(btts_yes_prob, 4),
        "btts_no_prob": round(btts_no_prob, 4),
    }


def build_score_map(stats: dict[str, float]) -> dict[str, float]:
    home_prob = stats["home_prob"]
    away_prob = stats["away_prob"]
    over25_prob = stats["over25_prob"]
    under25_prob = stats["under25_prob"]
    btts_yes_prob = stats["btts_yes_prob"]
    btts_no_prob = stats["btts_no_prob"]
    over15_prob = implied_probability(stats["over15"])
    over35_prob = implied_probability(stats["over35"])
    spread_presence = 0.08 if stats["spread_home"] or stats["spread_away"] else 0.0
    attack_balance = min(1.0, home_prob + away_prob)
    draw_bias = stats["draw_prob"]
    corners_signal = min(0.9, (over25_prob * 0.55) + (attack_balance * 0.35) + 0.1)
    cards_signal = min(0.88, (draw_bias * 0.50) + (attack_balance * 0.30) + 0.12)
    first_half_signal = min(0.9, (over15_prob * 0.7) + (over25_prob * 0.2) + 0.08)
    second_half_signal = min(0.92, (over25_prob * 0.45) + (btts_yes_prob * 0.20) + 0.22)
    return {
        "home_win": round(home_prob + spread_presence, 4),
        "away_win": round(away_prob + spread_presence, 4),
        "double_chance_home": round(min(0.96, home_prob + draw_bias), 4),
        "double_chance_away": round(min(0.96, away_prob + draw_bias), 4),
        "draw_no_bet_home": round(min(0.95, home_prob + (draw_bias * 0.25)), 4),
        "draw_no_bet_away": round(min(0.95, away_prob + (draw_bias * 0.25)), 4),
        "btts_yes": round(btts_yes_prob, 4),
        "btts_no": round(btts_no_prob, 4),
        "over25": round(over25_prob, 4),
        "under25": round(under25_prob, 4),
        "over15": round(over15_prob, 4),
        "over35": round(over35_prob, 4),
        "corners": round(corners_signal, 4),
        "cards": round(cards_signal, 4),
        "first_half": round(first_half_signal, 4),
        "second_half": round(second_half_signal, 4),
        "safe": round(max(home_prob, away_prob, over15_prob), 4),
        "value": round(max(stats["home_price"] * home_prob, stats["away_price"] * away_prob, stats["over25"] * over25_prob, stats["btts_yes"] * btts_yes_prob), 4),
    }


def upcoming_match_insights() -> list[MatchInsight]:
    now = datetime.now(UTC)
    cutoff = now + timedelta(hours=MATCH_WINDOW_HOURS)
    insights: list[MatchInsight] = []
    for event in get_upcoming_odds():
        kickoff = parse_iso_time(str(event.get("commence_time", "")))
        if not kickoff or kickoff < now or kickoff > cutoff:
            continue
        bookmaker, markets = select_bookmaker(event)
        home_team = str(event.get("home_team", "Unknown"))
        away_team = str(event.get("away_team", "Unknown"))
        support_stats = build_support_stats(markets, home_team, away_team)
        score_map = build_score_map(support_stats)
        insights.append(
            MatchInsight(
                match_id=str(event.get("id") or f"{home_team}-{away_team}-{event.get('commence_time', '')}"),
                competition=str(event.get("sport_title", "Football")),
                kickoff_utc=str(event.get("commence_time", "")),
                home_team=home_team,
                away_team=away_team,
                bookmaker=bookmaker,
                odds_home=support_stats["home_price"] or None,
                odds_draw=support_stats["draw_price"] or None,
                odds_away=support_stats["away_price"] or None,
                totals_over_25=support_stats["over25"] or None,
                totals_under_25=support_stats["under25"] or None,
                btts_yes=support_stats["btts_yes"] or None,
                btts_no=support_stats["btts_no"] or None,
                totals_over_15=support_stats["over15"] or None,
                totals_under_15=support_stats["under15"] or None,
                totals_over_35=support_stats["over35"] or None,
                totals_under_35=support_stats["under35"] or None,
                spread_home=support_stats["spread_home"] or None,
                spread_away=support_stats["spread_away"] or None,
                score_map=score_map,
                support_stats=support_stats,
                raw_event=event,
            )
        )
    insights.sort(key=lambda item: item.kickoff_utc)
    return insights


def confidence_from_probability(probability: float, floor: int = 48, ceiling: int = 86) -> int:
    bounded = max(0.0, min(1.0, probability))
    return max(floor, min(ceiling, round(floor + (bounded * (ceiling - floor)))))


def pick_for_mode(insight: MatchInsight, mode: str) -> dict[str, Any]:
    s = insight.score_map
    stats = insight.support_stats
    options = {
        "today": {
            "market": "Best Available",
            "bet": insight.home_team if s["home_win"] >= s["away_win"] else "Over 2.5 Goals",
            "confidence": confidence_from_probability(max(s["safe"], s["over25"])),
            "risk": "Low-Medium",
            "value": "B+",
            "odds": insight.odds_home if s["home_win"] >= s["away_win"] else insight.totals_over_25,
            "key": max(s["safe"], s["over25"]),
        },
        "live": {
            "market": "Live Opportunity",
            "bet": "Monitor for in-play entry on momentum side",
            "confidence": confidence_from_probability(max(s["value"] / 2.5, 0.48), 45, 75),
            "risk": "Medium",
            "value": "B",
            "odds": insight.odds_home if s["home_win"] >= s["away_win"] else insight.odds_away,
            "key": s["value"],
        },
        "safe": {
            "market": "Double Chance / Draw No Bet",
            "bet": f"{insight.home_team} or Draw" if s["double_chance_home"] >= s["double_chance_away"] else f"{insight.away_team} or Draw",
            "confidence": confidence_from_probability(max(s["double_chance_home"], s["double_chance_away"]), 56, 87),
            "risk": "Low",
            "value": "A-",
            "odds": insight.odds_home if s["double_chance_home"] >= s["double_chance_away"] else insight.odds_away,
            "key": max(s["double_chance_home"], s["double_chance_away"]),
        },
        "value": {
            "market": "Value Bet",
            "bet": "Over 2.5 Goals" if (insight.totals_over_25 and s["over25"] >= max(s["home_win"], s["away_win"])) else (insight.home_team if s["home_win"] >= s["away_win"] else insight.away_team),
            "confidence": confidence_from_probability(max(s["over25"], s["home_win"], s["away_win"]), 50, 82),
            "risk": "Medium",
            "value": "A",
            "odds": insight.totals_over_25 if (insight.totals_over_25 and s["over25"] >= max(s["home_win"], s["away_win"])) else (insight.odds_home if s["home_win"] >= s["away_win"] else insight.odds_away),
            "key": s["value"],
        },
        "acca": {
            "market": "Accumulator Leg",
            "bet": "Over 1.5 Goals" if insight.totals_over_15 else (insight.home_team if s["home_win"] >= s["away_win"] else insight.away_team),
            "confidence": confidence_from_probability(max(s["over15"], s["safe"]), 52, 84),
            "risk": "Medium",
            "value": "B+",
            "odds": insight.totals_over_15 if insight.totals_over_15 else (insight.odds_home if s["home_win"] >= s["away_win"] else insight.odds_away),
            "key": max(s["over15"], s["safe"]),
        },
        "corners": {
            "market": "Total Corners",
            "bet": "Over 8.5 Corners",
            "confidence": confidence_from_probability(s["corners"], 48, 78),
            "risk": "Medium",
            "value": "B",
            "odds": None,
            "key": s["corners"],
        },
        "cards": {
            "market": "Total Cards",
            "bet": "Over 3.5 Cards",
            "confidence": confidence_from_probability(s["cards"], 47, 77),
            "risk": "Medium-High",
            "value": "B",
            "odds": None,
            "key": s["cards"],
        },
        "goals": {
            "market": "Goals",
            "bet": "Over 2.5 Goals" if s["over25"] >= s["under25"] else "Under 2.5 Goals",
            "confidence": confidence_from_probability(max(s["over25"], s["under25"]), 49, 82),
            "risk": "Medium",
            "value": "B+",
            "odds": insight.totals_over_25 if s["over25"] >= s["under25"] else insight.totals_under_25,
            "key": max(s["over25"], s["under25"]),
        },
        "btts": {
            "market": "BTTS",
            "bet": "BTTS Yes" if s["btts_yes"] >= s["btts_no"] else "BTTS No",
            "confidence": confidence_from_probability(max(s["btts_yes"], s["btts_no"]), 48, 79),
            "risk": "Medium",
            "value": "B",
            "odds": insight.btts_yes if s["btts_yes"] >= s["btts_no"] else insight.btts_no,
            "key": max(s["btts_yes"], s["btts_no"]),
        },
        "over25": {
            "market": "Goals",
            "bet": "Over 2.5 Goals",
            "confidence": confidence_from_probability(s["over25"], 48, 82),
            "risk": "Medium",
            "value": "B+",
            "odds": insight.totals_over_25,
            "key": s["over25"],
        },
        "under25": {
            "market": "Goals",
            "bet": "Under 2.5 Goals",
            "confidence": confidence_from_probability(s["under25"], 46, 78),
            "risk": "Medium",
            "value": "B",
            "odds": insight.totals_under_25,
            "key": s["under25"],
        },
        "firsthalf": {
            "market": "First Half Goals",
            "bet": "Over 0.5 First Half Goals",
            "confidence": confidence_from_probability(s["first_half"], 46, 77),
            "risk": "Medium",
            "value": "B",
            "odds": insight.totals_over_15,
            "key": s["first_half"],
        },
        "secondhalf": {
            "market": "Second Half Goals",
            "bet": "Over 0.5 Second Half Goals",
            "confidence": confidence_from_probability(s["second_half"], 47, 79),
            "risk": "Medium",
            "value": "B",
            "odds": insight.totals_over_15,
            "key": s["second_half"],
        },
    }
    selected = options.get(mode, options["today"])
    selected["stake"] = max(1, min(10, round((selected["confidence"] - 40) / 6)))
    selected["support"] = support_summary(insight)
    return selected


def support_summary(insight: MatchInsight) -> str:
    stats = insight.support_stats
    parts = [
        f"1X2 implied: H {round(stats['home_prob'] * 100)}% | D {round(stats['draw_prob'] * 100)}% | A {round(stats['away_prob'] * 100)}%",
    ]
    if insight.totals_over_25 and insight.totals_under_25:
        parts.append(f"O2\.5 {insight.totals_over_25:.2f} vs U2\.5 {insight.totals_under_25:.2f}")
    if insight.btts_yes and insight.btts_no:
        parts.append(f"BTTS Yes {insight.btts_yes:.2f} vs No {insight.btts_no:.2f}")
    if insight.totals_over_15:
        parts.append(f"O1\.5 {insight.totals_over_15:.2f}")
    return " â€¢ ".join(parts)


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
                "competition": insight.competition,
                "kickoff": format_kickoff(insight.kickoff_utc),
                "market": pick["market"],
                "recommended_bet": pick["bet"],
                "confidence": pick["confidence"],
                "bookmaker_odds": pick["odds"],
                "supporting_stats": insight.support_stats,
            }
        )
    system_text = (
        "You are a football betting reasoning layer. Use only the supplied odds-derived data. "
        "Do not invent xG, possession, injuries, corners history, cards history, head-to-head, or live stats. "
        "Return valid JSON object where each key is match_id and each value is a concise explanation under 45 words. "
        "Explain uncertainty honestly and never claim certainty or fake percentages."
    )

    if settings.gemini_api_key:
        try:
            response = retry_request(
                "POST",
                GEMINI_URL,
                headers={"Content-Type": "application/json", "x-goog-api-key": settings.gemini_api_key},
                json_body={
                    "contents": [{"parts": [{"text": system_text + "\n\n" + json.dumps(prompt_items, ensure_ascii=False)}]}],
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
            log.warning("Gemini unavailable: %s", exc)

    if settings.openrouter_api_key:
        try:
            response = retry_request(
                "POST",
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://render.com",
                    "X-Title": "smart-football-bot",
                },
                json_body={
                    "model": "openrouter/auto",
                    "messages": [
                        {"role": "system", "content": system_text},
                        {"role": "user", "content": json.dumps(prompt_items, ensure_ascii=False)},
                    ],
                    "temperature": 0.2,
                    "response_format": {"type": "json_object"},
                },
                timeout=25,
            )
            if response.status_code == 200:
                data = response.json()
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    ai_cache.set(cache_key, parsed, AI_CACHE_TTL_SECONDS)
                    return parsed
        except Exception as exc:
            log.warning("OpenRouter unavailable: %s", exc)

    fallback = {
        insight.match_id: "Reasoning is based on bookmaker pricing, implied probability, and market alignment only."
        for insight, _ in picks
    }
    ai_cache.set(cache_key, fallback, AI_CACHE_TTL_SECONDS)
    return fallback


def sort_insights(insights: list[MatchInsight], mode: str) -> list[MatchInsight]:
    mode_key_map = {
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
    key = mode_key_map.get(mode, "safe")
    return sorted(insights, key=lambda item: item.score_map.get(key, 0), reverse=True)


def build_prediction_message(mode: str, insights: list[MatchInsight], page: int = 0) -> tuple[str, dict[str, Any]]:
    ordered = sort_insights(insights, mode)
    if not ordered:
        return (
            "âš ï¸ *No matches found*\n\nNo qualifying football matches were available from The Odds API in the next 24 hours\.",
            main_menu_keyboard(),
        )
    total_pages = max(1, (len(ordered) + MAX_PAGE_SIZE - 1) // MAX_PAGE_SIZE)
    page = max(0, min(page, total_pages - 1))
    start = page * MAX_PAGE_SIZE
    selected = ordered[start : start + MAX_PAGE_SIZE]
    picks = [(insight, pick_for_mode(insight, mode)) for insight in selected]
    ai_notes = ai_reasoning(mode, picks)
    headings = {
        "today": "ðŸ“… Today board",
        "live": "ðŸ”´ Live watchlist",
        "safe": "ðŸ›¡ï¸ Safe board",
        "value": "ðŸ’Ž Value board",
        "acca": "ðŸ§© Acca board",
        "corners": "ðŸš© Corners board",
        "cards": "ðŸŸ¨ Cards board",
        "goals": "ðŸ¥… Goals board",
        "btts": "ðŸ¤ BTTS board",
        "over25": "âš½ Over 2\.5 board",
        "under25": "ðŸ§Š Under 2\.5 board",
        "firsthalf": "â±ï¸ First half board",
        "secondhalf": "âŒ› Second half board",
    }
    lines = [f"{headings.get(mode, 'ðŸ“Š Betting board')} â€” page {page + 1}/{total_pages}"]
    for insight, pick in picks:
        odds_text = f"{pick['odds']:.2f}" if isinstance(pick["odds"], (int, float)) and pick["odds"] > 0 else "N/A"
        lines.extend(
            [
                "",
                f"âš½ *{markdown_escape(insight.home_team)} vs {markdown_escape(insight.away_team)}*",
                f"ðŸ† {markdown_escape(insight.competition)}",
                f"ðŸ•’ {markdown_escape(format_kickoff(insight.kickoff_utc))}",
                f"ðŸŽ¯ *Market:* {markdown_escape(pick['market'])}",
                f"âœ… *Recommended Bet:* {markdown_escape(pick['bet'])}",
                f"ðŸ”¥ *Confidence:* {pick['confidence']}%",
                f"ðŸ“Š *Supporting Statistics:* {markdown_escape(pick['support'])}",
                f"ðŸ’° *Bookmaker Odds:* {markdown_escape(odds_text)} via {markdown_escape(insight.bookmaker)}",
                f"âš ï¸ *Risk Level:* {markdown_escape(pick['risk'])}",
                f"â­ *Value Rating:* {markdown_escape(pick['value'])}",
                f"ðŸŽšï¸ *Suggested Stake:* {pick['stake']}/10",
                f"ðŸ§  *Why:* {markdown_escape(ai_notes.get(insight.match_id, 'Reasoning is based on the supplied market data only.'))}",
            ]
        )
    return "\n".join(lines), prediction_keyboard(mode, page, total_pages)


def main_menu_keyboard() -> dict[str, Any]:
    return {
        "inline_keyboard": [
            [
                {"text": "ðŸ“… Today", "callback_data": "menu:today:0"},
                {"text": "ðŸ›¡ï¸ Safe", "callback_data": "menu:safe:0"},
                {"text": "ðŸ’Ž Value", "callback_data": "menu:value:0"},
            ],
            [
                {"text": "ðŸ¥… Goals", "callback_data": "menu:goals:0"},
                {"text": "ðŸš© Corners", "callback_data": "menu:corners:0"},
                {"text": "ðŸŸ¨ Cards", "callback_data": "menu:cards:0"},
            ],
            [
                {"text": "ðŸ§© Acca", "callback_data": "menu:acca:0"},
                {"text": "ðŸ“Š Stats", "callback_data": "menu:stats:0"},
            ],
        ]
    }


def prediction_keyboard(mode: str, page: int, total_pages: int) -> dict[str, Any]:
    row: list[dict[str, str]] = []
    if page > 0:
        row.append({"text": "â¬…ï¸ Prev", "callback_data": f"menu:{mode}:{page - 1}"})
    if page + 1 < total_pages:
        row.append({"text": "Next âž¡ï¸", "callback_data": f"menu:{mode}:{page + 1}"})
    keyboard = [row] if row else []
    keyboard.append([
        {"text": "ðŸ  Menu", "callback_data": "menu:menu:0"},
        {"text": "ðŸ“Š Stats", "callback_data": "menu:stats:0"},
    ])
    return {"inline_keyboard": keyboard}


def help_text() -> str:
    rows = ["ðŸ‘‹ *Football Intelligence Pro*", "", "Premium betting assistance using bookmaker odds plus AI reasoning constrained to real fetched market data\.", ""]
    for item in COMMAND_DESCRIPTIONS:
        rows.append(f"â€¢ *\/{markdown_escape(item['command'])}* â€” {markdown_escape(item['description'])}")
    rows.append("")
    rows.append("Examples: /team arsenal, /match arsenal vs chelsea, /analyze arsenal vs chelsea")
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
    if not q:
        return []
    return [item for item in insights if q in item.home_team.lower() or q in item.away_team.lower()]


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
    ai_note = ai_reasoning(mode, [(insight, pick)]).get(insight.match_id, "Reasoning is based on supplied market data only.")
    odds_text = f"{pick['odds']:.2f}" if isinstance(pick["odds"], (int, float)) and pick["odds"] > 0 else "N/A"
    return (
        f"âš½ *{markdown_escape(insight.home_team)} vs {markdown_escape(insight.away_team)}*\n"
        f"ðŸ† {markdown_escape(insight.competition)}\n"
        f"ðŸ•’ {markdown_escape(format_kickoff(insight.kickoff_utc))}\n"
        f"ðŸŽ¯ *Market:* {markdown_escape(pick['market'])}\n"
        f"âœ… *Recommended Bet:* {markdown_escape(pick['bet'])}\n"
        f"ðŸ”¥ *Confidence:* {pick['confidence']}%\n"
        f"ðŸ“Š *Supporting Statistics:* {markdown_escape(pick['support'])}\n"
        f"ðŸ’° *Bookmaker Odds:* {markdown_escape(odds_text)} via {markdown_escape(insight.bookmaker)}\n"
        f"âš ï¸ *Risk Level:* {markdown_escape(pick['risk'])}\n"
        f"â­ *Value Rating:* {markdown_escape(pick['value'])}\n"
        f"ðŸŽšï¸ *Suggested Stake:* {pick['stake']}/10\n"
        f"ðŸ§  *Why:* {markdown_escape(ai_note)}"
    )


def stats_text(insights: list[MatchInsight]) -> str:
    if not insights:
        return "ðŸ“Š *Stats engine*\n\nNo current odds data available\."
    home_probs = [item.support_stats["home_prob"] for item in insights]
    over25_probs = [item.support_stats["over25_prob"] for item in insights if item.support_stats["over25_prob"] > 0]
    btts_probs = [item.support_stats["btts_yes_prob"] for item in insights if item.support_stats["btts_yes_prob"] > 0]
    avg_home = round(sum(home_probs) / len(home_probs) * 100)
    avg_over25 = round(sum(over25_probs) / len(over25_probs) * 100) if over25_probs else 0
    avg_btts = round(sum(btts_probs) / len(btts_probs) * 100) if btts_probs else 0
    return (
        "ðŸ“Š *Model inputs currently loaded*\n\n"
        f"â€¢ Matches in next {MATCH_WINDOW_HOURS}h: {len(insights)}\n"
        f"â€¢ Avg favorite implied win rate: {avg_home}%\n"
        f"â€¢ Avg over 2\.5 implied rate: {avg_over25}%\n"
        f"â€¢ Avg BTTS implied rate: {avg_btts}%\n"
        f"â€¢ Data sources: The Odds API for markets, Gemini/OpenRouter for constrained reasoning only\n"
        f"â€¢ No invented injuries, xG, possession, or historical stats are used"
    )


async def process_message(update: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    message = update.get("message") or update.get("edited_message") or {}
    text = str(message.get("text") or "").strip()
    user_id = str(message.get("from", {}).get("id", "0"))
    command, args = parse_command(text)

    if rate_limiter.is_spamming(user_id):
        return "â›” *Slow down*\n\nToo many requests in a short time\. Please wait a bit and try again\.", None

    if command not in {"start", "help", "menu"}:
        remaining = rate_limiter.cooldown_remaining(f"{user_id}:{command}")
        if remaining > 0:
            return f"â±ï¸ *Cooldown active*\n\nPlease wait {remaining}s before using /{markdown_escape(command)} again\.", None

    insights = upcoming_match_insights()

    if command == "start":
        return (
            "ðŸ‘‹ *Welcome to Football Intelligence Pro*\n\n"
            "A premium Telegram betting assistant using your existing Telegram, Odds API, Gemini, OpenRouter, and webhook setup only\."
        ), main_menu_keyboard()
    if command in {"help", "menu"}:
        return help_text(), main_menu_keyboard()
    if command in {"today", "live", "safe", "value", "acca", "corners", "cards", "goals", "btts", "over25", "under25", "firsthalf", "secondhalf"}:
        return build_prediction_message(command, insights, 0)
    if command == "stats":
        return stats_text(insights), {"inline_keyboard": [[{"text": "ðŸ  Menu", "callback_data": "menu:menu:0"}]]}
    if command == "team":
        if not args:
            return "ðŸ”Ž *Usage:* /team arsenal", None
        matches = filter_by_team(insights, args)
        if not matches:
            return f"âš ï¸ *No fixtures found* for {markdown_escape(args)} in the next {MATCH_WINDOW_HOURS}h\.", None
        lines = [f"ðŸ”Ž *Fixtures for {markdown_escape(args)}*", ""]
        for item in matches[:8]:
            lines.append(f"â€¢ {markdown_escape(item.home_team)} vs {markdown_escape(item.away_team)} â€” {markdown_escape(format_kickoff(item.kickoff_utc))}")
        return "\n".join(lines), None
    if command in {"match", "analyze"}:
        if args and " vs " in args.lower():
            left, right = re.split(r"\s+vs\s+", args, maxsplit=1, flags=re.IGNORECASE)
            insight = find_match(insights, left, right)
            if not insight:
                return "âš ï¸ *Match not found* in the current window\. Try /today first\.", None
            return single_match_text(insight, "today" if command == "match" else "value"), None
        if args:
            filtered = filter_by_team(insights, args)
            if filtered:
                return build_prediction_message("today", filtered, 0)
        return build_prediction_message("value" if command == "analyze" else "today", insights, 0)
    if command == "search":
        if not args:
            return "ðŸ”Ž *Usage:* /search arsenal", None
        filtered = filter_by_team(insights, args)
        if not filtered:
            return f"âš ï¸ *No fixtures found* for {markdown_escape(args)}\.", None
        return build_prediction_message("today", filtered, 0)
    return "â“ *Unknown command*\n\nUse /help to see the supported commands\.", main_menu_keyboard()


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
                if mode == "menu":
                    telegram.answer_callback(callback_id, "Opened menu")
                    telegram.send_message(chat_id, help_text(), main_menu_keyboard())
                elif mode == "stats":
                    telegram.answer_callback(callback_id, "Opened stats")
                    telegram.send_message(chat_id, stats_text(insights), {"inline_keyboard": [[{"text": "ðŸ  Menu", "callback_data": "menu:menu:0"}]]})
                else:
                    telegram.answer_callback(callback_id, f"Opened {mode}")
                    text_out, keyboard = build_prediction_message(mode, insights, page)
                    telegram.send_message(chat_id, text_out, keyboard)
            return jsonify({"ok": True}), 200

        message = update.get("message") or update.get("edited_message") or {}
        chat_id = int(message.get("chat", {}).get("id", 0) or 0)
        if not chat_id:
            return jsonify({"ok": True}), 200
        text_out, keyboard = asyncio.run(process_message(update))
        telegram.send_message(chat_id, text_out, keyboard)
        return jsonify({"ok": True}), 200
    except Exception as exc:
        log.exception("Webhook error: %s", exc)
        return jsonify({"ok": False, "error": "internal_error"}), 500


telegram.set_commands()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)