import os
import json
import logging
from datetime import datetime, timezone, timedelta
import requests
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("smart-football-bot")
app = Flask(__name__)

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "changeme")
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

TELEGRAM_API = "https://api.telegram.org/bot" + TELEGRAM_TOKEN
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/soccer/odds"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def send_message(chat_id, text):
    try:
        requests.post(TELEGRAM_API + "/sendMessage", json={"chat_id": chat_id, "text": text}, timeout=20)
    except Exception as e:
        log.exception("Telegram send failed: %s", e)


def parse_iso_time(value):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def get_odds(limit=30):
    if not ODDS_API_KEY:
        return []
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu",
        "markets": "h2h,totals",
        "oddsFormat": "decimal",
        "dateFormat": "iso"
    }
    try:
        r = requests.get(ODDS_API_URL, params=params, timeout=25)
        if r.status_code != 200:
            log.error("Odds API status=%s body=%s", r.status_code, r.text)
            return []
        data = r.json()
        return data[:limit] if isinstance(data, list) else []
    except Exception as e:
        log.exception("Odds API failed: %s", e)
        return []


def filter_next_12h(matches):
    now = datetime.now(timezone.utc)
    max_time = now + timedelta(hours=12)
    out = []
    for m in matches:
        t = parse_iso_time(str(m.get("commence_time", "")))
        if t and now <= t <= max_time:
            out.append(m)
    out.sort(key=lambda x: x.get("commence_time", ""))
    return out


def extract_market_text(m):
    parts = []
    bookmakers = m.get("bookmakers", [])
    if not bookmakers:
        return "No bookmaker markets available"
    try:
        markets = bookmakers[0].get("markets", [])
        for market in markets:
            key = market.get("key", "")
            outcomes = market.get("outcomes", [])
            if key == "h2h":
                row = []
                for o in outcomes:
                    row.append(str(o.get("name")) + "=" + str(o.get("price")))
                if row:
                    parts.append("h2h: " + ", ".join(row))
            elif key == "totals":
                row = []
                for o in outcomes:
                    row.append(str(o.get("name")) + " " + str(o.get("point")) + "=" + str(o.get("price")))
                if row:
                    parts.append("totals: " + ", ".join(row))
    except Exception:
        pass
    return " | ".join(parts) if parts else "No parsed markets"


def build_match_lines(matches):
    lines = []
    for i, m in enumerate(matches, 1):
        home = str(m.get("home_team", "?"))
        away = str(m.get("away_team", "?"))
        kickoff = str(m.get("commence_time", "?"))
        markets = extract_market_text(m)
        lines.append(str(i) + ". " + home + " vs " + away + " | kickoff: " + kickoff + " | " + markets)
    return "
".join(lines)


def ask_gemini(prompt):
    if not GEMINI_API_KEY:
        return None
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
    try:
        r = requests.post(GEMINI_URL, headers=headers, json=payload, timeout=40)
        if r.status_code != 200:
            log.error("Gemini status=%s body=%s", r.status_code, r.text)
            return None
        data = r.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            return None
        return parts[0].get("text", "") or None
    except Exception as e:
        log.exception("Gemini failed: %s", e)
        return None


def ask_openrouter(prompt):
    if not OPENROUTER_API_KEY:
        return None
    headers = {
        "Authorization": "Bearer " + OPENROUTER_API_KEY,
        "Content-Type": "application/json",
        "HTTP-Referer": "https://render.com",
        "X-Title": "smart-football-bot"
    }
    payload = {
        "model": "openrouter/auto",
        "messages": [
            {"role": "system", "content": "You are an elite football analyst. Be concise, sharp, realistic, and structured. Never promise certainty."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=40)
        if r.status_code != 200:
            log.error("OpenRouter status=%s body=%s", r.status_code, r.text)
            return None
        data = r.json()
        choices = data.get("choices", [])
        if not choices:
            return None
        return choices[0].get("message", {}).get("content", "") or None
    except Exception as e:
        log.exception("OpenRouter failed: %s", e)
        return None


def ask_ai(prompt):
    result = ask_gemini(prompt)
    if result:
        return result
    result = ask_openrouter(prompt)
    if result:
        return result
    return "Both Gemini and OpenRouter failed. Check keys or quotas."


def no_data_text():
    return "No good soccer matches found in the next 12 hours."


def now_prompt(match_block):
    return (
        "You are Football Intelligence Bot. Analyze only the matches below which start in the next 12 hours. "
        "Return only the best opportunities. Be premium, smart, selective, and realistic. "
        "For each chosen match give exactly this format: Match, Kickoff, Best market, Best odds angle, Confidence, Why it matters, Risk. "
        "Only include matches you truly like. Maximum 5 matches.

MATCH DATA:
" + match_block
    )


def corners_prompt(match_block):
    return (
        "You are Football Intelligence Bot focused on corners. Analyze only the matches below which start in the next 12 hours. "
        "Use the odds data carefully and infer likely pressure, wing play, attacking intent, and game state risk. "
        "For each chosen match give: Match, Kickoff, Best corners angle, Suggested line, Confidence, Why corners can land, Risk. "
        "Do not pretend to have exact live corner stats if not provided. Maximum 5 matches.

MATCH DATA:
" + match_block
    )


def perfect_prompt(match_block):
    return (
        "You are Football Intelligence Bot. From the matches below starting in the next 12 hours, select the strongest all-around picks. "
        "Be extremely selective. Output a luxury shortlist with: Match, Kickoff, Market, Odds angle, Confidence, Main reason, Main risk. "
        "Maximum 4 matches.

MATCH DATA:
" + match_block
    )


def predict_now_text():
    matches = filter_next_12h(get_odds(30))
    if not matches:
        return no_data_text()
    return ask_ai(now_prompt(build_match_lines(matches)))


def corners_text():
    matches = filter_next_12h(get_odds(30))
    if not matches:
        return no_data_text()
    return ask_ai(corners_prompt(build_match_lines(matches)))


def perfect_text():
    matches = filter_next_12h(get_odds(30))
    if not matches:
        return no_data_text()
    return ask_ai(perfect_prompt(build_match_lines(matches)))


def today_text():
    matches = filter_next_12h(get_odds(30))
    if not matches:
        return no_data_text()
    lines = []
    for i, m in enumerate(matches[:12], 1):
        lines.append(str(i) + ". " + str(m.get("home_team", "?")) + " vs " + str(m.get("away_team", "?")) + " | " + str(m.get("commence_time", "?")))
    return "Matches in next 12h:
" + "
".join(lines)


HELP_TEXT = "Football Intelligence Bot | /predict_now smart picks in next 12h | /corners corners angles in next 12h | /perfect best matches and odds in next 12h | /today next 12h fixtures"


@app.route("/", methods=["GET"])
def home():
    return "Bot is alive.", 200


@app.route("/webhook/" + WEBHOOK_SECRET, methods=["POST"])
def webhook():
    try:
        update = request.get_json(force=True, silent=True) or {}
        log.info("Incoming update: %s", json.dumps(update))
        message = update.get("message", {})
        chat_id = message.get("chat", {}).get("id")
        text = (message.get("text") or "").strip()
        if not chat_id:
            return jsonify(ok=True)
        lower = text.lower()
        if lower.startswith("/start") or lower.startswith("/help"):
            reply = HELP_TEXT
        elif lower.startswith("/predict_now"):
            reply = predict_now_text()
        elif lower.startswith("/corners"):
            reply = corners_text()
        elif lower.startswith("/perfect"):
            reply = perfect_text()
        elif lower.startswith("/today"):
            reply = today_text()
        else:
            reply = "Unknown command. Send /help"
        send_message(chat_id, reply)
        return jsonify(ok=True)
    except Exception as e:
        log.exception("Webhook error: %s", e)
        return jsonify(ok=False, error=str(e)), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)