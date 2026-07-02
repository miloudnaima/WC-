import os
import json
import logging
import requests
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("predict-bot")
app = Flask(__name__)

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "changeme")

TELEGRAM_API = "https://api.telegram.org/bot" + TELEGRAM_TOKEN
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/soccer/odds"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=" + GEMINI_API_KEY


def send_message(chat_id, text):
    try:
        r = requests.post(TELEGRAM_API + "/sendMessage", json={"chat_id": chat_id, "text": text}, timeout=20)
        log.info("sendMessage status=%s", r.status_code)
    except Exception as e:
        log.exception("Telegram send failed: %s", e)


def get_odds(limit=6):
    if not ODDS_API_KEY:
        return []
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu",
        "markets": "h2h,totals",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    try:
        r = requests.get(ODDS_API_URL, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        return data[:limit] if isinstance(data, list) else []
    except Exception as e:
        log.exception("Odds API failed: %s", e)
        return []


def build_matches_text(matches):
    lines = []
    for m in matches:
        home = str(m.get("home_team", "?"))
        away = str(m.get("away_team", "?"))
        commence = str(m.get("commence_time", "?"))
        line = home + " vs " + away + " (" + commence + ")"
        bookmakers = m.get("bookmakers", [])
        if bookmakers:
            try:
                outcomes = bookmakers[0]["markets"][0]["outcomes"]
                parts = []
                for o in outcomes:
                    parts.append(str(o.get("name")) + ": " + str(o.get("price")))
                line = line + " | Odds: " + ", ".join(parts)
            except Exception:
                pass
        lines.append(line)
    return chr(10).join(lines) if lines else "No upcoming matches found."


def ask_gemini(prompt):
    if not GEMINI_API_KEY:
        return "Gemini API key missing in Render environment variables."
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        r = requests.post(GEMINI_URL, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        log.exception("Gemini API failed: %s", e)
        return "Analysis engine unavailable right now."


def predict_today_text():
    matches = get_odds(6)
    matches_text = build_matches_text(matches)
    if matches_text == "No upcoming matches found.":
        return matches_text
    prompt = "You are a football betting analyst. Use only the odds data below. Pick the 3 best betting angles for today. For each give match, market, confidence and one short reason. Do not promise certainty." + chr(10) + chr(10) + matches_text
    result = ask_gemini(prompt)
    return result + chr(10) + chr(10) + "Warning: this is analysis, not guaranteed winning advice."


def predict_now_text():
    matches = get_odds(8)
    matches_text = build_matches_text(matches)
    if matches_text == "No upcoming matches found.":
        return matches_text
    prompt = "You are a football betting analyst. From the odds data below, give short picks for matches starting soon. Keep it practical and careful." + chr(10) + chr(10) + matches_text
    result = ask_gemini(prompt)
    return result + chr(10) + chr(10) + "Warning: this is analysis, not guaranteed winning advice."


def value_bets_text():
    matches = get_odds(8)
    matches_text = build_matches_text(matches)
    if matches_text == "No upcoming matches found.":
        return matches_text
    prompt = "You are a sharp football bettor. Using the odds below, flag possible value bets and explain briefly why the odds may be interesting. Avoid certainty." + chr(10) + chr(10) + matches_text
    result = ask_gemini(prompt)
    return result + chr(10) + chr(10) + "Warning: this is analysis, not guaranteed winning advice."


def simple_ai_text(user_text, mode_name):
    prompt = "You are a football analyst. Give a short and practical answer for " + mode_name + " about: " + user_text
    result = ask_gemini(prompt)
    return result + chr(10) + chr(10) + "Warning: this is analysis, not guaranteed winning advice."


HELP_TEXT = "Commands: /start /help /ping /predict_today /predict_now /value_bets /corners team1 vs team2 /cards team1 vs team2 /form team /h2h team1 vs team2"


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
        if lower.startswith("/start"):
            reply = "Bot connected. Send /help"
        elif lower.startswith("/help"):
            reply = HELP_TEXT
        elif lower.startswith("/ping"):
            reply = "pong"
        elif lower.startswith("/predict_today"):
            reply = predict_today_text()
        elif lower.startswith("/predict_now"):
            reply = predict_now_text()
        elif lower.startswith("/value_bets"):
            reply = value_bets_text()
        elif lower.startswith("/corners"):
            query_text = text.replace("/corners", "").strip() or "general matches today"
            reply = simple_ai_text(query_text, "corners")
        elif lower.startswith("/cards"):
            query_text = text.replace("/cards", "").strip() or "general matches today"
            reply = simple_ai_text(query_text, "cards")
        elif lower.startswith("/form"):
            query_text = text.replace("/form", "").strip() or "general team"
            reply = simple_ai_text(query_text, "recent form")
        elif lower.startswith("/h2h"):
            query_text = text.replace("/h2h", "").strip() or "general matchup"
            reply = simple_ai_text(query_text, "head to head")
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