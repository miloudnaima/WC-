"""
Telegram Football Prediction Bot
Stack: Telegram Bot API + The Odds API + Google Gemini API
Hosting: Render (Web Service) + GitHub (source control / auto-deploy)
"""

import os
import logging
import requests
from flask import Flask, request, jsonify

NL = chr(10)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("predict-bot")

app = Flask(__name__)

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
ODDS_API_KEY = os.environ["ODDS_API_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "changeme")

TELEGRAM_API = "https://api.telegram.org/bot" + TELEGRAM_TOKEN
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/soccer/odds"
GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
              "gemini-2.0-flash:generateContent?key=" + GEMINI_API_KEY)


def send_message(chat_id, text):
    try:
        requests.post(
            TELEGRAM_API + "/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=15,
        )
    except Exception as e:
        log.error("Telegram send failed: " + str(e))


def get_odds(regions="eu", markets="h2h,totals", limit=15):
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    try:
        r = requests.get(ODDS_API_URL, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        return data[:limit] if isinstance(data, list) else []
    except Exception as e:
        log.error("Odds API failed: " + str(e))
        return []


def ask_gemini(prompt):
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        r = requests.post(GEMINI_URL, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        log.error("Gemini API failed: " + str(e))
        return "Sorry, the analysis engine is temporarily unavailable. Try again shortly."


def build_matches_text(matches):
    lines = []
    for m in matches:
        home = m.get("home_team")
        away = m.get("away_team")
        commence = m.get("commence_time")
        line = str(home) + " vs " + str(away) + " (" + str(commence) + ")"
        bookmakers = m.get("bookmakers", [])
        if bookmakers:
            outcomes = bookmakers[0]["markets"][0]["outcomes"]
            parts = []
            for o in outcomes:
                parts.append(str(o.get("name")) + ": " + str(o.get("price")))
            odds_str = ", ".join(parts)
            line = line + " | Odds: " + odds_str
        lines.append(line)
    if lines:
        return NL.join(lines)
    return "No upcoming matches found."


DISCLAIMER = (NL + NL +
    "WARNING: This is data-based analysis, not a guarantee. Bet responsibly, "
    "never stake more than you can afford to lose.")


def cmd_predict_today():
    matches = get_odds(limit=20)
    matches_text = build_matches_text(matches)
    prompt = (
        "You are a professional football betting analyst. Based ONLY on the "
        "odds data below, identify the 5 matches today with the clearest "
        "statistical edge. For each, give: match, suggested bet type "
        "(1X2/over-under), confidence (Low/Medium/High), and a 1-line reason. "
        "Be realistic, never claim certainty." + NL + NL +
        "Match data:" + NL + matches_text
    )
    return ask_gemini(prompt) + DISCLAIMER


def cmd_predict_now():
    matches = get_odds(limit=30)
    matches_text = build_matches_text(matches)
    prompt = (
        "From the match list below, select only matches starting within the "
        "next 2 hours (check commence_time against current UTC time). Give a "
        "quick betting read for each: suggested market, confidence level, "
        "and reasoning based on the odds shown. If none start soon, say so." + NL + NL
        + matches_text
    )
    return ask_gemini(prompt) + DISCLAIMER


def cmd_value_bets():
    matches = get_odds(limit=25)
    matches_text = build_matches_text(matches)
    prompt = (
        "Act as a sharp bettor looking for value. Using the odds below, "
        "estimate implied probability for each outcome, then flag any match "
        "where you believe the market odds look mispriced or offer value "
        "compared to typical team strength assumptions. Explain briefly." + NL + NL
        + matches_text
    )
    return ask_gemini(prompt) + DISCLAIMER


def cmd_corners_cards(query_text):
    prompt = (
        "You are a football stats analyst. Based on general team tendencies "
        "and recent known patterns, give an estimate for corners and cards "
        "markets for: " + query_text + ". Be clear this is a general estimate "
        "since live corner/card statistics require a dedicated stats API "
        "(e.g. API-Football) which is not yet connected."
    )
    return ask_gemini(prompt) + DISCLAIMER


def cmd_form_h2h(query_text, mode="form"):
    if mode == "form":
        prompt = "Summarize the likely recent form and current trend for: " + query_text + ". Keep it factual and concise."
    else:
        prompt = "Summarize the historical head-to-head tendency between: " + query_text + ". Keep it factual and concise."
    return ask_gemini(prompt) + DISCLAIMER


HELP_LINES = [
    "Available commands:",
    "/predict_today - Best data-backed picks for today",
    "/predict_now - Matches starting in the next 2 hours",
    "/value_bets - Matches where odds may be mispriced",
    "/corners team1 vs team2 - Corner market estimate",
    "/cards team1 vs team2 - Card market estimate",
    "/form team - Recent form summary",
    "/h2h team1 vs team2 - Head-to-head history",
    "/help - Show this menu",
]
HELP_TEXT = NL.join(HELP_LINES)


@app.route("/webhook/" + WEBHOOK_SECRET, methods=["POST"])
def webhook():
    update = request.get_json(force=True, silent=True) or {}
    message = update.get("message", {})
    chat_id = message.get("chat", {}).get("id")
    text = (message.get("text") or "").strip()

    if not chat_id or not text:
        return jsonify(ok=True)

    lower = text.lower()

    if lower.startswith("/start") or lower.startswith("/help"):
        reply = HELP_TEXT
    elif lower.startswith("/predict_today"):
        reply = cmd_predict_today()
    elif lower.startswith("/predict_now"):
        reply = cmd_predict_now()
    elif lower.startswith("/value_bets"):
        reply = cmd_value_bets()
    elif lower.startswith("/corners"):
        reply = cmd_corners_cards(text.replace("/corners", "").strip() or "general matches today")
    elif lower.startswith("/cards"):
        reply = cmd_corners_cards(text.replace("/cards", "").strip() or "general matches today")
    elif lower.startswith("/form"):
        reply = cmd_form_h2h(text.replace("/form", "").strip(), mode="form")
    elif lower.startswith("/h2h"):
        reply = cmd_form_h2h(text.replace("/h2h", "").strip(), mode="h2h")
    else:
        reply = "Unknown command. Send /help to see what I can do."

    send_message(chat_id, reply)
    return jsonify(ok=True)


@app.route("/", methods=["GET"])
def health():
    return "Bot is alive.", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)