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
        r = requests.post(
            TELEGRAM_API + "/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=20,
        )
        log.info("sendMessage status=%s body=%s", r.status_code, r.text)
        return r
    except Exception as e:
        log.exception("Telegram send failed: %s", e)
        return None


def get_odds(limit=8):
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
        home = m.get("home_team", "?")
        away = m.get("away_team", "?")
        commence = m.get("commence_time", "?")
        line = f"{home} vs {away} ({commence})"

        bookmakers = m.get("bookmakers", [])
        if bookmakers:
            try:
                outcomes = bookmakers[0]["markets"][0]["outcomes"]
                parts = []
                for o in outcomes:
                    parts.append(f"{o.get('name')}: {o.get('price')}")
                line += " | Odds: " + ", ".join(parts)
            except Exception:
                pass

        lines.append(line)

    return "
".join(lines) if lines else "No upcoming matches found."


def ask_gemini(prompt):
    if not GEMINI_API_KEY:
        return "Gemini API key missing in Render environment variables."

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }

    try:
        r = requests.post(GEMINI_URL, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        log.exception("Gemini API failed: %s", e)
        return "Analysis engine unavailable right now."


HELP_TEXT = """Available commands:
/start - start the bot
/help - show commands
/ping - simple test
/predict_today - today's best data-backed picks
/predict_now - matches starting soon
/value_bets - possible value bets
/corners team1 vs team2 - corners estimate
/cards team1 vs team2 - cards estimate
/form team - recent form summary
/h2h team1 vs team2 - head-to-head summary"""


DISCLAIMER = "

Warning: this is data-assisted analysis, not a guaranteed winning signal."


def cmd_predict_today():
    matches = get_odds(limit=8)
    matches_text = build_matches_text(matches)

    if matches_text == "No upcoming matches found.":
        return matches_text

    prompt = (
        "You are a football betting analyst. "
        "Use ONLY the odds data below. "
        "Choose the 3 strongest betting angles for today. "
        "For each one give: match, suggested market, confidence "
        "(Low/Medium/High), and one short reason. "
        "Do not promise certainty.

"
        f"Match data:
{matches_text}"
    )

    return ask_gemini(prompt) + DISCLAIMER


def cmd_predict_now():
    matches = get_odds(limit=10)
    matches_text = build_matches_text(matches)

    if matches_text == "No upcoming matches found.":
        return matches_text

    prompt = (
        "You are a football betting analyst. "
        "From the match list below, identify matches starting soon "
        "and give a short betting read for each one. "
        "If exact start timing is unclear, still provide the best short analysis. "
        "Keep it practical and not overconfident.

"
        f"Match data:
{matches_text}"
    )

    return ask_gemini(prompt) + DISCLAIMER


def cmd_value_bets():
    matches = get_odds(limit=10)
    matches_text = build_matches_text(matches)

    if matches_text == "No upcoming matches found.":
        return matches_text

    prompt = (
        "You are a sharp football bettor. "
        "Using the odds below, flag any possible value bets and explain "
        "briefly why the price may be interesting. "
        "Keep it realistic and avoid certainty.

"
        f"Match data:
{matches_text}"
    )

    return ask_gemini(prompt) + DISCLAIMER


def cmd_corners_cards(query_text, mode):
    prompt = (
        f"You are a football stats analyst. "
        f"Give a careful estimate for {mode} on this match: {query_text}. "
        f"Keep it short, practical, and not overconfident."
    )
    return ask_gemini(prompt) + DISCLAIMER


def cmd_form_h2h(query_text, mode):
    if mode == "form":
        prompt = (
            f"Summarize the likely recent form for this football team or match-up: "
            f"{query_text}. Keep it concise."
        )
    else:
        prompt = (
            f"Summarize the head-to-head tendency for this football match-up: "
            f"{query_text}. Keep it concise."
        )

    return ask_gemini(prompt) + DISCLAIMER


@app.route("/", methods=["GET"])
def health():
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
            reply = cmd_predict_today()
        elif lower.startswith("/predict_now"):
            reply = cmd_predict_now()
        elif lower.startswith("/value_bets"):
            reply = cmd_value_bets()
        elif lower.startswith("/corners"):
            query_text = text.replace("/corners", "").strip() or "general matches today"
            reply = cmd_corners_cards(query_text, "corners")
        elif lower.startswith("/cards"):
            query_text = text.replace("/cards", "").strip() or "general matches today"
            reply = cmd_corners_cards(query_text, "cards")
        elif lower.startswith("/form"):
            query_text = text.replace("/form", "").strip() or "general team"
            reply = cmd_form_h2h(query_text, "form")
        elif lower.startswith("/h2h"):
            query_text = text.replace("/h2h", "").strip() or "general matchup"
            reply = cmd_form_h2h(query_text, "h2h")
        else:
            reply = "Unknown command. Send /help."

        send_message(chat_id, reply)
        return jsonify(ok=True)

    except Exception as e:
        log.exception("Webhook error: %s", e)
        return jsonify(ok=False, error=str(e)), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)