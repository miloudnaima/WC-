import os
import json
import logging
import requests
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ultimate-football-bot")
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
        requests.post(
            TELEGRAM_API + "/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=20
        )
    except Exception as e:
        log.exception("Telegram send failed: %s", e)


def get_odds(limit=12):
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


def build_match_lines(matches):
    lines = []

    for idx, m in enumerate(matches, 1):
        home = str(m.get("home_team", "?"))
        away = str(m.get("away_team", "?"))
        commence = str(m.get("commence_time", "?"))

        line = str(idx) + ". " + home + " vs " + away + " | kickoff: " + commence

        bookmakers = m.get("bookmakers", [])
        if bookmakers:
            try:
                markets = bookmakers[0].get("markets", [])

                if len(markets) > 0:
                    outcomes = markets[0].get("outcomes", [])
                    parts = []
                    for o in outcomes:
                        parts.append(str(o.get("name")) + "=" + str(o.get("price")))
                    if parts:
                        line = line + " | h2h: " + ", ".join(parts)

                if len(markets) > 1:
                    totals = markets[1].get("outcomes", [])
                    tparts = []
                    for t in totals:
                        name = str(t.get("name"))
                        point = str(t.get("point"))
                        price = str(t.get("price"))
                        tparts.append(name + " " + point + "=" + price)
                    if tparts:
                        line = line + " | totals: " + ", ".join(tparts)

            except Exception:
                pass

        lines.append(line)

    return chr(10).join(lines)


def ask_gemini(prompt):
    if not GEMINI_API_KEY:
        return None

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

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }

    try:
        r = requests.post(GEMINI_URL, headers=headers, json=payload, timeout=40)

        if r.status_code != 200:
            log.error("Gemini status=%s body=%s", r.status_code, r.text)
            return None

        data = r.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return None

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts:
            return None

        text = parts[0].get("text", "")
        if not text:
            return None

        return text

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
        "X-Title": "ultimate-football-bot"
    }

    payload = {
        "model": "openrouter/auto",
        "messages": [
            {
                "role": "system",
                "content": "You are a premium football betting analyst. Be realistic, sharp, and detailed. Never promise certainty."
            },
            {
                "role": "user",
                "content": prompt
            }
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

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if not content:
            return None

        return content

    except Exception as e:
        log.exception("OpenRouter failed: %s", e)
        return None


def ask_ai(prompt):
    gemini_result = ask_gemini(prompt)
    if gemini_result:
        return gemini_result

    openrouter_result = ask_openrouter(prompt)
    if openrouter_result:
        return openrouter_result

    return "Both Gemini and OpenRouter failed. Check your API keys or quotas."


def no_data_text():
    return "No matches found right now or odds API unavailable."


def luxury_prompt(mode_name, match_block):
    intro = "You are Football Intelligence Bot, a premium football analyst. Write like an elite paid research assistant. Be realistic, polished, detailed, and structured. Never promise certainty."

    if mode_name == "today":
        body = "Using only the football odds data below plus careful reasoning, produce a luxury today report. For each match include: match title, likely score, win draw probabilities in percentages, tactical strengths and weaknesses, injury or suspension note as assumed if unknown, and betting insight. Use elegant formatting with separators."
    elif mode_name == "banker":
        body = "Using the football odds data below, choose the safest single bet of the day. Give match, exact market, confidence, detailed reasoning, and one backup banker."
    elif mode_name == "acca":
        body = "Using the football odds data below, build one smart 3-leg accumulator with safer logic. Give each leg, reason, and one caution note."
    elif mode_name == "goals":
        body = "Using the football odds data below, analyze over under 2.5 goals for the listed matches. For each one give the best lean, confidence, likely score and short logic."
    elif mode_name == "predict":
        body = "Using the football odds data below, produce a deep premium pre-match analysis for the listed fixtures. Include likely score, probabilities, tactical angle, risk note, and betting insight."
    elif mode_name == "predict_now":
        body = "Using the football odds data below, create a live style momentum report for the listed fixtures. Mention likely pressure direction, next goal lean, and smart market angle, while clearly staying cautious if live information is limited."
    elif mode_name == "standings":
        body = "Give a concise major-league standings style overview, but clearly mention that this is a general AI summary because no standings API is connected."
    elif mode_name == "usdt":
        body = "Explain briefly how to check USDT DZD manually and mention that a direct live parser is not connected in this bot."
    else:
        body = "Analyze the football odds data below in a premium and realistic way."

    return intro + chr(10) + chr(10) + body + chr(10) + chr(10) + "MATCH DATA:" + chr(10) + match_block


def today_text():
    matches = get_odds(10)
    if not matches:
        return no_data_text()
    prompt = luxury_prompt("today", build_match_lines(matches))
    return ask_ai(prompt)


def banker_text():
    matches = get_odds(10)
    if not matches:
        return no_data_text()
    prompt = luxury_prompt("banker", build_match_lines(matches))
    return ask_ai(prompt)


def acca_text():
    matches = get_odds(10)
    if not matches:
        return no_data_text()
    prompt = luxury_prompt("acca", build_match_lines(matches))
    return ask_ai(prompt)


def goals_text():
    matches = get_odds(10)
    if not matches:
        return no_data_text()
    prompt = luxury_prompt("goals", build_match_lines(matches))
    return ask_ai(prompt)


def predict_text():
    matches = get_odds(10)
    if not matches:
        return no_data_text()
    prompt = luxury_prompt("predict", build_match_lines(matches))
    return ask_ai(prompt)


def predict_now_text():
    matches = get_odds(10)
    if not matches:
        return no_data_text()
    prompt = luxury_prompt("predict_now", build_match_lines(matches))
    return ask_ai(prompt)


def standings_text():
    prompt = luxury_prompt("standings", "No standings API connected.")
    return ask_ai(prompt)


def usdt_text():
    prompt = luxury_prompt("usdt", "No live USDT parser connected.")
    return ask_ai(prompt)


HELP_TEXT = "Football Intelligence Bot | /banker best single bet | /acca 3-leg parlay | /goals over-under 2.5 analysis | /predict deep pre-match analysis | /predict_now live style update | /today full fixtures | /standings major league tables | /usdt USDT DZD guidance"

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
        elif lower.startswith("/banker"):
            reply = banker_text()
        elif lower.startswith("/acca"):
            reply = acca_text()
        elif lower.startswith("/goals"):
            reply = goals_text()
        elif lower.startswith("/predict_now"):
            reply = predict_now_text()
        elif lower.startswith("/predict"):
            reply = predict_text()
        elif lower.startswith("/today"):
            reply = today_text()
        elif lower.startswith("/standings"):
            reply = standings_text()
        elif lower.startswith("/usdt"):
            reply = usdt_text()
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