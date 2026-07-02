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


def send_message(chat_id, text):
    try:
        r = requests.post(
            TELEGRAM_API + "/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text
            },
            timeout=20,
        )
        log.info("sendMessage status=%s body=%s", r.status_code, r.text)
        return r
    except Exception as e:
        log.exception("Telegram send failed: %s", e)
        return None


HELP_TEXT = """Available commands:
/start
/help
/ping
/predict_today
/predict_now
/value_bets"""


@app.route("/", methods=["GET"])
def health():
    return "Bot is alive.", 200


@app.route("/webhook/" + WEBHOOK_SECRET, methods=["POST"])
def webhook():
    try:
        update = request.get_json(force=True, silent=True) or {}
        log.info("Incoming update: %s", json.dumps(update))

        message = update.get("message", {})
        chat = message.get("chat", {})
        chat_id = chat.get("id")
        text = (message.get("text") or "").strip()

        log.info("Parsed chat_id=%s text=%s", chat_id, text)

        if not chat_id:
            log.info("No chat_id found, returning ok")
            return jsonify(ok=True)

        lower = text.lower()

        if lower.startswith("/start"):
            reply = "Bot connected successfully. Send /help"
        elif lower.startswith("/help"):
            reply = HELP_TEXT
        elif lower.startswith("/ping"):
            reply = "pong"
        elif lower.startswith("/predict_today"):
            reply = "Bot is working. Prediction module comes next."
        elif lower.startswith("/predict_now"):
            reply = "Bot is working. Live prediction module comes next."
        elif lower.startswith("/value_bets"):
            reply = "Bot is working. Value bets module comes next."
        else:
            reply = "I received: " + (text or "[empty message]")

        send_message(chat_id, reply)
        return jsonify(ok=True)

    except Exception as e:
        log.exception("Webhook error: %s", e)
        return jsonify(ok=False, error=str(e)), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)