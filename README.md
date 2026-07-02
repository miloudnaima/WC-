# Telegram Football Prediction Bot

Stack: Telegram Bot API + The Odds API + Google Gemini + GitHub + Render

## 1. Get your free API keys
- Telegram bot token: from @BotFather on Telegram
- The Odds API key: https://the-odds-api.com (free tier: 500 requests/month)
- Gemini API key: https://aistudio.google.com/app/apikey (free tier)

## 2. Push these files to a new GitHub repo
Files: bot.py, requirements.txt, render.yaml, .env.example, README.md

## 3. Deploy on Render
1. Go to render.com -> New -> Web Service -> connect your GitHub repo
2. Render will detect render.yaml automatically
3. In the Render dashboard, add these Environment Variables (do NOT commit real keys to GitHub):
   - TELEGRAM_TOKEN
   - ODDS_API_KEY
   - GEMINI_API_KEY
   - WEBHOOK_SECRET (any random string you choose)
4. Deploy. Render gives you a public URL like: https://your-app.onrender.com

## 4. Connect Telegram to your Render app
Run this once in your browser (replace values):
https://api.telegram.org/bot<TELEGRAM_TOKEN>/setWebhook?url=https://your-app.onrender.com/webhook/<WEBHOOK_SECRET>

## 5. Test it
Open Telegram, message your bot: /help

## Commands
- /predict_today
- /predict_now
- /value_bets
- /corners team1 vs team2
- /cards team1 vs team2
- /form team
- /h2h team1 vs team2

## Notes
- Render's free tier sleeps after inactivity; use UptimeRobot (free) to ping your health endpoint (/) every 10 min to keep it awake.
- The Odds API free tier = 500 requests/month, so avoid spamming commands.
- Corners/cards commands currently use Gemini's general reasoning, not live stats.
  To get real corner/card numbers, add API-Football later and pass its data into the prompt the same way odds data is passed.
- No AI or bot can guarantee betting outcomes. This tool is for data-assisted decisions only.
