# Telegram NLP Summarizer

A Python app that summarizes daily messages from specified Telegram channels and groups with an NLP focus.

It filters out non‑NLP messages using an OpenAI‑compatible API and posts the summarized results to your own Telegram channel. The app also attempts to detect reposts and duplicates.

## Setup
1. Copy `.env.example` to `.env` and fill in the required values.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the summarizer:
   ```bash
   python summarizer.py
   ```

## Environment Variables
- `TELEGRAM_API_ID` and `TELEGRAM_API_HASH`: Telegram API credentials.
- `TELEGRAM_BOT_TOKEN`: Bot token used to send messages.
- `TARGET_CHANNEL`: Channel username or ID where summaries will be posted.
- `OPENAI_API_KEY`: API key for the OpenAI‑compatible service.
- `SOURCE_CHANNELS`: Comma‑separated list of channels/groups to summarize.

## Features
- Fetches messages from configured channels for the last 24 hours.
- Filters messages to keep only those related to NLP.
- Detects duplicates through link matching and LLM checks.
- Sends a daily summary to your target channel.
