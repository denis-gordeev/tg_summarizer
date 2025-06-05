import os
import asyncio
from datetime import datetime, timedelta
from difflib import SequenceMatcher
import re

from dotenv import load_dotenv
from telethon import TelegramClient, events
from openai import OpenAI

load_dotenv()

API_ID = int(os.getenv('TELEGRAM_API_ID'))
API_HASH = os.getenv('TELEGRAM_API_HASH')
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TARGET_CHANNEL = os.getenv('TARGET_CHANNEL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SOURCE_CHANNELS = [c.strip() for c in os.getenv('SOURCE_CHANNELS', '').split(',') if c.strip()]

openai_client = OpenAI(api_key=OPENAI_API_KEY)

LINK_REGEX = re.compile(r"https?://\S+")

client = TelegramClient('session', API_ID, API_HASH).start(bot_token=BOT_TOKEN)

SIMILARITY_THRESHOLD = 0.9

def extract_links(text: str) -> list[str]:
    """Return all URLs from a string."""
    return LINK_REGEX.findall(text)

async def are_messages_duplicate(msg_a: str, msg_b: str) -> bool:
    """Use the LLM to see if two messages cover the same topic."""
    prompt = (
        "Are the following two Telegram messages describing the same content or article?\n"
        "Reply yes or no."
    )
    try:
        response = openai_client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": f"Message 1:\n{msg_a}\n\nMessage 2:\n{msg_b}",
                },
            ],
            model="gpt-3.5-turbo",
            max_tokens=1,
        )
        answer = response.choices[0].message.content.strip().lower()
        return answer.startswith("y")
    except Exception as e:
        print(f"OpenAI duplicate check error: {e}")
        return False

async def is_nlp_related(text: str) -> bool:
    """Use the LLM to decide if a message is NLP related."""
    prompt = (
        "You are a classifier. Answer yes if the text is about NLP or machine learning. "
        "Otherwise answer no. Only reply with yes or no."
    )
    try:
        response = openai_client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": text}],
            model="gpt-3.5-turbo",
            max_tokens=1,
        )
        answer = response.choices[0].message.content.strip().lower()
        return answer.startswith('y')
    except Exception as e:
        print(f"OpenAI error: {e}")
        return False

async def summarize_text(text: str) -> str:
    """Call LLM to summarize the given text."""
    prompt = (
        "Summarize the following Telegram messages in a concise daily digest focused on NLP."""
    )
    response = openai_client.chat.completions.create(
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": text}],
        model="gpt-3.5-turbo",
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()

async def fetch_messages():
    """Fetch messages from source channels in the last 24 hours."""
    since = datetime.utcnow() - timedelta(days=1)
    all_msgs = []
    for channel in SOURCE_CHANNELS:
        async for msg in client.iter_messages(channel, offset_date=None, min_id=0, reverse=False):
            if msg.date < since:
                break
            if msg.message:
                all_msgs.append(msg.message)
    return all_msgs

async def remove_duplicates(messages):
    unique_msgs = []
    seen_links = set()
    for msg in messages:
        links = extract_links(msg)
        if any(l in seen_links for l in links):
            continue
        duplicate = False
        for u in unique_msgs:
            if SequenceMatcher(None, msg, u).ratio() > SIMILARITY_THRESHOLD:
                duplicate = True
                break
            if await are_messages_duplicate(msg, u):
                duplicate = True
                break
        if not duplicate:
            unique_msgs.append(msg)
            seen_links.update(links)
    return unique_msgs

async def main():
    messages = await fetch_messages()
    print(f"Fetched {len(messages)} messages")
    filtered = []
    for msg in messages:
        if await is_nlp_related(msg):
            filtered.append(msg)
    print(f"{len(filtered)} messages after NLP filter")
    unique = await remove_duplicates(filtered)
    print(f"{len(unique)} messages after deduplication")
    if not unique:
        print("No NLP messages found")
        return
    big_text = "\n".join(unique)
    summary = await summarize_text(big_text)
    await client.send_message(TARGET_CHANNEL, summary)
    print("Summary sent")

if __name__ == "__main__":
    asyncio.run(main())
