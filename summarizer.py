import os
import asyncio
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
import re

from dotenv import load_dotenv
from telethon import TelegramClient
from openai import OpenAI

load_dotenv()

# Get environment variables with proper error handling
api_id_str = os.getenv('TELEGRAM_API_ID')
if not api_id_str:
    raise ValueError("TELEGRAM_API_ID environment variable is required")
API_ID = int(api_id_str)

api_hash = os.getenv('TELEGRAM_API_HASH')
if not api_hash:
    raise ValueError("TELEGRAM_API_HASH environment variable is required")
API_HASH = api_hash

bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
if not bot_token:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")
BOT_TOKEN = bot_token

target_channel = os.getenv('TARGET_CHANNEL')
if not target_channel:
    raise ValueError("TARGET_CHANNEL environment variable is required")
TARGET_CHANNEL = target_channel

openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")
OPENAI_API_KEY = openai_api_key

source_channels_str = os.getenv('SOURCE_CHANNELS', '')
SOURCE_CHANNELS = [c.strip() for c in source_channels_str.split(',') if c.strip()]

openai_client = OpenAI(api_key=OPENAI_API_KEY)

LINK_REGEX = re.compile(r"https?://\S+")

# Create separate clients for user (reading) and bot (sending)
user_client = TelegramClient('tg_summarizer_user', API_ID, API_HASH)
bot_client = TelegramClient('tg_summarizer_bot', API_ID, API_HASH)

SIMILARITY_THRESHOLD = 0.9


async def call_openai(system_prompt: str, user_content: str, max_tokens: int = 300) -> str:
    """Универсальная функция для вызова OpenAI API."""
    try:
        response = openai_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            model="gpt-4o-mini",
            max_tokens=max_tokens,
        )
        result = response.choices[0].message.content
        if result is None:
            return ""
        return result.strip()
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return ""


def extract_links(text: str) -> list[str]:
    """Return all URLs from a string."""
    return LINK_REGEX.findall(text)


async def are_messages_duplicate(msg_a: str, msg_b: str) -> bool:
    """Use the LLM to see if two messages cover the same topic."""
    system_prompt = (
        "Описывают ли следующие два сообщения Telegram одинаковый контент или статью?\n"
        "Ответьте да или нет."
    )
    user_content = f"Message 1:\n{msg_a}\n\nMessage 2:\n{msg_b}"
    
    answer = await call_openai(system_prompt, user_content, max_tokens=1)
    return answer.lower().startswith("y")


async def is_nlp_related(text: str) -> bool:
    """Use the LLM to decide if a message is NLP related."""
    system_prompt = (
        "Вы классификатор. Ответьте да, если текст касается NLP или машинного обучения. "
        "Иначе ответьте нет. Отвечайте только да или нет."
    )
    
    answer = await call_openai(system_prompt, text, max_tokens=1)
    return answer.lower().startswith('y')


async def summarize_text(text: str) -> str:
    """Call LLM to summarize the given text."""
    system_prompt = (
        "Обобщите следующие сообщения Telegram в краткий ежедневный дайджест, сфокусированный на NLP."
    )
    
    result = await call_openai(system_prompt, text, max_tokens=300)
    if not result:
        return "Ошибка: Не удалось сгенерировать обобщение"
    return result


async def fetch_messages():
    """Fetch messages from source channels in the last 24 hours."""
    since = datetime.now(timezone.utc) - timedelta(days=1)
    all_msgs = []
    for channel in SOURCE_CHANNELS:
        print(f"Fetching messages from {channel}...")
        channel_msgs = []
        async for msg in user_client.iter_messages(channel, offset_date=None, min_id=0, reverse=False):
            if msg.date < since:
                break
            if msg.message:
                channel_msgs.append(msg.message)
                all_msgs.append(msg.message)
        print(f"  Found {len(channel_msgs)} messages from {channel}")
    return all_msgs


async def remove_duplicates(messages):
    unique_msgs = []
    seen_links = set()
    for msg in messages:
        links = extract_links(msg)
        if any(link in seen_links for link in links):
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
    # Start both clients
    await user_client.start()
    await bot_client.start(bot_token=BOT_TOKEN)
    
    try:
        messages = await fetch_messages()
        print(f"Fetched {len(messages)} messages")
        # For testing, include all messages (bypass NLP filter)
        filtered = messages
        print(f"Testing mode: including all {len(filtered)} messages (bypassing NLP filter)")
        
        # Uncomment the following lines to re-enable NLP filtering:
        # filtered = []
        # for i, msg in enumerate(messages):
        #     print(f"Checking message {i+1}/{len(messages)}...")
        #     if await is_nlp_related(msg):
        #         filtered.append(msg)
        #         print(f"  ✓ Message {i+1} is NLP-related")
        #     else:
        #         print(f"  ✗ Message {i+1} is not NLP-related")
        # print(f"{len(filtered)} messages after NLP filter")
        unique = await remove_duplicates(filtered)
        print(f"{len(unique)} messages after deduplication")
        if not unique:
            print("No NLP messages found")
            return
        big_text = "\n".join(unique)
        summary = await summarize_text(big_text)
        await user_client.send_message(TARGET_CHANNEL, summary)
        print("Summary sent")
    finally:
        # Disconnect both clients
        await user_client.disconnect()
        await bot_client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
