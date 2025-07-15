import os
import asyncio
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
import re
from dataclasses import dataclass
from typing import List

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


@dataclass
class MessageInfo:
    """Информация о сообщении Telegram"""
    text: str
    channel: str
    message_id: int
    date: datetime
    link: str
    
    def get_telegram_link(self) -> str:
        """Генерирует ссылку на оригинальное сообщение в Telegram"""
        # Убираем @ из названия канала для формирования ссылки
        channel_name = self.channel.lstrip('@')
        return f"https://t.me/{channel_name}/{self.message_id}"


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


async def are_messages_duplicate(msg_a: MessageInfo, msg_b: MessageInfo) -> bool:
    """Use the LLM to see if two messages cover the same topic."""
    system_prompt = (
        "Описывают ли следующие два сообщения Telegram одинаковый контент или статью?\n"
        "Ответьте да или нет."
    )
    user_content = f"Message 1:\n{msg_a.text}\n\nMessage 2:\n{msg_b.text}"
    
    answer = await call_openai(system_prompt, user_content, max_tokens=1)
    return answer.lower().startswith("y")


async def is_nlp_related(text: str) -> bool:
    """Use the LLM to decide if a message is NLP related and not advertising."""
    system_prompt = (
        "Вы классификатор. Ответьте да, если текст касается NLP или машинного обучения И НЕ является рекламой. "
        "Реклама включает: курсы, платные программы обучения, рекламу русскоязычных LLM, коммерческие предложения. "
        "Отдавайте предпочтение техническим статьям, новым подходам и исследованиям. "
        "Иначе ответьте нет. Отвечайте только да или нет."
    )
    
    answer = await call_openai(system_prompt, text, max_tokens=1)
    return answer.lower().startswith('y')


async def summarize_text(messages: List[MessageInfo]) -> str:
    """Call LLM to summarize the given messages with links."""
    # Подготавливаем текст для суммаризации
    messages_text = "\n\n".join([msg.text for msg in messages])
    
    system_prompt = (
        "Обобщите следующие сообщения Telegram в краткий ежедневный дайджест, сфокусированный на NLP. "
        "Структурируйте дайджест по темам или категориям. Для каждого блока информации указывайте "
        "номер источника в квадратных скобках [1], [2], [3] и т.д. Если несколько источников говорят "
        "об одном и том же, указывайте все номера через запятую, например [1,3] или [2,4,5]. "
        "Для статей и новых подходов кратко опишите технические аспекты: архитектуру, методологию, результаты."
    )
    
    result = await call_openai(system_prompt, messages_text, max_tokens=400)
    if not result:
        return "Ошибка: Не удалось сгенерировать обобщение"
    
    # Заменяем номера источников на HTML-ссылки прямо в тексте
    import re

    def replace_all_sources_html(match):
        content = match.group(1)  # содержимое внутри скобок
        numbers = [num.strip() for num in content.split(',')]
        source_links = []
        for num_str in numbers:
            try:
                num = int(num_str)
                if 1 <= num <= len(messages):
                    source_links.append(f'<a href="{messages[num-1].get_telegram_link()}">[{num}]</a>')
            except ValueError:
                continue
        return ', '.join(source_links)

    # Паттерн для поиска всех ссылок на источники [1], [1,2], [1,2,3] и т.д.
    all_sources_pattern = r'\[(\d+(?:,\s*\d+)*)\]'
    result = re.sub(all_sources_pattern, replace_all_sources_html, result)

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
                message_info = MessageInfo(
                    text=msg.message,
                    channel=channel,
                    message_id=msg.id,
                    date=msg.date,
                    link=msg.get_web_preview() if hasattr(msg, 'get_web_preview') else ""
                )
                channel_msgs.append(message_info)
                all_msgs.append(message_info)
        print(f"  Found {len(channel_msgs)} messages from {channel}")
    return all_msgs


async def remove_duplicates(messages: List[MessageInfo]) -> List[MessageInfo]:
    unique_msgs: List[MessageInfo] = []
    seen_links = set()
    for msg in messages:
        links = extract_links(msg.text)
        if any(link in seen_links for link in links):
            continue
        duplicate = False
        for u in unique_msgs:
            if SequenceMatcher(None, msg.text, u.text).ratio() > SIMILARITY_THRESHOLD:
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
        #     if await is_nlp_related(msg.text):
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
        summary = await summarize_text(unique)
        await user_client.send_message(TARGET_CHANNEL, summary, parse_mode='html')
        print("Summary sent")
    finally:
        # Disconnect both clients
        await user_client.disconnect()
        await bot_client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
