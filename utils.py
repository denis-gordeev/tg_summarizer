import os
import re
import json
import asyncio
import logging
from datetime import datetime, timezone
from openai import OpenAI, APIError, RateLimitError, APIConnectionError
from config import OPENAI_API_KEY, OPENAI_DEFAULT_MAX_TOKENS, OPENAI_MODEL

logger = logging.getLogger(__name__)


def load_json_file(filepath: str, default: dict = None) -> dict:
    """Generic JSON file loader with error handling."""
    if default is None:
        default = {}
    if not os.path.exists(filepath):
        return default
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error("Error loading %s: %s", filepath, e)
        return default


def save_json_file(filepath: str, data: dict, error_msg: str) -> bool:
    """Generic JSON file saver with error handling."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error("%s: %s", error_msg, e)
        return False


def now_iso() -> str:
    """Returns current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()


openai_client = None

LINK_REGEX = re.compile(r"https?://\S+")
TELEGRAM_CHANNEL_REGEX = re.compile(r"https://t\.me/([^/]+)/\d+")
ABBREVIATION_REGEX = re.compile(r'\[([A-Z0-9]+)\]')


def count_characters(text: str) -> int:
    """Подсчитывает количество символов в тексте, исключая HTML-теги."""
    # Удаляем HTML-теги для корректного подсчета символов
    clean_text = re.sub(r'<[^>]+>', '', text)
    return len(clean_text)


def extract_telegram_channels(text: str) -> list[str]:
    """Извлекает названия каналов из ссылок вида https://t.me/channel_name/message_id."""
    channels = []
    matches = TELEGRAM_CHANNEL_REGEX.findall(text)
    for match in matches:
        channel_name = match.strip()
        if channel_name and channel_name not in channels:
            channels.append(channel_name)
    return channels


def extract_channels_from_abbreviations(text: str) -> list[str]:
    """Извлекает названия каналов из аббревиатур в квадратных скобках."""
    from channel_manager import load_channel_abbreviations
    
    # Загружаем словарь аббревиатур
    abbreviations = load_channel_abbreviations()
    
    # Создаем обратный словарь: аббревиатура -> название канала
    reverse_abbreviations = {v: k for k, v in abbreviations.items()}
    
    # Ищем все аббревиатуры в квадратных скобках
    matches = ABBREVIATION_REGEX.findall(text)
    channels = []
    
    for match in matches:
        abbreviation = match.strip()
        if abbreviation in reverse_abbreviations:
            channel_name = reverse_abbreviations[abbreviation]
            if channel_name not in channels:
                channels.append(channel_name)
    
    return channels


def extract_all_channels(text: str) -> list[str]:
    """Извлекает все каналы из текста: из ссылок и из аббревиатур."""
    # Получаем каналы из ссылок
    link_channels = extract_telegram_channels(text)
    
    # Получаем каналы из аббревиатур
    abbreviation_channels = extract_channels_from_abbreviations(text)
    
    # Объединяем и убираем дубликаты
    all_channels = list(set(link_channels + abbreviation_channels))
    
    return all_channels


async def call_openai(
    system_prompt: str,
    user_content: str,
    max_tokens: int = OPENAI_DEFAULT_MAX_TOKENS,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> str:
    """Универсальная функция для вызова OpenAI API с retry и exponential backoff."""
    global openai_client
    if openai_client is None:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)

    for attempt in range(max_retries + 1):
        try:
            response = openai_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                model=OPENAI_MODEL,
                max_tokens=max_tokens,
            )
            result = response.choices[0].message.content
            if result is None:
                return ""
            return result.strip()
        except (RateLimitError, APIConnectionError) as e:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning("OpenAI retryable error (attempt %d/%d): %s; retrying in %.1fs", attempt + 1, max_retries + 1, e, delay)
                await asyncio.sleep(delay)
            else:
                logger.error("OpenAI API error after %d retries: %s", max_retries, e)
                return ""
        except APIError as e:
            if e.status_code and e.status_code >= 500 and attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning("OpenAI 5xx error (attempt %d/%d): %s; retrying in %.1fs", attempt + 1, max_retries + 1, e, delay)
                await asyncio.sleep(delay)
            else:
                logger.error("OpenAI API error: %s", e)
                return ""
        except Exception as e:
            logger.error("OpenAI API error: %s", e)
            return ""
    return ""


def extract_links(text: str) -> list[str]:
    """Return all URLs from a string."""
    return LINK_REGEX.findall(text) 
