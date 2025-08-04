import re
from openai import OpenAI
from config import OPENAI_API_KEY

openai_client = OpenAI(api_key=OPENAI_API_KEY)

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