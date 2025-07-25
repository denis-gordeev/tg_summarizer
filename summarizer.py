import os
import asyncio
import json
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
import random
import re
from dataclasses import dataclass
from typing import List, Set

from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.tl.functions.channels import GetChannelRecommendationsRequest
from telethon.tl.types import InputChannel
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
SOURCE_CHANNELS = set([c.strip() for c in source_channels_str.split(',') if c.strip()])

# Новые переменные для групп
source_groups_str = os.getenv('SOURCE_GROUPS', '')
SOURCE_GROUPS = [g.strip() for g in source_groups_str.split(',') if g.strip()]

openai_client = OpenAI(api_key=OPENAI_API_KEY)

LINK_REGEX = re.compile(r"https?://\S+")

# Create separate clients for user (reading) and bot (sending)
user_client = TelegramClient('tg_summarizer_user', API_ID, API_HASH)
bot_client = TelegramClient('tg_summarizer_bot', API_ID, API_HASH)

SIMILARITY_THRESHOLD = 0.9
HISTORY_FILE = 'summarization_history.json'
SUMMARIES_HISTORY_FILE = 'summaries_history.json'
DISCOVERED_CHANNELS_FILE = 'discovered_channels.json'

# Новые файлы для групп
GROUP_HISTORY_FILE = 'group_summarization_history.json'
GROUP_SUMMARIES_HISTORY_FILE = 'group_summaries_history.json'
GROUP_LAST_RUN_FILE = 'group_last_run.json'

# Флаг для отключения проверки покрытия в предыдущих саммари
ENABLE_SUMMARIES_DEDUPLICATION = True


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
    
    def to_dict(self) -> dict:
        """Конвертирует объект в словарь для сохранения в JSON"""
        return {
            'text': self.text,
            'channel': self.channel,
            'message_id': self.message_id,
            'date': self.date.isoformat(),
            'link': self.link
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MessageInfo':
        """Создает объект из словаря"""
        return cls(
            text=data['text'],
            channel=data['channel'],
            message_id=data['message_id'],
            date=datetime.fromisoformat(data['date']),
            link=data['link']
        )


@dataclass
class SummaryInfo:
    """Информация о созданном саммари"""
    content: str
    date: datetime
    message_count: int
    channels: List[str]
    
    def to_dict(self) -> dict:
        """Конвертирует объект в словарь для сохранения в JSON"""
        return {
            'content': self.content,
            'date': self.date.isoformat(),
            'message_count': self.message_count,
            'channels': self.channels
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SummaryInfo':
        """Создает объект из словаря"""
        return cls(
            content=data['content'],
            date=datetime.fromisoformat(data['date']),
            message_count=data['message_count'],
            channels=data['channels']
        )


def count_characters(text: str) -> int:
    """Подсчитывает количество символов в тексте, исключая HTML-теги."""
    # Удаляем HTML-теги для корректного подсчета символов
    import re
    clean_text = re.sub(r'<[^>]+>', '', text)
    return len(clean_text)


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


def load_channel_abbreviations() -> dict:
    """Загружает существующие аббревиатуры каналов из JSON файла."""
    abbreviations_file = 'channel_abbreviations.json'
    if not os.path.exists(abbreviations_file):
        return {}
    
    try:
        with open(abbreviations_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('channel_abbreviations', {})
    except Exception as e:
        print(f"Ошибка при загрузке аббревиатур каналов: {e}")
        return {}


def save_channel_abbreviation(channel_name: str, abbreviation: str) -> None:
    """Сохраняет новую аббревиатуру канала в JSON файл."""
    abbreviations_file = 'channel_abbreviations.json'
    
    try:
        # Загружаем существующие данные
        if os.path.exists(abbreviations_file):
            with open(abbreviations_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {'channel_abbreviations': {}, 'last_updated': ''}
        
        # Добавляем новую аббревиатуру
        data['channel_abbreviations'][channel_name] = abbreviation
        data['last_updated'] = datetime.now().isoformat()
        
        # Сохраняем обновленные данные
        with open(abbreviations_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"Ошибка при сохранении аббревиатуры канала: {e}")


def create_channel_abbreviation(channel_name: str) -> str:
    """Создает аббревиатуру из названия канала."""
    # Сначала проверяем существующие аббревиатуры
    existing_abbreviations = load_channel_abbreviations()
    
    if channel_name in existing_abbreviations:
        return existing_abbreviations[channel_name]
    
    # Если аббревиатуры нет, создаем новую
    clean_name = channel_name.lstrip('@')
    words = re.split(r'[\s\-_]+', clean_name)
    abbreviation = ''.join(word[0].upper() for word in words if word)
    
    # Если аббревиатура слишком длинная, берем только первые 3-4 буквы
    if len(abbreviation) > 4:
        abbreviation = abbreviation[:4]
    
    # Если аббревиатура пустая или слишком короткая, используем первые буквы названия
    if len(abbreviation) < 2:
        abbreviation = clean_name[:3].upper()
    
    # Проверяем, не конфликтует ли новая аббревиатура с существующими
    existing_values = set(existing_abbreviations.values())
    if abbreviation in existing_values:
        # Если конфликт, добавляем цифру
        counter = 1
        while f"{abbreviation}{counter}" in existing_values:
            counter += 1
        abbreviation = f"{abbreviation}{counter}"
    
    # Сохраняем новую аббревиатуру
    save_channel_abbreviation(channel_name, abbreviation)
    
    return abbreviation


def load_summarization_history() -> Set[str]:
    """Загружает историю уже обработанных сообщений из файла."""
    if not os.path.exists(HISTORY_FILE):
        return set()
    
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Создаем уникальные идентификаторы для каждого сообщения
            processed_messages = set()
            for msg_data in data.get('processed_messages', []):
                msg = MessageInfo.from_dict(msg_data)
                # Создаем уникальный идентификатор: канал + message_id + хеш текста
                msg_id = f"{msg.channel}_{msg.message_id}_{hash(msg.text)}"
                processed_messages.add(msg_id)
            return processed_messages
    except Exception as e:
        print(f"Ошибка при загрузке истории: {e}")
        return set()


def save_summarization_history(messages: List[MessageInfo]) -> None:
    """Сохраняет обработанные сообщения в историю."""
    try:
        # Загружаем существующую историю
        existing_data = []
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                existing_data = data.get('processed_messages', [])
        
        # Добавляем новые сообщения
        new_messages = [msg.to_dict() for msg in messages]
        all_messages = existing_data + new_messages
        
        # Ограничиваем историю последними 1000 сообщениями
        if len(all_messages) > 1000:
            all_messages = all_messages[-1000:]
        
        # Сохраняем обновленную историю
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump({
                'processed_messages': all_messages,
                'last_updated': datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"Ошибка при сохранении истории: {e}")


def load_summaries_history() -> List[SummaryInfo]:
    """Загружает историю созданных саммари из файла."""
    if not os.path.exists(SUMMARIES_HISTORY_FILE):
        return []
    
    try:
        with open(SUMMARIES_HISTORY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            summaries = []
            for summary_data in data.get('summaries', []):
                summaries.append(SummaryInfo.from_dict(summary_data))
            return summaries
    except Exception as e:
        print(f"Ошибка при загрузке истории саммари: {e}")
        return []


def save_summary_to_history(summary: SummaryInfo) -> None:
    """Сохраняет новое саммари в историю."""
    try:
        # Загружаем существующую историю
        existing_summaries = []
        if os.path.exists(SUMMARIES_HISTORY_FILE):
            with open(SUMMARIES_HISTORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                existing_summaries = data.get('summaries', [])
        
        # Добавляем новое саммари
        new_summary = summary.to_dict()
        all_summaries = existing_summaries + [new_summary]
        
        # Ограничиваем историю последними 50 саммари
        if len(all_summaries) > 50:
            all_summaries = all_summaries[-50:]
        
        # Сохраняем обновленную историю
        with open(SUMMARIES_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump({
                'summaries': all_summaries,
                'last_updated': datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"Ошибка при сохранении истории саммари: {e}")


def load_discovered_channels() -> List[str]:
    """Загружает список обнаруженных каналов из файла."""
    if not os.path.exists(DISCOVERED_CHANNELS_FILE):
        return []
    
    try:
        with open(DISCOVERED_CHANNELS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('discovered_channels', [])
    except Exception as e:
        print(f"Ошибка при загрузке обнаруженных каналов: {e}")
        return []


def load_similar_channels() -> List[str]:
    """Загружает список похожих каналов из файла."""
    if not os.path.exists(DISCOVERED_CHANNELS_FILE):
        return []
    
    try:
        with open(DISCOVERED_CHANNELS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('similar_channels', [])
    except Exception as e:
        print(f"Ошибка при загрузке похожих каналов: {e}")
        return []


def load_banned_channels() -> List[str]:
    """Загружает список заблокированных каналов из файла."""
    if not os.path.exists(DISCOVERED_CHANNELS_FILE):
        return []
    
    try:
        with open(DISCOVERED_CHANNELS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('banned_channels', [])
    except Exception as e:
        print(f"Ошибка при загрузке заблокированных каналов: {e}")
        return []


def save_discovered_channel(channel_name: str) -> None:
    """Сохраняет новый обнаруженный канал в JSON файл."""
    try:
        # Загружаем существующие данные
        if os.path.exists(DISCOVERED_CHANNELS_FILE):
            with open(DISCOVERED_CHANNELS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {'discovered_channels': [], 'similar_channels': [], 'last_updated': ''}
        
        # Проверяем, нет ли уже такого канала
        if channel_name not in data['discovered_channels']:
            data['discovered_channels'].append(channel_name)
            data['last_updated'] = datetime.now().isoformat()
            
            # Сохраняем обновленные данные
            with open(DISCOVERED_CHANNELS_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
    except Exception as e:
        print(f"Ошибка при сохранении обнаруженного канала: {e}")


def save_similar_channel(channel_name: str) -> None:
    """Сохраняет новый похожий канал в JSON файл."""
    try:
        # Загружаем существующие данные
        if os.path.exists(DISCOVERED_CHANNELS_FILE):
            with open(DISCOVERED_CHANNELS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {'discovered_channels': [], 'similar_channels': [], 'banned_channels': [], 'last_updated': ''}
        
        # Проверяем, нет ли уже такого канала
        if channel_name not in data['similar_channels']:
            data['similar_channels'].append(channel_name)
            data['last_updated'] = datetime.now().isoformat()
            
            # Сохраняем обновленные данные
            with open(DISCOVERED_CHANNELS_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
    except Exception as e:
        print(f"Ошибка при сохранении похожего канала: {e}")


def save_banned_channel(channel_name: str) -> None:
    """Сохраняет новый заблокированный канал в JSON файл."""
    try:
        # Загружаем существующие данные
        if os.path.exists(DISCOVERED_CHANNELS_FILE):
            with open(DISCOVERED_CHANNELS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {'discovered_channels': [], 'similar_channels': [], 'banned_channels': [], 'last_updated': ''}
        
        # Проверяем, нет ли уже такого канала
        if channel_name not in data['banned_channels']:
            data['banned_channels'].append(channel_name)
            data['last_updated'] = datetime.now().isoformat()
            
            # Сохраняем обновленные данные
            with open(DISCOVERED_CHANNELS_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
    except Exception as e:
        print(f"Ошибка при сохранении заблокированного канала: {e}")


async def get_similar_channels_from_telegram(channel_username: str = None) -> List[str]:
    """Получает список похожих каналов через Telegram API."""
    try:
        if not user_client.is_connected():
            await user_client.start()
        
        # Если указан канал, получаем рекомендации для него
        if channel_username:
            # Убираем @ если есть
            channel_username = channel_username.lstrip('@')
            
            # Получаем информацию о канале
            try:
                channel_entity = await user_client.get_entity(f"@{channel_username}")
                channel_input = InputChannel(channel_entity.id, channel_entity.access_hash)
                
                # Получаем рекомендации
                result = await user_client(GetChannelRecommendationsRequest(
                    channel=channel_input
                ))
                
                similar_channels = []
                for chat in result.chats:
                    if hasattr(chat, 'username') and chat.username:
                        similar_channels.append(f"@{chat.username}")
                
                return similar_channels
                
            except Exception as e:
                print(f"Ошибка при получении рекомендаций для канала {channel_username}: {e}")
                return []
        
        # Если канал не указан, получаем глобальные рекомендации
        else:
            result = await user_client(GetChannelRecommendationsRequest())
            
            similar_channels = []
            for chat in result.chats:
                if hasattr(chat, 'username') and chat.username:
                    similar_channels.append(f"@{chat.username}")
            
            return similar_channels
            
    except Exception as e:
        print(f"Ошибка при получении похожих каналов: {e}")
        return []


async def discover_and_save_similar_channels(channel_username: str = None) -> None:
    """Обнаруживает и сохраняет похожие каналы."""
    try:
        # Загружаем существующие каналы для проверки дубликатов
        existing_discovered = set(load_discovered_channels())
        existing_similar = set(load_similar_channels())
        existing_banned = set(load_banned_channels())
        
        # Объединяем все существующие каналы
        existing_channels = SOURCE_CHANNELS | existing_discovered | existing_similar | existing_banned
        
        if channel_username:
            # Если указан конкретный канал, проверяем что он в SOURCE_CHANNELS
            if channel_username not in SOURCE_CHANNELS:
                print(f"Канал {channel_username} не найден в SOURCE_CHANNELS. "
                      f"Поиск похожих каналов только для каналов из SOURCE_CHANNELS.")
                return
            
            print(f"Получаем похожие каналы для {channel_username}...")
            similar_channels = await get_similar_channels_from_telegram(channel_username)
        else:
            # Если канал не указан, ищем похожие для всех каналов из SOURCE_CHANNELS
            print(f"Получаем похожие каналы для всех каналов из SOURCE_CHANNELS...")
            all_similar_channels = set()
            
            for source_channel in SOURCE_CHANNELS:
                print(f"  Ищем похожие для {source_channel}...")
                similar_channels = await get_similar_channels_from_telegram(source_channel)
                all_similar_channels.update(similar_channels)
                # Небольшая пауза между запросами
                await asyncio.sleep(1)
            
            similar_channels = list(all_similar_channels)
        
        if similar_channels:
            # Фильтруем каналы, которые уже существуют
            new_similar_channels = []
            skipped_channels = []
            
            for channel in similar_channels:
                if channel in existing_channels:
                    skipped_channels.append(channel)
                else:
                    new_similar_channels.append(channel)
            
            if new_similar_channels:
                print(f"Найдено {len(new_similar_channels)} новых уникальных похожих каналов:")
                for channel in new_similar_channels:
                    print(f"  - {channel}")
                    save_similar_channel(channel)
                print("Новые похожие каналы сохранены в discovered_channels.json")
            else:
                print("Все найденные каналы уже существуют в системе")
            
            if skipped_channels:
                print(f"Пропущено {len(skipped_channels)} каналов (уже существуют):")
                for channel in skipped_channels[:10]:  # Показываем только первые 10
                    print(f"  - {channel}")
                if len(skipped_channels) > 10:
                    print(f"  ... и еще {len(skipped_channels) - 10} каналов")
        else:
            print("Похожих каналов не найдено")
            
    except Exception as e:
        print(f"Ошибка при обнаружении похожих каналов: {e}")


def load_group_summarization_history() -> Set[str]:
    """Загружает историю уже обработанных сообщений из групп из файла."""
    if not os.path.exists(GROUP_HISTORY_FILE):
        return set()
    
    try:
        with open(GROUP_HISTORY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Создаем уникальные идентификаторы для каждого сообщения
            processed_messages = set()
            for msg_data in data.get('processed_messages', []):
                msg = MessageInfo.from_dict(msg_data)
                # Создаем уникальный идентификатор: канал + message_id + хеш текста
                msg_id = f"{msg.channel}_{msg.message_id}_{hash(msg.text)}"
                processed_messages.add(msg_id)
            return processed_messages
    except Exception as e:
        print(f"Ошибка при загрузке истории групп: {e}")
        return set()


def save_group_summarization_history(messages: List[MessageInfo]) -> None:
    """Сохраняет историю обработанных сообщений из групп в файл."""
    try:
        # Загружаем существующие данные
        if os.path.exists(GROUP_HISTORY_FILE):
            with open(GROUP_HISTORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {'processed_messages': [], 'last_updated': ''}
        
        # Добавляем новые сообщения
        for msg in messages:
            data['processed_messages'].append(msg.to_dict())
        
        # Ограничиваем историю последними 1000 сообщениями
        if len(data['processed_messages']) > 1000:
            data['processed_messages'] = data['processed_messages'][-1000:]
        
        data['last_updated'] = datetime.now().isoformat()
        
        # Сохраняем обновленные данные
        with open(GROUP_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"Ошибка при сохранении истории групп: {e}")


def load_group_summaries_history() -> List[SummaryInfo]:
    """Загружает историю созданных саммари из групп из файла."""
    if not os.path.exists(GROUP_SUMMARIES_HISTORY_FILE):
        return []
    
    try:
        with open(GROUP_SUMMARIES_HISTORY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            summaries = []
            for summary_data in data.get('summaries', []):
                summaries.append(SummaryInfo.from_dict(summary_data))
            return summaries
    except Exception as e:
        print(f"Ошибка при загрузке истории саммари групп: {e}")
        return []


def save_group_summary_to_history(summary: SummaryInfo) -> None:
    """Сохраняет саммари из групп в историю."""
    try:
        # Загружаем существующие данные
        if os.path.exists(GROUP_SUMMARIES_HISTORY_FILE):
            with open(GROUP_SUMMARIES_HISTORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {'summaries': [], 'last_updated': ''}
        
        # Добавляем новое саммари
        data['summaries'].append(summary.to_dict())
        
        # Ограничиваем историю последними 100 саммари
        if len(data['summaries']) > 100:
            data['summaries'] = data['summaries'][-100:]
        
        data['last_updated'] = datetime.now().isoformat()
        
        # Сохраняем обновленные данные
        with open(GROUP_SUMMARIES_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"Ошибка при сохранении саммари групп в историю: {e}")


def should_run_group_summarization() -> bool:
    """Проверяет, нужно ли запускать суммаризацию групп (раз в сутки)."""
    if not os.path.exists(GROUP_LAST_RUN_FILE):
        return True
    
    try:
        with open(GROUP_LAST_RUN_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            last_run_str = data.get('last_run', '')
            if not last_run_str:
                return True
            
            last_run = datetime.fromisoformat(last_run_str)
            now = datetime.now(timezone.utc)
            
            # Проверяем, прошло ли больше 24 часов с последнего запуска
            return (now - last_run).total_seconds() > 24 * 60 * 60
            
    except Exception as e:
        print(f"Ошибка при проверке времени последнего запуска групп: {e}")
        return True


def update_group_last_run() -> None:
    """Обновляет время последнего запуска суммаризации групп."""
    try:
        data = {
            'last_run': datetime.now(timezone.utc).isoformat(),
            'last_updated': datetime.now().isoformat()
        }
        
        with open(GROUP_LAST_RUN_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"Ошибка при обновлении времени последнего запуска групп: {e}")


def get_recent_group_summaries_context(days: int = 7) -> str:
    """Получает контекст последних саммари из групп для дедупликации."""
    summaries = load_group_summaries_history()
    if not summaries:
        return ""
    
    # Фильтруем саммари за последние N дней
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    recent_summaries = [s for s in summaries if s.date >= cutoff_date]
    
    if not recent_summaries:
        return ""
    
    # Объединяем содержимое саммари
    context_parts = []
    for summary in recent_summaries:
        context_parts.append(f"Дата: {summary.date.strftime('%Y-%m-%d')}")
        context_parts.append(f"Содержание: {summary.content}")
        context_parts.append("---")
    
    return "\n".join(context_parts)


def get_all_source_channels() -> List[str]:
    """Возвращает объединенный список каналов из .env, обнаруженных и похожих каналов."""
    discovered_channels = set(load_discovered_channels())
    similar_channels = set(load_similar_channels())
    banned_channels = set(load_banned_channels())
    
    # Исключаем заблокированные каналы
    all_channels_set = SOURCE_CHANNELS | discovered_channels | similar_channels
    all_channels_set = all_channels_set - banned_channels
    
    all_channels = list(all_channels_set)
    random.shuffle(all_channels)
    print(f"Используем {len(SOURCE_CHANNELS)} каналов из .env, "
          f"{len(discovered_channels)} обнаруженных, {len(similar_channels)} похожих каналов "
          f"(исключено {len(banned_channels)} заблокированных)")
    return all_channels


def get_recent_summaries_context(days: int = 3) -> str:
    """Возвращает контекст последних саммари для дедупликации."""
    summaries = load_summaries_history()
    if not summaries:
        return ""
    
    # Фильтруем саммари за последние N дней
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    recent_summaries = [s for s in summaries if s.date >= cutoff_date]
    
    if not recent_summaries:
        return ""
    
    # Создаем контекст из последних 50 саммари
    context_parts = []
    for i, summary in enumerate(recent_summaries[-50:], 1):
        # Очищаем HTML теги для лучшего сравнения
        clean_content = re.sub(r'<[^>]+>', '', summary.content)
        # Берем первые 500 символов для контекста
        context_parts.append(f"Саммари {i} ({summary.date.strftime('%Y-%m-%d')}):\n{clean_content[:500]}...")
    
    return "\n\n".join(context_parts)


def is_message_processed(msg: MessageInfo, processed_messages: Set[str]) -> bool:
    """Проверяет, было ли сообщение уже обработано ранее."""
    msg_id = f"{msg.channel}_{msg.message_id}_{hash(msg.text)}"
    return msg_id in processed_messages


async def are_messages_duplicate(msg_a: MessageInfo, msg_b: MessageInfo) -> bool:
    """Use the LLM to see if two messages cover the same topic."""
    system_prompt = (
        "Описывают ли следующие два сообщения Telegram одинаковый контент или статью?\n"
        "Ответьте да или нет."
    )
    user_content = f"Message 1:\n{msg_a.text}\n\nMessage 2:\n{msg_b.text}"
    
    answer = await call_openai(system_prompt, user_content, max_tokens=1)
    return answer.lower().startswith("y")


async def is_message_covered_in_summaries(msg: MessageInfo) -> bool:
    """Проверяет, была ли тема сообщения уже освещена в предыдущих саммари."""
    if not ENABLE_SUMMARIES_DEDUPLICATION:
        return False
    
    recent_summaries = get_recent_summaries_context()
    if not recent_summaries:
        return False
    
    system_prompt = """Ты эксперт по анализу текстов. Твоя задача - определить, была ли тема из нового сообщения уже освещена в предыдущих дайджестах.

    Правила:
    1. Сравнивай ТЕМУ и ОСНОВНУЮ ИДЕЮ сообщения, а не точные формулировки
    2. Если тема уже была освещена в предыдущих дайджестах, отвечай "ДА"
    3. Если это новая тема, отвечай "НЕТ"
    4. Учитывай только существенную информацию, игнорируй мелкие детали

    Примеры:
    - "Новый алгоритм GPT-5" и "OpenAI анонсировал GPT-5" = ДА (та же тема)
    - "Новый алгоритм GPT-5" и "Новый алгоритм BERT" = НЕТ (разные темы)
    - "Цена акций NVIDIA выросла" и "NVIDIA показала хорошие результаты" = ДА (та же тема)

    Отвечай только "ДА" или "НЕТ"."""

    user_content = f"""Предыдущие дайджесты:
        {recent_summaries}

        Новое сообщение:
        {msg.text}

        Была ли эта тема уже освещена в предыдущих дайджестах?"""

    try:
        result = await call_openai(system_prompt, user_content, max_tokens=10)
        return result.strip().upper() == "ДА"
    except Exception as e:
        print(f"Ошибка при проверке покрытия в саммари: {e}")
        return False


async def is_message_covered_in_group_summaries(msg: MessageInfo) -> bool:
    """Проверяет, была ли тема сообщения уже освещена в предыдущих саммари групп."""
    if not ENABLE_SUMMARIES_DEDUPLICATION:
        return False
    
    recent_summaries = get_recent_group_summaries_context()
    if not recent_summaries:
        return False
    
    system_prompt = """Ты эксперт по анализу текстов. Твоя задача - определить, была ли тема из нового сообщения уже освещена в предыдущих дайджестах групп.

Правила:
1. Сравнивай ТЕМУ и ОСНОВНУЮ ИДЕЮ сообщения, а не точные формулировки
2. Если тема уже была освещена в предыдущих дайджестах групп, отвечай "ДА"
3. Если это новая тема, отвечай "НЕТ"
4. Учитывай только существенную информацию, игнорируй мелкие детали

Примеры:
- "Новый алгоритм GPT-5" и "OpenAI анонсировал GPT-5" = ДА (та же тема)
- "Новый алгоритм GPT-5" и "Новый алгоритм BERT" = НЕТ (разные темы)
- "Цена акций NVIDIA выросла" и "NVIDIA показала хорошие результаты" = ДА (та же тема)

Отвечай только "ДА" или "НЕТ"."""

    user_content = f"""Предыдущие дайджесты групп:
{recent_summaries}

Новое сообщение:
{msg.text}

Была ли эта тема уже освещена в предыдущих дайджестах групп?"""

    try:
        result = await call_openai(system_prompt, user_content, max_tokens=10)
        return result.strip().upper() == "ДА"
    except Exception as e:
        print(f"Ошибка при проверке покрытия в саммари групп: {e}")
        return False


async def is_nlp_related(text: str) -> bool:
    """Use the LLM to decide if a message is NLP related and not advertising."""
    system_prompt = (
        "Вы классификатор. Определите, является ли текст релевантным для NLP/ML/AI/programming/Python дайджеста.\n\n"
        "ПРИНИМАЙТЕ (ответьте 'да'):\n"
        "- Научные статьи и исследования\n"
        "- Новые модели, библиотеки, инструменты\n"
        "- Технические обзоры и бенчмарки\n"
        "- Академические конференции и воркшопы\n"
        "- Открытые проекты и датасеты\n"
        "- Новости про Kaggle, хакатоны и соревнования\n"
        "- Информация и советы о карьере и вакансиях\n"
        "- Информация о пет-проектах\n"
        "- Тексты по ии и bigtech компании\n"
        "- Тексты про стартапы\n"
        "- Тексты про ии-ассистентов и приложения (ChatGPT, Claude, Gemini, Grok, DeepSeek, KlingAi, Midjourney etc.)\n"
        "- Любые новости про Сэма Альтмана (Sam Altman), Марка Цукерберга"
        "OpenAI, Anthropic, Google, Meta, Microsoft, Nvidia, FAANG и т.д.\n"
        "- Новости про покупку, продажу компаний и поглощения\n:"
        "- Новости про увольнения и ИИ-экономику\n:"
        "- Новости про GPU и железо\n:"
        "- Новости про вайбкодинг (vibe coding) и ии-агентов (часто просто агенты, agents)\n:"
        "- Новости про LLM и их использование\n:"
        "- Новости про опыт тренировки моделей машинного обучения\n:"
        "- Мемы из мира ии и технологий\n:"
        "- Промпт-инжениринг\n:"

        "ОТКЛОНЯЙТЕ (ответьте 'нет'):\n"
        "- Курсы, обучение, платные программы\n"
        "- Реклама русскоязычных LLM (GigaChat, YandexGPT)\n"
        "- Коммерческие предложения и услуги\n"
        "- Вебинары с продажами\n"
        "- Мастер-классы с сертификатами\n"
        "- Hiring days\n"
        "- В сообщении есть ссылки ведущие на ботов, при этом не указано перед ссылкой, что это ссылка на бота\n"
        "- Сообщение содержит ссылку на бота (например, @ChattyEnglishBot)\n"
        "Отвечайте только 'да' или 'нет'."
    )
    
    answer = await call_openai(system_prompt, text, max_tokens=5)
    return answer.lower().strip().startswith('да')


async def summarize_text(messages: List[MessageInfo]) -> str:
    """Call LLM to summarize the given messages with links."""
    # Подготавливаем текст для суммаризации с указанием номеров источников
    messages_with_sources = []
    total_original_length = 0
    
    for i, msg in enumerate(messages, 1):
        # Извлекаем ссылки из текста сообщения
        links = extract_links(msg.text)
        source_info = f"[{i}] {msg.text}"
        if links:
            source_info += f" (Ссылки: {', '.join(links)})"
        messages_with_sources.append(source_info)
        total_original_length += count_characters(msg.text)
    
    messages_text = "\n\n".join(messages_with_sources)
    
    # Устанавливаем максимальную длину саммари
    max_summary_length = min(total_original_length * 2, 16000)
    
    system_prompt = (
        "Обобщите следующие сообщения Telegram в краткий ежедневный дайджест, сфокусированный на NLP. "
        "Структурируйте дайджест по темам или категориям. Для каждого блока информации указывайте "
        "номер источника в квадратных скобках [1], [2], [3] и т.д. Если несколько источников говорят "
        "об одном и том же, то указывайте все. "
        "Для статей и новых подходов кратко опишите технические аспекты: архитектуру, методологию, результаты. "
        "ВАЖНО: НЕ добавляйте введение и заключение. Начинайте сразу с содержания дайджеста. "
        "ВАЖНО: Учитывайте, что Telegram-каналы выражают свое мнение и могут содержать субъективные оценки. "
        "Если автор канала рекомендует что-то или дает советы, НЕ дословно переписывайте эти рекомендации. "
        "Вместо этого объективно опишите факты и события, о которых идет речь. "
        "ВАЖНО: ПРИДЕРЖИВАЙТЕСЬ НЕЙТРАЛЬНОГО СТИЛЯ, даже если исходные сообщения эмоциональные, "
        "восторженные или содержат преувеличения. Передавайте информацию объективно, без эмоциональной окраски. "
        "ВАЖНО: Используйте стиль заголовков, который соответствует оригинальным сообщениям из Telegram-каналов. "
        "Анализируйте стиль исходных сообщений и создавайте заголовки в том же духе - используйте эмодзи, "
        "короткие и яркие формулировки, характерные для Telegram. НЕ используйте формальные академические заголовки. "
        "ВАЖНО: В начале каждого заголовка добавляйте 1-5 эмодзи, которые креативно и метафорично "
        "описывают суть новости. Используйте неожиданные, но понятные комбинации эмодзи. "
        "Эмодзи НЕ должны быть синонимичными. "
        "Примеры стиля: '<b>🧠💧 Переходы между AI-компаниями</b>', "
        "'<b>🤖🎭👅 Новый подход к обучению языковых моделей</b>', "
        "'<b>🫱👁️🫲 Прорыв в компьютерном зрении</b>', "
        "'<b>💡👅 Инновации в NLP</b>', "
        "'<b>🎯🪿📈 Новые результаты на Kaggle</b>', "
        "'<b>💼🌪️📉 Рынок AI-вакансий</b>'. "
        "ВАЖНО: Используйте ТОЛЬКО HTML-теги <b>текст</b> для жирного шрифта, "
        "НЕ используйте **текст** или другие Markdown разметки. "
        "ВАЖНО: НЕ создавайте собственные ссылки в тексте, используйте только номера источников [1], [2], [3] и т.д. "
        "Ссылки будут добавлены автоматически. "
        f"КРИТИЧЕСКИ ВАЖНО: Максимальная длина саммари НЕ ДОЛЖНА превышать {max_summary_length} символов. "
        "Для коротких сообщений создавайте развернутые саммари, "
        "но строго соблюдайте лимит символов."
        "denissexy - это канал про машинное обучение и искусственный интеллект"
    )
    
    # Вычисляем max_tokens на основе максимальной длины саммари (примерно 4 символа на токен)
    max_tokens = 16000
    
    result = await call_openai(system_prompt, messages_text, max_tokens=max_tokens)
    if not result:
        return "Ошибка: Не удалось сгенерировать обобщение"
    
    print(f"Длина исходного текста: {total_original_length} символов")
    print(f"Длина саммари: {count_characters(result)} символов")
    
    # Заменяем номера источников на HTML-ссылки
    def replace_source_with_links(match):
        content = match.group(1)  # содержимое внутри скобок
        numbers = [num.strip() for num in content.split(',')]
        source_links = []
        
        for num_str in numbers:
            try:
                num = int(num_str)
                if 1 <= num <= len(messages):
                    msg = messages[num-1]
                    # Извлекаем ссылки из текста сообщения
                    links = extract_links(msg.text)
                    telegram_link = msg.get_telegram_link()
                    
                    # Всегда создаем аббревиатуру канала
                    channel_abbr = create_channel_abbreviation(msg.channel)
                    
                    if links:
                        # Если есть внешние ссылки, добавляем и внешнюю ссылку, и ссылку на Telegram-пост
                        main_link = links[0]
                        source_links.append(
                            f'<a href="{main_link}">[{num}]</a> <a href="{telegram_link}">[{channel_abbr}]</a>'
                        )
                    else:
                        # Если внешних ссылок нет, используем только ссылку на Telegram-сообщение с аббревиатурой
                        source_links.append(f'<a href="{telegram_link}">[{channel_abbr}]</a>')
                        
            except ValueError:
                continue
        return ', '.join(source_links)

    # Паттерн для поиска всех ссылок на источники [1], [1,2], [1,2,3] и т.д.
    all_sources_pattern = r'\[(\d+(?:,\s*\d+)*)\]'
    result = re.sub(all_sources_pattern, replace_source_with_links, result)

    return result


async def summarize_group_text(messages: List[MessageInfo]) -> str:
    """Call LLM to summarize group messages with community review header."""
    # Подготавливаем текст для суммаризации с указанием номеров источников
    messages_with_sources = []
    total_original_length = 0
    
    for i, msg in enumerate(messages, 1):
        # Извлекаем ссылки из текста сообщения
        links = extract_links(msg.text)
        source_info = f"[{i}] {msg.text}"
        if links:
            source_info += f" (Ссылки: {', '.join(links)})"
        messages_with_sources.append(source_info)
        total_original_length += count_characters(msg.text)
    
    messages_text = "\n\n".join(messages_with_sources)
    
    # Устанавливаем максимальную длину саммари
    max_summary_length = min(total_original_length * 2, 16000)
    
    system_prompt = (
        "Обобщите следующие сообщения из Telegram-группы в краткий дайджест, сфокусированный на NLP. "
        "Структурируйте дайджест по темам или категориям. Для каждого блока информации указывайте "
        "номер источника в квадратных скобках [1], [2], [3] и т.д. Если несколько источников говорят "
        "об одном и том же, то указывайте все. "
        "Для статей и новых подходов кратко опишите технические аспекты: архитектуру, методологию, результаты. "
        "ВАЖНО: НЕ добавляйте введение и заключение. Начинайте сразу с содержания дайджеста. "
        "ВАЖНО: Учитывайте, что сообщения из групп могут содержать обсуждения, вопросы и мнения участников. "
        "Если участники рекомендуют что-то или дают советы, НЕ дословно переписывайте эти рекомендации. "
        "Вместо этого объективно опишите факты и события, о которых идет речь. "
        "ВАЖНО: ПРИДЕРЖИВАЙТЕСЬ НЕЙТРАЛЬНОГО СТИЛЯ, даже если исходные сообщения эмоциональные, "
        "восторженные или содержат преувеличения. Передавайте информацию объективно, без эмоциональной окраски. "
        "ВАЖНО: Используйте стиль заголовков, который соответствует оригинальным сообщениям из Telegram-групп. "
        "Анализируйте стиль исходных сообщений и создавайте заголовки в том же духе - используйте эмодзи, "
        "короткие и яркие формулировки, характерные для Telegram. НЕ используйте формальные академические заголовки. "
        "ВАЖНО: В начале каждого заголовка добавляйте 1-5 эмодзи, которые креативно и метафорично "
        "описывают суть новости. Используйте неожиданные, но понятные комбинации эмодзи. "
        "Эмодзи НЕ должны быть синонимичными. "
        "Примеры стиля: '<b>🧠💧 Переходы между AI-компаниями</b>', "
        "'<b>🤖🎭👅 Новый подход к обучению языковых моделей</b>', "
        "'<b>🫱👁️🫲 Прорыв в компьютерном зрении</b>', "
        "'<b>💡👅 Инновации в NLP</b>', "
        "'<b>🎯🪿📈 Новые результаты на Kaggle</b>', "
        "'<b>💼🌪️📉 Рынок AI-вакансий</b>'. "
        "ВАЖНО: Используйте ТОЛЬКО HTML-теги <b>текст</b> для жирного шрифта, "
        "НЕ используйте **текст** или другие Markdown разметки. "
        "ВАЖНО: НЕ создавайте собственные ссылки в тексте, используйте только номера источников [1], [2], [3] и т.д. "
        "Ссылки будут добавлены автоматически. "
        f"КРИТИЧЕСКИ ВАЖНО: Максимальная длина саммари НЕ ДОЛЖНА превышать {max_summary_length} символов. "
        "Для коротких сообщений создавайте развернутые саммари, "
        "но строго соблюдайте лимит символов."
    )
    
    # Вычисляем max_tokens на основе максимальной длины саммари (примерно 4 символа на токен)
    max_tokens = 16000
    
    result = await call_openai(system_prompt, messages_text, max_tokens=max_tokens)
    if not result:
        return "Ошибка: Не удалось сгенерировать обобщение"
    
    print(f"Длина исходного текста групп: {total_original_length} символов")
    print(f"Длина саммари групп: {count_characters(result)} символов")
    
    # Заменяем номера источников на HTML-ссылки
    def replace_source_with_links(match):
        content = match.group(1)  # содержимое внутри скобок
        numbers = [num.strip() for num in content.split(',')]
        source_links = []
        
        for num_str in numbers:
            try:
                num = int(num_str)
                if 1 <= num <= len(messages):
                    msg = messages[num-1]
                    # Извлекаем ссылки из текста сообщения
                    links = extract_links(msg.text)
                    telegram_link = msg.get_telegram_link()
                    
                    # Всегда создаем аббревиатуру канала
                    channel_abbr = create_channel_abbreviation(msg.channel)
                    
                    if links:
                        # Если есть внешние ссылки, добавляем и внешнюю ссылку, и ссылку на Telegram-пост
                        main_link = links[0]
                        source_links.append(
                            f'<a href="{main_link}">[{num}]</a> <a href="{telegram_link}">[{channel_abbr}]</a>'
                        )
                    else:
                        # Если внешних ссылок нет, используем только ссылку на Telegram-сообщение с аббревиатурой
                        source_links.append(f'<a href="{telegram_link}">[{channel_abbr}]</a>')
                        
            except ValueError:
                continue
        return ', '.join(source_links)

    # Паттерн для поиска всех ссылок на источники [1], [1,2], [1,2,3] и т.д.
    all_sources_pattern = r'\[(\d+(?:,\s*\d+)*)\]'
    result = re.sub(all_sources_pattern, replace_source_with_links, result)
    
    # Добавляем заголовок "Обзор сообщества"
    group_names = list(set(msg.channel.lstrip('@') for msg in messages))
    community_name = ', '.join(group_names)
    header = f"<b>👥 Обзор сообщества {community_name}</b>\n\n"
    
    return header + result


async def fetch_messages():
    """Fetch messages from source channels in the last 24 hours."""
    since = datetime.now(timezone.utc) - timedelta(days=1)
    all_msgs = []
    
    # Загружаем историю обработанных сообщений
    processed_messages = load_summarization_history()
    print(f"Загружено {len(processed_messages)} уже обработанных сообщений из истории")
    
    # Получаем объединенный список каналов
    all_channels = get_all_source_channels()
    
    for channel in all_channels:
        print(f"Fetching messages from {channel}...")
        channel_msgs = []
        async for msg in user_client.iter_messages(channel, offset_date=None, min_id=0, reverse=False):
            if msg.date < since:
                break
            if msg.message:
                # Извлекаем ссылки из текста сообщения
                links = extract_links(msg.message)
                main_link = links[0] if links else ""
                
                message_info = MessageInfo(
                    text=msg.message,
                    channel=channel,
                    message_id=msg.id,
                    date=msg.date,
                    link=main_link
                )
                
                # Проверяем, не было ли сообщение уже обработано
                if not is_message_processed(message_info, processed_messages):
                    channel_msgs.append(message_info)
                    all_msgs.append(message_info)
                else:
                    print(f"  Пропускаем уже обработанное сообщение {msg.id} из {channel}")
                    
        print(f"  Found {len(channel_msgs)} новых сообщений from {channel}")
    return all_msgs


async def fetch_group_messages():
    """Fetch messages from source groups in the last 24 hours."""
    since = datetime.now(timezone.utc) - timedelta(days=1)
    all_msgs = []
    
    # Загружаем историю обработанных сообщений из групп
    processed_messages = load_group_summarization_history()
    print(f"Загружено {len(processed_messages)} уже обработанных сообщений из групп из истории")
    
    # Получаем список групп
    all_groups = SOURCE_GROUPS
    
    for group in all_groups:
        print(f"Fetching messages from group {group}...")
        group_msgs = []
        async for msg in user_client.iter_messages(group, offset_date=None, min_id=0, reverse=False):
            if msg.date < since:
                break
            if msg.message:
                # Извлекаем ссылки из текста сообщения
                links = extract_links(msg.message)
                main_link = links[0] if links else ""
                
                message_info = MessageInfo(
                    text=msg.message,
                    channel=group,
                    message_id=msg.id,
                    date=msg.date,
                    link=main_link
                )
                
                # Проверяем, не было ли сообщение уже обработано
                if not is_message_processed(message_info, processed_messages):
                    group_msgs.append(message_info)
                    all_msgs.append(message_info)
                else:
                    print(f"  Пропускаем уже обработанное сообщение {msg.id} из группы {group}")
                    
        print(f"  Found {len(group_msgs)} новых сообщений from group {group}")
    return all_msgs


async def remove_duplicates(messages: List[MessageInfo]) -> List[MessageInfo]:
    unique_msgs: List[MessageInfo] = []
    seen_links = set()
    
    for msg in messages:
        links = extract_links(msg.text)
        
        # Сначала проверяем по ссылкам - если ссылка уже была, пропускаем
        if links and any(link in seen_links for link in links):
            print(f"  Пропускаем дубликат по ссылке: {links[0]}")
            continue
        
        # Проверяем дубликаты по тексту
        duplicate = False
        for u in unique_msgs:
            if SequenceMatcher(None, msg.text, u.text).ratio() > SIMILARITY_THRESHOLD:
                print(f"  Пропускаем дубликат по тексту: {msg.text[:50]}...")
                duplicate = True
                break
        
        # Если не нашли дубликат по тексту, проверяем через LLM
        if not duplicate:
            for u in unique_msgs:
                try:
                    if await are_messages_duplicate(msg, u):
                        print(f"  Пропускаем дубликат по LLM: {msg.text[:50]}...")
                        duplicate = True
                        break
                except Exception as e:
                    print(f"  Ошибка при проверке дубликата через LLM: {e}")
                    # В случае ошибки LLM, считаем сообщения разными
                    continue
        
        # Проверяем, не была ли тема уже освещена в предыдущих саммари
        if not duplicate and ENABLE_SUMMARIES_DEDUPLICATION:
            try:
                if await is_message_covered_in_summaries(msg):
                    print(f"  Пропускаем сообщение, уже освещенное в саммари: {msg.text[:50]}...")
                    duplicate = True
            except Exception as e:
                print(f"  Ошибка при проверке покрытия в саммари: {e}")
                # В случае ошибки, считаем сообщение новым
                pass
        
        if not duplicate:
            unique_msgs.append(msg)
            seen_links.update(links)
            print(f"  Добавляем уникальное сообщение: {msg.text[:50]}...")
    
    return unique_msgs


async def remove_group_duplicates(messages: List[MessageInfo]) -> List[MessageInfo]:
    """Remove duplicates from group messages, checking against group summaries history."""
    unique_msgs: List[MessageInfo] = []
    seen_links = set()
    
    for msg in messages:
        links = extract_links(msg.text)
        
        # Сначала проверяем по ссылкам - если ссылка уже была, пропускаем
        if links and any(link in seen_links for link in links):
            print(f"  Пропускаем дубликат по ссылке: {links[0]}")
            continue
        
        # Проверяем дубликаты по тексту
        duplicate = False
        for u in unique_msgs:
            if SequenceMatcher(None, msg.text, u.text).ratio() > SIMILARITY_THRESHOLD:
                print(f"  Пропускаем дубликат по тексту: {msg.text[:50]}...")
                duplicate = True
                break
        
        # Если не нашли дубликат по тексту, проверяем через LLM
        if not duplicate:
            for u in unique_msgs:
                try:
                    if await are_messages_duplicate(msg, u):
                        print(f"  Пропускаем дубликат по LLM: {msg.text[:50]}...")
                        duplicate = True
                        break
                except Exception as e:
                    print(f"  Ошибка при проверке дубликата через LLM: {e}")
                    # В случае ошибки LLM, считаем сообщения разными
                    continue
        
        # Проверяем, не была ли тема уже освещена в предыдущих саммари групп
        if not duplicate and ENABLE_SUMMARIES_DEDUPLICATION:
            try:
                if await is_message_covered_in_group_summaries(msg):
                    print(f"  Пропускаем сообщение, уже освещенное в саммари групп: {msg.text[:50]}...")
                    duplicate = True
            except Exception as e:
                print(f"  Ошибка при проверке покрытия в саммари групп: {e}")
                # В случае ошибки, считаем сообщение новым
                pass
        
        if not duplicate:
            unique_msgs.append(msg)
            seen_links.update(links)
            print(f"  Добавляем уникальное сообщение из группы: {msg.text[:50]}...")
    
    return unique_msgs


async def main():
    # Start both clients
    await user_client.start()
    await bot_client.start(bot_token=BOT_TOKEN)
    
    try:
        # Опционально: автоматическое обнаружение похожих каналов
        # Раскомментируйте следующую строку для автоматического обнаружения
        # await discover_and_save_similar_channels()
        
        # Обработка каналов (как обычно)
        print("=== Обработка каналов ===")
        messages = await fetch_messages()
        print(f"Fetched {len(messages)} new messages from channels")
        
        if messages:
            # Apply NLP filtering to remove advertising
            filtered = []
            all_checked_messages = []  # Все сообщения, которые были проверены через is_nlp_related
            discovered_channels = set()  # Каналы, которые репостнули и прошли проверку
            
            for i, msg in enumerate(messages):
                print(f"Checking message {i+1}/{len(messages)}...")
                # Добавляем сообщение в список проверенных независимо от результата
                all_checked_messages.append(msg)
                
                if await is_nlp_related(msg.text):
                    filtered.append(msg)
                    # Если канал не из основного списка SOURCE_CHANNELS, добавляем его в обнаруженные
                    if msg.channel not in SOURCE_CHANNELS:
                        discovered_channels.add(msg.channel)
                    print(f"  ✓ Message {i+1} is NLP-related: {msg.text[:100]}; {msg.link}")
                else:
                    print(f"  ✗ Message {i+1} is not NLP-related (likely advertising): {msg.text[:100]}; "
                          f"{msg.channel}; {msg.link}")
            print(f"{len(filtered)} messages after NLP filter")
            
            # Сохраняем все проверенные сообщения в историю
            save_summarization_history(all_checked_messages)
            print(f"Saved {len(all_checked_messages)} checked messages to history")
            
            # Сохраняем новые обнаруженные каналы
            for channel in discovered_channels:
                save_discovered_channel(channel)
            if discovered_channels:
                print(f"Discovered {len(discovered_channels)} new channels: {', '.join(discovered_channels)}")
            
            if filtered:
                unique = await remove_duplicates(filtered)
                print(f"{len(unique)} messages after deduplication")
                
                if unique:
                    summary = await summarize_text(unique)
                    await user_client.send_message(TARGET_CHANNEL, summary, parse_mode='html')
                    print("Channel summary sent")
                    
                    # Сохраняем саммари в историю
                    channels = list(set(msg.channel for msg in unique))
                    summary_info = SummaryInfo(
                        content=summary,
                        date=datetime.now(timezone.utc),
                        message_count=len(unique),
                        channels=channels
                    )
                    save_summary_to_history(summary_info)
                    print(f"Channel summary saved to history (channels: {channels})")
                else:
                    print("No unique NLP messages found in channels")
            else:
                print("No new NLP-related messages found in channels")
        else:
            print("No new messages found in channels")
        
        # Обработка групп (раз в сутки)
        print("\n=== Обработка групп ===")
        if SOURCE_GROUPS and should_run_group_summarization():
            print("Starting daily group summarization...")
            
            group_messages = await fetch_group_messages()
            print(f"Fetched {len(group_messages)} new messages from groups")
            
            if group_messages:
                # Apply NLP filtering to remove advertising
                group_filtered = []
                all_checked_group_messages = []
                
                for i, msg in enumerate(group_messages):
                    print(f"Checking group message {i+1}/{len(group_messages)}...")
                    all_checked_group_messages.append(msg)
                    
                    if await is_nlp_related(msg.text):
                        group_filtered.append(msg)
                        print(f"  ✓ Group message {i+1} is NLP-related: {msg.text[:100]}; {msg.link}")
                    else:
                        print(f"  ✗ Group message {i+1} is not NLP-related: {msg.text[:100]}; {msg.link}")
                print(f"{len(group_filtered)} group messages after NLP filter")
                
                # Сохраняем все проверенные сообщения из групп в историю
                save_group_summarization_history(all_checked_group_messages)
                print(f"Saved {len(all_checked_group_messages)} checked group messages to history")
                
                if group_filtered:
                    unique_group = await remove_group_duplicates(group_filtered)
                    print(f"{len(unique_group)} group messages after deduplication")
                    
                    if unique_group:
                        group_summary = await summarize_group_text(unique_group)
                        await user_client.send_message(TARGET_CHANNEL, group_summary, parse_mode='html')
                        print("Group summary sent")
                        
                        # Сохраняем саммари групп в историю
                        groups = list(set(msg.channel for msg in unique_group))
                        group_summary_info = SummaryInfo(
                            content=group_summary,
                            date=datetime.now(timezone.utc),
                            message_count=len(unique_group),
                            channels=groups
                        )
                        save_group_summary_to_history(group_summary_info)
                        print(f"Group summary saved to history (groups: {groups})")
                        
                        # Обновляем время последнего запуска суммаризации групп
                        update_group_last_run()
                        print("Group summarization completed for today")
                    else:
                        print("No unique NLP messages found in groups")
                else:
                    print("No new NLP-related messages found in groups")
            else:
                print("No new messages found in groups")
        else:
            if SOURCE_GROUPS:
                print("Group summarization skipped (already run today)")
            else:
                print("No groups configured for summarization")
        
    finally:
        # Disconnect both clients
        await user_client.disconnect()
        await bot_client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
