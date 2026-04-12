import os
import json
import re
import logging
from datetime import datetime, timezone
from config import ABBREVIATIONS_FILE, DISCOVERED_CHANNELS_FILE

logger = logging.getLogger(__name__)


def _load_json_file(filepath: str, default: dict = None) -> dict:
    """Generic JSON file loader with error handling."""
    if default is None:
        default = {}
    if not os.path.exists(filepath):
        return default
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return default


def _save_json_file(filepath: str, data: dict, error_msg: str) -> bool:
    """Generic JSON file saver with error handling."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"{error_msg}: {e}")
        return False


def _now_iso() -> str:
    """Returns current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _load_channel_list(key: str) -> list[str]:
    """Generic function to load a channel list from the discovered channels file."""
    data = _load_json_file(DISCOVERED_CHANNELS_FILE, {})
    return data.get(key, [])


def _save_channel_list(key: str, channel_name: str, success_msg: str) -> None:
    """Generic function to add a channel to a list in the discovered channels file."""
    data = _load_json_file(DISCOVERED_CHANNELS_FILE, {
        'discovered_channels': [],
        'similar_channels': [],
        'banned_channels': [],
        'last_updated': ''
    })

    if channel_name not in data[key]:
        data[key].append(channel_name)
        data['last_updated'] = _now_iso()
        _save_json_file(DISCOVERED_CHANNELS_FILE, data, success_msg)


def load_channel_abbreviations() -> dict:
    """Загружает существующие аббревиатуры каналов из JSON файла."""
    return _load_json_file(ABBREVIATIONS_FILE, {}).get('channel_abbreviations', {})


def save_channel_abbreviation(channel_name: str, abbreviation: str) -> None:
    """Сохраняет новую аббревиатуру канала в JSON файл."""
    abbreviations = load_channel_abbreviations()
    abbreviations[channel_name] = abbreviation
    data = {"channel_abbreviations": abbreviations, "last_updated": _now_iso()}
    _save_json_file(ABBREVIATIONS_FILE, data, "Error saving channel abbreviation")


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


def load_discovered_channels() -> list[str]:
    """Загружает список обнаруженных каналов из файла."""
    return _load_channel_list('discovered_channels')


def load_similar_channels() -> list[str]:
    """Загружает список похожих каналов из файла."""
    return _load_channel_list('similar_channels')


def load_banned_channels() -> list[str]:
    """Загружает список заблокированных каналов из файла."""
    return _load_channel_list('banned_channels')


def save_discovered_channel(channel_name: str) -> None:
    """Сохраняет новый обнаруженный канал в JSON файл."""
    _save_channel_list('discovered_channels', channel_name, "Error saving discovered channel")


def save_similar_channel(channel_name: str) -> None:
    """Сохраняет новый похожий канал в JSON файл."""
    _save_channel_list('similar_channels', channel_name, "Error saving similar channel")


def save_banned_channel(channel_name: str) -> None:
    """Сохраняет новый заблокированный канал в JSON файл."""
    _save_channel_list('banned_channels', channel_name, "Error saving banned channel")