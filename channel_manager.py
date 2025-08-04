import os
import json
import re
from datetime import datetime
from config import ABBREVIATIONS_FILE, DISCOVERED_CHANNELS_FILE


def load_channel_abbreviations() -> dict:
    """Загружает существующие аббревиатуры каналов из JSON файла."""
    abbreviations_file = ABBREVIATIONS_FILE
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
    abbreviations_file = ABBREVIATIONS_FILE
    try:
        abbreviations = load_channel_abbreviations()
        abbreviations[channel_name] = abbreviation
        data = {"channel_abbreviations": abbreviations, "last_updated": datetime.now().isoformat()}
        
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


def load_discovered_channels() -> list[str]:
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


def load_similar_channels() -> list[str]:
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


def load_banned_channels() -> list[str]:
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
            data = {
                'discovered_channels': [], 
                'similar_channels': [], 
                'last_updated': ''
            }
        
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
            data = {
                'discovered_channels': [], 
                'similar_channels': [], 
                'banned_channels': [], 
                'last_updated': ''
            }
        
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
            data = {
                'discovered_channels': [], 
                'similar_channels': [], 
                'banned_channels': [], 
                'last_updated': ''
            }
        
        # Проверяем, нет ли уже такого канала
        if channel_name not in data['banned_channels']:
            data['banned_channels'].append(channel_name)
            data['last_updated'] = datetime.now().isoformat()
            
            # Сохраняем обновленные данные
            with open(DISCOVERED_CHANNELS_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
    except Exception as e:
        print(f"Ошибка при сохранении заблокированного канала: {e}") 