import os
import json
from datetime import datetime, timezone, timedelta
from typing import List, Set
from models import MessageInfo, SummaryInfo
from config import (
    HISTORY_FILE, SUMMARIES_HISTORY_FILE, GROUP_HISTORY_FILE,
    GROUP_SUMMARIES_HISTORY_FILE, GROUP_LAST_RUN_FILE
)


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
            
            # Handle both old format (array) and new format (object with summaries key)
            if isinstance(data, list):
                # Old format: direct array of summary objects
                for summary_data in data:
                    summaries.append(SummaryInfo.from_dict(summary_data))
            else:
                # New format: object with summaries key
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
        existing_summaries = load_summaries_history()
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
        return False
    
    try:
        with open(GROUP_LAST_RUN_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            last_run_str = data.get('last_run', '')
            if not last_run_str:
                return True
            
            last_run = datetime.fromisoformat(last_run_str)
            now = datetime.now(timezone.utc)
            
            time_since_last_run = (now - last_run).total_seconds()
            print(f"Last run: {last_run}; now: {now}; time since last run: {time_since_last_run}")
            # Проверяем, прошло ли больше 24 часов с последнего запуска
            return time_since_last_run > 24 * 60 * 60
            
    except Exception as e:
        print(f"Ошибка при проверке времени последнего запуска групп: {e}")
        return False


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
        import re
        clean_content = re.sub(r'<[^>]+>', '', summary.content)
        # Берем первые 500 символов для контекста
        context_parts.append(
            f"Саммари {i} ({summary.date.strftime('%Y-%m-%d')}):\n{clean_content[:500]}..."
        )
    
    return "\n\n".join(context_parts) 