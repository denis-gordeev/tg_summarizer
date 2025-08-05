import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from typing import List, Set

from channel_manager import create_channel_abbreviation
from config import (
    GROUP_HISTORY_FILE,
    GROUP_LAST_RUN_FILE,
    GROUP_SUMMARIES_HISTORY_FILE,
    HISTORY_FILE,
    SUMMARIES_HISTORY_FILE,
    TARGET_CHANNEL,
)
from models import MessageInfo, SummaryInfo
from prompts import prompts
from utils import call_openai, extract_channels_from_abbreviations, extract_links


def load_summarization_history() -> Set[str]:
    """Загружает историю уже обработанных сообщений из файла."""
    if not os.path.exists(HISTORY_FILE):
        return set()

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Создаем уникальные идентификаторы для каждого сообщения
            processed_messages = set()
            for msg_data in data.get("processed_messages", []):
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
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                existing_data = data.get("processed_messages", [])

        # Добавляем новые сообщения
        new_messages = [msg.to_dict() for msg in messages]
        all_messages = existing_data + new_messages

        # Ограничиваем историю последними 1000 сообщениями
        if len(all_messages) > 1000:
            all_messages = all_messages[-1000:]

        # Сохраняем обновленную историю
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {"processed_messages": all_messages, "last_updated": datetime.now().isoformat()},
                f,
                ensure_ascii=False,
                indent=2,
            )

    except Exception as e:
        print(f"Ошибка при сохранении истории: {e}")


def load_summaries_history() -> List[SummaryInfo]:
    """Загружает историю созданных саммари из файла."""
    if not os.path.exists(SUMMARIES_HISTORY_FILE):
        return []

    try:
        with open(SUMMARIES_HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            summaries = []

            # Handle both old format (array) and new format (object with summaries key)
            if isinstance(data, list):
                # Old format: direct array of summary objects
                for summary_data in data:
                    summaries.append(SummaryInfo.from_dict(summary_data))
            else:
                # New format: object with summaries key
                for summary_data in data.get("summaries", []):
                    summaries.append(SummaryInfo.from_dict(summary_data))

            if not summaries:
                print("Пытаемся восстановить историю из канала...")
                return restore_summaries_from_channel_sync()
            summaries = sorted(summaries, key=lambda x: x.date)
            return summaries
    except Exception as e:
        print(f"Ошибка при загрузке истории саммари: {e}")
        print("Пытаемся восстановить историю из канала...")
        return restore_summaries_from_channel_sync()


async def restore_summaries_from_channel() -> List[SummaryInfo]:
    """Восстанавливает историю саммари из канала, читая сообщения за последнюю неделю."""
    try:
        from telegram_client import start_clients, user_client

        # Запускаем клиент если не запущен
        if not user_client.is_connected():
            await start_clients()

        # Получаем сообщения за последнюю неделю
        since = datetime.now(timezone.utc) - timedelta(days=7)
        summaries = []

        print(f"Читаем сообщения из канала {TARGET_CHANNEL} " f"за последнюю неделю...")

        async for msg in user_client.iter_messages(
            TARGET_CHANNEL, offset_date=None, min_id=0, reverse=False
        ):
            if msg.date < since:
                break

            if msg.message and msg.message.strip():
                # Все сообщения в канале - это саммари
                # Извлекаем каналы из ссылок и аббревиатур в сообщении
                from utils import extract_all_channels

                channels = extract_all_channels(msg.message)

                # Создаем SummaryInfo из сообщения
                summary_info = SummaryInfo(
                    content=msg.message,
                    date=msg.date,
                    message_count=0,  # Не можем определить количество исходных сообщений
                    channels=channels,  # Извлекаем каналы из ссылок и аббревиатур
                    message_id=msg.id,
                )
                summaries.append(summary_info)

        print(f"Восстановлено {len(summaries)} саммари из канала")
        summaries = sorted(summaries, key=lambda x: x.date)
        # Сохраняем восстановленную историю
        if summaries:
            # Создаем новый файл с восстановленными данными
            all_summaries = [summary.to_dict() for summary in summaries]

            with open(SUMMARIES_HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    {"summaries": all_summaries, "last_updated": datetime.now().isoformat()},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        return summaries

    except Exception as e:
        print(f"Ошибка при восстановлении истории из канала: {e}")
        return []


def restore_summaries_from_channel_sync() -> List[SummaryInfo]:
    """Синхронная обертка для восстановления истории саммари из канала."""
    try:
        # Проверяем, есть ли уже запущенный event loop
        try:
            asyncio.get_running_loop()
            # Если loop уже запущен, создаем новую задачу
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, restore_summaries_from_channel())
                return future.result()
        except RuntimeError:
            # Если нет активного event loop, создаем новый
            return asyncio.run(restore_summaries_from_channel())
    except Exception as e:
        print(f"Ошибка при синхронном восстановлении: {e}")
        return []


async def restore_group_summaries_from_channel() -> List[SummaryInfo]:
    """Восстанавливает историю групповых саммари из канала."""
    try:
        from telegram_client import start_clients, user_client

        # Запускаем клиент если не запущен
        if not user_client.is_connected():
            await start_clients()

        # Получаем сообщения за последнюю неделю
        since = datetime.now(timezone.utc) - timedelta(days=7)
        summaries = []

        print(f"Читаем групповые саммари из канала {TARGET_CHANNEL} " f"за последнюю неделю...")

        async for msg in user_client.iter_messages(
            TARGET_CHANNEL, offset_date=None, min_id=0, reverse=False
        ):
            if msg.date < since:
                break

            if msg.message and msg.message.strip():
                # Все сообщения в канале - это саммари
                # Извлекаем каналы из ссылок и аббревиатур в сообщении
                from utils import extract_all_channels

                channels = extract_all_channels(msg.message)

                # Создаем SummaryInfo из сообщения
                summary_info = SummaryInfo(
                    content=msg.message,
                    date=msg.date,
                    message_count=0,  # Не можем определить количество исходных сообщений
                    channels=channels,  # Извлекаем каналы из ссылок и аббревиатур
                    message_id=msg.id,
                )
                summaries.append(summary_info)

        print(f"Восстановлено {len(summaries)} групповых саммари из канала")

        summaries = sorted(summaries, key=lambda x: x.date)
        # Сохраняем восстановленную историю
        if summaries:
            # Создаем новый файл с восстановленными данными
            all_summaries = [summary.to_dict() for summary in summaries]

            with open(GROUP_SUMMARIES_HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    {"summaries": all_summaries, "last_updated": datetime.now().isoformat()},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        return summaries

    except Exception as e:
        print(f"Ошибка при восстановлении истории групповых саммари из канала: {e}")
        return []


def restore_group_summaries_from_channel_sync() -> List[SummaryInfo]:
    """Синхронная обертка для восстановления истории групповых саммари из канала."""
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(restore_group_summaries_from_channel())
    except RuntimeError:
        # Если нет активного event loop, создаем новый
        return asyncio.run(restore_group_summaries_from_channel())


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
        all_summaries_dict = [s.to_dict() for s in all_summaries if isinstance(s, SummaryInfo)]
        with open(SUMMARIES_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {"summaries": all_summaries_dict, "last_updated": datetime.now().isoformat()},
                f,
                ensure_ascii=False,
                indent=2,
            )

    except Exception as e:
        print(f"Ошибка при сохранении истории саммари: {e}")


def load_group_summarization_history() -> Set[str]:
    """Загружает историю уже обработанных сообщений из групп из файла."""
    if not os.path.exists(GROUP_HISTORY_FILE):
        return set()

    try:
        with open(GROUP_HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Создаем уникальные идентификаторы для каждого сообщения
            processed_messages = set()
            for msg_data in data.get("processed_messages", []):
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
            with open(GROUP_HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"processed_messages": [], "last_updated": ""}

        # Добавляем новые сообщения
        for msg in messages:
            data["processed_messages"].append(msg.to_dict())

        # Ограничиваем историю последними 1000 сообщениями
        if len(data["processed_messages"]) > 1000:
            data["processed_messages"] = data["processed_messages"][-1000:]

        data["last_updated"] = datetime.now().isoformat()

        # Сохраняем обновленные данные
        with open(GROUP_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"Ошибка при сохранении истории групп: {e}")


def load_group_summaries_history() -> List[SummaryInfo]:
    """Загружает историю созданных саммари из групп из файла."""
    if not os.path.exists(GROUP_SUMMARIES_HISTORY_FILE):
        return []

    try:
        with open(GROUP_SUMMARIES_HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            summaries = []
            for summary_data in data.get("summaries", []):
                summaries.append(SummaryInfo.from_dict(summary_data))
            return summaries
    except Exception as e:
        print(f"Ошибка при загрузке истории саммари групп: {e}")
        print("Пытаемся восстановить историю групповых саммари из канала...")
        return restore_group_summaries_from_channel_sync()


def save_group_summary_to_history(summary: SummaryInfo) -> None:
    """Сохраняет саммари из групп в историю."""
    try:
        # Загружаем существующие данные
        if os.path.exists(GROUP_SUMMARIES_HISTORY_FILE):
            with open(GROUP_SUMMARIES_HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"summaries": [], "last_updated": ""}

        # Добавляем новое саммари
        data["summaries"].append(summary.to_dict())

        # Ограничиваем историю последними 100 саммари
        if len(data["summaries"]) > 100:
            data["summaries"] = data["summaries"][-100:]

        data["last_updated"] = datetime.now().isoformat()

        # Сохраняем обновленные данные
        with open(GROUP_SUMMARIES_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"Ошибка при сохранении саммари групп в историю: {e}")


def should_run_group_summarization() -> bool:
    """Проверяет, нужно ли запускать суммаризацию групп (раз в сутки)."""
    if not os.path.exists(GROUP_LAST_RUN_FILE):
        return False

    try:
        with open(GROUP_LAST_RUN_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            last_run_str = data.get("last_run", "")
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
            "last_run": datetime.now(timezone.utc).isoformat(),
            "last_updated": datetime.now().isoformat(),
        }

        with open(GROUP_LAST_RUN_FILE, "w", encoding="utf-8") as f:
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
    summaries = sorted(summaries, key=lambda x: x.date)
    if not summaries:
        return ""

    # Фильтруем саммари за последние N дней
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    recent_summaries = [s for s in summaries if s.date >= cutoff_date]

    if not recent_summaries:
        return ""

    return "\n\n".join([m.content for m in recent_summaries])


async def find_relevant_summary_for_update(msg: MessageInfo, is_group: bool = False) -> SummaryInfo:
    """
    Находит наиболее подходящее существующее саммари для обновления.
    Возвращает None, если подходящее саммари не найдено.
    """
    if is_group:
        summaries = load_group_summaries_history()
    else:
        summaries = load_summaries_history()

    if not summaries:
        return None

    # Берем последние 3 саммари для поиска
    recent_summaries = summaries[-50:]

    # Создаем контекст для поиска
    summaries_text = ""
    for i, summary in enumerate(recent_summaries, 1):
        summaries_text += f"Саммари {i}:\n{summary.content}\n\n"

    user_content = f"""Существующие саммари:
{summaries_text}

Новое сообщение:
{msg.text}

В каком саммари (1, 2, 3) лучше всего добавить ссылку на это сообщение? 
Если ни одно не подходит, ответьте "НЕТ".
Отвечайте только номером (1, 2, 3) или "НЕТ"."""

    try:
        result = await call_openai(prompts.FIND_RELEVANT_SUMMARY_PROMPT, user_content, max_tokens=5)
        result = result.strip().upper()

        if result.isdigit():
            index = int(result) - 1
            if 0 <= index < len(recent_summaries):
                return recent_summaries[index]
    except Exception as e:
        print(f"Ошибка при поиске подходящего саммари: {e}")

    return None


async def update_existing_summary(
    summary: SummaryInfo, new_message: MessageInfo, is_group: bool = False
) -> SummaryInfo:
    """
    Обновляет существующее саммари, добавляя блок "Другие ссылки:" с новой ссылкой.
    """
    # Создаем ссылку для нового сообщения
    links = extract_links(new_message.text)
    telegram_link = new_message.get_telegram_link()
    channel_abbr = create_channel_abbreviation(new_message.channel)

    if links:
        new_link = (
            f'<a href="{links[0]}">[новое]</a> ' f'<a href="{telegram_link}">[{channel_abbr}]</a>'
        )
    else:
        new_link = f'<a href="{telegram_link}">[{channel_abbr}]</a>'

    # Ищем подходящее место для вставки блока "Другие ссылки:"
    # Обычно это в конце саммари или после последнего блока с ссылками
    content = summary.content

    # Проверяем, есть ли уже блок "Другие ссылки:"
    if "Другие ссылки:" in content:
        # Если есть, добавляем новую ссылку к существующему блоку
        lines = content.split("\n")
        updated_lines = []
        for line in lines:
            if line.strip().startswith("Другие ссылки:"):
                # Добавляем новую ссылку к существующему блоку
                updated_lines.append(line + f", {new_link}")
            else:
                updated_lines.append(line)
        updated_content = "\n".join(updated_lines)
    else:
        # Если нет, добавляем новый блок в конец
        updated_content = content + f"\n\nДругие ссылки: {new_link}"

    # Создаем обновленное саммари
    updated_channels = (
        summary.channels + [new_message.channel]
        if new_message.channel not in summary.channels
        else summary.channels
    )

    updated_summary = SummaryInfo(
        content=updated_content,
        date=summary.date,
        message_count=summary.message_count + 1,
        channels=updated_channels,
    )

    return updated_summary


async def save_updated_summary(
    original_summary: SummaryInfo, updated_summary: SummaryInfo, is_group: bool = False
) -> None:
    """
    Сохраняет обновленное саммари, заменяя оригинальное в истории.
    Также редактирует сообщение в канале, если есть message_id.
    """
    from telegram_client import edit_message_in_target_channel
    
    if is_group:
        summaries = load_group_summaries_history()
        # Находим индекс оригинального саммари
        for i, summary in enumerate(summaries):
            if (
                summary.content == original_summary.content
                and summary.date == original_summary.date
                and summary.message_count == original_summary.message_count
            ):
                summaries[i] = updated_summary
                break

        # Сохраняем обновленную историю
        with open(GROUP_SUMMARIES_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in summaries], f, ensure_ascii=False, indent=2)
    else:
        summaries = load_summaries_history()
        # Находим индекс оригинального саммари
        for i, summary in enumerate(summaries):
            if summary.content == original_summary.content:
                summaries[i] = updated_summary
                print("updated_summary:", "=" * 100, "\n", updated_summary, "\n", "=" * 100, "\n")
                break

        # Сохраняем обновленную историю в текущем формате (массив)
        with open(SUMMARIES_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in summaries], f, ensure_ascii=False, indent=2)
    
    # Редактируем сообщение в канале, если есть message_id
    if original_summary.message_id:
        try:
            await edit_message_in_target_channel(original_summary.message_id, updated_summary.content)
            print(f"Сообщение {original_summary.message_id} обновлено в канале")
        except Exception as e:
            print(f"Ошибка при редактировании сообщения {original_summary.message_id}: {e}")
