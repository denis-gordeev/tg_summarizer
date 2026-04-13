import asyncio
import hashlib
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Set, Optional, Callable, TypeVar

from channel_manager import create_channel_abbreviation
from config import (
    GROUP_HISTORY_FILE,
    GROUP_LAST_RUN_FILE,
    GROUP_SUMMARIES_HISTORY_FILE,
    HISTORY_FILE,
    SUMMARIES_HISTORY_FILE,
    TARGET_CHANNEL,
    MAX_CHANNEL_HISTORY_MESSAGES,
    MAX_CHANNEL_SUMMARIES,
    MAX_GROUP_HISTORY_MESSAGES,
    MAX_GROUP_SUMMARIES,
    GROUP_SUMMARIZATION_INTERVAL_SECONDS,
    RESTORE_HISTORY_DAYS,
)
from models import MessageInfo, SummaryInfo
from prompts import prompts
from utils import call_openai, extract_links, load_json_file, save_json_file, now_iso

logger = logging.getLogger(__name__)

T = TypeVar("T")


def load_summarization_history() -> Set[str]:
    """Загружает историю уже обработанных сообщений из файла."""
    data = load_json_file(HISTORY_FILE, {"processed_messages": []})
    processed_messages = set()
    for msg_data in data.get("processed_messages", []):
        msg = MessageInfo.from_dict(msg_data)
        text_hash = hashlib.sha256(msg.text.encode()).hexdigest()[:16]
        msg_id = f"{msg.channel}_{msg.message_id}_{text_hash}"
        processed_messages.add(msg_id)
    return processed_messages


def save_summarization_history(messages: List[MessageInfo]) -> None:
    """Сохраняет обработанные сообщения в историю."""
    existing_data = load_json_file(HISTORY_FILE, {"processed_messages": []})
    existing_messages = existing_data.get("processed_messages", [])

    new_messages = [msg.to_dict() for msg in messages]
    all_messages = existing_messages + new_messages

    if len(all_messages) > MAX_CHANNEL_HISTORY_MESSAGES:
        all_messages = all_messages[-MAX_CHANNEL_HISTORY_MESSAGES:]

    data = {"processed_messages": all_messages, "last_updated": now_iso()}
    save_json_file(HISTORY_FILE, data, "Error saving summarization history")


def _parse_summaries_from_data(data: Any) -> List[SummaryInfo]:
    """Parses summary data from JSON format (handles both old and new formats)."""
    summaries = []
    if isinstance(data, list):
        for summary_data in data:
            summaries.append(SummaryInfo.from_dict(summary_data))
    else:
        for summary_data in data.get("summaries", []):
            summaries.append(SummaryInfo.from_dict(summary_data))
    return summaries


def load_summaries_history() -> List[SummaryInfo]:
    """Загружает историю созданных саммари из файла."""
    data = load_json_file(SUMMARIES_HISTORY_FILE, None)
    if data is None:
        return []

    summaries = _parse_summaries_from_data(data)
    if not summaries:
        logger.info("No summaries found in history file, attempting channel restore...")
        return restore_summaries_from_channel_sync()

    summaries = sorted(summaries, key=lambda x: x.date)
    return summaries


async def restore_summaries_from_channel() -> List[SummaryInfo]:
    """Восстанавливает историю саммари из канала, читая сообщения за последнюю неделю."""
    try:
        import telegram_client as tg
        # Запускаем клиент если не запущен
        if tg.user_client is None or not tg.user_client.is_connected():
            logger.info("Starting Telegram client...")
            await tg.start_clients()
            logger.info("Telegram client started")

        # Получаем сообщения за последнюю неделю
        since = datetime.now(timezone.utc) - timedelta(days=RESTORE_HISTORY_DAYS)
        logger.debug(f"Restoring summaries since {since}")
        summaries = []

        logger.info(f"Reading messages from channel {TARGET_CHANNEL} for the last {RESTORE_HISTORY_DAYS} days...")

        async for msg in tg.user_client.iter_messages(
            TARGET_CHANNEL, offset_date=None, min_id=0, reverse=False
        ):
            if msg.date < since:
                break
            logger.debug(f"Processing message: id={msg.id}, date={msg.date}")
            if msg.message and msg.message.strip():
                # Все сообщения в канале - это саммари
                # Извлекаем каналы из ссылок и аббревиатур в сообщении
                from utils import extract_all_channels

                channels = extract_all_channels(msg.text)

                # Создаем SummaryInfo из сообщения
                summary_info = SummaryInfo(
                    content=msg.text,
                    date=msg.date,
                    message_count=0,  # Не можем определить количество исходных сообщений
                    channels=channels,  # Извлекаем каналы из ссылок и аббревиатур
                    message_id=msg.id,
                )
                summaries.append(summary_info)

        logger.info(f"Restored {len(summaries)} summaries from channel")
        summaries = sorted(summaries, key=lambda x: x.date)
        # Сохраняем восстановленную историю
        if summaries:
            all_summaries = [summary.to_dict() for summary in summaries]
            data = {"summaries": all_summaries, "last_updated": now_iso()}
            save_json_file(SUMMARIES_HISTORY_FILE, data, "Error saving restored summary history")

        return summaries

    except Exception as e:
        logger.error(f"Error restoring history from channel: {e}")
        return []


def restore_summaries_from_channel_sync() -> List[SummaryInfo]:
    """Синхронная обёртка для restore_summaries_from_channel."""
    import telegram_client as tg
    tg_loop = getattr(tg, "clients_loop", None)
    
    try:
        current_loop = asyncio.get_running_loop()
        # We're inside a running event loop
        if tg_loop and tg_loop.is_running():
            if tg_loop is current_loop:
                logger.debug("Restore: same event loop, skipping to avoid deadlock")
                return []
            logger.debug("Restore: clients running in separate loop")
            future = asyncio.run_coroutine_threadsafe(
                restore_summaries_from_channel(), tg_loop
            )
            try:
                return future.result(timeout=float(os.getenv("RESTORE_TIMEOUT_SEC", "30")))
            except Exception as wait_err:
                logger.error(f"Restore: timeout/wait error: {wait_err}")
                return []
        logger.debug("Restore: clients not running in separate loop")
        return []
    except RuntimeError:
        # No event loop in current thread
        if tg_loop and tg_loop.is_running():
            logger.debug("Restore: clients running in separate loop (RuntimeError path)")
            future = asyncio.run_coroutine_threadsafe(
                restore_summaries_from_channel(), tg_loop
            )
            try:
                return future.result(timeout=float(os.getenv("RESTORE_TIMEOUT_SEC", "30")))
            except Exception as wait_err:
                logger.error(f"Restore: timeout/wait error: {wait_err}")
                return []
        return asyncio.run(restore_summaries_from_channel())
    except Exception as e:
        logger.error(f"Error restoring history from channel: {e}")
        return []


async def restore_group_summaries_from_channel() -> List[SummaryInfo]:
    """Восстанавливает историю групповых саммари из канала."""
    try:
        import telegram_client as tg
        # Запускаем клиент если не запущен
        if tg.user_client is None or not tg.user_client.is_connected():
            await tg.start_clients()

        # Получаем сообщения за последнюю неделю
        since = datetime.now(timezone.utc) - timedelta(days=RESTORE_HISTORY_DAYS)
        summaries = []

        logger.info(f"Reading group summaries from channel {TARGET_CHANNEL} for the last {RESTORE_HISTORY_DAYS} days...")

        async for msg in tg.user_client.iter_messages(
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

        logger.info(f"Restored {len(summaries)} group summaries from channel")

        summaries = sorted(summaries, key=lambda x: x.date)
        # Сохраняем восстановленную историю
        if summaries:
            all_summaries = [summary.to_dict() for summary in summaries]
            data = {"summaries": all_summaries, "last_updated": now_iso()}
            save_json_file(GROUP_SUMMARIES_HISTORY_FILE, data, "Error saving restored group summary history")

        return summaries

    except Exception as e:
        logger.error(f"Error restoring group summary history from channel: {e}")
        return []


def restore_group_summaries_from_channel_sync() -> List[SummaryInfo]:
    """Синхронная обертка для восстановления истории групповых саммари из канала."""
    import telegram_client as tg
    tg_loop = getattr(tg, "clients_loop", None)
    try:
        asyncio.get_running_loop()
        if tg_loop and tg_loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                restore_group_summaries_from_channel(), tg_loop
            )
            return future.result()
        logger.debug("Restore group summaries: clients not running in separate loop")
        return []
    except RuntimeError:
        if tg_loop and tg_loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                restore_group_summaries_from_channel(), tg_loop
            )
            return future.result()
        return asyncio.run(restore_group_summaries_from_channel())
    except Exception as e:
        logger.error(f"Error restoring group summary history from channel: {e}")
        return []


def save_summary_to_history(summary: SummaryInfo) -> None:
    """Сохраняет новое саммари в историю."""
    existing_summaries = load_summaries_history()
    all_summaries = existing_summaries + [summary]

    if len(all_summaries) > MAX_CHANNEL_SUMMARIES:
        all_summaries = all_summaries[-MAX_CHANNEL_SUMMARIES:]

    all_summaries_dict = [s.to_dict() for s in all_summaries if isinstance(s, SummaryInfo)]
    data = {"summaries": all_summaries_dict, "last_updated": now_iso()}
    save_json_file(SUMMARIES_HISTORY_FILE, data, "Error saving summary history")


def load_group_summarization_history() -> Set[str]:
    """Загружает историю уже обработанных сообщений из групп из файла."""
    data = load_json_file(GROUP_HISTORY_FILE, {"processed_messages": []})
    processed_messages = set()
    for msg_data in data.get("processed_messages", []):
        msg = MessageInfo.from_dict(msg_data)
        text_hash = hashlib.sha256(msg.text.encode()).hexdigest()[:16]
        msg_id = f"{msg.channel}_{msg.message_id}_{text_hash}"
        processed_messages.add(msg_id)
    return processed_messages


def save_group_summarization_history(messages: List[MessageInfo]) -> None:
    """Сохраняет историю обработанных сообщений из групп в файл."""
    data = load_json_file(GROUP_HISTORY_FILE, {"processed_messages": [], "last_updated": ""})

    for msg in messages:
        data["processed_messages"].append(msg.to_dict())

    if len(data["processed_messages"]) > MAX_GROUP_HISTORY_MESSAGES:
        data["processed_messages"] = data["processed_messages"][-MAX_GROUP_HISTORY_MESSAGES:]

    data["last_updated"] = now_iso()
    save_json_file(GROUP_HISTORY_FILE, data, "Error saving group summarization history")


def load_group_summaries_history() -> List[SummaryInfo]:
    """Загружает историю созданных саммари из групп из файла."""
    data = load_json_file(GROUP_SUMMARIES_HISTORY_FILE, None)
    if data is None:
        return []

    summaries = _parse_summaries_from_data(data)
    if not summaries:
        return []
    return summaries


def save_group_summary_to_history(summary: SummaryInfo) -> None:
    """Сохраняет саммари из групп в историю."""
    data = load_json_file(GROUP_SUMMARIES_HISTORY_FILE, {"summaries": [], "last_updated": ""})

    data["summaries"].append(summary.to_dict())

    if len(data["summaries"]) > MAX_GROUP_SUMMARIES:
        data["summaries"] = data["summaries"][-MAX_GROUP_SUMMARIES:]

    data["last_updated"] = now_iso()
    save_json_file(GROUP_SUMMARIES_HISTORY_FILE, data, "Error saving group summary history")


def should_run_group_summarization() -> bool:
    """Проверяет, нужно ли запускать суммаризацию групп (раз в сутки)."""
    if not os.path.exists(GROUP_LAST_RUN_FILE):
        return True

    data = load_json_file(GROUP_LAST_RUN_FILE, {})
    last_run_str = data.get("last_run", "")
    if not last_run_str:
        return True

    try:
        last_run = datetime.fromisoformat(last_run_str)
        if last_run.tzinfo is None:
            last_run = last_run.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        time_since_last_run = (now - last_run).total_seconds()
        logger.debug(f"Last run: {last_run}; now: {now}; time since last run: {time_since_last_run}")

        # Check if more than 24 hours have passed
        return time_since_last_run > GROUP_SUMMARIZATION_INTERVAL_SECONDS
    except (ValueError, TypeError) as e:
        logger.error(f"Error parsing group last run time: {e}")
        return False


def update_group_last_run() -> None:
    """Обновляет время последнего запуска суммаризации групп."""
    data = {
        "last_run": now_iso(),
        "last_updated": now_iso(),
    }
    save_json_file(GROUP_LAST_RUN_FILE, data, "Error updating group last run")


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


async def find_relevant_summary_for_update(
    msg: MessageInfo, is_group: bool = False
) -> Optional[SummaryInfo]:
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
        save_json_file(
            GROUP_SUMMARIES_HISTORY_FILE,
            [s.to_dict() for s in summaries],
            "Error saving updated group summary history",
        )
    else:
        summaries = load_summaries_history()
        # Находим индекс оригинального саммари
        for i, summary in enumerate(summaries):
            if summary.content == original_summary.content:
                summaries[i] = updated_summary
                logger.debug("updated_summary: %s", updated_summary)
                break

        # Сохраняем обновленную историю
        save_json_file(
            SUMMARIES_HISTORY_FILE,
            [s.to_dict() for s in summaries],
            "Error saving updated summary history",
        )

    # Редактируем сообщение в канале, если есть message_id
    if original_summary.message_id:
        try:
            await edit_message_in_target_channel(
                original_summary.message_id, updated_summary.content
            )
            logger.info("Message %s updated in channel", original_summary.message_id)
        except Exception as e:
            logger.error("Error updating message %s in channel: %s", original_summary.message_id, e)
