import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

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
    COVERAGE_CHECK_MAX_SUMMARIES,
    COVERAGE_CHECK_MAX_CHARS_PER_SUMMARY,
    UPDATE_MATCH_MAX_SUMMARIES,
    UPDATE_MATCH_MAX_CHARS_PER_SUMMARY,
    UPDATE_SUMMARY_MAX_TOKENS,
)
from models import MessageInfo, SummaryInfo
from prompts import prompts
from utils import call_openai, extract_links, load_json_file, save_json_file, now_iso, text_hash

logger = logging.getLogger(__name__)

_cache: Dict[str, Any] = {}


def _cache_key(filepath: str) -> str:
    return f"file:{filepath}"


def invalidate_cache(filepath: str = None) -> None:
    if filepath:
        _cache.pop(_cache_key(filepath), None)
    else:
        _cache.clear()


def _load_processed_messages(filepath: str) -> Set[str]:
    """Load processed message IDs from a history file."""
    data = load_json_file(filepath, {"processed_messages": []})
    processed_messages = set()
    for msg_data in data.get("processed_messages", []):
        msg = MessageInfo.from_dict(msg_data)
        msg_id = f"{msg.channel}_{msg.message_id}_{text_hash(msg.text)}"
        processed_messages.add(msg_id)
    return processed_messages


def _save_processed_messages(
    filepath: str, messages: List[MessageInfo], max_messages: int, error_msg: str
) -> None:
    """Append messages to a processed-messages history file, truncating to max_messages."""
    data = load_json_file(filepath, {"processed_messages": []})
    existing = data.get("processed_messages", [])
    for msg in messages:
        existing.append(msg.to_dict())
    if len(existing) > max_messages:
        existing = existing[-max_messages:]
    data = {"processed_messages": existing, "last_updated": now_iso()}
    save_json_file(filepath, data, error_msg)
    invalidate_cache(filepath)


def load_summarization_history() -> Set[str]:
    """Загружает историю уже обработанных сообщений из файла."""
    return _load_processed_messages(HISTORY_FILE)


def save_summarization_history(messages: List[MessageInfo]) -> None:
    """Сохраняет обработанные сообщения в историю."""
    _save_processed_messages(
        HISTORY_FILE, messages, MAX_CHANNEL_HISTORY_MESSAGES,
        "Error saving summarization history",
    )


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
    cache_key = _cache_key(SUMMARIES_HISTORY_FILE)
    if cache_key in _cache:
        return _cache[cache_key]

    data = load_json_file(SUMMARIES_HISTORY_FILE, None)
    if data is None:
        logger.info("History file missing or corrupt, attempting channel restore...")
        summaries = restore_summaries_from_channel_sync()
        summaries = sorted(summaries, key=lambda x: x.date)
        _cache[cache_key] = summaries
        return summaries

    summaries = _parse_summaries_from_data(data)
    summaries = sorted(summaries, key=lambda x: x.date)
    _cache[cache_key] = summaries
    return summaries


async def _restore_summaries_from_channel(
    history_file: str, label: str
) -> List[SummaryInfo]:
    """Restore summaries from the target Telegram channel for the last N days."""
    try:
        import telegram_client as tg
        if tg.user_client is None or not tg.user_client.is_connected():
            logger.info("Starting Telegram client...")
            await tg.start_clients()

        since = datetime.now(timezone.utc) - timedelta(days=RESTORE_HISTORY_DAYS)
        logger.debug("Restoring %s since %s", label, since)
        summaries = []

        logger.info("Reading %s from channel %s for the last %d days...", label, TARGET_CHANNEL, RESTORE_HISTORY_DAYS)

        async for msg in tg.user_client.iter_messages(
            TARGET_CHANNEL, offset_date=None, min_id=0, reverse=False
        ):
            if msg.date < since:
                break
            logger.debug("Processing message: id=%d, date=%s", msg.id, msg.date)
            if msg.message and msg.message.strip():
                from utils import extract_all_channels

                channels = extract_all_channels(msg.text)

                summary_info = SummaryInfo(
                    content=msg.text,
                    date=msg.date,
                    message_count=0,
                    channels=channels,
                    message_id=msg.id,
                )
                summaries.append(summary_info)

        logger.info("Restored %d %s from channel", len(summaries), label)
        summaries = sorted(summaries, key=lambda x: x.date)
        if summaries:
            all_summaries = [summary.to_dict() for summary in summaries]
            data = {"summaries": all_summaries, "last_updated": now_iso()}
            save_json_file(history_file, data, f"Error saving restored {label}")
            invalidate_cache(history_file)

        return summaries

    except Exception as e:
        logger.error("Error restoring %s from channel: %s", label, e)
        return []


async def restore_summaries_from_channel() -> List[SummaryInfo]:
    return await _restore_summaries_from_channel(SUMMARIES_HISTORY_FILE, "summaries")


async def restore_group_summaries_from_channel() -> List[SummaryInfo]:
    return await _restore_summaries_from_channel(GROUP_SUMMARIES_HISTORY_FILE, "group summaries")


def _run_async_with_loop(coro):
    """Run an async coroutine from sync context, handling event loop edge cases.

    In Lambda, `asyncio.run()` in the handler creates a fresh loop each time,
    and `stop_clients()` resets `clients_loop`. So the common cases are:
    1. No running loop → use `asyncio.run()`
    2. Running loop (e.g. called from async `fetch_messages`) → schedule
       via `run_coroutine_threadsafe` on the Telegram client's loop.
    """
    import telegram_client as tg
    tg_loop = getattr(tg, "clients_loop", None)

    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    if current_loop is None:
        if tg_loop and tg_loop.is_running():
            future = asyncio.run_coroutine_threadsafe(coro, tg_loop)
            try:
                return future.result(timeout=float(os.getenv("RESTORE_TIMEOUT_SEC", "30")))
            except Exception as e:
                logger.error("Restore: timeout/wait error: %s", e)
                return []
        try:
            return asyncio.run(coro)
        except Exception as e:
            logger.error("Restore: error running coroutine: %s", e)
            return []

    if tg_loop and tg_loop.is_running() and tg_loop is not current_loop:
        future = asyncio.run_coroutine_threadsafe(coro, tg_loop)
        try:
            return future.result(timeout=float(os.getenv("RESTORE_TIMEOUT_SEC", "30")))
        except Exception as e:
            logger.error("Restore: timeout/wait error: %s", e)
            return []

    import threading
    result_box = [None]
    exception_box = [None]

    def _run_in_thread():
        try:
            result_box[0] = asyncio.run(coro)
        except Exception as e:
            exception_box[0] = e

    t = threading.Thread(target=_run_in_thread, daemon=True)
    t.start()
    t.join(timeout=float(os.getenv("RESTORE_TIMEOUT_SEC", "30")))
    if t.is_alive():
        logger.error("Restore: timed out in background thread")
        return []
    if exception_box[0]:
        logger.error("Restore: error in background thread: %s", exception_box[0])
        return []
    return result_box[0] or []


def restore_summaries_from_channel_sync() -> List[SummaryInfo]:
    return _run_async_with_loop(restore_summaries_from_channel())


def restore_group_summaries_from_channel_sync() -> List[SummaryInfo]:
    return _run_async_with_loop(restore_group_summaries_from_channel())


def save_summary_to_history(summary: SummaryInfo) -> None:
    """Сохраняет новое саммари в историю."""
    existing_summaries = load_summaries_history()
    all_summaries = existing_summaries + [summary]

    if len(all_summaries) > MAX_CHANNEL_SUMMARIES:
        all_summaries = all_summaries[-MAX_CHANNEL_SUMMARIES:]

    all_summaries_dict = [s.to_dict() for s in all_summaries if isinstance(s, SummaryInfo)]
    data = {"summaries": all_summaries_dict, "last_updated": now_iso()}
    save_json_file(SUMMARIES_HISTORY_FILE, data, "Error saving summary history")
    invalidate_cache(SUMMARIES_HISTORY_FILE)


def load_group_summarization_history() -> Set[str]:
    """Загружает историю уже обработанных сообщений из групп из файла."""
    return _load_processed_messages(GROUP_HISTORY_FILE)


def save_group_summarization_history(messages: List[MessageInfo]) -> None:
    """Сохраняет историю обработанных сообщений из групп в файл."""
    _save_processed_messages(
        GROUP_HISTORY_FILE, messages, MAX_GROUP_HISTORY_MESSAGES,
        "Error saving group summarization history",
    )


def load_group_summaries_history() -> List[SummaryInfo]:
    """Загружает историю созданных саммари из групп из файла."""
    cache_key = _cache_key(GROUP_SUMMARIES_HISTORY_FILE)
    if cache_key in _cache:
        return _cache[cache_key]

    data = load_json_file(GROUP_SUMMARIES_HISTORY_FILE, None)
    if data is None:
        logger.info("Group history file missing or corrupt, attempting channel restore...")
        summaries = restore_group_summaries_from_channel_sync()
        summaries = sorted(summaries, key=lambda x: x.date)
        _cache[cache_key] = summaries
        return summaries

    summaries = _parse_summaries_from_data(data)
    if not summaries:
        return []
    _cache[cache_key] = summaries
    return summaries


def save_group_summary_to_history(summary: SummaryInfo) -> None:
    """Сохраняет саммари из групп в историю."""
    data = load_json_file(GROUP_SUMMARIES_HISTORY_FILE, {"summaries": [], "last_updated": ""})

    data["summaries"].append(summary.to_dict())

    if len(data["summaries"]) > MAX_GROUP_SUMMARIES:
        data["summaries"] = data["summaries"][-MAX_GROUP_SUMMARIES:]

    data["last_updated"] = now_iso()
    save_json_file(GROUP_SUMMARIES_HISTORY_FILE, data, "Error saving group summary history")
    invalidate_cache(GROUP_SUMMARIES_HISTORY_FILE)


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
        logger.debug("Last run: %s; now: %s; time since last run: %.0f seconds", last_run, now, time_since_last_run)

        # Check if more than 24 hours have passed
        return time_since_last_run > GROUP_SUMMARIZATION_INTERVAL_SECONDS
    except (ValueError, TypeError) as e:
        logger.error("Error parsing group last run time: %s", e)
        return False


def update_group_last_run() -> None:
    """Обновляет время последнего запуска суммаризации групп."""
    data = {
        "last_run": now_iso(),
        "last_updated": now_iso(),
    }
    save_json_file(GROUP_LAST_RUN_FILE, data, "Error updating group last run")


def _get_recent_summaries_context(
    summaries: list, days: int, max_summaries: int, max_chars: int,
) -> str:
    """Shared helper to build truncated context from recent summaries."""
    if not summaries:
        return ""
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    recent_summaries = [s for s in summaries if s.date >= cutoff_date]
    if not recent_summaries:
        return ""
    recent_summaries = recent_summaries[-max_summaries:]
    context_parts = []
    for summary in recent_summaries:
        truncated = summary.content[:max_chars]
        context_parts.append(f"Дата: {summary.date.strftime('%Y-%m-%d')}")
        context_parts.append(f"Содержание: {truncated}")
        context_parts.append("---")
    return "\n".join(context_parts)


def get_recent_summaries_context(days: int = 3) -> str:
    """Возвращает контекст последних саммари для дедупликации."""
    summaries = load_summaries_history()
    return _get_recent_summaries_context(
        summaries, days, COVERAGE_CHECK_MAX_SUMMARIES, COVERAGE_CHECK_MAX_CHARS_PER_SUMMARY,
    )


def get_recent_group_summaries_context(days: int = 7) -> str:
    """Получает контекст последних саммари из групп для дедупликации."""
    summaries = load_group_summaries_history()
    return _get_recent_summaries_context(
        summaries, days, COVERAGE_CHECK_MAX_SUMMARIES, COVERAGE_CHECK_MAX_CHARS_PER_SUMMARY,
    )


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

    recent_summaries = summaries[-UPDATE_MATCH_MAX_SUMMARIES:]

    summaries_text = ""
    for i, summary in enumerate(recent_summaries, 1):
        truncated = summary.content[:UPDATE_MATCH_MAX_CHARS_PER_SUMMARY]
        summaries_text += f"Саммари {i}:\n{truncated}\n\n"

    num_summaries = len(recent_summaries)
    numbers = ", ".join(str(i) for i in range(1, num_summaries + 1))

    user_content = f"""Существующие саммари:
{summaries_text}

Новое сообщение:
{msg.text}

В каком саммари ({numbers}) лучше всего добавить ссылку на это сообщение? 
Если ни одно не подходит, ответьте "НЕТ".
Отвечайте только номером ({numbers}) или "НЕТ"."""

    try:
        result = await call_openai(prompts.FIND_RELEVANT_SUMMARY_PROMPT, user_content, max_tokens=3, temperature=0)
        result = result.strip().upper()

        if result.isdigit():
            index = int(result) - 1
            if 0 <= index < len(recent_summaries):
                return recent_summaries[index]
    except Exception as e:
        logger.error("Error finding relevant summary for update: %s", e)

    return None


async def update_existing_summary(
    summary: SummaryInfo, new_message: MessageInfo, is_group: bool = False
) -> SummaryInfo:
    """
    Обновляет существующее саммари, интегрируя новую информацию через LLM.
    """
    links = extract_links(new_message.text)
    telegram_link = new_message.get_telegram_link()
    channel_abbr = create_channel_abbreviation(new_message.channel)

    if links:
        new_link = f'<a href="{links[0]}">[новое]</a> <a href="{telegram_link}">[{channel_abbr}]</a>'
    else:
        new_link = f'<a href="{telegram_link}">[{channel_abbr}]</a>'

    update_prompt = f"""Вставь ссылку на новое сообщение в подходящее место саммари.
Не переписывай всё саммари — только добавь ссылку {new_link} рядом с релевантным абзацем.
Если нет подходящего места — добавь строку "Другие ссылки: {new_link}" в конец.
Ответь только обновлённое саммари."""

    user_content = f"Саммари:\n{summary.content}\n\nНовое сообщение:\n{new_message.text}"

    try:
        updated_content = await call_openai(update_prompt, user_content, max_tokens=UPDATE_SUMMARY_MAX_TOKENS, temperature=0)
        if not updated_content:
            updated_content = summary.content + f"\n\nДругие ссылки: {new_link}"
    except Exception as e:
        logger.error("Error updating summary via LLM: %s; falling back to append", e)
        updated_content = summary.content + f"\n\nДругие ссылки: {new_link}"

    updated_channels = (
        summary.channels + [new_message.channel]
        if new_message.channel not in summary.channels
        else summary.channels
    )

    return SummaryInfo(
        content=updated_content,
        date=summary.date,
        message_count=summary.message_count + 1,
        channels=updated_channels,
        message_id=summary.message_id,
    )


async def save_updated_summary(
    original_summary: SummaryInfo, updated_summary: SummaryInfo, is_group: bool = False
) -> None:
    """Save updated summary, replacing the original in history and editing the channel message."""
    from telegram_client import edit_message_in_target_channel

    history_file = GROUP_SUMMARIES_HISTORY_FILE if is_group else SUMMARIES_HISTORY_FILE
    summaries = load_group_summaries_history() if is_group else load_summaries_history()

    found = False
    for i, summary in enumerate(summaries):
        if original_summary.message_id is not None and summary.message_id == original_summary.message_id:
            summaries[i] = updated_summary
            found = True
            break
        if summary.content == original_summary.content and summary.date == original_summary.date and summary.message_count == original_summary.message_count:
            summaries[i] = updated_summary
            found = True
            break

    if not found:
        logger.warning("Could not find original summary to update — skipping save and edit")
        return

    all_summaries_dict = [s.to_dict() for s in summaries]
    data = {"summaries": all_summaries_dict, "last_updated": now_iso()}
    save_json_file(
        history_file,
        data,
        "Error saving updated %s summary history" % ("group" if is_group else ""),
    )
    invalidate_cache(history_file)

    if original_summary.message_id:
        try:
            await edit_message_in_target_channel(
                original_summary.message_id, updated_summary.content
            )
            logger.info("Message %s updated in channel", original_summary.message_id)
        except Exception as e:
            logger.error("Error updating message %s in channel: %s", original_summary.message_id, e)
