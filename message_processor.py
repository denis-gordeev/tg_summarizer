import hashlib
import random
import re
from typing import List, Set
from difflib import SequenceMatcher
from models import MessageInfo, SummaryInfo
from utils import call_openai, extract_links, count_characters
from config import (
    SIMILARITY_THRESHOLD,
    ENABLE_SUMMARIES_DEDUPLICATION,
    OPENAI_CHANNEL_SUMMARY_MAX_TOKENS,
    OPENAI_GROUP_SUMMARY_MAX_TOKENS,
    DEBUG,
)
from history_manager import get_recent_summaries_context, get_recent_group_summaries_context
from channel_manager import (
    load_discovered_channels,
    load_similar_channels,
    load_banned_channels,
    create_channel_abbreviation,
)
from prompts import prompts


def _calculate_channel_summary_limit(total_original_length: int) -> int:
    return min(max(total_original_length // 3, 800), 4000)


def _calculate_group_summary_limit(total_original_length: int) -> int:
    return min(max(total_original_length, 2000), 12000)


def _truncate_html_preserving_tags(text: str, max_visible_chars: int) -> str:
    if max_visible_chars <= 0:
        return ""

    result: list[str] = []
    open_tags: list[str] = []
    visible_chars = 0
    i = 0
    truncated = False

    while i < len(text):
        char = text[i]
        if char == "<":
            end = text.find(">", i)
            if end == -1:
                break
            tag = text[i:end + 1]
            result.append(tag)

            tag_body = tag[1:-1].strip()
            if tag_body and not tag_body.startswith(("!", "?")):
                is_closing = tag_body.startswith("/")
                tag_name = tag_body[1:].split()[0].lower() if is_closing else tag_body.split()[0].lower()
                is_self_closing = tag_body.endswith("/")
                if is_closing:
                    if open_tags and open_tags[-1] == tag_name:
                        open_tags.pop()
                elif not is_self_closing:
                    open_tags.append(tag_name)
            i = end + 1
            continue

        if visible_chars >= max_visible_chars:
            truncated = True
            break

        result.append(char)
        visible_chars += 1
        i += 1

    output = "".join(result).rstrip()
    if truncated and visible_chars > 0 and not output.endswith("..."):
        output = output.rstrip(" ,;:\n") + "..."

    for tag_name in reversed(open_tags):
        output += f"</{tag_name}>"

    return output.strip()


def enforce_summary_length(summary: str, max_visible_chars: int) -> str:
    if count_characters(summary) <= max_visible_chars:
        return summary.strip()

    blocks = [block.strip() for block in summary.split("\n\n") if block.strip()]
    if blocks:
        kept_blocks: list[str] = []
        for block in blocks:
            candidate = "\n\n".join(kept_blocks + [block])
            if count_characters(candidate) > max_visible_chars:
                break
            kept_blocks.append(block)
        if kept_blocks:
            return "\n\n".join(kept_blocks).strip()

    return _truncate_html_preserving_tags(summary, max_visible_chars)


def get_all_source_channels() -> List[str]:
    """Возвращает объединенный список каналов из .env, обнаруженных и похожих каналов."""
    from config import SOURCE_CHANNELS

    discovered_channels = set(load_discovered_channels())
    similar_channels = set(load_similar_channels())
    banned_channels = set(load_banned_channels())

    # Исключаем заблокированные каналы
    all_channels_set = SOURCE_CHANNELS | discovered_channels | similar_channels
    all_channels_set = all_channels_set - banned_channels

    all_channels = list(all_channels_set)
    random.shuffle(all_channels)
    print(
        f"Используем {len(SOURCE_CHANNELS)} каналов из .env, "
        f"{len(discovered_channels)} обнаруженных, {len(similar_channels)} похожих каналов "
        f"(исключено {len(banned_channels)} заблокированных)"
    )
    return all_channels


def is_message_processed(msg: MessageInfo, processed_messages: Set[str]) -> bool:
    """Проверяет, было ли сообщение уже обработано ранее."""
    text_hash = hashlib.sha256(msg.text.encode()).hexdigest()[:16]
    msg_id = f"{msg.channel}_{msg.message_id}_{text_hash}"
    return msg_id in processed_messages


async def are_messages_duplicate(msg_a: MessageInfo, msg_b: MessageInfo) -> bool:
    """Use the LLM to see if two messages cover the same topic."""
    user_content = f"Message 1:\n{msg_a.text}\n\nMessage 2:\n{msg_b.text}"

    answer = await call_openai(prompts.DUPLICATE_CHECK_PROMPT, user_content, max_tokens=1)
    return answer.lower().startswith("y")


async def is_message_covered_in_summaries(msg: MessageInfo) -> bool:
    """Проверяет, была ли тема сообщения уже освещена в предыдущих саммари."""
    if not ENABLE_SUMMARIES_DEDUPLICATION:
        return False

    recent_summaries = get_recent_summaries_context()
    if not recent_summaries:
        return False

    user_content = f"""Предыдущие дайджесты:
        {recent_summaries}

        Новое сообщение:
        {msg.text}

        Была ли эта тема уже освещена в предыдущих дайджестах?"""

    try:
        result = await call_openai(
            prompts.SUMMARY_COVERAGE_CHECK_PROMPT, user_content, max_tokens=10
        )
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

    user_content = f"""Предыдущие дайджесты групп:
{recent_summaries}

Новое сообщение:
{msg.text}

Была ли эта тема уже освещена в предыдущих дайджестах групп?"""

    try:
        result = await call_openai(
            prompts.GROUP_SUMMARY_COVERAGE_CHECK_PROMPT, user_content, max_tokens=10
        )
        return result.strip().upper() == "ДА"
    except Exception as e:
        print(f"Ошибка при проверке покрытия в саммари групп: {e}")
        return False


async def is_nlp_related(text: str) -> tuple[bool, str]:
    """Use the LLM to decide if a message is NLP related and not advertising."""
    if len(text) < 100:
        return False, "too_short"
    answer = await call_openai(prompts.NLP_RELEVANCE_PROMPT, text, max_tokens=30)
    return answer.lower().strip().startswith("да"), answer


async def _remove_duplicates_generic(
    messages: List[MessageInfo],
    coverage_check_fn,
    unique_label: str,
    duplicate_label: str,
) -> List[MessageInfo]:
    """Generic deduplication shared by channel and group message streams."""
    unique_msgs: List[MessageInfo] = []
    seen_links = set()

    for msg in messages:
        links = extract_links(msg.text)

        if links and any(link in seen_links for link in links):
            print(f"  Пропускаем дубликат по ссылке: {links[0]}")
            continue

        duplicate = False
        for u in unique_msgs:
            if SequenceMatcher(None, msg.text, u.text).ratio() > SIMILARITY_THRESHOLD:
                print(f"  Пропускаем дубликат по тексту: {msg.text[:50]}...")
                duplicate = True
                break

        if not duplicate:
            for u in unique_msgs:
                try:
                    if await are_messages_duplicate(msg, u):
                        print(f"  Пропускаем дубликат по LLM: {msg.text[:50]}...")
                        duplicate = True
                        break
                except Exception as e:
                    print(f"  Ошибка при проверке дубликата через LLM: {e}")
                    continue

        if not duplicate and ENABLE_SUMMARIES_DEDUPLICATION:
            try:
                if await coverage_check_fn(msg):
                    print(f"  Пропускаем {duplicate_label}: {msg.text[:50]}...")
                    duplicate = True
            except Exception as e:
                print(f"  Ошибка при проверке покрытия в саммари: {e}")

        if not duplicate:
            unique_msgs.append(msg)
            seen_links.update(links)
            print(f"  {unique_label}: {msg.text[:50]}...")

    return unique_msgs


async def remove_duplicates(messages: List[MessageInfo]) -> List[MessageInfo]:
    return await _remove_duplicates_generic(
        messages,
        coverage_check_fn=is_message_covered_in_summaries,
        unique_label="Добавляем уникальное сообщение",
        duplicate_label="сообщение, уже освещенное в саммари",
    )


async def remove_group_duplicates(messages: List[MessageInfo]) -> List[MessageInfo]:
    """Remove duplicates from group messages, checking against group summaries history."""
    return await _remove_duplicates_generic(
        messages,
        coverage_check_fn=is_message_covered_in_group_summaries,
        unique_label="Добавляем уникальное сообщение из группы",
        duplicate_label="сообщение, уже освещенное в саммари групп",
    )


def _replace_source_with_links(messages: List[MessageInfo], result: str) -> str:
    """Replace source numbers [1], [2,3], etc. with HTML links in LLM output."""
    def _replacer(match):
        numbers = [num.strip() for num in match.group(1).split(",")]
        source_links = []
        for num_str in numbers:
            try:
                num = int(num_str)
                if 1 <= num <= len(messages):
                    msg = messages[num - 1]
                    links = extract_links(msg.text)
                    telegram_link = msg.get_telegram_link()
                    channel_abbr = create_channel_abbreviation(msg.channel)
                    if links:
                        main_link = links[0]
                        source_links.append(
                            f'<a href="{main_link}">[{num}]</a> <a href="{telegram_link}">[{channel_abbr}]</a>'
                        )
                    else:
                        source_links.append(f'<a href="{telegram_link}">[{channel_abbr}]</a>')
            except ValueError:
                continue
        return ", ".join(source_links)

    return re.sub(r"\[(\d+(?:,\s*\d+)*)\]", _replacer, result)


def _prepare_messages_text(messages: List[MessageInfo]) -> tuple[str, int]:
    """Prepare messages text with source numbering and return (text, total_length)."""
    messages_with_sources = []
    total_original_length = 0
    for i, msg in enumerate(messages, 1):
        links = extract_links(msg.text)
        source_info = f"[{i}] {msg.text}"
        if links:
            source_info += f" (Ссылки: {', '.join(links)})"
        messages_with_sources.append(source_info)
        total_original_length += count_characters(msg.text)
    return "\n\n".join(messages_with_sources), total_original_length


async def summarize_text(messages: List[MessageInfo]) -> str:
    """Call LLM to summarize the given messages with links."""
    messages_text, total_original_length = _prepare_messages_text(messages)
    max_summary_length = _calculate_channel_summary_limit(total_original_length)

    system_prompt = prompts.CHANNEL_SUMMARY_PROMPT.format(max_summary_length=max_summary_length)

    result = await call_openai(
        system_prompt,
        messages_text,
        max_tokens=OPENAI_CHANNEL_SUMMARY_MAX_TOKENS,
    )
    if not result:
        return "Ошибка: Не удалось сгенерировать обобщение"

    print(f"Длина исходного текста: {total_original_length} символов")
    print(f"Длина саммари: {count_characters(result)} символов")

    result = _replace_source_with_links(messages, result)
    result = enforce_summary_length(result, max_summary_length)
    if DEBUG:
        print("result:", "=" * 100, "\n", result, "\n", "=" * 100, "\n")

    return result


async def summarize_group_text(messages: List[MessageInfo]) -> str:
    """Call LLM to summarize group messages with community review header."""
    messages_text, total_original_length = _prepare_messages_text(messages)
    max_summary_length = _calculate_group_summary_limit(total_original_length)

    system_prompt = prompts.GROUP_SUMMARY_PROMPT.format(max_summary_length=max_summary_length)

    result = await call_openai(
        system_prompt,
        messages_text,
        max_tokens=OPENAI_GROUP_SUMMARY_MAX_TOKENS,
    )
    if not result:
        return "Ошибка: Не удалось сгенерировать обобщение"

    print(f"Длина исходного текста групп: {total_original_length} символов")
    print(f"Длина саммари групп: {count_characters(result)} символов")

    result = _replace_source_with_links(messages, result)

    group_names = list(set(msg.channel.lstrip("@") for msg in messages))
    community_name = ", ".join(group_names)
    header = f"<b>👥 Обзор сообщества {community_name}</b>\n\n"

    result = header + result
    result = enforce_summary_length(result, max_summary_length)
    if DEBUG:
        print("group_result:", "=" * 100, "\n", result, "\n", "=" * 100, "\n")
    return result


async def process_messages(
    messages: List[MessageInfo], save_changes: bool, send_message: bool, is_group: bool = False
):
    """Processes a list of messages: filters, deduplicates, summarizes, and sends."""
    from config import SOURCE_CHANNELS
    from history_manager import (
        save_summarization_history,
        save_group_summarization_history,
        save_summary_to_history,
        save_group_summary_to_history,
        update_group_last_run,
    )
    from channel_manager import save_discovered_channel
    from telegram_client import send_message_to_target_channel_with_id
    from datetime import datetime, timezone

    print(f"Fetched {len(messages)} new messages from {'groups' if is_group else 'channels'}")

    if not messages:
        print(f"No new messages found in {'groups' if is_group else 'channels'}")
        return

    all_checked_messages = []
    discovered_channels = set()  # Only relevant for channels, but kept for consistency

    for i, msg in enumerate(messages):
        if DEBUG:
            print(f"Checking message {i+1}/{len(messages)}...")
            print("<" * 100)
            print(f"msg: {msg.to_dict()}")
            print(">" * 100)
        all_checked_messages.append(msg)

        # Проверяем, является ли сообщение NLP-релевантным
        msg.is_nlp_related, msg.is_nlp_related_reason = await is_nlp_related(msg.text)

        is_nlp_related_message = "✅" if msg.is_nlp_related else "❌"
        is_nlp_related_message += (
            f" {msg.text} {msg.channel} {msg.message_id} \nReason: {msg.is_nlp_related_reason}"
        )
        print(is_nlp_related_message)

        if msg.is_nlp_related:
            if not is_group and msg.channel not in SOURCE_CHANNELS:
                discovered_channels.add(msg.channel)
        if ENABLE_SUMMARIES_DEDUPLICATION and msg.is_nlp_related:
            msg.is_covered_in_summaries = False
            if is_group:
                msg.is_covered_in_summaries = await is_message_covered_in_group_summaries(msg)
            else:
                msg.is_covered_in_summaries = await is_message_covered_in_summaries(msg)
            print(f"\tmsg.is_covered_in_summaries: {msg.is_covered_in_summaries}")
            if msg.is_covered_in_summaries:
                await process_covered_message(msg, is_group=is_group)

    unique = [
        msg
        for msg in all_checked_messages
        if msg.is_nlp_related and not msg.is_covered_in_summaries
    ]
    summary = None
    if unique:
        summary = await (summarize_group_text(unique) if is_group else summarize_text(unique))
        message_id = None
        if send_message:
            message_id = await send_message_to_target_channel_with_id(summary)
            print(f"{'Group' if is_group else 'Channel'} summary sent with ID: {message_id}")

    if save_changes:
        if is_group:
            save_group_summarization_history(all_checked_messages)
        else:
            save_summarization_history(all_checked_messages)
        if not is_group:  # Only save discovered channels for non-group messages
            for channel in discovered_channels:
                save_discovered_channel(channel)
            if discovered_channels:
                print(f"Discovered {len(discovered_channels)} new channels: {discovered_channels}")
        channels = list(set(msg.channel for msg in unique))
        if summary:
            summary_info = SummaryInfo(
                content=summary,
                date=datetime.now(timezone.utc),
                message_count=len(unique),
                channels=channels,
                message_id=message_id,
            )
            if is_group:
                save_group_summary_to_history(summary_info)
                print(f"Group summary saved to history (groups: {channels})")
                update_group_last_run()
                print("Group summarization completed for today")
            else:
                save_summary_to_history(summary_info)
                print(f"Channel summary saved to history (channels: {channels})")


async def process_covered_message(msg: MessageInfo, is_group: bool = False):
    """Process a message that is already covered in previous summaries."""
    from config import ENABLE_SUMMARY_UPDATES
    from history_manager import (
        find_relevant_summary_for_update,
        update_existing_summary,
        save_updated_summary,
    )

    if not ENABLE_SUMMARY_UPDATES:
        print(f"  Пропускаем уже освещенное сообщение: {msg.text[:50]}...")
        return

    print(f"  Проверяем возможность обновления саммари для: {msg.text[:50]}...")

    # Ищем подходящее саммари для обновления
    relevant_summary = await find_relevant_summary_for_update(msg, is_group)
    if relevant_summary:
        print(f"  Найдено подходящее саммари для обновления")
        print("relevant_summary:", "=" * 100, "\n", relevant_summary, "\n", "=" * 100, "\n")
        updated_summary = await update_existing_summary(relevant_summary, msg, is_group)
        print("updated_summary:", "=" * 100, "\n", updated_summary, "\n", "=" * 100, "\n")
        if updated_summary:
            await save_updated_summary(relevant_summary, updated_summary, is_group)
            print(f"  Саммари обновлено с новым сообщением")
        else:
            print(f"  Не удалось обновить саммари")
    else:
        print(f"  Не найдено подходящее саммари для обновления")
