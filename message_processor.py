import logging
import re
from datetime import datetime, timezone
from typing import List, Set
from difflib import SequenceMatcher
from models import MessageInfo, SummaryInfo
from utils import call_openai, extract_links, count_characters, text_hash
from config import (
    SIMILARITY_LLM_LOWER,
    SIMILARITY_LLM_UPPER,
    ENABLE_SUMMARIES_DEDUPLICATION,
    OPENAI_CHANNEL_SUMMARY_MAX_TOKENS,
    OPENAI_GROUP_SUMMARY_MAX_TOKENS,
    DEBUG,
    SUMMARY_MIN_RATIO,
    SUMMARY_MIN_LENGTH,
    SUMMARY_MAX_LENGTH,
    GROUP_SUMMARY_MIN_LENGTH,
    GROUP_SUMMARY_MAX_LENGTH,
    NLP_CHECK_MAX_INPUT_CHARS,
)
from history_manager import get_recent_summaries_context, get_recent_group_summaries_context
from channel_manager import (
    create_channel_abbreviation,
)
from prompts import prompts

logger = logging.getLogger(__name__)


def _calculate_channel_summary_limit(total_original_length: int) -> int:
    return min(max(total_original_length // SUMMARY_MIN_RATIO, SUMMARY_MIN_LENGTH), SUMMARY_MAX_LENGTH)


def _calculate_group_summary_limit(total_original_length: int) -> int:
    return min(max(total_original_length, GROUP_SUMMARY_MIN_LENGTH), GROUP_SUMMARY_MAX_LENGTH)


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


def is_message_processed(msg: MessageInfo, processed_messages: Set[str]) -> bool:
    """Проверяет, было ли сообщение уже обработано ранее."""
    text_hash_value = text_hash(msg.text)
    msg_id = f"{msg.channel}_{msg.message_id}_{text_hash_value}"
    return msg_id in processed_messages


async def are_messages_duplicate(msg_a: MessageInfo, msg_b: MessageInfo) -> bool:
    """Use the LLM to see if two messages cover the same topic."""
    user_content = f"Message 1:\n{msg_a.text}\n\nMessage 2:\n{msg_b.text}"

    answer = await call_openai(prompts.DUPLICATE_CHECK_PROMPT, user_content, max_tokens=3, temperature=0)
    return answer.strip().lower().startswith("да")


async def _check_coverage(
    msg: MessageInfo,
    recent_summaries: str,
    prompt,
    label: str,
) -> bool:
    """Check if a message topic is already covered in previous summaries."""
    if not ENABLE_SUMMARIES_DEDUPLICATION:
        return False

    if not recent_summaries:
        return False

    user_content = f"""Предыдущие дайджесты{label}:
{recent_summaries}

Новое сообщение:
{msg.text}

Была ли эта тема уже освещена в предыдущих дайджестах{label}?"""

    try:
        result = await call_openai(prompt, user_content, max_tokens=2, temperature=0)
        return result.strip().upper() == "ДА"
    except Exception as e:
        logger.error("Error checking %s coverage: %s", label.strip(), e)
        return False


async def is_message_covered_in_summaries(msg: MessageInfo) -> bool:
    """Проверяет, была ли тема сообщения уже освещена в предыдущих саммари."""
    return await _check_coverage(
        msg,
        get_recent_summaries_context(),
        prompts.SUMMARY_COVERAGE_CHECK_PROMPT,
        "",
    )


async def is_message_covered_in_group_summaries(msg: MessageInfo) -> bool:
    """Проверяет, была ли тема сообщения уже освещена в предыдущих саммари групп."""
    return await _check_coverage(
        msg,
        get_recent_group_summaries_context(),
        prompts.GROUP_SUMMARY_COVERAGE_CHECK_PROMPT,
        " групп",
    )


async def is_nlp_related(text: str) -> tuple[bool, str]:
    """Use the LLM to decide if a message is NLP related and not advertising."""
    if len(text) < 100:
        return False, "too_short"
    truncated = text[:NLP_CHECK_MAX_INPUT_CHARS]
    answer = await call_openai(prompts.NLP_RELEVANCE_PROMPT, truncated, max_tokens=20, temperature=0)
    return answer.lower().strip().startswith("да"), answer


async def _remove_duplicates_generic(
    messages: List[MessageInfo],
    coverage_check_fn,
    unique_label: str,
) -> List[MessageInfo]:
    """Generic deduplication shared by channel and group message streams.

    Uses a three-band SequenceMatcher strategy to minimize LLM calls:
    - ratio > SIMILARITY_LLM_UPPER (0.95): definite duplicate, no LLM
    - ratio < SIMILARITY_LLM_LOWER (0.7): definite different, no LLM
    - between: call LLM to decide
    """
    unique_msgs: List[MessageInfo] = []
    seen_links = set()

    for msg in messages:
        links = extract_links(msg.text)

        if links and any(link in seen_links for link in links):
            logger.debug("Skipping link duplicate: %s", links[0])
            continue

        duplicate = False
        for u in unique_msgs:
            ratio = SequenceMatcher(None, msg.text, u.text).ratio()
            if ratio > SIMILARITY_LLM_UPPER:
                logger.debug("Skipping text duplicate (ratio=%.2f): %s...", ratio, msg.text[:50])
                duplicate = True
                break
            if ratio < SIMILARITY_LLM_LOWER:
                continue
            try:
                if await are_messages_duplicate(msg, u):
                    logger.debug("Skipping LLM duplicate (ratio=%.2f): %s...", ratio, msg.text[:50])
                    duplicate = True
                    break
            except Exception as e:
                logger.error("Error checking LLM duplicate: %s", e)
                continue

        if not duplicate and ENABLE_SUMMARIES_DEDUPLICATION:
            try:
                if await coverage_check_fn(msg):
                    logger.debug("Skipping duplicate: %s...", msg.text[:50])
                    duplicate = True
            except Exception as e:
                logger.error("Error checking coverage: %s", e)

        if not duplicate:
            unique_msgs.append(msg)
            seen_links.update(links)
            logger.debug("%s: %s...", unique_label, msg.text[:50])

    return unique_msgs


async def remove_duplicates(messages: List[MessageInfo]) -> List[MessageInfo]:
    return await _remove_duplicates_generic(
        messages,
        coverage_check_fn=is_message_covered_in_summaries,
        unique_label="Добавляем уникальное сообщение",
    )


async def remove_group_duplicates(messages: List[MessageInfo]) -> List[MessageInfo]:
    """Remove duplicates from group messages, checking against group summaries history."""
    return await _remove_duplicates_generic(
        messages,
        coverage_check_fn=is_message_covered_in_group_summaries,
        unique_label="Добавляем уникальное сообщение из группы",
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

    logger.debug("Source length: %d chars, summary: %d chars", total_original_length, count_characters(result))

    result = _replace_source_with_links(messages, result)
    result = enforce_summary_length(result, max_summary_length)
    if DEBUG:
        logger.debug("Summary result:\n%s", result)

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

    logger.debug("Group source length: %d chars, summary: %d chars", total_original_length, count_characters(result))

    result = _replace_source_with_links(messages, result)

    group_names = list(set(msg.channel.lstrip("@") for msg in messages))
    community_name = ", ".join(group_names)
    header = f"<b>👥 Обзор сообщества {community_name}</b>\n\n"

    result = header + result
    result = enforce_summary_length(result, max_summary_length)
    if DEBUG:
        logger.debug("Group summary result:\n%s", result)
    return result


async def _classify_message(
    msg: MessageInfo, is_group: bool, source_channels
) -> tuple[set, bool]:
    """Check NLP relevance, discover new channels, and check summary coverage.

    Returns (discovered_channels_set, is_covered_in_summaries).
    """
    discovered = set()
    msg.is_nlp_related, msg.is_nlp_related_reason = await is_nlp_related(msg.text)

    if DEBUG:
        logger.debug(
            "NLP check: %s | %s | %s | Reason: %s",
            "✅" if msg.is_nlp_related else "❌",
            msg.text[:80], msg.channel, msg.is_nlp_related_reason,
        )

    if msg.is_nlp_related and not is_group and msg.channel not in source_channels:
        discovered.add(msg.channel)

    is_covered = False
    if ENABLE_SUMMARIES_DEDUPLICATION and msg.is_nlp_related:
        coverage_fn = (
            is_message_covered_in_group_summaries if is_group
            else is_message_covered_in_summaries
        )
        is_covered = await coverage_fn(msg)
        msg.is_covered_in_summaries = is_covered
        logger.debug("is_covered_in_summaries: %s", is_covered)
        if is_covered:
            await process_covered_message(msg, is_group=is_group)

    return discovered, is_covered


async def _create_summary_info(
    summary: str, unique_messages: List[MessageInfo], message_id, is_group: bool
) -> SummaryInfo:
    channels = list(set(msg.channel for msg in unique_messages))
    return SummaryInfo(
        content=summary,
        date=datetime.now(timezone.utc),
        message_count=len(unique_messages),
        channels=channels,
        message_id=message_id,
    )


async def _save_processing_results(
    all_checked_messages: List[MessageInfo],
    unique_messages: List[MessageInfo],
    summary: str,
    message_id,
    discovered_channels: set,
    is_group: bool,
) -> None:
    """Save history, discovered channels, and summary info."""
    from history_manager import (
        save_summarization_history,
        save_group_summarization_history,
        save_summary_to_history,
        save_group_summary_to_history,
        update_group_last_run,
    )
    from channel_manager import save_discovered_channel

    if all_checked_messages:
        if is_group:
            save_group_summarization_history(all_checked_messages)
        else:
            save_summarization_history(all_checked_messages)

    if not is_group and discovered_channels:
        for channel in discovered_channels:
            save_discovered_channel(channel)
        logger.info("Discovered %d new channels: %s", len(discovered_channels), discovered_channels)

    if summary and unique_messages:
        summary_info = await _create_summary_info(summary, unique_messages, message_id, is_group)
        if is_group:
            save_group_summary_to_history(summary_info)
            logger.info("Group summary saved to history (groups: %s)", summary_info.channels)
            update_group_last_run()
            logger.info("Group summarization completed for today")
        else:
            save_summary_to_history(summary_info)
            logger.info("Channel summary saved to history (channels: %s)", summary_info.channels)


async def process_messages(
    messages: List[MessageInfo], save_changes: bool, send_message: bool, is_group: bool = False
):
    """Processes a list of messages: filters, deduplicates, summarizes, and sends."""
    from config import SOURCE_CHANNELS
    from telegram_client import send_message_to_target_channel_with_id

    stream_label = "groups" if is_group else "channels"
    logger.info("Fetched %d new messages from %s", len(messages), stream_label)

    if not messages:
        logger.info("No new messages found in %s", stream_label)
        return

    all_checked_messages: List[MessageInfo] = []
    discovered_channels: set = set()
    unique_messages: List[MessageInfo] = []

    for msg in messages:
        if DEBUG:
            logger.debug("Checking message: %s", msg.to_dict())

        all_checked_messages.append(msg)
        disc, is_covered = await _classify_message(msg, is_group, SOURCE_CHANNELS)
        discovered_channels.update(disc)

        if msg.is_nlp_related and not is_covered:
            unique_messages.append(msg)

    summary = None
    message_id = None
    if unique_messages:
        summary_fn = summarize_group_text if is_group else summarize_text
        summary = await summary_fn(unique_messages)
        if send_message:
            message_id = await send_message_to_target_channel_with_id(summary)
            logger.info("%s summary sent with ID: %s", "Group" if is_group else "Channel", message_id)

    if save_changes:
        await _save_processing_results(
            all_checked_messages, unique_messages, summary, message_id,
            discovered_channels, is_group,
        )


async def process_covered_message(msg: MessageInfo, is_group: bool = False):
    """Process a message that is already covered in previous summaries."""
    from config import ENABLE_SUMMARY_UPDATES
    from history_manager import (
        find_relevant_summary_for_update,
        update_existing_summary,
        save_updated_summary,
    )

    if not ENABLE_SUMMARY_UPDATES:
        logger.debug("Skipping already covered message: %s...", msg.text[:50])
        return

    logger.debug("Checking summary update for: %s...", msg.text[:50])

    # Ищем подходящее саммари для обновления
    relevant_summary = await find_relevant_summary_for_update(msg, is_group)
    if relevant_summary:
        logger.debug("Found relevant summary for update")
        updated_summary = await update_existing_summary(relevant_summary, msg, is_group)
        if updated_summary:
            await save_updated_summary(relevant_summary, updated_summary, is_group)
            logger.info("Summary updated with new message")
        else:
            logger.warning("Failed to update summary")
    else:
        logger.debug("No relevant summary found for update")
