import asyncio
import json as _json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import List, Optional, Set
from models import MessageInfo, SummaryInfo
from utils import call_openai, extract_links, count_characters, enforce_summary_length, strip_meta_artifacts, text_hash, is_circuit_breaker_open
from config import (
    SIMILARITY_LLM_UPPER,
    ENABLE_SUMMARIES_DEDUPLICATION,
    OPENAI_CHANNEL_SUMMARY_MAX_TOKENS,
    OPENAI_GROUP_SUMMARY_MAX_TOKENS,
    OPENAI_SUMMARY_TEMPERATURE,
    SOURCE_CHANNELS,
    DEBUG,
    SUMMARY_MIN_RATIO,
    SUMMARY_MIN_LENGTH,
    SUMMARY_MAX_LENGTH,
    GROUP_SUMMARY_MIN_LENGTH,
    GROUP_SUMMARY_MAX_LENGTH,
    NLP_CHECK_MAX_INPUT_CHARS,
    COVERAGE_CHECK_MAX_INPUT_CHARS,
    NLP_MIN_TEXT_LENGTH,
    SUMMARY_MAX_INPUT_CHARS_PER_MESSAGE,
    NLP_CONCURRENT_CHECKS,
    NLP_AD_KEYWORDS,
    UPDATE_MATCH_MAX_SUMMARIES,
    UPDATE_MATCH_MAX_CHARS_PER_SUMMARY,
    ENABLE_SUMMARY_UPDATES,
    MAX_COVERED_MESSAGE_UPDATES,
)
from history_manager import (
    load_summaries_history,
    load_group_summaries_history,
    save_summarization_history,
    save_group_summarization_history,
    save_summary_to_history,
    save_group_summary_to_history,
    update_group_last_run,
    update_existing_summary,
    save_updated_summary,
)
from channel_manager import (
    create_channel_abbreviation,
    save_discovered_channel,
)
from prompts import prompts
logger = logging.getLogger(__name__)


def _emit_coverage_dedup_metric(covered: int, total: int, new: int, is_group: bool) -> None:
    try:
        import time as _time
        emf = {
            "_aws": {
                "Timestamp": int(_time.time() * 1000),
                "CloudWatchMetrics": [{
                    "Namespace": "tg_summarizer/Coverage",
                    "Dimensions": [["Function", "StreamType"]],
                    "Metrics": [
                        {"Name": "CoveredMessages", "Unit": "None"},
                        {"Name": "TotalNlpMessages", "Unit": "None"},
                        {"Name": "NewMessages", "Unit": "None"},
                    ]
                }]
            },
            "Function": os.getenv("AWS_LAMBDA_FUNCTION_NAME", "local"),
            "StreamType": "group" if is_group else "channel",
            "CoveredMessages": covered,
            "TotalNlpMessages": total,
            "NewMessages": new,
        }
        sys.stdout.write(_json.dumps(emf, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    except Exception:
        pass


def _calculate_channel_summary_limit(total_original_length: int) -> int:
    return min(max(total_original_length // SUMMARY_MIN_RATIO, SUMMARY_MIN_LENGTH), SUMMARY_MAX_LENGTH)


def _calculate_group_summary_limit(total_original_length: int) -> int:
    return min(max(total_original_length, GROUP_SUMMARY_MIN_LENGTH), GROUP_SUMMARY_MAX_LENGTH)


def is_message_processed(msg: MessageInfo, processed_messages: Set[str]) -> bool:
    """Проверяет, было ли сообщение уже обработано ранее."""
    text_hash_value = text_hash(msg.text)
    msg_id = f"{msg.channel}_{msg.message_id}_{text_hash_value}"
    return msg_id in processed_messages


async def _check_coverage_and_match(
    msg: MessageInfo,
    summaries: List[SummaryInfo],
) -> Optional[SummaryInfo]:
    """Combined coverage check + summary match. Returns matching summary if covered, None otherwise.

    Saves one LLM call per covered message vs separate _check_coverage + find_relevant_summary_for_update.
    """
    if not summaries:
        return None

    recent = summaries[-UPDATE_MATCH_MAX_SUMMARIES:]
    context_parts = []
    for i, s in enumerate(recent, 1):
        truncated = s.content[:UPDATE_MATCH_MAX_CHARS_PER_SUMMARY]
        context_parts.append(f"Дайджест {i}:\n{truncated}\n")

    user_content = f"""{"".join(context_parts)}Новое сообщение:
{msg.text[:COVERAGE_CHECK_MAX_INPUT_CHARS]}"""

    try:
        result = await call_openai(
            prompts.COVERAGE_AND_MATCH_PROMPT, user_content, max_tokens=3, temperature=0,
        )
        result = result.strip().upper()
        digits = re.match(r"^(\d+)", result)
        if digits:
            index = int(digits.group(1)) - 1
            if 0 <= index < len(recent):
                return recent[index]
            logger.debug("Coverage+match: LLM returned out-of-range index %d (have %d summaries)", index + 1, len(recent))
    except Exception as e:
        logger.error("Error in coverage+match check: %s", e)
    return None


_AD_PATTERN = re.compile("|".join(re.escape(kw) for kw in NLP_AD_KEYWORDS), re.IGNORECASE)


def _is_obvious_non_nlp(text: str) -> bool:
    """Quick regex pre-check for obvious ad/course content before LLM call."""
    return bool(_AD_PATTERN.search(text))


async def is_nlp_related(text: str) -> tuple[bool, str]:
    """Use the LLM to decide if a message is NLP related and not advertising."""
    if len(text) < NLP_MIN_TEXT_LENGTH:
        return False, "too_short"
    if _is_obvious_non_nlp(text):
        return False, "ad_keyword"
    if is_circuit_breaker_open():
        return False, "circuit_breaker_open"
    truncated = text[:NLP_CHECK_MAX_INPUT_CHARS]
    answer = await call_openai(prompts.NLP_RELEVANCE_PROMPT, truncated, max_tokens=10, temperature=0)
    return answer.lower().strip().startswith("да"), answer


def _remove_intra_batch_duplicates(messages: List[MessageInfo]) -> List[MessageInfo]:
    """Remove obvious duplicates within a batch using SequenceMatcher only.

    No LLM calls — catches near-identical messages (ratio > SIMILARITY_LLM_UPPER)
    and link duplicates. Ambiguous cases are left for the summary LLM to consolidate.
    """
    unique_msgs: List[MessageInfo] = []
    seen_links = set()
    seen_hashes = set()

    for msg in messages:
        links = extract_links(msg.text)
        if links and any(link in seen_links for link in links):
            logger.debug("Skipping intra-batch link duplicate: %s", links[0])
            continue

        msg_hash = text_hash(msg.text)
        if msg_hash in seen_hashes:
            logger.debug("Skipping intra-batch exact text duplicate (hash=%s)", msg_hash)
            continue

        duplicate = False
        for u in unique_msgs:
            len_a, len_b = len(msg.text), len(u.text)
            if len_a and len_b and abs(len_a - len_b) / max(len_a, len_b) > 0.5:
                continue
            ratio = SequenceMatcher(None, msg.text, u.text).ratio()
            if ratio > SIMILARITY_LLM_UPPER:
                logger.debug("Skipping intra-batch text duplicate (ratio=%.2f)", ratio)
                duplicate = True
                break

        if not duplicate:
            unique_msgs.append(msg)
            seen_links.update(links)
            seen_hashes.add(msg_hash)

    return unique_msgs


def _replace_source_with_links(messages: List[MessageInfo], result: str, msg_links: dict[int, list[str]] | None = None) -> str:
    """Replace source numbers [1], [2,3], etc. with HTML links in LLM output."""
    msg_data: dict[int, tuple[list[str], str, str]] = {}
    for i, msg in enumerate(messages, 1):
        links = msg_links[i] if msg_links and i in msg_links else extract_links(msg.text)
        telegram_link = msg.get_telegram_link()
        channel_abbr = create_channel_abbreviation(msg.channel)
        msg_data[i] = (links, telegram_link, channel_abbr)

    def _replacer(match):
        numbers = [num.strip() for num in match.group(1).split(",")]
        source_links = []
        for num_str in numbers:
            try:
                num = int(num_str)
                data = msg_data.get(num)
                if data is None:
                    continue
                links, telegram_link, channel_abbr = data
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


def _prepare_messages_text(messages: List[MessageInfo]) -> tuple[str, int, dict[int, list[str]]]:
    """Prepare messages text with source numbering and return (text, total_length, msg_links).

    msg_links maps message index (1-based) to its extracted links, so
    _replace_source_with_links can reuse them without a second regex pass.
    """
    messages_with_sources = []
    total_original_length = 0
    msg_links: dict[int, list[str]] = {}
    for i, msg in enumerate(messages, 1):
        links = extract_links(msg.text)
        msg_links[i] = links
        truncated_text = msg.text[:SUMMARY_MAX_INPUT_CHARS_PER_MESSAGE]
        source_info = f"[{i}] {truncated_text}"
        if links:
            source_info += f" (Ссылки: {', '.join(links)})"
        messages_with_sources.append(source_info)
        total_original_length += count_characters(truncated_text)
    return "\n\n".join(messages_with_sources), total_original_length, msg_links


async def summarize_text(messages: List[MessageInfo]) -> str:
    """Call LLM to summarize the given messages with links."""
    messages_text, total_original_length, msg_links = _prepare_messages_text(messages)
    max_summary_length = _calculate_channel_summary_limit(total_original_length)

    system_prompt = prompts.CHANNEL_SUMMARY_PROMPT.format(max_summary_length=max_summary_length)

    result = await call_openai(
        system_prompt,
        messages_text,
        max_tokens=OPENAI_CHANNEL_SUMMARY_MAX_TOKENS,
        temperature=OPENAI_SUMMARY_TEMPERATURE,
    )
    if not result:
        logger.error("Failed to generate channel summary for %d messages", len(messages))
        return None

    logger.debug("Source length: %d chars, summary: %d chars", total_original_length, count_characters(result))

    result = _replace_source_with_links(messages, result, msg_links)
    result = strip_meta_artifacts(result)
    result = enforce_summary_length(result, max_summary_length)
    if DEBUG:
        logger.debug("Summary result:\n%s", result)

    return result


async def summarize_group_text(messages: List[MessageInfo]) -> str:
    """Call LLM to summarize group messages with community review header."""
    messages_text, total_original_length, msg_links = _prepare_messages_text(messages)
    max_summary_length = _calculate_group_summary_limit(total_original_length)

    system_prompt = prompts.GROUP_SUMMARY_PROMPT.format(max_summary_length=max_summary_length)

    result = await call_openai(
        system_prompt,
        messages_text,
        max_tokens=OPENAI_GROUP_SUMMARY_MAX_TOKENS,
        temperature=OPENAI_SUMMARY_TEMPERATURE,
    )
    if not result:
        logger.error("Failed to generate group summary for %d messages", len(messages))
        return None

    logger.debug("Group source length: %d chars, summary: %d chars", total_original_length, count_characters(result))

    result = _replace_source_with_links(messages, result, msg_links)

    group_names = list(set(msg.channel.lstrip("@") for msg in messages))
    community_name = ", ".join(group_names)
    header = f"<b>👥 Обзор сообщества {community_name}</b>\n\n"

    result = header + enforce_summary_length(strip_meta_artifacts(result), max_summary_length - count_characters(header))
    if DEBUG:
        logger.debug("Group summary result:\n%s", result)
    return result


def _create_summary_info(
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
        summary_info = _create_summary_info(summary, unique_messages, message_id, is_group)
        if is_group:
            save_group_summary_to_history(summary_info)
            logger.info("Group summary saved to history (groups: %s)", summary_info.channels)
            update_group_last_run()
            logger.info("Group summarization completed for today")
        else:
            save_summary_to_history(summary_info)
            logger.info("Channel summary saved to history (channels: %s)", summary_info.channels)


async def _dedup_covered_messages(
    messages: List[MessageInfo],
    is_group: bool,
    sem: asyncio.Semaphore,
    _deadline: float,
) -> List[MessageInfo]:
    """Run coverage dedup: check each message against existing summaries.

    Covered messages are updated in-place (is_covered_in_summaries flag)
    and processed via process_covered_message. Returns messages not covered.
    """
    summaries = load_group_summaries_history() if is_group else load_summaries_history()
    if not summaries:
        return messages, summaries

    async def _check_msg_coverage(msg: MessageInfo) -> Optional[SummaryInfo]:
        async with sem:
            return await _check_coverage_and_match(msg, summaries)

    coverage_results = await asyncio.gather(
        *[_check_msg_coverage(msg) for msg in messages]
    )

    updated_count = 0
    for msg, matching_summary in zip(messages, coverage_results):
        if matching_summary is not None:
            if _deadline and time.monotonic() > _deadline:
                logger.warning("Deadline exceeded during covered message processing — un-marking remaining covered messages")
                msg.is_covered_in_summaries = False
                continue
            if updated_count >= MAX_COVERED_MESSAGE_UPDATES:
                logger.info("Reached MAX_COVERED_MESSAGE_UPDATES (%d) — un-marking remaining covered messages", MAX_COVERED_MESSAGE_UPDATES)
                msg.is_covered_in_summaries = False
                continue
            msg.is_covered_in_summaries = True
            await process_covered_message(msg, matching_summary=matching_summary, is_group=is_group, summaries=summaries)
            updated_count += 1
        else:
            msg.is_covered_in_summaries = False

    return [msg for msg in messages if not getattr(msg, 'is_covered_in_summaries', False)], summaries


async def process_messages(
    messages: List[MessageInfo], save_changes: bool, send_message: bool, is_group: bool = False,
    _deadline: float = 0.0,
):
    """Processes a list of messages: filters, deduplicates, summarizes, and sends."""
    from telegram_client import send_message_to_target_channel_with_id

    stream_label = "groups" if is_group else "channels"
    logger.info("Fetched %d new messages from %s", len(messages), stream_label)

    if not messages:
        logger.info("No new messages found in %s", stream_label)
        return

    all_checked_messages: List[MessageInfo] = []
    discovered_channels: set = set()
    nlp_related_messages: List[MessageInfo] = []

    if _deadline and time.monotonic() > _deadline:
        logger.warning("Deadline exceeded before NLP checks in %s — skipping", stream_label)
        return

    sem = asyncio.Semaphore(NLP_CONCURRENT_CHECKS)

    async def _check_nlp(text: str) -> tuple[bool, str]:
        async with sem:
            return await is_nlp_related(text)

    nlp_results = await asyncio.gather(*[_check_nlp(msg.text) for msg in messages])

    for msg, (is_related, reason) in zip(messages, nlp_results):
        if DEBUG:
            logger.debug("Checking message: %s", msg.to_dict())

        all_checked_messages.append(msg)
        msg.is_nlp_related = is_related
        msg.is_nlp_related_reason = reason

        if DEBUG:
            logger.debug(
                "NLP check: %s | %s | %s | Reason: %s",
                "✅" if msg.is_nlp_related else "❌",
                msg.text[:80], msg.channel, msg.is_nlp_related_reason,
            )

        if msg.is_nlp_related and not is_group and msg.channel not in SOURCE_CHANNELS:
            discovered_channels.add(msg.channel)

        if msg.is_nlp_related:
            nlp_related_messages.append(msg)

    ad_filtered = sum(1 for m in all_checked_messages if getattr(m, 'is_nlp_related_reason', '') == "ad_keyword")
    short_filtered = sum(1 for m in all_checked_messages if getattr(m, 'is_nlp_related_reason', '') == "too_short")
    rejected = len(all_checked_messages) - len(nlp_related_messages)
    logger.info(
        "NLP filter (%s): %d total, %d accepted, %d rejected (ad=%d, short=%d, other=%d)",
        stream_label, len(messages), len(nlp_related_messages), rejected,
        ad_filtered, short_filtered, rejected - ad_filtered - short_filtered,
    )

    unique_messages = nlp_related_messages
    if unique_messages:
        unique_messages = _remove_intra_batch_duplicates(unique_messages)
        logger.info("After intra-batch dedup: %d unique messages from %s", len(unique_messages), stream_label)

    if ENABLE_SUMMARIES_DEDUPLICATION and unique_messages:
        if _deadline and time.monotonic() > _deadline:
            logger.warning("Deadline exceeded before coverage checks in %s — skipping dedup", stream_label)
        else:
            unique_messages, _summaries = await _dedup_covered_messages(unique_messages, is_group, sem, _deadline)

    covered_count = sum(1 for msg in nlp_related_messages if getattr(msg, 'is_covered_in_summaries', False))
    if ENABLE_SUMMARIES_DEDUPLICATION and covered_count > 0:
        logger.info(
            "Coverage dedup (%s): %d covered, %d new (of %d deduped)",
            stream_label, covered_count, len(unique_messages), len(nlp_related_messages),
        )
        _emit_coverage_dedup_metric(covered_count, len(nlp_related_messages), len(unique_messages), is_group)

    summary = None
    message_id = None
    if unique_messages:
        if _deadline and time.monotonic() > _deadline:
            logger.warning("Deadline exceeded before summary generation in %s — skipping", stream_label)
        else:
            summary_fn = summarize_group_text if is_group else summarize_text
            summary = await summary_fn(unique_messages)
            if summary and send_message:
                message_id = await send_message_to_target_channel_with_id(summary)
                logger.info("%s summary sent with ID: %s", "Group" if is_group else "Channel", message_id)

    if save_changes:
        await _save_processing_results(
            all_checked_messages, unique_messages, summary, message_id,
            discovered_channels, is_group,
        )


async def process_covered_message(msg: MessageInfo, matching_summary: SummaryInfo = None, is_group: bool = False, summaries: List[SummaryInfo] | None = None):
    """Process a message that is already covered in previous summaries."""
    if not ENABLE_SUMMARY_UPDATES:
        logger.debug("Skipping already covered message: %s...", msg.text[:50])
        return

    if matching_summary is None:
        logger.debug("No matching summary provided — skipping update")
        return

    if matching_summary.message_id is not None:
        if summaries is None:
            summaries = load_group_summaries_history() if is_group else load_summaries_history()
        refreshed = None
        for s in summaries:
            if s.message_id == matching_summary.message_id:
                refreshed = s
                break
        if refreshed is None:
            logger.debug("Summary %s not found in history — skipping update", matching_summary.message_id)
            return
        matching_summary = refreshed

    logger.debug("Checking summary update for: %s...", msg.text[:50])
    updated_summary = await update_existing_summary(matching_summary, msg, is_group)
    if updated_summary:
        await save_updated_summary(matching_summary, updated_summary, is_group, summaries=summaries)
        logger.info("Summary updated with new message")
    else:
        logger.warning("Failed to update summary")
