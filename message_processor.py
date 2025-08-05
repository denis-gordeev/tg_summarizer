import random
import re
from typing import List, Set
from difflib import SequenceMatcher
from models import MessageInfo, SummaryInfo
from utils import call_openai, extract_links, count_characters
from config import SIMILARITY_THRESHOLD, ENABLE_SUMMARIES_DEDUPLICATION
from history_manager import (
    get_recent_summaries_context, get_recent_group_summaries_context
)
from channel_manager import (
    load_discovered_channels, load_similar_channels, load_banned_channels,
    create_channel_abbreviation
)
from prompts import prompts


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
    print(f"Используем {len(SOURCE_CHANNELS)} каналов из .env, "
          f"{len(discovered_channels)} обнаруженных, {len(similar_channels)} похожих каналов "
          f"(исключено {len(banned_channels)} заблокированных)")
    return all_channels


def is_message_processed(msg: MessageInfo, processed_messages: Set[str]) -> bool:
    """Проверяет, было ли сообщение уже обработано ранее."""
    msg_id = f"{msg.channel}_{msg.message_id}_{hash(msg.text)}"
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
        result = await call_openai(prompts.SUMMARY_COVERAGE_CHECK_PROMPT, user_content, max_tokens=10)
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
        result = await call_openai(prompts.GROUP_SUMMARY_COVERAGE_CHECK_PROMPT, user_content, max_tokens=10)
        return result.strip().upper() == "ДА"
    except Exception as e:
        print(f"Ошибка при проверке покрытия в саммари групп: {e}")
        return False


async def is_nlp_related(text: str) -> bool:
    """Use the LLM to decide if a message is NLP related and not advertising."""
    answer = await call_openai(prompts.NLP_RELEVANCE_PROMPT, text, max_tokens=5)
    return answer.lower().strip().startswith('да')


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
                    print(f"  Пропускаем сообщение, уже освещенное в саммари групп: "
                          f"{msg.text[:50]}...")
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
    max_summary_length = min(total_original_length // 3, 50)
    
    system_prompt = prompts.CHANNEL_SUMMARY_PROMPT.format(max_summary_length=max_summary_length)
    
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
    print("result:", "="*100, "\n", result, "\n", "="*100, "\n")

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
    
    system_prompt = prompts.GROUP_SUMMARY_PROMPT.format(max_summary_length=max_summary_length)
    
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
    
    result = header + result
    print("group_result:", "="*100, "\n", result, "\n", "="*100, "\n")
    return result


async def process_messages(messages: List[MessageInfo], save_changes: bool, send_message: bool, is_group: bool = False):
    """Processes a list of messages: filters, deduplicates, summarizes, and sends."""
    from config import SOURCE_CHANNELS
    from history_manager import save_summarization_history, save_group_summarization_history, save_summary_to_history, save_group_summary_to_history, update_group_last_run
    from channel_manager import save_discovered_channel
    from telegram_client import send_message_to_target_channel
    from datetime import datetime, timezone
    
    print(f"Fetched {len(messages)} new messages from {'groups' if is_group else 'channels'}")

    if not messages:
        print(f"No new messages found in {'groups' if is_group else 'channels'}")
        return

    filtered = []
    all_checked_messages = []
    discovered_channels = set() # Only relevant for channels, but kept for consistency

    for i, msg in enumerate(messages):
        print(f"Checking message {i+1}/{len(messages)}...")
        all_checked_messages.append(msg)

        if await is_nlp_related(msg.text):
            filtered.append(msg)
            if not is_group and msg.channel not in SOURCE_CHANNELS:
                discovered_channels.add(msg.channel)
            print(f"  ✓ Message {i+1} is NLP-related: {msg.text[:100]}; {msg.link}")
        else:
            print(f"  ✗ Message {i+1} is not NLP-related (likely advertising): {msg.text[:100]}; "
                  f"{msg.channel}; {msg.link}")
    print(f"{len(filtered)} messages after NLP filter")

    if save_changes:
        if is_group:
            save_group_summarization_history(all_checked_messages)
            print(f"Saved {len(all_checked_messages)} checked group messages to history")
        else:
            save_summarization_history(all_checked_messages)
            print(f"Saved {len(all_checked_messages)} checked messages to history")

        if not is_group: # Only save discovered channels for non-group messages
            for channel in discovered_channels:
                save_discovered_channel(channel)
            if discovered_channels:
                print(f"Discovered {len(discovered_channels)} new channels: {', '.join(discovered_channels)}")

    if filtered:
        unique = await (remove_group_duplicates(filtered) if is_group else remove_duplicates(filtered))
        print(f"{len(unique)} messages after deduplication")

        for msg in unique:
            if ENABLE_SUMMARIES_DEDUPLICATION:
                if await is_message_covered_in_summaries(msg):
                    await process_covered_message(msg)
                elif await is_message_covered_in_group_summaries(msg):
                    await process_covered_message(msg, is_group=True)

        if unique:
            summary = await (summarize_group_text(unique) if is_group else summarize_text(unique))
            if send_message:
                await send_message_to_target_channel(summary)
                print(f"{'Group' if is_group else 'Channel'} summary sent")

            if save_changes:
                channels = list(set(msg.channel for msg in unique))
                summary_info = SummaryInfo(
                    content=summary,
                    date=datetime.now(timezone.utc),
                    message_count=len(unique),
                    channels=channels
                )
                if is_group:
                    save_group_summary_to_history(summary_info)
                    print(f"Group summary saved to history (groups: {channels})")
                    update_group_last_run()
                    print("Group summarization completed for today")
                else:
                    save_summary_to_history(summary_info)
                    print(f"Channel summary saved to history (channels: {channels})")
        else:
            print(f"No unique NLP messages found in {'groups' if is_group else 'channels'}")
    else:
        print(f"No new NLP-related messages found in {'groups' if is_group else 'channels'}")


async def process_covered_message(msg: MessageInfo, is_group: bool = False):
    """Process a message that is already covered in previous summaries."""
    from config import ENABLE_SUMMARY_UPDATES
    from history_manager import find_relevant_summary_for_update, update_existing_summary, save_updated_summary
    
    if not ENABLE_SUMMARY_UPDATES:
        print(f"  Пропускаем уже освещенное сообщение: {msg.text[:50]}...")
        return
    
    print(f"  Проверяем возможность обновления саммари для: {msg.text[:50]}...")
    
    # Ищем подходящее саммари для обновления
    relevant_summary = await find_relevant_summary_for_update(msg, is_group)
    if relevant_summary:
        print(f"  Найдено подходящее саммари для обновления")
        print("relevant_summary:", "="*100, "\n", relevant_summary, "\n", "="*100, "\n")
        updated_summary = await update_existing_summary(relevant_summary, msg, is_group)
        print("updated_summary:", "="*100, "\n", updated_summary, "\n", "="*100, "\n")
        if updated_summary:
            await save_updated_summary(updated_summary, is_group)
            print(f"  Саммари обновлено с новым сообщением")
        else:
            print(f"  Не удалось обновить саммари")
    else:
        print(f"  Не найдено подходящее саммари для обновления")
