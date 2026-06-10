import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Any

from telethon import TelegramClient  # type: ignore[reportMissingTypeStubs]
from telethon.tl.functions.channels import (  # type: ignore[reportMissingTypeStubs]
    GetChannelRecommendationsRequest,
)
from telethon.tl.types import InputChannel  # type: ignore[reportMissingTypeStubs]

from config import API_HASH, API_ID, BOT_TOKEN, SOURCE_GROUPS, MAX_MESSAGES_PER_SOURCE
from history_manager import load_group_summarization_history, load_summarization_history
from channel_manager import get_all_source_channels
from message_processor import is_message_processed
from models import MessageInfo
from utils import extract_links

logger = logging.getLogger(__name__)

# Clients are created lazily inside the active event loop
user_client: Any = None
bot_client: Any = None
clients_loop: Optional[asyncio.AbstractEventLoop] = None


async def _ensure_clients() -> None:
    """Ensure Telegram clients are connected, starting them if needed."""
    if user_client is None or not user_client.is_connected():
        await start_clients()


async def _ensure_bot_client() -> None:
    """Ensure bot client is connected, starting it if needed."""
    await _ensure_clients()


async def get_similar_channels_from_telegram(channel_username: Optional[str] = None) -> List[str]:
    """Получает список похожих каналов через Telegram API."""
    try:
        global user_client
        await _ensure_clients()

        # Если указан канал, получаем рекомендации для него
        if channel_username:
            # Убираем @ если есть
            channel_username = channel_username.lstrip("@")

            # Получаем информацию о канале
            try:
                channel_entity = await user_client.get_entity(f"@{channel_username}")
                channel_input = InputChannel(channel_entity.id, channel_entity.access_hash)

                # Получаем рекомендации
                result = await user_client(GetChannelRecommendationsRequest(channel=channel_input))

                similar_channels = []
                for chat in result.chats:
                    if hasattr(chat, "username") and chat.username:
                        similar_channels.append(f"@{chat.username}")

                return similar_channels

            except Exception as e:
                logger.error("Error getting recommendations for channel %s: %s", channel_username, e)
                return []

        # Если канал не указан, получаем глобальные рекомендации
        else:
            result = await user_client(GetChannelRecommendationsRequest())

            similar_channels = []
            for chat in result.chats:
                if hasattr(chat, "username") and chat.username:
                    similar_channels.append(f"@{chat.username}")

            return similar_channels

    except Exception as e:
        logger.error("Error getting similar channels: %s", e)
        return []


async def _fetch_from_sources(
    sources: List[str],
    processed_messages: set,
    source_label: str,
    include_today_processed: bool = False,
    _deadline: float = 0.0,
) -> List[MessageInfo]:
    """Generic message fetcher for both channels and groups."""
    since = datetime.now(timezone.utc) - timedelta(days=1)
    all_msgs = []

    await _ensure_clients()

    for source in sources:
        if _deadline and time.monotonic() > _deadline:
            logger.warning("Deadline exceeded while fetching %s — returning %d messages fetched so far", source_label, len(all_msgs))
            break
        logger.info("Fetching messages from %s %s", source_label, source)
        source_count = 0
        total_examined = 0
        try:
            async for msg in user_client.iter_messages(
                source, offset_date=None, min_id=0, reverse=False
            ):
                if _deadline and time.monotonic() > _deadline:
                    logger.warning("Deadline exceeded during fetch from %s %s — returning %d messages fetched so far", source_label, source, len(all_msgs))
                    break
                if msg.date < since:
                    break
                total_examined += 1
                if total_examined > MAX_MESSAGES_PER_SOURCE * 3:
                    logger.debug("Reached total examined limit (%d) for %s %s", total_examined, source_label, source)
                    break
                if source_count >= MAX_MESSAGES_PER_SOURCE:
                    logger.debug("Reached MAX_MESSAGES_PER_SOURCE (%d) for %s %s", MAX_MESSAGES_PER_SOURCE, source_label, source)
                    break
                if msg.text:
                    links = extract_links(msg.text)
                    main_link = links[0] if links else ""

                    message_info = MessageInfo(
                        text=msg.text,
                        channel=source,
                        message_id=msg.id,
                        date=msg.date,
                        link=main_link,
                    )

                    if include_today_processed or not is_message_processed(
                        message_info, processed_messages
                    ):
                        source_count += 1
                        all_msgs.append(message_info)
                    else:
                        logger.debug("Skipping already processed message %s from %s", msg.id, source)

            logger.info("Found %d new messages from %s %s", source_count, source_label, source)
        except Exception as e:
            logger.error("Error fetching messages from %s %s: %s", source_label, source, e, exc_info=True)
            continue
    return all_msgs


async def fetch_messages(include_today_processed_messages: bool = False, _deadline: float = 0.0) -> List[MessageInfo]:
    """Fetch messages from source channels in the last 24 hours."""
    processed_messages = load_summarization_history()
    logger.info("Loaded %d already processed messages from history", len(processed_messages))

    all_channels = get_all_source_channels()
    return await _fetch_from_sources(
        all_channels, processed_messages, "channel", include_today_processed_messages, _deadline,
    )


async def fetch_group_messages(include_today_processed_messages: bool = False, _deadline: float = 0.0) -> List[MessageInfo]:
    """Fetch messages from source groups in the last 24 hours."""
    processed_messages = load_group_summarization_history()
    logger.info("Loaded %d already processed group messages from history", len(processed_messages))

    return await _fetch_from_sources(
        list(SOURCE_GROUPS), processed_messages, "group", include_today_processed_messages, _deadline,
    )


async def edit_message_in_target_channel(message_id: int, new_message: str) -> None:
    """Редактирует сообщение в целевом канале."""
    from config import TARGET_CHANNEL

    try:
        await _ensure_bot_client()
        await bot_client.edit_message(
            TARGET_CHANNEL, message_id, new_message, parse_mode="html"
        )
        logger.info("Message %s edited in target channel", message_id)
    except Exception as e:
        logger.error("Error editing message %s in target channel: %s", message_id, e)


async def send_message_to_target_channel_with_id(message: str) -> Optional[int]:
    """Отправляет сообщение в целевой канал и возвращает message_id."""
    from config import TARGET_CHANNEL

    try:
        await _ensure_bot_client()
        sent_message = await bot_client.send_message(
            TARGET_CHANNEL, message, parse_mode="html"
        )
        logger.info("Message sent to target channel with ID: %s", sent_message.id)
        return sent_message.id
    except Exception as e:
        logger.error("Error sending message to target channel: %s", e)
        return None


async def start_clients() -> None:
    """Запускает клиенты Telegram."""
    global user_client, bot_client, clients_loop
    current_loop = asyncio.get_running_loop()

    # If clients were created on a different loop (e.g., previous Lambda invoke), recreate them
    if clients_loop is not None and clients_loop is not current_loop:
        try:
            if user_client is not None and user_client.is_connected():
                await user_client.disconnect()
            if bot_client is not None and bot_client.is_connected():
                await bot_client.disconnect()
        except Exception:
            pass
        user_client = None
        bot_client = None

    clients_loop = current_loop

    if user_client is None:
        user_client = TelegramClient(
            "tg_summarizer_user", API_ID, API_HASH,
            connection_retries=3, retry_delay=2, timeout=15,
        )
        user_client.parse_mode = "html"
    if bot_client is None:
        bot_client = TelegramClient(
            "tg_summarizer_bot", API_ID, API_HASH,
            connection_retries=3, retry_delay=2, timeout=15,
        )
    if not user_client.is_connected():
        await user_client.start()
    if not bot_client.is_connected():
        await bot_client.start(bot_token=BOT_TOKEN)


async def stop_clients() -> None:
    """Останавливает клиенты Telegram."""
    global user_client, bot_client, clients_loop
    if user_client is not None and user_client.is_connected():
        await user_client.disconnect()
    if bot_client is not None and bot_client.is_connected():
        await bot_client.disconnect()
    # Ensure fresh clients on next run (prevents event loop mismatch)
    user_client = None
    bot_client = None
    clients_loop = None
