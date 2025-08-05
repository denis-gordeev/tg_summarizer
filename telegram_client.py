import asyncio
import random
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from telethon import TelegramClient
from telethon.tl.functions.channels import GetChannelRecommendationsRequest
from telethon.tl.types import InputChannel

from config import API_HASH, API_ID, BOT_TOKEN, SOURCE_CHANNELS, SOURCE_GROUPS
from history_manager import load_group_summarization_history, load_summarization_history
from message_processor import get_all_source_channels, is_message_processed
from models import MessageInfo
from utils import extract_links

# Create separate clients for user (reading) and bot (sending)
user_client = TelegramClient("tg_summarizer_user", API_ID, API_HASH)
bot_client = TelegramClient("tg_summarizer_bot", API_ID, API_HASH)


async def get_similar_channels_from_telegram(channel_username: Optional[str] = None) -> List[str]:
    """Получает список похожих каналов через Telegram API."""
    try:
        if not user_client.is_connected():
            await user_client.start()

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
                print(f"Ошибка при получении рекомендаций для канала {channel_username}: {e}")
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
        print(f"Ошибка при получении похожих каналов: {e}")
        return []


async def fetch_messages(include_today_processed_messages: bool = False) -> List[MessageInfo]:
    """Fetch messages from source channels in the last 24 hours."""
    since = datetime.now(timezone.utc) - timedelta(days=1)
    all_msgs = []

    # Загружаем историю обработанных сообщений
    processed_messages = load_summarization_history()
    print(f"Загружено {len(processed_messages)} уже обработанных сообщений из истории")

    # Получаем объединенный список каналов
    all_channels = get_all_source_channels()
    random.shuffle(all_channels)

    for channel in all_channels:
        print(f"Fetching messages from {channel}...")
        channel_msgs = []
        async for msg in user_client.iter_messages(
            channel, offset_date=None, min_id=0, reverse=False
        ):
            if msg.date < since:
                break
            if msg.message:
                # Извлекаем ссылки из текста сообщения
                links = extract_links(msg.message)
                main_link = links[0] if links else ""

                message_info = MessageInfo(
                    text=msg.message,
                    channel=channel,
                    message_id=msg.id,
                    date=msg.date,
                    link=main_link,
                )

                # Проверяем, не было ли сообщение уже обработано (если не игнорируем)
                if include_today_processed_messages or not is_message_processed(
                    message_info, processed_messages
                ):
                    channel_msgs.append(message_info)
                    all_msgs.append(message_info)
                else:
                    print(f"  Пропускаем уже обработанное сообщение {msg.id} из {channel}")

        print(f"  Found {len(channel_msgs)} новых сообщений from {channel}")
    return all_msgs


async def fetch_group_messages(include_today_processed_messages: bool = False) -> List[MessageInfo]:
    """Fetch messages from source groups in the last 24 hours."""
    since = datetime.now(timezone.utc) - timedelta(days=1)
    all_msgs = []

    # Загружаем историю обработанных сообщений из групп
    processed_messages = load_group_summarization_history()
    print(f"Загружено {len(processed_messages)} уже обработанных сообщений из групп из истории")

    # Получаем список групп
    all_groups = SOURCE_GROUPS

    for group in all_groups:
        print(f"Fetching messages from group {group}...")
        group_msgs = []
        async for msg in user_client.iter_messages(
            group, offset_date=None, min_id=0, reverse=False
        ):
            if msg.date < since:
                break
            if msg.message:
                # Извлекаем ссылки из текста сообщения
                links = extract_links(msg.message)
                main_link = links[0] if links else ""

                message_info = MessageInfo(
                    text=msg.message,
                    channel=group,
                    message_id=msg.id,
                    date=msg.date,
                    link=main_link,
                )

                # Проверяем, не было ли сообщение уже обработано (если не игнорируем)
                if include_today_processed_messages or not is_message_processed(
                    message_info, processed_messages
                ):
                    group_msgs.append(message_info)
                    all_msgs.append(message_info)
                else:
                    print(f"  Пропускаем уже обработанное сообщение {msg.id} из группы {group}")

        print(f"  Found {len(group_msgs)} новых сообщений from group {group}")
    return all_msgs


async def send_message_to_target_channel(message: str) -> None:
    """Отправляет сообщение в целевой канал."""
    from config import TARGET_CHANNEL

    try:
        await bot_client.send_message(TARGET_CHANNEL, message, parse_mode="html")
        print("Message sent to target channel")
    except Exception as e:
        print(f"Error sending message to target channel: {e}")


async def edit_message_in_target_channel(message_id: int, new_message: str) -> None:
    """Редактирует сообщение в целевом канале."""
    from config import TARGET_CHANNEL

    try:
        await bot_client.edit_message(TARGET_CHANNEL, message_id, new_message, parse_mode="html")
        print(f"Message {message_id} edited in target channel")
    except Exception as e:
        print(f"Error editing message {message_id} in target channel: {e}")


async def send_message_to_target_channel_with_id(message: str) -> Optional[int]:
    """Отправляет сообщение в целевой канал и возвращает message_id."""
    from config import TARGET_CHANNEL

    try:
        sent_message = await bot_client.send_message(TARGET_CHANNEL, message, parse_mode="html")
        print(f"Message sent to target channel with ID: {sent_message.id}")
        return sent_message.id
    except Exception as e:
        print(f"Error sending message to target channel: {e}")
        return None


async def start_clients() -> None:
    """Запускает клиенты Telegram."""
    await user_client.start()
    await bot_client.start(bot_token=BOT_TOKEN)


async def stop_clients() -> None:
    """Останавливает клиенты Telegram."""
    await user_client.disconnect()
    await bot_client.disconnect()
