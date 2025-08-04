import asyncio
import fire
from telegram_client import start_clients, stop_clients, fetch_messages, fetch_group_messages
from message_processor import process_messages
from config import SOURCE_GROUPS
from history_manager import should_run_group_summarization


async def run_summarizer(
    send_message: bool = True,
    save_changes: bool = True,
    include_today_processed_groups: bool = False,
    include_today_processed_messages: bool = False
):
    await start_clients()
    try:
        # Process channels
        print("=== Processing Channels ===")
        channel_messages = await fetch_messages(include_today_processed_messages)
        await process_messages(channel_messages, save_changes, send_message)

        # Process groups
        if SOURCE_GROUPS and (should_run_group_summarization() or include_today_processed_groups):
            print("\n=== Processing Groups ===")
            group_messages = await fetch_group_messages(include_today_processed_messages)
            await process_messages(group_messages, save_changes, send_message, is_group=True)
        else:
            print("\nGroup summarization skipped.")
    finally:
        await stop_clients()


def main(
    send_message: bool = True,
    save_changes: bool = True,
    include_today_processed_groups: bool = False,
    include_today_processed_messages: bool = False
):
    """Main function to run the summarizer."""
    asyncio.run(run_summarizer(
        send_message, save_changes, include_today_processed_groups, include_today_processed_messages
    ))


if __name__ == "__main__":
    fire.Fire(main)
