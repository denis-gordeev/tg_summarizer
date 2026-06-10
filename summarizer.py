import asyncio
import logging
import time
from telegram_client import start_clients, stop_clients, fetch_messages, fetch_group_messages
from message_processor import process_messages
from config import SOURCE_GROUPS
from history_manager import should_run_group_summarization

logger = logging.getLogger(__name__)


class DeadlineExceededError(Exception):
    pass


def check_deadline(deadline: float) -> None:
    """Raise DeadlineExceededError if we've passed the safety deadline."""
    if deadline and time.monotonic() > deadline:
        raise DeadlineExceededError(
            f"Approaching Lambda timeout — stopping with {deadline - time.monotonic():.1f}s remaining"
        )


async def run_summarizer(
    send_message: bool = True,
    save_changes: bool = True,
    include_today_processed_groups: bool = False,
    include_today_processed_messages: bool = False,
    _deadline: float = 0.0,
):
    await start_clients()
    try:
        logger.info("=== Processing Channels ===")
        channel_messages = await fetch_messages(include_today_processed_messages, _deadline=_deadline)
        check_deadline(_deadline)
        await process_messages(channel_messages, save_changes, send_message, _deadline=_deadline)

        if SOURCE_GROUPS and (should_run_group_summarization() or include_today_processed_groups):
            check_deadline(_deadline)
            logger.info("=== Processing Groups ===")
            group_messages = await fetch_group_messages(include_today_processed_messages, _deadline=_deadline)
            await process_messages(group_messages, save_changes, send_message, is_group=True, _deadline=_deadline)
        else:
            logger.info("Group summarization skipped.")
    except DeadlineExceededError:
        logger.warning("Deadline exceeded — saving partial results and exiting gracefully")
    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)
    finally:
        await stop_clients()


def main(
    send_message: bool = True,
    save_changes: bool = True,
    include_today_processed_groups: bool = False,
    include_today_processed_messages: bool = False
):
    """Main function to run the summarizer."""
    from config import validate_config
    validate_config()
    asyncio.run(run_summarizer(
        send_message, save_changes, include_today_processed_groups, include_today_processed_messages
    ))


if __name__ == "__main__":
    import fire

    fire.Fire(main)
