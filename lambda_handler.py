import os
import logging
import asyncio
import time
from typing import Any, Dict
from s3_sync import download_from_s3, upload_to_s3
from summarizer import run_summarizer, DeadlineExceededError
from utils import get_circuit_breaker_state, reset_circuit_breaker, get_token_usage, reset_token_usage

SAFETY_MARGIN_SECONDS = 10

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)

logger = logging.getLogger(__name__)


def _classify_error(exc: Exception) -> str:
    if isinstance(exc, DeadlineExceededError):
        return "timeout"
    if isinstance(exc, ValueError):
        return "config"
    exc_name = type(exc).__name__.lower()
    if "connection" in exc_name or "timeout" in exc_name:
        return "connection"
    return "runtime"

def _parse_event_flag(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    return default


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    request_id = getattr(context, 'aws_request_id', None) if context else None
    if request_id:
        logger.info("Lambda invocation %s", request_id)

    start_time = time.monotonic()

    reset_circuit_breaker()
    reset_token_usage()

    # Ensure we can write files (sessions, history) in Lambda
    try:
        os.chdir('/tmp')
    except Exception:
        pass

    # Pull state from S3 if configured
    t0 = time.monotonic()
    download_from_s3()
    t_download = time.monotonic() - t0

    # Defaults can be overridden via EventBridge input transform
    send_message = _parse_event_flag(event.get('send_message'), True)
    save_changes = _parse_event_flag(event.get('save_changes'), True)
    include_today_processed_groups = _parse_event_flag(
        event.get('include_today_processed_groups'), False
    )
    include_today_processed_messages = _parse_event_flag(
        event.get('include_today_processed_messages'), False
    )

    logger.info(
        "Lambda event flags: send=%s, save=%s, today_groups=%s, today_msgs=%s",
        send_message, save_changes, include_today_processed_groups,
        include_today_processed_messages,
    )

    # Compute a hard deadline so we never exceed Lambda timeout
    if context is not None and hasattr(context, 'get_remaining_time_in_millis'):
        deadline = time.monotonic() + context.get_remaining_time_in_millis() / 1000.0 - SAFETY_MARGIN_SECONDS
    else:
        deadline = time.monotonic() + 180 - SAFETY_MARGIN_SECONDS

    t0 = time.monotonic()
    try:
        from config import validate_config
        validate_config()

        # Run the summarizer
        asyncio.run(
            run_summarizer(
                send_message=send_message,
                save_changes=save_changes,
                include_today_processed_groups=include_today_processed_groups,
                include_today_processed_messages=include_today_processed_messages,
                _deadline=deadline,
            )
        )
    except Exception as e:
        t_summarizer = time.monotonic() - t0
        elapsed = time.monotonic() - start_time
        error_type = _classify_error(e)
        logger.error("Lambda execution failed after %.1fs [%s]: %s", elapsed, error_type, e, exc_info=True)
        upload_to_s3()
        cb_state = get_circuit_breaker_state()
        token_usage = get_token_usage()
        return {
            'status': 'error',
            'error': str(e),
            'error_type': error_type,
            'request_id': request_id,
            'elapsed_seconds': round(elapsed, 1),
            'send_message': send_message,
            'save_changes': save_changes,
            'circuit_breaker': cb_state,
            'token_usage': token_usage,
        }
    t_summarizer = time.monotonic() - t0

    # Push updated state back to S3
    t0 = time.monotonic()
    upload_to_s3()
    t_upload = time.monotonic() - t0

    elapsed = time.monotonic() - start_time
    cb_state = get_circuit_breaker_state()
    token_usage = get_token_usage()
    logger.info(
        "Lambda completed in %.1fs (download=%.1fs, summarizer=%.1fs, upload=%.1fs) "
        "[send=%s, save=%s, today_groups=%s, today_msgs=%s] [cb=%s] [tokens=%d+%d]",
        elapsed, t_download, t_summarizer, t_upload,
        send_message, save_changes, include_today_processed_groups,
        include_today_processed_messages, cb_state["state"],
        token_usage["prompt_tokens"], token_usage["completion_tokens"],
    )

    return {
        'status': 'ok',
        'request_id': request_id,
        'elapsed_seconds': round(elapsed, 1),
        'send_message': send_message,
        'save_changes': save_changes,
        'include_today_processed_groups': include_today_processed_groups,
        'include_today_processed_messages': include_today_processed_messages,
        'circuit_breaker': cb_state,
        'token_usage': token_usage,
    }
