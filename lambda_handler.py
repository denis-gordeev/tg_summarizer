import json as _json
import os
import logging
import asyncio
import sys
import time
from typing import Any, Dict
from s3_sync import download_from_s3, upload_to_s3
from summarizer import run_summarizer, DeadlineExceededError
from utils import get_circuit_breaker_state, reset_circuit_breaker, get_token_usage, reset_token_usage, _estimate_cost_usd
from config import OPENAI_MODEL

SAFETY_MARGIN_SECONDS = 10

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)

logger = logging.getLogger(__name__)


def _emit_invocation_summary(
    prompt_tokens: int, completion_tokens: int, elapsed_seconds: float, model: str
) -> None:
    """Emit a per-invocation EMF metric with cumulative cost/tokens (no Model dimension).

    Unlike the per-call EMF in utils._emit_openai_latency (which has Function+Model
    dimensions), this metric uses only the Function dimension, enabling CloudWatch
    aggregation across models for daily cost alerting.
    """
    if prompt_tokens == 0 and completion_tokens == 0:
        return
    try:
        cost_usd = _estimate_cost_usd(model, prompt_tokens, completion_tokens)
        emf = {
            "_aws": {
                "Timestamp": int(time.time() * 1000),
                "CloudWatchMetrics": [{
                    "Namespace": "tg_summarizer/Invocation",
                    "Dimensions": [["Function"]],
                    "Metrics": [
                        {"Name": "CumulativePromptTokens", "Unit": "None"},
                        {"Name": "CumulativeCompletionTokens", "Unit": "None"},
                        {"Name": "CumulativeCostUSD", "Unit": "None"},
                        {"Name": "ElapsedSeconds", "Unit": "Seconds"},
                    ]
                }]
            },
            "Function": os.getenv("AWS_LAMBDA_FUNCTION_NAME", "local"),
            "CumulativePromptTokens": prompt_tokens,
            "CumulativeCompletionTokens": completion_tokens,
            "CumulativeCostUSD": round(cost_usd, 6),
            "ElapsedSeconds": round(elapsed_seconds, 1),
        }
        sys.stdout.write(_json.dumps(emf, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    except Exception:
        pass


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


async def _warmup_telegram():
    """Start and stop Telegram clients to refresh sessions and keep the execution environment warm."""
    from telegram_client import start_clients, stop_clients
    await start_clients()
    await stop_clients()


def _emit_warmup_metric(success: bool, elapsed_seconds: float) -> None:
    try:
        emf = {
            "_aws": {
                "Timestamp": int(time.time() * 1000),
                "CloudWatchMetrics": [{
                    "Namespace": "tg_summarizer/Warmup",
                    "Dimensions": [["Function"]],
                    "Metrics": [
                        {"Name": "Success", "Unit": "None"},
                        {"Name": "ElapsedSeconds", "Unit": "Seconds"},
                    ]
                }]
            },
            "Function": os.getenv("AWS_LAMBDA_FUNCTION_NAME", "local"),
            "Success": 1 if success else 0,
            "ElapsedSeconds": round(elapsed_seconds, 1),
        }
        sys.stdout.write(_json.dumps(emf, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    except Exception:
        pass


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    if not isinstance(event, dict):
        event = {}
    request_id = getattr(context, 'aws_request_id', None) if context else None
    if request_id:
        logger.info("Lambda invocation %s", request_id)

    start_time = time.monotonic()

    reset_circuit_breaker()
    reset_token_usage()

    try:
        os.chdir('/tmp')
    except Exception:
        pass

    t0 = time.monotonic()
    download_from_s3()
    t_download = time.monotonic() - t0

    warmup = _parse_event_flag(event.get('warmup'), False)
    if warmup:
        logger.info("Lambda warmup — connecting Telegram clients")
        warmup_success = True
        try:
            asyncio.run(_warmup_telegram())
        except Exception as e:
            logger.warning("Warmup Telegram connection failed: %s", e)
            warmup_success = False
        upload_to_s3()
        elapsed = time.monotonic() - start_time
        _emit_warmup_metric(warmup_success, elapsed)
        return {
            'status': 'warmed',
            'request_id': request_id,
            'elapsed_seconds': round(elapsed, 1),
        }

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
        s3_upload_result = upload_to_s3()
        cb_state = get_circuit_breaker_state()
        token_usage = get_token_usage()
        _emit_invocation_summary(
            token_usage["prompt_tokens"], token_usage["completion_tokens"], elapsed, OPENAI_MODEL
        )
        return {
            'status': 'error',
            'error': str(e),
            'error_type': error_type,
            'request_id': request_id,
            'elapsed_seconds': round(elapsed, 1),
            'send_message': send_message,
            'save_changes': save_changes,
            'model': OPENAI_MODEL,
            'circuit_breaker': cb_state,
            'token_usage': token_usage,
            's3_upload': s3_upload_result,
        }
    t_summarizer = time.monotonic() - t0

    # Push updated state back to S3
    t0 = time.monotonic()
    s3_upload_result = upload_to_s3()
    t_upload = time.monotonic() - t0

    if s3_upload_result.get("failed", 0) > 0:
        logger.warning(
            "S3 upload had %d failures out of %d files — state may be incomplete",
            s3_upload_result["failed"],
            s3_upload_result["uploaded"] + s3_upload_result["failed"] + s3_upload_result.get("skipped_empty", 0),
        )

    elapsed = time.monotonic() - start_time
    cb_state = get_circuit_breaker_state()
    token_usage = get_token_usage()
    _emit_invocation_summary(
        token_usage["prompt_tokens"], token_usage["completion_tokens"], elapsed, OPENAI_MODEL
    )
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
        'model': OPENAI_MODEL,
        'circuit_breaker': cb_state,
        'token_usage': token_usage,
        's3_upload': s3_upload_result,
    }
