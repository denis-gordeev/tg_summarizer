import os
import json
import logging
import time
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


DEFAULT_SYNC_FILES = (
    "tg_summarizer_user.session",
    "tg_summarizer_bot.session",
    os.getenv("ABBREVIATIONS_FILE", "channel_abbreviations.json"),
    os.getenv("HISTORY_FILE", "summarization_history.json"),
    os.getenv("SUMMARIES_HISTORY_FILE", "summaries_history.json"),
    os.getenv("DISCOVERED_CHANNELS_FILE", "discovered_channels.json"),
    os.getenv("GROUP_HISTORY_FILE", "group_summarization_history.json"),
    os.getenv("GROUP_SUMMARIES_HISTORY_FILE", "group_summaries_history.json"),
    os.getenv("GROUP_LAST_RUN_FILE", "group_last_run.json"),
    os.getenv("PROMPTS_FILE", "prompts.json"),
)


_s3_client = None


def _get_s3_client():
    global _s3_client
    if _s3_client is not None:
        return _s3_client
    try:
        import boto3
    except ImportError:
        logger.error("boto3 is not installed; skipping S3 sync")
        return None
    _s3_client = boto3.client("s3")
    return _s3_client


def _get_sync_files() -> list[str]:
    configured = os.getenv("STATE_SYNC_FILES", "")
    if configured.strip():
        files = [item.strip() for item in configured.split(",") if item.strip()]
    else:
        files = [item for item in DEFAULT_SYNC_FILES if item]
    return list(dict.fromkeys(files))


def _iter_local_files() -> Iterable[tuple[str, Path]]:
    for relative_name in _get_sync_files():
        yield relative_name, Path(relative_name)


def _build_s3_key(prefix: str, relative_name: str) -> str:
    clean_prefix = prefix.strip("/")
    if clean_prefix:
        return f"{clean_prefix}/{relative_name}"
    return relative_name


def _is_json_state_file(relative_name: str) -> bool:
    return relative_name.endswith(".json")


def _validate_json_file(path: Path) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as e:
        logger.warning("Downloaded file %s is not valid JSON (%s) — removing", path, e)
        try:
            path.unlink()
        except OSError:
            pass
        return False


_S3_UPLOAD_RETRIES = 1
_S3_RETRY_DELAY_SEC = 1.0


def download_from_s3() -> None:
    bucket = os.getenv("STATE_S3_BUCKET", "").strip()
    if not bucket:
        logger.debug("STATE_S3_BUCKET is not set; skipping S3 download")
        return

    client = _get_s3_client()
    if client is None:
        return

    prefix = os.getenv("STATE_S3_PREFIX", "")
    downloaded = 0
    skipped = 0
    for relative_name, local_path in _iter_local_files():
        key = _build_s3_key(prefix, relative_name)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            client.download_file(bucket, key, str(local_path))
            if _is_json_state_file(relative_name) and not _validate_json_file(local_path):
                skipped += 1
                continue
            logger.debug("Downloaded s3://%s/%s -> %s", bucket, key, local_path)
            downloaded += 1
        except Exception as exc:
            logger.debug("Skipping download for s3://%s/%s: %s", bucket, key, exc)
            skipped += 1

    logger.info("S3 download: %d files downloaded, %d skipped", downloaded, skipped)

    if downloaded:
        try:
            from history_manager import invalidate_cache
            invalidate_cache()
        except ImportError:
            pass


def upload_to_s3() -> None:
    bucket = os.getenv("STATE_S3_BUCKET", "").strip()
    if not bucket:
        logger.debug("STATE_S3_BUCKET is not set; skipping S3 upload")
        return

    client = _get_s3_client()
    if client is None:
        return

    prefix = os.getenv("STATE_S3_PREFIX", "")
    uploaded = 0
    failed = 0
    skipped_empty = 0
    for relative_name, local_path in _iter_local_files():
        if not local_path.exists():
            continue
        if local_path.stat().st_size == 0:
            logger.warning("Skipping upload of empty file %s — would overwrite valid S3 state", relative_name)
            skipped_empty += 1
            continue
        key = _build_s3_key(prefix, relative_name)
        uploaded_ok = False
        for attempt in range(_S3_UPLOAD_RETRIES + 1):
            try:
                client.upload_file(str(local_path), bucket, key)
                logger.debug("Uploaded %s -> s3://%s/%s", local_path, bucket, key)
                uploaded_ok = True
                break
            except Exception as exc:
                if attempt < _S3_UPLOAD_RETRIES:
                    logger.warning("Upload attempt %d failed for %s, retrying: %s", attempt + 1, relative_name, exc)
                    time.sleep(_S3_RETRY_DELAY_SEC)
                else:
                    logger.error("Failed to upload %s to s3://%s/%s after %d attempts: %s", local_path, bucket, key, attempt + 1, exc)
        if uploaded_ok:
            uploaded += 1
        else:
            failed += 1

    logger.info("S3 upload: %d files uploaded, %d failed, %d skipped (empty)", uploaded, failed, skipped_empty)
