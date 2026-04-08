import os
from pathlib import Path
from typing import Iterable


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


def _get_s3_client():
    try:
        import boto3
    except ImportError:
        print("boto3 is not installed; skipping S3 sync")
        return None
    return boto3.client("s3")


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


def download_from_s3() -> None:
    bucket = os.getenv("STATE_S3_BUCKET", "").strip()
    if not bucket:
        print("STATE_S3_BUCKET is not set; skipping S3 download")
        return

    client = _get_s3_client()
    if client is None:
        return

    prefix = os.getenv("STATE_S3_PREFIX", "")
    for relative_name, local_path in _iter_local_files():
        key = _build_s3_key(prefix, relative_name)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            client.download_file(bucket, key, str(local_path))
            print(f"Downloaded s3://{bucket}/{key} -> {local_path}")
        except Exception as exc:
            print(f"Skipping download for s3://{bucket}/{key}: {exc}")


def upload_to_s3() -> None:
    bucket = os.getenv("STATE_S3_BUCKET", "").strip()
    if not bucket:
        print("STATE_S3_BUCKET is not set; skipping S3 upload")
        return

    client = _get_s3_client()
    if client is None:
        return

    prefix = os.getenv("STATE_S3_PREFIX", "")
    for relative_name, local_path in _iter_local_files():
        if not local_path.exists():
            continue
        key = _build_s3_key(prefix, relative_name)
        try:
            client.upload_file(str(local_path), bucket, key)
            print(f"Uploaded {local_path} -> s3://{bucket}/{key}")
        except Exception as exc:
            print(f"Failed to upload {local_path} to s3://{bucket}/{key}: {exc}")
