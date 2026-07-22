import os
from typing import List

import boto3  # type: ignore
from botocore.exceptions import ClientError  # type: ignore


def get_s3_client():
    bucket = os.getenv("S3_BUCKET_NAME")
    if not bucket:
        return None, None
    region = os.getenv("AWS_REGION")
    s3 = boto3.client("s3", region_name=region) if region else boto3.client("s3")
    return s3, bucket


def files_to_sync() -> List[str]:
    """List of state files to synchronize with S3 if configured."""
    return [
        os.getenv("ABBREVIATIONS_FILE", "channel_abbreviations.json"),
        os.getenv("HISTORY_FILE", "summarization_history.json"),
        os.getenv("SUMMARIES_HISTORY_FILE", "summaries_history.json"),
        os.getenv("DISCOVERED_CHANNELS_FILE", "discovered_channels.json"),
        os.getenv("GROUP_HISTORY_FILE", "group_summarization_history.json"),
        os.getenv("GROUP_SUMMARIES_HISTORY_FILE", "group_summaries_history.json"),
        os.getenv("GROUP_LAST_RUN_FILE", "group_last_run.json"),
        os.getenv("PROMPTS_FILE", "prompts.json"),
    ]


def session_files_to_sync() -> List[str]:
    """Telethon session files to persist across runs."""
    sessions_dir = os.getenv("SESSIONS_DIR", "sessions")
    return [
        "tg_summarizer_user.session",
        "tg_summarizer_bot.session",
        os.path.join(sessions_dir, "tg_summarizer_user.session"),
        os.path.join(sessions_dir, "tg_summarizer_bot.session"),
    ]


def ensure_dirs(paths: List[str]) -> None:
    for path in paths:
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)


def s3_key_for_local(path: str) -> str:
    prefix = os.getenv("S3_PREFIX", "").strip("/")
    key = path.lstrip("./")
    return f"{prefix}/{key}" if prefix else key


def download_from_s3() -> None:
    s3, bucket = get_s3_client()
    if not s3:
        print("S3 not configured, skipping download")
        return
    paths = files_to_sync() + session_files_to_sync()
    ensure_dirs(paths)
    for path in paths:
        key = s3_key_for_local(path)
        try:
            s3.download_file(bucket, key, path)
            print(f"Downloaded {key} -> {path}")
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey"):
                print(f"No object {key} in bucket {bucket}, skipping")
            else:
                print(f"Failed to download {key}: {error}")


def upload_to_s3() -> None:
    s3, bucket = get_s3_client()
    if not s3:
        print("S3 not configured, skipping upload")
        return
    paths = [
        path
        for path in files_to_sync() + session_files_to_sync()
        if os.path.exists(path)
    ]
    for path in paths:
        key = s3_key_for_local(path)
        try:
            s3.upload_file(path, bucket, key)
            print(f"Uploaded {path} -> {key}")
        except ClientError as error:
            print(f"Failed to upload {path}: {error}")
