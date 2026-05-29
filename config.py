import os
from dotenv import load_dotenv

load_dotenv()


def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return parsed


_REQUIRED_VARS = {
    "TELEGRAM_API_ID": "API_ID",
    "TELEGRAM_API_HASH": "API_HASH",
    "TELEGRAM_BOT_TOKEN": "BOT_TOKEN",
    "TARGET_CHANNEL": "TARGET_CHANNEL",
    "OPENAI_API_KEY": "OPENAI_API_KEY",
}

_api_id_str = os.getenv('TELEGRAM_API_ID')
API_ID = int(_api_id_str) if _api_id_str else None

API_HASH = os.getenv('TELEGRAM_API_HASH') or None

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN') or None

_target_channel = os.getenv('TARGET_CHANNEL') or None
TARGET_CHANNEL: int | str | None = _target_channel
if isinstance(TARGET_CHANNEL, str) and TARGET_CHANNEL.startswith("-"):
    TARGET_CHANNEL = int(TARGET_CHANNEL)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or None


def validate_config() -> None:
    """Validate that all required environment variables are set.

    Call this at entry points (lambda handler, CLI) before doing real work.
    Importing the module alone will NOT raise — this allows testing and
    partial module loading without a full .env file.
    """
    missing = []
    for env_name, attr_name in _REQUIRED_VARS.items():
        if globals().get(attr_name) is None:
            missing.append(env_name)
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}"
        )


OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_DEFAULT_MAX_TOKENS = _get_int_env("OPENAI_DEFAULT_MAX_TOKENS", 300)
OPENAI_CHANNEL_SUMMARY_MAX_TOKENS = _get_int_env(
    "OPENAI_CHANNEL_SUMMARY_MAX_TOKENS", 4000
)
OPENAI_GROUP_SUMMARY_MAX_TOKENS = _get_int_env(
    "OPENAI_GROUP_SUMMARY_MAX_TOKENS", 4000
)

source_channels_str = os.getenv('SOURCE_CHANNELS', '')
SOURCE_CHANNELS = set([c.strip() for c in source_channels_str.split(',') if c.strip()])

# Новые переменные для групп
source_groups_str = os.getenv('SOURCE_GROUPS', '')
SOURCE_GROUPS = set([g.strip() for g in source_groups_str.split(',') if g.strip()])

# File paths
ABBREVIATIONS_FILE = os.getenv('ABBREVIATIONS_FILE', 'channel_abbreviations.json')
HISTORY_FILE = os.getenv('HISTORY_FILE', 'summarization_history.json')
SUMMARIES_HISTORY_FILE = os.getenv('SUMMARIES_HISTORY_FILE', 'summaries_history.json')
DISCOVERED_CHANNELS_FILE = os.getenv('DISCOVERED_CHANNELS_FILE', 'discovered_channels.json')
GROUP_HISTORY_FILE = os.getenv('GROUP_HISTORY_FILE', 'group_summarization_history.json')
GROUP_SUMMARIES_HISTORY_FILE = os.getenv('GROUP_SUMMARIES_HISTORY_FILE', 'group_summaries_history.json')
GROUP_LAST_RUN_FILE = os.getenv('GROUP_LAST_RUN_FILE', 'group_last_run.json')
PROMPTS_FILE = os.getenv("PROMPTS_FILE", "prompts.json")

# Constants
SIMILARITY_LLM_LOWER = float(os.getenv("SIMILARITY_LLM_LOWER", "0.7"))
SIMILARITY_LLM_UPPER = float(os.getenv("SIMILARITY_LLM_UPPER", "0.95"))
ENABLE_SUMMARIES_DEDUPLICATION = True
ENABLE_SUMMARY_UPDATES = True

# History limits (configurable via environment variables)
MAX_CHANNEL_HISTORY_MESSAGES = int(os.getenv("MAX_CHANNEL_HISTORY_MESSAGES", "1000"))
MAX_CHANNEL_SUMMARIES = int(os.getenv("MAX_CHANNEL_SUMMARIES", "50"))
MAX_GROUP_HISTORY_MESSAGES = int(os.getenv("MAX_GROUP_HISTORY_MESSAGES", "1000"))
MAX_GROUP_SUMMARIES = int(os.getenv("MAX_GROUP_SUMMARIES", "100"))

# Time intervals (in seconds)
GROUP_SUMMARIZATION_INTERVAL_SECONDS = int(os.getenv("GROUP_SUMMARIZATION_INTERVAL_SECONDS", str(24 * 60 * 60)))
RESTORE_HISTORY_DAYS = int(os.getenv("RESTORE_HISTORY_DAYS", "7"))

# Summary size guardrails
SUMMARY_MIN_RATIO = 3  # total_original_length // SUMMARY_MIN_RATIO
SUMMARY_MIN_LENGTH = 800
SUMMARY_MAX_LENGTH = 4000
GROUP_SUMMARY_MIN_LENGTH = 2000
GROUP_SUMMARY_MAX_LENGTH = 12000
TEXT_PREVIEW_LENGTH = 50

# Coverage check context limits (major cost optimization)
COVERAGE_CHECK_MAX_SUMMARIES = int(os.getenv("COVERAGE_CHECK_MAX_SUMMARIES", "10"))
COVERAGE_CHECK_MAX_CHARS_PER_SUMMARY = int(os.getenv("COVERAGE_CHECK_MAX_CHARS_PER_SUMMARY", "300"))
UPDATE_MATCH_MAX_SUMMARIES = int(os.getenv("UPDATE_MATCH_MAX_SUMMARIES", "5"))
UPDATE_MATCH_MAX_CHARS_PER_SUMMARY = int(os.getenv("UPDATE_MATCH_MAX_CHARS_PER_SUMMARY", "500"))

# Debug mode (set DEBUG=1 in environment to enable verbose logging)
DEBUG = os.getenv("DEBUG", "").lower() in {"1", "true", "yes"}
