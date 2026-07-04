import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_ssm_client = None


def _get_ssm_client():
    """Get or create a cached SSM client."""
    global _ssm_client
    if _ssm_client is None:
        try:
            import boto3
            _ssm_client = boto3.client("ssm")
        except Exception as e:
            logger.debug("Could not create SSM client: %s", e)
            return None
    return _ssm_client


def _get_ssm_param(path: str) -> str | None:
    """Fetch a parameter from AWS SSM Parameter Store. Returns None on failure."""
    if not path:
        return None
    try:
        client = _get_ssm_client()
        if client is None:
            return None
        response = client.get_parameter(Name=path, WithDecryption=True)
        return response["Parameter"]["Value"]
    except Exception as e:
        logger.debug("SSM param %s not available: %s", path, e)
        return None


def _get_secret(env_name: str, ssm_env_name: str | None = None) -> str | None:
    """Resolve a secret: SSM path first (if configured), then env var."""
    ssm_path = os.getenv(ssm_env_name, "") if ssm_env_name else os.getenv(f"{env_name}_SSM_PATH", "")
    if ssm_path:
        value = _get_ssm_param(ssm_path)
        if value is not None:
            return value
    return os.getenv(env_name) or None


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


def _get_float_env(name: str, default: float, min_val: float = 0.0, max_val: float = float("inf")) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if parsed < min_val or parsed > max_val:
        raise ValueError(f"{name} must be between {min_val} and {max_val}")
    return parsed


_REQUIRED_VARS = {
    "TELEGRAM_API_ID": "API_ID",
    "TELEGRAM_API_HASH": "API_HASH",
    "TELEGRAM_BOT_TOKEN": "BOT_TOKEN",
    "TARGET_CHANNEL": "TARGET_CHANNEL",
    "OPENAI_API_KEY": "OPENAI_API_KEY",
}

_api_id_str = _get_secret('TELEGRAM_API_ID', 'TELEGRAM_API_ID_SSM_PATH')
API_ID = int(_api_id_str) if _api_id_str else None

API_HASH = _get_secret('TELEGRAM_API_HASH', 'TELEGRAM_API_HASH_SSM_PATH')

BOT_TOKEN = _get_secret('TELEGRAM_BOT_TOKEN', 'TELEGRAM_BOT_TOKEN_SSM_PATH')

_target_channel = os.getenv('TARGET_CHANNEL') or None
TARGET_CHANNEL: int | str | None = _target_channel
if isinstance(TARGET_CHANNEL, str) and TARGET_CHANNEL.startswith("-"):
    TARGET_CHANNEL = int(TARGET_CHANNEL)

OPENAI_API_KEY = _get_secret('OPENAI_API_KEY', 'OPENAI_API_KEY_SSM_PATH')


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


OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
OPENAI_DEFAULT_MAX_TOKENS = _get_int_env("OPENAI_DEFAULT_MAX_TOKENS", 300)
OPENAI_CHANNEL_SUMMARY_MAX_TOKENS = _get_int_env(
    "OPENAI_CHANNEL_SUMMARY_MAX_TOKENS", 1500
)
OPENAI_GROUP_SUMMARY_MAX_TOKENS = _get_int_env(
    "OPENAI_GROUP_SUMMARY_MAX_TOKENS", 4000
)
OPENAI_REQUEST_TIMEOUT = _get_int_env("OPENAI_REQUEST_TIMEOUT", 30)
OPENAI_SUMMARY_TEMPERATURE = _get_float_env("OPENAI_SUMMARY_TEMPERATURE", 0.1, min_val=0.0, max_val=2.0)

NLP_CHECK_MAX_INPUT_CHARS = _get_int_env("NLP_CHECK_MAX_INPUT_CHARS", 500)
COVERAGE_CHECK_MAX_INPUT_CHARS = _get_int_env("COVERAGE_CHECK_MAX_INPUT_CHARS", 500)
NLP_MIN_TEXT_LENGTH = _get_int_env("NLP_MIN_TEXT_LENGTH", 100)
MAX_MESSAGES_PER_SOURCE = _get_int_env("MAX_MESSAGES_PER_SOURCE", 100)
FETCH_EXAMINED_MULTIPLIER = _get_int_env("FETCH_EXAMINED_MULTIPLIER", 3)
SUMMARY_MAX_INPUT_CHARS_PER_MESSAGE = _get_int_env("SUMMARY_MAX_INPUT_CHARS_PER_MESSAGE", 3000)
NLP_CONCURRENT_CHECKS = _get_int_env("NLP_CONCURRENT_CHECKS", 5)
UPDATE_SUMMARY_MAX_INPUT_CHARS = _get_int_env("UPDATE_SUMMARY_MAX_INPUT_CHARS", 1000)

NLP_AD_KEYWORDS = [
    "курс", "вебинар", "регистраци", "скидк", "промокод",
    "мастер-класс", "стажировк", "hire", "hiring day",
    "карьерный трек", "bootcamp", "boot camp",
]

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
SIMILARITY_LLM_UPPER = _get_float_env("SIMILARITY_LLM_UPPER", 0.95, min_val=0.0, max_val=1.0)
ENABLE_SUMMARIES_DEDUPLICATION = os.getenv("ENABLE_SUMMARIES_DEDUPLICATION", "true").lower() not in {"0", "false", "no", "off"}
ENABLE_SUMMARY_UPDATES = os.getenv("ENABLE_SUMMARY_UPDATES", "true").lower() not in {"0", "false", "no", "off"}

# History limits (configurable via environment variables)
MAX_CHANNEL_HISTORY_MESSAGES = _get_int_env("MAX_CHANNEL_HISTORY_MESSAGES", 1000)
MAX_CHANNEL_SUMMARIES = _get_int_env("MAX_CHANNEL_SUMMARIES", 50)
MAX_GROUP_HISTORY_MESSAGES = _get_int_env("MAX_GROUP_HISTORY_MESSAGES", 1000)
MAX_GROUP_SUMMARIES = _get_int_env("MAX_GROUP_SUMMARIES", 100)

# Time intervals (in seconds)
GROUP_SUMMARIZATION_INTERVAL_SECONDS = _get_int_env("GROUP_SUMMARIZATION_INTERVAL_SECONDS", 24 * 60 * 60)
RESTORE_HISTORY_DAYS = _get_int_env("RESTORE_HISTORY_DAYS", 7)

# Summary size guardrails
SUMMARY_MIN_RATIO = 3  # total_original_length // SUMMARY_MIN_RATIO
SUMMARY_MIN_LENGTH = 800
SUMMARY_MAX_LENGTH = 4000
GROUP_SUMMARY_MIN_LENGTH = 2000
GROUP_SUMMARY_MAX_LENGTH = 12000
TEXT_PREVIEW_LENGTH = 50

# Coverage+match context limits (used by _check_coverage_and_match)
UPDATE_MATCH_MAX_SUMMARIES = _get_int_env("UPDATE_MATCH_MAX_SUMMARIES", 5)
UPDATE_MATCH_MAX_CHARS_PER_SUMMARY = _get_int_env("UPDATE_MATCH_MAX_CHARS_PER_SUMMARY", 500)
UPDATE_SUMMARY_MAX_TOKENS = _get_int_env("UPDATE_SUMMARY_MAX_TOKENS", 2000)
MAX_COVERED_MESSAGE_UPDATES = _get_int_env("MAX_COVERED_MESSAGE_UPDATES", 5)

# Debug mode (set DEBUG=1 in environment to enable verbose logging)
DEBUG = os.getenv("DEBUG", "").lower() in {"1", "true", "yes"}

# Restore timeout (seconds) for channel restore from Telegram
RESTORE_TIMEOUT_SEC = _get_int_env("RESTORE_TIMEOUT_SEC", 30)

# Telegram message length limit (platform constant, not configurable)
TELEGRAM_MAX_MESSAGE_LENGTH = 4096

# Circuit breaker settings
CIRCUIT_BREAKER_THRESHOLD = _get_int_env("CIRCUIT_BREAKER_THRESHOLD", 3)
CIRCUIT_BREAKER_RESET_SEC = _get_int_env("CIRCUIT_BREAKER_RESET_SEC", 60)
