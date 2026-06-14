import hashlib
import json as _json
import os
import re
import sys
import asyncio
import logging
import time as _time
from datetime import datetime, timezone
from openai import AsyncOpenAI, APIError, RateLimitError, APIConnectionError
from config import OPENAI_API_KEY, OPENAI_DEFAULT_MAX_TOKENS, OPENAI_MODEL, OPENAI_REQUEST_TIMEOUT

logger = logging.getLogger(__name__)


def load_json_file(filepath: str, default: dict = None) -> dict:
    """Generic JSON file loader with error handling."""
    if default is None:
        default = {}
    if not os.path.exists(filepath):
        return default
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return _json.load(f)
    except Exception as e:
        logger.error("Error loading %s: %s", filepath, e)
        return default


def save_json_file(filepath: str, data: dict, error_msg: str) -> bool:
    """Generic JSON file saver with error handling."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            _json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error("%s: %s", error_msg, e)
        return False


def now_iso() -> str:
    """Returns current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()


openai_client = None

LINK_REGEX = re.compile(r"https?://\S+")
TRAILING_PUNCTUATION_REGEX = re.compile(r"""[.,;:!?)\]}'">]+$""")
TELEGRAM_CHANNEL_REGEX = re.compile(r"https://t\.me/([^/]+)/\d+")
ABBREVIATION_REGEX = re.compile(r'\[([A-Z0-9]+)\]')
HTML_TAG_REGEX = re.compile(r'<[^>]+>')


def count_characters(text: str) -> int:
    """Подсчитывает количество символов в тексте, исключая HTML-теги."""
    clean_text = HTML_TAG_REGEX.sub('', text)
    return len(clean_text)


def extract_telegram_channels(text: str) -> list[str]:
    """Извлекает названия каналов из ссылок вида https://t.me/channel_name/message_id."""
    channels = []
    matches = TELEGRAM_CHANNEL_REGEX.findall(text)
    for match in matches:
        channel_name = match.strip()
        if channel_name and channel_name not in channels:
            channels.append(channel_name)
    return channels


def extract_channels_from_abbreviations(text: str) -> list[str]:
    """Извлекает названия каналов из аббревиатур в квадратных скобках."""
    from channel_manager import load_channel_abbreviations

    abbreviations = load_channel_abbreviations()
    reverse_abbreviations = {v: k for k, v in abbreviations.items()}
    matches = ABBREVIATION_REGEX.findall(text)
    channels = []

    for match in matches:
        abbreviation = match.strip()
        if abbreviation in reverse_abbreviations:
            channel_name = reverse_abbreviations[abbreviation]
            if channel_name not in channels:
                channels.append(channel_name)

    return channels


def extract_all_channels(text: str) -> list[str]:
    """Извлекает все каналы из текста: из ссылок и из аббревиатур."""
    link_channels = extract_telegram_channels(text)
    abbreviation_channels = extract_channels_from_abbreviations(text)
    all_channels = list(set(link_channels + abbreviation_channels))
    return all_channels


def _emit_openai_latency(elapsed: float, model: str, total_tokens: int) -> None:
    """Emit OpenAI latency as a CloudWatch Embedded Metric Format (EMF) metric.

    Prints a JSON line to stdout that CloudWatch automatically processes into a metric.
    No additional API calls or dependencies required.
    """
    try:
        emf = {
            "_aws": {
                "Timestamp": int(_time.time() * 1000),
                "CloudWatchMetrics": [{
                    "Namespace": "tg_summarizer/OpenAI",
                    "Dimensions": [["Function", "Model"]],
                    "Metrics": [{"Name": "Latency", "Unit": "Seconds"}]
                }]
            },
            "Function": os.getenv("AWS_LAMBDA_FUNCTION_NAME", "local"),
            "Model": model,
            "Latency": round(elapsed, 2),
            "TotalTokens": total_tokens,
        }
        sys.stdout.write(_json.dumps(emf, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    except Exception:
        pass


async def call_openai(
    system_prompt: str,
    user_content: str,
    max_tokens: int = OPENAI_DEFAULT_MAX_TOKENS,
    max_retries: int = 3,
    base_delay: float = 1.0,
    temperature: float | None = None,
) -> str:
    """Универсальная функция для вызова OpenAI API с retry и exponential backoff."""
    global openai_client
    if openai_client is None:
        if not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY is not set — cannot create OpenAI client")
            return ""
        openai_client = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            timeout=float(OPENAI_REQUEST_TIMEOUT),
        )

    kwargs = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "model": OPENAI_MODEL,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    for attempt in range(max_retries + 1):
        try:
            t0 = _time.monotonic()
            response = await openai_client.chat.completions.create(**kwargs)
            elapsed = _time.monotonic() - t0
            result = response.choices[0].message.content
            if result is None:
                return ""
            total_tokens = response.usage.total_tokens if hasattr(response, 'usage') and response.usage else 0
            if hasattr(response, 'usage') and response.usage:
                logger.info(
                    "OpenAI usage: model=%s prompt=%d completion=%d total=%d latency=%.1fs",
                    OPENAI_MODEL,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    response.usage.total_tokens,
                    elapsed,
                )
            else:
                logger.info("OpenAI call: model=%s latency=%.1fs", OPENAI_MODEL, elapsed)
            _emit_openai_latency(elapsed, OPENAI_MODEL, total_tokens)
            return result.strip()
        except (RateLimitError, APIConnectionError) as e:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning("OpenAI retryable error (attempt %d/%d): %s; retrying in %.1fs", attempt + 1, max_retries + 1, e, delay)
                await asyncio.sleep(delay)
            else:
                logger.error("OpenAI API error after %d retries: %s", max_retries, e)
                return ""
        except APIError as e:
            if e.status_code and e.status_code >= 500 and attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning("OpenAI 5xx error (attempt %d/%d): %s; retrying in %.1fs", attempt + 1, max_retries + 1, e, delay)
                await asyncio.sleep(delay)
            elif e.status_code and e.status_code in (401, 403):
                logger.error("OpenAI auth error (status %d): check OPENAI_API_KEY — %s", e.status_code, e)
                openai_client = None
                return ""
            else:
                logger.error("OpenAI API error (status %s): %s", getattr(e, 'status_code', '?'), e)
                return ""
        except Exception as e:
            logger.error("OpenAI API error: %s", e)
            return ""
    return ""


def text_hash(text: str) -> str:
    """Deterministic SHA-256 hash of text (None-safe)."""
    return hashlib.sha256((text or "").encode()).hexdigest()[:16]


def extract_links(text: str) -> list[str]:
    """Return all URLs from a string, with trailing punctuation stripped."""
    raw = LINK_REGEX.findall(text)
    return [TRAILING_PUNCTUATION_REGEX.sub("", url) for url in raw]


def _truncate_html_preserving_tags(text: str, max_visible_chars: int) -> str:
    if max_visible_chars <= 0:
        return ""

    result: list[str] = []
    open_tags: list[str] = []
    visible_chars = 0
    i = 0
    truncated = False

    while i < len(text):
        char = text[i]
        if char == "<":
            end = text.find(">", i)
            if end == -1:
                break
            tag = text[i:end + 1]
            result.append(tag)

            tag_body = tag[1:-1].strip()
            if tag_body and not tag_body.startswith(("!", "?")):
                is_closing = tag_body.startswith("/")
                tag_name = tag_body[1:].split()[0].lower() if is_closing else tag_body.split()[0].lower()
                is_self_closing = tag_body.endswith("/")
                if is_closing:
                    if open_tags and open_tags[-1] == tag_name:
                        open_tags.pop()
                elif not is_self_closing:
                    open_tags.append(tag_name)
            i = end + 1
            continue

        if visible_chars >= max_visible_chars:
            truncated = True
            break

        result.append(char)
        visible_chars += 1
        i += 1

    output = "".join(result).rstrip()
    if truncated and visible_chars > 0 and not output.endswith("..."):
        output = output.rstrip(" ,;:\n") + "..."

    for tag_name in reversed(open_tags):
        output += f"</{tag_name}>"

    return output.strip()


def enforce_summary_length(summary: str, max_visible_chars: int) -> str:
    if count_characters(summary) <= max_visible_chars:
        return summary.strip()

    blocks = [block.strip() for block in summary.split("\n\n") if block.strip()]
    if blocks:
        kept_blocks: list[str] = []
        for block in blocks:
            candidate = "\n\n".join(kept_blocks + [block])
            if count_characters(candidate) > max_visible_chars:
                break
            kept_blocks.append(block)
        if kept_blocks:
            return "\n\n".join(kept_blocks).strip()

    return _truncate_html_preserving_tags(summary, max_visible_chars)
