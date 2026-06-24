import hashlib
import html as _html
import json as _json
import os
import re
import sys
import asyncio
import logging
import tempfile
import time as _time
from datetime import datetime, timezone
from openai import AsyncOpenAI, APIError, RateLimitError, APIConnectionError
from config import OPENAI_API_KEY, OPENAI_DEFAULT_MAX_TOKENS, OPENAI_MODEL, OPENAI_REQUEST_TIMEOUT, CIRCUIT_BREAKER_THRESHOLD, CIRCUIT_BREAKER_RESET_SEC

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
    """Generic JSON file saver with error handling.

    Uses atomic write (write to temp file, then rename) to prevent
    file corruption if the process is killed mid-write (e.g. Lambda timeout).
    """
    try:
        dir_name = os.path.dirname(filepath) or "."
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                _json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, filepath)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        return True
    except Exception as e:
        logger.error("%s: %s", error_msg, e)
        return False


def now_iso() -> str:
    """Returns current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()


openai_client = None

_CIRCUIT_BREAKER_FAILURES = 0
_CIRCUIT_BREAKER_OPEN_SINCE: float = 0.0

_cumulative_prompt_tokens: int = 0
_cumulative_completion_tokens: int = 0

LINK_REGEX = re.compile(r"https?://\S+")
TRAILING_PUNCTUATION_REGEX = re.compile(r"""[.,;:!?)\]}'">]+$""")
TELEGRAM_CHANNEL_REGEX = re.compile(r"https://t\.me/([^/]+)/\d+")
ABBREVIATION_REGEX = re.compile(r'\[([A-Z0-9]+)\]')
HTML_TAG_REGEX = re.compile(r'<[^>]+>')

VOID_HTML_ELEMENTS = frozenset({
    "br", "hr", "img", "input", "meta", "link", "area", "base",
    "col", "embed", "source", "track", "wbr",
})


def count_characters(text: str) -> int:
    """Подсчитывает количество видимых символов в тексте, исключая HTML-теги и раскрывая сущности."""
    clean_text = HTML_TAG_REGEX.sub('', text)
    clean_text = _html.unescape(clean_text)
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


_COST_PER_MILLION = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
}

def _estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    input_per_m, output_per_m = _COST_PER_MILLION.get(model, (0.15, 0.60))
    return prompt_tokens * input_per_m / 1_000_000 + completion_tokens * output_per_m / 1_000_000


def _emit_openai_latency(elapsed: float, model: str, total_tokens: int, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
    """Emit OpenAI latency as a CloudWatch Embedded Metric Format (EMF) metric.

    Prints a JSON line to stdout that CloudWatch automatically processes into a metric.
    No additional API calls or dependencies required.
    """
    try:
        cost_usd = _estimate_cost_usd(model, prompt_tokens, completion_tokens)
        emf = {
            "_aws": {
                "Timestamp": int(_time.time() * 1000),
                "CloudWatchMetrics": [{
                    "Namespace": "tg_summarizer/OpenAI",
                    "Dimensions": [["Function", "Model"]],
                    "Metrics": [
                        {"Name": "Latency", "Unit": "Seconds"},
                        {"Name": "PromptTokens", "Unit": "None"},
                        {"Name": "CompletionTokens", "Unit": "None"},
                        {"Name": "EstimatedCostUSD", "Unit": "None"},
                    ]
                }]
            },
            "Function": os.getenv("AWS_LAMBDA_FUNCTION_NAME", "local"),
            "Model": model,
            "Latency": round(elapsed, 2),
            "PromptTokens": prompt_tokens,
            "CompletionTokens": completion_tokens,
            "TotalTokens": total_tokens,
            "EstimatedCostUSD": round(cost_usd, 6),
        }
        sys.stdout.write(_json.dumps(emf, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    except Exception:
        pass


def _cb_record_failure():
    global _CIRCUIT_BREAKER_FAILURES, _CIRCUIT_BREAKER_OPEN_SINCE
    _CIRCUIT_BREAKER_FAILURES += 1
    if _CIRCUIT_BREAKER_FAILURES == CIRCUIT_BREAKER_THRESHOLD:
        _CIRCUIT_BREAKER_OPEN_SINCE = _time.monotonic()


def _cb_record_success():
    global _CIRCUIT_BREAKER_FAILURES, _CIRCUIT_BREAKER_OPEN_SINCE
    _CIRCUIT_BREAKER_FAILURES = 0
    _CIRCUIT_BREAKER_OPEN_SINCE = 0.0


def get_circuit_breaker_state() -> dict:
    if _CIRCUIT_BREAKER_FAILURES < CIRCUIT_BREAKER_THRESHOLD:
        return {"state": "closed", "failures": _CIRCUIT_BREAKER_FAILURES}
    if _CIRCUIT_BREAKER_OPEN_SINCE and (_time.monotonic() - _CIRCUIT_BREAKER_OPEN_SINCE) < CIRCUIT_BREAKER_RESET_SEC:
        return {"state": "open", "failures": _CIRCUIT_BREAKER_FAILURES, "open_since_elapsed": round(_time.monotonic() - _CIRCUIT_BREAKER_OPEN_SINCE, 1)}
    return {"state": "half_open", "failures": _CIRCUIT_BREAKER_FAILURES, "open_since_elapsed": round(_time.monotonic() - _CIRCUIT_BREAKER_OPEN_SINCE, 1)}


def reset_circuit_breaker() -> None:
    global _CIRCUIT_BREAKER_FAILURES, _CIRCUIT_BREAKER_OPEN_SINCE
    _CIRCUIT_BREAKER_FAILURES = 0
    _CIRCUIT_BREAKER_OPEN_SINCE = 0.0


def get_token_usage() -> dict:
    return {"prompt_tokens": _cumulative_prompt_tokens, "completion_tokens": _cumulative_completion_tokens}


def reset_token_usage() -> None:
    global _cumulative_prompt_tokens, _cumulative_completion_tokens
    _cumulative_prompt_tokens = 0
    _cumulative_completion_tokens = 0


async def call_openai(
    system_prompt: str,
    user_content: str,
    max_tokens: int = OPENAI_DEFAULT_MAX_TOKENS,
    max_retries: int = 3,
    base_delay: float = 1.0,
    temperature: float | None = None,
) -> str:
    """Универсальная функция для вызова OpenAI API с retry и exponential backoff."""
    global openai_client, _CIRCUIT_BREAKER_FAILURES, _CIRCUIT_BREAKER_OPEN_SINCE, _cumulative_prompt_tokens, _cumulative_completion_tokens

    if _CIRCUIT_BREAKER_FAILURES >= CIRCUIT_BREAKER_THRESHOLD:
        elapsed_since_open = _time.monotonic() - _CIRCUIT_BREAKER_OPEN_SINCE
        if elapsed_since_open < CIRCUIT_BREAKER_RESET_SEC:
            logger.warning(
                "OpenAI circuit breaker open (%d consecutive failures, %.0fs until reset) — skipping call",
                _CIRCUIT_BREAKER_FAILURES,
                CIRCUIT_BREAKER_RESET_SEC - elapsed_since_open,
            )
            return ""
        logger.info(
            "OpenAI circuit breaker half-open (%d failures, %.0fs elapsed) — probing",
            _CIRCUIT_BREAKER_FAILURES, elapsed_since_open,
        )

    if openai_client is None:
        if not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY is not set — cannot create OpenAI client")
            return ""
        if OPENAI_MODEL not in _COST_PER_MILLION:
            logger.warning("OPENAI_MODEL=%s not in cost table — may exceed gpt-4o-mini pricing", OPENAI_MODEL)
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
            if not response.choices:
                logger.warning("OpenAI returned empty choices for model=%s", OPENAI_MODEL)
                _cb_record_failure()
                return ""
            result = response.choices[0].message.content
            if result is None:
                _cb_record_failure()
                return ""
            _cb_record_success()
            total_tokens = response.usage.total_tokens if hasattr(response, 'usage') and response.usage else 0
            prompt_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') and response.usage else 0
            completion_tokens = response.usage.completion_tokens if hasattr(response, 'usage') and response.usage else 0
            _cumulative_prompt_tokens += prompt_tokens
            _cumulative_completion_tokens += completion_tokens
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
            _emit_openai_latency(elapsed, OPENAI_MODEL, total_tokens, prompt_tokens, completion_tokens)
            return result.strip()
        except (RateLimitError, APIConnectionError) as e:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning("OpenAI retryable error (attempt %d/%d): %s; retrying in %.1fs", attempt + 1, max_retries + 1, e, delay)
                await asyncio.sleep(delay)
            else:
                logger.error("OpenAI API error after %d retries: %s", max_retries, e)
                _cb_record_failure()
                return ""
        except APIError as e:
            if e.status_code and e.status_code >= 500 and attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning("OpenAI 5xx error (attempt %d/%d): %s; retrying in %.1fs", attempt + 1, max_retries + 1, e, delay)
                await asyncio.sleep(delay)
            elif e.status_code and e.status_code in (401, 403):
                logger.error("OpenAI auth error (status %d): check OPENAI_API_KEY — %s", e.status_code, e)
                openai_client = None
                _cb_record_failure()
                return ""
            else:
                logger.error("OpenAI API error (status %s): %s", getattr(e, 'status_code', '?'), e)
                _cb_record_failure()
                return ""
        except Exception as e:
            logger.error("OpenAI API error: %s", e)
            _cb_record_failure()
            return ""
    _cb_record_failure()
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
                elif not is_self_closing and tag_name not in VOID_HTML_ELEMENTS:
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


_META_ARTIFACT_PATTERNS = [
    re.compile(r"^[^<\n]*?(?:в этом дайджесте|итого|в заключение|подведя итог|в итоге|итак|в общем|вкратце|как видно|обратите внимание|напомним)[^\n]*\n*", re.IGNORECASE),
    re.compile(r"\n[^<\n]*?(?:в этом дайджесте|итого|в заключение|подведя итог|в итоге|итак|в общем|вкратце|как видно|обратите внимание|напомним)[^\n]*$", re.IGNORECASE),
    re.compile(r"\n[^<\n]*?другие ссылки:\s*[^\n]*$", re.IGNORECASE),
]


def strip_meta_artifacts(text: str) -> str:
    for pattern in _META_ARTIFACT_PATTERNS:
        text = pattern.sub("", text)
    return text.strip()


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
