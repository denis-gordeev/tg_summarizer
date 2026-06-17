import asyncio
import importlib
import hashlib
import re
import sys
import time
import types
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from models import MessageInfo, SummaryInfo


def _truncate_html_preserving_tags(text, max_visible_chars):
    if max_visible_chars <= 0:
        return ""
    result = []
    open_tags = []
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


def _setup_stubs():
    """Install all fake modules before importing message_processor."""
    # --- Stub utils ---
    async def fake_call_openai(system_prompt, user_content, max_tokens=300, **kwargs):
        return "AI breakthrough reported [1]."

    fake_utils = types.ModuleType("utils")
    fake_utils.call_openai = fake_call_openai
    fake_utils.extract_links = lambda text: []
    fake_utils.count_characters = lambda text: len(text)
    fake_utils.text_hash = lambda text: hashlib.sha256(text.encode()).hexdigest()[:16]
    fake_utils.enforce_summary_length = lambda text, max_chars: text[:max_chars]
    sys.modules["utils"] = fake_utils

    # --- Stub config ---
    fake_config = types.ModuleType("config")
    fake_config.SIMILARITY_LLM_UPPER = 0.95
    fake_config.ENABLE_SUMMARIES_DEDUPLICATION = False
    fake_config.OPENAI_CHANNEL_SUMMARY_MAX_TOKENS = 4000
    fake_config.OPENAI_GROUP_SUMMARY_MAX_TOKENS = 4000
    fake_config.OPENAI_SUMMARY_TEMPERATURE = 0.3
    fake_config.SOURCE_CHANNELS = set()
    fake_config.DEBUG = False
    fake_config.ENABLE_SUMMARY_UPDATES = True
    fake_config.SUMMARY_MIN_RATIO = 3
    fake_config.SUMMARY_MIN_LENGTH = 800
    fake_config.SUMMARY_MAX_LENGTH = 4000
    fake_config.GROUP_SUMMARY_MIN_LENGTH = 2000
    fake_config.GROUP_SUMMARY_MAX_LENGTH = 12000
    fake_config.NLP_CHECK_MAX_INPUT_CHARS = 2000
    fake_config.COVERAGE_CHECK_MAX_INPUT_CHARS = 2000
    fake_config.NLP_MIN_TEXT_LENGTH = 100
    fake_config.SUMMARY_MAX_INPUT_CHARS_PER_MESSAGE = 3000
    fake_config.NLP_CONCURRENT_CHECKS = 5
    fake_config.NLP_AD_KEYWORDS = [
        "курс", "вебинар", "регистраци", "скидк", "промокод",
        "мастер-класс", "стажировк", "hire", "hiring day",
        "карьерный трек", "bootcamp", "boot camp",
    ]
    fake_config.UPDATE_MATCH_MAX_SUMMARIES = 5
    fake_config.UPDATE_MATCH_MAX_CHARS_PER_SUMMARY = 500
    fake_config.MAX_COVERED_MESSAGE_UPDATES = 5
    sys.modules["config"] = fake_config

    # --- Stub history_manager ---
    fake_history = types.ModuleType("history_manager")
    fake_history.load_summaries_history = MagicMock(return_value=[])
    fake_history.load_group_summaries_history = MagicMock(return_value=[])
    fake_history.save_summarization_history = MagicMock()
    fake_history.save_group_summarization_history = MagicMock()
    fake_history.save_summary_to_history = MagicMock()
    fake_history.save_group_summary_to_history = MagicMock()
    fake_history.update_group_last_run = MagicMock()
    fake_history.update_existing_summary = AsyncMock(return_value=None)
    fake_history.save_updated_summary = AsyncMock()
    sys.modules["history_manager"] = fake_history

    # --- Stub telegram_client ---
    fake_tg = types.ModuleType("telegram_client")
    fake_tg.send_message_to_target_channel_with_id = AsyncMock(return_value=42)
    sys.modules["telegram_client"] = fake_tg

    # --- Stub channel_manager ---
    fake_cm = types.ModuleType("channel_manager")
    fake_cm.load_discovered_channels = MagicMock(return_value=[])
    fake_cm.load_similar_channels = MagicMock(return_value=[])
    fake_cm.load_banned_channels = MagicMock(return_value=[])
    fake_cm.create_channel_abbreviation = lambda ch: ch.lstrip("@")[:5].upper()
    fake_cm.save_discovered_channel = MagicMock()
    fake_cm.get_all_source_channels = lambda: []
    sys.modules["channel_manager"] = fake_cm

    # --- Stub prompts ---
    fake_prompts = types.ModuleType("prompts")
    fake_prompts.prompts = types.SimpleNamespace(
        NLP_RELEVANCE_PROMPT="nlp",
        COVERAGE_AND_MATCH_PROMPT="covmatch",
        CHANNEL_SUMMARY_PROMPT="channel summary, max {max_summary_length} chars",
        GROUP_SUMMARY_PROMPT="group summary, max {max_summary_length} chars",
    )
    sys.modules["prompts"] = fake_prompts

    return fake_config, fake_history, fake_tg


def _make_message(
    text: str = "Test message",
    channel: str = "@test_channel",
    message_id: int = 1,
):
    from models import MessageInfo
    return MessageInfo(
        text=text,
        channel=channel,
        message_id=message_id,
        date=datetime.now(timezone.utc),
        link=f"https://t.me/{channel.lstrip('@')}/{message_id}",
    )


class ProcessMessagesIntegrationTests(unittest.TestCase):
    """End-to-end pipeline tests for process_messages() with mocked Telegram and OpenAI."""

    @classmethod
    def setUpClass(cls):
        cls._stubs = _setup_stubs()
        cls._fake_config, cls._fake_history, cls._fake_tg = cls._stubs

        # Force reimport of message_processor with our stubs
        for mod_name in list(sys.modules.keys()):
            if mod_name in ("message_processor", "models"):
                del sys.modules[mod_name]

        cls.mp = importlib.import_module("message_processor")

    def setUp(self):
        # Reset mock call counts between tests
        self._fake_history.save_summarization_history.reset_mock()
        self._fake_history.save_group_summarization_history.reset_mock()
        self._fake_history.save_summary_to_history.reset_mock()
        self._fake_history.save_group_summary_to_history.reset_mock()
        self._fake_history.update_group_last_run.reset_mock()
        self._fake_tg.send_message_to_target_channel_with_id = AsyncMock(return_value=42)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _run_pipeline(self, messages, **kwargs):
        import asyncio
        return asyncio.run(self.mp.process_messages(messages, **kwargs))

    # -------------------------------------------------------------------------
    # Tests
    # -------------------------------------------------------------------------
    def test_process_messages_channel_flow(self):
        """Full channel pipeline: filter NLP, deduplicate, summarize, save history."""
        messages = [
            _make_message(
                text="This is a long enough text about AI and machine learning breakthroughs "
                     "that should pass the NLP relevance check because it has sufficient length "
                     "and topic coverage for the model to analyze.",
                channel="@ai_news",
                message_id=100,
            ),
            _make_message(
                text="Another substantial article about natural language processing transformers "
                     "and their applications in real-world scenarios with detailed analysis.",
                channel="@ml_daily",
                message_id=200,
            ),
        ]

        with patch.object(self.mp, "is_nlp_related", new_callable=AsyncMock) as mock_nlp:
            mock_nlp.return_value = (True, "NLP related")

            self._fake_config.SOURCE_CHANNELS = {"@ai_news", "@ml_daily"}
            self._fake_config.ENABLE_SUMMARIES_DEDUPLICATION = False

            self._run_pipeline(messages, save_changes=True, send_message=False, is_group=False)

            self.assertEqual(mock_nlp.call_count, len(messages))
            self._fake_history.save_summarization_history.assert_called_once()

    def test_process_messages_skips_non_nlp(self):
        """Messages marked as non-NLP should not produce a summary."""
        messages = [
            _make_message(
                text="Buy cheap watches at the best prices!!! Visit our website now!!!",
                channel="@spam_channel",
                message_id=300,
            ),
        ]

        with patch.object(self.mp, "is_nlp_related", new_callable=AsyncMock) as mock_nlp:
            mock_nlp.return_value = (False, "advertising")

            self._fake_config.SOURCE_CHANNELS = set()
            self._fake_config.ENABLE_SUMMARIES_DEDUPLICATION = False

            self._run_pipeline(messages, save_changes=True, send_message=True, is_group=False)

            mock_nlp.assert_called_once()
            # History still gets saved (even with empty unique batch)
            self._fake_history.save_summarization_history.assert_called_once()

    def test_process_messages_empty_list(self):
        """Empty message list should return early without errors."""
        self._fake_config.SOURCE_CHANNELS = set()
        self._run_pipeline([], save_changes=True, send_message=True, is_group=False)

    def test_process_messages_group_flow(self):
        """Full group pipeline: filter, summarize group, update last run."""
        messages = [
            _make_message(
                text="Discussion about GPT-5 capabilities and the future of AGI development "
                     "in the context of recent research papers and industry Trends. Very long "
                     "detailed technical analysis of transformer architectures and scaling laws.",
                channel="@group_a",
                message_id=400,
            ),
        ]

        with patch.object(self.mp, "is_nlp_related", new_callable=AsyncMock) as mock_nlp:
            mock_nlp.return_value = (True, "NLP related")

            self._fake_config.SOURCE_CHANNELS = set()
            self._fake_config.ENABLE_SUMMARIES_DEDUPLICATION = False

            self._run_pipeline(messages, save_changes=True, send_message=True, is_group=True)

            mock_nlp.assert_called_once()
            self._fake_history.update_group_last_run.assert_called_once()

    def test_process_messages_sends_message(self):
        """When send_message=True, the Telegram send function should be called."""
        messages = [
            _make_message(
                text="Research paper on reinforcement learning with LLMs as agents. This is a "
                     "detailed discussion of the methodology and results with code examples.",
                channel="@rl_group",
                message_id=500,
            ),
        ]

        with patch.object(self.mp, "is_nlp_related", new_callable=AsyncMock) as mock_nlp:
            mock_nlp.return_value = (True, "NLP related")

            fake_tg_call = AsyncMock(return_value=12345)
            self._fake_tg.send_message_to_target_channel_with_id = fake_tg_call

            self._fake_config.SOURCE_CHANNELS = set()
            self._fake_config.ENABLE_SUMMARIES_DEDUPLICATION = False

            self._run_pipeline(messages, save_changes=True, send_message=True, is_group=False)

            fake_tg_call.assert_called_once()

    def test_coverage_checks_run_in_parallel(self):
        """Coverage checks should be dispatched concurrently via asyncio.gather."""
        messages = [
            _make_message(
                text="First article about transformer models in NLP research with detailed "
                     "methodology and experimental results.",
                channel="@ai_ch1",
                message_id=601,
            ),
            _make_message(
                text="Second article about reinforcement learning with LLMs as agents "
                "and their applications in real-world scenarios.",
                channel="@ai_ch2",
                message_id=602,
            ),
        ]

        self._fake_config.SOURCE_CHANNELS = {"@ai_ch1", "@ai_ch2"}
        fake_summary = SummaryInfo(
            content="Previous summary about transformers",
            date=datetime.now(timezone.utc),
            message_count=1,
            channels=["@ai_ch1"],
        )
        self._fake_history.load_summaries_history.return_value = [fake_summary]

        with patch.object(self.mp, "is_nlp_related", new_callable=AsyncMock) as mock_nlp, \
             patch.object(self.mp, "ENABLE_SUMMARIES_DEDUPLICATION", True), \
             patch.object(self.mp, "_check_coverage_and_match", new_callable=AsyncMock) as mock_cov:
            mock_nlp.return_value = (True, "NLP related")
            mock_cov.return_value = None

            self._run_pipeline(messages, save_changes=True, send_message=False, is_group=False)

            self.assertEqual(mock_cov.call_count, 2,
                             "Coverage+match check should be called for each unique NLP-related message")

    def test_covered_messages_excluded_from_summary(self):
        """Messages covered in previous summaries should not appear in the new summary."""
        from models import SummaryInfo

        messages = [
            _make_message(
                text="Article about AI that was already covered in a previous digest "
                     "with similar content and analysis.",
                channel="@ai_ch",
                message_id=701,
            ),
        ]

        self._fake_config.SOURCE_CHANNELS = {"@ai_ch"}
        fake_summary = SummaryInfo(
            content="Previous summary about AI",
            date=datetime.now(timezone.utc),
            message_count=1,
            channels=["@ai_ch"],
            message_id=100,
        )
        self._fake_history.load_summaries_history.return_value = [fake_summary]

        with patch.object(self.mp, "is_nlp_related", new_callable=AsyncMock) as mock_nlp, \
             patch.object(self.mp, "ENABLE_SUMMARIES_DEDUPLICATION", True), \
             patch.object(self.mp, "_check_coverage_and_match", new_callable=AsyncMock) as mock_cov, \
             patch.object(self.mp, "summarize_text", new_callable=AsyncMock) as mock_sum:
            mock_nlp.return_value = (True, "NLP related")
            mock_cov.return_value = fake_summary

            self._run_pipeline(messages, save_changes=True, send_message=False, is_group=False)

            mock_sum.assert_not_called()


class SummaryTemperatureTests(unittest.TestCase):
    """Tests for OPENAI_SUMMARY_TEMPERATURE being passed to call_openai."""

    @classmethod
    def setUpClass(cls):
        _setup_stubs()
        import message_processor
        cls.mp = message_processor

    def test_summarize_text_passes_temperature(self):
        """summarize_text should pass OPENAI_SUMMARY_TEMPERATURE to call_openai."""
        from models import MessageInfo
        from unittest.mock import AsyncMock, patch

        messages = [
            MessageInfo(
                text="AI research breakthrough with detailed methodology",
                channel="@ai",
                message_id=1,
                date=datetime.now(timezone.utc),
                link="",
            ),
        ]

        with patch.object(self.mp, "call_openai", new_callable=AsyncMock) as mock_openai:
            mock_openai.return_value = "<b>AI news</b> [1]"
            asyncio.run(self.mp.summarize_text(messages))
            call_kwargs = mock_openai.call_args
            self.assertAlmostEqual(call_kwargs.kwargs.get("temperature"), 0.3)

    def test_summarize_group_text_passes_temperature(self):
        """summarize_group_text should pass OPENAI_SUMMARY_TEMPERATURE to call_openai."""
        from models import MessageInfo
        from unittest.mock import AsyncMock, patch

        messages = [
            MessageInfo(
                text="Group discussion about GPT-5 capabilities",
                channel="@group",
                message_id=1,
                date=datetime.now(timezone.utc),
                link="",
            ),
        ]

        with patch.object(self.mp, "call_openai", new_callable=AsyncMock) as mock_openai:
            mock_openai.return_value = "<b>GPT-5</b> [1]"
            asyncio.run(self.mp.summarize_group_text(messages))
            call_kwargs = mock_openai.call_args
            self.assertAlmostEqual(call_kwargs.kwargs.get("temperature"), 0.3)

class NlpCheckTruncationTests(unittest.TestCase):
    """Tests for NLP check input truncation."""

    @classmethod
    def setUpClass(cls):
        cls._stubs = _setup_stubs()
        cls._fake_config, _, _ = cls._stubs
        import message_processor
        cls.mp = message_processor

    def test_nlp_check_truncates_long_input(self):
        """is_nlp_related should truncate input text to NLP_CHECK_MAX_INPUT_CHARS."""
        from unittest.mock import AsyncMock, patch

        long_text = "x" * 5000
        self._fake_config.NLP_CHECK_MAX_INPUT_CHARS = 2000

        with patch.object(self.mp, "call_openai", new_callable=AsyncMock) as mock_openai:
            mock_openai.return_value = "да"
            asyncio.run(self.mp.is_nlp_related(long_text))
            call_args = mock_openai.call_args
            user_content = call_args[0][1]
            self.assertLessEqual(len(user_content), 2000)

    def test_nlp_check_keeps_short_input(self):
        """is_nlp_related should not truncate short text."""
        from unittest.mock import AsyncMock, patch

        short_text = "A" * 500
        self._fake_config.NLP_CHECK_MAX_INPUT_CHARS = 2000

        with patch.object(self.mp, "call_openai", new_callable=AsyncMock) as mock_openai:
            mock_openai.return_value = "да"
            asyncio.run(self.mp.is_nlp_related(short_text))
            call_args = mock_openai.call_args
            user_content = call_args[0][1]
            self.assertEqual(len(user_content), 500)

    def test_nlp_check_uses_max_tokens_10(self):
        """is_nlp_related should use max_tokens=10 for cost optimization."""
        from unittest.mock import AsyncMock, patch

        long_enough_text = "A" * 200

        with patch.object(self.mp, "call_openai", new_callable=AsyncMock) as mock_openai:
            mock_openai.return_value = "да"
            asyncio.run(self.mp.is_nlp_related(long_enough_text))
            call_kwargs = mock_openai.call_args
            self.assertEqual(call_kwargs.kwargs.get("max_tokens"), 10)


class SummaryFailureReturnsNoneTests(unittest.TestCase):
    """Tests for summarize_text/summarize_group_text returning None on failure."""

    @classmethod
    def setUpClass(cls):
        cls._stubs = _setup_stubs()
        cls._fake_config, cls._fake_history, cls._fake_tg = cls._stubs
        for mod_name in list(sys.modules.keys()):
            if mod_name in ("message_processor", "models"):
                del sys.modules[mod_name]
        cls.mp = importlib.import_module("message_processor")

    def test_summarize_text_returns_none_on_empty_openai_response(self):
        from models import MessageInfo
        from unittest.mock import AsyncMock, patch

        messages = [
            MessageInfo(
                text="AI research breakthrough",
                channel="@ai",
                message_id=1,
                date=datetime.now(timezone.utc),
                link="",
            ),
        ]

        with patch.object(self.mp, "call_openai", new_callable=AsyncMock) as mock_openai:
            mock_openai.return_value = ""
            result = asyncio.run(self.mp.summarize_text(messages))
            self.assertIsNone(result)

    def test_summarize_group_text_returns_none_on_empty_openai_response(self):
        from models import MessageInfo
        from unittest.mock import AsyncMock, patch

        messages = [
            MessageInfo(
                text="Group discussion about GPT",
                channel="@group",
                message_id=1,
                date=datetime.now(timezone.utc),
                link="",
            ),
        ]

        with patch.object(self.mp, "call_openai", new_callable=AsyncMock) as mock_openai:
            mock_openai.return_value = ""
            result = asyncio.run(self.mp.summarize_group_text(messages))
            self.assertIsNone(result)

    def test_process_messages_skips_send_when_summary_is_none(self):
        from models import MessageInfo
        from unittest.mock import AsyncMock, patch

        messages = [
            _make_message(
                text="AI text that is long enough to pass NLP check and be summarized",
                channel="@ai_ch",
                message_id=900,
            ),
        ]

        self._fake_config.ENABLE_SUMMARIES_DEDUPLICATION = False

        with patch.object(self.mp, "is_nlp_related", new_callable=AsyncMock) as mock_nlp, \
             patch.object(self.mp, "summarize_text", new_callable=AsyncMock) as mock_sum, \
             patch.object(self._fake_tg, "send_message_to_target_channel_with_id", new_callable=AsyncMock) as mock_send:
            mock_nlp.return_value = (True, "NLP related")
            mock_sum.return_value = None

            asyncio.run(self.mp.process_messages(
                messages, save_changes=False, send_message=True, is_group=False
            ))

            mock_send.assert_not_called()


class IntraBatchDedupBeforeCoverageTests(unittest.TestCase):
    """Tests that intra-batch dedup runs before coverage checks, saving LLM calls."""

    @classmethod
    def setUpClass(cls):
        cls._stubs = _setup_stubs()
        cls._fake_config, cls._fake_history, cls._fake_tg = cls._stubs
        for mod_name in list(sys.modules.keys()):
            if mod_name in ("message_processor", "models"):
                del sys.modules[mod_name]
        cls.mp = importlib.import_module("message_processor")

    def test_dedup_before_coverage_saves_llm_calls(self):
        """When two near-identical messages are in a batch, only the surviving one
        should get a coverage check (the duplicate is removed first)."""
        base_text = "AI breakthrough: GPT-5 announced with new capabilities and improved reasoning over previous models."
        messages = [
            _make_message(
                text=base_text,
                channel="@ai_ch1",
                message_id=801,
            ),
            _make_message(
                text=base_text,
                channel="@ai_ch2",
                message_id=802,
            ),
        ]

        self._fake_config.SOURCE_CHANNELS = {"@ai_ch1", "@ai_ch2"}
        fake_summary = SummaryInfo(
            content="Previous summary",
            date=datetime.now(timezone.utc),
            message_count=1,
            channels=["@ai_ch1"],
        )
        self._fake_history.load_summaries_history.return_value = [fake_summary]

        with patch.object(self.mp, "is_nlp_related", new_callable=AsyncMock) as mock_nlp, \
             patch.object(self.mp, "ENABLE_SUMMARIES_DEDUPLICATION", True), \
             patch.object(self.mp, "_check_coverage_and_match", new_callable=AsyncMock) as mock_cov, \
             patch.object(self.mp, "summarize_text", new_callable=AsyncMock) as mock_sum:
            mock_nlp.return_value = (True, "NLP related")
            mock_cov.return_value = None
            mock_sum.return_value = "<b>test</b>"

            asyncio.run(self.mp.process_messages(
                messages, save_changes=False, send_message=False, is_group=False
            ))

            self.assertEqual(mock_nlp.call_count, 2)
            self.assertLessEqual(mock_cov.call_count, 1,
                                 "Coverage check should only run for deduped messages")


class NlpRelatedMessagesIndentationTests(unittest.TestCase):
    """Regression test: nlp_related_messages.append must be inside the for loop."""

    @classmethod
    def setUpClass(cls):
        cls._stubs = _setup_stubs()
        cls._fake_config, cls._fake_history, cls._fake_tg = cls._stubs
        for mod_name in list(sys.modules.keys()):
            if mod_name in ("message_processor", "models"):
                del sys.modules[mod_name]
        cls.mp = importlib.import_module("message_processor")

    def test_all_nlp_messages_appended(self):
        """All NLP-related messages should be appended to nlp_related_messages,
        not just the last one in the loop."""
        messages = [
            _make_message(text="AI research breakthrough with methodology " * 5, channel="@ch1", message_id=1),
            _make_message(text="ML paper about transformers " * 5, channel="@ch2", message_id=2),
            _make_message(text="NLP model evaluation results " * 5, channel="@ch3", message_id=3),
        ]

        self._fake_config.SOURCE_CHANNELS = set()
        self._fake_config.ENABLE_SUMMARIES_DEDUPLICATION = False

        with patch.object(self.mp, "is_nlp_related", new_callable=AsyncMock) as mock_nlp, \
             patch.object(self.mp, "summarize_text", new_callable=AsyncMock) as mock_sum:
            mock_nlp.return_value = (True, "NLP related")
            mock_sum.return_value = "<b>test</b>"

            asyncio.run(self.mp.process_messages(
                messages, save_changes=False, send_message=False, is_group=False
            ))

            nlp_related_count = sum(1 for m in messages if m.is_nlp_related)
            self.assertEqual(nlp_related_count, 3,
                             "All 3 messages should be marked as NLP related")


class NlpAdKeywordFilterTests(unittest.TestCase):
    """Tests for _is_obvious_non_nlp keyword pre-filter."""

    @classmethod
    def setUpClass(cls):
        cls._stubs = _setup_stubs()
        cls._fake_config, cls._fake_history, cls._fake_tg = cls._stubs
        for mod_name in list(sys.modules.keys()):
            if mod_name in ("message_processor", "models"):
                del sys.modules[mod_name]
        cls.mp = importlib.import_module("message_processor")

    def test_rejects_course_ad(self):
        result = self.mp._is_obvious_non_nlp("Запишитесь на наш курс по машинному обучению!")
        self.assertTrue(result)

    def test_rejects_webinar_ad(self):
        result = self.mp._is_obvious_non_nlp("Бесплатный вебинар по AI завтра в 18:00")
        self.assertTrue(result)

    def test_rejects_promocode_ad(self):
        result = self.mp._is_obvious_non_nlp("Промокод на скидку 50% на обучение")
        self.assertTrue(result)

    def test_allows_ai_research(self):
        result = self.mp._is_obvious_non_nlp("New research paper on transformer architectures and attention mechanisms")
        self.assertFalse(result)

    def test_allows_vacancy(self):
        result = self.mp._is_obvious_non_nlp("Ищем ML инженера в команду, удаленка, опыт с LLM")
        self.assertFalse(result)

    def test_nlp_related_skips_llm_on_ad_keyword(self):
        """is_nlp_related should return (False, 'ad_keyword') without calling LLM for ads."""
        ad_text = "Курс по Data Science с сертификатом! Регистрация открыта!" + " x" * 80

        with patch.object(self.mp, "call_openai", new_callable=AsyncMock) as mock_openai:
            result = asyncio.run(self.mp.is_nlp_related(ad_text))
            self.assertFalse(result[0])
            self.assertEqual(result[1], "ad_keyword")
            mock_openai.assert_not_called()


class CoverageAndMatchCheckTests(unittest.TestCase):
    """Tests for _check_coverage_and_match combined check."""

    @classmethod
    def setUpClass(cls):
        cls._stubs = _setup_stubs()
        cls._fake_config, cls._fake_history, cls._fake_tg = cls._stubs
        for mod_name in list(sys.modules.keys()):
            if mod_name in ("message_processor", "models"):
                del sys.modules[mod_name]
        cls.mp = importlib.import_module("message_processor")

    def test_returns_matching_summary_on_covered(self):
        """When LLM returns a valid summary number, return that summary."""
        summaries = [
            SummaryInfo(
                content="Summary about GPT-5 release",
                date=datetime.now(timezone.utc),
                message_count=2,
                channels=["@ai"],
                message_id=100,
            ),
            SummaryInfo(
                content="Summary about BERT improvements",
                date=datetime.now(timezone.utc),
                message_count=1,
                channels=["@nlp"],
                message_id=101,
            ),
        ]
        msg = _make_message(text="New details about GPT-5 capabilities", channel="@ai", message_id=1)

        with patch.object(self.mp, "call_openai", new_callable=AsyncMock) as mock_openai:
            mock_openai.return_value = "1"
            result = asyncio.run(self.mp._check_coverage_and_match(msg, summaries))
            self.assertIsNotNone(result)
            self.assertEqual(result.message_id, 100)

    def test_returns_none_on_not_covered(self):
        """When LLM returns НЕТ, return None."""
        summaries = [
            SummaryInfo(
                content="Summary about quantum computing",
                date=datetime.now(timezone.utc),
                message_count=1,
                channels=["@quantum"],
            ),
        ]
        msg = _make_message(text="New LLM benchmark results", channel="@ml", message_id=1)

        with patch.object(self.mp, "call_openai", new_callable=AsyncMock) as mock_openai:
            mock_openai.return_value = "НЕТ"
            result = asyncio.run(self.mp._check_coverage_and_match(msg, summaries))
            self.assertIsNone(result)

    def test_returns_none_on_empty_summaries(self):
        """When summaries list is empty, return None without LLM call."""
        msg = _make_message(text="Some text", channel="@ch", message_id=1)

        with patch.object(self.mp, "call_openai", new_callable=AsyncMock) as mock_openai:
            result = asyncio.run(self.mp._check_coverage_and_match(msg, []))
            self.assertIsNone(result)
            mock_openai.assert_not_called()

    def test_returns_none_on_invalid_response(self):
        """When LLM returns invalid response, return None."""
        summaries = [
            SummaryInfo(
                content="Some summary",
                date=datetime.now(timezone.utc),
                message_count=1,
                channels=["@ch"],
            ),
        ]
        msg = _make_message(text="Some text", channel="@ch", message_id=1)

        with patch.object(self.mp, "call_openai", new_callable=AsyncMock) as mock_openai:
            mock_openai.return_value = "maybe"
            result = asyncio.run(self.mp._check_coverage_and_match(msg, summaries))
            self.assertIsNone(result)

    def test_returns_none_on_digit_out_of_range(self):
        """When LLM returns a digit outside the valid range, return None."""
        summaries = [
            SummaryInfo(
                content="Some summary",
                date=datetime.now(timezone.utc),
                message_count=1,
                channels=["@ch"],
            ),
        ]
        msg = _make_message(text="Some text", channel="@ch", message_id=1)

        with patch.object(self.mp, "call_openai", new_callable=AsyncMock) as mock_openai:
            mock_openai.return_value = "5"
            result = asyncio.run(self.mp._check_coverage_and_match(msg, summaries))
            self.assertIsNone(result)


class CoverageMatchTruncationTests(unittest.TestCase):
    """Tests for _check_coverage_and_match truncating msg.text using COVERAGE_CHECK_MAX_INPUT_CHARS."""

    @classmethod
    def setUpClass(cls):
        cls._stubs = _setup_stubs()
        cls._fake_config, cls._fake_history, cls._fake_tg = cls._stubs
        for mod_name in list(sys.modules.keys()):
            if mod_name in ("message_processor", "models"):
                del sys.modules[mod_name]
        cls.mp = importlib.import_module("message_processor")

    def test_coverage_match_truncates_long_message(self):
        summaries = [
            SummaryInfo(
                content="Summary about AI",
                date=datetime.now(timezone.utc),
                message_count=1,
                channels=["@ai"],
                message_id=100,
            ),
        ]
        long_text = "AI breakthrough " * 500
        self._fake_config.COVERAGE_CHECK_MAX_INPUT_CHARS = 2000
        msg = _make_message(text=long_text, channel="@ai", message_id=1)

        with patch.object(self.mp, "call_openai", new_callable=AsyncMock) as mock_openai:
            mock_openai.return_value = "НЕТ"
            asyncio.run(self.mp._check_coverage_and_match(msg, summaries))
            user_content = mock_openai.call_args[0][1]
            self.assertIn(long_text[:2000], user_content)
            self.assertNotIn(long_text[2001:], user_content)

    def test_coverage_match_uses_coverage_check_max_input_chars(self):
        """_check_coverage_and_match should use COVERAGE_CHECK_MAX_INPUT_CHARS, not NLP_CHECK_MAX_INPUT_CHARS."""
        summaries = [
            SummaryInfo(
                content="Summary about AI",
                date=datetime.now(timezone.utc),
                message_count=1,
                channels=["@ai"],
                message_id=100,
            ),
        ]
        long_text = "AI breakthrough " * 500
        self._fake_config.COVERAGE_CHECK_MAX_INPUT_CHARS = 1500
        self._fake_config.NLP_CHECK_MAX_INPUT_CHARS = 2000
        msg = _make_message(text=long_text, channel="@ai", message_id=1)

        with patch.object(self.mp, "call_openai", new_callable=AsyncMock) as mock_openai:
            mock_openai.return_value = "НЕТ"
            asyncio.run(self.mp._check_coverage_and_match(msg, summaries))
            user_content = mock_openai.call_args[0][1]
            self.assertNotIn(long_text[1501:], user_content)


class ReplaceSourceWithLinksPrecomputeTests(unittest.TestCase):
    """Tests for _replace_source_with_links pre-computing per-message data."""

    @classmethod
    def setUpClass(cls):
        cls._stubs = _setup_stubs()
        cls._fake_config, cls._fake_history, cls._fake_tg = cls._stubs
        for mod_name in list(sys.modules.keys()):
            if mod_name in ("message_processor", "models"):
                del sys.modules[mod_name]
        cls.mp = importlib.import_module("message_processor")

    def test_replaces_source_numbers_with_links(self):
        """_replace_source_with_links should replace [1] with HTML link."""
        messages = [
            MessageInfo(
                text="AI text https://example.com/article",
                channel="@ai_ch",
                message_id=1,
                date=datetime.now(timezone.utc),
                link="https://example.com/article",
            ),
        ]

        result = self.mp._replace_source_with_links(messages, "AI news [1]")
        self.assertIn('<a href="https://t.me/ai_ch/1">[AI_CH]</a>', result)
        self.assertNotIn("[1]", result.replace('<a href="https://t.me/ai_ch/1">[AI_CH]</a>', ""))

    def test_handles_multiple_references_to_same_source(self):
        """When [1] appears multiple times, each should be replaced."""
        messages = [
            MessageInfo(
                text="AI text https://example.com/a",
                channel="@ai_ch",
                message_id=1,
                date=datetime.now(timezone.utc),
                link="https://example.com/a",
            ),
        ]

        result = self.mp._replace_source_with_links(messages, "AI news [1] and more [1]")
        self.assertEqual(result.count('<a href="https://t.me/ai_ch/1">[AI_CH]</a>'), 2)


class CoverageDedupStatsTests(unittest.TestCase):
    """Tests for coverage dedup statistics logging."""

    @classmethod
    def setUpClass(cls):
        cls._stubs = _setup_stubs()
        cls._fake_config, cls._fake_history, cls._fake_tg = cls._stubs
        for mod_name in list(sys.modules.keys()):
            if mod_name in ("message_processor", "models"):
                del sys.modules[mod_name]
        cls.mp = importlib.import_module("message_processor")

    def test_logs_coverage_dedup_stats(self):
        """process_messages should log coverage dedup stats when some messages are covered."""
        msgs = [
            _make_message(text="AI text that is long enough for NLP check " * 5, channel="@ai", message_id=1),
            _make_message(text="Another AI text that passes NLP check " * 5, channel="@ai", message_id=2),
        ]

        matching = SummaryInfo(
            content="Existing summary",
            date=datetime.now(timezone.utc),
            message_count=1,
            channels=["@ai"],
            message_id=100,
        )

        self._fake_history.load_summaries_history.return_value = [matching]

        with patch.object(self.mp, "is_nlp_related", new_callable=AsyncMock, return_value=(True, "да")), \
             patch.object(self.mp, "_check_coverage_and_match", new_callable=AsyncMock, return_value=matching), \
             patch.object(self.mp, "process_covered_message", new_callable=AsyncMock), \
             patch.object(self.mp, "ENABLE_SUMMARIES_DEDUPLICATION", True), \
             patch.object(self.mp, "summarize_text", new_callable=AsyncMock, return_value="new summary"), \
             patch("telegram_client.send_message_to_target_channel_with_id", new_callable=AsyncMock, return_value=42), \
             patch.object(self.mp, "_save_processing_results", new_callable=AsyncMock), \
             patch.object(self.mp.logger, "info") as mock_log:
            asyncio.run(self.mp.process_messages(
                msgs, save_changes=False, send_message=True, is_group=False,
            ))

        dedup_logs = [call for call in mock_log.call_args_list
                      if "Coverage dedup" in str(call)]
        self.assertTrue(len(dedup_logs) > 0, "Expected coverage dedup stats log message")


class ProcessMessagesDeadlineTests(unittest.TestCase):
    """Tests for deadline propagation in process_messages."""

    @classmethod
    def setUpClass(cls):
        cls._stubs = _setup_stubs()
        cls._fake_config, cls._fake_history, cls._fake_tg = cls._stubs
        for mod_name in list(sys.modules.keys()):
            if mod_name in ("message_processor", "models"):
                del sys.modules[mod_name]
        cls.mp = importlib.import_module("message_processor")

    def test_deadline_skips_summary_generation(self):
        messages = [
            _make_message(
                text="AI text long enough for NLP check " * 5,
                channel="@ai_ch",
                message_id=900,
            ),
        ]

        self._fake_config.SOURCE_CHANNELS = set()
        self._fake_config.ENABLE_SUMMARIES_DEDUPLICATION = False

        past_deadline = time.monotonic() - 100

        with patch.object(self.mp, "is_nlp_related", new_callable=AsyncMock) as mock_nlp:
            mock_nlp.return_value = (True, "NLP related")

            asyncio.run(self.mp.process_messages(
                messages, save_changes=False, send_message=False, is_group=False,
                _deadline=past_deadline,
            ))

            self._fake_tg.send_message_to_target_channel_with_id.assert_not_called()


class StaleSummaryRefreshTests(unittest.TestCase):
    """Tests for process_covered_message refreshing stale matching_summary."""

    @classmethod
    def setUpClass(cls):
        cls._stubs = _setup_stubs()
        cls._fake_config, cls._fake_history, cls._fake_tg = cls._stubs
        for mod_name in list(sys.modules.keys()):
            if mod_name in ("message_processor", "models"):
                del sys.modules[mod_name]
        cls.mp = importlib.import_module("message_processor")

    def test_refreshes_matching_summary_from_file(self):
        original_summary = SummaryInfo(
            content="Original content",
            date=datetime.now(timezone.utc),
            message_count=1,
            channels=["@ai"],
            message_id=100,
        )
        updated_summary = SummaryInfo(
            content="Updated content by another message",
            date=datetime.now(timezone.utc),
            message_count=2,
            channels=["@ai"],
            message_id=100,
        )

        msg = _make_message(text="New AI details", channel="@ai", message_id=200)

        with patch.object(self.mp, "load_summaries_history", return_value=[updated_summary]) as mock_load, \
             patch.object(self.mp, "update_existing_summary", new_callable=AsyncMock, return_value=updated_summary) as mock_update, \
             patch.object(self.mp, "save_updated_summary"):
            asyncio.run(self.mp.process_covered_message(
                msg, matching_summary=original_summary, is_group=False,
            ))

            mock_load.assert_called()
            mock_update.assert_called_once()
            call_args = mock_update.call_args
            self.assertEqual(call_args[0][0].content, "Updated content by another message")


class ProcessCoveredMessageNoFallbackTests(unittest.TestCase):
    """Tests for process_covered_message not falling back to find_relevant_summary_for_update."""

    @classmethod
    def setUpClass(cls):
        cls._stubs = _setup_stubs()
        cls._fake_config, cls._fake_history, cls._fake_tg = cls._stubs
        for mod_name in list(sys.modules.keys()):
            if mod_name in ("message_processor", "models"):
                del sys.modules[mod_name]
        cls.mp = importlib.import_module("message_processor")

    def test_skips_update_when_no_matching_summary(self):
        """process_covered_message should skip update when matching_summary is None."""
        msg = _make_message(text="New AI details", channel="@ai", message_id=200)

        with patch.object(self.mp, "update_existing_summary", new_callable=AsyncMock) as mock_update:
            asyncio.run(self.mp.process_covered_message(
                msg, matching_summary=None, is_group=False,
            ))

            mock_update.assert_not_called()


class ProcessCoveredMessageRefreshTests(unittest.TestCase):
    """Tests for process_covered_message refreshing and skipping when summary not found in history."""

    @classmethod
    def setUpClass(cls):
        cls._stubs = _setup_stubs()
        cls._fake_config, cls._fake_history, cls._fake_tg = cls._stubs
        for mod_name in list(sys.modules.keys()):
            if mod_name in ("message_processor", "models"):
                del sys.modules[mod_name]
        cls.mp = importlib.import_module("message_processor")

    def test_skips_update_when_summary_not_found_in_history(self):
        """process_covered_message should skip update when the summary is not found
        in the history file (e.g., was deleted between invocations)."""
        original_summary = SummaryInfo(
            content="Deleted summary",
            date=datetime.now(timezone.utc),
            message_count=1,
            channels=["@ai"],
            message_id=999,
        )

        msg = _make_message(text="New AI details", channel="@ai", message_id=200)

        with patch.object(self.mp, "load_summaries_history", return_value=[]) as mock_load, \
             patch.object(self.mp, "update_existing_summary", new_callable=AsyncMock) as mock_update:
            asyncio.run(self.mp.process_covered_message(
                msg, matching_summary=original_summary, is_group=False,
            ))

            mock_load.assert_called()
            mock_update.assert_not_called()


class DeadlineInCoveredMessageProcessingTests(unittest.TestCase):
    """Tests for deadline check during covered message processing loop."""

    @classmethod
    def setUpClass(cls):
        cls._stubs = _setup_stubs()
        cls._fake_config, cls._fake_history, cls._fake_tg = cls._stubs
        for mod_name in list(sys.modules.keys()):
            if mod_name in ("message_processor", "models"):
                del sys.modules[mod_name]
        cls.mp = importlib.import_module("message_processor")

    def test_deadline_skips_remaining_covered_updates(self):
        """When deadline is exceeded during covered message processing,
        remaining updates should be skipped."""
        summaries = [
            SummaryInfo(
                content="Summary about AI",
                date=datetime.now(timezone.utc),
                message_count=1,
                channels=["@ai"],
                message_id=100,
            ),
        ]

        messages = [
            _make_message(text="AI update one " * 20, channel="@ai", message_id=201),
            _make_message(text="AI update two " * 20, channel="@ai", message_id=202),
        ]

        self._fake_config.SOURCE_CHANNELS = {"@ai"}
        self._fake_history.load_summaries_history.return_value = summaries

        past_deadline = time.monotonic() - 100

        with patch.object(self.mp, "is_nlp_related", new_callable=AsyncMock) as mock_nlp, \
             patch.object(self.mp, "_check_coverage_and_match", new_callable=AsyncMock) as mock_cov, \
             patch.object(self.mp, "process_covered_message", new_callable=AsyncMock) as mock_process:
            mock_nlp.return_value = (True, "NLP related")
            mock_cov.return_value = summaries[0]

            asyncio.run(self.mp.process_messages(
                messages, save_changes=False, send_message=False, is_group=False,
                _deadline=past_deadline,
            ))

            mock_process.assert_not_called()


class CreateSummaryInfoSyncTests(unittest.TestCase):
    """Tests that _create_summary_info is a sync function."""

    @classmethod
    def setUpClass(cls):
        cls._stubs = _setup_stubs()
        cls._fake_config, cls._fake_history, cls._fake_tg = cls._stubs
        for mod_name in list(sys.modules.keys()):
            if mod_name in ("message_processor", "models"):
                del sys.modules[mod_name]
        cls.mp = importlib.import_module("message_processor")

    def test_create_summary_info_returns_summary_info(self):
        """_create_summary_info should return a SummaryInfo without awaiting."""
        from models import MessageInfo, SummaryInfo
        messages = [
            MessageInfo(
                text="AI research",
                channel="@ai",
                message_id=1,
                date=datetime.now(timezone.utc),
                link="",
            ),
        ]

        result = self.mp._create_summary_info("summary text", messages, 42, False)
        self.assertIsInstance(result, SummaryInfo)
        self.assertEqual(result.content, "summary text")
        self.assertEqual(result.message_id, 42)
        self.assertEqual(result.channels, ["@ai"])
        self.assertEqual(result.message_count, 1)


class DeadlineUnmarksCoveredMessagesTests(unittest.TestCase):
    """Tests for deadline un-marking covered messages so they're included in summary."""

    @classmethod
    def setUpClass(cls):
        cls._stubs = _setup_stubs()
        cls._fake_config, cls._fake_history, cls._fake_tg = cls._stubs
        for mod_name in list(sys.modules.keys()):
            if mod_name in ("message_processor", "models"):
                del sys.modules[mod_name]
        cls.mp = importlib.import_module("message_processor")

    def test_deadline_unmarks_covered_messages(self):
        """When deadline is exceeded during covered message processing,
        remaining covered messages should be un-marked so they're included
        in the new summary instead of being silently dropped."""
        summaries = [
            SummaryInfo(
                content="Summary about AI",
                date=datetime.now(timezone.utc),
                message_count=1,
                channels=["@ai"],
                message_id=100,
            ),
        ]

        messages = [
            _make_message(text="AI update one " * 20, channel="@ai", message_id=201),
            _make_message(text="AI update two " * 20, channel="@ai", message_id=202),
        ]

        self._fake_config.SOURCE_CHANNELS = {"@ai"}
        self._fake_history.load_summaries_history.return_value = summaries

        past_deadline = time.monotonic() - 100

        with patch.object(self.mp, "is_nlp_related", new_callable=AsyncMock) as mock_nlp, \
             patch.object(self.mp, "_check_coverage_and_match", new_callable=AsyncMock) as mock_cov, \
             patch.object(self.mp, "process_covered_message", new_callable=AsyncMock) as mock_process, \
             patch.object(self.mp, "summarize_text", new_callable=AsyncMock) as mock_sum:
            mock_nlp.return_value = (True, "NLP related")
            mock_cov.return_value = summaries[0]
            mock_sum.return_value = "<b>test</b>"

            asyncio.run(self.mp.process_messages(
                messages, save_changes=False, send_message=False, is_group=False,
                _deadline=past_deadline,
            ))

            unmarked = [m for m in messages if not getattr(m, 'is_covered_in_summaries', False)]
            self.assertGreater(len(unmarked), 0,
                               "At least some covered messages should be un-marked when deadline exceeded")


class NlpMinTextLengthTests(unittest.TestCase):
    """Tests for configurable NLP_MIN_TEXT_LENGTH."""

    @classmethod
    def setUpClass(cls):
        cls._stubs = _setup_stubs()
        cls._fake_config, cls._fake_history, cls._fake_tg = cls._stubs
        for mod_name in list(sys.modules.keys()):
            if mod_name in ("message_processor", "models"):
                del sys.modules[mod_name]
        cls.mp = importlib.import_module("message_processor")

    def test_nlp_rejects_short_text_with_configurable_threshold(self):
        """is_nlp_related should reject text shorter than NLP_MIN_TEXT_LENGTH."""
        with patch.object(self.mp, "NLP_MIN_TEXT_LENGTH", 200), \
             patch.object(self.mp, "call_openai", new_callable=AsyncMock) as mock_openai:
            short_text = "x" * 150
            result = asyncio.run(self.mp.is_nlp_related(short_text))
            self.assertFalse(result[0])
            self.assertEqual(result[1], "too_short")
            mock_openai.assert_not_called()

    def test_nlp_accepts_text_above_threshold(self):
        """is_nlp_related should not reject text >= NLP_MIN_TEXT_LENGTH."""
        with patch.object(self.mp, "NLP_MIN_TEXT_LENGTH", 100), \
             patch.object(self.mp, "call_openai", new_callable=AsyncMock) as mock_openai:
            mock_openai.return_value = "да"
            text = "x" * 150
            result = asyncio.run(self.mp.is_nlp_related(text))
            mock_openai.assert_called()


class CoverageAndMatchResponseRobustnessTests(unittest.TestCase):
    """Tests for _check_coverage_and_match handling edge-case LLM responses."""

    @classmethod
    def setUpClass(cls):
        cls._stubs = _setup_stubs()
        cls._fake_config, cls._fake_history, cls._fake_tg = cls._stubs
        for mod_name in list(sys.modules.keys()):
            if mod_name in ("message_processor", "models"):
                del sys.modules[mod_name]
        cls.mp = importlib.import_module("message_processor")

    def test_matches_digit_with_trailing_period(self):
        """LLM returns '1.' instead of '1' — should still match."""
        summaries = [
            SummaryInfo(
                content="Summary about GPT-5",
                date=datetime.now(timezone.utc),
                message_count=1,
                channels=["@ai"],
                message_id=100,
            ),
        ]
        msg = _make_message(text="GPT-5 details", channel="@ai", message_id=1)

        with patch.object(self.mp, "call_openai", new_callable=AsyncMock) as mock_openai:
            mock_openai.return_value = "1."
            result = asyncio.run(self.mp._check_coverage_and_match(msg, summaries))
            self.assertIsNotNone(result)
            self.assertEqual(result.message_id, 100)

    def test_matches_digit_with_trailing_comma(self):
        """LLM returns '2,' instead of '2' — should still match."""
        summaries = [
            SummaryInfo(
                content="Summary about GPT-5",
                date=datetime.now(timezone.utc),
                message_count=1,
                channels=["@ai"],
                message_id=100,
            ),
            SummaryInfo(
                content="Summary about BERT",
                date=datetime.now(timezone.utc),
                message_count=1,
                channels=["@nlp"],
                message_id=101,
            ),
        ]
        msg = _make_message(text="BERT details", channel="@nlp", message_id=1)

        with patch.object(self.mp, "call_openai", new_callable=AsyncMock) as mock_openai:
            mock_openai.return_value = "2,"
            result = asyncio.run(self.mp._check_coverage_and_match(msg, summaries))
            self.assertIsNotNone(result)
            self.assertEqual(result.message_id, 101)


class GroupHeaderLengthAccountingTests(unittest.TestCase):
    """Tests for group summary header being accounted for in length enforcement."""

    @classmethod
    def setUpClass(cls):
        cls._stubs = _setup_stubs()
        cls._fake_config, cls._fake_history, cls._fake_tg = cls._stubs
        for mod_name in list(sys.modules.keys()):
            if mod_name in ("message_processor", "models"):
                del sys.modules[mod_name]
        cls.mp = importlib.import_module("message_processor")

    def test_group_summary_total_length_within_limit(self):
        """Group summary including header should not exceed GROUP_SUMMARY_MAX_LENGTH."""
        from models import MessageInfo

        messages = [
            MessageInfo(
                text="AI discussion " * 200,
                channel="@test_group",
                message_id=1,
                date=datetime.now(timezone.utc),
                link="",
            ),
        ]

        long_response = "x" * 12000
        with patch.object(self.mp, "call_openai", new_callable=AsyncMock) as mock_openai:
            mock_openai.return_value = long_response
            result = asyncio.run(self.mp.summarize_group_text(messages))
            self.assertLessEqual(len(result), 12000)
            self.assertTrue(result.startswith("<b>"))


class TelegramMessageLengthGuardTests(unittest.TestCase):
    """Tests for Telegram message length truncation guard."""

    def test_send_truncates_oversized_message(self):
        """send_message_to_target_channel_with_id should truncate messages > 4096 chars."""
        import importlib
        import sys
        import types
        from unittest.mock import AsyncMock, patch

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_telethon = types.ModuleType("telethon")
        fake_telethon.TelegramClient = MagicMock
        fake_telethon.errors = types.ModuleType("errors")
        fake_telethon.errors.FloodWaitError = type("FloodWaitError", (Exception,), {})
        fake_telethon.tl = types.ModuleType("tl")
        fake_telethon.tl.functions = types.ModuleType("functions")
        fake_telethon.tl.functions.channels = types.ModuleType("channels")
        fake_telethon.tl.functions.channels.GetChannelRecommendationsRequest = MagicMock
        fake_telethon.tl.types = types.ModuleType("types")
        fake_telethon.tl.types.InputChannel = MagicMock
        fake_config = types.ModuleType("config")
        fake_config.API_ID = 1
        fake_config.API_HASH = "h"
        fake_config.BOT_TOKEN = "t"
        fake_config.TARGET_CHANNEL = "@target"
        fake_config.SOURCE_GROUPS = set()
        fake_config.MAX_MESSAGES_PER_SOURCE = 100
        fake_config.FETCH_EXAMINED_MULTIPLIER = 3
        fake_config.TELEGRAM_MAX_MESSAGE_LENGTH = 4096
        fake_history = types.ModuleType("history_manager")
        fake_history.load_group_summarization_history = lambda: set()
        fake_history.load_summarization_history = lambda: set()
        fake_cm = types.ModuleType("channel_manager")
        fake_cm.get_all_source_channels = lambda: []
        fake_mp = types.ModuleType("message_processor")
        fake_mp.is_message_processed = lambda msg, proc: False
        fake_models = types.ModuleType("models")
        fake_models.MessageInfo = MagicMock
        fake_utils = types.ModuleType("utils")
        fake_utils.extract_links = lambda text: []
        fake_utils.count_characters = lambda text: len(re.sub(r'<[^>]+>', '', text))
        fake_utils._truncate_html_preserving_tags = _truncate_html_preserving_tags

        with patch.dict(sys.modules, {
            "dotenv": fake_dotenv,
            "telethon": fake_telethon,
            "telethon.errors": fake_telethon.errors,
            "telethon.tl": fake_telethon.tl,
            "telethon.tl.functions": fake_telethon.tl.functions,
            "telethon.tl.functions.channels": fake_telethon.tl.functions.channels,
            "telethon.tl.types": fake_telethon.tl.types,
            "config": fake_config,
            "history_manager": fake_history,
            "channel_manager": fake_cm,
            "message_processor": fake_mp,
            "models": fake_models,
            "utils": fake_utils,
        }):
            sys.modules.pop("telegram_client", None)
            tg = importlib.import_module("telegram_client")

            sent_message = types.SimpleNamespace(id=123)
            fake_bot = MagicMock()
            fake_bot.is_connected = lambda: True
            fake_bot.send_message = AsyncMock(return_value=sent_message)
            fake_bot.start = AsyncMock()
            tg.bot_client = fake_bot
            tg.user_client = None

            long_msg = "x" * 5000
            with patch.object(tg, "_ensure_bot_client", new_callable=AsyncMock):
                result = asyncio.run(tg.send_message_to_target_channel_with_id(long_msg))

            self.assertIsNotNone(result)
            call_args = fake_bot.send_message.call_args
            sent_text = call_args[0][1]
            self.assertLessEqual(fake_utils.count_characters(sent_text), 4096)

    def test_edit_truncates_oversized_message(self):
        """edit_message_in_target_channel should truncate messages > 4096 chars."""
        import importlib
        import sys
        import types
        from unittest.mock import AsyncMock, MagicMock, patch

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_telethon = types.ModuleType("telethon")
        fake_telethon.TelegramClient = MagicMock
        fake_telethon.errors = types.ModuleType("errors")
        fake_telethon.errors.FloodWaitError = type("FloodWaitError", (Exception,), {"seconds": 60})
        fake_telethon.tl = types.ModuleType("tl")
        fake_telethon.tl.functions = types.ModuleType("functions")
        fake_telethon.tl.functions.channels = types.ModuleType("channels")
        fake_telethon.tl.functions.channels.GetChannelRecommendationsRequest = MagicMock
        fake_telethon.tl.types = types.ModuleType("types")
        fake_telethon.tl.types.InputChannel = MagicMock
        fake_config = types.ModuleType("config")
        fake_config.API_ID = 1
        fake_config.API_HASH = "h"
        fake_config.BOT_TOKEN = "t"
        fake_config.TARGET_CHANNEL = "@target"
        fake_config.SOURCE_GROUPS = set()
        fake_config.MAX_MESSAGES_PER_SOURCE = 100
        fake_config.FETCH_EXAMINED_MULTIPLIER = 3
        fake_config.TELEGRAM_MAX_MESSAGE_LENGTH = 4096
        fake_history = types.ModuleType("history_manager")
        fake_history.load_group_summarization_history = lambda: set()
        fake_history.load_summarization_history = lambda: set()
        fake_cm = types.ModuleType("channel_manager")
        fake_cm.get_all_source_channels = lambda: []
        fake_mp = types.ModuleType("message_processor")
        fake_mp.is_message_processed = lambda msg, proc: False
        fake_models = types.ModuleType("models")
        fake_models.MessageInfo = MagicMock
        fake_utils = types.ModuleType("utils")
        fake_utils.extract_links = lambda text: []
        fake_utils.count_characters = lambda text: len(re.sub(r'<[^>]+>', '', text))
        fake_utils._truncate_html_preserving_tags = _truncate_html_preserving_tags

        with patch.dict(sys.modules, {
            "dotenv": fake_dotenv,
            "telethon": fake_telethon,
            "telethon.errors": fake_telethon.errors,
            "telethon.tl": fake_telethon.tl,
            "telethon.tl.functions": fake_telethon.tl.functions,
            "telethon.tl.functions.channels": fake_telethon.tl.functions.channels,
            "telethon.tl.types": fake_telethon.tl.types,
            "config": fake_config,
            "history_manager": fake_history,
            "channel_manager": fake_cm,
            "message_processor": fake_mp,
            "models": fake_models,
            "utils": fake_utils,
        }):
            sys.modules.pop("telegram_client", None)
            tg = importlib.import_module("telegram_client")

            fake_bot = MagicMock()
            fake_bot.is_connected = lambda: True
            fake_bot.edit_message = AsyncMock()
            fake_bot.start = AsyncMock()
            tg.bot_client = fake_bot
            tg.user_client = None

            long_msg = "x" * 5000
            with patch.object(tg, "_ensure_bot_client", new_callable=AsyncMock):
                asyncio.run(tg.edit_message_in_target_channel(1, long_msg))

            call_args = fake_bot.edit_message.call_args
            sent_text = call_args[0][2]
            self.assertLessEqual(fake_utils.count_characters(sent_text), 4096)

    def test_send_uses_html_aware_truncation(self):
        """send_message_to_target_channel_with_id should use _truncate_html_preserving_tags
        instead of raw string slicing, preserving HTML tags on truncation."""
        import importlib
        import sys
        import types
        from unittest.mock import AsyncMock, MagicMock, patch

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_telethon = types.ModuleType("telethon")
        fake_telethon.TelegramClient = MagicMock
        fake_telethon.errors = types.ModuleType("errors")
        fake_telethon.errors.FloodWaitError = type("FloodWaitError", (Exception,), {})
        fake_telethon.tl = types.ModuleType("tl")
        fake_telethon.tl.functions = types.ModuleType("functions")
        fake_telethon.tl.functions.channels = types.ModuleType("channels")
        fake_telethon.tl.functions.channels.GetChannelRecommendationsRequest = MagicMock
        fake_telethon.tl.types = types.ModuleType("types")
        fake_telethon.tl.types.InputChannel = MagicMock
        fake_config = types.ModuleType("config")
        fake_config.API_ID = 1
        fake_config.API_HASH = "h"
        fake_config.BOT_TOKEN = "t"
        fake_config.TARGET_CHANNEL = "@target"
        fake_config.SOURCE_GROUPS = set()
        fake_config.MAX_MESSAGES_PER_SOURCE = 100
        fake_config.FETCH_EXAMINED_MULTIPLIER = 3
        fake_config.TELEGRAM_MAX_MESSAGE_LENGTH = 4096
        fake_history = types.ModuleType("history_manager")
        fake_history.load_group_summarization_history = lambda: set()
        fake_history.load_summarization_history = lambda: set()
        fake_cm = types.ModuleType("channel_manager")
        fake_cm.get_all_source_channels = lambda: []
        fake_mp = types.ModuleType("message_processor")
        fake_mp.is_message_processed = lambda msg, proc: False
        fake_models = types.ModuleType("models")
        fake_models.MessageInfo = MagicMock

        truncate_called = [False]
        def _fake_truncate(text, max_visible):
            truncate_called[0] = True
            return _truncate_html_preserving_tags(text, max_visible)

        fake_utils = types.ModuleType("utils")
        fake_utils.extract_links = lambda text: []
        fake_utils.count_characters = lambda text: len(re.sub(r'<[^>]+>', '', text))
        fake_utils._truncate_html_preserving_tags = _fake_truncate

        with patch.dict(sys.modules, {
            "dotenv": fake_dotenv,
            "telethon": fake_telethon,
            "telethon.errors": fake_telethon.errors,
            "telethon.tl": fake_telethon.tl,
            "telethon.tl.functions": fake_telethon.tl.functions,
            "telethon.tl.functions.channels": fake_telethon.tl.functions.channels,
            "telethon.tl.types": fake_telethon.tl.types,
            "config": fake_config,
            "history_manager": fake_history,
            "channel_manager": fake_cm,
            "message_processor": fake_mp,
            "models": fake_models,
            "utils": fake_utils,
        }):
            sys.modules.pop("telegram_client", None)
            tg = importlib.import_module("telegram_client")

            sent_message = types.SimpleNamespace(id=123)
            fake_bot = MagicMock()
            fake_bot.is_connected = lambda: True
            fake_bot.send_message = AsyncMock(return_value=sent_message)
            fake_bot.start = AsyncMock()
            tg.bot_client = fake_bot
            tg.user_client = None

            msg_with_html = "<b>" + "x" * 5000 + "</b>"
            with patch.object(tg, "_ensure_bot_client", new_callable=AsyncMock):
                asyncio.run(tg.send_message_to_target_channel_with_id(msg_with_html))

            self.assertTrue(truncate_called[0], "_truncate_html_preserving_tags should be called for oversized messages")
            call_args = fake_bot.send_message.call_args
            sent_text = call_args[0][1]
            self.assertTrue(sent_text.endswith("</b>"), "Truncated message should close HTML tags")


class FloodWaitErrorHandlingTests(unittest.TestCase):
    """Tests for FloodWaitError handling in _fetch_from_sources."""

    def test_fetch_skips_source_on_flood_wait(self):
        """_fetch_from_sources should skip a source that raises FloodWaitError
        and continue to the next source."""
        import importlib
        import sys
        import types
        from unittest.mock import AsyncMock, MagicMock, patch

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_telethon = types.ModuleType("telethon")
        fake_telethon.TelegramClient = MagicMock
        FloodWaitError = type("FloodWaitError", (Exception,), {"seconds": 60})
        fake_telethon.errors = types.ModuleType("errors")
        fake_telethon.errors.FloodWaitError = FloodWaitError
        fake_telethon.tl = types.ModuleType("tl")
        fake_telethon.tl.functions = types.ModuleType("functions")
        fake_telethon.tl.functions.channels = types.ModuleType("channels")
        fake_telethon.tl.functions.channels.GetChannelRecommendationsRequest = MagicMock
        fake_telethon.tl.types = types.ModuleType("types")
        fake_telethon.tl.types.InputChannel = MagicMock
        fake_config = types.ModuleType("config")
        fake_config.API_ID = 1
        fake_config.API_HASH = "h"
        fake_config.BOT_TOKEN = "t"
        fake_config.TARGET_CHANNEL = "@target"
        fake_config.SOURCE_GROUPS = set()
        fake_config.MAX_MESSAGES_PER_SOURCE = 100
        fake_config.FETCH_EXAMINED_MULTIPLIER = 3
        fake_config.TELEGRAM_MAX_MESSAGE_LENGTH = 4096
        fake_history = types.ModuleType("history_manager")
        fake_history.load_group_summarization_history = lambda: set()
        fake_history.load_summarization_history = lambda: set()
        fake_cm = types.ModuleType("channel_manager")
        fake_cm.get_all_source_channels = lambda: []
        fake_mp = types.ModuleType("message_processor")
        fake_mp.is_message_processed = lambda msg, proc: False
        fake_models = types.ModuleType("models")
        fake_models.MessageInfo = MagicMock
        fake_utils = types.ModuleType("utils")
        fake_utils.extract_links = lambda text: []
        fake_utils.count_characters = lambda text: len(text)
        fake_utils._truncate_html_preserving_tags = _truncate_html_preserving_tags

        with patch.dict(sys.modules, {
            "dotenv": fake_dotenv,
            "telethon": fake_telethon,
            "telethon.errors": fake_telethon.errors,
            "telethon.tl": fake_telethon.tl,
            "telethon.tl.functions": fake_telethon.tl.functions,
            "telethon.tl.functions.channels": fake_telethon.tl.functions.channels,
            "telethon.tl.types": fake_telethon.tl.types,
            "config": fake_config,
            "history_manager": fake_history,
            "channel_manager": fake_cm,
            "message_processor": fake_mp,
            "models": fake_models,
            "utils": fake_utils,
        }):
            sys.modules.pop("telegram_client", None)
            tg = importlib.import_module("telegram_client")

            fake_user = MagicMock()
            fake_user.is_connected = lambda: True
            fake_user.start = AsyncMock()

            async def iter_flood(*args, **kwargs):
                raise FloodWaitError("flood", request=60)
                yield

            fake_user.iter_messages = iter_flood
            tg.user_client = fake_user
            tg.bot_client = None

            result = asyncio.run(tg._fetch_from_sources(
                ["@ch1", "@ch2"], set(), "channel"
            ))

            self.assertEqual(result, [])


class TelegramHTMLAwareLengthTests(unittest.TestCase):
    """Tests for Telegram length checks using count_characters() instead of len()."""

    def test_send_uses_count_characters_not_len(self):
        """send_message_to_target_channel_with_id should use count_characters for 4096 limit,
        allowing messages with HTML tags that have <= 4096 visible characters."""
        import importlib
        import re
        import sys
        import types
        from unittest.mock import AsyncMock, MagicMock, patch

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_telethon = types.ModuleType("telethon")
        fake_telethon.TelegramClient = MagicMock
        fake_telethon.errors = types.ModuleType("errors")
        fake_telethon.errors.FloodWaitError = type("FloodWaitError", (Exception,), {})
        fake_telethon.tl = types.ModuleType("tl")
        fake_telethon.tl.functions = types.ModuleType("functions")
        fake_telethon.tl.functions.channels = types.ModuleType("channels")
        fake_telethon.tl.functions.channels.GetChannelRecommendationsRequest = MagicMock
        fake_telethon.tl.types = types.ModuleType("types")
        fake_telethon.tl.types.InputChannel = MagicMock
        fake_config = types.ModuleType("config")
        fake_config.API_ID = 1
        fake_config.API_HASH = "h"
        fake_config.BOT_TOKEN = "t"
        fake_config.TARGET_CHANNEL = "@target"
        fake_config.SOURCE_GROUPS = set()
        fake_config.MAX_MESSAGES_PER_SOURCE = 100
        fake_config.FETCH_EXAMINED_MULTIPLIER = 3
        fake_config.TELEGRAM_MAX_MESSAGE_LENGTH = 4096
        fake_history = types.ModuleType("history_manager")
        fake_history.load_group_summarization_history = lambda: set()
        fake_history.load_summarization_history = lambda: set()
        fake_cm = types.ModuleType("channel_manager")
        fake_cm.get_all_source_channels = lambda: []
        fake_mp = types.ModuleType("message_processor")
        fake_mp.is_message_processed = lambda msg, proc: False
        fake_models = types.ModuleType("models")
        fake_models.MessageInfo = MagicMock
        fake_utils = types.ModuleType("utils")
        fake_utils.extract_links = lambda text: []
        fake_utils.count_characters = lambda text: len(re.sub(r'<[^>]+>', '', text))
        fake_utils._truncate_html_preserving_tags = _truncate_html_preserving_tags

        with patch.dict(sys.modules, {
            "dotenv": fake_dotenv,
            "telethon": fake_telethon,
            "telethon.errors": fake_telethon.errors,
            "telethon.tl": fake_telethon.tl,
            "telethon.tl.functions": fake_telethon.tl.functions,
            "telethon.tl.functions.channels": fake_telethon.tl.functions.channels,
            "telethon.tl.types": fake_telethon.tl.types,
            "config": fake_config,
            "history_manager": fake_history,
            "channel_manager": fake_cm,
            "message_processor": fake_mp,
            "models": fake_models,
            "utils": fake_utils,
        }):
            sys.modules.pop("telegram_client", None)
            tg = importlib.import_module("telegram_client")

            sent_message = types.SimpleNamespace(id=123)
            fake_bot = MagicMock()
            fake_bot.is_connected = lambda: True
            fake_bot.send_message = AsyncMock(return_value=sent_message)
            fake_bot.start = AsyncMock()
            tg.bot_client = fake_bot
            tg.user_client = None

            visible_chars = 3900
            html_tags = '<a href="https://example.com/very/long/url/path">[1]</a> ' * 20
            html_msg = html_tags + "x" * visible_chars
            self.assertGreater(len(html_msg), 4096)

            with patch.object(tg, "_ensure_bot_client", new_callable=AsyncMock):
                result = asyncio.run(tg.send_message_to_target_channel_with_id(html_msg))

            self.assertIsNotNone(result)
            fake_bot.send_message.assert_called_once()
            sent_text = fake_bot.send_message.call_args[0][1]
            self.assertEqual(sent_text, html_msg)


class EnforceSummaryLengthInUtilsTests(unittest.TestCase):
    """Tests that enforce_summary_length is available from utils module."""

    def test_enforce_summary_length_importable_from_utils(self):
        """enforce_summary_length should be importable directly from utils."""
        from utils import enforce_summary_length
        self.assertTrue(callable(enforce_summary_length))


class NoCircularImportTests(unittest.TestCase):
    """Tests that history_manager no longer imports from message_processor."""

    def test_history_manager_no_message_processor_import(self):
        """history_manager should not import enforce_summary_length or count_characters
        from message_processor — they should come from utils."""
        import ast
        with open("history_manager.py") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "message_processor":
                for alias in node.names:
                    self.assertNotIn(alias.name, ("enforce_summary_length", "count_characters"),
                                     f"history_manager should not import {alias.name} from message_processor")


class NlpFilterStatsTests(unittest.TestCase):
    """Tests for NLP filter statistics logging."""

    @classmethod
    def setUpClass(cls):
        cls._stubs = _setup_stubs()
        cls._fake_config, cls._fake_history, cls._fake_tg = cls._stubs
        for mod_name in list(sys.modules.keys()):
            if mod_name in ("message_processor", "models"):
                del sys.modules[mod_name]
        cls.mp = importlib.import_module("message_processor")

    def test_process_messages_logs_nlp_filter_stats(self):
        """process_messages should log NLP filter stats (accepted, rejected, ad, short)."""
        msgs = [
            _make_message(text="A" * 200, channel="@ch1", message_id=1),
            _make_message(text="short", channel="@ch2", message_id=2),
            _make_message(text="Курс по ML с сертификацией! " + "B" * 200, channel="@ch3", message_id=3),
        ]

        async def _fake_nlp(text):
            if len(text) < 100:
                return (False, "too_short")
            if "курс" in text.lower():
                return (False, "ad_keyword")
            return (True, "да")

        with patch.object(self.mp, "is_nlp_related", side_effect=_fake_nlp), \
             patch.object(self.mp, "call_openai", new_callable=AsyncMock, return_value="summary"), \
             patch("telegram_client.send_message_to_target_channel_with_id", new_callable=AsyncMock, return_value=42), \
             patch.object(self.mp.logger, "info") as mock_log:
            asyncio.run(self.mp.process_messages(
                msgs, save_changes=False, send_message=True, is_group=False,
            ))

        stats_logs = [call for call in mock_log.call_args_list
                      if "NLP filter" in str(call)]
        self.assertTrue(len(stats_logs) > 0, "Expected NLP filter stats log message")

        log_text = str(stats_logs[0])
        self.assertIn("accepted", log_text)
        self.assertIn("rejected", log_text)


class GroupSummaryHeaderLengthTests(unittest.TestCase):
    """Tests that group summary header length is accounted for using count_characters."""

    @classmethod
    def setUpClass(cls):
        cls._stubs = _setup_stubs()
        cls._fake_config, cls._fake_history, cls._fake_tg = cls._stubs
        for mod_name in list(sys.modules.keys()):
            if mod_name in ("message_processor", "models"):
                del sys.modules[mod_name]
        cls.mp = importlib.import_module("message_processor")

    def test_group_summary_uses_count_characters_for_header(self):
        """summarize_group_text should account for header length using count_characters,
        not len(), so HTML tags in the header don't over-reduce the body allowance."""
        import ast
        with open("message_processor.py") as f:
            source = f.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "summarize_group_text":
                found_count_characters = False
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        func = child.func
                        if isinstance(func, ast.Name) and func.id == "count_characters":
                            found_count_characters = True
                self.assertTrue(
                    found_count_characters,
                    "summarize_group_text should call count_characters for header length"
                )


class MaxCoveredMessageUpdatesTests(unittest.TestCase):
    """Tests for MAX_COVERED_MESSAGE_UPDATES cap on summary updates."""

    @classmethod
    def setUpClass(cls):
        cls._stubs = _setup_stubs()
        cls._fake_config, cls._fake_history, cls._fake_tg = cls._stubs
        for mod_name in list(sys.modules.keys()):
            if mod_name in ("message_processor", "models"):
                del sys.modules[mod_name]
        cls.mp = importlib.import_module("message_processor")

    def test_covered_updates_capped_at_max(self):
        """When more messages are covered than MAX_COVERED_MESSAGE_UPDATES,
        excess messages should be un-marked and included in the new summary."""
        msgs = [
            MessageInfo(text=f"NLP text about AI topic {i} " * 10, channel=f"@ch{i}",
                        message_id=i, date=datetime.now(timezone.utc), link="")
            for i in range(8)
        ]

        matching = SummaryInfo(
            content="Existing summary",
            date=datetime.now(timezone.utc),
            message_count=1,
            channels=["@ch0"],
            message_id=100,
        )

        self._fake_history.load_summaries_history.return_value = [matching]

        with patch.object(self.mp, "is_nlp_related", new_callable=AsyncMock, return_value=(True, "да")), \
             patch.object(self.mp, "_check_coverage_and_match", new_callable=AsyncMock, return_value=matching), \
             patch.object(self.mp, "process_covered_message", new_callable=AsyncMock), \
             patch.object(self.mp, "summarize_text", new_callable=AsyncMock, return_value="new summary"), \
             patch.object(self.mp, "ENABLE_SUMMARIES_DEDUPLICATION", True), \
             patch.object(self.mp, "MAX_COVERED_MESSAGE_UPDATES", 3), \
             patch("telegram_client.send_message_to_target_channel_with_id", new_callable=AsyncMock, return_value=42), \
             patch.object(self.mp, "_save_processing_results", new_callable=AsyncMock), \
             patch.object(self.mp.logger, "info") as mock_log:
            asyncio.run(self.mp.process_messages(
                msgs, save_changes=False, send_message=True, is_group=False,
            ))

        cap_logs = [call for call in mock_log.call_args_list
                    if "MAX_COVERED_MESSAGE_UPDATES" in str(call)]
        self.assertTrue(len(cap_logs) > 0, "Expected a cap log message")

    def test_covered_messages_beyond_cap_unmarked(self):
        """Messages beyond the cap should have is_covered_in_summaries=False
        so they're included in the new summary instead of dropped."""
        msgs = [
            MessageInfo(text=f"NLP AI text {i} " * 10, channel=f"@ch{i}",
                        message_id=i, date=datetime.now(timezone.utc), link="")
            for i in range(6)
        ]

        matching = SummaryInfo(
            content="Existing summary",
            date=datetime.now(timezone.utc),
            message_count=1,
            channels=["@ch0"],
            message_id=100,
        )

        self._fake_history.load_summaries_history.return_value = [matching]

        with patch.object(self.mp, "is_nlp_related", new_callable=AsyncMock, return_value=(True, "да")), \
             patch.object(self.mp, "_check_coverage_and_match", new_callable=AsyncMock, return_value=matching), \
             patch.object(self.mp, "process_covered_message", new_callable=AsyncMock) as mock_process, \
             patch.object(self.mp, "summarize_text", new_callable=AsyncMock, return_value="new summary"), \
             patch.object(self.mp, "ENABLE_SUMMARIES_DEDUPLICATION", True), \
             patch.object(self.mp, "MAX_COVERED_MESSAGE_UPDATES", 2), \
             patch("telegram_client.send_message_to_target_channel_with_id", new_callable=AsyncMock, return_value=42), \
             patch.object(self.mp, "_save_processing_results", new_callable=AsyncMock):
            asyncio.run(self.mp.process_messages(
                msgs, save_changes=False, send_message=True, is_group=False,
            ))

        self.assertEqual(mock_process.call_count, 2, "Only 2 covered messages should be updated (MAX_COVERED_MESSAGE_UPDATES=2)")


class DedupCoveredMessagesExtractionTests(unittest.TestCase):
    """Tests for _dedup_covered_messages extracted function."""

    def _import_mp(self):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_config = types.ModuleType("config")
        fake_config.SIMILARITY_LLM_UPPER = 0.95
        fake_config.ENABLE_SUMMARIES_DEDUPLICATION = True
        fake_config.OPENAI_CHANNEL_SUMMARY_MAX_TOKENS = 1500
        fake_config.OPENAI_GROUP_SUMMARY_MAX_TOKENS = 4000
        fake_config.OPENAI_SUMMARY_TEMPERATURE = 0.3
        fake_config.SOURCE_CHANNELS = set()
        fake_config.DEBUG = False
        fake_config.SUMMARY_MIN_RATIO = 3
        fake_config.SUMMARY_MIN_LENGTH = 800
        fake_config.SUMMARY_MAX_LENGTH = 4000
        fake_config.GROUP_SUMMARY_MIN_LENGTH = 2000
        fake_config.GROUP_SUMMARY_MAX_LENGTH = 12000
        fake_config.NLP_CHECK_MAX_INPUT_CHARS = 2000
        fake_config.COVERAGE_CHECK_MAX_INPUT_CHARS = 2000
        fake_config.NLP_MIN_TEXT_LENGTH = 100
        fake_config.SUMMARY_MAX_INPUT_CHARS_PER_MESSAGE = 3000
        fake_config.NLP_CONCURRENT_CHECKS = 5
        fake_config.NLP_AD_KEYWORDS = ["курс"]
        fake_config.UPDATE_MATCH_MAX_SUMMARIES = 5
        fake_config.UPDATE_MATCH_MAX_CHARS_PER_SUMMARY = 500
        fake_config.ENABLE_SUMMARY_UPDATES = True
        fake_config.MAX_COVERED_MESSAGE_UPDATES = 5

        fake_channel_manager = types.ModuleType("channel_manager")
        fake_channel_manager.create_channel_abbreviation = lambda ch: "AB"
        fake_channel_manager.save_discovered_channel = MagicMock()

        fake_models = types.ModuleType("models")
        fake_models.MessageInfo = type("FakeMI", (), {
            "__init__": lambda self, **kw: None,
            "to_dict": lambda self: {},
        })
        fake_models.SummaryInfo = type("FakeSI", (), {
            "__init__": lambda self, **kw: None,
            "to_dict": lambda self: {},
        })

        fake_prompts = types.ModuleType("prompts")
        fake_prompts.prompts = types.SimpleNamespace(
            CHANNEL_SUMMARY_PROMPT="prompt",
            GROUP_SUMMARY_PROMPT="prompt",
            NLP_RELEVANCE_PROMPT="prompt",
            COVERAGE_AND_MATCH_PROMPT="prompt",
        )

        fake_history = types.ModuleType("history_manager")
        fake_history.load_summaries_history = MagicMock(return_value=[])
        fake_history.load_group_summaries_history = MagicMock(return_value=[])
        fake_history.save_summarization_history = MagicMock()
        fake_history.save_group_summarization_history = MagicMock()
        fake_history.save_summary_to_history = MagicMock()
        fake_history.save_group_summary_to_history = MagicMock()
        fake_history.update_group_last_run = MagicMock()
        fake_history.update_existing_summary = AsyncMock()
        fake_history.save_updated_summary = AsyncMock()

        fake_utils = types.ModuleType("utils")
        fake_utils.call_openai = AsyncMock(return_value="ok")
        fake_utils.extract_links = lambda text: []
        fake_utils.count_characters = len
        fake_utils.enforce_summary_length = lambda text, max_chars: text
        fake_utils.text_hash = lambda text: hashlib.sha256(text.encode()).hexdigest()[:16]

        fake_telegram_client = types.ModuleType("telegram_client")
        fake_telegram_client.send_message_to_target_channel_with_id = AsyncMock(return_value=42)

        with patch.dict(sys.modules, {
            "dotenv": fake_dotenv,
            "config": fake_config,
            "channel_manager": fake_channel_manager,
            "models": fake_models,
            "prompts": fake_prompts,
            "history_manager": fake_history,
            "utils": fake_utils,
            "telegram_client": fake_telegram_client,
        }):
            sys.modules.pop("message_processor", None)
            mp = importlib.import_module("message_processor")
        return mp

    def test_dedup_covered_messages_returns_uncovered(self):
        """_dedup_covered_messages should return only non-covered messages."""
        mp = self._import_mp()
        msg = MagicMock()
        msg.is_covered_in_summaries = None
        msg.text = "test message about NLP"
        msgs = [msg]

        with patch.object(mp, "_check_coverage_and_match", new_callable=AsyncMock, return_value=None), \
             patch.object(mp, "load_summaries_history", return_value=[MagicMock()]):
            sem = asyncio.Semaphore(5)
            result = asyncio.run(mp._dedup_covered_messages(msgs, False, sem, 0.0))

        self.assertEqual(len(result), 1)
        self.assertIs(result[0].is_covered_in_summaries, False)

    def test_dedup_covered_messages_filters_covered(self):
        """_dedup_covered_messages should filter out covered messages."""
        mp = self._import_mp()
        msg = MagicMock()
        msg.is_covered_in_summaries = None
        msg.text = "test message"
        msgs = [msg]
        matching = MagicMock()

        with patch.object(mp, "_check_coverage_and_match", new_callable=AsyncMock, return_value=matching), \
             patch.object(mp, "process_covered_message", new_callable=AsyncMock), \
             patch.object(mp, "load_summaries_history", return_value=[matching]):
            sem = asyncio.Semaphore(5)
            result = asyncio.run(mp._dedup_covered_messages(msgs, False, sem, 0.0))

        self.assertEqual(len(result), 0)
        self.assertIs(msg.is_covered_in_summaries, True)

    def test_dedup_covered_messages_returns_all_when_no_summaries(self):
        """_dedup_covered_messages should return all messages when no summaries exist."""
        mp = self._import_mp()
        msg = MagicMock()
        msgs = [msg]

        with patch.object(mp, "load_summaries_history", return_value=[]):
            sem = asyncio.Semaphore(5)
            result = asyncio.run(mp._dedup_covered_messages(msgs, False, sem, 0.0))

        self.assertEqual(len(result), 1)
