import asyncio
import importlib
import sys
import time
import types
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from models import MessageInfo, SummaryInfo


def _setup_stubs():
    """Install all fake modules before importing message_processor."""
    # --- Stub utils ---
    async def fake_call_openai(system_prompt, user_content, max_tokens=300, **kwargs):
        return "AI breakthrough reported [1]."

    fake_utils = types.ModuleType("utils")
    fake_utils.call_openai = fake_call_openai
    fake_utils.extract_links = lambda text: []
    fake_utils.count_characters = lambda text: len(text)
    fake_utils.text_hash = lambda text: "abc123"
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
    fake_config.ENABLE_SUMMARY_UPDATES = True
    fake_config.SUMMARY_MIN_RATIO = 3
    fake_config.SUMMARY_MIN_LENGTH = 800
    fake_config.SUMMARY_MAX_LENGTH = 4000
    fake_config.GROUP_SUMMARY_MIN_LENGTH = 2000
    fake_config.GROUP_SUMMARY_MAX_LENGTH = 12000
    fake_config.NLP_CHECK_MAX_INPUT_CHARS = 2000
    fake_config.SUMMARY_MAX_INPUT_CHARS_PER_MESSAGE = 3000
    fake_config.NLP_CONCURRENT_CHECKS = 5
    fake_config.NLP_AD_KEYWORDS = [
        "курс", "вебинар", "регистраци", "скидк", "промокод", "бесплатный курс",
        "платный курс", "мастер-класс", "стажировк", "hire", "hiring day",
        "карьерный трек", "bootcamp", "boot camp",
    ]
    fake_config.UPDATE_MATCH_MAX_SUMMARIES = 5
    fake_config.UPDATE_MATCH_MAX_CHARS_PER_SUMMARY = 500
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
    """Tests for _check_coverage_and_match truncating msg.text."""

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
        self._fake_config.NLP_CHECK_MAX_INPUT_CHARS = 2000
        msg = _make_message(text=long_text, channel="@ai", message_id=1)

        with patch.object(self.mp, "call_openai", new_callable=AsyncMock) as mock_openai:
            mock_openai.return_value = "НЕТ"
            asyncio.run(self.mp._check_coverage_and_match(msg, summaries))
            user_content = mock_openai.call_args[0][1]
            self.assertIn(long_text[:2000], user_content)
            self.assertNotIn(long_text[2001:], user_content)


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
