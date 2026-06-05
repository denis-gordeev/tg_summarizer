import asyncio
import importlib
import sys
import types
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch


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
    fake_config.SIMILARITY_LLM_LOWER = 0.7
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
    sys.modules["config"] = fake_config

    # --- Stub history_manager ---
    fake_history = types.ModuleType("history_manager")
    fake_history.get_recent_summaries_context = MagicMock(return_value="")
    fake_history.get_recent_group_summaries_context = MagicMock(return_value="")
    fake_history.save_summarization_history = MagicMock()
    fake_history.save_group_summarization_history = MagicMock()
    fake_history.save_summary_to_history = MagicMock()
    fake_history.save_group_summary_to_history = MagicMock()
    fake_history.update_group_last_run = MagicMock()
    fake_history.find_relevant_summary_for_update = AsyncMock(return_value=None)
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
        DUPLICATE_CHECK_PROMPT="dup",
        NLP_RELEVANCE_PROMPT="nlp",
        SUMMARY_COVERAGE_CHECK_PROMPT="cov",
        GROUP_SUMMARY_COVERAGE_CHECK_PROMPT="gcov",
        CHANNEL_SUMMARY_PROMPT="channel summary, max {max_summary_length} chars",
        GROUP_SUMMARY_PROMPT="group summary, max {max_summary_length} chars",
        FIND_RELEVANT_SUMMARY_PROMPT="find",
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

    def test_three_band_dedup_upper_threshold_uses_llm(self):
        """Messages with ratio between LOWER and UPPER should call LLM, not auto-duplicate."""
        base_text = "A detailed article about machine learning and neural networks " * 5
        similar_text = "A detailed article about machine learning and neural networks " * 4 + "A different topic entirely about space exploration "

        messages = [
            _make_message(text=base_text, channel="@ch", message_id=1),
            _make_message(text=similar_text, channel="@ch", message_id=2),
        ]

        self._fake_config.ENABLE_SUMMARIES_DEDUPLICATION = False

        with patch.object(self.mp, "are_messages_duplicate", new_callable=AsyncMock) as mock_dup:
            mock_dup.return_value = False

            result = asyncio.run(self.mp.remove_duplicates(messages))

            self.assertTrue(mock_dup.called, "LLM duplicate check should be called for ambiguous ratio")

    def test_three_band_dedup_near_identical_skips_llm(self):
        """Nearly identical messages (ratio > UPPER) should be auto-deduplicated without LLM."""
        base_text = "This is a detailed article about transformer models in NLP research " * 5
        near_identical = base_text + " extra"

        messages = [
            _make_message(text=base_text, channel="@ch", message_id=1),
            _make_message(text=near_identical, channel="@ch", message_id=2),
        ]

        self._fake_config.ENABLE_SUMMARIES_DEDUPLICATION = False

        with patch.object(self.mp, "are_messages_duplicate", new_callable=AsyncMock) as mock_dup:
            result = asyncio.run(self.mp.remove_duplicates(messages))

            self.assertFalse(mock_dup.called, "LLM should NOT be called for near-identical messages")
            self.assertEqual(len(result), 1, "Near-identical message should be auto-deduplicated")


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


class AreMessagesDuplicateRussianResponseTests(unittest.TestCase):
    """Tests for are_messages_duplicate Russian response handling."""

    @classmethod
    def setUpClass(cls):
        _setup_stubs()
        import message_processor
        cls.mp = message_processor

    def test_are_messages_duplicate_recognizes_russian_yes(self):
        """are_messages_duplicate should recognize 'да' (Russian yes) as duplicate."""
        from models import MessageInfo
        from unittest.mock import AsyncMock, patch

        msg_a = MessageInfo(text="Some text about AI", channel="@ch", message_id=1,
                            date=datetime.now(timezone.utc), link="")
        msg_b = MessageInfo(text="Some text about AI too", channel="@ch", message_id=2,
                            date=datetime.now(timezone.utc), link="")

        with patch.object(self.mp, "call_openai", new_callable=AsyncMock) as mock_openai:
            mock_openai.return_value = "да"
            result = asyncio.run(self.mp.are_messages_duplicate(msg_a, msg_b))
            self.assertTrue(result, "Russian 'да' should be recognized as duplicate")

    def test_are_messages_duplicate_recognizes_russian_no(self):
        """are_messages_duplicate should recognize 'нет' (Russian no) as not duplicate."""
        from models import MessageInfo
        from unittest.mock import AsyncMock, patch

        msg_a = MessageInfo(text="Some text about AI", channel="@ch", message_id=1,
                            date=datetime.now(timezone.utc), link="")
        msg_b = MessageInfo(text="Completely different topic", channel="@ch", message_id=2,
                            date=datetime.now(timezone.utc), link="")

        with patch.object(self.mp, "call_openai", new_callable=AsyncMock) as mock_openai:
            mock_openai.return_value = "нет"
            result = asyncio.run(self.mp.are_messages_duplicate(msg_a, msg_b))
            self.assertFalse(result, "Russian 'нет' should be recognized as not duplicate")


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


if __name__ == "__main__":
    unittest.main()
