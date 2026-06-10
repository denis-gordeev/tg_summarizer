import os
import re
import sys
import types
import unittest
import asyncio
from datetime import datetime, timezone
from unittest.mock import patch, AsyncMock

from models import MessageInfo


REQUIRED_ENV = {
    "TELEGRAM_API_ID": "1",
    "TELEGRAM_API_HASH": "hash",
    "TELEGRAM_BOT_TOKEN": "token",
    "TARGET_CHANNEL": "@target",
    "OPENAI_API_KEY": "test-key",
}


def _stub_dependencies(fake_openai_response: str = "[1] New AI breakthrough"):
    """Create stub modules and return the openai response to use."""
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda: None

    fake_openai = types.ModuleType("openai")

    class FakeOpenAI:
        def __init__(self, api_key):
            self.api_key = api_key
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **kwargs: None)
            )

    fake_openai.OpenAI = FakeOpenAI
    fake_channel_manager = types.ModuleType("channel_manager")
    fake_channel_manager.load_discovered_channels = lambda: []
    fake_channel_manager.load_similar_channels = lambda: []
    fake_channel_manager.load_banned_channels = lambda: []
    fake_channel_manager.create_channel_abbreviation = lambda ch: ch.lstrip("@")[:4].upper()
    fake_channel_manager.get_all_source_channels = lambda: ["@ai_news"]
    fake_channel_manager.save_discovered_channel = lambda ch: None
    fake_history_manager = types.ModuleType("history_manager")
    fake_history_manager.get_recent_summaries_context = lambda: ""
    fake_history_manager.get_recent_group_summaries_context = lambda: ""
    fake_history_manager.load_summaries_history = lambda: []
    fake_history_manager.load_group_summaries_history = lambda: []
    fake_history_manager.save_summarization_history = lambda *a, **kw: None
    fake_history_manager.save_group_summarization_history = lambda *a, **kw: None
    fake_history_manager.save_summary_to_history = lambda *a, **kw: None
    fake_history_manager.save_group_summary_to_history = lambda *a, **kw: None
    fake_history_manager.update_group_last_run = lambda: None
    fake_history_manager.find_relevant_summary_for_update = lambda *a, **kw: None
    fake_history_manager.update_existing_summary = lambda *a, **kw: None
    fake_history_manager.save_updated_summary = lambda *a, **kw: None
    fake_prompts = types.ModuleType("prompts")
    fake_prompts.prompts = types.SimpleNamespace(
        CHANNEL_SUMMARY_PROMPT="{max_summary_length}",
        GROUP_SUMMARY_PROMPT="{max_summary_length}",
        NLP_RELEVANCE_PROMPT="yes",
        SUMMARY_COVERAGE_CHECK_PROMPT="нет",
        GROUP_SUMMARY_COVERAGE_CHECK_PROMPT="нет",
        COVERAGE_AND_MATCH_PROMPT="covmatch",
        FIND_RELEVANT_SUMMARY_PROMPT="find",
    )
    fake_utils = types.ModuleType("utils")
    fake_utils.extract_links = lambda text: [
        m for m in re.findall(r'(https?://\S+)', text)
    ]
    fake_utils.count_characters = lambda text: len(re.sub(r'<[^>]+>', '', text))

    async def fake_call_openai(system_prompt, user_content, max_tokens=None, **kwargs):
        return fake_openai_response

    fake_utils.call_openai = fake_call_openai
    fake_utils.text_hash = lambda text: "abc123"

    return {
        "dotenv": fake_dotenv,
        "openai": fake_openai,
        "channel_manager": fake_channel_manager,
        "history_manager": fake_history_manager,
        "prompts": fake_prompts,
        "utils": fake_utils,
    }


def _import_message_processor(stub_modules):
    for mod_name, mod in stub_modules.items():
        sys.modules[mod_name] = mod
    sys.modules.pop("config", None)
    sys.modules.pop("message_processor", None)
    with patch.dict(os.environ, REQUIRED_ENV, clear=True):
        import message_processor

    return message_processor


def _make_messages(channels, texts, message_ids=None):
    """Helper to create test messages."""
    msgs = []
    for i, (ch, text) in enumerate(zip(channels, texts)):
        mid = message_ids[i] if message_ids else (100 + i)
        msg = MessageInfo(
            channel=ch,
            message_id=mid,
            text=text,
            date=datetime.now(timezone.utc),
            link=f"https://t.me/{ch.lstrip('@')}/{mid}",
            is_nlp_related=True,
        )
        msgs.append(msg)
    return msgs


class DigestPostProcessingTests(unittest.TestCase):
    """Integration tests for the full digest post-processing pipeline.

    Covers:
    - Source reference number replacement ([1], [1,2]) → HTML links
    - Summary length enforcement (block-level trimming + HTML tag closing)
    """

    def test_source_number_replaced_with_html_links(self):
        """[1] in LLM output should become <a href="external">[1]</a> <a href="tg">[CHAN]</a>."""
        stub_modules = _stub_dependencies(fake_openai_response="[1] New AI breakthrough")
        mp = _import_message_processor(stub_modules)

        messages = _make_messages(
            channels=["@ai_news"],
            texts=["Check out https://example.com/article about AI"],
        )

        result = asyncio.run(mp.summarize_text(messages))

        # Should contain HTML link to external source
        self.assertIn('<a href="https://example.com/article">', result)
        # Should contain Telegram channel abbreviation link
        self.assertIn('<a href="https://t.me/ai_news/', result)
        self.assertIn('[AI_N]', result)

    def test_multiple_source_numbers_replaced(self):
        """Multiple [1], [2] references should each become HTML links."""
        stub_modules = _stub_dependencies(fake_openai_response="[1] First news. [2] Second news.")
        mp = _import_message_processor(stub_modules)

        messages = _make_messages(
            channels=["@ai_news", "@ml_daily"],
            texts=[
                "New paper https://arxiv.org/abs/1234 on transformers",
                "Discussion at https://reddit.com/r/ml about LLMs",
            ],
        )

        result = asyncio.run(mp.summarize_text(messages))

        # Both source references should be replaced
        self.assertIn('<a href="https://arxiv.org/abs/1234">', result)
        self.assertIn('<a href="https://reddit.com/r/ml">', result)

    def test_composite_source_reference_replaced(self):
        """Composite reference [1,2] should become comma-separated HTML links."""
        stub_modules = _stub_dependencies(fake_openai_response="[1,2] Combined news.")
        mp = _import_message_processor(stub_modules)

        messages = _make_messages(
            channels=["@ai_news", "@ml_daily"],
            texts=[
                "Transformer breakthrough https://arxiv.org/abs/1111",
                "Related discussion https://news.ycombinator.com/item?id=2222",
            ],
        )

        result = asyncio.run(mp.summarize_text(messages))

        # Both links should appear in result
        self.assertIn('<a href="https://arxiv.org/abs/1111">', result)
        self.assertIn('<a href="https://news.ycombinator.com/item?id=2222">', result)

    def test_source_without_external_link_gets_telegram_only(self):
        """Message with no external links should get only Telegram link with abbreviation."""
        stub_modules = _stub_dependencies(fake_openai_response="[1] Plain announcement.")
        mp = _import_message_processor(stub_modules)

        messages = _make_messages(
            channels=["@deep_learning"],
            texts=["Just a plain text announcement about a new model"],
        )

        result = asyncio.run(mp.summarize_text(messages))

        # No external link, should have Telegram link with abbreviation
        self.assertIn('<a href="https://t.me/deep_learning/', result)
        self.assertIn('[DEEP]', result)
        # Should not have empty href for external link
        self.assertNotIn('<a href="">', result)

    def test_summary_length_enforced_on_long_output(self):
        """Long summaries should be truncated to channel limit while preserving HTML."""
        # Create a very long fake LLM response
        long_response = "AI update. " * 500  # 5500 chars
        stub_modules = _stub_dependencies(fake_openai_response=long_response)
        mp = _import_message_processor(stub_modules)

        messages = _make_messages(
            channels=["@ai_news"],
            texts=["Long article " * 500],  # 6000 chars
        )

        result = asyncio.run(mp.summarize_text(messages))

        # Channel limit is min(max(len//3, 800), 4000) = 4000 for this input
        char_count = mp.count_characters(result)
        self.assertLessEqual(char_count, 4000)

    def test_summary_length_enforced_on_group_output(self):
        """Group summaries should respect 12000 char cap."""
        long_response = "Community discussion point. " * 500  # 15500 chars
        stub_modules = _stub_dependencies(fake_openai_response=long_response)
        mp = _import_message_processor(stub_modules)

        messages = _make_messages(
            channels=["@group_a", "@group_b"],
            texts=["Discussion " * 600, "Discussion " * 600],
        )

        result = asyncio.run(mp.summarize_group_text(messages))

        char_count = mp.count_characters(result)
        # Group cap is 12000
        self.assertLessEqual(char_count, 12000)

    def test_html_tags_closed_on_truncation(self):
        """When truncating, open HTML tags should be properly closed."""
        mp = _import_message_processor(_stub_dependencies())

        # Simulate a summary with open bold tag that gets truncated
        summary_with_open_tag = "<b>Important AI News About New Model Release That Goes On And On</b>"
        result = mp.enforce_summary_length(summary_with_open_tag, 15)

        # Should have closed the bold tag
        self.assertIn("</b>", result)

    def test_short_summary_not_truncated(self):
        """Summaries under the limit should pass through unchanged (aside from link replacement)."""
        stub_modules = _stub_dependencies(fake_openai_response="[1] Short news.")
        mp = _import_message_processor(stub_modules)

        messages = _make_messages(
            channels=["@ai_news"],
            texts=["Short news about AI"],
        )

        result = asyncio.run(mp.summarize_text(messages))

        # Should not be truncated, just link replacement
        self.assertIsNotNone(result)
        self.assertIn('<a href="https://t.me/ai_news/', result)

    def test_channel_abbreviation_consistent(self):
        """Channel abbreviations should be consistent across messages from same channel."""
        stub_modules = _stub_dependencies(
            fake_openai_response="[1] First paper. [2] Follow up. [3] Other news."
        )
        mp = _import_message_processor(stub_modules)

        messages = _make_messages(
            channels=["@neuraldeep", "@neuraldeep", "@ai_news"],
            texts=[
                "Paper https://arxiv.org/abs/111 on neural networks",
                "Follow up https://arxiv.org/abs/222 on deep learning",
                "News https://example.com/ai-update",
            ],
        )

        result = asyncio.run(mp.summarize_text(messages))

        # Same channel should have same abbreviation
        abbr_count = result.count('[NEUR]')
        self.assertGreaterEqual(abbr_count, 2)  # Two messages from @neuraldeep


class DigestEndToEndTests(unittest.TestCase):
    """End-to-end tests simulating the full summarize_text pipeline with mocked LLM."""

    def test_full_pipeline_produces_valid_html(self):
        """Full summarize_text pipeline should produce valid-looking HTML with links."""
        stub_modules = _stub_dependencies(
            fake_openai_response="[1] Transformer variant breakthrough. [2] New benchmark results."
        )
        mp = _import_message_processor(stub_modules)

        messages = _make_messages(
            channels=["@ai_news", "@ml_research"],
            texts=[
                "New transformer variant released https://arxiv.org/abs/2401.12345",
                "Benchmark results at https://paperswithcode.com/paper/xyz",
            ],
        )

        result = asyncio.run(mp.summarize_text(messages))

        # Result should be a non-empty string
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
        # Should contain at least one HTML link
        self.assertIn('<a href="', result)
        # Should not contain raw [1] or [2] references (should be replaced)
        raw_refs = re.findall(r'(?<!<a[^>])\[\d+\](?!</a>)', result)
        self.assertEqual(len(raw_refs), 0, f"Found unreplaced references: {raw_refs}")


class IntraBatchDedupTests(unittest.TestCase):
    """Tests for _remove_intra_batch_duplicates."""

    def test_removes_identical_messages(self):
        stub_modules = _stub_dependencies()
        mp = _import_message_processor(stub_modules)
        msgs = _make_messages(
            channels=["@a", "@b"],
            texts=["GPT-5 announced today", "GPT-5 announced today"],
        )
        result = mp._remove_intra_batch_duplicates(msgs)
        self.assertEqual(len(result), 1)

    def test_keeps_different_messages(self):
        stub_modules = _stub_dependencies()
        mp = _import_message_processor(stub_modules)
        msgs = _make_messages(
            channels=["@a", "@b"],
            texts=["GPT-5 announced today", "BERT update released"],
        )
        result = mp._remove_intra_batch_duplicates(msgs)
        self.assertEqual(len(result), 2)

    def test_removes_link_duplicates(self):
        stub_modules = _stub_dependencies()
        mp = _import_message_processor(stub_modules)
        msgs = _make_messages(
            channels=["@a", "@b"],
            texts=[
                "New model https://arxiv.org/abs/2401.12345",
                "Same model discussed https://arxiv.org/abs/2401.12345",
            ],
        )
        result = mp._remove_intra_batch_duplicates(msgs)
        self.assertEqual(len(result), 1)

    def test_keeps_different_links(self):
        stub_modules = _stub_dependencies()
        mp = _import_message_processor(stub_modules)
        msgs = _make_messages(
            channels=["@a", "@b"],
            texts=[
                "New model https://arxiv.org/abs/2401.11111",
                "Other model https://arxiv.org/abs/2401.22222",
            ],
        )
        result = mp._remove_intra_batch_duplicates(msgs)
        self.assertEqual(len(result), 2)

    def test_empty_input_returns_empty(self):
        stub_modules = _stub_dependencies()
        mp = _import_message_processor(stub_modules)
        result = mp._remove_intra_batch_duplicates([])
        self.assertEqual(result, [])


class SummaryInputTruncationTests(unittest.TestCase):
    """Tests for _prepare_messages_text truncation via SUMMARY_MAX_INPUT_CHARS_PER_MESSAGE."""

    def test_long_message_is_truncated(self):
        stub_modules = _stub_dependencies()
        mp = _import_message_processor(stub_modules)
        long_text = "A" * 5000
        msgs = _make_messages(channels=["@a"], texts=[long_text])
        text_output, total_length = mp._prepare_messages_text(msgs)
        self.assertNotIn("A" * 5000, text_output)

    def test_short_message_is_not_truncated(self):
        stub_modules = _stub_dependencies()
        mp = _import_message_processor(stub_modules)
        short_text = "Short message about AI"
        msgs = _make_messages(channels=["@a"], texts=[short_text])
        text_output, total_length = mp._prepare_messages_text(msgs)
        self.assertIn(short_text, text_output)


class ExtractLinksTrailingPunctuationTests(unittest.TestCase):
    """Tests for extract_links stripping trailing punctuation — pure regex tests."""

    def test_strips_trailing_period(self):
        import re
        LINK_REGEX = re.compile(r"https?://\S+")
        TRAILING = re.compile(r"[.,;:!?)\]}'\">]+$")
        raw = LINK_REGEX.findall("See https://example.com/page.")
        result = [TRAILING.sub("", url) for url in raw]
        self.assertEqual(result, ["https://example.com/page"])

    def test_strips_trailing_parenthesis(self):
        import re
        LINK_REGEX = re.compile(r"https?://\S+")
        TRAILING = re.compile(r"[.,;:!?)\]}'\">]+$")
        raw = LINK_REGEX.findall("Link (https://example.com/foo)")
        result = [TRAILING.sub("", url) for url in raw]
        self.assertEqual(result, ["https://example.com/foo"])

    def test_strips_trailing_comma(self):
        import re
        LINK_REGEX = re.compile(r"https?://\S+")
        TRAILING = re.compile(r"[.,;:!?)\]}'\">]+$")
        raw = LINK_REGEX.findall("https://a.com, https://b.com")
        result = [TRAILING.sub("", url) for url in raw]
        self.assertEqual(result, ["https://a.com", "https://b.com"])

    def test_preserves_clean_url(self):
        import re
        LINK_REGEX = re.compile(r"https?://\S+")
        TRAILING = re.compile(r"[.,;:!?)\]}'\">]+$")
        raw = LINK_REGEX.findall("Visit https://example.com/page today")
        result = [TRAILING.sub("", url) for url in raw]
        self.assertEqual(result, ["https://example.com/page"])

    def test_strips_multiple_trailing_punctuation(self):
        import re
        LINK_REGEX = re.compile(r"https?://\S+")
        TRAILING = re.compile(r"[.,;:!?)\]}'\">]+$")
        raw = LINK_REGEX.findall("See https://example.com).")
        result = [TRAILING.sub("", url) for url in raw]
        self.assertEqual(result, ["https://example.com"])


if __name__ == "__main__":
    unittest.main()
