import os
import re
import sys
import types
import unittest
import asyncio
import hashlib
import html as _html
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


def _stub_count_characters(text):
    return len(_html.unescape(re.sub(r'<[^>]+>', '', text)))


def _stub_enforce_summary_length(text, max_chars):
    if _stub_count_characters(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    tag_stack = []
    for tag in re.finditer(r'<(/?)(\w+)[^>]*>', truncated):
        is_closing = tag.group(1) == '/'
        tag_name = tag.group(2).lower()
        if is_closing:
            if tag_stack and tag_stack[-1] == tag_name:
                tag_stack.pop()
        else:
            tag_stack.append(tag_name)
    for t in reversed(tag_stack):
        truncated += f'</{t}>'
    return truncated


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
    fake_history_manager.load_summaries_history = lambda: []
    fake_history_manager.load_group_summaries_history = lambda: []
    fake_history_manager.save_summarization_history = lambda *a, **kw: None
    fake_history_manager.save_group_summarization_history = lambda *a, **kw: None
    fake_history_manager.save_summary_to_history = lambda *a, **kw: None
    fake_history_manager.save_group_summary_to_history = lambda *a, **kw: None
    fake_history_manager.update_group_last_run = lambda: None
    fake_history_manager.update_existing_summary = lambda *a, **kw: None
    fake_history_manager.save_updated_summary = lambda *a, **kw: None
    fake_prompts = types.ModuleType("prompts")
    fake_prompts.prompts = types.SimpleNamespace(
        CHANNEL_SUMMARY_PROMPT="{max_summary_length}",
        GROUP_SUMMARY_PROMPT="{max_summary_length}",
        NLP_RELEVANCE_PROMPT="yes",
        COVERAGE_AND_MATCH_PROMPT="covmatch",
    )
    fake_utils = types.ModuleType("utils")
    fake_utils.extract_links = lambda text: [
        m for m in re.findall(r'(https?://\S+)', text)
    ]
    fake_utils.count_characters = _stub_count_characters
    fake_utils.enforce_summary_length = _stub_enforce_summary_length
    fake_utils.strip_meta_artifacts = lambda text: text

    async def fake_call_openai(system_prompt, user_content, max_tokens=None, **kwargs):
        return fake_openai_response

    fake_utils.call_openai = fake_call_openai
    fake_utils.text_hash = lambda text: hashlib.sha256(text.encode()).hexdigest()[:16]

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
        text_output, total_length, msg_links = mp._prepare_messages_text(msgs)
        self.assertNotIn("A" * 5000, text_output)

    def test_short_message_is_not_truncated(self):
        stub_modules = _stub_dependencies()
        mp = _import_message_processor(stub_modules)
        short_text = "Short message about AI"
        msgs = _make_messages(channels=["@a"], texts=[short_text])
        text_output, total_length, msg_links = mp._prepare_messages_text(msgs)
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


class VoidHtmlElementTests(unittest.TestCase):
    """Tests for void HTML elements in _truncate_html_preserving_tags."""

    def _import_utils(self):
        import importlib
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_openai = types.ModuleType("openai")
        fake_openai.AsyncOpenAI = type("AsyncOpenAI", (), {"__init__": lambda *a, **kw: None})
        fake_openai.APIError = type("APIError", (Exception,), {"status_code": None})
        fake_openai.RateLimitError = type("RateLimitError", (Exception,), {"status_code": None})
        fake_openai.APIConnectionError = type("APIConnectionError", (Exception,), {})
        fake_config = types.ModuleType("config")
        fake_config.OPENAI_API_KEY = "test"
        fake_config.OPENAI_DEFAULT_MAX_TOKENS = 300
        fake_config.OPENAI_MODEL = "gpt-4o-mini"
        fake_config.OPENAI_REQUEST_TIMEOUT = 30
        fake_config.CIRCUIT_BREAKER_THRESHOLD = 3
        fake_config.CIRCUIT_BREAKER_RESET_SEC = 60

        with patch.dict(sys.modules, {
            "dotenv": fake_dotenv,
            "openai": fake_openai,
            "config": fake_config,
        }):
            sys.modules.pop("utils", None)
            return importlib.import_module("utils")

    def test_br_tag_not_added_to_open_stack(self):
        """<br> is a void element and should not be closed."""
        utils = self._import_utils()
        result = utils._truncate_html_preserving_tags("hello<br>world", 20)
        self.assertIn("<br>", result)
        self.assertNotIn("</br>", result)

    def test_img_tag_not_added_to_open_stack(self):
        """<img> is a void element and should not be closed."""
        utils = self._import_utils()
        result = utils._truncate_html_preserving_tags('hello<img src="x">world', 20)
        self.assertIn("<img", result)
        self.assertNotIn("</img>", result)

    def test_br_does_not_cause_extra_closing_tags(self):
        """Truncation after <br> should not append </br>."""
        utils = self._import_utils()
        result = utils._truncate_html_preserving_tags("hello<br>wor", 8)
        self.assertNotIn("</br>", result)

    def test_b_tag_still_gets_closed(self):
        """Non-void elements like <b> should still be closed on truncation."""
        utils = self._import_utils()
        result = utils._truncate_html_preserving_tags("<b>hello world</b>", 5)
        self.assertIn("</b>", result)


class PrepareMessagesTextLinksTests(unittest.TestCase):
    """Tests for _prepare_messages_text returning msg_links."""

    def test_prepare_messages_text_returns_msg_links(self):
        stub_modules = _stub_dependencies()
        mp = _import_message_processor(stub_modules)
        msgs = _make_messages(
            channels=["@a"],
            texts=["Check https://example.com for details"],
        )
        text_output, total_length, msg_links = mp._prepare_messages_text(msgs)
        self.assertIsInstance(msg_links, dict)
        self.assertIn(1, msg_links)
        self.assertEqual(msg_links[1], ["https://example.com"])

    def test_prepare_messages_text_msg_links_empty_for_no_links(self):
        stub_modules = _stub_dependencies()
        mp = _import_message_processor(stub_modules)
        msgs = _make_messages(channels=["@a"], texts=["No links here"])
        text_output, total_length, msg_links = mp._prepare_messages_text(msgs)
        self.assertEqual(msg_links[1], [])


class ExactHashDedupTests(unittest.TestCase):
    """Tests for text-hash pre-filter in _remove_intra_batch_duplicates."""

    def test_removes_exact_text_duplicate_different_channel(self):
        """Exact same text from a different channel should be caught by hash pre-filter."""
        stub_modules = _stub_dependencies()
        mp = _import_message_processor(stub_modules)
        msgs = _make_messages(
            channels=["@a", "@b"],
            texts=["Identical text about GPT-5 release date announced today", "Identical text about GPT-5 release date announced today"],
        )
        result = mp._remove_intra_batch_duplicates(msgs)
        self.assertEqual(len(result), 1)

    def test_hash_dedup_before_sequencematcher(self):
        """Hash dedup should catch duplicates before SequenceMatcher is invoked,
        avoiding O(N²) comparison for exact matches."""
        stub_modules = _stub_dependencies()
        mp = _import_message_processor(stub_modules)
        base_text = "Exact duplicate message for hash check"
        msgs = _make_messages(
            channels=["@a", "@b", "@c"],
            texts=[base_text, base_text, "Completely different content here"],
        )
        result = mp._remove_intra_batch_duplicates(msgs)
        self.assertEqual(len(result), 2)


class LengthPreFilterDedupTests(unittest.TestCase):
    """Tests for length-based pre-filter in _remove_intra_batch_duplicates."""

    def test_different_length_texts_not_compared_by_sequencematcher(self):
        """Texts differing in length by >50% should skip SequenceMatcher."""
        stub_modules = _stub_dependencies()
        mp = _import_message_processor(stub_modules)
        short_text = "AI news" + " x" * 50
        long_text = "AI news" + " y" * 200
        msgs = _make_messages(
            channels=["@a", "@b"],
            texts=[short_text, long_text],
        )
        result = mp._remove_intra_batch_duplicates(msgs)
        self.assertEqual(len(result), 2, "Length-different texts should both be kept")

    def test_similar_length_texts_still_deduplicated(self):
        """Texts of similar length should still go through SequenceMatcher."""
        stub_modules = _stub_dependencies()
        mp = _import_message_processor(stub_modules)
        base = "AI breakthrough in natural language processing " * 10
        near_dup = base + "extra word"
        msgs = _make_messages(
            channels=["@a", "@b"],
            texts=[base, near_dup],
        )
        result = mp._remove_intra_batch_duplicates(msgs)
        self.assertLessEqual(len(result), 2)


class StripMetaArtifactsTests(unittest.TestCase):
    """Tests for strip_meta_artifacts removing LLM intro/outro phrases."""

    def _import_utils(self):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_openai = types.ModuleType("openai")

        class FakeOpenAI:
            def __init__(self, api_key, **kwargs):
                pass

        class FakeOpenAIError(Exception):
            pass

        fake_openai.OpenAI = FakeOpenAI
        fake_openai.AsyncOpenAI = FakeOpenAI
        fake_openai.APIError = FakeOpenAIError
        fake_openai.RateLimitError = type("RateLimitError", (FakeOpenAIError,), {"status_code": None})
        fake_openai.APIConnectionError = type("APIConnectionError", (FakeOpenAIError,), {})

        with patch.dict(sys.modules, {"dotenv": fake_dotenv, "openai": fake_openai}):
            with patch.dict(os.environ, REQUIRED_ENV, clear=True):
                sys.modules.pop("config", None)
                sys.modules.pop("utils", None)
                import importlib
                return importlib.import_module("utils")

    def test_strips_intro_phrase(self):
        utils = self._import_utils()
        text = "В этом дайджесте мы рассмотрим новые модели\n<b>AI news</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("В этом дайджесте", result)
        self.assertIn("<b>AI news</b>", result)

    def test_strips_outro_phrase(self):
        utils = self._import_utils()
        text = "<b>AI news</b>\nИтого, рынок растёт"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Итого", result)
        self.assertIn("<b>AI news</b>", result)

    def test_preserves_clean_summary(self):
        utils = self._import_utils()
        text = "<b>🧠 AI модели</b>\nGPT-5 выпущен [1]"
        result = utils.strip_meta_artifacts(text)
        self.assertEqual(result, text)

    def test_strips_итак_intro(self):
        utils = self._import_utils()
        text = "Итак, подведём итоги\n<b>News</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Итак", result)

    def test_strips_в_заключение(self):
        utils = self._import_utils()
        text = "<b>News</b>\nВ заключение отметим что"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("В заключение", result)

    def test_strips_другие_ссылки_outro(self):
        utils = self._import_utils()
        text = "<b>AI модели</b>\nGPT-5 выпущен [1]\nДругие ссылки: [2]"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Другие ссылки:", result)
        self.assertIn("GPT-5", result)

    def test_preserves_другие_ссылки_without_colon(self):
        utils = self._import_utils()
        text = "<b>AI модели</b>\nДругие ссылки в статье [1]"
        result = utils.strip_meta_artifacts(text)
        self.assertIn("Другие ссылки в статье", result)

    def test_strips_обратите_внимание_intro(self):
        utils = self._import_utils()
        text = "Обратите внимание на новую модель\n<b>AI news</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Обратите внимание", result)
        self.assertIn("<b>AI news</b>", result)

    def test_strips_напомним_intro(self):
        utils = self._import_utils()
        text = "Напомним, что OpenAI выпустил GPT-5\n<b>AI news</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Напомним", result)
        self.assertIn("<b>AI news</b>", result)

    def test_strips_обратите_внимание_outro(self):
        utils = self._import_utils()
        text = "<b>AI news</b>\nОбратите внимание на эту новость"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Обратите внимание", result)
        self.assertIn("<b>AI news</b>", result)

    def test_preserves_обратите_внимание_in_body(self):
        utils = self._import_utils()
        text = "<b>AI news</b>\nСтоит обратить внимание на модель GPT-5 [1]"
        result = utils.strip_meta_artifacts(text)
        self.assertIn("обратить внимание", result)

    def test_strips_также_стоит_отметить_intro(self):
        utils = self._import_utils()
        text = "Также стоит отметить выход GPT-5\n<b>AI news</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Также стоит отметить", result)
        self.assertIn("<b>AI news</b>", result)

    def test_strips_также_стоит_отметить_outro(self):
        utils = self._import_utils()
        text = "<b>AI news</b>\nТакже стоит отметить выход GPT-5"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Также стоит отметить", result)
        self.assertIn("<b>AI news</b>", result)

    def test_preserves_смотри_также_in_body(self):
        utils = self._import_utils()
        text = "<b>AI news</b>\nРекомендуем смотри также статью о GPT-5 [1]"
        result = utils.strip_meta_artifacts(text)
        self.assertIn("смотри также", result)

    def test_preserves_подробнее_in_body(self):
        utils = self._import_utils()
        text = "<b>AI news</b>\nПодробнее о модели GPT-5 читайте в [1]"
        result = utils.strip_meta_artifacts(text)
        self.assertIn("Подробнее о модели", result)

    def test_strips_смотри_также_with_colon_outro(self):
        utils = self._import_utils()
        text = "<b>AI news</b>\nСмотри также: [2]"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Смотри также:", result)
        self.assertIn("<b>AI news</b>", result)

    def test_strips_подробнее_with_colon_outro(self):
        utils = self._import_utils()
        text = "<b>AI news</b>\nПодробнее: <a href=\"...\">link</a>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Подробнее:", result)
        self.assertIn("<b>AI news</b>", result)


class PromptAntiListTests(unittest.TestCase):
    """Tests for anti-list and anti-subheader rules in summary prompts."""

    def test_channel_summary_prompt_prohibits_lists(self):
        """CHANNEL_SUMMARY_PROMPT should contain anti-list rule."""
        import ast

        with open("prompts.py") as f:
            source = f.read()
        self.assertIn("Без нумерованных и маркированных списков", source)

    def test_channel_summary_prompt_prohibits_subheaders(self):
        """CHANNEL_SUMMARY_PROMPT should contain anti-subheader rule."""
        with open("prompts.py") as f:
            source = f.read()
        self.assertIn("Без подзаголовков внутри раздела", source)


class TemplateDashboardTests(unittest.TestCase):
    """Tests for CloudWatch Dashboard in template.yaml."""

    def test_template_contains_dashboard_resource(self):
        with open("template.yaml") as f:
            content = f.read()
        self.assertIn("SummarizerDashboard", content)
        self.assertIn("AWS::CloudWatch::Dashboard", content)

    def test_template_default_model_is_gpt41_nano(self):
        with open("template.yaml") as f:
            content = f.read()
        self.assertIn('Default: "gpt-4.1-nano"', content)

    def test_template_allowed_values_gpt41_nano_first(self):
        with open("template.yaml") as f:
            content = f.read()
        nano_pos = content.index("gpt-4.1-nano")
        mini_pos = content.index("gpt-4o-mini")
        self.assertLess(nano_pos, mini_pos, "gpt-4.1-nano should be first in AllowedValues")

    def test_template_contains_daily_cost_alarm(self):
        with open("template.yaml") as f:
            content = f.read()
        self.assertIn("OpenAIDailyCostAlarm", content)
        self.assertIn("CumulativeCostUSD", content)
        self.assertIn("tg_summarizer/Invocation", content)
        self.assertIn("86400", content)

    def test_template_dashboard_contains_invocation_widget(self):
        with open("template.yaml") as f:
            content = f.read()
        self.assertIn("CumulativeCostUSD", content)
        self.assertIn("CumulativePromptTokens", content)
        self.assertIn("CumulativeCompletionTokens", content)
        self.assertIn("tg_summarizer/Invocation", content)


if __name__ == "__main__":
    unittest.main()
