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
    fake_utils.is_circuit_breaker_open = lambda: False
    fake_utils._emit_emf = lambda *a, **kw: None

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

    def test_strips_стоит_отметить_intro(self):
        utils = self._import_utils()
        text = "Стоит отметить выход новой модели\n<b>AI news</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Стоит отметить", result)
        self.assertIn("<b>AI news</b>", result)

    def test_strips_следует_отметить_intro(self):
        utils = self._import_utils()
        text = "Следует отметить важность модели\n<b>AI news</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Следует отметить", result)
        self.assertIn("<b>AI news</b>", result)

    def test_strips_важно_отметить_intro(self):
        utils = self._import_utils()
        text = "Важно отметить, что GPT-5 вышел\n<b>AI news</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Важно отметить", result)
        self.assertIn("<b>AI news</b>", result)

    def test_strips_подводя_итог_outro(self):
        utils = self._import_utils()
        text = "<b>AI news</b>\nПодводя итог, рынок растёт"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Подводя итог", result)
        self.assertIn("<b>AI news</b>", result)

    def test_strips_резюмируя_outro(self):
        utils = self._import_utils()
        text = "<b>AI news</b>\nРезюмируя, прогресс налицо"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Резюмируя", result)
        self.assertIn("<b>AI news</b>", result)

    def test_preserves_стоит_отметить_in_body(self):
        utils = self._import_utils()
        text = "<b>AI news</b>\nМодель GPT-5 стоит отметить среди прочих [1]"
        result = utils.strip_meta_artifacts(text)
        self.assertIn("стоит отметить", result)

    def test_strips_ключевые_выводы_intro(self):
        utils = self._import_utils()
        text = "Ключевые выводы дня\n<b>AI news</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Ключевые выводы", result)
        self.assertIn("<b>AI news</b>", result)

    def test_strips_ключевые_выводы_outro(self):
        utils = self._import_utils()
        text = "<b>AI news</b>\nКлючевые выводы: модели развиваются"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Ключевые выводы", result)
        self.assertIn("<b>AI news</b>", result)

    def test_strips_в_целом_outro(self):
        utils = self._import_utils()
        text = "<b>AI news</b>\nВ целом, рынок растёт"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("В целом", result)
        self.assertIn("<b>AI news</b>", result)

    def test_strips_в_целом_intro(self):
        utils = self._import_utils()
        text = "В целом, ситуация стабильная\n<b>AI news</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("В целом", result)
        self.assertIn("<b>AI news</b>", result)


class PromptAntiListTests(unittest.TestCase):
    """Tests for anti-list and anti-subheader rules in summary prompts."""

    def test_channel_summary_prompt_prohibits_lists(self):
        """CHANNEL_SUMMARY_PROMPT should contain anti-list rule."""
        import ast

        with open("prompts.py") as f:
            source = f.read()
        self.assertIn("Без списков — сплошной текст с абзацами", source)

    def test_channel_summary_prompt_prohibits_subheaders(self):
        """CHANNEL_SUMMARY_PROMPT should contain anti-subheader rule."""
        with open("prompts.py") as f:
            source = f.read()
        self.assertIn("Один заголовок на тему, без подзаголовков", source)


class PromptBrevityTests(unittest.TestCase):
    """Tests for brevity emphasis in summary prompts."""

    def test_channel_summary_prompt_has_brevity_rule(self):
        with open("prompts.py") as f:
            source = f.read()
        self.assertIn("вводных слов", source)

    def test_group_summary_prompt_has_brevity_rule(self):
        with open("prompts.py") as f:
            source = f.read()
        count = source.count("вводных слов")
        self.assertGreaterEqual(count, 2, "Both prompts should have brevity rule")

    def test_channel_summary_prompt_has_one_fact_per_sentence_rule(self):
        with open("prompts.py") as f:
            source = f.read()
        self.assertIn("Один факт на предложение", source)

    def test_group_summary_prompt_has_one_fact_per_sentence_rule(self):
        with open("prompts.py") as f:
            source = f.read()
        count = source.count("Один факт на предложение")
        self.assertGreaterEqual(count, 2, "Both prompts should have one-fact-per-sentence rule")


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

    def test_template_outputs_at_top_level(self):
        """Outputs section should be at YAML top level, not nested in Resources."""
        with open("template.yaml") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip() == "Outputs:":
                indent = len(line) - len(line.lstrip())
                self.assertEqual(indent, 0, "Outputs should be at top level (0 indent)")
                break
        else:
            self.fail("Outputs section not found in template.yaml")

    def test_template_contains_warmup_schedule_parameter(self):
        with open("template.yaml") as f:
            content = f.read()
        self.assertIn("WarmupScheduleExpression", content)

    def test_template_contains_warmup_event(self):
        with open("template.yaml") as f:
            content = f.read()
        self.assertIn("WarmupRun", content)
        self.assertIn("HasWarmupSchedule", content)


class NewMetaArtifactPatternsTests(unittest.TestCase):
    """Tests for newly added meta-artifact patterns (таким образом, отметим, etc.)."""

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

    def test_strips_таким_образом_outro(self):
        """'Таким образом' at start of text should be stripped."""
        utils = self._import_utils()
        text = "Таким образом, рынок растёт"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Таким образом", result)

    def test_strips_таким_образом_intro(self):
        """'Таким образом' at start of text should be stripped."""
        utils = self._import_utils()
        text = "Таким образом, всё хорошо\n<b>AI news</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Таким образом", result)

    def test_preserves_таким_образом_in_middle(self):
        """'таким образом' in the middle of body text should be preserved."""
        utils = self._import_utils()
        text = "<b>AI</b>\nМодель обрабатывает данные, и таким образом результат получается лучше"
        result = utils.strip_meta_artifacts(text)
        self.assertIn("таким образом", result.lower())

    def test_strips_отметим_intro(self):
        utils = self._import_utils()
        text = "Отметим, что модель обновлена\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Отметим", result)

    def test_strips_как_уже_упоминалось_outro(self):
        utils = self._import_utils()
        text = "<b>AI</b>\nКак уже упоминалось, это важно"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Как уже упоминалось", result)

    def test_strips_в_данном_обзоре_intro(self):
        utils = self._import_utils()
        text = "В данном обзоре мы рассмотрим\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("В данном обзоре", result)

    def test_strips_ниже_представлены_intro(self):
        utils = self._import_utils()
        text = "Ниже представлены результаты\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Ниже представлены", result)

    def test_strips_давайте_рассмотрим_intro(self):
        utils = self._import_utils()
        text = "Давайте рассмотрим новые модели\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Давайте рассмотрим", result)

    def test_strips_напоследок_outro(self):
        utils = self._import_utils()
        text = "<b>AI news</b>\nНапоследок, упомянем GPT-5"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Напоследок", result)
        self.assertIn("<b>AI news</b>", result)

    def test_strips_кратко_говоря_outro(self):
        utils = self._import_utils()
        text = "<b>AI news</b>\nКратко говоря, прогресс налицо"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Кратко говоря", result)
        self.assertIn("<b>AI news</b>", result)

    def test_strips_среди_прочего_intro(self):
        utils = self._import_utils()
        text = "Среди прочего, модель обновлена\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Среди прочего", result)
        self.assertIn("<b>AI</b>", result)

    def test_strips_во_первых_intro(self):
        utils = self._import_utils()
        text = "Во-первых, GPT-5 вышел\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Во-первых", result)
        self.assertIn("<b>AI</b>", result)

    def test_preserves_во_первых_in_body(self):
        utils = self._import_utils()
        text = "<b>AI</b>\nМодель во-первых превосходит конкурентов [1]"
        result = utils.strip_meta_artifacts(text)
        self.assertIn("во-первых", result.lower())

    def test_strips_в_конечном_итоге_intro(self):
        utils = self._import_utils()
        text = "В конечном итоге, проект завершён\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("В конечном итоге", result)
        self.assertIn("<b>AI</b>", result)

    def test_strips_в_конечном_итоге_outro(self):
        utils = self._import_utils()
        text = "<b>AI news</b>\nВ конечном итоге, прогресс налицо"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("В конечном итоге", result)
        self.assertIn("<b>AI news</b>", result)

    def test_strips_собственно_говоря_intro(self):
        utils = self._import_utils()
        text = "Собственно говоря, ничего нового\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Собственно говоря", result)
        self.assertIn("<b>AI</b>", result)

    def test_strips_к_слову_outro(self):
        utils = self._import_utils()
        text = "<b>AI news</b>\nК слову, модель обновлена"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("К слову", result)
        self.assertIn("<b>AI news</b>", result)

    def test_strips_в_первую_очередь_intro(self):
        utils = self._import_utils()
        text = "В первую очередь, стоит отметить\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("В первую очередь", result)
        self.assertIn("<b>AI</b>", result)

    def test_strips_короче_говоря_intro(self):
        utils = self._import_utils()
        text = "Короче говоря, это важно\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Короче говоря", result)
        self.assertIn("<b>AI</b>", result)

    def test_strips_подытоживая_intro(self):
        utils = self._import_utils()
        text = "Подытоживая сказанное\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Подытоживая", result)
        self.assertIn("<b>AI</b>", result)

    def test_strips_в_частности_intro(self):
        utils = self._import_utils()
        text = "В частности, модель обновлена\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("В частности", result)
        self.assertIn("<b>AI</b>", result)

    def test_strips_между_прочим_intro(self):
        utils = self._import_utils()
        text = "Между прочим, это важно\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Между прочим", result)
        self.assertIn("<b>AI</b>", result)

    def test_strips_между_прочим_outro(self):
        utils = self._import_utils()
        text = "<b>AI news</b>\nМежду прочим, модель обновлена"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Между прочим", result)
        self.assertIn("<b>AI news</b>", result)

    def test_strips_перейдём_к_intro(self):
        utils = self._import_utils()
        text = "Перейдём к результатам\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Перейдём к", result)
        self.assertIn("<b>AI</b>", result)

    def test_strips_перейдем_к_intro(self):
        utils = self._import_utils()
        text = "Перейдем к итогам\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Перейдем к", result)
        self.assertIn("<b>AI</b>", result)

    def test_strips_возвращаясь_к_intro(self):
        utils = self._import_utils()
        text = "Возвращаясь к теме моделей\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Возвращаясь к", result)
        self.assertIn("<b>AI</b>", result)

    def test_strips_как_показано_intro(self):
        utils = self._import_utils()
        text = "Как показано выше\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Как показано", result)
        self.assertIn("<b>AI</b>", result)

    def test_preserves_в_частности_in_body(self):
        utils = self._import_utils()
        text = "<b>AI</b>\nМодель превосходит конкурентов в частности благодаря архитектуре [1]"
        result = utils.strip_meta_artifacts(text)
        self.assertIn("в частности", result.lower())


class CallOpenaiUsesCircuitBreakerStateTests(unittest.TestCase):
    """Tests that call_openai uses get_circuit_breaker_state() instead of inline checks."""

    def test_call_openai_uses_get_circuit_breaker_state(self):
        with open("utils.py") as f:
            source = f.read()
        self.assertIn("cb_state = get_circuit_breaker_state()", source)

    def test_call_openai_no_inline_failure_threshold_check(self):
        with open("utils.py") as f:
            source = f.read()
        self.assertNotIn("_CIRCUIT_BREAKER_FAILURES >= CIRCUIT_BREAKER_THRESHOLD", source)
    """Tests for is_circuit_breaker_open helper."""

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

    def test_returns_false_when_closed(self):
        utils = self._import_utils()
        utils.reset_circuit_breaker()
        self.assertFalse(utils.is_circuit_breaker_open())


class NLPCircuitBreakerTests(unittest.TestCase):
    """Tests for is_nlp_related returning circuit_breaker_open reason."""

    def test_nlp_related_checks_circuit_breaker_before_llm(self):
        """is_nlp_related should check circuit breaker and return early with circuit_breaker_open reason."""
        import ast
        with open("message_processor.py") as f:
            source = f.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "is_nlp_related":
                body_source = ast.get_source_segment(source, node)
                self.assertIn("is_circuit_breaker_open", body_source,
                              "is_nlp_related should check is_circuit_breaker_open before calling call_openai")
                return
        self.fail("is_nlp_related function not found")


class LambdaEventValidationTests(unittest.TestCase):
    """Tests for Lambda event input validation."""

    @staticmethod
    def _load_lambda_handler():
        import importlib
        fake_summarizer = types.ModuleType("summarizer")
        fake_summarizer.run_summarizer = AsyncMock()
        fake_summarizer.DeadlineExceededError = type("DeadlineExceededError", (Exception,), {})
        fake_config = types.ModuleType("config")
        fake_config.validate_config = lambda: None
        fake_config.OPENAI_MODEL = "gpt-4.1-nano"
        fake_utils = types.ModuleType("utils")
        fake_utils.get_circuit_breaker_state = lambda: {"state": "closed", "failures": 0}
        fake_utils.reset_circuit_breaker = lambda: None
        fake_utils.get_token_usage = lambda: {"prompt_tokens": 0, "completion_tokens": 0}
        fake_utils.reset_token_usage = lambda: None
        fake_utils._estimate_cost_usd = lambda model, pt, ct: 0.0
        fake_utils._emit_emf = lambda *a, **kw: None
        sys.modules["summarizer"] = fake_summarizer
        sys.modules["config"] = fake_config
        sys.modules["utils"] = fake_utils
        sys.modules.pop("lambda_handler", None)
        return importlib.import_module("lambda_handler")

    def test_handler_handles_none_event(self):
        lambda_handler = self._load_lambda_handler()
        with patch.object(lambda_handler.os, "chdir"), \
             patch.object(lambda_handler, "download_from_s3"), \
             patch.object(lambda_handler, "upload_to_s3", return_value={"uploaded": 0, "failed": 0, "skipped_empty": 0}):
            result = lambda_handler.handler(None, None)
            self.assertIn("status", result)

    def test_handler_handles_string_event(self):
        lambda_handler = self._load_lambda_handler()
        with patch.object(lambda_handler.os, "chdir"), \
             patch.object(lambda_handler, "download_from_s3"), \
             patch.object(lambda_handler, "upload_to_s3", return_value={"uploaded": 0, "failed": 0, "skipped_empty": 0}):
            result = lambda_handler.handler("not a dict", None)
            self.assertIn("status", result)


class NLPPromptSimpleYesNoTests(unittest.TestCase):
    """Tests that NLP prompt asks for simple да/нет response."""

    def test_nlp_prompt_asks_only_yes_no(self):
        with open("prompts.py") as f:
            source = f.read()
        self.assertIn("Ответь только 'да' или 'нет'", source)
        self.assertNotIn("напиши \"нет, причина:\"", source)


class CostFallbackMatchesDefaultModelTests(unittest.TestCase):
    """Tests that cost fallback uses gpt-4.1-nano pricing."""

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

    def test_cost_fallback_is_nano_pricing(self):
        utils = self._import_utils()

        cost = utils._estimate_cost_usd("unknown-model", 1_000_000, 0)
        self.assertAlmostEqual(cost, 0.10, places=2,
                               msg="Fallback cost for unknown model should match gpt-4.1-nano input pricing ($0.10/M)")

        cost2 = utils._estimate_cost_usd("unknown-model", 0, 1_000_000)
        self.assertAlmostEqual(cost2, 0.40, places=2,
                               msg="Fallback cost for unknown model should match gpt-4.1-nano output pricing ($0.40/M)")


class TemplateConfigSyncTests(unittest.TestCase):
    """Tests that template.yaml includes all tunable config parameters."""

    def test_template_has_update_match_max_summaries(self):
        with open("template.yaml") as f:
            content = f.read()
        self.assertIn("UpdateMatchMaxSummaries", content)

    def test_template_has_max_channel_history_messages(self):
        with open("template.yaml") as f:
            content = f.read()
        self.assertIn("MaxChannelHistoryMessages", content)

    def test_template_has_restore_history_days(self):
        with open("template.yaml") as f:
            content = f.read()
        self.assertIn("RestoreHistoryDays", content)

    def test_template_has_group_summarization_interval(self):
        with open("template.yaml") as f:
            content = f.read()
        self.assertIn("GroupSummarizationIntervalSeconds", content)


class NewMetaArtifactTests(unittest.TestCase):
    """Tests for newly added meta-artifact patterns."""

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

    def test_strips_как_мы_видим_intro(self):
        utils = self._import_utils()
        text = "Как мы видим, модели улучшаются\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Как мы видим", result)
        self.assertIn("<b>AI</b>", result)

    def test_strips_как_можно_заметить_intro(self):
        utils = self._import_utils()
        text = "Как можно заметить, прогресс очевиден\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Как можно заметить", result)

    def test_strips_рассмотрим_intro(self):
        utils = self._import_utils()
        text = "Рассмотрим новые модели\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Рассмотрим", result)

    def test_strips_остановимся_на_intro(self):
        utils = self._import_utils()
        text = "Остановимся на архитектуре\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Остановимся на", result)

    def test_strips_исходя_из_вышесказанного_intro(self):
        utils = self._import_utils()
        text = "Исходя из вышесказанного, модели развиваются\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Исходя из вышесказанного", result)

    def test_strips_необходимо_отметить_intro(self):
        utils = self._import_utils()
        text = "Необходимо отметить рост качества\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Необходимо отметить", result)

    def test_strips_важно_подчеркнуть_intro(self):
        utils = self._import_utils()
        text = "Важно подчеркнуть результат\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Важно подчеркнуть", result)

    def test_strips_не_стоит_забывать_intro(self):
        utils = self._import_utils()
        text = "Не стоит забывать про GPU\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Не стоит забывать", result)

    def test_strips_суммируя_вышесказанное_outro(self):
        utils = self._import_utils()
        text = "<b>AI</b>\nСуммируя вышесказанное, прогресс налицо"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Суммируя вышесказанное", result)

    def test_strips_в_конечном_счёте_outro(self):
        utils = self._import_utils()
        text = "<b>AI</b>\nВ конечном счёте, это важно"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("В конечном счёте", result)

    def test_strips_в_конечном_счете_outro(self):
        utils = self._import_utils()
        text = "<b>AI</b>\nВ конечном счете, это важно"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("В конечном счете", result)

    def test_strips_очевидно_intro(self):
        utils = self._import_utils()
        text = "Очевидно, модель улучшилась\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Очевидно", result)

    def test_strips_подведём_итог_intro(self):
        utils = self._import_utils()
        text = "Подведём итог обсуждения\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Подведём итог", result)

    def test_strips_главное_colon_intro(self):
        utils = self._import_utils()
        text = "Главное: модель GPT-5 вышла\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Главное:", result)

    def test_strips_резюме_colon_intro(self):
        utils = self._import_utils()
        text = "Резюме: модель GPT-5 вышла\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Резюме:", result)

    def test_preserves_необходимо_в_body(self):
        utils = self._import_utils()
        text = "<b>AI</b>\nДля обучения необходимо 100 GPU"
        result = utils.strip_meta_artifacts(text)
        self.assertIn("необходимо", result.lower())

    def test_preserves_рассмотрим_в_body(self):
        utils = self._import_utils()
        text = "<b>AI</b>\nМодель, которую рассмотрим ниже — GPT-5"
        result = utils.strip_meta_artifacts(text)
        self.assertIn("рассмотрим", result.lower())

    def test_strips_проще_говоря_intro(self):
        utils = self._import_utils()
        text = "Проще говоря, модель работает лучше\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Проще говоря", result)
        self.assertIn("<b>AI</b>", result)

    def test_strips_проще_говоря_outro(self):
        utils = self._import_utils()
        text = "<b>AI</b>\nПроще говоря, это улучшение"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Проще говоря", result)

    def test_strips_иными_словами_intro(self):
        utils = self._import_utils()
        text = "Иными словами, архитектура изменилась\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Иными словами", result)

    def test_strips_иными_словами_outro(self):
        utils = self._import_utils()
        text = "<b>AI</b>\nИными словами, результат тот же"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Иными словами", result)

    def test_strips_иначе_говоря_intro(self):
        utils = self._import_utils()
        text = "Иначе говоря, подход другой\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Иначе говоря", result)

    def test_strips_кстати_intro(self):
        utils = self._import_utils()
        text = "Кстати, вышла новая модель\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Кстати", result)
        self.assertIn("<b>AI</b>", result)

    def test_strips_кстати_outro(self):
        utils = self._import_utils()
        text = "<b>AI</b>\nКстати, модель обновлена"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Кстати", result)

    def test_strips_по_сути_intro(self):
        utils = self._import_utils()
        text = "По сути, это тот же подход\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("По сути", result)
        self.assertIn("<b>AI</b>", result)

    def test_preserves_по_сути_in_body(self):
        utils = self._import_utils()
        text = "<b>AI</b>\nМодель по сути использует attention [1]"
        result = utils.strip_meta_artifacts(text)
        self.assertIn("по сути", result.lower())

    def test_strips_впрочем_intro(self):
        utils = self._import_utils()
        text = "Впрочем, это не важно\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Впрочем", result)

    def test_strips_помимо_прочего_intro(self):
        utils = self._import_utils()
        text = "Помимо прочего, вышла GPT-5\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Помимо прочего", result)
        self.assertIn("<b>AI</b>", result)

    def test_strips_кроме_того_intro(self):
        utils = self._import_utils()
        text = "Кроме того, модель обновлена\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Кроме того", result)
        self.assertIn("<b>AI</b>", result)

    def test_strips_кроме_того_outro(self):
        utils = self._import_utils()
        text = "<b>AI</b>\nКроме того, модель обновлена"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Кроме того", result)

    def test_strips_более_того_intro(self):
        utils = self._import_utils()
        text = "Более того, результаты улучшились\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Более того", result)
        self.assertIn("<b>AI</b>", result)

    def test_strips_более_того_outro(self):
        utils = self._import_utils()
        text = "<b>AI</b>\nБолее того, результаты улучшились"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Более того", result)

    def test_strips_помимо_этого_intro(self):
        utils = self._import_utils()
        text = "Помимо этого, вышла GPT-5\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Помимо этого", result)
        self.assertIn("<b>AI</b>", result)

    def test_strips_что_касается_intro(self):
        utils = self._import_utils()
        text = "Что касается архитектуры, она изменилась\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Что касается", result)
        self.assertIn("<b>AI</b>", result)

    def test_strips_переходя_к_intro(self):
        utils = self._import_utils()
        text = "Переходя к результатам, модель лучше\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Переходя к", result)
        self.assertIn("<b>AI</b>", result)

    def test_strips_в_связи_с_этим_intro(self):
        utils = self._import_utils()
        text = "В связи с этим, модель обновлена\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("В связи с этим", result)
        self.assertIn("<b>AI</b>", result)

    def test_strips_что_важно_intro(self):
        utils = self._import_utils()
        text = "Что важно, модель быстрее\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Что важно", result)
        self.assertIn("<b>AI</b>", result)

    def test_strips_стоит_понимать_intro(self):
        utils = self._import_utils()
        text = "Стоит понимать, что архитектура другая\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Стоит понимать", result)
        self.assertIn("<b>AI</b>", result)

    def test_strips_как_известно_intro(self):
        utils = self._import_utils()
        text = "Как известно, GPT-5 вышел\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Как известно", result)
        self.assertIn("<b>AI</b>", result)

    def test_strips_естественно_intro(self):
        utils = self._import_utils()
        text = "Естественно, результаты улучшились\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Естественно", result)
        self.assertIn("<b>AI</b>", result)

    def test_strips_разумеется_outro(self):
        utils = self._import_utils()
        text = "<b>AI</b>\nРазумеется, модель лучше"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Разумеется", result)

    def test_strips_само_собой_outro(self):
        utils = self._import_utils()
        text = "<b>AI</b>\nСамо собой, результат ожидаемый"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Само собой", result)

    def test_strips_понятно_outro(self):
        utils = self._import_utils()
        text = "<b>AI</b>\nПонятно, что модель лучше"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Понятно", result)

    def test_preserves_естественно_in_body(self):
        utils = self._import_utils()
        text = "<b>AI</b>\nМодель естественно обрабатывает текст"
        result = utils.strip_meta_artifacts(text)
        self.assertIn("естественно", result.lower())

    def test_preserves_как_известно_in_body(self):
        utils = self._import_utils()
        text = "<b>AI</b>\nМодели как известно давно существуют"
        result = utils.strip_meta_artifacts(text)
        self.assertIn("как известно", result.lower())

    def test_strips_одним_словом_intro(self):
        utils = self._import_utils()
        text = "Одним словом, модель обновлена\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Одним словом", result)

    def test_strips_одним_словом_outro(self):
        utils = self._import_utils()
        text = "<b>AI</b>\nОдним словом, всё понятно"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Одним словом", result)

    def test_strips_в_общем_и_целом_intro(self):
        utils = self._import_utils()
        text = "В общем и целом, прогресс есть\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("В общем и целом", result)

    def test_strips_в_общем_и_целом_outro(self):
        utils = self._import_utils()
        text = "<b>AI</b>\nВ общем и целом, результат положительный"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("В общем и целом", result)

    def test_strips_к_слову_сказать_intro(self):
        utils = self._import_utils()
        text = "К слову сказать, модель лучше\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("К слову сказать", result)

    def test_strips_к_слову_сказать_outro(self):
        utils = self._import_utils()
        text = "<b>AI</b>\nК слову сказать, это важно"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("К слову сказать", result)

    def test_strips_строго_говоря_intro(self):
        utils = self._import_utils()
        text = "Строго говоря, это не совсем так\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Строго говоря", result)

    def test_strips_честно_говоря_intro(self):
        utils = self._import_utils()
        text = "Честно говоря, модель не справилась\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Честно говоря", result)

    def test_strips_откровенно_говоря_intro(self):
        utils = self._import_utils()
        text = "Откровенно говоря, результат слабый\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("Откровенно говоря", result)

    def test_strips_по_существу_intro(self):
        utils = self._import_utils()
        text = "По существу, алгоритм работает\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("По существу", result)

    def test_strips_в_сущности_intro(self):
        utils = self._import_utils()
        text = "В сущности, метод не нов\n<b>AI</b>"
        result = utils.strip_meta_artifacts(text)
        self.assertNotIn("В сущности", result)

    def test_preserves_строго_говоря_in_body(self):
        utils = self._import_utils()
        text = "<b>AI</b>\nМодель строго говоря не является новой"
        result = utils.strip_meta_artifacts(text)
        self.assertIn("строго говоря", result.lower())

    def test_preserves_честно_говоря_in_body(self):
        utils = self._import_utils()
        text = "<b>AI</b>\nМодель честно говоря не справилась с задачей"
        result = utils.strip_meta_artifacts(text)
        self.assertIn("честно говоря", result.lower())

    def test_preserves_по_существу_in_body(self):
        utils = self._import_utils()
        text = "<b>AI</b>\nМетод по существу делает следующее"
        result = utils.strip_meta_artifacts(text)
        self.assertIn("по существу", result.lower())

    def test_preserves_в_сущности_in_body(self):
        utils = self._import_utils()
        text = "<b>AI</b>\nПодход в сущности эквивалентен"
        result = utils.strip_meta_artifacts(text)
        self.assertIn("в сущности", result.lower())


class PromptCompactnessTests(unittest.TestCase):
    """Tests that prompt rules are merged and compact."""

    def test_channel_prompt_has_merged_intro_rule(self):
        with open("prompts.py") as f:
            source = f.read()
        self.assertIn("Без введения, заключения, мета-комментариев и вводных слов", source)
        self.assertNotIn("Без введения, заключения и мета-комментариев", source)

    def test_channel_prompt_has_merged_fact_rule(self):
        with open("prompts.py") as f:
            source = f.read()
        self.assertIn("Один факт на предложение, одна мысль на абзац", source)
        self.assertNotIn("Одно предложение на факт, без пояснений", source)

    def test_channel_prompt_has_merged_style_rule(self):
        with open("prompts.py") as f:
            source = f.read()
        self.assertIn("Конкретные факты и результаты, нейтральный стиль без мнений", source)
        self.assertNotIn("Конкретные результаты и факты, не общие описания", source)

    def test_group_prompt_has_merged_intro_rule(self):
        with open("prompts.py") as f:
            source = f.read()
        self.assertIn("Без введения, заключения, мета-комментариев и вводных слов", source)

    def test_group_prompt_still_has_question_rule(self):
        with open("prompts.py") as f:
            source = f.read()
        self.assertIn("Не добавляй вопросы пользователей", source)

    def test_prompts_no_leading_newline(self):
        with open("prompts.py") as f:
            source = f.read()
        self.assertNotIn('"COVERAGE_AND_MATCH_PROMPT": """\n', source)
        self.assertNotIn('"NLP_RELEVANCE_PROMPT": """\n', source)
        self.assertNotIn('"CHANNEL_SUMMARY_PROMPT": """\n', source)
        self.assertNotIn('"GROUP_SUMMARY_PROMPT": """\n', source)

    def test_channel_prompt_has_compact_structure_instruction(self):
        with open("prompts.py") as f:
            source = f.read()
        self.assertIn("Группируй по темам с [1], [2]", source)
        self.assertNotIn("Структура: группируй по темам, указывай", source)

    def test_channel_prompt_has_compact_list_rule(self):
        with open("prompts.py") as f:
            source = f.read()
        self.assertIn("Без списков — сплошной текст с абзацами", source)
        self.assertNotIn("Без нумерованных и маркированных списков", source)

    def test_channel_prompt_has_compact_subheading_rule(self):
        with open("prompts.py") as f:
            source = f.read()
        self.assertIn("Один заголовок на тему, без подзаголовков", source)
        self.assertNotIn("Без подзаголовков внутри раздела", source)

    def test_channel_prompt_has_compact_html_rule(self):
        with open("prompts.py") as f:
            source = f.read()
        self.assertIn("HTML: <b>текст</b>, не Markdown", source)
        self.assertNotIn("Только HTML:", source)

    def test_channel_prompt_has_compact_source_rule(self):
        with open("prompts.py") as f:
            source = f.read()
        self.assertIn("Источники: [1], [2], без своих ссылок", source)
        self.assertNotIn("Только номера источников", source)

    def test_channel_prompt_has_compact_length_rule(self):
        with open("prompts.py") as f:
            source = f.read()
        self.assertIn("≤{max_summary_length} символов", source)
        self.assertNotIn("Максимум {max_summary_length} символов", source)

    def test_group_prompt_has_compact_structure_instruction(self):
        with open("prompts.py") as f:
            source = f.read()
        self.assertIn("Группируй по темам с [1], [2]", source)

    def test_update_prompt_is_compact_merged(self):
        with open("prompts.py") as f:
            source = f.read()
        self.assertIn("Вставь {new_link} в подходящее место саммари рядом с релевантным абзацем", source)
        self.assertNotIn("Скопируй саммари, вставь ссылку", source)

    def test_update_prompt_still_has_html_preservation(self):
        with open("prompts.py") as f:
            source = f.read()
        self.assertIn("Сохрани HTML-форматирование", source)

    def test_update_prompt_still_has_fallback_instruction(self):
        with open("prompts.py") as f:
            source = f.read()
        self.assertIn("Доп. источники:", source)


class EmitEmfDebugLogTests(unittest.TestCase):
    """Tests that _emit_emf logs debug on exception."""

    def test_emit_emf_has_debug_log_on_exception(self):
        with open("utils.py") as f:
            source = f.read()
        self.assertIn("logger.debug", source)
        self.assertIn("_emit_emf failed", source)


class CircuitBreakerAlarmTests(unittest.TestCase):
    """Tests for circuit breaker half-open alarm and DLQ alarm in template."""

    def test_template_contains_circuit_breaker_alarm(self):
        with open("template.yaml") as f:
            content = f.read()
        self.assertIn("CircuitBreakerHalfOpenAlarm", content)
        self.assertIn("CircuitBreakerState", content)

    def test_circuit_breaker_alarm_threshold_is_1(self):
        with open("template.yaml") as f:
            content = f.read()
        alarm_section = content[content.index("CircuitBreakerHalfOpenAlarm"):]
        self.assertIn("Threshold: 1", alarm_section[:500])

    def test_template_contains_dlq_depth_alarm(self):
        with open("template.yaml") as f:
            content = f.read()
        self.assertIn("DLQDepthAlarm", content)
        self.assertIn("ApproximateNumberOfMessagesVisible", content)

    def test_template_dashboard_contains_circuit_breaker_widget(self):
        with open("template.yaml") as f:
            content = f.read()
        self.assertIn("Circuit Breaker State", content)


class NlpCheckCostAlarmTests(unittest.TestCase):
    """Tests for NLP check cost alarm in template."""

    def test_template_contains_nlp_check_cost_alarm(self):
        with open("template.yaml") as f:
            content = f.read()
        self.assertIn("NlpCheckCostAlarm", content)

    def test_nlp_check_cost_alarm_uses_estimated_cost_metric(self):
        with open("template.yaml") as f:
            content = f.read()
        alarm_section = content[content.index("NlpCheckCostAlarm"):]
        self.assertIn("EstimatedCostUSD", alarm_section[:500])

    def test_nlp_check_cost_alarm_has_call_type_nlp_dimension(self):
        with open("template.yaml") as f:
            content = f.read()
        alarm_section = content[content.index("NlpCheckCostAlarm"):]
        self.assertIn("CallType", alarm_section[:1000])
        self.assertIn("Value: nlp", alarm_section[:1000])

    def test_nlp_check_cost_alarm_period_is_86400(self):
        with open("template.yaml") as f:
            content = f.read()
        alarm_section = content[content.index("NlpCheckCostAlarm"):]
        self.assertIn("Period: 86400", alarm_section[:1000])


class SummaryCostAlarmTests(unittest.TestCase):
    """Tests for summary generation cost alarm in template."""

    def test_template_contains_summary_cost_alarm(self):
        with open("template.yaml") as f:
            content = f.read()
        self.assertIn("SummaryCostAlarm", content)

    def test_summary_cost_alarm_uses_estimated_cost_metric(self):
        with open("template.yaml") as f:
            content = f.read()
        alarm_section = content[content.index("SummaryCostAlarm"):]
        self.assertIn("EstimatedCostUSD", alarm_section[:2000])

    def test_summary_cost_alarm_includes_channel_summary(self):
        with open("template.yaml") as f:
            content = f.read()
        alarm_section = content[content.index("SummaryCostAlarm"):]
        self.assertIn("channel_summary", alarm_section[:2000])

    def test_summary_cost_alarm_includes_group_summary(self):
        with open("template.yaml") as f:
            content = f.read()
        alarm_section = content[content.index("SummaryCostAlarm"):]
        self.assertIn("group_summary", alarm_section[:2000])

    def test_summary_cost_alarm_uses_metric_math(self):
        with open("template.yaml") as f:
            content = f.read()
        alarm_section = content[content.index("SummaryCostAlarm"):]
        self.assertIn("m1 + m2", alarm_section[:2000])

    def test_summary_cost_alarm_period_is_86400(self):
        with open("template.yaml") as f:
            content = f.read()
        alarm_section = content[content.index("SummaryCostAlarm"):]
        self.assertIn("Period: 86400", alarm_section[:2000])


class UpdateCostAlarmTests(unittest.TestCase):
    """Tests for summary update cost alarm in template."""

    def test_template_contains_update_cost_alarm(self):
        with open("template.yaml") as f:
            content = f.read()
        self.assertIn("UpdateCostAlarm", content)

    def test_update_cost_alarm_uses_estimated_cost_metric(self):
        with open("template.yaml") as f:
            content = f.read()
        alarm_section = content[content.index("UpdateCostAlarm"):]
        self.assertIn("EstimatedCostUSD", alarm_section[:500])

    def test_update_cost_alarm_has_call_type_update_dimension(self):
        with open("template.yaml") as f:
            content = f.read()
        alarm_section = content[content.index("UpdateCostAlarm"):]
        self.assertIn("CallType", alarm_section[:1000])
        self.assertIn("Value: update", alarm_section[:1000])

    def test_update_cost_alarm_period_is_86400(self):
        with open("template.yaml") as f:
            content = f.read()
        alarm_section = content[content.index("UpdateCostAlarm"):]
        self.assertIn("Period: 86400", alarm_section[:1000])


class CoveragePromptCompactTests(unittest.TestCase):
    """Tests for compact COVERAGE_AND_MATCH_PROMPT."""

    def test_coverage_prompt_is_compact(self):
        with open("prompts.py") as f:
            source = f.read()
        self.assertIn("Та же тема → номер", source)
        self.assertNotIn("Та же тема → номер дайджеста", source)

    def test_coverage_prompt_no_verbose_bullets(self):
        with open("prompts.py") as f:
            source = f.read()
        prompt_section = source[source.index("COVERAGE_AND_MATCH_PROMPT"):]
        self.assertNotIn("Новая тема или существенные новые детали", prompt_section)

    def test_coverage_prompt_has_model_rule(self):
        with open("prompts.py") as f:
            source = f.read()
        self.assertIn("разные модели = разные темы", source)

    def test_coverage_prompt_still_has_response_instruction(self):
        with open("prompts.py") as f:
            source = f.read()
        self.assertIn('Ответь: номер или "НЕТ"', source)


class RestoreNoneDateGuardTests(unittest.TestCase):
    """Tests for None-date guard in _restore_summaries_from_channel."""

    def test_restore_guards_none_date(self):
        with open("history_manager.py") as f:
            source = f.read()
        self.assertIn("msg.date is None", source)


if __name__ == "__main__":
    unittest.main()
