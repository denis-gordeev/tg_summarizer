import importlib
import os
import re
import sys
import types
import unittest
import html as _html
from unittest.mock import MagicMock, patch


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


def _stub_dependencies():
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda: None
    fake_openai = types.ModuleType("openai")

    class FakeOpenAIError(Exception):
        pass

    class FakeOpenAI:
        def __init__(self, api_key):
            self.api_key = api_key
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **kwargs: None)
            )

    fake_openai.OpenAI = FakeOpenAI
    fake_openai.AsyncOpenAI = FakeOpenAI
    fake_openai.APIError = FakeOpenAIError
    fake_openai.RateLimitError = type("RateLimitError", (FakeOpenAIError,), {"status_code": None})
    fake_openai.APIConnectionError = type("APIConnectionError", (FakeOpenAIError,), {})
    fake_channel_manager = types.ModuleType("channel_manager")
    fake_channel_manager.load_discovered_channels = lambda: []
    fake_channel_manager.load_similar_channels = lambda: []
    fake_channel_manager.load_banned_channels = lambda: []
    fake_channel_manager.create_channel_abbreviation = lambda channel: channel.lstrip("@")[:4]
    fake_channel_manager.get_all_source_channels = lambda: []
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
        NLP_RELEVANCE_PROMPT="nlp",
        COVERAGE_AND_MATCH_PROMPT="covmatch",
    )
    fake_utils = types.ModuleType("utils")
    fake_utils.call_openai = MagicMock()
    fake_utils.extract_links = lambda text: []
    fake_utils.count_characters = _stub_count_characters
    fake_utils.enforce_summary_length = _stub_enforce_summary_length
    fake_utils.text_hash = lambda text: __import__('hashlib').sha256(text.encode()).hexdigest()[:16]

    return {
        "dotenv": fake_dotenv,
        "openai": fake_openai,
        "channel_manager": fake_channel_manager,
        "history_manager": fake_history_manager,
        "prompts": fake_prompts,
        "utils": fake_utils,
    }


def _import_message_processor():
    sys.modules.pop("config", None)
    sys.modules.pop("utils", None)
    sys.modules.pop("message_processor", None)
    with patch.dict(os.environ, REQUIRED_ENV, clear=True), \
         patch.dict(sys.modules, _stub_dependencies()):
        import message_processor

    return message_processor


class SummaryLengthGuardrailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.message_processor = _import_message_processor()

    def test_channel_limit_has_reasonable_floor_and_cap(self):
        self.assertEqual(self.message_processor._calculate_channel_summary_limit(300), 800)
        self.assertEqual(self.message_processor._calculate_channel_summary_limit(18000), 4000)

    def test_group_limit_has_reasonable_floor_and_cap(self):
        self.assertEqual(self.message_processor._calculate_group_summary_limit(500), 2000)
        self.assertEqual(self.message_processor._calculate_group_summary_limit(20000), 12000)

    def test_enforce_summary_length_prefers_whole_blocks(self):
        summary = "<b>Block 1</b>\nКороткий текст.\n\n<b>Block 2</b>\nЕще один блок."
        limited = self.message_processor.enforce_summary_length(summary, 25)
        self.assertIn("Block 1", limited)
        self.assertNotIn("Block 2", limited)

    def test_enforce_summary_length_closes_open_html_tags(self):
        summary = "<b>Очень длинный заголовок без конца"
        limited = self.message_processor.enforce_summary_length(summary, 12)
        self.assertTrue(limited.endswith("</b>"))
        self.assertLessEqual(self.message_processor.count_characters(limited), 15)


class CountCharactersHtmlEntityTests(unittest.TestCase):
    """Tests for count_characters handling HTML entities."""

    def _import_utils(self):
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

        with patch.dict(sys.modules, {
            "dotenv": fake_dotenv,
            "openai": fake_openai,
            "config": fake_config,
        }):
            sys.modules.pop("utils", None)
            return importlib.import_module("utils")

    def test_amp_entity_counts_as_one_char(self):
        """&amp; should count as 1 visible character, not 5."""
        utils = self._import_utils()
        self.assertEqual(utils.count_characters("a &amp; b"), 5)

    def test_lt_entity_counts_as_one_char(self):
        """&lt; should count as 1 visible character, not 3."""
        utils = self._import_utils()
        self.assertEqual(utils.count_characters("a &lt; b"), 5)

    def test_gt_entity_counts_as_one_char(self):
        """&gt; should count as 1 visible character, not 3."""
        utils = self._import_utils()
        self.assertEqual(utils.count_characters("a &gt; b"), 5)

    def test_numeric_entity_counts_as_one_char(self):
        """&#39; should count as 1 visible character."""
        utils = self._import_utils()
        self.assertEqual(utils.count_characters("a&#39;b"), 3)

    def test_plain_text_unchanged(self):
        """Plain text without entities should count the same as before."""
        utils = self._import_utils()
        self.assertEqual(utils.count_characters("hello world"), 11)


class SaveJsonFileFsyncTests(unittest.TestCase):
    """Tests for save_json_file using fsync before atomic replace."""

    def test_save_json_file_writes_with_fsync(self):
        """save_json_file should flush and fsync before atomic replace."""
        import importlib
        import json
        import tempfile

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

        with patch.dict(sys.modules, {
            "dotenv": fake_dotenv,
            "openai": fake_openai,
            "config": fake_config,
        }):
            sys.modules.pop("utils", None)
            utils = importlib.import_module("utils")

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.json")
            data = {"key": "value"}
            result = utils.save_json_file(filepath, data, "test error")
            self.assertTrue(result)
            with open(filepath) as f:
                saved = json.load(f)
            self.assertEqual(saved, data)


if __name__ == "__main__":
    unittest.main()
