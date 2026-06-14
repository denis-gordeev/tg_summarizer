import os
import re
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


REQUIRED_ENV = {
    "TELEGRAM_API_ID": "1",
    "TELEGRAM_API_HASH": "hash",
    "TELEGRAM_BOT_TOKEN": "token",
    "TARGET_CHANNEL": "@target",
    "OPENAI_API_KEY": "test-key",
}


def _stub_count_characters(text):
    return len(re.sub(r'<[^>]+>', '', text))


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
    fake_utils.text_hash = lambda text: "abc123"

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


if __name__ == "__main__":
    unittest.main()
