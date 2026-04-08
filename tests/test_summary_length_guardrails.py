import os
import sys
import types
import unittest
from unittest.mock import patch


REQUIRED_ENV = {
    "TELEGRAM_API_ID": "1",
    "TELEGRAM_API_HASH": "hash",
    "TELEGRAM_BOT_TOKEN": "token",
    "TARGET_CHANNEL": "@target",
    "OPENAI_API_KEY": "test-key",
}


def _stub_dependencies():
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
    fake_channel_manager.create_channel_abbreviation = lambda channel: channel.lstrip("@")[:4]
    fake_history_manager = types.ModuleType("history_manager")
    fake_history_manager.get_recent_summaries_context = lambda: ""
    fake_history_manager.get_recent_group_summaries_context = lambda: ""
    fake_prompts = types.ModuleType("prompts")
    fake_prompts.prompts = types.SimpleNamespace(
        CHANNEL_SUMMARY_PROMPT="{max_summary_length}",
        GROUP_SUMMARY_PROMPT="{max_summary_length}",
    )
    return {
        "dotenv": fake_dotenv,
        "openai": fake_openai,
        "channel_manager": fake_channel_manager,
        "history_manager": fake_history_manager,
        "prompts": fake_prompts,
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
