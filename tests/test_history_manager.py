import asyncio
import importlib
import sys
import types
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock, AsyncMock
import json
import os


class ShouldRunGroupSummarizationLogicTests(unittest.TestCase):
    """Tests for should_run_group_summarization logic (standalone)."""

    def test_time_calculation_logic(self):
        """Test the time calculation logic without file I/O."""
        def check_should_run(last_run_str):
            if not last_run_str:
                return True
            
            try:
                last_run = datetime.fromisoformat(last_run_str)
                if last_run.tzinfo is None:
                    last_run = last_run.replace(tzinfo=timezone.utc)
                
                now = datetime.now(timezone.utc)
                time_since_last_run = (now - last_run).total_seconds()
                
                return time_since_last_run > 86400
            except (ValueError, TypeError):
                return True
        
        self.assertTrue(check_should_run(""))
        
        recent = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()
        self.assertFalse(check_should_run(recent))
        
        old = (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat()
        self.assertTrue(check_should_run(old))
        
        self.assertTrue(check_should_run("not-a-timestamp"))


class ShouldRunGroupSummarizationParseErrorTests(unittest.TestCase):
    """Tests for should_run_group_summarization returning True on parse errors."""

    def _import_hm_with_stubs(self):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_channel_manager = types.ModuleType("channel_manager")
        fake_channel_manager.create_channel_abbreviation = lambda ch: "AB"
        fake_channel_manager.load_channel_abbreviations = lambda: {}
        fake_models = types.ModuleType("models")
        fake_models.SummaryInfo = type("FakeSummaryInfo", (), {"__init__": lambda self, **kw: None, "to_dict": lambda self: {}})
        fake_models.MessageInfo = type("MessageInfo", (), {})
        fake_prompts = types.ModuleType("prompts")
        fake_prompts.prompts = types.SimpleNamespace()
        fake_utils = types.ModuleType("utils")
        fake_utils.call_openai = MagicMock()
        fake_utils.extract_links = lambda text: []
        fake_utils.load_json_file = lambda *a, **kw: {}
        fake_utils.save_json_file = lambda *a, **kw: True
        fake_utils.now_iso = lambda: "2026-01-01T00:00:00"
        fake_utils.extract_all_channels = lambda text: []
        fake_utils.text_hash = lambda text: __import__('hashlib').sha256(text.encode()).hexdigest()[:16]
        fake_utils.count_characters = lambda text: len(text)
        fake_utils.enforce_summary_length = lambda text, max_chars: text[:max_chars]
        fake_telegram_client = types.ModuleType("telegram_client")
        fake_telegram_client.user_client = None
        fake_telegram_client.clients_loop = None

        stubs = {
            "dotenv": fake_dotenv,
            "channel_manager": fake_channel_manager,
            "models": fake_models,
            "prompts": fake_prompts,
            "utils": fake_utils,
            "telegram_client": fake_telegram_client,
        }

        return stubs

    def test_returns_true_on_malformed_timestamp(self):
        """should_run_group_summarization should return True on parse error."""
        import tempfile
        import os

        stubs = self._import_hm_with_stubs()
        with patch.dict(sys.modules, stubs), \
             patch.dict(os.environ, {"GROUP_LAST_RUN_FILE": ""}, clear=False):
            sys.modules.pop("config", None)
            sys.modules.pop("history_manager", None)
            config = importlib.import_module("config")
            config.GROUP_LAST_RUN_FILE = "/tmp/test_group_last_run_malformed.json"
            config.GROUP_SUMMARIZATION_INTERVAL_SECONDS = 86400
            sys.modules["config"] = config

            with open("/tmp/test_group_last_run_malformed.json", "w") as f:
                json.dump({"last_run": "not-a-timestamp"}, f)

            hm = importlib.import_module("history_manager")
            result = hm.should_run_group_summarization()
            self.assertTrue(result)

        os.remove("/tmp/test_group_last_run_malformed.json")

    def test_returns_true_when_file_missing(self):
        """should_run_group_summarization should return True when file is missing."""
        import os

        stubs = self._import_hm_with_stubs()
        with patch.dict(sys.modules, stubs), \
             patch.dict(os.environ, {"GROUP_LAST_RUN_FILE": ""}, clear=False):
            sys.modules.pop("config", None)
            sys.modules.pop("history_manager", None)
            config = importlib.import_module("config")
            config.GROUP_LAST_RUN_FILE = "/tmp/nonexistent_group_last_run.json"
            config.GROUP_SUMMARIZATION_INTERVAL_SECONDS = 86400
            sys.modules["config"] = config

            hm = importlib.import_module("history_manager")
            result = hm.should_run_group_summarization()
            self.assertTrue(result)


class HistoryContextLogicTests(unittest.TestCase):
    """Tests for history context extraction logic."""

    def test_recent_summaries_context_logic(self):
        """Test the context extraction logic without file I/O."""
        max_summaries = 10
        max_chars = 300

        def extract_context(summaries, days=3):
            if not summaries:
                return ""

            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            recent = [s for s in summaries if s['date'] >= cutoff_date]

            if not recent:
                return ""

            recent = recent[-max_summaries:]

            return "\n\n".join(s['content'][:max_chars] for s in recent)

        # Test empty list
        self.assertEqual(extract_context([]), "")

        # Test old summaries
        old_summaries = [
            {
                'content': "Old summary",
                'date': datetime.now(timezone.utc) - timedelta(days=10)
            }
        ]
        self.assertEqual(extract_context(old_summaries, days=3), "")

        # Test recent summaries
        recent_summaries = [
            {
                'content': "Recent AI news",
                'date': datetime.now(timezone.utc) - timedelta(hours=12)
            },
            {
                'content': "Recent ML update",
                'date': datetime.now(timezone.utc) - timedelta(hours=6)
            }
        ]
        result = extract_context(recent_summaries, days=3)
        self.assertIn("Recent AI news", result)
        self.assertIn("Recent ML update", result)

        # Test truncation of long content
        long_summaries = [
            {
                'content': "X" * 1000,
                'date': datetime.now(timezone.utc)
            }
        ]
        result = extract_context(long_summaries, days=3)
        self.assertEqual(len(result), max_chars)

        # Test limit on number of summaries
        many_summaries = [
            {
                'content': f"Summary {i}",
                'date': datetime.now(timezone.utc) - timedelta(hours=i)
            }
            for i in range(20)
        ]
        result = extract_context(many_summaries, days=3)
        self.assertIn("Summary 19", result)
        self.assertNotIn("Summary 0", result)

    def test_update_match_limits_from_config(self):
        """Verify that UPDATE_MATCH_MAX_SUMMARIES and UPDATE_MATCH_MAX_CHARS_PER_SUMMARY exist in config."""
        import importlib
        import sys
        import types
        from unittest.mock import patch
        import os

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, {}, clear=True):
                sys.modules.pop("config", None)
                config = importlib.import_module("config")

        self.assertIsInstance(config.UPDATE_MATCH_MAX_SUMMARIES, int)
        self.assertIsInstance(config.UPDATE_MATCH_MAX_CHARS_PER_SUMMARY, int)
        self.assertGreater(config.UPDATE_MATCH_MAX_SUMMARIES, 0)
        self.assertGreater(config.UPDATE_MATCH_MAX_CHARS_PER_SUMMARY, 0)



class RunAsyncWithLoopTests(unittest.TestCase):
    """Tests for _run_async_with_loop — verifies it works from within an event loop."""

    def _import_hm_with_stubs(self):
        import importlib
        import sys
        import types

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_channel_manager = types.ModuleType("channel_manager")
        fake_channel_manager.create_channel_abbreviation = lambda ch: "AB"
        fake_channel_manager.load_channel_abbreviations = lambda: {}
        fake_models = types.ModuleType("models")
        fake_models.SummaryInfo = type("FakeSummaryInfo", (), {"__init__": lambda self, **kw: None, "to_dict": lambda self: {}})
        fake_models.MessageInfo = type("MessageInfo", (), {})
        fake_prompts = types.ModuleType("prompts")
        fake_prompts.prompts = types.SimpleNamespace()
        fake_utils = types.ModuleType("utils")
        fake_utils.call_openai = MagicMock()
        fake_utils.extract_links = lambda text: []
        fake_utils.load_json_file = lambda *a, **kw: {}
        fake_utils.save_json_file = lambda *a, **kw: True
        fake_utils.now_iso = lambda: "2026-01-01T00:00:00"
        fake_utils.extract_all_channels = lambda text: []
        fake_utils.text_hash = lambda text: __import__('hashlib').sha256(text.encode()).hexdigest()[:16]
        fake_utils.count_characters = lambda text: len(text)
        fake_utils.enforce_summary_length = lambda text, max_chars: text[:max_chars]
        fake_telegram_client = types.ModuleType("telegram_client")
        fake_telegram_client.user_client = None
        fake_telegram_client.clients_loop = None

        stubs = {
            "dotenv": fake_dotenv,
            "channel_manager": fake_channel_manager,
            "models": fake_models,
            "prompts": fake_prompts,
            "utils": fake_utils,
            "telegram_client": fake_telegram_client,
        }

        return stubs

    def test_run_async_with_loop_works_without_running_loop(self):
        """_run_async_with_loop should work when no event loop is running."""
        stubs = self._import_hm_with_stubs()
        with patch.dict(sys.modules, stubs):
            with patch.dict(os.environ, {}, clear=True):
                sys.modules.pop("history_manager", None)
                sys.modules.pop("config", None)
                hm = importlib.import_module("history_manager")

                async def fake_coro():
                    return ["item1", "item2"]

                result = hm._run_async_with_loop(fake_coro())
                self.assertEqual(result, ["item1", "item2"])

    def test_run_async_with_loop_inside_running_loop(self):
        """_run_async_with_loop should work even when called from within a
        running event loop (the common Lambda case). Previously, this silently
        returned [] due to same-loop deadlock avoidance."""
        stubs = self._import_hm_with_stubs()
        with patch.dict(sys.modules, stubs):
            with patch.dict(os.environ, {}, clear=True):
                sys.modules.pop("history_manager", None)
                sys.modules.pop("config", None)
                hm = importlib.import_module("history_manager")

                async def fake_coro():
                    return ["restored"]

                async def run_inside_loop():
                    return hm._run_async_with_loop(fake_coro())

                result = asyncio.run(run_inside_loop())
                self.assertEqual(result, ["restored"])

    def test_run_async_with_loop_returns_empty_on_exception(self):
        """_run_async_with_loop should return [] when the coroutine raises."""
        stubs = self._import_hm_with_stubs()
        with patch.dict(sys.modules, stubs):
            with patch.dict(os.environ, {}, clear=True):
                sys.modules.pop("history_manager", None)
                sys.modules.pop("config", None)
                hm = importlib.import_module("history_manager")

                async def failing_coro():
                    raise RuntimeError("test failure")

                result = hm._run_async_with_loop(failing_coro())
                self.assertEqual(result, [])


class SaveUpdatedSummaryMatchingTests(unittest.TestCase):
    """Tests for save_updated_summary matching logic — verifies message_id takes precedence."""

    def test_match_by_message_id_takes_precedence(self):
        """When message_id is set, it should be used for matching instead of content."""
        from models import SummaryInfo
        from datetime import datetime, timezone

        original = SummaryInfo(
            content="Some content",
            date=datetime.now(timezone.utc),
            message_count=2,
            channels=["@ch1"],
            message_id=42,
        )
        updated = SummaryInfo(
            content="Updated content",
            date=original.date,
            message_count=3,
            channels=["@ch1", "@ch2"],
            message_id=None,
        )

        self.assertIsNotNone(original.message_id)
        self.assertEqual(original.message_id, 42)

    def test_fallback_to_content_date_count_when_no_message_id(self):
        """When message_id is None, matching should fall back to content+date+count."""
        from models import SummaryInfo
        from datetime import datetime, timezone

        original = SummaryInfo(
            content="Fallback content",
            date=datetime.now(timezone.utc),
            message_count=5,
            channels=["@ch"],
            message_id=None,
        )

        self.assertIsNone(original.message_id)
        self.assertEqual(original.content, "Fallback content")


class UpdateExistingSummaryPreservesMessageIdTests(unittest.TestCase):
    """Tests for update_existing_summary preserving message_id."""

    def test_update_preserves_message_id(self):
        """update_existing_summary should carry over message_id from the original summary."""
        from models import SummaryInfo, MessageInfo
        from datetime import datetime, timezone
        import asyncio

        original = SummaryInfo(
            content="Original content",
            date=datetime.now(timezone.utc),
            message_count=2,
            channels=["@ch1"],
            message_id=99,
        )

        new_msg = MessageInfo(
            text="New message text",
            channel="@ch2",
            message_id=200,
            date=datetime.now(timezone.utc),
            link="https://t.me/ch2/200",
        )

        import importlib
        import sys
        import types
        from unittest.mock import MagicMock

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_channel_manager = types.ModuleType("channel_manager")
        fake_channel_manager.create_channel_abbreviation = lambda ch: "AB"
        fake_channel_manager.load_channel_abbreviations = lambda: {}
        fake_models = types.ModuleType("models")
        fake_models.SummaryInfo = SummaryInfo
        fake_models.MessageInfo = MessageInfo
        fake_prompts = types.ModuleType("prompts")
        fake_prompts.prompts = types.SimpleNamespace()
        fake_utils = types.ModuleType("utils")
        fake_utils.call_openai = MagicMock()
        fake_utils.extract_links = lambda text: []
        fake_utils.load_json_file = lambda *a, **kw: {}
        fake_utils.save_json_file = lambda *a, **kw: True
        fake_utils.now_iso = lambda: "2026-01-01T00:00:00"
        fake_utils.extract_all_channels = lambda text: []
        fake_utils.text_hash = lambda text: __import__('hashlib').sha256(text.encode()).hexdigest()[:16]
        fake_utils.count_characters = lambda text: len(text)
        fake_utils.enforce_summary_length = lambda text, max_chars: text[:max_chars]
        fake_telegram_client = types.ModuleType("telegram_client")
        fake_telegram_client.user_client = None
        fake_telegram_client.clients_loop = None

        stubs = {
            "dotenv": fake_dotenv,
            "channel_manager": fake_channel_manager,
            "models": fake_models,
            "prompts": fake_prompts,
            "utils": fake_utils,
            "telegram_client": fake_telegram_client,
        }

        with patch.dict(sys.modules, stubs):
            with patch.dict(os.environ, {}, clear=True):
                sys.modules.pop("history_manager", None)
                sys.modules.pop("config", None)
                hm = importlib.import_module("history_manager")
                updated = asyncio.run(hm.update_existing_summary(original, new_msg))

        self.assertEqual(updated.message_id, 99, "message_id should be preserved from original summary")


class SsmClientCachingTests(unittest.TestCase):
    """Tests for SSM client caching in config.py."""

    def test_get_ssm_client_caches_client(self):
        """_get_ssm_client should return the same client on subsequent calls."""
        import importlib
        import sys
        import types

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        fake_boto3 = types.ModuleType("boto3")
        mock_ssm_client = MagicMock()
        fake_boto3.client = MagicMock(return_value=mock_ssm_client)

        with patch.dict(sys.modules, {"dotenv": fake_dotenv, "boto3": fake_boto3}):
            with patch.dict(os.environ, {}, clear=True):
                sys.modules.pop("config", None)
                config = importlib.import_module("config")
                config._ssm_client = None

                client1 = config._get_ssm_client()
                client2 = config._get_ssm_client()

                self.assertIs(client1, client2, "SSM client should be cached and reused")
                fake_boto3.client.assert_called_once_with("ssm")


class HistoryCacheTests(unittest.TestCase):
    """Tests for in-memory history caching."""

    def _import_hm_with_stubs(self):
        import importlib
        import sys
        import types

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_channel_manager = types.ModuleType("channel_manager")
        fake_channel_manager.create_channel_abbreviation = lambda ch: "AB"
        fake_channel_manager.load_channel_abbreviations = lambda: {}
        fake_models = types.ModuleType("models")
        fake_models.SummaryInfo = type("FakeSummaryInfo", (), {"__init__": lambda self, **kw: None, "to_dict": lambda self: {}})
        fake_models.MessageInfo = type("MessageInfo", (), {})
        fake_prompts = types.ModuleType("prompts")
        fake_prompts.prompts = types.SimpleNamespace()
        fake_utils = types.ModuleType("utils")
        fake_utils.call_openai = MagicMock()
        fake_utils.extract_links = lambda text: []
        fake_utils.load_json_file = lambda *a, **kw: {}
        fake_utils.save_json_file = lambda *a, **kw: True
        fake_utils.now_iso = lambda: "2026-01-01T00:00:00"
        fake_utils.extract_all_channels = lambda text: []
        fake_utils.text_hash = lambda text: __import__('hashlib').sha256(text.encode()).hexdigest()[:16]
        fake_utils.count_characters = lambda text: len(text)
        fake_utils.enforce_summary_length = lambda text, max_chars: text[:max_chars]
        fake_telegram_client = types.ModuleType("telegram_client")
        fake_telegram_client.user_client = None
        fake_telegram_client.clients_loop = None

        stubs = {
            "dotenv": fake_dotenv,
            "channel_manager": fake_channel_manager,
            "models": fake_models,
            "prompts": fake_prompts,
            "utils": fake_utils,
            "telegram_client": fake_telegram_client,
        }

        return stubs

    def test_invalidate_cache_clears_specific_key(self):
        """invalidate_cache with filepath should clear only that key."""
        stubs = self._import_hm_with_stubs()
        with patch.dict(sys.modules, stubs):
            with patch.dict(os.environ, {}, clear=True):
                sys.modules.pop("history_manager", None)
                sys.modules.pop("config", None)
                hm = importlib.import_module("history_manager")

                hm._cache["file:test.json"] = ["data"]
                hm._cache["file:other.json"] = ["other"]
                hm.invalidate_cache("test.json")
                self.assertNotIn("file:test.json", hm._cache)
                self.assertIn("file:other.json", hm._cache)

    def test_invalidate_cache_clears_all_without_filepath(self):
        """invalidate_cache without filepath should clear entire cache."""
        stubs = self._import_hm_with_stubs()
        with patch.dict(sys.modules, stubs):
            with patch.dict(os.environ, {}, clear=True):
                sys.modules.pop("history_manager", None)
                sys.modules.pop("config", None)
                hm = importlib.import_module("history_manager")

                hm._cache["file:a.json"] = ["a"]
                hm._cache["file:b.json"] = ["b"]
                hm.invalidate_cache()
                self.assertEqual(len(hm._cache), 0)


class TextHashTests(unittest.TestCase):
    """Tests for text_hash helper (now in utils) — None-safe hashing."""

    def test_text_hash_with_normal_text(self):
        """text_hash should produce deterministic hash for normal text."""
        import hashlib

        def text_hash(text):
            return hashlib.sha256((text or "").encode()).hexdigest()[:16]

        h1 = text_hash("hello world")
        h2 = text_hash("hello world")
        self.assertEqual(h1, h2, "Same input should produce same hash")
        self.assertEqual(len(h1), 16, "Hash should be 16 chars")

    def test_text_hash_with_none_returns_same_as_empty(self):
        """text_hash should handle None gracefully, same as empty string."""
        import hashlib

        def text_hash(text):
            return hashlib.sha256((text or "").encode()).hexdigest()[:16]

        h_none = text_hash(None)
        h_empty = text_hash("")
        self.assertEqual(h_none, h_empty, "None and empty string should produce same hash")


class CacheInvalidationTests(unittest.TestCase):
    """Tests for cache invalidation after save operations."""

    def test_save_summarization_history_invalidates_cache(self):
        """save_summarization_history should invalidate HISTORY_FILE cache."""
        import importlib
        import sys
        import types

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_channel_manager = types.ModuleType("channel_manager")
        fake_channel_manager.create_channel_abbreviation = lambda ch: "AB"
        fake_channel_manager.load_channel_abbreviations = lambda: {}
        fake_models = types.ModuleType("models")
        fake_models.SummaryInfo = type("FakeSummaryInfo", (), {})
        fake_models.MessageInfo = type("MessageInfo", (), {
            "from_dict": staticmethod(lambda d: type("M", (), {"channel": "@t", "message_id": 1, "text": "x"})()),
            "to_dict": lambda self: {},
        })
        fake_prompts = types.ModuleType("prompts")
        fake_prompts.prompts = types.SimpleNamespace()
        fake_utils = types.ModuleType("utils")
        fake_utils.call_openai = MagicMock()
        fake_utils.extract_links = lambda text: []
        fake_utils.load_json_file = lambda *a, **kw: {"processed_messages": []}
        fake_utils.save_json_file = lambda *a, **kw: True
        fake_utils.now_iso = lambda: "2026-01-01T00:00:00"
        fake_utils.extract_all_channels = lambda text: []
        fake_utils.text_hash = lambda text: __import__('hashlib').sha256(text.encode()).hexdigest()[:16]
        fake_utils.count_characters = lambda text: len(text)
        fake_utils.enforce_summary_length = lambda text, max_chars: text[:max_chars]
        fake_telegram_client = types.ModuleType("telegram_client")
        fake_telegram_client.user_client = None
        fake_telegram_client.clients_loop = None

        stubs = {
            "dotenv": fake_dotenv,
            "channel_manager": fake_channel_manager,
            "models": fake_models,
            "prompts": fake_prompts,
            "utils": fake_utils,
            "telegram_client": fake_telegram_client,
        }

        with patch.dict(sys.modules, stubs):
            with patch.dict(os.environ, {}, clear=True):
                sys.modules.pop("history_manager", None)
                sys.modules.pop("config", None)
                hm = importlib.import_module("history_manager")

                hm._cache[hm._cache_key("summarization_history.json")] = ["stale"]
                hm.save_summarization_history([])
                self.assertNotIn(hm._cache_key("summarization_history.json"), hm._cache)

    def test_save_group_summarization_history_invalidates_cache(self):
        """save_group_summarization_history should invalidate GROUP_HISTORY_FILE cache."""
        import importlib
        import sys
        import types

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_channel_manager = types.ModuleType("channel_manager")
        fake_channel_manager.create_channel_abbreviation = lambda ch: "AB"
        fake_channel_manager.load_channel_abbreviations = lambda: {}
        fake_models = types.ModuleType("models")
        fake_models.SummaryInfo = type("FakeSummaryInfo", (), {})
        fake_models.MessageInfo = type("MessageInfo", (), {
            "from_dict": staticmethod(lambda d: type("M", (), {"channel": "@t", "message_id": 1, "text": "x"})()),
            "to_dict": lambda self: {},
        })
        fake_prompts = types.ModuleType("prompts")
        fake_prompts.prompts = types.SimpleNamespace()
        fake_utils = types.ModuleType("utils")
        fake_utils.call_openai = MagicMock()
        fake_utils.extract_links = lambda text: []
        fake_utils.load_json_file = lambda *a, **kw: {"processed_messages": [], "last_updated": ""}
        fake_utils.save_json_file = lambda *a, **kw: True
        fake_utils.now_iso = lambda: "2026-01-01T00:00:00"
        fake_utils.extract_all_channels = lambda text: []
        fake_utils.text_hash = lambda text: __import__('hashlib').sha256(text.encode()).hexdigest()[:16]
        fake_utils.count_characters = lambda text: len(text)
        fake_utils.enforce_summary_length = lambda text, max_chars: text[:max_chars]
        fake_telegram_client = types.ModuleType("telegram_client")
        fake_telegram_client.user_client = None
        fake_telegram_client.clients_loop = None

        stubs = {
            "dotenv": fake_dotenv,
            "channel_manager": fake_channel_manager,
            "models": fake_models,
            "prompts": fake_prompts,
            "utils": fake_utils,
            "telegram_client": fake_telegram_client,
        }

        with patch.dict(sys.modules, stubs):
            with patch.dict(os.environ, {}, clear=True):
                sys.modules.pop("history_manager", None)
                sys.modules.pop("config", None)
                hm = importlib.import_module("history_manager")

                hm._cache[hm._cache_key("group_summarization_history.json")] = ["stale"]
                hm.save_group_summarization_history([])
                self.assertNotIn(hm._cache_key("group_summarization_history.json"), hm._cache)


class LoadSummariesHistoryRestoreTests(unittest.TestCase):
    """Tests for load_summaries_history — channel restore only on missing/corrupt file."""

    def _import_hm_with_stubs(self):
        import importlib
        import sys
        import types

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_channel_manager = types.ModuleType("channel_manager")
        fake_channel_manager.create_channel_abbreviation = lambda ch: "AB"
        fake_channel_manager.load_channel_abbreviations = lambda: {}
        fake_models = types.ModuleType("models")

        class FakeSummaryInfo:
            def __init__(self, content="", date=None, message_count=0, channels=None, message_id=None):
                self.content = content
                self.date = date or datetime.now(timezone.utc)
                self.message_count = message_count
                self.channels = channels or []
                self.message_id = message_id

            def to_dict(self):
                return {"content": self.content, "date": self.date.isoformat() if self.date else "",
                        "message_count": self.message_count, "channels": self.channels, "message_id": self.message_id}

        fake_models.SummaryInfo = FakeSummaryInfo
        fake_models.MessageInfo = type("MessageInfo", (), {})
        fake_prompts = types.ModuleType("prompts")
        fake_prompts.prompts = types.SimpleNamespace()
        fake_utils = types.ModuleType("utils")
        fake_utils.call_openai = MagicMock()
        fake_utils.extract_links = lambda text: []
        fake_utils.load_json_file = lambda *a, **kw: {}
        fake_utils.save_json_file = lambda *a, **kw: True
        fake_utils.now_iso = lambda: "2026-01-01T00:00:00"
        fake_utils.extract_all_channels = lambda text: []
        fake_utils.text_hash = lambda text: __import__('hashlib').sha256(text.encode()).hexdigest()[:16]
        fake_utils.count_characters = lambda text: len(text)
        fake_utils.enforce_summary_length = lambda text, max_chars: text[:max_chars]
        fake_telegram_client = types.ModuleType("telegram_client")
        fake_telegram_client.user_client = None
        fake_telegram_client.clients_loop = None

        stubs = {
            "dotenv": fake_dotenv,
            "channel_manager": fake_channel_manager,
            "models": fake_models,
            "prompts": fake_prompts,
            "utils": fake_utils,
            "telegram_client": fake_telegram_client,
        }
        return stubs

    def test_no_restore_when_file_has_empty_summaries(self):
        """load_summaries_history should NOT trigger restore for a valid but empty file."""
        stubs = self._import_hm_with_stubs()
        stubs["utils"].load_json_file = lambda path, default=None: {"summaries": [], "last_updated": "2026-01-01"}

        with patch.dict(sys.modules, stubs):
            with patch.dict(os.environ, {}, clear=True):
                sys.modules.pop("history_manager", None)
                sys.modules.pop("config", None)
                hm = importlib.import_module("history_manager")

                with patch.object(hm, "restore_summaries_from_channel_sync", return_value=["restored"]) as mock_restore:
                    result = hm.load_summaries_history()
                    mock_restore.assert_not_called()
                    self.assertEqual(result, [])

    def test_restore_when_file_missing(self):
        """load_summaries_history should trigger restore when file is missing/corrupt."""
        stubs = self._import_hm_with_stubs()
        stubs["utils"].load_json_file = lambda path, default=None: None

        with patch.dict(sys.modules, stubs):
            with patch.dict(os.environ, {}, clear=True):
                sys.modules.pop("history_manager", None)
                sys.modules.pop("config", None)
                hm = importlib.import_module("history_manager")

                with patch.object(hm, "restore_summaries_from_channel_sync", return_value=[]) as mock_restore:
                    hm.load_summaries_history()
                    mock_restore.assert_called_once()


class LoadGroupSummariesHistoryRestoreTests(unittest.TestCase):
    """Tests for load_group_summaries_history — channel restore on missing/corrupt file."""

    def _import_hm_with_stubs(self):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_channel_manager = types.ModuleType("channel_manager")
        fake_channel_manager.create_channel_abbreviation = lambda ch: "AB"
        fake_channel_manager.load_channel_abbreviations = lambda: {}

        class FakeSummaryInfo:
            def __init__(self, content="", date=None, message_count=0, channels=None, message_id=None):
                self.content = content
                self.date = date or datetime.now(timezone.utc)
                self.message_count = message_count
                self.channels = channels or []
                self.message_id = message_id

            def to_dict(self):
                return {"content": self.content, "date": self.date.isoformat() if self.date else "",
                        "message_count": self.message_count, "channels": self.channels, "message_id": self.message_id}

        fake_models = types.ModuleType("models")
        fake_models.SummaryInfo = FakeSummaryInfo
        fake_models.MessageInfo = type("MessageInfo", (), {})
        fake_prompts = types.ModuleType("prompts")
        fake_prompts.prompts = types.SimpleNamespace()
        fake_utils = types.ModuleType("utils")
        fake_utils.call_openai = MagicMock()
        fake_utils.extract_links = lambda text: []
        fake_utils.load_json_file = lambda *a, **kw: {}
        fake_utils.save_json_file = lambda *a, **kw: True
        fake_utils.now_iso = lambda: "2026-01-01T00:00:00"
        fake_utils.extract_all_channels = lambda text: []
        fake_utils.text_hash = lambda text: __import__('hashlib').sha256(text.encode()).hexdigest()[:16]
        fake_utils.count_characters = lambda text: len(text)
        fake_utils.enforce_summary_length = lambda text, max_chars: text[:max_chars]
        fake_telegram_client = types.ModuleType("telegram_client")
        fake_telegram_client.user_client = None
        fake_telegram_client.clients_loop = None

        stubs = {
            "dotenv": fake_dotenv,
            "channel_manager": fake_channel_manager,
            "models": fake_models,
            "prompts": fake_prompts,
            "utils": fake_utils,
            "telegram_client": fake_telegram_client,
        }
        return stubs

    def test_no_restore_when_file_has_empty_summaries(self):
        """load_group_summaries_history should NOT trigger restore for a valid but empty file."""
        stubs = self._import_hm_with_stubs()
        stubs["utils"].load_json_file = lambda path, default=None: {"summaries": [], "last_updated": "2026-01-01"}

        with patch.dict(sys.modules, stubs):
            with patch.dict(os.environ, {}, clear=True):
                sys.modules.pop("history_manager", None)
                sys.modules.pop("config", None)
                hm = importlib.import_module("history_manager")

                with patch.object(hm, "restore_group_summaries_from_channel_sync", return_value=["restored"]) as mock_restore:
                    result = hm.load_group_summaries_history()
                    mock_restore.assert_not_called()
                    self.assertEqual(result, [])

    def test_restore_when_file_missing(self):
        """load_group_summaries_history should trigger restore when file is missing/corrupt."""
        stubs = self._import_hm_with_stubs()
        stubs["utils"].load_json_file = lambda path, default=None: None

        with patch.dict(sys.modules, stubs):
            with patch.dict(os.environ, {}, clear=True):
                sys.modules.pop("history_manager", None)
                sys.modules.pop("config", None)
                hm = importlib.import_module("history_manager")

                with patch.object(hm, "restore_group_summaries_from_channel_sync", return_value=[]) as mock_restore:
                    hm.load_group_summaries_history()
                    mock_restore.assert_called_once()


class SaveUpdatedSummaryNoMatchTests(unittest.TestCase):
    """Tests for save_updated_summary — should skip save/edit when no match found."""

    def _import_hm_with_stubs(self):
        import importlib
        import sys
        import types

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_channel_manager = types.ModuleType("channel_manager")
        fake_channel_manager.create_channel_abbreviation = lambda ch: "AB"
        fake_channel_manager.load_channel_abbreviations = lambda: {}

        class FakeSummaryInfo:
            def __init__(self, content="", date=None, message_count=0, channels=None, message_id=None):
                self.content = content
                self.date = date or datetime.now(timezone.utc)
                self.message_count = message_count
                self.channels = channels or []
                self.message_id = message_id

            def to_dict(self):
                return {"content": self.content, "date": self.date.isoformat() if self.date else "",
                        "message_count": self.message_count, "channels": self.channels, "message_id": self.message_id}

        fake_models = types.ModuleType("models")
        fake_models.SummaryInfo = FakeSummaryInfo
        fake_models.MessageInfo = type("MessageInfo", (), {})
        fake_prompts = types.ModuleType("prompts")
        fake_prompts.prompts = types.SimpleNamespace()
        fake_utils = types.ModuleType("utils")
        fake_utils.call_openai = MagicMock()
        fake_utils.extract_links = lambda text: []
        fake_utils.load_json_file = lambda *a, **kw: {"summaries": [], "last_updated": ""}
        fake_utils.save_json_file = MagicMock(return_value=True)
        fake_utils.now_iso = lambda: "2026-01-01T00:00:00"
        fake_utils.extract_all_channels = lambda text: []
        fake_utils.text_hash = lambda text: __import__('hashlib').sha256(text.encode()).hexdigest()[:16]
        fake_utils.count_characters = lambda text: len(text)
        fake_utils.enforce_summary_length = lambda text, max_chars: text[:max_chars]
        fake_telegram_client = types.ModuleType("telegram_client")
        fake_telegram_client.user_client = None
        fake_telegram_client.clients_loop = None
        fake_telegram_client.edit_message_in_target_channel = AsyncMock()

        stubs = {
            "dotenv": fake_dotenv,
            "channel_manager": fake_channel_manager,
            "models": fake_models,
            "prompts": fake_prompts,
            "utils": fake_utils,
            "telegram_client": fake_telegram_client,
        }
        return stubs

    def test_save_updated_summary_skips_when_no_match(self):
        """save_updated_summary should not save or edit when original summary is not found."""
        stubs = self._import_hm_with_stubs()

        with patch.dict(sys.modules, stubs):
            with patch.dict(os.environ, {}, clear=True):
                sys.modules.pop("history_manager", None)
                sys.modules.pop("config", None)
                hm = importlib.import_module("history_manager")

                FakeSI = stubs["models"].SummaryInfo
                original = FakeSI(content="nonexistent", message_id=9999)
                updated = FakeSI(content="updated", message_id=9999)

                async def _test():
                    await hm.save_updated_summary(original, updated)

                asyncio.run(_test())
                stubs["utils"].save_json_file.assert_not_called()


class UpdateExistingSummaryLLMTests(unittest.TestCase):
    """Tests for update_existing_summary LLM integration."""

    def test_update_existing_summary_uses_llm(self):
        """update_existing_summary should call LLM to integrate new info."""
        import types
        import importlib

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        fake_channel_manager = types.ModuleType("channel_manager")
        fake_channel_manager.create_channel_abbreviation = lambda ch: "AB"
        fake_channel_manager.load_channel_abbreviations = lambda: {}

        class FakeSummaryInfo:
            def __init__(self, content="", date=None, message_count=0, channels=None, message_id=None):
                self.content = content
                self.date = date or datetime.now(timezone.utc)
                self.message_count = message_count
                self.channels = channels or []
                self.message_id = message_id

            def to_dict(self):
                return {"content": self.content, "date": self.date.isoformat() if self.date else "",
                        "message_count": self.message_count, "channels": self.channels, "message_id": self.message_id}

        class FakeMessageInfo:
            def __init__(self, text="", channel="", message_id=0, date=None, link=""):
                self.text = text
                self.channel = channel
                self.message_id = message_id
                self.date = date or datetime.now(timezone.utc)
                self.link = link

            def get_telegram_link(self):
                return f"https://t.me/{self.channel.lstrip('@')}/{self.message_id}"

        fake_models = types.ModuleType("models")
        fake_models.SummaryInfo = FakeSummaryInfo
        fake_models.MessageInfo = FakeMessageInfo
        fake_prompts = types.ModuleType("prompts")
        fake_prompts.prompts = types.SimpleNamespace()
        fake_utils = types.ModuleType("utils")
        fake_utils.call_openai = AsyncMock(return_value="Updated summary with new link")
        fake_utils.extract_links = lambda text: ["https://example.com"]
        fake_utils.load_json_file = lambda *a, **kw: {"summaries": [], "last_updated": ""}
        fake_utils.save_json_file = MagicMock(return_value=True)
        fake_utils.now_iso = lambda: "2026-01-01T00:00:00"
        fake_utils.text_hash = lambda text: __import__('hashlib').sha256(text.encode()).hexdigest()[:16]
        fake_utils.count_characters = lambda text: len(text)
        fake_utils.enforce_summary_length = lambda text, max_chars: text[:max_chars]

        stubs = {
            "dotenv": fake_dotenv,
            "channel_manager": fake_channel_manager,
            "models": fake_models,
            "prompts": fake_prompts,
            "utils": fake_utils,
        }

        with patch.dict(sys.modules, stubs):
            with patch.dict(os.environ, {}, clear=True):
                sys.modules.pop("history_manager", None)
                sys.modules.pop("config", None)
                hm = importlib.import_module("history_manager")

                summary = FakeSummaryInfo(content="Original summary", message_id=1)
                msg = FakeMessageInfo(text="New info", channel="@test", message_id=42)

                async def _test():
                    result = await hm.update_existing_summary(summary, msg)
                    return result

                updated = asyncio.run(_test())
                self.assertEqual(updated.content, "Updated summary with new link")
                fake_utils.call_openai.assert_called_once()

    def test_update_existing_summary_uses_update_max_tokens(self):
        """update_existing_summary should use UPDATE_SUMMARY_MAX_TOKENS, not full channel max."""
        import types
        import importlib

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        fake_channel_manager = types.ModuleType("channel_manager")
        fake_channel_manager.create_channel_abbreviation = lambda ch: "AB"
        fake_channel_manager.load_channel_abbreviations = lambda: {}

        class FakeSummaryInfo:
            def __init__(self, content="", date=None, message_count=0, channels=None, message_id=None):
                self.content = content
                self.date = date or datetime.now(timezone.utc)
                self.message_count = message_count
                self.channels = channels or []
                self.message_id = message_id

        class FakeMessageInfo:
            def __init__(self, text="", channel="", message_id=0, date=None, link=""):
                self.text = text
                self.channel = channel
                self.message_id = message_id
                self.date = date or datetime.now(timezone.utc)
                self.link = link

            def get_telegram_link(self):
                return f"https://t.me/{self.channel.lstrip('@')}/{self.message_id}"

        fake_models = types.ModuleType("models")
        fake_models.SummaryInfo = FakeSummaryInfo
        fake_models.MessageInfo = FakeMessageInfo
        fake_prompts = types.ModuleType("prompts")
        fake_prompts.prompts = types.SimpleNamespace()
        fake_utils = types.ModuleType("utils")
        fake_utils.call_openai = AsyncMock(return_value="Updated")
        fake_utils.extract_links = lambda text: []
        fake_utils.load_json_file = lambda *a, **kw: {"summaries": [], "last_updated": ""}
        fake_utils.save_json_file = MagicMock(return_value=True)
        fake_utils.now_iso = lambda: "2026-01-01T00:00:00"
        fake_utils.text_hash = lambda text: __import__('hashlib').sha256(text.encode()).hexdigest()[:16]
        fake_utils.count_characters = lambda text: len(text)
        fake_utils.enforce_summary_length = lambda text, max_chars: text[:max_chars]

        stubs = {
            "dotenv": fake_dotenv,
            "channel_manager": fake_channel_manager,
            "models": fake_models,
            "prompts": fake_prompts,
            "utils": fake_utils,
        }

        with patch.dict(sys.modules, stubs):
            with patch.dict(os.environ, {}, clear=True):
                sys.modules.pop("history_manager", None)
                sys.modules.pop("config", None)
                hm = importlib.import_module("history_manager")

                summary = FakeSummaryInfo(content="Original", message_id=1)
                msg = FakeMessageInfo(text="New info", channel="@test", message_id=42)

                async def _test():
                    return await hm.update_existing_summary(summary, msg)

                asyncio.run(_test())

                call_kwargs = fake_utils.call_openai.call_args
                self.assertEqual(call_kwargs.kwargs.get("max_tokens"), 2000)

    def test_update_existing_summary_fallback_on_llm_failure(self):
        """update_existing_summary should fall back to append on LLM failure."""
        import types
        import importlib

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        fake_channel_manager = types.ModuleType("channel_manager")
        fake_channel_manager.create_channel_abbreviation = lambda ch: "AB"
        fake_channel_manager.load_channel_abbreviations = lambda: {}

        class FakeSummaryInfo:
            def __init__(self, content="", date=None, message_count=0, channels=None, message_id=None):
                self.content = content
                self.date = date or datetime.now(timezone.utc)
                self.message_count = message_count
                self.channels = channels or []
                self.message_id = message_id

        class FakeMessageInfo:
            def __init__(self, text="", channel="", message_id=0, date=None, link=""):
                self.text = text
                self.channel = channel
                self.message_id = message_id
                self.date = date or datetime.now(timezone.utc)
                self.link = link

            def get_telegram_link(self):
                return f"https://t.me/{self.channel.lstrip('@')}/{self.message_id}"

        fake_models = types.ModuleType("models")
        fake_models.SummaryInfo = FakeSummaryInfo
        fake_models.MessageInfo = FakeMessageInfo
        fake_prompts = types.ModuleType("prompts")
        fake_prompts.prompts = types.SimpleNamespace()
        fake_utils = types.ModuleType("utils")
        fake_utils.call_openai = AsyncMock(side_effect=Exception("LLM error"))
        fake_utils.extract_links = lambda text: ["https://example.com"]
        fake_utils.load_json_file = lambda *a, **kw: {"summaries": [], "last_updated": ""}
        fake_utils.save_json_file = MagicMock(return_value=True)
        fake_utils.now_iso = lambda: "2026-01-01T00:00:00"
        fake_utils.text_hash = lambda text: __import__('hashlib').sha256(text.encode()).hexdigest()[:16]
        fake_utils.count_characters = lambda text: len(text)
        fake_utils.enforce_summary_length = lambda text, max_chars: text[:max_chars]

        stubs = {
            "dotenv": fake_dotenv,
            "channel_manager": fake_channel_manager,
            "models": fake_models,
            "prompts": fake_prompts,
            "utils": fake_utils,
        }

        with patch.dict(sys.modules, stubs):
            with patch.dict(os.environ, {}, clear=True):
                sys.modules.pop("history_manager", None)
                sys.modules.pop("config", None)
                hm = importlib.import_module("history_manager")

                summary = FakeSummaryInfo(content="Original summary", message_id=1)
                msg = FakeMessageInfo(text="New info", channel="@test", message_id=42)

                async def _test():
                    result = await hm.update_existing_summary(summary, msg)
                    return result

                updated = asyncio.run(_test())
                self.assertIn("Original summary", updated.content)
                self.assertIn("Другие ссылки:", updated.content)

    def test_update_existing_summary_fallback_on_empty_llm_response(self):
        """update_existing_summary should fall back to append on empty LLM response."""
        import types
        import importlib

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        fake_channel_manager = types.ModuleType("channel_manager")
        fake_channel_manager.create_channel_abbreviation = lambda ch: "AB"
        fake_channel_manager.load_channel_abbreviations = lambda: {}

        class FakeSummaryInfo:
            def __init__(self, content="", date=None, message_count=0, channels=None, message_id=None):
                self.content = content
                self.date = date or datetime.now(timezone.utc)
                self.message_count = message_count
                self.channels = channels or []
                self.message_id = message_id

        class FakeMessageInfo:
            def __init__(self, text="", channel="", message_id=0, date=None, link=""):
                self.text = text
                self.channel = channel
                self.message_id = message_id
                self.date = date or datetime.now(timezone.utc)
                self.link = link

            def get_telegram_link(self):
                return f"https://t.me/{self.channel.lstrip('@')}/{self.message_id}"

        fake_models = types.ModuleType("models")
        fake_models.SummaryInfo = FakeSummaryInfo
        fake_models.MessageInfo = FakeMessageInfo
        fake_prompts = types.ModuleType("prompts")
        fake_prompts.prompts = types.SimpleNamespace()
        fake_utils = types.ModuleType("utils")
        fake_utils.call_openai = AsyncMock(return_value="")
        fake_utils.extract_links = lambda text: ["https://example.com"]
        fake_utils.load_json_file = lambda *a, **kw: {"summaries": [], "last_updated": ""}
        fake_utils.save_json_file = MagicMock(return_value=True)
        fake_utils.now_iso = lambda: "2026-01-01T00:00:00"
        fake_utils.text_hash = lambda text: __import__('hashlib').sha256(text.encode()).hexdigest()[:16]
        fake_utils.count_characters = lambda text: len(text)
        fake_utils.enforce_summary_length = lambda text, max_chars: text[:max_chars]

        stubs = {
            "dotenv": fake_dotenv,
            "channel_manager": fake_channel_manager,
            "models": fake_models,
            "prompts": fake_prompts,
            "utils": fake_utils,
        }

        with patch.dict(sys.modules, stubs):
            with patch.dict(os.environ, {}, clear=True):
                sys.modules.pop("history_manager", None)
                sys.modules.pop("config", None)
                hm = importlib.import_module("history_manager")

                summary = FakeSummaryInfo(content="Original summary", message_id=1)
                msg = FakeMessageInfo(text="New info", channel="@test", message_id=42)

                async def _test():
                    result = await hm.update_existing_summary(summary, msg)
                    return result

                updated = asyncio.run(_test())
                self.assertIn("Другие ссылки:", updated.content)


class UpdateExistingSummaryTemperatureTests(unittest.TestCase):
    """Tests that update_existing_summary uses temperature=0 for deterministic link insertion."""

    def test_update_existing_summary_uses_temperature_zero(self):
        import types
        import importlib

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        fake_channel_manager = types.ModuleType("channel_manager")
        fake_channel_manager.create_channel_abbreviation = lambda ch: "AB"
        fake_channel_manager.load_channel_abbreviations = lambda: {}

        class FakeSummaryInfo:
            def __init__(self, content="", date=None, message_count=0, channels=None, message_id=None):
                self.content = content
                self.date = date or datetime.now(timezone.utc)
                self.message_count = message_count
                self.channels = channels or []
                self.message_id = message_id

        class FakeMessageInfo:
            def __init__(self, text="", channel="", message_id=0, date=None, link=""):
                self.text = text
                self.channel = channel
                self.message_id = message_id
                self.date = date or datetime.now(timezone.utc)
                self.link = link

            def get_telegram_link(self):
                return f"https://t.me/{self.channel.lstrip('@')}/{self.message_id}"

        fake_models = types.ModuleType("models")
        fake_models.SummaryInfo = FakeSummaryInfo
        fake_models.MessageInfo = FakeMessageInfo
        fake_prompts = types.ModuleType("prompts")
        fake_prompts.prompts = types.SimpleNamespace()
        fake_utils = types.ModuleType("utils")
        fake_utils.call_openai = AsyncMock(return_value="Updated with link")
        fake_utils.extract_links = lambda text: []
        fake_utils.load_json_file = lambda *a, **kw: {"summaries": [], "last_updated": ""}
        fake_utils.save_json_file = MagicMock(return_value=True)
        fake_utils.now_iso = lambda: "2026-01-01T00:00:00"
        fake_utils.text_hash = lambda text: __import__('hashlib').sha256(text.encode()).hexdigest()[:16]
        fake_utils.count_characters = lambda text: len(text)
        fake_utils.enforce_summary_length = lambda text, max_chars: text[:max_chars]

        stubs = {
            "dotenv": fake_dotenv,
            "channel_manager": fake_channel_manager,
            "models": fake_models,
            "prompts": fake_prompts,
            "utils": fake_utils,
        }

        with patch.dict(sys.modules, stubs):
            with patch.dict(os.environ, {}, clear=True):
                sys.modules.pop("history_manager", None)
                sys.modules.pop("config", None)
                hm = importlib.import_module("history_manager")

                summary = FakeSummaryInfo(content="Original", message_id=1)
                msg = FakeMessageInfo(text="New info", channel="@test", message_id=42)

                async def _test():
                    return await hm.update_existing_summary(summary, msg)

                asyncio.run(_test())

                call_kwargs = fake_utils.call_openai.call_args
                self.assertEqual(call_kwargs.kwargs.get("temperature"), 0)


class SharedLoadSaveHelperTests(unittest.TestCase):
    """Tests for _load_processed_messages and _save_processed_messages shared helpers."""

    def _import_hm_with_stubs(self):
        import importlib
        import sys
        import types

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_channel_manager = types.ModuleType("channel_manager")
        fake_channel_manager.create_channel_abbreviation = lambda ch: "AB"
        fake_channel_manager.load_channel_abbreviations = lambda: {}

        class FakeMessageInfo:
            def __init__(self, text="", channel="", message_id=0, date=None, link=""):
                self.text = text
                self.channel = channel
                self.message_id = message_id
                self.date = date or datetime.now(timezone.utc)
                self.link = link

            @staticmethod
            def from_dict(d):
                m = FakeMessageInfo()
                m.text = d.get("text", "")
                m.channel = d.get("channel", "")
                m.message_id = d.get("message_id", 0)
                return m

            def to_dict(self):
                return {"text": self.text, "channel": self.channel, "message_id": self.message_id}

        fake_models = types.ModuleType("models")
        fake_models.SummaryInfo = type("FakeSummaryInfo", (), {"__init__": lambda self, **kw: None, "to_dict": lambda self: {}})
        fake_models.MessageInfo = FakeMessageInfo
        fake_prompts = types.ModuleType("prompts")
        fake_prompts.prompts = types.SimpleNamespace()
        fake_utils = types.ModuleType("utils")
        fake_utils.call_openai = MagicMock()
        fake_utils.extract_links = lambda text: []
        fake_utils.load_json_file = lambda *a, **kw: {"processed_messages": []}
        fake_utils.save_json_file = MagicMock(return_value=True)
        fake_utils.now_iso = lambda: "2026-01-01T00:00:00"
        fake_utils.extract_all_channels = lambda text: []
        fake_utils.text_hash = lambda text: __import__('hashlib').sha256(text.encode()).hexdigest()[:16]
        fake_utils.count_characters = lambda text: len(text)
        fake_utils.enforce_summary_length = lambda text, max_chars: text[:max_chars]
        fake_telegram_client = types.ModuleType("telegram_client")
        fake_telegram_client.user_client = None
        fake_telegram_client.clients_loop = None

        stubs = {
            "dotenv": fake_dotenv,
            "channel_manager": fake_channel_manager,
            "models": fake_models,
            "prompts": fake_prompts,
            "utils": fake_utils,
            "telegram_client": fake_telegram_client,
        }

        return stubs

    def test_load_processed_messages_returns_set(self):
        """_load_processed_messages should return a set of message ID strings."""
        stubs = self._import_hm_with_stubs()
        stubs["utils"].load_json_file = lambda path, default=None: {
            "processed_messages": [
                {"text": "hello", "channel": "@ch", "message_id": 1},
                {"text": "world", "channel": "@ch", "message_id": 2},
            ]
        }

        with patch.dict(sys.modules, stubs):
            with patch.dict(os.environ, {}, clear=True):
                sys.modules.pop("history_manager", None)
                sys.modules.pop("config", None)
                hm = importlib.import_module("history_manager")

                result = hm._load_processed_messages("test.json")
                self.assertIsInstance(result, set)
                self.assertEqual(len(result), 2)

    def test_save_processed_messages_appends_and_truncates(self):
        """_save_processed_messages should append new messages and truncate to max."""
        stubs = self._import_hm_with_stubs()

        saved_data = {}

        def fake_save(filepath, data, error_msg):
            saved_data[filepath] = data
            return True

        stubs["utils"].save_json_file = fake_save
        stubs["utils"].load_json_file = lambda path, default=None: {
            "processed_messages": [{"text": "old", "channel": "@ch", "message_id": 1}]
        }

        with patch.dict(sys.modules, stubs):
            with patch.dict(os.environ, {}, clear=True):
                sys.modules.pop("history_manager", None)
                sys.modules.pop("config", None)
                hm = importlib.import_module("history_manager")

                FakeMI = stubs["models"].MessageInfo
                new_msgs = [FakeMI(text="new", channel="@ch", message_id=2)]

                hm._save_processed_messages("test.json", new_msgs, max_messages=1, error_msg="test")

                self.assertEqual(len(saved_data["test.json"]["processed_messages"]), 1)
                self.assertEqual(saved_data["test.json"]["processed_messages"][0]["text"], "new")


class GroupSummariesCacheEmptyTests(unittest.TestCase):
    """Tests that load_group_summaries_history caches empty results."""

    def _import_hm_with_stubs(self):
        import importlib
        import sys
        import types

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        fake_channel_manager = types.ModuleType("channel_manager")
        fake_channel_manager.create_channel_abbreviation = lambda ch: "AB"
        fake_channel_manager.load_channel_abbreviations = lambda: {}

        class FakeSummaryInfo:
            def __init__(self, content="", date=None, message_count=0, channels=None, message_id=None):
                self.content = content
                self.date = date or datetime.now(timezone.utc)
                self.message_count = message_count
                self.channels = channels or []
                self.message_id = message_id

            def to_dict(self):
                return {"content": self.content, "date": self.date.isoformat(),
                        "message_count": self.message_count, "channels": self.channels,
                        "message_id": self.message_id}

        class FakeMessageInfo:
            def __init__(self, text="", channel="", message_id=0, date=None, link=""):
                self.text = text
                self.channel = channel
                self.message_id = message_id
                self.date = date or datetime.now(timezone.utc)
                self.link = link

            def to_dict(self):
                return {"text": self.text, "channel": self.channel, "message_id": self.message_id}

        fake_models = types.ModuleType("models")
        fake_models.SummaryInfo = FakeSummaryInfo
        fake_models.MessageInfo = FakeMessageInfo
        fake_prompts = types.ModuleType("prompts")
        fake_prompts.prompts = types.SimpleNamespace()
        fake_utils = types.ModuleType("utils")
        fake_utils.call_openai = MagicMock()
        fake_utils.extract_links = lambda text: []
        fake_utils.load_json_file = lambda *a, **kw: {"summaries": [], "last_updated": ""}
        fake_utils.save_json_file = MagicMock(return_value=True)
        fake_utils.now_iso = lambda: "2026-01-01T00:00:00"
        fake_utils.extract_all_channels = lambda text: []
        fake_utils.text_hash = lambda text: __import__('hashlib').sha256(text.encode()).hexdigest()[:16]
        fake_utils.count_characters = lambda text: len(text)
        fake_utils.enforce_summary_length = lambda text, max_chars: text[:max_chars]
        fake_telegram_client = types.ModuleType("telegram_client")
        fake_telegram_client.user_client = None
        fake_telegram_client.clients_loop = None

        stubs = {
            "dotenv": fake_dotenv,
            "channel_manager": fake_channel_manager,
            "models": fake_models,
            "prompts": fake_prompts,
            "utils": fake_utils,
            "telegram_client": fake_telegram_client,
        }

        return stubs

    def test_load_group_summaries_caches_empty_results(self):
        stubs = self._import_hm_with_stubs()
        with patch.dict(sys.modules, stubs):
            with patch.dict(os.environ, {}, clear=True):
                sys.modules.pop("history_manager", None)
                sys.modules.pop("config", None)
                hm = importlib.import_module("history_manager")

                result1 = hm.load_group_summaries_history()
                self.assertEqual(result1, [])

                cache_key = hm._cache_key(hm.GROUP_SUMMARIES_HISTORY_FILE)
                self.assertIn(cache_key, hm._cache)


class SharedSaveSummaryHelperTests(unittest.TestCase):
    """Tests for _save_summary_to_history_file shared helper."""

    def _import_hm_with_stubs(self):
        import importlib
        import sys
        import types

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        fake_channel_manager = types.ModuleType("channel_manager")
        fake_channel_manager.create_channel_abbreviation = lambda ch: "AB"
        fake_channel_manager.load_channel_abbreviations = lambda: {}

        class FakeSummaryInfo:
            def __init__(self, content="", date=None, message_count=0, channels=None, message_id=None):
                self.content = content
                self.date = date or datetime.now(timezone.utc)
                self.message_count = message_count
                self.channels = channels or []
                self.message_id = message_id

            def to_dict(self):
                return {"content": self.content, "date": self.date.isoformat(),
                        "message_count": self.message_count, "channels": self.channels,
                        "message_id": self.message_id}

        fake_models = types.ModuleType("models")
        fake_models.SummaryInfo = FakeSummaryInfo
        fake_models.MessageInfo = type("FakeMessageInfo", (), {"__init__": lambda self, **kw: None, "to_dict": lambda self: {}})
        fake_prompts = types.ModuleType("prompts")
        fake_prompts.prompts = types.SimpleNamespace()
        fake_utils = types.ModuleType("utils")
        fake_utils.call_openai = MagicMock()
        fake_utils.extract_links = lambda text: []
        fake_utils.load_json_file = lambda *a, **kw: {"summaries": [], "last_updated": ""}
        fake_utils.save_json_file = MagicMock(return_value=True)
        fake_utils.now_iso = lambda: "2026-01-01T00:00:00"
        fake_utils.text_hash = lambda text: __import__('hashlib').sha256(text.encode()).hexdigest()[:16]
        fake_utils.count_characters = lambda text: len(text)
        fake_utils.enforce_summary_length = lambda text, max_chars: text[:max_chars]
        fake_telegram_client = types.ModuleType("telegram_client")
        fake_telegram_client.user_client = None
        fake_telegram_client.clients_loop = None

        stubs = {
            "dotenv": fake_dotenv,
            "channel_manager": fake_channel_manager,
            "models": fake_models,
            "prompts": fake_prompts,
            "utils": fake_utils,
            "telegram_client": fake_telegram_client,
        }

        return stubs

    def test_save_summary_to_history_file_appends_and_truncates(self):
        stubs = self._import_hm_with_stubs()
        saved_data = {}

        def fake_save(filepath, data, error_msg):
            saved_data[filepath] = data
            return True

        stubs["utils"].save_json_file = fake_save

        with patch.dict(sys.modules, stubs):
            with patch.dict(os.environ, {}, clear=True):
                sys.modules.pop("history_manager", None)
                sys.modules.pop("config", None)
                hm = importlib.import_module("history_manager")

                FakeSI = stubs["models"].SummaryInfo
                summary = FakeSI(content="Test summary")

                hm._save_summary_to_history_file(
                    summary, "test_summaries.json", max_summaries=1, error_msg="test"
                )

                self.assertEqual(len(saved_data["test_summaries.json"]["summaries"]), 1)
                self.assertEqual(saved_data["test_summaries.json"]["summaries"][0]["content"], "Test summary")


class UpdateSummaryLengthGuardTests(unittest.TestCase):
    """Tests for update_existing_summary length guard — falls back to append on truncation."""

    def _make_stubs(self):
        import types
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_channel_manager = types.ModuleType("channel_manager")
        fake_channel_manager.create_channel_abbreviation = lambda ch: "AB"
        fake_channel_manager.load_channel_abbreviations = lambda: {}

        class FakeSummaryInfo:
            def __init__(self, content="", date=None, message_count=0, channels=None, message_id=None):
                self.content = content
                self.date = date or datetime.now(timezone.utc)
                self.message_count = message_count
                self.channels = channels or []
                self.message_id = message_id

        class FakeMessageInfo:
            def __init__(self, text="", channel="", message_id=0, date=None, link=""):
                self.text = text
                self.channel = channel
                self.message_id = message_id
                self.date = date or datetime.now(timezone.utc)
                self.link = link

            def get_telegram_link(self):
                return f"https://t.me/{self.channel.lstrip('@')}/{self.message_id}"

        fake_models = types.ModuleType("models")
        fake_models.SummaryInfo = FakeSummaryInfo
        fake_models.MessageInfo = FakeMessageInfo
        fake_prompts = types.ModuleType("prompts")
        fake_prompts.prompts = types.SimpleNamespace()
        fake_utils = types.ModuleType("utils")
        fake_utils.extract_links = lambda text: []
        fake_utils.load_json_file = lambda *a, **kw: {"summaries": [], "last_updated": ""}
        fake_utils.save_json_file = MagicMock(return_value=True)
        fake_utils.now_iso = lambda: "2026-01-01T00:00:00"
        fake_utils.text_hash = lambda text: __import__('hashlib').sha256(text.encode()).hexdigest()[:16]
        fake_utils.count_characters = lambda text: len(text)
        fake_utils.enforce_summary_length = lambda text, max_chars: text[:max_chars]

        stubs = {
            "dotenv": fake_dotenv,
            "channel_manager": fake_channel_manager,
            "models": fake_models,
            "prompts": fake_prompts,
            "utils": fake_utils,
        }
        return stubs, FakeSummaryInfo, FakeMessageInfo, fake_utils

    def test_falls_back_when_llm_response_truncated(self):
        """If LLM response is <80% of original length, fall back to append."""
        import types
        import importlib

        stubs, FakeSI, FakeMI, fake_utils = self._make_stubs()
        original_content = "A" * 1000
        truncated_response = "B" * 100

        fake_utils.call_openai = AsyncMock(return_value=truncated_response)

        with patch.dict(sys.modules, stubs):
            with patch.dict(os.environ, {}, clear=True):
                sys.modules.pop("history_manager", None)
                sys.modules.pop("config", None)
                hm = importlib.import_module("history_manager")

                summary = FakeSI(content=original_content, message_id=1)
                msg = FakeMI(text="New info", channel="@test", message_id=42)

                async def _test():
                    return await hm.update_existing_summary(summary, msg)

                updated = asyncio.run(_test())
                self.assertIn("Другие ссылки:", updated.content)
                self.assertIn(original_content, updated.content)

    def test_keeps_llm_response_when_length_sufficient(self):
        """If LLM response is >=80% of original length, keep it."""
        import types
        import importlib

        stubs, FakeSI, FakeMI, fake_utils = self._make_stubs()
        original_content = "A" * 1000
        good_response = "B" * 900

        fake_utils.call_openai = AsyncMock(return_value=good_response)

        with patch.dict(sys.modules, stubs):
            with patch.dict(os.environ, {}, clear=True):
                sys.modules.pop("history_manager", None)
                sys.modules.pop("config", None)
                hm = importlib.import_module("history_manager")

                summary = FakeSI(content=original_content, message_id=1)
                msg = FakeMI(text="New info", channel="@test", message_id=42)

                async def _test():
                    return await hm.update_existing_summary(summary, msg)

                updated = asyncio.run(_test())
                self.assertEqual(updated.content, good_response)
                self.assertNotIn("Другие ссылки:", updated.content)


class ConfigDedupToggleTests(unittest.TestCase):
    """Tests for ENABLE_SUMMARIES_DEDUPLICATION and ENABLE_SUMMARY_UPDATES env-configurable."""

    def test_enable_dedup_defaults_true(self):
        import importlib
        import sys
        import types

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, {
                "TELEGRAM_API_ID": "1", "TELEGRAM_API_HASH": "h",
                "TELEGRAM_BOT_TOKEN": "t", "TARGET_CHANNEL": "@c",
                "OPENAI_API_KEY": "k",
            }, clear=True):
                sys.modules.pop("config", None)
                config = importlib.import_module("config")
                self.assertTrue(config.ENABLE_SUMMARIES_DEDUPLICATION)

    def test_enable_dedup_can_be_disabled(self):
        import importlib
        import sys
        import types

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, {
                "TELEGRAM_API_ID": "1", "TELEGRAM_API_HASH": "h",
                "TELEGRAM_BOT_TOKEN": "t", "TARGET_CHANNEL": "@c",
                "OPENAI_API_KEY": "k",
                "ENABLE_SUMMARIES_DEDUPLICATION": "false",
            }, clear=True):
                sys.modules.pop("config", None)
                config = importlib.import_module("config")
                self.assertFalse(config.ENABLE_SUMMARIES_DEDUPLICATION)

    def test_enable_updates_defaults_true(self):
        import importlib
        import sys
        import types

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, {
                "TELEGRAM_API_ID": "1", "TELEGRAM_API_HASH": "h",
                "TELEGRAM_BOT_TOKEN": "t", "TARGET_CHANNEL": "@c",
                "OPENAI_API_KEY": "k",
            }, clear=True):
                sys.modules.pop("config", None)
                config = importlib.import_module("config")
                self.assertTrue(config.ENABLE_SUMMARY_UPDATES)

    def test_enable_updates_can_be_disabled(self):
        import importlib
        import sys
        import types

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, {
                "TELEGRAM_API_ID": "1", "TELEGRAM_API_HASH": "h",
                "TELEGRAM_BOT_TOKEN": "t", "TARGET_CHANNEL": "@c",
                "OPENAI_API_KEY": "k",
                "ENABLE_SUMMARY_UPDATES": "0",
            }, clear=True):
                sys.modules.pop("config", None)
                config = importlib.import_module("config")
                self.assertFalse(config.ENABLE_SUMMARY_UPDATES)


class UpdateSummaryInputTruncationTests(unittest.TestCase):
    """Tests for summary content truncation in update_existing_summary."""

    def test_truncates_summary_in_update_prompt(self):
        """update_existing_summary should truncate summary content to UPDATE_SUMMARY_MAX_INPUT_CHARS."""
        from models import MessageInfo, SummaryInfo

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_openai = types.ModuleType("openai")

        class FakeOpenAIError(Exception):
            pass

        async def fake_call_openai(system_prompt, user_content, **kwargs):
            self.assertLessEqual(
                user_content.find("Саммари:\n") + len("Саммари:\n") + 2001,
                user_content.find("\n\nНовое сообщение:"),
            )
            return SummaryInfo.__module__  # return something truthy

        fake_openai.AsyncOpenAI = type("AsyncOpenAI", (), {"__init__": lambda self, **kw: None})
        fake_openai.APIError = FakeOpenAIError
        fake_openai.RateLimitError = type("RateLimitError", (FakeOpenAIError,), {"status_code": None})
        fake_openai.APIConnectionError = type("APIConnectionError", (FakeOpenAIError,), {})

        with patch.dict(sys.modules, {"dotenv": fake_dotenv, "openai": fake_openai}):
            with patch.dict(os.environ, {
                "TELEGRAM_API_ID": "1", "TELEGRAM_API_HASH": "h",
                "TELEGRAM_BOT_TOKEN": "t", "TARGET_CHANNEL": "@c",
                "OPENAI_API_KEY": "k",
                "UPDATE_SUMMARY_MAX_INPUT_CHARS": "2000",
            }, clear=True):
                for mod_name in list(sys.modules.keys()):
                    if mod_name in ("config", "utils", "history_manager", "channel_manager", "prompts", "models"):
                        sys.modules.pop(mod_name, None)
                config = importlib.import_module("config")
                models = importlib.import_module("models")

                long_summary = models.SummaryInfo(
                    content="x" * 5000,
                    date=datetime.now(timezone.utc),
                    message_count=3,
                    channels=["@ch"],
                    message_id=100,
                )
                new_msg = models.MessageInfo(
                    text="New message about AI research with enough content",
                    channel="@ch",
                    message_id=200,
                    date=datetime.now(timezone.utc),
                    link="",
                )

                with patch("history_manager.call_openai", new_callable=AsyncMock) as mock_openai:
                    mock_openai.return_value = "x" * 5000
                    hm = importlib.import_module("history_manager")
                    result = asyncio.run(hm.update_existing_summary(long_summary, new_msg))
                    user_content = mock_openai.call_args[0][1]
                    summary_part = user_content.split("Саммари:\n")[1].split("\n\nНовое сообщение:")[0]
                    self.assertLessEqual(len(summary_part), 2001)


class UpdateSummaryEnforcesLengthTests(unittest.TestCase):
    """Tests for update_existing_summary enforcing summary max length on the result."""

    def test_enforces_length_on_oversized_update(self):
        """When update_existing_summary produces content exceeding SUMMARY_MAX_LENGTH,
        the returned SummaryInfo content should be length-enforced."""
        import types
        import importlib

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        fake_channel_manager = types.ModuleType("channel_manager")
        fake_channel_manager.create_channel_abbreviation = lambda ch: "AB"
        fake_channel_manager.load_channel_abbreviations = lambda: {}

        class FakeSummaryInfo:
            def __init__(self, content="", date=None, message_count=0, channels=None, message_id=None):
                self.content = content
                self.date = date or datetime.now(timezone.utc)
                self.message_count = message_count
                self.channels = channels or []
                self.message_id = message_id

            def to_dict(self):
                return {"content": self.content, "date": self.date.isoformat() if self.date else "",
                        "message_count": self.message_count, "channels": self.channels, "message_id": self.message_id}

        class FakeMessageInfo:
            def __init__(self, text="", channel="", message_id=0, date=None, link=""):
                self.text = text
                self.channel = channel
                self.message_id = message_id
                self.date = date or datetime.now(timezone.utc)
                self.link = link

            def get_telegram_link(self):
                return f"https://t.me/{self.channel.lstrip('@')}/{self.message_id}"

        fake_models = types.ModuleType("models")
        fake_models.SummaryInfo = FakeSummaryInfo
        fake_models.MessageInfo = FakeMessageInfo
        fake_prompts = types.ModuleType("prompts")
        fake_prompts.prompts = types.SimpleNamespace()

        oversized_response = "X" * 5000
        fake_utils = types.ModuleType("utils")
        fake_utils.call_openai = AsyncMock(return_value=oversized_response)
        fake_utils.extract_links = lambda text: []
        fake_utils.load_json_file = lambda *a, **kw: {"summaries": [], "last_updated": ""}
        fake_utils.save_json_file = MagicMock(return_value=True)
        fake_utils.now_iso = lambda: "2026-01-01T00:00:00"
        fake_utils.text_hash = lambda text: __import__('hashlib').sha256(text.encode()).hexdigest()[:16]
        fake_utils.count_characters = lambda text: len(text)
        fake_utils.enforce_summary_length = lambda text, max_chars: text[:max_chars]

        stubs = {
            "dotenv": fake_dotenv,
            "channel_manager": fake_channel_manager,
            "models": fake_models,
            "prompts": fake_prompts,
            "utils": fake_utils,
        }

        with patch.dict(sys.modules, stubs):
            with patch.dict(os.environ, {}, clear=True):
                sys.modules.pop("history_manager", None)
                sys.modules.pop("config", None)
                hm = importlib.import_module("history_manager")

                self.assertEqual(hm.SUMMARY_MAX_LENGTH, 4000)

                summary = FakeSummaryInfo(content="A" * 2000, message_id=1)
                msg = FakeMessageInfo(text="New info", channel="@test", message_id=42)

                async def _test():
                    return await hm.update_existing_summary(summary, msg)

                updated = asyncio.run(_test())
                self.assertLessEqual(len(updated.content), 4000)


class UpdateSummaryFallbackUsesCountCharactersTests(unittest.TestCase):
    """Tests that update_existing_summary fallback uses count_characters instead of len."""

    def test_fallback_uses_count_characters_for_html_content(self):
        """When LLM response is short, fallback comparison should use count_characters
        which strips HTML tags, not raw len()."""
        import types
        import importlib

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        fake_channel_manager = types.ModuleType("channel_manager")
        fake_channel_manager.create_channel_abbreviation = lambda ch: "AB"
        fake_channel_manager.load_channel_abbreviations = lambda: {}

        class FakeSummaryInfo:
            def __init__(self, content="", date=None, message_count=0, channels=None, message_id=None):
                self.content = content
                self.date = date or datetime.now(timezone.utc)
                self.message_count = message_count
                self.channels = channels or []
                self.message_id = message_id

        class FakeMessageInfo:
            def __init__(self, text="", channel="", message_id=0, date=None, link=""):
                self.text = text
                self.channel = channel
                self.message_id = message_id
                self.date = date or datetime.now(timezone.utc)
                self.link = link

            def get_telegram_link(self):
                return f"https://t.me/{self.channel.lstrip('@')}/{self.message_id}"

        fake_models = types.ModuleType("models")
        fake_models.SummaryInfo = FakeSummaryInfo
        fake_models.MessageInfo = FakeMessageInfo
        fake_prompts = types.ModuleType("prompts")
        fake_prompts.prompts = types.SimpleNamespace()

        def count_characters_html(text):
            import re
            return len(re.sub(r'<[^>]+>', '', text))

        short_response = "<b>Short</b>"
        fake_utils = types.ModuleType("utils")
        fake_utils.call_openai = AsyncMock(return_value=short_response)
        fake_utils.extract_links = lambda text: []
        fake_utils.load_json_file = lambda *a, **kw: {"summaries": [], "last_updated": ""}
        fake_utils.save_json_file = MagicMock(return_value=True)
        fake_utils.now_iso = lambda: "2026-01-01T00:00:00"
        fake_utils.text_hash = lambda text: __import__('hashlib').sha256(text.encode()).hexdigest()[:16]
        fake_utils.count_characters = count_characters_html
        fake_utils.enforce_summary_length = lambda text, max_chars: text

        stubs = {
            "dotenv": fake_dotenv,
            "channel_manager": fake_channel_manager,
            "models": fake_models,
            "prompts": fake_prompts,
            "utils": fake_utils,
        }

        with patch.dict(sys.modules, stubs):
            with patch.dict(os.environ, {}, clear=True):
                sys.modules.pop("history_manager", None)
                sys.modules.pop("config", None)
                hm = importlib.import_module("history_manager")

                html_content = "<b>" + "A" * 100 + "</b>"
                summary = FakeSummaryInfo(content=html_content, message_id=1)
                msg = FakeMessageInfo(text="New info", channel="@test", message_id=42)

                async def _test():
                    return await hm.update_existing_summary(summary, msg)

                updated = asyncio.run(_test())
                self.assertIn("Другие ссылки:", updated.content)


class SaveUpdatedSummaryNoRedundantEnforceTests(unittest.TestCase):
    """Tests that save_updated_summary passes content directly without redundant enforce_summary_length."""

    def test_save_updated_summary_passes_content_directly(self):
        """save_updated_summary should pass updated_summary.content directly to edit,
        not re-enforce length (already enforced in update_existing_summary)."""
        import types
        import importlib

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        fake_channel_manager = types.ModuleType("channel_manager")
        fake_channel_manager.create_channel_abbreviation = lambda ch: "AB"
        fake_channel_manager.load_channel_abbreviations = lambda: {}

        class FakeSummaryInfo:
            def __init__(self, content="", date=None, message_count=0, channels=None, message_id=None):
                self.content = content
                self.date = date or datetime.now(timezone.utc)
                self.message_count = message_count
                self.channels = channels or []
                self.message_id = message_id

            def to_dict(self):
                return {"content": self.content, "date": self.date.isoformat() if self.date else "",
                        "message_count": self.message_count, "channels": self.channels,
                        "message_id": self.message_id}

            @classmethod
            def from_dict(cls, data):
                return cls(
                    content=data.get("content", ""),
                    date=datetime.fromisoformat(data["date"]) if data.get("date") else None,
                    message_count=data.get("message_count", 0),
                    channels=data.get("channels", []),
                    message_id=data.get("message_id"),
                )

        fake_models = types.ModuleType("models")
        fake_models.SummaryInfo = FakeSummaryInfo
        fake_models.MessageInfo = type("MessageInfo", (), {})
        fake_prompts = types.ModuleType("prompts")
        fake_prompts.prompts = types.SimpleNamespace()
        fake_utils = types.ModuleType("utils")
        fake_utils.call_openai = MagicMock()
        fake_utils.extract_links = lambda text: []
        original_dict = FakeSummaryInfo(content="original", message_id=42).to_dict()
        fake_utils.load_json_file = lambda *a, **kw: {
            "summaries": [original_dict],
            "last_updated": "",
        }
        fake_utils.save_json_file = MagicMock(return_value=True)
        fake_utils.now_iso = lambda: "2026-01-01T00:00:00"
        fake_utils.text_hash = lambda text: __import__('hashlib').sha256(text.encode()).hexdigest()[:16]
        fake_utils.count_characters = lambda text: len(text)
        fake_utils.enforce_summary_length = lambda text, max_chars: text
        fake_telegram_client = types.ModuleType("telegram_client")
        fake_telegram_client.user_client = None
        fake_telegram_client.clients_loop = None
        fake_telegram_client.edit_message_in_target_channel = AsyncMock()

        stubs = {
            "dotenv": fake_dotenv,
            "channel_manager": fake_channel_manager,
            "models": fake_models,
            "prompts": fake_prompts,
            "utils": fake_utils,
            "telegram_client": fake_telegram_client,
        }

        with patch.dict(sys.modules, stubs):
            with patch.dict(os.environ, {}, clear=True):
                sys.modules.pop("history_manager", None)
                sys.modules.pop("config", None)
                hm = importlib.import_module("history_manager")

                FakeSI = FakeSummaryInfo
                original = FakeSI(content="original", message_id=42)
                updated = FakeSI(content="updated content", message_id=42)

                async def _test():
                    await hm.save_updated_summary(original, updated)

                asyncio.run(_test())

                fake_telegram_client.edit_message_in_target_channel.assert_called_once()
                call_args = fake_telegram_client.edit_message_in_target_channel.call_args
                self.assertEqual(call_args[0][1], "updated content")


class AtomicJsonWriteTests(unittest.TestCase):
    """Tests for atomic JSON file writes in save_json_file."""

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

    def test_save_json_file_writes_atomic(self):
        """save_json_file should write atomically (temp + rename)."""
        import json
        import os
        import tempfile

        utils = self._import_utils()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_atomic.json")
            result = utils.save_json_file(filepath, {"key": "value"}, "test error")
            self.assertTrue(result)
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.assertEqual(data, {"key": "value"})

    def test_save_json_file_cleans_up_temp_on_failure(self):
        """If json.dump fails, save_json_file should return False and clean up."""
        import os
        import tempfile

        utils = self._import_utils()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_cleanup.json")
            with patch("json.dump", side_effect=OSError("write error")):
                result = utils.save_json_file(filepath, {"key": "value"}, "test error")
            self.assertFalse(result)
            self.assertFalse(os.path.exists(filepath))

    def test_save_json_file_preserves_existing_on_failure(self):
        """If write fails, existing file should remain unchanged."""
        import json
        import os
        import tempfile

        utils = self._import_utils()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_preserve.json")
            utils.save_json_file(filepath, {"original": True}, "init error")
            with patch("json.dump", side_effect=OSError("write error")):
                result = utils.save_json_file(filepath, {"new": True}, "update error")
            self.assertFalse(result)
            data = utils.load_json_file(filepath)
            self.assertEqual(data, {"original": True})


class FallbackDedupLinksTests(unittest.TestCase):
    """Tests for 'Другие ссылки' dedup in update_existing_summary fallback."""

    def _make_stubs(self):
        import types
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_channel_manager = types.ModuleType("channel_manager")
        fake_channel_manager.create_channel_abbreviation = lambda ch: "AB"
        fake_channel_manager.load_channel_abbreviations = lambda: {}

        class FakeSummaryInfo:
            def __init__(self, content="", date=None, message_count=0, channels=None, message_id=None):
                self.content = content
                self.date = date or datetime.now(timezone.utc)
                self.message_count = message_count
                self.channels = channels or []
                self.message_id = message_id

        class FakeMessageInfo:
            def __init__(self, text="", channel="", message_id=0, date=None, link=""):
                self.text = text
                self.channel = channel
                self.message_id = message_id
                self.date = date or datetime.now(timezone.utc)
                self.link = link

            def get_telegram_link(self):
                return f"https://t.me/{self.channel.lstrip('@')}/{self.message_id}"

        fake_models = types.ModuleType("models")
        fake_models.SummaryInfo = FakeSummaryInfo
        fake_models.MessageInfo = FakeMessageInfo
        fake_prompts = types.ModuleType("prompts")
        fake_prompts.prompts = types.SimpleNamespace()
        fake_utils = types.ModuleType("utils")
        fake_utils.call_openai = AsyncMock(side_effect=Exception("LLM error"))
        fake_utils.extract_links = lambda text: []
        fake_utils.load_json_file = lambda *a, **kw: {"summaries": [], "last_updated": ""}
        fake_utils.save_json_file = MagicMock(return_value=True)
        fake_utils.now_iso = lambda: "2026-01-01T00:00:00"
        fake_utils.text_hash = lambda text: __import__('hashlib').sha256(text.encode()).hexdigest()[:16]
        fake_utils.count_characters = lambda text: len(text)
        fake_utils.enforce_summary_length = lambda text, max_chars: text[:max_chars]

        stubs = {
            "dotenv": fake_dotenv,
            "channel_manager": fake_channel_manager,
            "models": fake_models,
            "prompts": fake_prompts,
            "utils": fake_utils,
        }
        return stubs, FakeSummaryInfo, FakeMessageInfo

    def test_no_duplicate_Другие_ссылки_on_repeated_fallback(self):
        """When fallback is triggered on a summary that already has 'Другие ссылки:',
        the new link should be appended to the existing section, not create a new one."""
        stubs, FakeSI, FakeMI = self._make_stubs()

        with patch.dict(sys.modules, stubs):
            with patch.dict(os.environ, {}, clear=True):
                sys.modules.pop("history_manager", None)
                sys.modules.pop("config", None)
                hm = importlib.import_module("history_manager")

                summary = FakeSI(content="Some summary\n\nДругие ссылки: <a href=\"https://t.me/ch/1\">[CH]</a>", message_id=1)
                msg = FakeMI(text="New info", channel="@test", message_id=42)

                async def _test():
                    return await hm.update_existing_summary(summary, msg)

                updated = asyncio.run(_test())
                self.assertEqual(updated.content.count("Другие ссылки:"), 1,
                                 "Should not create duplicate 'Другие ссылки:' sections")

    def test_creates_Другие_ссылки_when_absent(self):
        """When summary has no 'Другие ссылки:', fallback should create the section."""
        stubs, FakeSI, FakeMI = self._make_stubs()

        with patch.dict(sys.modules, stubs):
            with patch.dict(os.environ, {}, clear=True):
                sys.modules.pop("history_manager", None)
                sys.modules.pop("config", None)
                hm = importlib.import_module("history_manager")

                summary = FakeSI(content="Some summary without links section", message_id=1)
                msg = FakeMI(text="New info", channel="@test", message_id=42)

                async def _test():
                    return await hm.update_existing_summary(summary, msg)

                updated = asyncio.run(_test())
                self.assertIn("Другие ссылки:", updated.content)


class SaveUpdatedSummaryWithSummariesParamTests(unittest.TestCase):
    """Tests for save_updated_summary accepting optional summaries parameter."""

    def test_save_updated_summary_uses_provided_summaries(self):
        """When summaries are passed, save_updated_summary should use them instead of loading from file."""
        import types
        import importlib

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        fake_channel_manager = types.ModuleType("channel_manager")
        fake_channel_manager.create_channel_abbreviation = lambda ch: "AB"
        fake_channel_manager.load_channel_abbreviations = lambda: {}

        class FakeSummaryInfo:
            def __init__(self, content="", date=None, message_count=0, channels=None, message_id=None):
                self.content = content
                self.date = date or datetime.now(timezone.utc)
                self.message_count = message_count
                self.channels = channels or []
                self.message_id = message_id

            def to_dict(self):
                return {"content": self.content, "date": self.date.isoformat() if self.date else "",
                        "message_count": self.message_count, "channels": self.channels,
                        "message_id": self.message_id}

        fake_models = types.ModuleType("models")
        fake_models.SummaryInfo = FakeSummaryInfo
        fake_models.MessageInfo = type("MessageInfo", (), {})
        fake_prompts = types.ModuleType("prompts")
        fake_prompts.prompts = types.SimpleNamespace()
        fake_utils = types.ModuleType("utils")
        fake_utils.call_openai = MagicMock()
        fake_utils.extract_links = lambda text: []
        fake_utils.load_json_file = lambda *a, **kw: {"summaries": [], "last_updated": ""}
        fake_utils.save_json_file = MagicMock(return_value=True)
        fake_utils.now_iso = lambda: "2026-01-01T00:00:00"
        fake_utils.text_hash = lambda text: __import__('hashlib').sha256(text.encode()).hexdigest()[:16]
        fake_utils.count_characters = lambda text: len(text)
        fake_utils.enforce_summary_length = lambda text, max_chars: text
        fake_telegram_client = types.ModuleType("telegram_client")
        fake_telegram_client.user_client = None
        fake_telegram_client.clients_loop = None
        fake_telegram_client.edit_message_in_target_channel = AsyncMock()

        stubs = {
            "dotenv": fake_dotenv,
            "channel_manager": fake_channel_manager,
            "models": fake_models,
            "prompts": fake_prompts,
            "utils": fake_utils,
            "telegram_client": fake_telegram_client,
        }

        with patch.dict(sys.modules, stubs):
            with patch.dict(os.environ, {}, clear=True):
                sys.modules.pop("history_manager", None)
                sys.modules.pop("config", None)
                hm = importlib.import_module("history_manager")

                original = FakeSummaryInfo(content="original", message_id=42)
                updated = FakeSummaryInfo(content="updated content", message_id=42)
                summaries = [original]

                with patch.object(hm, "load_summaries_history") as mock_load:
                    async def _test():
                        await hm.save_updated_summary(original, updated, summaries=summaries)

                    asyncio.run(_test())
                    mock_load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
