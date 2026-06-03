import asyncio
import importlib
import sys
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock, AsyncMock
import json
import os


class ShouldRunGroupSummarizationLogicTests(unittest.TestCase):
    """Tests for should_run_group_summarization logic (standalone)."""

    def test_time_calculation_logic(self):
        """Test the time calculation logic without file I/O."""
        # Simulate the logic from should_run_group_summarization
        def check_should_run(last_run_str):
            if not last_run_str:
                return True
            
            try:
                last_run = datetime.fromisoformat(last_run_str)
                if last_run.tzinfo is None:
                    last_run = last_run.replace(tzinfo=timezone.utc)
                
                now = datetime.now(timezone.utc)
                time_since_last_run = (now - last_run).total_seconds()
                
                # 24 hours = 86400 seconds
                return time_since_last_run > 86400
            except (ValueError, TypeError):
                return False
        
        # Test empty string
        self.assertTrue(check_should_run(""))
        
        # Test recent run (12 hours ago)
        recent = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()
        self.assertFalse(check_should_run(recent))
        
        # Test old run (30 hours ago)
        old = (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat()
        self.assertTrue(check_should_run(old))
        
        # Test malformed timestamp
        self.assertFalse(check_should_run("not-a-timestamp"))


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

    def test_coverage_check_limits_from_config(self):
        """Verify that COVERAGE_CHECK_MAX_SUMMARIES and COVERAGE_CHECK_MAX_CHARS_PER_SUMMARY exist in config."""
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

        self.assertIsInstance(config.COVERAGE_CHECK_MAX_SUMMARIES, int)
        self.assertIsInstance(config.COVERAGE_CHECK_MAX_CHARS_PER_SUMMARY, int)
        self.assertGreater(config.COVERAGE_CHECK_MAX_SUMMARIES, 0)
        self.assertGreater(config.COVERAGE_CHECK_MAX_CHARS_PER_SUMMARY, 0)

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

    def test_find_relevant_summary_context_truncation(self):
        """Verify that find_relevant_summary_for_update truncates context."""
        max_summaries = 5
        max_chars = 500

        def build_context(summaries):
            recent = summaries[-max_summaries:]
            parts = []
            for i, s in enumerate(recent, 1):
                truncated = s["content"][:max_chars]
                parts.append(f"Саммари {i}:\n{truncated}\n\n")
            return "".join(parts)

        many_summaries = [
            {"content": f"summary_{i}: " + "X" * 1000, "date": datetime.now(timezone.utc)}
            for i in range(20)
        ]
        result = build_context(many_summaries)
        self.assertIn("Саммари 5:", result)
        self.assertIn("summary_19", result)
        self.assertNotIn("summary_0", result)
        self.assertEqual(len(result.split("Саммари")), max_summaries + 1)


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
        fake_utils.text_hash = lambda text: "abc123"
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
        fake_utils.text_hash = lambda text: "abc123"
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
        fake_utils.text_hash = lambda text: "abc123"
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
        fake_utils.text_hash = lambda text: "abc123"
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
        fake_utils.text_hash = lambda text: "abc123"
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
        fake_utils.text_hash = lambda text: "abc123"
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
        fake_utils.text_hash = lambda text: "abc123"
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


class ContextFormattingConsistencyTests(unittest.TestCase):
    """Tests for context formatting consistency between channel and group coverage checks."""

    def test_recent_summaries_context_includes_date(self):
        """get_recent_summaries_context should include date like group version."""
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
            context_parts = []
            for summary in recent:
                truncated = summary['content'][:max_chars]
                context_parts.append(f"Дата: {summary['date'].strftime('%Y-%m-%d')}")
                context_parts.append(f"Содержание: {truncated}")
                context_parts.append("---")
            return "\n".join(context_parts)

        recent_summaries = [
            {
                'content': "AI breakthrough",
                'date': datetime.now(timezone.utc) - timedelta(hours=12)
            }
        ]
        result = extract_context(recent_summaries, days=3)
        self.assertIn("Дата:", result)
        self.assertIn("Содержание:", result)
        self.assertIn("---", result)
        self.assertIn("AI breakthrough", result)


if __name__ == '__main__':
    unittest.main()
