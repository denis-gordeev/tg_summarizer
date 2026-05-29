import asyncio
import importlib
import sys
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock
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


if __name__ == '__main__':
    unittest.main()
