import asyncio
import importlib
import sys
import time
import types
import unittest
from unittest.mock import AsyncMock, patch


def _load_lambda_handler():
    fake_summarizer = types.ModuleType("summarizer")

    async def fake_run_summarizer(**kwargs):
        return kwargs

    fake_summarizer.run_summarizer = fake_run_summarizer
    fake_config = types.ModuleType("config")
    fake_config.validate_config = lambda: None
    sys.modules["summarizer"] = fake_summarizer
    sys.modules["config"] = fake_config
    sys.modules.pop("lambda_handler", None)
    return importlib.import_module("lambda_handler")


class ParseEventFlagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lambda_handler = _load_lambda_handler()

    def test_parse_event_flag_handles_string_false(self):
        result = self.lambda_handler._parse_event_flag("false", True)
        self.assertFalse(result)

    def test_parse_event_flag_handles_truthy_strings(self):
        for value in ("true", "1", "yes", "on", "Y"):
            with self.subTest(value=value):
                result = self.lambda_handler._parse_event_flag(value, False)
                self.assertTrue(result)

    def test_parse_event_flag_returns_default_for_unknown_string(self):
        result = self.lambda_handler._parse_event_flag("unexpected", False)
        self.assertFalse(result)


class HandlerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lambda_handler = _load_lambda_handler()

    def test_handler_runs_s3_sync_and_summarizer_with_parsed_flags(self):
        event = {
            "send_message": "false",
            "save_changes": "true",
            "include_today_processed_groups": "1",
            "include_today_processed_messages": "",
        }

        async_mock = AsyncMock()

        def _run_and_close(coro):
            coro.close()

        with patch.object(self.lambda_handler.os, "chdir") as mock_chdir, \
             patch.object(self.lambda_handler, "download_from_s3") as mock_download, \
             patch.object(self.lambda_handler, "upload_to_s3") as mock_upload, \
             patch.object(self.lambda_handler, "run_summarizer", async_mock) as mock_run, \
             patch.object(self.lambda_handler.asyncio, "run", side_effect=_run_and_close) as mock_asyncio_run:
            result = self.lambda_handler.handler(event, context=None)

        mock_chdir.assert_called_once_with("/tmp")
        mock_download.assert_called_once_with()
        call_kwargs = mock_run.call_args[1]
        self.assertEqual(call_kwargs["send_message"], False)
        self.assertEqual(call_kwargs["save_changes"], True)
        self.assertEqual(call_kwargs["include_today_processed_groups"], True)
        self.assertEqual(call_kwargs["include_today_processed_messages"], False)
        self.assertIsInstance(call_kwargs["_deadline"], float)
        mock_asyncio_run.assert_called_once()
        mock_upload.assert_called_once_with()
        self.assertEqual(result["status"], "ok")
        self.assertIsNone(result["request_id"])
        self.assertEqual(result["send_message"], False)
        self.assertEqual(result["save_changes"], True)
        self.assertEqual(result["include_today_processed_groups"], True)
        self.assertEqual(result["include_today_processed_messages"], False)
        self.assertIn("elapsed_seconds", result)

    def test_handler_uses_context_remaining_time_for_deadline(self):
        event = {"send_message": True, "save_changes": True}

        async_mock = AsyncMock()

        def _run_and_close(coro):
            coro.close()

        fake_context = types.SimpleNamespace(
            get_remaining_time_in_millis=lambda: 60000
        )

        with patch.object(self.lambda_handler.os, "chdir"), \
             patch.object(self.lambda_handler, "download_from_s3"), \
             patch.object(self.lambda_handler, "upload_to_s3"), \
             patch.object(self.lambda_handler, "run_summarizer", async_mock) as mock_run, \
             patch.object(self.lambda_handler.asyncio, "run", side_effect=_run_and_close):
            self.lambda_handler.handler(event, context=fake_context)

        call_kwargs = mock_run.call_args[1]
        now = time.monotonic()
        self.assertGreater(call_kwargs["_deadline"], now)
        self.assertLess(call_kwargs["_deadline"], now + 60)

    def test_handler_includes_request_id_in_response(self):
        event = {"send_message": True, "save_changes": True}

        async_mock = AsyncMock()

        def _run_and_close(coro):
            coro.close()

        fake_context = types.SimpleNamespace(
            aws_request_id="test-req-123",
            get_remaining_time_in_millis=lambda: 180000,
        )

        with patch.object(self.lambda_handler.os, "chdir"), \
             patch.object(self.lambda_handler, "download_from_s3"), \
             patch.object(self.lambda_handler, "upload_to_s3"), \
             patch.object(self.lambda_handler, "run_summarizer", async_mock), \
             patch.object(self.lambda_handler.asyncio, "run", side_effect=_run_and_close):
            result = self.lambda_handler.handler(event, context=fake_context)

        self.assertEqual(result["request_id"], "test-req-123")

    def test_handler_passes_deadline_to_run_summarizer(self):
        """Verify the handler computes a deadline and passes it as _deadline kwarg."""
        event = {"send_message": True, "save_changes": True}

        async_mock = AsyncMock()

        def _run_and_close(coro):
            coro.close()

        with patch.object(self.lambda_handler.os, "chdir"), \
             patch.object(self.lambda_handler, "download_from_s3"), \
             patch.object(self.lambda_handler, "upload_to_s3"), \
             patch.object(self.lambda_handler, "run_summarizer", async_mock) as mock_run, \
             patch.object(self.lambda_handler.asyncio, "run", side_effect=_run_and_close):
            self.lambda_handler.handler(event, context=None)

        call_kwargs = mock_run.call_args[1]
        self.assertIn("_deadline", call_kwargs)
        self.assertIsInstance(call_kwargs["_deadline"], float)

    def test_handler_logs_duration_on_success(self):
        """Handler should log completion duration on success."""
        event = {"send_message": True, "save_changes": True}

        async_mock = AsyncMock()

        def _run_and_close(coro):
            coro.close()

        with patch.object(self.lambda_handler.os, "chdir"), \
             patch.object(self.lambda_handler, "download_from_s3"), \
             patch.object(self.lambda_handler, "upload_to_s3"), \
             patch.object(self.lambda_handler, "run_summarizer", async_mock), \
             patch.object(self.lambda_handler.asyncio, "run", side_effect=_run_and_close), \
             patch.object(self.lambda_handler.logger, "info") as mock_log:
            self.lambda_handler.handler(event, context=None)

        duration_logs = [call for call in mock_log.call_args_list
                         if "completed" in str(call).lower()]
        self.assertTrue(len(duration_logs) > 0, "Expected a completion duration log message")

    def test_handler_logs_duration_on_error(self):
        """Handler should log duration when execution fails."""
        event = {"send_message": True, "save_changes": True}

        async def _raise(**kwargs):
            raise RuntimeError("test error")

        with patch.object(self.lambda_handler.os, "chdir"), \
             patch.object(self.lambda_handler, "download_from_s3"), \
             patch.object(self.lambda_handler, "upload_to_s3"), \
             patch.object(self.lambda_handler, "run_summarizer", _raise):
            result = self.lambda_handler.handler(event, context=None)

        self.assertEqual(result["status"], "error")
        self.assertIn("elapsed_seconds", result)

    def test_handler_includes_elapsed_seconds_on_success(self):
        event = {"send_message": True, "save_changes": True}

        async_mock = AsyncMock()

        def _run_and_close(coro):
            coro.close()

        with patch.object(self.lambda_handler.os, "chdir"), \
             patch.object(self.lambda_handler, "download_from_s3"), \
             patch.object(self.lambda_handler, "upload_to_s3"), \
             patch.object(self.lambda_handler, "run_summarizer", async_mock), \
             patch.object(self.lambda_handler.asyncio, "run", side_effect=_run_and_close):
            result = self.lambda_handler.handler(event, context=None)

        self.assertIn("elapsed_seconds", result)
        self.assertIsInstance(result["elapsed_seconds"], float)

    def test_handler_includes_elapsed_seconds_on_error(self):
        event = {"send_message": True, "save_changes": True}

        async def _raise(**kwargs):
            raise RuntimeError("test error")

        with patch.object(self.lambda_handler.os, "chdir"), \
             patch.object(self.lambda_handler, "download_from_s3"), \
             patch.object(self.lambda_handler, "upload_to_s3"), \
             patch.object(self.lambda_handler, "run_summarizer", _raise):
            result = self.lambda_handler.handler(event, context=None)

        self.assertIn("elapsed_seconds", result)
        self.assertIsInstance(result["elapsed_seconds"], float)


if __name__ == "__main__":
    unittest.main()
