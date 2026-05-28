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
        self.assertEqual(
            result,
            {
                "status": "ok",
                "request_id": None,
                "send_message": False,
                "save_changes": True,
                "include_today_processed_groups": True,
                "include_today_processed_messages": False,
            },
        )

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


if __name__ == "__main__":
    unittest.main()
