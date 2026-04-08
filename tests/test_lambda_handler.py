import importlib
import sys
import types
import unittest
from unittest.mock import AsyncMock, patch


def _load_lambda_handler():
    fake_summarizer = types.ModuleType("summarizer")

    async def fake_run_summarizer(**kwargs):
        return kwargs

    fake_summarizer.run_summarizer = fake_run_summarizer
    sys.modules["summarizer"] = fake_summarizer
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
        mock_run.assert_called_once_with(
            send_message=False,
            save_changes=True,
            include_today_processed_groups=True,
            include_today_processed_messages=False,
        )
        mock_asyncio_run.assert_called_once()
        mock_upload.assert_called_once_with()
        self.assertEqual(
            result,
            {
                "status": "ok",
                "send_message": False,
                "save_changes": True,
                "include_today_processed_groups": True,
                "include_today_processed_messages": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
