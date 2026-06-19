import asyncio
import importlib
import sys
import time
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


def _load_lambda_handler():
    fake_summarizer = types.ModuleType("summarizer")

    async def fake_run_summarizer(**kwargs):
        return kwargs

    class DeadlineExceededError(Exception):
        pass

    fake_summarizer.run_summarizer = fake_run_summarizer
    fake_summarizer.DeadlineExceededError = DeadlineExceededError
    fake_config = types.ModuleType("config")
    fake_config.validate_config = lambda: None
    fake_utils = types.ModuleType("utils")
    fake_utils.get_circuit_breaker_state = lambda: {"state": "closed", "failures": 0}
    fake_utils.reset_circuit_breaker = lambda: None
    fake_utils.get_token_usage = lambda: {"prompt_tokens": 0, "completion_tokens": 0}
    fake_utils.reset_token_usage = lambda: None
    sys.modules["summarizer"] = fake_summarizer
    sys.modules["config"] = fake_config
    sys.modules["utils"] = fake_utils
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


    def test_handler_returns_structured_error_on_validate_config_failure(self):
        """validate_config inside try/except should return structured error, not unhandled exception."""
        event = {"send_message": True, "save_changes": True}

        def _raise_value_error():
            raise ValueError("Missing required environment variables: FOO, BAR")

        with patch.object(self.lambda_handler.os, "chdir"), \
             patch.object(self.lambda_handler, "download_from_s3"), \
             patch.object(self.lambda_handler, "upload_to_s3"), \
             patch.dict(sys.modules, {"config": types.SimpleNamespace(validate_config=_raise_value_error)}):
            result = self.lambda_handler.handler(event, context=None)

        self.assertEqual(result["status"], "error")
        self.assertIn("Missing required", result["error"])

    def test_handler_logs_phase_timing_on_success(self):
        """Handler should log per-phase timing (download, summarizer, upload) on success."""
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

        phase_logs = [call for call in mock_log.call_args_list
                      if "download=" in str(call) and "summarizer=" in str(call) and "upload=" in str(call)]
        self.assertTrue(len(phase_logs) > 0, "Expected a per-phase timing log message")

    def test_error_type_timeout_for_deadline_exceeded(self):
        DeadlineExceededError = self.lambda_handler.DeadlineExceededError
        exc = DeadlineExceededError("approaching timeout")
        self.assertEqual(self.lambda_handler._classify_error(exc), "timeout")

    def test_error_type_config_for_value_error(self):
        exc = ValueError("Missing required environment variables")
        self.assertEqual(self.lambda_handler._classify_error(exc), "config")

    def test_error_type_connection_for_connection_error(self):
        exc = ConnectionError("refused")
        self.assertEqual(self.lambda_handler._classify_error(exc), "connection")

    def test_error_type_runtime_for_generic_exception(self):
        exc = RuntimeError("something went wrong")
        self.assertEqual(self.lambda_handler._classify_error(exc), "runtime")

    def test_handler_includes_error_type_in_error_response(self):
        event = {"send_message": True, "save_changes": True}

        async def _raise(**kwargs):
            raise RuntimeError("test error")

        with patch.object(self.lambda_handler.os, "chdir"), \
             patch.object(self.lambda_handler, "download_from_s3"), \
             patch.object(self.lambda_handler, "upload_to_s3"), \
             patch.object(self.lambda_handler, "run_summarizer", _raise):
            result = self.lambda_handler.handler(event, context=None)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "runtime")

    def test_handler_includes_error_type_config_on_value_error(self):
        event = {"send_message": True, "save_changes": True}

        def _raise_value_error():
            raise ValueError("Missing required environment variables: FOO, BAR")

        with patch.object(self.lambda_handler.os, "chdir"), \
             patch.object(self.lambda_handler, "download_from_s3"), \
             patch.object(self.lambda_handler, "upload_to_s3"), \
             patch.dict(sys.modules, {"config": types.SimpleNamespace(validate_config=_raise_value_error)}):
            result = self.lambda_handler.handler(event, context=None)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "config")


    def test_handler_logs_event_flags_in_completion_log(self):
        """Handler should include event flags in the completion log."""
        event = {"send_message": "false", "save_changes": "true"}

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

        flag_logs = [call for call in mock_log.call_args_list
                     if "send=" in str(call) and "save=" in str(call)]
        self.assertTrue(len(flag_logs) > 0, "Expected event flags in completion log")

    def test_handler_includes_circuit_breaker_state_on_success(self):
        """Handler should include circuit_breaker state dict in success response."""
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

        self.assertIn("circuit_breaker", result)
        self.assertIn("state", result["circuit_breaker"])
        self.assertIn("failures", result["circuit_breaker"])

    def test_handler_includes_circuit_breaker_state_on_error(self):
        """Handler should include circuit_breaker state dict in error response."""
        event = {"send_message": True, "save_changes": True}

        async def _raise(**kwargs):
            raise RuntimeError("test error")

        with patch.object(self.lambda_handler.os, "chdir"), \
             patch.object(self.lambda_handler, "download_from_s3"), \
             patch.object(self.lambda_handler, "upload_to_s3"), \
             patch.object(self.lambda_handler, "run_summarizer", _raise):
            result = self.lambda_handler.handler(event, context=None)

        self.assertEqual(result["status"], "error")
        self.assertIn("circuit_breaker", result)
        self.assertIn("state", result["circuit_breaker"])

    def test_handler_logs_circuit_breaker_state_in_completion_log(self):
        """Handler completion log should include circuit breaker state."""
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

        cb_logs = [call for call in mock_log.call_args_list
                   if "cb=" in str(call)]
        self.assertTrue(len(cb_logs) > 0, "Expected 'cb=' in completion log")


    def test_handler_logs_event_flags_at_start(self):
        """Handler should log parsed event flags at the start of invocation."""
        event = {"send_message": "false", "save_changes": "true"}

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

        start_flag_logs = [call for call in mock_log.call_args_list
                           if "Lambda event flags" in str(call)]
        self.assertTrue(len(start_flag_logs) > 0, "Expected 'Lambda event flags' log at start")

    def test_handler_includes_token_usage_in_success_response(self):
        """Handler should include token_usage dict in success response."""
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

        self.assertIn("token_usage", result)
        self.assertIn("prompt_tokens", result["token_usage"])
        self.assertIn("completion_tokens", result["token_usage"])

    def test_handler_includes_token_usage_in_error_response(self):
        """Handler should include token_usage dict in error response."""
        event = {"send_message": True, "save_changes": True}

        async def _raise(**kwargs):
            raise RuntimeError("test error")

        with patch.object(self.lambda_handler.os, "chdir"), \
             patch.object(self.lambda_handler, "download_from_s3"), \
             patch.object(self.lambda_handler, "upload_to_s3"), \
             patch.object(self.lambda_handler, "run_summarizer", _raise):
            result = self.lambda_handler.handler(event, context=None)

        self.assertEqual(result["status"], "error")
        self.assertIn("token_usage", result)

    def test_handler_resets_circuit_breaker_on_start(self):
        """Handler should call reset_circuit_breaker at the start of each invocation."""
        event = {"send_message": True, "save_changes": True}

        async_mock = AsyncMock()

        def _run_and_close(coro):
            coro.close()

        with patch.object(self.lambda_handler.os, "chdir"), \
             patch.object(self.lambda_handler, "download_from_s3"), \
             patch.object(self.lambda_handler, "upload_to_s3"), \
             patch.object(self.lambda_handler, "run_summarizer", async_mock), \
             patch.object(self.lambda_handler.asyncio, "run", side_effect=_run_and_close), \
             patch.object(self.lambda_handler, "reset_circuit_breaker") as mock_reset:
            self.lambda_handler.handler(event, context=None)

        mock_reset.assert_called_once()

    def test_handler_resets_token_usage_on_start(self):
        """Handler should call reset_token_usage at the start of each invocation."""
        event = {"send_message": True, "save_changes": True}

        async_mock = AsyncMock()

        def _run_and_close(coro):
            coro.close()

        with patch.object(self.lambda_handler.os, "chdir"), \
             patch.object(self.lambda_handler, "download_from_s3"), \
             patch.object(self.lambda_handler, "upload_to_s3"), \
             patch.object(self.lambda_handler, "run_summarizer", async_mock), \
             patch.object(self.lambda_handler.asyncio, "run", side_effect=_run_and_close), \
             patch.object(self.lambda_handler, "reset_token_usage") as mock_reset:
            self.lambda_handler.handler(event, context=None)

        mock_reset.assert_called_once()

    def test_handler_logs_token_usage_in_completion_log(self):
        """Handler completion log should include token counts."""
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

        token_logs = [call for call in mock_log.call_args_list
                      if "tokens=" in str(call)]
        self.assertTrue(len(token_logs) > 0, "Expected 'tokens=' in completion log")


class SummarizerGroupDeadlineTests(unittest.TestCase):
    """Tests for deadline check after group fetch in summarizer.py."""

    def test_check_deadline_between_group_fetch_and_process(self):
        """summarizer.py should have check_deadline between fetch_group_messages and
        process_messages for groups (verified by AST inspection)."""
        import ast

        with open("summarizer.py") as f:
            source = f.read()
        tree = ast.parse(source)

        source_lines = source.splitlines()
        fetch_line = None
        deadline_after_fetch = None
        process_line = None
        for i, line in enumerate(source_lines):
            if "fetch_group_messages" in line and fetch_line is None:
                fetch_line = i + 1
            if "check_deadline" in line and fetch_line is not None and deadline_after_fetch is None:
                deadline_after_fetch = i + 1
            if "process_messages" in line and "is_group" in line and fetch_line is not None:
                process_line = i + 1
                break
        self.assertIsNotNone(fetch_line, "fetch_group_messages call not found")
        self.assertIsNotNone(deadline_after_fetch, "check_deadline after fetch_group_messages not found")
        self.assertIsNotNone(process_line, "process_messages for groups not found")
        self.assertLess(fetch_line, deadline_after_fetch,
                        "check_deadline should come after fetch_group_messages")
        self.assertLess(deadline_after_fetch, process_line,
                        "check_deadline should come before process_messages for groups")


class PerSourceFetchTimingTests(unittest.TestCase):
    """Tests for per-source timing in _fetch_from_sources."""

    def test_fetch_logs_per_source_timing(self):
        """_fetch_from_sources should log time spent per source (AST check)."""
        with open("telegram_client.py") as f:
            source = f.read()

        self.assertIn("source_start", source, "_fetch_from_sources should track source_start time")
        self.assertIn("time.monotonic() - source_start", source,
                       "_fetch_from_sources should log elapsed time per source")


if __name__ == "__main__":
    unittest.main()
