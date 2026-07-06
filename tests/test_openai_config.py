import importlib
import os
import sys
import time as _time
import types
import unittest
from unittest.mock import Mock, MagicMock, patch


REQUIRED_ENV = {
    "TELEGRAM_API_ID": "1",
    "TELEGRAM_API_HASH": "hash",
    "TELEGRAM_BOT_TOKEN": "token",
    "TARGET_CHANNEL": "@target",
    "OPENAI_API_KEY": "test-key",
}


def _reload_module(name: str):
    sys.modules.pop(name, None)
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda: None
    fake_openai = types.ModuleType("openai")

    class FakeOpenAIError(Exception):
        pass

    class FakeOpenAI:
        def __init__(self, api_key, **kwargs):
            self.api_key = api_key
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=Mock())
            )

    fake_openai.OpenAI = FakeOpenAI
    fake_openai.AsyncOpenAI = FakeOpenAI
    fake_openai.APIError = FakeOpenAIError
    fake_openai.RateLimitError = type("RateLimitError", (FakeOpenAIError,), {"status_code": None})
    fake_openai.APIConnectionError = type("APIConnectionError", (FakeOpenAIError,), {})

    with patch.dict(sys.modules, {"dotenv": fake_dotenv, "openai": fake_openai}):
        return importlib.import_module(name)


class ConfigTests(unittest.TestCase):
    def test_config_reads_openai_overrides_from_env(self):
        env = {
            **REQUIRED_ENV,
            "OPENAI_MODEL": "gpt-4o-mini",
            "OPENAI_DEFAULT_MAX_TOKENS": "123",
            "OPENAI_CHANNEL_SUMMARY_MAX_TOKENS": "456",
            "OPENAI_GROUP_SUMMARY_MAX_TOKENS": "789",
        }

        with patch.dict(os.environ, env, clear=True):
            config = _reload_module("config")

        self.assertEqual(config.OPENAI_MODEL, "gpt-4o-mini")
        self.assertEqual(config.OPENAI_DEFAULT_MAX_TOKENS, 123)
        self.assertEqual(config.OPENAI_CHANNEL_SUMMARY_MAX_TOKENS, 456)
        self.assertEqual(config.OPENAI_GROUP_SUMMARY_MAX_TOKENS, 789)

    def test_config_openai_model_default_is_gpt41_nano(self):
        with patch.dict(os.environ, REQUIRED_ENV, clear=True):
            config = _reload_module("config")
        self.assertEqual(config.OPENAI_MODEL, "gpt-4.1-nano")

    def test_config_rejects_non_positive_token_limits(self):
        env = {
            **REQUIRED_ENV,
            "OPENAI_DEFAULT_MAX_TOKENS": "0",
        }

        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENAI_DEFAULT_MAX_TOKENS"):
                _reload_module("config")

    def test_config_does_not_raise_on_import_without_env(self):
        with patch.dict(os.environ, {}, clear=True):
            config = _reload_module("config")
        self.assertIsNone(config.API_ID)
        self.assertIsNone(config.OPENAI_API_KEY)

    def test_validate_config_raises_for_missing_vars(self):
        with patch.dict(os.environ, {}, clear=True):
            config = _reload_module("config")
        with self.assertRaisesRegex(ValueError, "Missing required environment variables"):
            config.validate_config()

    def test_validate_config_passes_when_all_required_set(self):
        with patch.dict(os.environ, REQUIRED_ENV, clear=True):
            config = _reload_module("config")
        config.validate_config()

    def test_config_reads_request_timeout_from_env(self):
        env = {
            **REQUIRED_ENV,
            "OPENAI_REQUEST_TIMEOUT": "45",
        }
        with patch.dict(os.environ, env, clear=True):
            config = _reload_module("config")
        self.assertEqual(config.OPENAI_REQUEST_TIMEOUT, 45)

    def test_config_reads_nlp_check_max_input_chars_from_env(self):
        env = {
            **REQUIRED_ENV,
            "NLP_CHECK_MAX_INPUT_CHARS": "3000",
        }
        with patch.dict(os.environ, env, clear=True):
            config = _reload_module("config")
        self.assertEqual(config.NLP_CHECK_MAX_INPUT_CHARS, 3000)

    def test_config_reads_max_messages_per_source_from_env(self):
        env = {
            **REQUIRED_ENV,
            "MAX_MESSAGES_PER_SOURCE": "50",
        }
        with patch.dict(os.environ, env, clear=True):
            config = _reload_module("config")
        self.assertEqual(config.MAX_MESSAGES_PER_SOURCE, 50)

    def test_config_reads_update_summary_max_tokens_from_env(self):
        env = {
            **REQUIRED_ENV,
            "UPDATE_SUMMARY_MAX_TOKENS": "750",
        }
        with patch.dict(os.environ, env, clear=True):
            config = _reload_module("config")
        self.assertEqual(config.UPDATE_SUMMARY_MAX_TOKENS, 750)

    def test_config_update_summary_max_tokens_default(self):
        with patch.dict(os.environ, REQUIRED_ENV, clear=True):
            config = _reload_module("config")
        self.assertEqual(config.UPDATE_SUMMARY_MAX_TOKENS, 1000)

    def test_config_reads_summary_temperature_from_env(self):
        env = {
            **REQUIRED_ENV,
            "OPENAI_SUMMARY_TEMPERATURE": "0.5",
        }
        with patch.dict(os.environ, env, clear=True):
            config = _reload_module("config")
        self.assertAlmostEqual(config.OPENAI_SUMMARY_TEMPERATURE, 0.5)

    def test_config_summary_temperature_default(self):
        with patch.dict(os.environ, REQUIRED_ENV, clear=True):
            config = _reload_module("config")
        self.assertAlmostEqual(config.OPENAI_SUMMARY_TEMPERATURE, 0.1)

    def test_config_summary_temperature_rejects_invalid(self):
        env = {
            **REQUIRED_ENV,
            "OPENAI_SUMMARY_TEMPERATURE": "not_a_number",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENAI_SUMMARY_TEMPERATURE"):
                _reload_module("config")

    def test_config_summary_temperature_rejects_out_of_range(self):
        env = {
            **REQUIRED_ENV,
            "OPENAI_SUMMARY_TEMPERATURE": "3.0",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENAI_SUMMARY_TEMPERATURE"):
                _reload_module("config")

    def test_config_similarity_llm_upper_from_env(self):
        env = {
            **REQUIRED_ENV,
            "SIMILARITY_LLM_UPPER": "0.99",
        }
        with patch.dict(os.environ, env, clear=True):
            config = _reload_module("config")
        self.assertAlmostEqual(config.SIMILARITY_LLM_UPPER, 0.99)

    def test_config_similarity_llm_upper_default(self):
        with patch.dict(os.environ, REQUIRED_ENV, clear=True):
            config = _reload_module("config")
        self.assertAlmostEqual(config.SIMILARITY_LLM_UPPER, 0.95)

    def test_config_similarity_llm_upper_rejects_out_of_range(self):
        env = {
            **REQUIRED_ENV,
            "SIMILARITY_LLM_UPPER": "1.5",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "SIMILARITY_LLM_UPPER"):
                _reload_module("config")


class SsmSecretResolutionTests(unittest.TestCase):
    def test_get_secret_prefers_ssm_over_env(self):
        with patch.dict(os.environ, {
            **REQUIRED_ENV,
            "OPENAI_API_KEY_SSM_PATH": "/test/openai-key",
        }, clear=True):
            config = _reload_module("config")
            with patch.object(config, "_get_ssm_param", return_value="ssm-key-value"):
                result = config._get_secret("OPENAI_API_KEY", "OPENAI_API_KEY_SSM_PATH")
                self.assertEqual(result, "ssm-key-value")

    def test_get_secret_falls_back_to_env_when_ssm_empty(self):
        with patch.dict(os.environ, {
            **REQUIRED_ENV,
            "OPENAI_API_KEY_SSM_PATH": "",
        }, clear=True):
            config = _reload_module("config")
            result = config._get_secret("OPENAI_API_KEY", "OPENAI_API_KEY_SSM_PATH")
            self.assertEqual(result, "test-key")

    def test_get_secret_falls_back_to_env_when_ssm_fails(self):
        with patch.dict(os.environ, {
            **REQUIRED_ENV,
            "OPENAI_API_KEY_SSM_PATH": "/test/openai-key",
        }, clear=True):
            config = _reload_module("config")
            with patch.object(config, "_get_ssm_param", return_value=None):
                result = config._get_secret("OPENAI_API_KEY", "OPENAI_API_KEY_SSM_PATH")
                self.assertEqual(result, "test-key")


class CallOpenAITests(unittest.IsolatedAsyncioTestCase):
    async def test_call_openai_uses_async_client(self):
        """Verify that call_openai uses AsyncOpenAI (not sync OpenAI)."""
        import importlib
        import sys
        import types
        from unittest.mock import AsyncMock, patch

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_openai = types.ModuleType("openai")

        class FakeAsyncOpenAI:
            def __init__(self, api_key, **kwargs):
                self.api_key = api_key
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(
                        create=AsyncMock(return_value=type(
                            "Response",
                            (),
                            {
                                "choices": [
                                    type(
                                        "Choice",
                                        (),
                                        {"message": type("Message", (), {"content": "async done"})()},
                                    )()
                                ]
                            },
                        )())
                    )
                )

        class FakeOpenAIError(Exception):
            pass

        fake_openai.AsyncOpenAI = FakeAsyncOpenAI
        fake_openai.OpenAI = FakeAsyncOpenAI
        fake_openai.APIError = FakeOpenAIError
        fake_openai.RateLimitError = type("RateLimitError", (FakeOpenAIError,), {"status_code": None})
        fake_openai.APIConnectionError = type("APIConnectionError", (FakeOpenAIError,), {})

        with patch.dict(sys.modules, {"dotenv": fake_dotenv, "openai": fake_openai}):
            with patch.dict(os.environ, {
                **REQUIRED_ENV,
                "OPENAI_MODEL": "gpt-4o-mini",
                "OPENAI_DEFAULT_MAX_TOKENS": "321",
            }, clear=True):
                sys.modules.pop("config", None)
                sys.modules.pop("utils", None)
                config = importlib.import_module("config")
                utils = importlib.import_module("utils")

        result = await utils.call_openai("system", "user")
        self.assertEqual(result, "async done")

    async def test_call_openai_passes_timeout_to_client(self):
        """Verify that call_openai passes OPENAI_REQUEST_TIMEOUT to AsyncOpenAI."""
        import importlib
        import sys
        import types
        from unittest.mock import AsyncMock, patch

        captured_kwargs = {}

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_openai = types.ModuleType("openai")

        class FakeAsyncOpenAITracking:
            def __init__(self, api_key, **kwargs):
                self.api_key = api_key
                captured_kwargs.update(kwargs)
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(
                        create=AsyncMock(return_value=type(
                            "Response", (),
                            {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()]},
                        )())
                    )
                )

        class FakeOpenAIError(Exception):
            pass

        fake_openai.AsyncOpenAI = FakeAsyncOpenAITracking
        fake_openai.OpenAI = FakeAsyncOpenAITracking
        fake_openai.APIError = FakeOpenAIError
        fake_openai.RateLimitError = type("RateLimitError", (FakeOpenAIError,), {"status_code": None})
        fake_openai.APIConnectionError = type("APIConnectionError", (FakeOpenAIError,), {})

        with patch.dict(sys.modules, {"dotenv": fake_dotenv, "openai": fake_openai}):
            with patch.dict(os.environ, {
                **REQUIRED_ENV,
                "OPENAI_REQUEST_TIMEOUT": "25",
            }, clear=True):
                sys.modules.pop("config", None)
                sys.modules.pop("utils", None)
                config = importlib.import_module("config")
                utils = importlib.import_module("utils")

        result = await utils.call_openai("system", "user")
        self.assertEqual(result, "ok")
        self.assertAlmostEqual(captured_kwargs.get("timeout"), 25.0)

    async def test_call_openai_passes_temperature_to_api(self):
        """Verify that call_openai passes temperature kwarg to the API create call."""
        import importlib
        import sys
        import types
        from unittest.mock import AsyncMock, patch

        captured_create_kwargs = {}

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_openai = types.ModuleType("openai")

        class FakeAsyncOpenAI:
            def __init__(self, api_key, **kwargs):
                self.api_key = api_key

                async def _create(**kwargs):
                    captured_create_kwargs.update(kwargs)
                    return type(
                        "Response", (),
                        {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()]},
                    )()

                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(create=_create)
                )

        class FakeOpenAIError(Exception):
            pass

        fake_openai.AsyncOpenAI = FakeAsyncOpenAI
        fake_openai.OpenAI = FakeAsyncOpenAI
        fake_openai.APIError = FakeOpenAIError
        fake_openai.RateLimitError = type("RateLimitError", (FakeOpenAIError,), {"status_code": None})
        fake_openai.APIConnectionError = type("APIConnectionError", (FakeOpenAIError,), {})

        with patch.dict(sys.modules, {"dotenv": fake_dotenv, "openai": fake_openai}):
            with patch.dict(os.environ, {
                **REQUIRED_ENV,
            }, clear=True):
                sys.modules.pop("config", None)
                sys.modules.pop("utils", None)
                config = importlib.import_module("config")
                utils = importlib.import_module("utils")

        result = await utils.call_openai("system", "user", temperature=0)
        self.assertEqual(result, "ok")
        self.assertEqual(captured_create_kwargs.get("temperature"), 0)

    async def test_call_openai_omits_temperature_when_none(self):
        """Verify that call_openai omits temperature when not specified."""
        import importlib
        import sys
        import types
        from unittest.mock import AsyncMock, patch

        captured_create_kwargs = {}

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_openai = types.ModuleType("openai")

        class FakeAsyncOpenAI:
            def __init__(self, api_key, **kwargs):
                self.api_key = api_key

                async def _create(**kwargs):
                    captured_create_kwargs.update(kwargs)
                    return type(
                        "Response", (),
                        {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()]},
                    )()

                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(create=_create)
                )

        class FakeOpenAIError(Exception):
            pass

        fake_openai.AsyncOpenAI = FakeAsyncOpenAI
        fake_openai.OpenAI = FakeAsyncOpenAI
        fake_openai.APIError = FakeOpenAIError
        fake_openai.RateLimitError = type("RateLimitError", (FakeOpenAIError,), {"status_code": None})
        fake_openai.APIConnectionError = type("APIConnectionError", (FakeOpenAIError,), {})

        with patch.dict(sys.modules, {"dotenv": fake_dotenv, "openai": fake_openai}):
            with patch.dict(os.environ, {
                **REQUIRED_ENV,
            }, clear=True):
                sys.modules.pop("config", None)
                sys.modules.pop("utils", None)
                config = importlib.import_module("config")
                utils = importlib.import_module("utils")

        result = await utils.call_openai("system", "user")
        self.assertEqual(result, "ok")
        self.assertNotIn("temperature", captured_create_kwargs)

    async def test_call_openai_returns_empty_when_api_key_missing(self):
        """call_openai should return '' early when OPENAI_API_KEY is not set."""
        import importlib
        import sys
        import types
        from unittest.mock import MagicMock, patch

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_openai = types.ModuleType("openai")

        class FakeOpenAIError(Exception):
            pass

        fake_openai.AsyncOpenAI = MagicMock()
        fake_openai.APIError = FakeOpenAIError
        fake_openai.RateLimitError = type("RateLimitError", (FakeOpenAIError,), {"status_code": None})
        fake_openai.APIConnectionError = type("APIConnectionError", (FakeOpenAIError,), {})

        with patch.dict(sys.modules, {"dotenv": fake_dotenv, "openai": fake_openai}):
            with patch.dict(os.environ, {}, clear=True):
                sys.modules.pop("config", None)
                sys.modules.pop("utils", None)
                config = importlib.import_module("config")
                utils = importlib.import_module("utils")

        result = await utils.call_openai("system", "user")
        self.assertEqual(result, "")
        fake_openai.AsyncOpenAI.assert_not_called()


class NewConfigConstantsTests(unittest.TestCase):
    """Tests for SUMMARY_MAX_INPUT_CHARS_PER_MESSAGE and NLP_CONCURRENT_CHECKS config."""

    def test_config_reads_summary_max_input_chars_from_env(self):
        import importlib
        import sys
        import types
        from unittest.mock import patch

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, {
                **REQUIRED_ENV,
                "SUMMARY_MAX_INPUT_CHARS_PER_MESSAGE": "5000",
            }, clear=True):
                sys.modules.pop("config", None)
                config = importlib.import_module("config")
                self.assertEqual(config.SUMMARY_MAX_INPUT_CHARS_PER_MESSAGE, 5000)

    def test_config_summary_max_input_chars_default(self):
        import importlib
        import sys
        import types
        from unittest.mock import patch

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, REQUIRED_ENV, clear=True):
                sys.modules.pop("config", None)
                config = importlib.import_module("config")
                self.assertEqual(config.SUMMARY_MAX_INPUT_CHARS_PER_MESSAGE, 2000)

    def test_config_reads_nlp_concurrent_checks_from_env(self):
        import importlib
        import sys
        import types
        from unittest.mock import patch

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, {
                **REQUIRED_ENV,
                "NLP_CONCURRENT_CHECKS": "10",
            }, clear=True):
                sys.modules.pop("config", None)
                config = importlib.import_module("config")
                self.assertEqual(config.NLP_CONCURRENT_CHECKS, 10)

    def test_config_nlp_concurrent_checks_default(self):
        import importlib
        import sys
        import types
        from unittest.mock import patch

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, REQUIRED_ENV, clear=True):
                sys.modules.pop("config", None)
                config = importlib.import_module("config")
                self.assertEqual(config.NLP_CONCURRENT_CHECKS, 5)


class OpenAITokenUsageLoggingTests(unittest.IsolatedAsyncioTestCase):
    async def test_call_openai_logs_token_usage(self):
        import importlib
        import sys
        import types
        from unittest.mock import AsyncMock, MagicMock, patch

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_openai = types.ModuleType("openai")

        fake_usage = MagicMock()
        fake_usage.prompt_tokens = 100
        fake_usage.completion_tokens = 50
        fake_usage.total_tokens = 150

        class FakeAsyncOpenAI:
            def __init__(self, api_key, **kwargs):
                self.api_key = api_key
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(
                        create=AsyncMock(return_value=type(
                            "Response", (),
                            {
                                "choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()],
                                "usage": fake_usage,
                            },
                        )())
                    )
                )

        class FakeOpenAIError(Exception):
            pass

        fake_openai.AsyncOpenAI = FakeAsyncOpenAI
        fake_openai.OpenAI = FakeAsyncOpenAI
        fake_openai.APIError = FakeOpenAIError
        fake_openai.RateLimitError = type("RateLimitError", (FakeOpenAIError,), {"status_code": None})
        fake_openai.APIConnectionError = type("APIConnectionError", (FakeOpenAIError,), {})

        with patch.dict(sys.modules, {"dotenv": fake_dotenv, "openai": fake_openai}):
            with patch.dict(os.environ, REQUIRED_ENV, clear=True):
                sys.modules.pop("config", None)
                sys.modules.pop("utils", None)
                config = importlib.import_module("config")
                utils = importlib.import_module("utils")

        with patch.object(utils.logger, "info") as mock_log:
            result = await utils.call_openai("system", "user")
            self.assertEqual(result, "ok")
            usage_logs = [call for call in mock_log.call_args_list
                          if "usage" in str(call).lower() or "prompt" in str(call).lower()]
            self.assertTrue(len(usage_logs) > 0, "Expected a token usage log message")

    async def test_call_openai_logs_latency(self):
        """call_openai should log response latency."""
        import importlib
        import sys
        import types
        from unittest.mock import AsyncMock, MagicMock, patch

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_openai = types.ModuleType("openai")

        fake_usage = MagicMock()
        fake_usage.prompt_tokens = 10
        fake_usage.completion_tokens = 5
        fake_usage.total_tokens = 15

        class FakeAsyncOpenAI:
            def __init__(self, api_key, **kwargs):
                self.api_key = api_key
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(
                        create=AsyncMock(return_value=type(
                            "Response", (),
                            {
                                "choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()],
                                "usage": fake_usage,
                            },
                        )())
                    )
                )

        class FakeOpenAIError(Exception):
            pass

        fake_openai.AsyncOpenAI = FakeAsyncOpenAI
        fake_openai.OpenAI = FakeAsyncOpenAI
        fake_openai.APIError = FakeOpenAIError
        fake_openai.RateLimitError = type("RateLimitError", (FakeOpenAIError,), {"status_code": None})
        fake_openai.APIConnectionError = type("APIConnectionError", (FakeOpenAIError,), {})

        with patch.dict(sys.modules, {"dotenv": fake_dotenv, "openai": fake_openai}):
            with patch.dict(os.environ, REQUIRED_ENV, clear=True):
                sys.modules.pop("config", None)
                sys.modules.pop("utils", None)
                config = importlib.import_module("config")
                utils = importlib.import_module("utils")

        with patch.object(utils.logger, "info") as mock_log:
            result = await utils.call_openai("system", "user")
            self.assertEqual(result, "ok")
            latency_logs = [call for call in mock_log.call_args_list
                            if "latency" in str(call).lower()]
            self.assertTrue(len(latency_logs) > 0, "Expected a latency log message")


class UpdateSummaryMaxInputCharsTests(unittest.TestCase):
    """Tests for UPDATE_SUMMARY_MAX_INPUT_CHARS config."""

    def test_config_reads_update_summary_max_input_chars_from_env(self):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, {
                **REQUIRED_ENV,
                "UPDATE_SUMMARY_MAX_INPUT_CHARS": "3000",
            }, clear=True):
                sys.modules.pop("config", None)
                config = importlib.import_module("config")
                self.assertEqual(config.UPDATE_SUMMARY_MAX_INPUT_CHARS, 3000)

    def test_config_update_summary_max_input_chars_default(self):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, REQUIRED_ENV, clear=True):
                sys.modules.pop("config", None)
                config = importlib.import_module("config")
                self.assertEqual(config.UPDATE_SUMMARY_MAX_INPUT_CHARS, 1000)


class NlpAdKeywordsConfigTests(unittest.TestCase):
    """Tests for NLP_AD_KEYWORDS config."""

    def test_nlp_ad_keywords_is_list(self):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, REQUIRED_ENV, clear=True):
                sys.modules.pop("config", None)
                config = importlib.import_module("config")
                self.assertIsInstance(config.NLP_AD_KEYWORDS, list)
                self.assertTrue(len(config.NLP_AD_KEYWORDS) > 0)

    def test_nlp_ad_keywords_contains_course(self):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, REQUIRED_ENV, clear=True):
                sys.modules.pop("config", None)
                config = importlib.import_module("config")
                self.assertIn("курс", config.NLP_AD_KEYWORDS)


class ConfigValidationConsistencyTests(unittest.TestCase):
    """Tests that _get_int_env validates all integer configs."""

    def test_nlp_check_max_input_chars_rejects_zero(self):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, {
                **REQUIRED_ENV,
                "NLP_CHECK_MAX_INPUT_CHARS": "0",
            }, clear=True):
                sys.modules.pop("config", None)
                with self.assertRaises(ValueError):
                    importlib.import_module("config")

    def test_max_messages_per_source_rejects_negative(self):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, {
                **REQUIRED_ENV,
                "MAX_MESSAGES_PER_SOURCE": "-5",
            }, clear=True):
                sys.modules.pop("config", None)
                with self.assertRaises(ValueError):
                    importlib.import_module("config")

    def test_channel_summary_max_tokens_default_is_1500(self):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, REQUIRED_ENV, clear=True):
                sys.modules.pop("config", None)
                config = importlib.import_module("config")
                self.assertEqual(config.OPENAI_CHANNEL_SUMMARY_MAX_TOKENS, 1500)

    def test_channel_summary_max_tokens_from_env(self):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, {**REQUIRED_ENV, "OPENAI_CHANNEL_SUMMARY_MAX_TOKENS": "2000"}, clear=True):
                sys.modules.pop("config", None)
                config = importlib.import_module("config")
                self.assertEqual(config.OPENAI_CHANNEL_SUMMARY_MAX_TOKENS, 2000)

    def test_config_reads_restore_timeout_sec_from_env(self):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, {**REQUIRED_ENV, "RESTORE_TIMEOUT_SEC": "60"}, clear=True):
                sys.modules.pop("config", None)
                config = importlib.import_module("config")
                self.assertEqual(config.RESTORE_TIMEOUT_SEC, 60)

    def test_config_restore_timeout_sec_default(self):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, REQUIRED_ENV, clear=True):
                sys.modules.pop("config", None)
                config = importlib.import_module("config")
                self.assertEqual(config.RESTORE_TIMEOUT_SEC, 30)

    def test_config_restore_timeout_sec_rejects_zero(self):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, {**REQUIRED_ENV, "RESTORE_TIMEOUT_SEC": "0"}, clear=True):
                sys.modules.pop("config", None)
                with self.assertRaises(ValueError):
                    importlib.import_module("config")


    async def test_call_openai_emits_emf_metric(self):
        """call_openai should emit EMF metric JSON to stdout on success."""
        import importlib
        import sys
        import types
        import io
        from unittest.mock import AsyncMock, MagicMock, patch

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_openai = types.ModuleType("openai")

        fake_usage = MagicMock()
        fake_usage.prompt_tokens = 10
        fake_usage.completion_tokens = 5
        fake_usage.total_tokens = 15

        class FakeAsyncOpenAI:
            def __init__(self, api_key, **kwargs):
                self.api_key = api_key
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(
                        create=AsyncMock(return_value=type(
                            "Response", (),
                            {
                                "choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()],
                                "usage": fake_usage,
                            },
                        )())
                    )
                )

        class FakeOpenAIError(Exception):
            pass

        fake_openai.AsyncOpenAI = FakeAsyncOpenAI
        fake_openai.OpenAI = FakeAsyncOpenAI
        fake_openai.APIError = FakeOpenAIError
        fake_openai.RateLimitError = type("RateLimitError", (FakeOpenAIError,), {"status_code": None})
        fake_openai.APIConnectionError = type("APIConnectionError", (FakeOpenAIError,), {})

        with patch.dict(sys.modules, {"dotenv": fake_dotenv, "openai": fake_openai}):
            with patch.dict(os.environ, REQUIRED_ENV, clear=True):
                sys.modules.pop("config", None)
                sys.modules.pop("utils", None)
                config = importlib.import_module("config")
                utils = importlib.import_module("utils")

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            result = await utils.call_openai("system", "user")
            self.assertEqual(result, "ok")

        output = captured.getvalue()
        self.assertIn("_aws", output)
        self.assertIn("CloudWatchMetrics", output)
        self.assertIn("Latency", output)


class NlpMinTextLengthConfigTests(unittest.TestCase):
    """Tests for NLP_MIN_TEXT_LENGTH config."""

    def test_config_reads_nlp_min_text_length_from_env(self):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, {
                **REQUIRED_ENV,
                "NLP_MIN_TEXT_LENGTH": "200",
            }, clear=True):
                sys.modules.pop("config", None)
                config = importlib.import_module("config")
                self.assertEqual(config.NLP_MIN_TEXT_LENGTH, 200)

    def test_config_nlp_min_text_length_default(self):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, REQUIRED_ENV, clear=True):
                sys.modules.pop("config", None)
                config = importlib.import_module("config")
                self.assertEqual(config.NLP_MIN_TEXT_LENGTH, 100)

    def test_config_nlp_min_text_length_rejects_zero(self):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, {
                **REQUIRED_ENV,
                "NLP_MIN_TEXT_LENGTH": "0",
            }, clear=True):
                sys.modules.pop("config", None)
                with self.assertRaises(ValueError):
                    importlib.import_module("config")


class NlpAdKeywordsNoRedundancyTests(unittest.TestCase):
    """Tests that NLP_AD_KEYWORDS has no redundant entries."""

    def test_no_substring_redundancy_in_ad_keywords(self):
        """No keyword should be a substring of another keyword that appears
        earlier in the list, since the regex alternation matches left-to-right."""
        import importlib
        import sys
        import types
        from unittest.mock import patch

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, REQUIRED_ENV, clear=True):
                sys.modules.pop("config", None)
                config = importlib.import_module("config")

        keywords = config.NLP_AD_KEYWORDS
        for i, kw in enumerate(keywords):
            for j, earlier in enumerate(keywords[:i]):
                self.assertNotIn(
                    earlier.lower(), kw.lower(),
                    f"Keyword '{kw}' at index {i} is redundant — '{earlier}' at index {j} is a substring"
                )


class EmfTokenMetricsTests(unittest.TestCase):
    """Tests for EMF token metrics in call_openai output."""

    async def _run_call_openai_with_fake(self, env_extra=None):
        import importlib
        import sys
        import types
        import io
        from unittest.mock import AsyncMock, MagicMock, patch

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_openai = types.ModuleType("openai")

        fake_usage = MagicMock()
        fake_usage.prompt_tokens = 42
        fake_usage.completion_tokens = 7
        fake_usage.total_tokens = 49

        class FakeAsyncOpenAI:
            def __init__(self, api_key, **kwargs):
                self.api_key = api_key
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(
                        create=AsyncMock(return_value=type(
                            "Response", (),
                            {
                                "choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()],
                                "usage": fake_usage,
                            },
                        )())
                    )
                )

        class FakeOpenAIError(Exception):
            pass

        fake_openai.AsyncOpenAI = FakeAsyncOpenAI
        fake_openai.OpenAI = FakeAsyncOpenAI
        fake_openai.APIError = FakeOpenAIError
        fake_openai.RateLimitError = type("RateLimitError", (FakeOpenAIError,), {"status_code": None})
        fake_openai.APIConnectionError = type("APIConnectionError", (FakeOpenAIError,), {})

        env = {**REQUIRED_ENV}
        if env_extra:
            env.update(env_extra)

        with patch.dict(sys.modules, {"dotenv": fake_dotenv, "openai": fake_openai}):
            with patch.dict(os.environ, env, clear=True):
                sys.modules.pop("config", None)
                sys.modules.pop("utils", None)
                config = importlib.import_module("config")
                utils = importlib.import_module("utils")

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            await utils.call_openai("system", "user")

        return captured.getvalue()

    def test_emf_includes_prompt_tokens(self):
        import asyncio
        output = asyncio.run(self._run_call_openai_with_fake())
        self.assertIn("PromptTokens", output)
        self.assertIn('"PromptTokens":42', output.replace(" ", ""))

    def test_emf_includes_completion_tokens(self):
        import asyncio
        output = asyncio.run(self._run_call_openai_with_fake())
        self.assertIn("CompletionTokens", output)
        self.assertIn('"CompletionTokens":7', output.replace(" ", ""))


class MaxCoveredMessageUpdatesConfigTests(unittest.TestCase):
    """Tests for MAX_COVERED_MESSAGE_UPDATES config."""

    def test_config_max_covered_message_updates_default(self):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, REQUIRED_ENV, clear=True):
                sys.modules.pop("config", None)
                config = importlib.import_module("config")
                self.assertEqual(config.MAX_COVERED_MESSAGE_UPDATES, 5)

    def test_config_reads_max_covered_message_updates_from_env(self):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, {
                **REQUIRED_ENV,
                "MAX_COVERED_MESSAGE_UPDATES": "10",
            }, clear=True):
                sys.modules.pop("config", None)
                config = importlib.import_module("config")
                self.assertEqual(config.MAX_COVERED_MESSAGE_UPDATES, 10)

    def test_config_max_covered_message_updates_rejects_zero(self):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, {
                **REQUIRED_ENV,
                "MAX_COVERED_MESSAGE_UPDATES": "0",
            }, clear=True):
                sys.modules.pop("config", None)
                with self.assertRaises(ValueError):
                    importlib.import_module("config")


class CoverageCheckMaxInputCharsConfigTests(unittest.TestCase):
    """Tests for COVERAGE_CHECK_MAX_INPUT_CHARS config."""

    def test_config_coverage_check_max_input_chars_default(self):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, REQUIRED_ENV, clear=True):
                sys.modules.pop("config", None)
                config = importlib.import_module("config")
                self.assertEqual(config.COVERAGE_CHECK_MAX_INPUT_CHARS, 500)

    def test_config_reads_coverage_check_max_input_chars_from_env(self):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, {
                **REQUIRED_ENV,
                "COVERAGE_CHECK_MAX_INPUT_CHARS": "3000",
            }, clear=True):
                sys.modules.pop("config", None)
                config = importlib.import_module("config")
                self.assertEqual(config.COVERAGE_CHECK_MAX_INPUT_CHARS, 3000)

    def test_config_coverage_check_max_input_chars_rejects_zero(self):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, {
                **REQUIRED_ENV,
                "COVERAGE_CHECK_MAX_INPUT_CHARS": "0",
            }, clear=True):
                sys.modules.pop("config", None)
                with self.assertRaises(ValueError):
                    importlib.import_module("config")


class FetchExaminedMultiplierConfigTests(unittest.TestCase):
    """Tests for FETCH_EXAMINED_MULTIPLIER config."""

    def test_config_fetch_examined_multiplier_default(self):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, REQUIRED_ENV, clear=True):
                sys.modules.pop("config", None)
                config = importlib.import_module("config")
                self.assertEqual(config.FETCH_EXAMINED_MULTIPLIER, 3)

    def test_config_reads_fetch_examined_multiplier_from_env(self):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, {
                **REQUIRED_ENV,
                "FETCH_EXAMINED_MULTIPLIER": "5",
            }, clear=True):
                sys.modules.pop("config", None)
                config = importlib.import_module("config")
                self.assertEqual(config.FETCH_EXAMINED_MULTIPLIER, 5)

    def test_config_fetch_examined_multiplier_rejects_zero(self):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, {
                **REQUIRED_ENV,
                "FETCH_EXAMINED_MULTIPLIER": "0",
            }, clear=True):
                sys.modules.pop("config", None)
                with self.assertRaises(ValueError):
                    importlib.import_module("config")


class EmfCostMetricTests(unittest.TestCase):
    """Tests for EMF estimated cost metric in call_openai output."""

    async def _run_call_openai_with_fake(self, env_extra=None):
        import importlib
        import sys
        import types
        import io
        from unittest.mock import AsyncMock, MagicMock, patch

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_openai = types.ModuleType("openai")

        fake_usage = MagicMock()
        fake_usage.prompt_tokens = 100
        fake_usage.completion_tokens = 50
        fake_usage.total_tokens = 150

        class FakeAsyncOpenAI:
            def __init__(self, api_key, **kwargs):
                self.api_key = api_key
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(
                        create=AsyncMock(return_value=type(
                            "Response", (),
                            {
                                "choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()],
                                "usage": fake_usage,
                            },
                        )())
                    )
                )

        class FakeOpenAIError(Exception):
            pass

        fake_openai.AsyncOpenAI = FakeAsyncOpenAI
        fake_openai.OpenAI = FakeAsyncOpenAI
        fake_openai.APIError = FakeOpenAIError
        fake_openai.RateLimitError = type("RateLimitError", (FakeOpenAIError,), {"status_code": None})
        fake_openai.APIConnectionError = type("APIConnectionError", (FakeOpenAIError,), {})

        env = {**REQUIRED_ENV}
        if env_extra:
            env.update(env_extra)

        with patch.dict(sys.modules, {"dotenv": fake_dotenv, "openai": fake_openai}):
            with patch.dict(os.environ, env, clear=True):
                sys.modules.pop("config", None)
                sys.modules.pop("utils", None)
                config = importlib.import_module("config")
                utils = importlib.import_module("utils")

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            await utils.call_openai("system", "user")

        return captured.getvalue()

    def test_emf_includes_estimated_cost_usd(self):
        import asyncio
        output = asyncio.run(self._run_call_openai_with_fake())
        self.assertIn("EstimatedCostUSD", output)
        compact = output.replace(" ", "")
        self.assertIn('"EstimatedCostUSD":', compact)

    def test_cost_estimate_for_gpt4o_mini(self):
        cost = 1000 * 0.15 / 1_000_000 + 500 * 0.60 / 1_000_000
        self.assertAlmostEqual(cost, cost, places=10)
        input_per_m, output_per_m = 0.15, 0.60
        expected = 1000 * input_per_m / 1_000_000 + 500 * output_per_m / 1_000_000
        self.assertAlmostEqual(cost, expected, places=10)


class ModelCostGuardTests(unittest.IsolatedAsyncioTestCase):
    """Tests for OPENAI_MODEL cost guard warning in call_openai."""

    async def test_warns_on_unknown_model(self):
        """call_openai should log a warning when OPENAI_MODEL is not in cost table."""
        import importlib
        import sys
        import types
        from unittest.mock import AsyncMock, MagicMock, patch

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_openai = types.ModuleType("openai")

        fake_usage = MagicMock()
        fake_usage.prompt_tokens = 10
        fake_usage.completion_tokens = 5
        fake_usage.total_tokens = 15

        class FakeAsyncOpenAI:
            def __init__(self, api_key, **kwargs):
                self.api_key = api_key
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(
                        create=AsyncMock(return_value=type(
                            "Response", (),
                            {
                                "choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()],
                                "usage": fake_usage,
                            },
                        )())
                    )
                )

        class FakeOpenAIError(Exception):
            pass

        fake_openai.AsyncOpenAI = FakeAsyncOpenAI
        fake_openai.OpenAI = FakeAsyncOpenAI
        fake_openai.APIError = FakeOpenAIError
        fake_openai.RateLimitError = type("RateLimitError", (FakeOpenAIError,), {"status_code": None})
        fake_openai.APIConnectionError = type("APIConnectionError", (FakeOpenAIError,), {})

        with patch.dict(sys.modules, {"dotenv": fake_dotenv, "openai": fake_openai}):
            with patch.dict(os.environ, {
                **REQUIRED_ENV,
                "OPENAI_MODEL": "gpt-4o",
            }, clear=True):
                sys.modules.pop("config", None)
                sys.modules.pop("utils", None)
                config = importlib.import_module("config")
                utils = importlib.import_module("utils")

        with patch.object(utils.logger, "warning") as mock_warn:
            await utils.call_openai("system", "user")
            warn_msgs = [str(c) for c in mock_warn.call_args_list]
            self.assertTrue(any("cost table" in w or "gpt-4o-mini" in w for w in warn_msgs),
                            "Expected cost guard warning for unknown model")

    async def test_no_warning_for_known_model(self):
        """call_openai should NOT warn when OPENAI_MODEL is gpt-4o-mini."""
        import importlib
        import sys
        import types
        from unittest.mock import AsyncMock, MagicMock, patch

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_openai = types.ModuleType("openai")

        fake_usage = MagicMock()
        fake_usage.prompt_tokens = 10
        fake_usage.completion_tokens = 5
        fake_usage.total_tokens = 15

        class FakeAsyncOpenAI:
            def __init__(self, api_key, **kwargs):
                self.api_key = api_key
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(
                        create=AsyncMock(return_value=type(
                            "Response", (),
                            {
                                "choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()],
                                "usage": fake_usage,
                            },
                        )())
                    )
                )

        class FakeOpenAIError(Exception):
            pass

        fake_openai.AsyncOpenAI = FakeAsyncOpenAI
        fake_openai.OpenAI = FakeAsyncOpenAI
        fake_openai.APIError = FakeOpenAIError
        fake_openai.RateLimitError = type("RateLimitError", (FakeOpenAIError,), {"status_code": None})
        fake_openai.APIConnectionError = type("APIConnectionError", (FakeOpenAIError,), {})

        with patch.dict(sys.modules, {"dotenv": fake_dotenv, "openai": fake_openai}):
            with patch.dict(os.environ, {
                **REQUIRED_ENV,
                "OPENAI_MODEL": "gpt-4o-mini",
            }, clear=True):
                sys.modules.pop("config", None)
                sys.modules.pop("utils", None)
                config = importlib.import_module("config")
                utils = importlib.import_module("utils")

        with patch.object(utils.logger, "warning") as mock_warn:
            await utils.call_openai("system", "user")
            warn_msgs = [str(c) for c in mock_warn.call_args_list]
            self.assertFalse(any("cost table" in w for w in warn_msgs),
                             "Should not warn for gpt-4o-mini")


class CircuitBreakerTests(unittest.IsolatedAsyncioTestCase):
    """Tests for OpenAI circuit breaker in call_openai."""

    async def test_circuit_breaker_opens_after_consecutive_failures(self):
        """After _CIRCUIT_BREAKER_THRESHOLD failures, call_openai should skip the API call."""
        import importlib
        import sys
        import types
        from unittest.mock import AsyncMock, MagicMock, patch

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_openai = types.ModuleType("openai")

        fake_usage = MagicMock()
        fake_usage.prompt_tokens = 10
        fake_usage.completion_tokens = 5
        fake_usage.total_tokens = 15

        class FakeAsyncOpenAI:
            def __init__(self, api_key, **kwargs):
                self.api_key = api_key
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(
                        create=AsyncMock(return_value=type(
                            "Response", (),
                            {
                                "choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()],
                                "usage": fake_usage,
                            },
                        )())
                    )
                )

        class FakeOpenAIError(Exception):
            pass

        fake_openai.AsyncOpenAI = FakeAsyncOpenAI
        fake_openai.OpenAI = FakeAsyncOpenAI
        fake_openai.APIError = FakeOpenAIError
        fake_openai.RateLimitError = type("RateLimitError", (FakeOpenAIError,), {"status_code": None})
        fake_openai.APIConnectionError = type("APIConnectionError", (FakeOpenAIError,), {})

        with patch.dict(sys.modules, {"dotenv": fake_dotenv, "openai": fake_openai}):
            with patch.dict(os.environ, REQUIRED_ENV, clear=True):
                sys.modules.pop("config", None)
                sys.modules.pop("utils", None)
                config = importlib.import_module("config")
                utils = importlib.import_module("utils")

        utils._CIRCUIT_BREAKER_FAILURES = config.CIRCUIT_BREAKER_THRESHOLD
        utils._CIRCUIT_BREAKER_OPEN_SINCE = _time.monotonic()

        with patch.object(utils.logger, "warning") as mock_warn:
            result = await utils.call_openai("system", "user")
            self.assertEqual(result, "")
            warn_msgs = [str(c) for c in mock_warn.call_args_list]
            self.assertTrue(any("circuit breaker open" in w.lower() for w in warn_msgs),
                            "Expected circuit breaker open warning")

    async def test_circuit_breaker_increments_on_api_error(self):
        """Circuit breaker failure counter should increment on API errors."""
        import importlib
        import sys
        import types
        from unittest.mock import AsyncMock, MagicMock, patch

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_openai = types.ModuleType("openai")

        class FakeAsyncOpenAI:
            def __init__(self, api_key, **kwargs):
                self.api_key = api_key
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(
                        create=AsyncMock(side_effect=Exception("API error"))
                    )
                )

        class FakeOpenAIError(Exception):
            pass

        fake_openai.AsyncOpenAI = FakeAsyncOpenAI
        fake_openai.OpenAI = FakeAsyncOpenAI
        fake_openai.APIError = FakeOpenAIError
        fake_openai.RateLimitError = type("RateLimitError", (FakeOpenAIError,), {"status_code": None})
        fake_openai.APIConnectionError = type("APIConnectionError", (FakeOpenAIError,), {})

        with patch.dict(sys.modules, {"dotenv": fake_dotenv, "openai": fake_openai}):
            with patch.dict(os.environ, REQUIRED_ENV, clear=True):
                sys.modules.pop("config", None)
                sys.modules.pop("utils", None)
                config = importlib.import_module("config")
                utils = importlib.import_module("utils")

        utils._CIRCUIT_BREAKER_FAILURES = 0
        result = await utils.call_openai("system", "user")
        self.assertEqual(result, "")
        self.assertGreater(utils._CIRCUIT_BREAKER_FAILURES, 0)

    async def test_circuit_breaker_resets_on_success(self):
        """Circuit breaker counter should reset to 0 on a successful call."""
        import importlib
        import sys
        import types
        from unittest.mock import AsyncMock, MagicMock, patch

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_openai = types.ModuleType("openai")

        fake_usage = MagicMock()
        fake_usage.prompt_tokens = 10
        fake_usage.completion_tokens = 5
        fake_usage.total_tokens = 15

        class FakeAsyncOpenAI:
            def __init__(self, api_key, **kwargs):
                self.api_key = api_key
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(
                        create=AsyncMock(return_value=type(
                            "Response", (),
                            {
                                "choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()],
                                "usage": fake_usage,
                            },
                        )())
                    )
                )

        class FakeOpenAIError(Exception):
            pass

        fake_openai.AsyncOpenAI = FakeAsyncOpenAI
        fake_openai.OpenAI = FakeAsyncOpenAI
        fake_openai.APIError = FakeOpenAIError
        fake_openai.RateLimitError = type("RateLimitError", (FakeOpenAIError,), {"status_code": None})
        fake_openai.APIConnectionError = type("APIConnectionError", (FakeOpenAIError,), {})

        with patch.dict(sys.modules, {"dotenv": fake_dotenv, "openai": fake_openai}):
            with patch.dict(os.environ, REQUIRED_ENV, clear=True):
                sys.modules.pop("config", None)
                sys.modules.pop("utils", None)
                config = importlib.import_module("config")
                utils = importlib.import_module("utils")

        utils._CIRCUIT_BREAKER_FAILURES = 2
        result = await utils.call_openai("system", "user")
        self.assertEqual(result, "ok")
        self.assertEqual(utils._CIRCUIT_BREAKER_FAILURES, 0)


class CircuitBreakerConfigTests(unittest.TestCase):
    """Tests for CIRCUIT_BREAKER_THRESHOLD and CIRCUIT_BREAKER_RESET_SEC config."""

    def test_config_circuit_breaker_threshold_default(self):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, REQUIRED_ENV, clear=True):
                sys.modules.pop("config", None)
                config = importlib.import_module("config")
                self.assertEqual(config.CIRCUIT_BREAKER_THRESHOLD, 3)

    def test_config_reads_circuit_breaker_threshold_from_env(self):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, {
                **REQUIRED_ENV,
                "CIRCUIT_BREAKER_THRESHOLD": "5",
            }, clear=True):
                sys.modules.pop("config", None)
                config = importlib.import_module("config")
                self.assertEqual(config.CIRCUIT_BREAKER_THRESHOLD, 5)

    def test_config_circuit_breaker_threshold_rejects_zero(self):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, {
                **REQUIRED_ENV,
                "CIRCUIT_BREAKER_THRESHOLD": "0",
            }, clear=True):
                sys.modules.pop("config", None)
                with self.assertRaises(ValueError):
                    importlib.import_module("config")

    def test_config_circuit_breaker_reset_sec_default(self):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, REQUIRED_ENV, clear=True):
                sys.modules.pop("config", None)
                config = importlib.import_module("config")
                self.assertEqual(config.CIRCUIT_BREAKER_RESET_SEC, 60)

    def test_config_reads_circuit_breaker_reset_sec_from_env(self):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, {
                **REQUIRED_ENV,
                "CIRCUIT_BREAKER_RESET_SEC": "120",
            }, clear=True):
                sys.modules.pop("config", None)
                config = importlib.import_module("config")
                self.assertEqual(config.CIRCUIT_BREAKER_RESET_SEC, 120)

    def test_config_circuit_breaker_reset_sec_rejects_zero(self):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, {
                **REQUIRED_ENV,
                "CIRCUIT_BREAKER_RESET_SEC": "0",
            }, clear=True):
                sys.modules.pop("config", None)
                with self.assertRaises(ValueError):
                    importlib.import_module("config")


class CircuitBreakerHalfOpenTests(unittest.IsolatedAsyncioTestCase):
    """Tests for circuit breaker half-open (auto-reset) behavior."""

    async def _setup_utils(self, env_extra=None):
        import importlib
        import sys
        import types
        from unittest.mock import AsyncMock, MagicMock, patch

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_openai = types.ModuleType("openai")

        fake_usage = MagicMock()
        fake_usage.prompt_tokens = 10
        fake_usage.completion_tokens = 5
        fake_usage.total_tokens = 15

        class FakeAsyncOpenAI:
            def __init__(self, api_key, **kwargs):
                self.api_key = api_key
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(
                        create=AsyncMock(return_value=type(
                            "Response", (),
                            {
                                "choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()],
                                "usage": fake_usage,
                            },
                        )())
                    )
                )

        class FakeOpenAIError(Exception):
            pass

        fake_openai.AsyncOpenAI = FakeAsyncOpenAI
        fake_openai.OpenAI = FakeAsyncOpenAI
        fake_openai.APIError = FakeOpenAIError
        fake_openai.RateLimitError = type("RateLimitError", (FakeOpenAIError,), {"status_code": None})
        fake_openai.APIConnectionError = type("APIConnectionError", (FakeOpenAIError,), {})

        env = {**REQUIRED_ENV}
        if env_extra:
            env.update(env_extra)

        with patch.dict(sys.modules, {"dotenv": fake_dotenv, "openai": fake_openai}):
            with patch.dict(os.environ, env, clear=True):
                sys.modules.pop("config", None)
                sys.modules.pop("utils", None)
                config = importlib.import_module("config")
                utils = importlib.import_module("utils")

        return config, utils

    async def test_circuit_breaker_skips_call_before_reset_timeout(self):
        """When circuit breaker is open and reset timeout hasn't elapsed, call should be skipped."""
        config, utils = await self._setup_utils()

        utils._CIRCUIT_BREAKER_FAILURES = config.CIRCUIT_BREAKER_THRESHOLD
        utils._CIRCUIT_BREAKER_OPEN_SINCE = _time.monotonic()

        with patch.object(utils.logger, "warning") as mock_warn:
            result = await utils.call_openai("system", "user")
            self.assertEqual(result, "")
            warn_msgs = [str(c) for c in mock_warn.call_args_list]
            self.assertTrue(any("circuit breaker open" in w.lower() for w in warn_msgs))

    async def test_circuit_breaker_allows_probe_after_reset_timeout(self):
        """When circuit breaker is open but reset timeout has elapsed, a probe call should be allowed."""
        config, utils = await self._setup_utils()

        utils._CIRCUIT_BREAKER_FAILURES = config.CIRCUIT_BREAKER_THRESHOLD
        utils._CIRCUIT_BREAKER_OPEN_SINCE = _time.monotonic() - config.CIRCUIT_BREAKER_RESET_SEC - 1

        with patch.object(utils.logger, "info") as mock_info:
            result = await utils.call_openai("system", "user")
            self.assertEqual(result, "ok")
            info_msgs = [str(c) for c in mock_info.call_args_list]
            self.assertTrue(any("half-open" in w.lower() for w in info_msgs))

    async def test_circuit_breaker_resets_on_successful_probe(self):
        """A successful probe call should fully reset the circuit breaker."""
        config, utils = await self._setup_utils()

        utils._CIRCUIT_BREAKER_FAILURES = config.CIRCUIT_BREAKER_THRESHOLD
        utils._CIRCUIT_BREAKER_OPEN_SINCE = _time.monotonic() - config.CIRCUIT_BREAKER_RESET_SEC - 1

        result = await utils.call_openai("system", "user")
        self.assertEqual(result, "ok")
        self.assertEqual(utils._CIRCUIT_BREAKER_FAILURES, 0)
        self.assertEqual(utils._CIRCUIT_BREAKER_OPEN_SINCE, 0.0)

    async def test_cb_record_failure_sets_open_since_on_threshold(self):
        """_cb_record_failure should set _CIRCUIT_BREAKER_OPEN_SINCE when threshold is first reached."""
        config, utils = await self._setup_utils()

        self.assertEqual(utils._CIRCUIT_BREAKER_OPEN_SINCE, 0.0)

        for i in range(config.CIRCUIT_BREAKER_THRESHOLD - 1):
            utils._cb_record_failure()
            self.assertEqual(utils._CIRCUIT_BREAKER_OPEN_SINCE, 0.0)

        utils._cb_record_failure()
        self.assertGreater(utils._CIRCUIT_BREAKER_OPEN_SINCE, 0.0)

    async def test_cb_record_success_resets_open_since(self):
        """_cb_record_success should reset _CIRCUIT_BREAKER_OPEN_SINCE to 0."""
        config, utils = await self._setup_utils()

        for _ in range(config.CIRCUIT_BREAKER_THRESHOLD):
            utils._cb_record_failure()
        self.assertGreater(utils._CIRCUIT_BREAKER_OPEN_SINCE, 0.0)

        utils._cb_record_success()
        self.assertEqual(utils._CIRCUIT_BREAKER_FAILURES, 0)
        self.assertEqual(utils._CIRCUIT_BREAKER_OPEN_SINCE, 0.0)

    async def test_get_circuit_breaker_state_closed(self):
        """get_circuit_breaker_state returns 'closed' when failures < threshold."""
        config, utils = await self._setup_utils()

        utils._CIRCUIT_BREAKER_FAILURES = 0
        state = utils.get_circuit_breaker_state()
        self.assertEqual(state["state"], "closed")
        self.assertEqual(state["failures"], 0)

    async def test_get_circuit_breaker_state_open(self):
        """get_circuit_breaker_state returns 'open' when failures >= threshold and reset not elapsed."""
        config, utils = await self._setup_utils()

        utils._CIRCUIT_BREAKER_FAILURES = config.CIRCUIT_BREAKER_THRESHOLD
        utils._CIRCUIT_BREAKER_OPEN_SINCE = _time.monotonic()
        state = utils.get_circuit_breaker_state()
        self.assertEqual(state["state"], "open")
        self.assertIn("open_since_elapsed", state)

    async def test_get_circuit_breaker_state_half_open(self):
        """get_circuit_breaker_state returns 'half_open' when reset timeout has elapsed."""
        config, utils = await self._setup_utils()

        utils._CIRCUIT_BREAKER_FAILURES = config.CIRCUIT_BREAKER_THRESHOLD
        utils._CIRCUIT_BREAKER_OPEN_SINCE = _time.monotonic() - config.CIRCUIT_BREAKER_RESET_SEC - 1
        state = utils.get_circuit_breaker_state()
        self.assertEqual(state["state"], "half_open")
        self.assertIn("open_since_elapsed", state)

    async def test_cost_estimate_for_gpt41_mini(self):
        """Cost estimate for gpt-4.1-mini uses its own pricing."""
        config, utils = await self._setup_utils()

        cost = utils._estimate_cost_usd("gpt-4.1-mini", 1_000_000, 1_000_000)
        self.assertAlmostEqual(cost, 0.40 + 1.60, places=2)

    async def test_no_warning_for_gpt41_mini(self):
        """gpt-4.1-mini is in cost table — no warning should be emitted."""
        config, utils = await self._setup_utils()

        config.OPENAI_MODEL = "gpt-4.1-mini"
        with patch.object(utils, "openai_client", None), \
             patch.object(utils, "AsyncOpenAI", return_value=Mock()) as mock_openai, \
             patch.object(utils.logger, "warning") as mock_warn:
            await utils.call_openai("sys", "user")
        cost_warnings = [c for c in mock_warn.call_args_list if "cost table" in str(c)]
        self.assertEqual(len(cost_warnings), 0)

    async def test_cost_estimate_for_gpt41_nano(self):
        """Cost estimate for gpt-4.1-nano uses its own pricing ($0.10/$0.40 per 1M)."""
        config, utils = await self._setup_utils()

        cost = utils._estimate_cost_usd("gpt-4.1-nano", 1_000_000, 1_000_000)
        self.assertAlmostEqual(cost, 0.10 + 0.40, places=2)

    async def test_no_warning_for_gpt41_nano(self):
        """gpt-4.1-nano is in cost table — no warning should be emitted."""
        config, utils = await self._setup_utils()

        config.OPENAI_MODEL = "gpt-4.1-nano"
        with patch.object(utils, "openai_client", None), \
             patch.object(utils, "AsyncOpenAI", return_value=Mock()) as mock_openai, \
             patch.object(utils.logger, "warning") as mock_warn:
            await utils.call_openai("sys", "user")
        cost_warnings = [c for c in mock_warn.call_args_list if "cost table" in str(c)]
        self.assertEqual(len(cost_warnings), 0)


class EmptyChoicesGuardTests(unittest.IsolatedAsyncioTestCase):
    """Tests for empty choices guard in call_openai."""

    async def test_call_openai_returns_empty_on_empty_choices(self):
        """call_openai should return '' when OpenAI returns empty choices array."""
        import importlib
        import sys
        import types
        from unittest.mock import AsyncMock, MagicMock, patch

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_openai = types.ModuleType("openai")

        class FakeAsyncOpenAI:
            def __init__(self, api_key, **kwargs):
                self.api_key = api_key
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(
                        create=AsyncMock(return_value=type("Response", (), {"choices": []})())
                    )
                )

        class FakeOpenAIError(Exception):
            pass

        fake_openai.AsyncOpenAI = FakeAsyncOpenAI
        fake_openai.OpenAI = FakeAsyncOpenAI
        fake_openai.APIError = FakeOpenAIError
        fake_openai.RateLimitError = type("RateLimitError", (FakeOpenAIError,), {"status_code": None})
        fake_openai.APIConnectionError = type("APIConnectionError", (FakeOpenAIError,), {})

        with patch.dict(sys.modules, {"dotenv": fake_dotenv, "openai": fake_openai}):
            with patch.dict(os.environ, REQUIRED_ENV, clear=True):
                sys.modules.pop("config", None)
                sys.modules.pop("utils", None)
                config = importlib.import_module("config")
                utils = importlib.import_module("utils")

        with patch.object(utils.logger, "warning") as mock_warn:
            result = await utils.call_openai("system", "user")
            self.assertEqual(result, "")
            warn_msgs = [str(c) for c in mock_warn.call_args_list]
            self.assertTrue(any("empty choices" in w.lower() for w in warn_msgs),
                            "Expected empty choices warning")


class ResetCircuitBreakerTests(unittest.IsolatedAsyncioTestCase):
    """Tests for reset_circuit_breaker function."""

    async def test_reset_circuit_breaker_clears_state(self):
        """reset_circuit_breaker should clear failures and open_since."""
        import importlib
        import sys
        import types
        from unittest.mock import MagicMock, patch

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_openai = types.ModuleType("openai")

        class FakeAsyncOpenAI:
            def __init__(self, api_key, **kwargs):
                pass

        class FakeOpenAIError(Exception):
            pass

        fake_openai.AsyncOpenAI = FakeAsyncOpenAI
        fake_openai.OpenAI = FakeAsyncOpenAI
        fake_openai.APIError = FakeOpenAIError
        fake_openai.RateLimitError = type("RateLimitError", (FakeOpenAIError,), {"status_code": None})
        fake_openai.APIConnectionError = type("APIConnectionError", (FakeOpenAIError,), {})

        with patch.dict(sys.modules, {"dotenv": fake_dotenv, "openai": fake_openai}):
            with patch.dict(os.environ, REQUIRED_ENV, clear=True):
                sys.modules.pop("config", None)
                sys.modules.pop("utils", None)
                config = importlib.import_module("config")
                utils = importlib.import_module("utils")

        utils._CIRCUIT_BREAKER_FAILURES = 5
        utils._CIRCUIT_BREAKER_OPEN_SINCE = 123.45

        utils.reset_circuit_breaker()

        self.assertEqual(utils._CIRCUIT_BREAKER_FAILURES, 0)
        self.assertEqual(utils._CIRCUIT_BREAKER_OPEN_SINCE, 0.0)


class TokenUsageTests(unittest.IsolatedAsyncioTestCase):
    """Tests for cumulative token usage tracking."""

    async def _setup_utils(self):
        import importlib
        import sys
        import types
        from unittest.mock import AsyncMock, MagicMock, patch

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_openai = types.ModuleType("openai")

        fake_usage = MagicMock()
        fake_usage.prompt_tokens = 100
        fake_usage.completion_tokens = 50
        fake_usage.total_tokens = 150

        class FakeAsyncOpenAI:
            def __init__(self, api_key, **kwargs):
                self.api_key = api_key
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(
                        create=AsyncMock(return_value=type(
                            "Response", (),
                            {
                                "choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()],
                                "usage": fake_usage,
                            },
                        )())
                    )
                )

        class FakeOpenAIError(Exception):
            pass

        fake_openai.AsyncOpenAI = FakeAsyncOpenAI
        fake_openai.OpenAI = FakeAsyncOpenAI
        fake_openai.APIError = FakeOpenAIError
        fake_openai.RateLimitError = type("RateLimitError", (FakeOpenAIError,), {"status_code": None})
        fake_openai.APIConnectionError = type("APIConnectionError", (FakeOpenAIError,), {})

        with patch.dict(sys.modules, {"dotenv": fake_dotenv, "openai": fake_openai}):
            with patch.dict(os.environ, REQUIRED_ENV, clear=True):
                sys.modules.pop("config", None)
                sys.modules.pop("utils", None)
                config = importlib.import_module("config")
                utils = importlib.import_module("utils")

        return utils

    async def test_token_usage_accumulates_across_calls(self):
        """Cumulative token counters should accumulate across multiple call_openai calls."""
        utils = await self._setup_utils()

        utils.reset_token_usage()
        self.assertEqual(utils.get_token_usage(), {"prompt_tokens": 0, "completion_tokens": 0})

        await utils.call_openai("system", "user")
        usage = utils.get_token_usage()
        self.assertEqual(usage["prompt_tokens"], 100)
        self.assertEqual(usage["completion_tokens"], 50)

        await utils.call_openai("system", "user")
        usage = utils.get_token_usage()
        self.assertEqual(usage["prompt_tokens"], 200)
        self.assertEqual(usage["completion_tokens"], 100)

    async def test_reset_token_usage_clears_counters(self):
        """reset_token_usage should reset both counters to 0."""
        utils = await self._setup_utils()

        await utils.call_openai("system", "user")
        self.assertGreater(utils.get_token_usage()["prompt_tokens"], 0)

        utils.reset_token_usage()
        self.assertEqual(utils.get_token_usage(), {"prompt_tokens": 0, "completion_tokens": 0})


class EmptyInputValidationTests(unittest.IsolatedAsyncioTestCase):
    """Tests for call_openai empty input validation."""

    async def _setup_utils(self):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_openai = types.ModuleType("openai")

        class FakeAsyncOpenAI:
            def __init__(self, api_key, **kwargs):
                pass

            class chat:
                class completions:
                    @staticmethod
                    async def create(**kwargs):
                        raise AssertionError("API should not be called with empty input")

        class FakeOpenAIError(Exception):
            pass

        fake_openai.AsyncOpenAI = FakeAsyncOpenAI
        fake_openai.APIError = FakeOpenAIError
        fake_openai.RateLimitError = type("RateLimitError", (FakeOpenAIError,), {"status_code": None})
        fake_openai.APIConnectionError = type("APIConnectionError", (FakeOpenAIError,), {})

        with patch.dict(sys.modules, {"dotenv": fake_dotenv, "openai": fake_openai}):
            with patch.dict(os.environ, REQUIRED_ENV, clear=True):
                sys.modules.pop("config", None)
                sys.modules.pop("utils", None)
                return importlib.import_module("utils")

    async def test_call_openai_returns_empty_on_empty_system_prompt(self):
        utils = await self._setup_utils()
        result = await utils.call_openai("", "user content")
        self.assertEqual(result, "")

    async def test_call_openai_returns_empty_on_whitespace_system_prompt(self):
        utils = await self._setup_utils()
        result = await utils.call_openai("   ", "user content")
        self.assertEqual(result, "")

    async def test_call_openai_returns_empty_on_empty_user_content(self):
        utils = await self._setup_utils()
        result = await utils.call_openai("system prompt", "")
        self.assertEqual(result, "")

    async def test_call_openai_returns_empty_on_whitespace_user_content(self):
        utils = await self._setup_utils()
        result = await utils.call_openai("system prompt", "   ")
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
