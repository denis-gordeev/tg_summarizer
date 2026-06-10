import importlib
import os
import sys
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
        self.assertEqual(config.UPDATE_SUMMARY_MAX_TOKENS, 2000)

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
        self.assertAlmostEqual(config.OPENAI_SUMMARY_TEMPERATURE, 0.3)


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
                self.assertEqual(config.SUMMARY_MAX_INPUT_CHARS_PER_MESSAGE, 3000)

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
                self.assertEqual(config.UPDATE_SUMMARY_MAX_INPUT_CHARS, 2000)


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


if __name__ == "__main__":
    unittest.main()
