import importlib
import os
import sys
import types
import unittest
from unittest.mock import Mock, patch


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
        def __init__(self, api_key):
            self.api_key = api_key
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=Mock())
            )

    fake_openai.OpenAI = FakeOpenAI
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
    async def test_call_openai_uses_configured_model_and_default_token_limit(self):
        env = {
            **REQUIRED_ENV,
            "OPENAI_MODEL": "gpt-4o-mini",
            "OPENAI_DEFAULT_MAX_TOKENS": "321",
        }

        with patch.dict(os.environ, env, clear=True):
            config = _reload_module("config")
            utils = _reload_module("utils")

        create_mock = Mock(
            return_value=type(
                "Response",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {"message": type("Message", (), {"content": "done"})()},
                        )()
                    ]
                },
            )()
        )

        # Trigger lazy initialization of openai_client
        _ = utils.openai_client  # will be None initially
        # The client is initialized inside call_openai, so we patch the class
        with patch.object(utils, "openai_client", create=True) as mock_client:
            mock_client.chat.completions.create = create_mock
            result = await utils.call_openai("system", "user")

        self.assertEqual(result, "done")
        create_mock.assert_called_once_with(
            messages=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ],
            model=config.OPENAI_MODEL,
            max_tokens=config.OPENAI_DEFAULT_MAX_TOKENS,
        )


if __name__ == "__main__":
    unittest.main()
