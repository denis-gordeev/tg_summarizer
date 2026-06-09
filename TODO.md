# TODO

Живой список задач для автоматических раундов.

## Completed in 2026-06-02 round 2 (Lambda hardening round 8, timeouts, limits, cost optimization)

- **OpenAI request timeout**: Added `OPENAI_REQUEST_TIMEOUT` config (default 30s) in [`config.py`](config.py). Applied in [`call_openai()`](utils.py) — `AsyncOpenAI` client now created with `timeout=float(OPENAI_REQUEST_TIMEOUT)`. Previously, the default httpx timeout was 10 minutes, which could cause the entire Lambda invocation to time out on a single hung OpenAI request.
- **NLP check input truncation**: Added `NLP_CHECK_MAX_INPUT_CHARS` config (default 2000) in [`config.py`](config.py). Applied in [`is_nlp_related()`](message_processor.py) — message text is now truncated to `NLP_CHECK_MAX_INPUT_CHARS` before being sent to the NLP relevance LLM call. Previously, very long messages (full articles) would waste thousands of input tokens on a yes/no classification.
- **Message fetch limit per source**: Added `MAX_MESSAGES_PER_SOURCE` config (default 100) in [`config.py`](config.py). Applied in [`_fetch_from_sources()`](telegram_client.py) — iteration breaks when the per-source limit is reached. Previously, a very active channel could return thousands of messages, consuming excessive Lambda time and tokens.
- **Telegram client connection timeout**: Added `connection_retries=3`, `retry_delay=2`, `timeout=15` to both `TelegramClient` constructors in [`start_clients()`](telegram_client.py). Previously, default Telethon settings could hang indefinitely on connection attempts.
- **Code cleanup**: Replaced redundant `source_msgs` list with a simple `source_count` counter in [`_fetch_from_sources()`](telegram_client.py) — the list was only used for its length, never its contents.
- **.env.example sync**: Updated `OPENAI_CHANNEL_SUMMARY_MAX_TOKENS` and `OPENAI_GROUP_SUMMARY_MAX_TOKENS` defaults from 16000 to 4000 in [`.env.example`](.env.example) to match the actual defaults in [`config.py`](config.py).
- **SAM template updated**: Added `OPENAI_REQUEST_TIMEOUT`, `NLP_CHECK_MAX_INPUT_CHARS`, `MAX_MESSAGES_PER_SOURCE` parameters and env vars in [`template.yaml`](template.yaml).
- **Tests added**: 6 new tests (total 87, up from 81):
  - `test_call_openai_passes_timeout_to_client`: verifies `OPENAI_REQUEST_TIMEOUT` is passed to `AsyncOpenAI`
  - `test_config_reads_request_timeout_from_env`: verifies `OPENAI_REQUEST_TIMEOUT` env var parsing
  - `test_config_reads_nlp_check_max_input_chars_from_env`: verifies `NLP_CHECK_MAX_INPUT_CHARS` env var parsing
  - `test_config_reads_max_messages_per_source_from_env`: verifies `MAX_MESSAGES_PER_SOURCE` env var parsing
  - `test_nlp_check_truncates_long_input`: verifies text truncation in `is_nlp_related()`
  - `test_nlp_check_keeps_short_input`: verifies short text is not truncated
- Updated test stubs in [`tests/test_openai_config.py`](tests/test_openai_config.py) and [`tests/test_process_messages_integration.py`](tests/test_process_messages_integration.py) to support new config constants and `timeout` kwarg.
- All 87 tests pass without errors.

## Completed in 2026-04-14 round (Lambda hardening & test coverage)

- **Lambda error handling**: Добавлен try/except в [`lambda_handler.handler()`](lambda_handler.py) — теперь ошибки суммаризации возвращают структурированный `{'status': 'error', 'error': str(e)}` вместо unhandled exception. State всё равно загружается в S3 при ошибке для сохранения частичных обновлений.
- **Unused imports removed**: Удалены неиспользуемые импорты:
  - `random` в [`message_processor.py`](message_processor.py)
  - `Callable`, `TypeVar`, `T` в [`history_manager.py`](history_manager.py)
  - `datetime`, `timezone` в [`channel_manager.py`](channel_manager.py)
- **Logging hardening**: Конвертировано ~10 eager f-strings в [`history_manager.py`](history_manager.py) в lazy %-format (e.g. `logger.error("msg: %s", e)` вместо `logger.error(f"msg: {e}")`), что предотвращает вычисление строк при DEBUG-уровне.
- **Print statement replaced**: Заменён module-level `print()` в [`history_manager.py`](history_manager.py) на `logger.error()` для структурированного логирования в CloudWatch.
- **Tests added**: Создано 3 новых тестовых файла (+13 тестов, общее количество увеличено с 30 до 43):
  - [`tests/test_history_manager.py`](tests/test_history_manager.py): тесты логики `should_run_group_summarization()` и извлечения контекста истории (2 теста)
  - [`tests/test_channel_manager.py`](tests/test_channel_manager.py): тесты логики создания аббревиатур и слияния каналов (2 теста)
  - [`tests/test_models.py`](tests/test_models.py): тесты сериализации/десериализации `MessageInfo` и `SummaryInfo` (9 тестов), включая регрессионный тест на восстановление `is_nlp_related_reason`
- Все 43 теста проходят без ошибок.

## Completed in 2026-04-13 round 2 (dedup optimization & refactoring)

- Оптимизирована дедупликация в [`message_processor.py`](message_processor.py): добавлена трёхзональная стратегия `SequenceMatcher` для минимизации LLM-вызовов:
  - `ratio > SIMILARITY_LLM_UPPER` (0.95): почти дубликаты → вызов LLM
  - `ratio < SIMILARITY_LLM_LOWER` (0.7): явно разные → без LLM
  - Между зонами → вызов LLM
- Добавлены константы `SIMILARITY_LLM_LOWER` и `SIMILARITY_LLM_UPPER` в [`config.py`](config.py) с возможностью переопределения через env.
- Рефакторинг `process_messages()` в [`message_processor.py`](message_processor.py): выделены три функции — `_classify_message()`, `_create_summary_info()`, `_save_processing_results()`. Основная функция сокращена с 120+ строк до ~30, улучшена читаемость и тестируемость.
- Перенесена `get_all_source_channels()` из [`message_processor.py`](message_processor.py) в [`channel_manager.py`](channel_manager.py) для устранения циклического импорта и логической группировки функций работы с каналами.
- Обновлён импорт в [`telegram_client.py`](telegram_client.py): `get_all_source_channels` теперь импортируется из `channel_manager`.
- Обновлён [`pyproject.toml`](pyproject.toml): placeholder автора заменён на реального (Denis Gordeev).
- Обновлены тестовые стабы в [`tests/test_digest_post_processing.py`](tests/test_digest_post_processing.py), [`tests/test_process_messages_integration.py`](tests/test_process_messages_integration.py), [`tests/test_summary_length_guardrails.py`](tests/test_summary_length_guardrails.py) для поддержки новых импортов.
- Все 30 тестов проходят без ошибок.

## Completed in 2026-04-13 round (logging hardening & code quality)

- **Критический баг**: Добавлен импорт `RESTORE_HISTORY_DAYS` в [`history_manager.py`](history_manager.py) — ранее функция восстановления истории падала с `NameError` при вызове.
- Перенесены `_load_json_file()`, `_save_json_file()`, `now_iso()` в [`utils.py`](utils.py) как единый источник, устранено дублирование между [`history_manager.py`](history_manager.py) и [`channel_manager.py`](channel_manager.py) (~70 строк сокращено).
- Lazy-init OpenAI клиента в [`utils.py`](utils.py): `openai_client` теперь инициализируется внутри `call_openai()`, а не при импорте модуля. Это упрощает тестирование и避免了 ошибки при отсутствии `OPENAI_API_KEY` на импорте.
- Исправлена `save_updated_summary()` в [`history_manager.py`](history_manager.py): теперь использует `save_json_file()` вместо прямого `json.dump()`, добавлен `last_updated` timestamp.
- Добавлен `logging.basicConfig()` в [`lambda_handler.py`](lambda_handler.py) для структурированного логирования в CloudWatch.
- Конвертировано ~70 `print()` statements в `logging` across [`message_processor.py`](message_processor.py), [`summarizer.py`](summarizer.py), [`s3_sync.py`](s3_sync.py), [`telegram_client.py`](telegram_client.py), [`prompts.py`](prompts.py), [`history_manager.py`](history_manager.py), [`utils.py`](utils.py). DEBUG-уровень для отладочных сообщений, INFO для информационных, ERROR для ошибок.
- Выделены хелперы `_ensure_clients()` и `_ensure_bot_client()` в [`telegram_client.py`](telegram_client.py), устранено дублирование проверки подключения клиента (~15 строк сокращено).
- `SOURCE_GROUPS` теперь `set` вместо `list` в [`config.py`](config.py), что соответствует типу `SOURCE_CHANNELS` и предотвращает дубликаты.
- Обновлён [`tests/test_openai_config.py`](tests/test_openai_config.py) для поддержки lazy-init OpenAI клиента.
- Все 30 тестов проходят без ошибок.

## Completed in 2026-04-12 round 2 (code quality & refactoring)

- Выделены универсальные функции `_load_json_file()`, `_save_json_file()` и `_now_iso()` в [`history_manager.py`](history_manager.py), устранено дублирование кода загрузки/сохранения JSON (~150 строк сокращено).
- Рефакторинг `load_summarization_history`/`save_summarization_history` и их group-аналогов в [`history_manager.py`](history_manager.py): убраны `try/except` с `print()`, заменены на `logger.error()` и универсальные хелперы.
- Выделены универсальные функции `_load_channel_list()` и `_save_channel_list()` в [`channel_manager.py`](channel_manager.py), устранено дублирование `load_discovered_channels`/`load_similar_channels`/`load_banned_channels` и их save-аналогов (~80 строк сокращено).
- Все `datetime.now().isoformat()` заменены на `_now_iso()` (UTC timezone), что устраняет неоднозначность часовых поясов при сравнении timestamps в Lambda и между машинами.
- Магические числа вынесены в именованные константы в [`config.py`](config.py): `MAX_CHANNEL_HISTORY_MESSAGES`, `MAX_CHANNEL_SUMMARIES`, `MAX_GROUP_HISTORY_MESSAGES`, `MAX_GROUP_SUMMARIES`, `GROUP_SUMMARIZATION_INTERVAL_SECONDS`, `RESTORE_HISTORY_DAYS`, `SUMMARY_MIN_RATIO`, `SUMMARY_MIN_LENGTH`, `SUMMARY_MAX_LENGTH`, `GROUP_SUMMARY_MIN_LENGTH`, `GROUP_SUMMARY_MAX_LENGTH`, `TEXT_PREVIEW_LENGTH`.
- [`message_processor.py`](message_processor.py) обновлён для использования новых констант из `config.py`.
- Все `print()` в функциях восстановления истории ([`history_manager.py`](history_manager.py)) заменены на `logger.info()`/`logger.debug()`/`logger.error()`, что обеспечивает структурированное логирование в CloudWatch.
- Обновлён test stub в [`tests/test_process_messages_integration.py`](tests/test_process_messages_integration.py) для поддержки новых констант.
- Все 30 тестов проходят без ошибок.

## Completed in 2026-04-08 round

- Восстановлен [`s3_sync.py`](s3_sync.py), без которого чистый импорт [`lambda_handler.py`](lambda_handler.py) падал с `ModuleNotFoundError`.
- Исправлен разбор Lambda event-флагов в [`lambda_handler.py`](lambda_handler.py): строковые значения `false` / `true` теперь интерпретируются корректно.
- Восстановлены и расширены smoke/regression-тесты для Lambda и OpenAI-конфига в [`tests/test_lambda_handler.py`](tests/test_lambda_handler.py), [`tests/test_s3_sync.py`](tests/test_s3_sync.py) и [`tests/test_openai_config.py`](tests/test_openai_config.py).
- Добавлены guardrails на длину итогового дайджеста в [`message_processor.py`](message_processor.py), чтобы итоговые саммари оставались лаконичными даже если модель пытается выйти за целевой объём.
- Исправлены и документированы OpenAI env-настройки с дефолтом `gpt-4o-mini` в [`config.py`](config.py), [`utils.py`](utils.py) и [`.env.example`](.env.example).
- Добавлен отдельный Lambda runbook [`docs/aws-lambda-runbook.md`](docs/aws-lambda-runbook.md), а [`README.md`](README.md) синхронизирован с текущим деревом проекта.
- Добавлен [`boto3`](https://pypi.org/project/boto3/) в [`pyproject.toml`](pyproject.toml) и [`requirements.txt`](requirements.txt), чтобы S3-синхронизация работала в container image и локальной отладке, а не только в AWS managed runtime.
- Создан AWS SAM шаблон [`template.yaml`](template.yaml) для воспроизводимого деплоя Lambda с полной инфраструктурой (S3 bucket, IAM роли, EventBridge триггер).
- Добавлен скрипт [`build_lambda_package.sh`](build_lambda_package.sh) для сборки deployment package с зависимостями под Python 3.12.
- Написана полная инструкция по деплою в [`docs/aws-lambda-deployment.md`](docs/aws-lambda-deployment.md) с двумя опциями: ручной деплой через AWS CLI и автоматический через AWS SAM.
- Добавлен пример конфигурации [`samconfig.toml.example`](samconfig.toml.example) для быстрого старта SAM деплоя.

## Completed in 2026-04-09 round

- Добавлен pre-push hook [`hooks/pre-push`](hooks/pre-push), который автоматически запускает `python3 -m unittest discover -s tests` перед каждым push.
- Добавлен интеграционный тест [`tests/test_digest_post_processing.py`](tests/test_digest_post_processing.py) (10 тестов) на полный пост-процессинг дайджеста:
  - замена ссылок `[1]`, `[2]`, `[1,2]` на HTML-ссылки;
  - ограничение длины channel/group summary;
  - корректное закрытие HTML-тегов при обрезке;
  - консистентность аббревиатур каналов.
- Общее количество тестов увеличено с 15 до 25.

## Completed in 2026-04-10 round

- Добавлен GitHub Actions workflow [`.github/workflows/ci.yml`](.github/workflows/ci.yml) для автоматического прогона тестов и сборки Lambda-пакета на каждый push и PR.
- Добавлен интеграционный тест [`tests/test_process_messages_integration.py`](tests/test_process_messages_integration.py) (5 тестов) на полный end-to-end pipeline `process_messages()`:
  - полный канал pipeline: фильтрация NLP, дедупликация, суммаризация, сохранение истории;
  - пропуск non-NLP сообщений;
  - обработка пустого списка сообщений;
  - group pipeline: фильтрация, групповая суммаризация, обновление last run;
  - отправка сообщения в Telegram при `send_message=True`.
- Общее количество тестов увеличено с 25 до 30.
- Обновлён [`docs/aws-lambda-runbook.md`](docs/aws-lambda-runbook.md): секция «Следующий шаг по инфраструктуре» заменена на актуальную «Инфраструктура» с ссылками на существующие `template.yaml`, `samconfig.toml.example` и deployment guide.

## Completed in 2026-04-10 round 2 (code quality)

- Удалён неиспользуемый импорт `from operator import is_` в [`message_processor.py`](message_processor.py).
- Исправлен `MessageInfo.from_dict()` в [`models.py`](models.py): теперь восстанавливается поле `is_nlp_related_reason`.
- Устранено дублирование кода дедупликации (~100 строк) в [`message_processor.py`](message_processor.py): `remove_duplicates` и `remove_group_duplicates` теперь делегируют общую логику `_remove_duplicates_generic`.
- Вынесена общая функция `_replace_source_with_links` и `_prepare_messages_text` в [`message_processor.py`](message_processor.py), убрано ~60 строк дублирования между `summarize_text` и `summarize_group_text`.
- Удалена захардкоженная привязка к конкретному каналу (`denissexy`) из `CHANNEL_SUMMARY_PROMPT` в [`prompts.py`](prompts.py).
- Все 30 тестов проходят без ошибок.

## Completed in 2026-04-11 round (bug fixes & hardening)

- **Критический баг**: Заменён `hash(msg.text)` на детерминированный `hashlib.sha256(msg.text.encode()).hexdigest()[:16]` в [`message_processor.py`](message_processor.py) и [`history_manager.py`](history_manager.py). Python's built-in `hash()` рандомизирован через `PYTHONHASHSEED`, что приводило к пропуску дедупликации между разными Lambda invocation.
- **Баг**: Исправлен `should_run_group_summarization()` в [`history_manager.py`](history_manager.py): теперь возвращает `True` при первом запуске (когда файл истории ещё не создан), вместо `False`.
- **Улучшение диагностики**: Добавлен `traceback.print_exc()` в top-level `except` блок [`summarizer.py`](summarizer.py) для полной трассировки ошибок.
- **Type hints**: Исправлен невалидный type hint `-> (bool, str)` на `-> tuple[bool, str]` в `is_nlp_related()` в [`message_processor.py`](message_processor.py).
- **Производительность/логи**: Добавлен `DEBUG` флаг в [`config.py`](config.py) (через env `DEBUG=1`).Verbose логирование сообщений и полных саммари теперь gated behind этим флагом в [`message_processor.py`](message_processor.py), что снижает объем CloudWatch логов в production.
- **Clean up**: Удалены module-level `print()` statements из [`config.py`](config.py), которые срабатывали при каждом импорте модуля.
- **Тесты**: Обновлён test stub в [`tests/test_process_messages_integration.py`](tests/test_process_messages_integration.py) для поддержки `DEBUG` флага. Все 30 тестов проходят.

## Completed in 2026-04-11 round 2 (error handling & monitoring)

- Добавлен per-channel error handling в [`telegram_client.py`](telegram_client.py): каждый канал обернут в try/catch, чтобы сбой одного канала не ломал весь цикл загрузки.
- Добавлен модуль `logging` в [`telegram_client.py`](telegram_client.py) для лучшей трассировки ошибок через `logger.error()`.
- Добавлены CloudWatch алерты в [`template.yaml`](template.yaml): автоматическое создание трех алертов (Errors, Duration, Throttles) при SAM деплое.
- Обновлена документация в [`docs/aws-lambda-runbook.md`](docs/aws-lambda-runbook.md): добавлена секция «CloudWatch алерты» с описанием метрик и инструкцией по настройке SNS уведомлений.
- Обновлена документация в [`docs/aws-lambda-deployment.md`](docs/aws-lambda-deployment.md): добавлена ссылка на CloudWatch алерты в секции Monitoring.
- Все 30 тестов проходят без ошибок.

## Completed in 2026-04-12 round (maintenance & cleanup)

- Расширен [`.gitignore`](.gitignore): добавлены session файлы (`*.session`), JSON state файлы, логи (`*.log`), артефакты сборки (`build/`, `dist/`, `*.zip`), Python кэш, IDE файлы, `.samconfig.toml`.
- Исправлено вводящее в заблуждение имя параметра `EnableSsmSend` → `EnableTelegramSend` в [`template.yaml`](template.yaml) (параметр управляет отправкой в Telegram, а не SSM).
- Обновлён [`AUTOWORK_INSTRUCTIONS.md`](AUTOWORK_INSTRUCTIONS.md): убрана устаревшая заметка "нет доки по Lambda" — теперь указаны ссылки на существующие `docs/aws-lambda-deployment.md`, `docs/aws-lambda-runbook.md` и `template.yaml`.
- Все 30 тестов проходят без ошибок.

## Completed in 2026-05-26 round (Lambda hardening round 2)

- **Critical bug**: Added missing `logger = logging.getLogger(__name__)` in [`lambda_handler.py`](lambda_handler.py) — `logger` was used on line 62 but never defined, causing `NameError` on the error path instead of structured logging.
- **OpenAI retry with exponential backoff**: Added retry logic in [`call_openai()`](utils.py) for `RateLimitError`, `APIConnectionError`, and 5xx `APIError`. Default: 3 retries with base delay 1s, doubling each attempt. This prevents Lambda failures on transient OpenAI API issues.
- **Duplicate error logging fixed**: Removed duplicate `logger.error()` calls in [`telegram_client.py`](telegram_client.py) — both `fetch_messages()` and `fetch_group_messages()` were logging the same exception twice (once with f-string, once with %-format). Now uses single lazy %-format call.
- **Lazy logging in utils.py**: Converted 2 remaining eager f-strings in `logger.error()` in [`utils.py`](utils.py) to lazy %-format (consistent with prior hardening in `history_manager.py`).
- **Lazy logging in history_manager.py**: Converted 1 remaining eager f-string in `logger.error()` in [`history_manager.py`](history_manager.py) to lazy %-format.
- **Double shuffle removed**: Removed redundant `random.shuffle(all_channels)` in [`telegram_client.py:fetch_messages()`](telegram_client.py) — channels are already shuffled in `get_all_source_channels()` in [`channel_manager.py`](channel_manager.py). Removed unused `import random`.
- **SQS Dead Letter Queue**: Added `SummarizerDLQ` (SQS queue with 14-day retention) in [`template.yaml`](template.yaml) and wired it to the Lambda function's `DeadLetterQueue` config. Failed invocations are now captured for inspection instead of being silently dropped.
- Updated test stubs in [`tests/test_openai_config.py`](tests/test_openai_config.py) and [`tests/test_summary_length_guardrails.py`](tests/test_summary_length_guardrails.py) to include `APIError`, `RateLimitError`, `APIConnectionError` in fake `openai` module.
- All 43 tests pass without errors.

## Completed in 2026-05-27 round (config lazy validation & Lambda hardening round 3)

- **Config lazy validation**: Refactored [`config.py`](config.py) — required env vars (`TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, `TELEGRAM_BOT_TOKEN`, `TARGET_CHANNEL`, `OPENAI_API_KEY`) now default to `None` instead of raising `ValueError` at import time. New `validate_config()` function checks all required vars and raises a clear error listing all missing ones. Called at entry points: [`lambda_handler.handler()`](lambda_handler.py) and [`summarizer.main()`](summarizer.py). This allows importing config for testing without a full `.env` file and prevents premature failures on Lambda cold start.
- **Lambda timeout guard**: Added wall-clock deadline tracking in [`lambda_handler.py`](lambda_handler.py) using `context.get_remaining_time_in_millis()` with a 10-second safety margin. Deadline is passed to [`run_summarizer()`](summarizer.py) as `_deadline` parameter. New `check_deadline()` and `DeadlineExceededError` in [`summarizer.py`](summarizer.py) — checked before channel processing and before group processing. When deadline is exceeded, the summarizer saves partial results and exits gracefully instead of being killed by Lambda timeout.
- **Event loop blocking fix**: Replaced `time.sleep()` with `await asyncio.sleep()` in [`call_openai()`](utils.py) retry backoff. The previous synchronous sleep blocked the entire event loop during OpenAI retry delays, preventing any other async work from progressing.
- **JSON format bug fix**: Fixed [`save_updated_summary()`](history_manager.py) — was saving raw list `[s.to_dict() for s in summaries]` instead of the expected `{"summaries": [...], "last_updated": "..."}` dict format, which would corrupt the file and cause parse errors on subsequent loads.
- **Removed unused import**: Removed `import time` from [`utils.py`](utils.py) (replaced by `asyncio.sleep`).
- **Tests added**: 4 new tests (total 47, up from 43):
  - `test_config_does_not_raise_on_import_without_env`: verifies config import succeeds without env vars
  - `test_validate_config_raises_for_missing_vars`: verifies `validate_config()` raises with clear message
  - `test_validate_config_passes_when_all_required_set`: verifies `validate_config()` passes with all vars
  - `test_handler_uses_context_remaining_time_for_deadline`: verifies Lambda context deadline calculation
- Updated [`tests/test_lambda_handler.py`](tests/test_lambda_handler.py) to stub `config.validate_config` and verify `_deadline` parameter.
- Updated [`tests/test_openai_config.py`](tests/test_openai_config.py) with new config validation tests.
- All 47 tests pass without errors.

## Completed in 2026-05-28 round (dedup bug fix & prompt optimization)

- **Critical bug**: Fixed three-band dedup logic in [`message_processor.py`](message_processor.py) — `SIMILARITY_THRESHOLD` (0.9) was checked before `SIMILARITY_LLM_UPPER` (0.95), making the upper band unreachable. Messages with ratio 0.9–0.95 were incorrectly auto-marked as duplicates without LLM verification. Now: ratio > UPPER (0.95) → auto-duplicate, LOWER (0.7) < ratio ≤ UPPER → LLM check, ratio ≤ LOWER → auto-different.
- **Removed SIMILARITY_THRESHOLD**: Deleted redundant `SIMILARITY_THRESHOLD = 0.9` from [`config.py`](config.py) — replaced by the two-band `SIMILARITY_LLM_LOWER` / `SIMILARITY_LLM_UPPER` constants. Updated [`tests/test_process_messages_integration.py`](tests/test_process_messages_integration.py) stubs accordingly.
- **Prompt optimization**: Reduced all prompts in [`prompts.py`](prompts.py) by ~50% (token savings):
  - `CHANNEL_SUMMARY_PROMPT`: 31 lines → 12 lines (removed redundant "ВАЖНО" prefixes, consolidated formatting rules)
  - `GROUP_SUMMARY_PROMPT`: 33 lines → 13 lines (same consolidation, kept group-specific rule about unanswered questions)
  - `NLP_RELEVANCE_PROMPT`: 44 lines → 24 lines (merged overlapping categories, compacted formatting)
  - `SUMMARY_COVERAGE_CHECK_PROMPT` / `GROUP_SUMMARY_COVERAGE_CHECK_PROMPT`: removed verbose "Ты эксперт" framing, kept essential rules and examples
  - `DUPLICATE_CHECK_PROMPT`: minor wording cleanup
- **Clean up**: Moved `from datetime import datetime, timezone` to module-level in [`message_processor.py`](message_processor.py), removed local import in `_create_summary_info()`.
- **Simplified `_run_async_with_loop`**: Extracted shared helper in [`history_manager.py`](history_manager.py) replacing 60+ lines of duplicated event-loop detection in `restore_summaries_from_channel_sync()` and `restore_group_summaries_from_channel_sync()` with a single 25-line function handling all three cases (no loop, same loop, different loop).
- **Tests added**: 2 new tests (total 49, up from 47):
  - `test_three_band_dedup_upper_threshold_uses_llm`: verifies ambiguous-ratio messages call LLM
  - `test_three_band_dedup_near_identical_skips_llm`: verifies near-identical messages auto-deduplicate without LLM
- All 49 tests pass without errors.

## Completed in 2026-05-29 round (dedup bug fix, cost optimization, code dedup)

- **Critical bug**: Fixed `are_messages_duplicate()` in [`message_processor.py`](message_processor.py) — was checking `answer.lower().startswith("y")` but `DUPLICATE_CHECK_PROMPT` asks "Ответь да или нет" (Russian). The LLM responds "да"/"нет", never "yes" — so LLM deduplication was **always skipped**. Changed to `answer.strip().lower().startswith("да")` and increased `max_tokens` from 1 to 3 to capture the Russian response.
- **Cost optimization — max output tokens**: Reduced `OPENAI_CHANNEL_SUMMARY_MAX_TOKENS` and `OPENAI_GROUP_SUMMARY_MAX_TOKENS` defaults from 16000 to 4000 in [`config.py`](config.py) and [`template.yaml`](template.yaml). With `SUMMARY_MAX_LENGTH=4000` chars and ~2-3 tokens/char for Russian, 4000 tokens is sufficient. This reduces output token cost by ~75% when the model generates verbose responses.
- **Cost optimization — update match truncation**: Added `UPDATE_MATCH_MAX_SUMMARIES` (default 5) and `UPDATE_MATCH_MAX_CHARS_PER_SUMMARY` (default 500) in [`config.py`](config.py). Applied in [`find_relevant_summary_for_update()`](history_manager.py) — previously sent last 50 full summaries as context (~50K+ chars). Now capped at ~2500 chars, saving ~95% input tokens on update match LLM calls.
- **Code dedup**: Merged `restore_summaries_from_channel()` and `restore_group_summaries_from_channel()` in [`history_manager.py`](history_manager.py) into shared `_restore_summaries_from_channel(history_file, label)` helper. Eliminated ~50 lines of duplicated async restore logic.
- **Tests added**: 4 new tests (total 55, up from 51):
  - `test_are_messages_duplicate_recognizes_russian_yes`: verifies "да" is recognized as duplicate
  - `test_are_messages_duplicate_recognizes_russian_no`: verifies "нет" is recognized as not duplicate
  - `test_update_match_limits_from_config`: verifies UPDATE_MATCH config constants
  - `test_find_relevant_summary_context_truncation`: verifies context truncation in update match
- All 55 tests pass without errors.

## Completed in 2026-05-28 round 2 (cost optimization, security hardening, prompt quality)

- **Major cost optimization — coverage check context truncation**: Added `COVERAGE_CHECK_MAX_SUMMARIES` (default 10) and `COVERAGE_CHECK_MAX_CHARS_PER_SUMMARY` (default 300) in [`config.py`](config.py). Applied in [`get_recent_summaries_context()`](history_manager.py) and [`get_recent_group_summaries_context()`](history_manager.py) — previously, these functions could send up to ~200K chars (channel) and ~1.2M chars (group) as input tokens to a yes/no coverage check LLM call. Now capped at ~3000 chars, saving ~90-99% input tokens on coverage checks.
- **Security: template.yaml NoEcho**: Set `NoEcho: true` on `TelegramApiId` and `TelegramApiHash` parameters in [`template.yaml`](template.yaml) — previously `NoEcho: false`, exposing credentials in CloudFormation console and API responses.
- **Security: Reserved concurrency**: Added `ReservedConcurrentExecutions: 1` to Lambda function in [`template.yaml`](template.yaml) — prevents concurrent invocations that could corrupt S3 state (non-atomic read-modify-write).
- **Prompt optimization round 2**: Further reduced token consumption in [`prompts.py`](prompts.py):
  - `FIND_RELEVANT_SUMMARY_PROMPT`: removed unnecessary "Ты эксперт по анализу текстов" framing
  - `CHANNEL_SUMMARY_PROMPT` / `GROUP_SUMMARY_PROMPT`: tightened emoji rule (1–5 → 1–3, removed 2 of 3 verbose example headers), merged redundant lines
  - `NLP_RELEVANCE_PROMPT`: consolidated overlapping categories (e.g., merged "Sam Altman, Mark Zuckerberg — любые новости" into BigTech line, removed "Незнакомые термины" catch-all), reduced by ~25%
- **Consistency fix**: Changed `restore_group_summaries_from_channel()` in [`history_manager.py`](history_manager.py) to use `msg.text` instead of `msg.message` for `content` and `extract_all_channels` — now consistent with the channel restore function. `msg.text` handles None safely.
- **OpenAI auth error clarity**: Added distinct logging for 401/403 auth errors in [`call_openai()`](utils.py) — now logs "OpenAI auth error (status 401): check OPENAI_API_KEY" instead of generic "OpenAI API error", making Lambda CloudWatch diagnostics faster.
- **Lambda request ID traceability**: Added `aws_request_id` to Lambda response and log output in [`lambda_handler.py`](lambda_handler.py) — enables correlating CloudWatch logs with Lambda invocation records.
- **Tests added**: 2 new tests (total 51, up from 49):
  - `test_handler_includes_request_id_in_response`: verifies request_id in Lambda response
  - `test_coverage_check_limits_from_config`: verifies COVERAGE_CHECK_MAX_SUMMARIES and COVERAGE_CHECK_MAX_CHARS_PER_SUMMARY in config
- Updated [`tests/test_lambda_handler.py`](tests/test_lambda_handler.py) for request_id in response structure.
- Updated [`tests/test_history_manager.py`](tests/test_history_manager.py) with context truncation tests.
- All 51 tests pass without errors.

## Completed in 2026-05-29 round 2 (Lambda hardening round 4, SSM secrets, code dedup)

- **Critical bug**: Fixed [`_run_async_with_loop()`](history_manager.py) — when called from within a running event loop (the common Lambda case via `asyncio.run()`), the old code silently returned `[]` due to "same loop deadlock avoidance", meaning `restore_summaries_from_channel_sync()` **never actually restored** anything. Now uses a background `threading.Thread` with `asyncio.run()` to avoid the deadlock while still executing the coroutine. Also added proper exception handling for the no-running-loop `asyncio.run()` path.
- **Code dedup**: Merged [`fetch_messages()`](telegram_client.py) and [`fetch_group_messages()`](telegram_client.py) into shared [`_fetch_from_sources()`](telegram_client.py) helper. Eliminated ~50 lines of duplicated message-fetching logic (the two functions differed only in source list, history loader, and log labels).
- **Code dedup**: Merged group/non-group branches in [`save_updated_summary()`](history_manager.py) into a single flow. Eliminated ~20 lines of duplicated summary-replacement logic.
- **Lambda memory**: Increased `MemorySize` from 512 to 1024 MB in [`template.yaml`](template.yaml). Lambda allocates CPU proportionally to memory, so this also speeds up execution and reduces timeout risk.
- **Secrets management — SSM Parameter Store**: Added optional SSM Parameter Store support for all four secrets (Telegram API ID/Hash, Bot Token, OpenAI API Key):
  - [`config.py`](config.py): New `_get_ssm_param()` and `_get_secret()` functions resolve secrets from SSM at runtime (supports key rotation). Falls back to env vars when SSM path is not configured or unavailable. `API_ID`, `API_HASH`, `BOT_TOKEN`, and `OPENAI_API_KEY` now use `_get_secret()`.
  - [`template.yaml`](template.yaml): New SSM path parameters (`TelegramApiIdSsmPath`, etc.) with conditional IAM policies and dual resolution (CloudFormation `{{resolve:ssm:...}}` at deploy time + runtime `boto3` for rotation).
- **Tests added**: 6 new tests (total 61, up from 55):
  - `test_run_async_with_loop_works_without_running_loop`: verifies basic coroutine execution
  - `test_run_async_with_loop_inside_running_loop`: verifies the fix for the same-loop deadlock
  - `test_run_async_with_loop_returns_empty_on_exception`: verifies graceful error handling
  - `test_get_secret_prefers_ssm_over_env`: verifies SSM takes precedence over env var
  - `test_get_secret_falls_back_to_env_when_ssm_empty`: verifies env var fallback when SSM path empty
  - `test_get_secret_falls_back_to_env_when_ssm_fails`: verifies env var fallback when SSM call fails
- All 61 tests pass without errors.

## Completed in 2026-06-01 round 2 (Lambda hardening round 7, caching, deadline, safety)

- **S3 client caching**: Added `_s3_client` module-level cache in [`s3_sync.py`](s3_sync.py) — `_get_s3_client()` now creates the boto3 S3 client once and reuses it (mirrors the SSM client caching already done in [`config.py`](config.py)). Previously created a new client per call (2 calls per Lambda invocation: download + upload).
- **In-memory history caching**: Added `_cache` dict and `invalidate_cache()` in [`history_manager.py`](history_manager.py). `load_summaries_history()` and `load_group_summaries_history()` now cache their results in memory, avoiding repeated disk reads within a single Lambda invocation. Cache is invalidated on every save operation (`save_summary_to_history`, `save_group_summary_to_history`, `save_updated_summary`, `_restore_summaries_from_channel`) and on S3 download (via `invalidate_cache()` call in [`s3_sync.py`](s3_sync.py)).
- **Deadline check during message iteration**: Added deadline check inside `async for msg in user_client.iter_messages(...)` loop in [`_fetch_from_sources()`](telegram_client.py) — previously only checked before each source, not between messages. If a single channel has thousands of messages, the Lambda could now exceed its timeout mid-fetch. The new check breaks out of the iteration loop early, returning already-fetched messages.
- **None-safe text handling**: Added `_text_hash()` helper in [`message_processor.py`](message_processor.py) and [`history_manager.py`](history_manager.py) that handles `None` text gracefully (`hashlib.sha256((text or "").encode())`). Also made [`MessageInfo.from_dict()`](models.py) defensive: `text`, `channel`, `link` default to `""` when `None` or missing, and `date` falls back to `datetime.now(timezone.utc)` when absent. Prevents `AttributeError` on `.encode()` if deserialized data has null fields.
- **Tests added**: 8 new tests (total 74, up from 66):
  - `test_from_dict_handles_none_text`: verifies `MessageInfo.from_dict` with `None` text
  - `test_from_dict_handles_missing_text`: verifies `MessageInfo.from_dict` without text key
  - `test_from_dict_handles_none_channel`: verifies `MessageInfo.from_dict` with `None` channel
  - `test_get_s3_client_caches_client`: verifies S3 client is cached and reused
  - `test_invalidate_cache_clears_specific_key`: verifies targeted cache invalidation
  - `test_invalidate_cache_clears_all_without_filepath`: verifies full cache invalidation
  - `test_text_hash_with_normal_text`: verifies deterministic hashing
  - `test_text_hash_with_none_returns_same_as_empty`: verifies None → empty string equivalence
- All 74 tests pass without errors.

## Completed in 2026-06-01 round (Lambda hardening round 6, async OpenAI, bug fixes, code dedup)

- **Critical fix**: Replaced sync `OpenAI` with `AsyncOpenAI` in [`utils.py`](utils.py). The sync client blocked the entire event loop during API calls (5-30s each), making deadline checks unreliable and preventing concurrent async work. Now `await openai_client.chat.completions.create(...)` properly yields control, so `check_deadline()` and Telegram client operations can progress during OpenAI wait times.
- **Bug fix**: Fixed [`update_existing_summary()`](history_manager.py) — was not carrying over `message_id` from the original summary to the updated one. This meant subsequent edits to the same channel message would fail because `save_updated_summary()` couldn't find the message by ID.
- **Code dedup**: Merged `is_message_covered_in_summaries()` and `is_message_covered_in_group_summaries()` in [`message_processor.py`](message_processor.py) into shared `_check_coverage()` helper. Eliminated ~20 lines of duplicated coverage-check logic (both differed only in prompt, context function, and label).
- **Code cleanup**: Removed unused `duplicate_label` parameter from [`_remove_duplicates_generic()`](message_processor.py) — was passed by `remove_duplicates()` and `remove_group_duplicates()` but never referenced.
- **Performance**: Cached SSM client in [`config.py`](config.py) — `_get_ssm_client()` now creates the boto3 SSM client once and reuses it, instead of creating a new client per `_get_ssm_param()` call (up to 4 calls per Lambda invocation).
- **Tests added**: 2 new tests (total 66, up from 64):
  - `test_update_preserves_message_id`: verifies `update_existing_summary` carries over `message_id` from original summary
  - `test_get_ssm_client_caches_client`: verifies SSM client is cached and reused across calls
- Updated test stubs in [`tests/test_openai_config.py`](tests/test_openai_config.py) and [`tests/test_summary_length_guardrails.py`](tests/test_summary_length_guardrails.py) to include `AsyncOpenAI` alongside `OpenAI`.
- All 66 tests pass without errors.

## Completed in 2026-05-30 round (Lambda hardening round 5, cost optimization)

- **Bug fix**: Fixed [`save_updated_summary()`](history_manager.py) matching — was using fragile `summary.content == original_summary.content` comparison which could match wrong summary if content happens to be identical. Now matches by `message_id` first (reliable unique key), falls back to content+date+count when `message_id` is None.
- **Lambda hardening**: Added deadline-aware fetching in [`telegram_client.py`](telegram_client.py) — [`_fetch_from_sources()`](telegram_client.py) now accepts `_deadline` parameter and checks it before each source, returning already-fetched messages if deadline is exceeded. [`fetch_messages()`](telegram_client.py) and [`fetch_group_messages()`](telegram_client.py) pass the deadline through from [`run_summarizer()`](summarizer.py). This prevents Lambda timeout during slow message fetching from Telegram API.
- **Dead code removal**: Removed unused [`send_message_to_target_channel()`](telegram_client.py) — the function without ID return was never called; only `send_message_to_target_channel_with_id()` is used.
- **Cost optimization — max_tokens for yes/no responses**: Reduced `max_tokens` for coverage check LLM calls in [`message_processor.py`](message_processor.py) from 10 → 5 (responses are only "ДА" or "НЕТ"). Reduced `max_tokens` for `find_relevant_summary_for_update()` in [`history_manager.py`](history_manager.py) from 5 → 3 (response is just a digit or "НЕТ").
- **Tests added**: 3 new tests (total 64, up from 61):
  - `test_match_by_message_id_takes_precedence`: verifies message_id-based matching in save_updated_summary
  - `test_fallback_to_content_date_count_when_no_message_id`: verifies fallback matching when message_id is None
  - `test_handler_passes_deadline_to_run_summarizer`: verifies deadline is passed through from Lambda handler
- Updated test stubs in [`tests/test_process_messages_integration.py`](tests/test_process_messages_integration.py) with `FIND_RELEVANT_SUMMARY_PROMPT` and `ENABLE_SUMMARY_UPDATES` config.
- All 64 tests pass without errors.

## Completed in 2026-06-02 round (bug fixes, code dedup, cost optimization)

- **Bug fix**: Added cache invalidation in [`save_summarization_history()`](history_manager.py) and [`save_group_summarization_history()`](history_manager.py) — previously, saving updated data to disk didn't clear the in-memory cache, so subsequent reads within the same Lambda invocation returned stale data.
- **Code dedup**: Moved `_text_hash()` from [`message_processor.py`](message_processor.py) and [`history_manager.py`](history_manager.py) to [`utils.py`](utils.py) as `text_hash()`. Eliminated duplicate implementation and removed unused `hashlib` imports from both modules.
- **Defensive deserialization**: Made [`SummaryInfo.from_dict()`](models.py) handle `None`/missing fields (`content`, `channels`, `date`, `message_count`) gracefully — now consistent with `MessageInfo.from_dict()` which was already hardened.
- **Cost optimization**: Reduced `max_tokens` for coverage check LLM calls in [`_check_coverage()`](message_processor.py) from 5 → 2 (responses are only "ДА"/"НЕТ"). Reduced `max_tokens` for NLP relevance check from 30 → 20. Saves ~30-40% output tokens on these yes/no classification calls.
- **Prompt fix**: Made [`find_relevant_summary_for_update()`](history_manager.py) prompt dynamic — previously hardcoded "(1, 2, 3)" but actual count varies with `UPDATE_MATCH_MAX_SUMMARIES`. Now correctly shows available indices.
- **Tests added**: 9 new tests (total 81, up from 74):
  - `test_from_dict_handles_none_content`: verifies `SummaryInfo.from_dict` with `None` content
  - `test_from_dict_handles_missing_content`: verifies `SummaryInfo.from_dict` without content key
  - `test_from_dict_handles_none_channels`: verifies `SummaryInfo.from_dict` with `None` channels
  - `test_from_dict_handles_missing_date`: verifies `SummaryInfo.from_dict` without date key
  - `test_from_dict_handles_missing_message_count`: verifies `SummaryInfo.from_dict` without message_count
  - `test_save_summarization_history_invalidates_cache`: verifies cache cleared after save
  - `test_save_group_summarization_history_invalidates_cache`: verifies cache cleared after save
  - `test_text_hash_with_normal_text`: verifies deterministic hashing (moved from stub-based to inline)
  - `test_text_hash_with_none_returns_same_as_empty`: verifies None → empty string equivalence (moved from stub-based to inline)
- Updated test stubs in [`tests/test_history_manager.py`](tests/test_history_manager.py), [`tests/test_process_messages_integration.py`](tests/test_process_messages_integration.py), [`tests/test_digest_post_processing.py`](tests/test_digest_post_processing.py) to include `text_hash` in fake `utils` module.
- All 81 tests pass without errors.

## Completed in 2026-06-03 round (bug fixes, prompt optimization, context consistency)

- **Bug fix**: Fixed [`load_summaries_history()`](history_manager.py) — previously triggered an expensive channel restore whenever the history file was valid but empty (`{"summaries": []}`). Now only triggers restore when the file is missing or corrupt (`data is None`), consistent with `load_group_summaries_history()` which already handled this correctly. An empty-but-valid history file (e.g., after clearing history) now returns `[]` without unnecessary API calls.
- **Bug fix**: Fixed [`save_updated_summary()`](history_manager.py) — previously continued with save and Telegram edit even when no matching summary was found in history, leading to no-op disk writes and misleading channel edit attempts. Now tracks `found` flag and returns early with a warning log if no match is found.
- **Consistency fix**: Unified context formatting in [`get_recent_summaries_context()`](history_manager.py) — now uses the same structured "Дата: / Содержание: / ---" format as `get_recent_group_summaries_context()`. Previously used bare `\n\n`-joined content without dates, causing inconsistent LLM coverage check behavior between channel and group paths.
- **Prompt optimization**: Tightened [`NLP_RELEVANCE_PROMPT`](prompts.py) — consolidated overlapping categories (merged "AI/BigTech новости" and BigTech line, merged "Карьера, вакансии" into one line, removed "новые детали уже вышедших моделей/библиотек" as redundant, removed "подготовительные" from ШАД exclusion). Reduced token count by ~15%.
- **Prompt optimization**: Tightened [`CHANNEL_SUMMARY_PROMPT`](prompts.py) and [`GROUP_SUMMARY_PROMPT`](prompts.py) — removed redundant "Кратко описывай архитектуру, методологию, результаты" instruction (already implied by "краткий дайджест"), removed "яркие" from header rule (redundant with "короткие").
- **Tests added**: 4 new tests (total 91, up from 87):
  - `test_no_restore_when_file_has_empty_summaries`: verifies empty-but-valid file does NOT trigger channel restore
  - `test_restore_when_file_missing`: verifies missing/corrupt file DOES trigger channel restore
  - `test_save_updated_summary_skips_when_no_match`: verifies no save/edit when original summary not found
  - `test_recent_summaries_context_includes_date`: verifies channel context now includes "Дата:" like group version
- All 91 tests pass without errors.

## Completed in 2026-06-03 round 2 (LLM update integration, deterministic classification, code dedup, prompt tightening)

- **Summary update quality**: Improved [`update_existing_summary()`](history_manager.py) — replaced mechanical "Другие ссылки:" append with LLM-integrated update. The LLM now inserts the new link next to the relevant paragraph in the summary. Falls back to the append approach on LLM failure or empty response. This addresses the "Summary update quality" next action from previous round.
- **Deterministic classification**: Added `temperature=0` parameter to all classification LLM calls in [`message_processor.py`](message_processor.py) and [`history_manager.py`](history_manager.py):
  - `are_messages_duplicate()`: `temperature=0` for duplicate detection
  - `_check_coverage()`: `temperature=0` for coverage check
  - `is_nlp_related()`: `temperature=0` for NLP relevance classification
  - `find_relevant_summary_for_update()`: `temperature=0` for summary matching
  - Added `temperature` parameter to [`call_openai()`](utils.py) — passed through to `chat.completions.create()` when provided, omitted when `None` (default). Classification calls now produce deterministic results, reducing variance and wasted retries.
- **Code dedup**: Merged [`get_recent_summaries_context()`](history_manager.py) and [`get_recent_group_summaries_context()`](history_manager.py) into shared [`_get_recent_summaries_context()`](history_manager.py) helper. Both functions now delegate to the same formatting logic, eliminating ~20 lines of duplicated context-building code.
- **Prompt optimization**: Tightened [`NLP_RELEVANCE_PROMPT`](prompts.py) — merged BigTech and AI-ассистенты lines into one (redundant split), merged "обучение/дообучение" into "дообучение" (shorter). Reduced token count by ~10%.
- **Unused imports removed**: Removed unused `import os` from [`channel_manager.py`](channel_manager.py) and unused imports (`load_discovered_channels`, `load_similar_channels`, `load_banned_channels`, `get_all_source_channels`) from [`message_processor.py`](message_processor.py).
- **Tests added**: 7 new tests (total 98, up from 91):
  - `test_call_openai_passes_temperature_to_api`: verifies `temperature=0` is passed to API create call
  - `test_call_openai_omits_temperature_when_none`: verifies `temperature` is omitted when not specified
  - `test_update_existing_summary_uses_llm`: verifies LLM is called for summary integration
  - `test_update_existing_summary_fallback_on_llm_failure`: verifies fallback to append on LLM exception
  - `test_update_existing_summary_fallback_on_empty_llm_response`: verifies fallback on empty LLM response
  - `test_shared_helper_returns_empty_for_no_summaries`: verifies `_get_recent_summaries_context` returns empty for no data
  - `test_shared_helper_truncates_long_content`: verifies content truncation in shared helper
- All 98 tests pass without errors.

## Completed in 2026-06-04 round (Lambda hardening round 9, bug fixes, cost optimization, observability)

- **Bug fix**: Fixed [`load_group_summaries_history()`](history_manager.py) — previously returned `[]` when file was missing or corrupt, unlike [`load_summaries_history()`](history_manager.py) which restores from channel. Now both functions consistently attempt channel restore when `data is None`, preventing data loss after group history file corruption.
- **Cost optimization**: Added `UPDATE_SUMMARY_MAX_TOKENS` config (default 500) in [`config.py`](config.py). Applied in [`update_existing_summary()`](history_manager.py) — previously used `OPENAI_CHANNEL_SUMMARY_MAX_TOKENS` (4000) for a simple link insertion, wasting ~87% output tokens. 500 tokens is sufficient for inserting one link into an existing summary.
- **Lambda duration logging**: Added `time.monotonic()` tracking in [`lambda_handler.handler()`](lambda_handler.py) — now logs `"Lambda completed in X.Xs"` on success and `"Lambda execution failed after X.Xs"` on error. Enables CloudWatch-based performance monitoring and timeout trend detection.
- **OpenAI API key early validation**: Added check in [`call_openai()`](utils.py) — returns `""` immediately with clear error log when `OPENAI_API_KEY` is not set, instead of creating an `AsyncOpenAI` client that would fail with a confusing 401 error.
- **SAM template updated**: Added `UpdateSummaryMaxTokens` parameter and env var in [`template.yaml`](template.yaml).
- **`.env.example` synced**: Added `UPDATE_SUMMARY_MAX_TOKENS=500` in [`.env.example`](.env.example).
- **Tests added**: 7 new tests (total 105, up from 98):
  - `test_config_reads_update_summary_max_tokens_from_env`: verifies `UPDATE_SUMMARY_MAX_TOKENS` env var parsing
  - `test_call_openai_returns_empty_when_api_key_missing`: verifies early return when API key is None
  - `test_handler_logs_duration_on_success`: verifies completion duration log
  - `test_handler_logs_duration_on_error`: verifies error path includes duration
  - `test_no_restore_when_file_has_empty_summaries` (group): verifies empty-but-valid group file does NOT trigger restore
  - `test_restore_when_file_missing` (group): verifies missing group file DOES trigger restore
  - `test_update_existing_summary_uses_update_max_tokens`: verifies `UPDATE_SUMMARY_MAX_TOKENS` (500) used instead of 4000
- All 105 tests pass without errors.

## Completed in 2026-06-05 round (Lambda hardening round 10, summary quality, code dedup)

- **Summary temperature**: Added `OPENAI_SUMMARY_TEMPERATURE` config (default 0.3) in [`config.py`](config.py). Applied in [`summarize_text()`](message_processor.py) and [`summarize_group_text()`](message_processor.py) — previously used OpenAI's default temperature (1.0), which produced inconsistent and often verbose summaries. A low temperature (0.3) produces more concise, consistent output, directly addressing the AUTOWORK_INSTRUCTIONS goal of improving quality and conciseness. Configurable via env for A/B testing.
- **Deterministic summary updates**: Added `temperature=0` to [`update_existing_summary()`](history_manager.py) — link insertion is a deterministic edit operation; randomness was unnecessary and could cause inconsistent updates.
- **Code dedup**: Merged `_ensure_clients()` and `_ensure_bot_client()` in [`telegram_client.py`](telegram_client.py) — both were identical wrappers that called `start_clients()` when client was disconnected. `_ensure_bot_client()` now delegates to `_ensure_clients()`.
- **Code dedup**: Extracted shared `_load_processed_messages()` and `_save_processed_messages()` helpers in [`history_manager.py`](history_manager.py). `load_summarization_history()` / `load_group_summarization_history()` and `save_summarization_history()` / `save_group_summarization_history()` now delegate to these shared helpers, eliminating ~40 lines of duplicated message-ID set building and append-truncate-save logic.
- **SAM template updated**: Added `OpenAISummaryTemperature` parameter and env var in [`template.yaml`](template.yaml).
- **`.env.example` synced**: Added `OPENAI_SUMMARY_TEMPERATURE=0.3` in [`.env.example`](.env.example).
- **Tests added**: 7 new tests (total 112, up from 105):
  - `test_config_reads_summary_temperature_from_env`: verifies `OPENAI_SUMMARY_TEMPERATURE` env var parsing
  - `test_config_summary_temperature_default`: verifies default value is 0.3
  - `test_summarize_text_passes_temperature`: verifies temperature passed to `call_openai` for channel summaries
  - `test_summarize_group_text_passes_temperature`: verifies temperature passed for group summaries
  - `test_update_existing_summary_uses_temperature_zero`: verifies `temperature=0` for deterministic link insertion
  - `test_load_processed_messages_returns_set`: verifies shared load helper returns set of message IDs
  - `test_save_processed_messages_appends_and_truncates`: verifies shared save helper appends and truncates
- Updated test stubs in [`tests/test_process_messages_integration.py`](tests/test_process_messages_integration.py) to include `OPENAI_SUMMARY_TEMPERATURE`.
- All 112 tests pass without errors.

## Completed in 2026-06-05 round 2 (Lambda hardening round 11, quality, dedup, cost, performance)

- **Bug fix**: Fixed [`load_group_summaries_history()`](history_manager.py) — previously returned `[]` without caching when the file had valid but empty data (`{"summaries": []}`). Every subsequent call re-read the file from disk. Now caches empty results consistently with [`load_summaries_history()`](history_manager.py), and adds `sorted()` for consistency.
- **Intra-batch dedup**: Added [`_remove_intra_batch_duplicates()`](message_processor.py) — lightweight SequenceMatcher-only dedup that removes obvious text duplicates (ratio > 0.95) and link duplicates within a single batch, without LLM calls. Applied in [`process_messages()`](message_processor.py) before summarization. Previously, two channels posting the same news would both be included in full, wasting input tokens and producing redundant summaries. Now near-identical messages are collapsed before the LLM sees them, directly improving summary conciseness (per AUTOWORK_INSTRUCTIONS goal).
- **Code dedup**: Merged [`save_summary_to_history()`](history_manager.py) and [`save_group_summary_to_history()`](history_manager.py) into shared [`_save_summary_to_history_file()`](history_manager.py) helper. The channel version previously loaded and re-serialized all existing `SummaryInfo` objects on every save; the new helper uses raw JSON append (like the group version already did), which is more efficient and avoids unnecessary parse/serialize overhead.
- **Cost optimization — summary input truncation**: Added `SUMMARY_MAX_INPUT_CHARS_PER_MESSAGE` config (default 3000) in [`config.py`](config.py). Applied in [`_prepare_messages_text()`](message_processor.py) — each message's text is now truncated before being sent to the summary LLM. Previously, a single very long message (full article) could consume the entire input token budget, leaving little room for other messages and producing incomplete summaries.
- **Performance — parallel NLP classification**: Added `NLP_CONCURRENT_CHECKS` config (default 5) in [`config.py`](config.py). Refactored [`process_messages()`](message_processor.py) to use `asyncio.gather` with a semaphore for NLP relevance checks — all messages are now classified concurrently (up to `NLP_CONCURRENT_CHECKS` at a time), reducing wall-clock time from N × OpenAI latency to ~N/5 × latency. Coverage checks and state-mutating operations remain sequential for safety.
- **Dead code removed**: Removed [`_classify_message()`](message_processor.py) — its logic was inlined into `process_messages()` to enable the parallel NLP check split (parallel NLP phase, then sequential coverage/update phase).
- **SAM template updated**: Added `SummaryMaxInputCharsPerMessage` and `NlpConcurrentChecks` parameters and env vars in [`template.yaml`](template.yaml).
- **`.env.example` synced**: Added `SUMMARY_MAX_INPUT_CHARS_PER_MESSAGE=3000` and `NLP_CONCURRENT_CHECKS=5` in [`.env.example`](.env.example).
- **Tests added**: 13 new tests (total 125, up from 112):
  - `test_load_group_summaries_caches_empty_results`: verifies empty group summaries are cached (not re-read from disk)
  - `test_save_summary_to_history_file_appends_and_truncates`: verifies shared save helper appends and truncates
  - `test_removes_identical_messages`: verifies intra-batch dedup removes identical messages
  - `test_keeps_different_messages`: verifies intra-batch dedup keeps different messages
  - `test_removes_link_duplicates`: verifies intra-batch dedup removes same-link messages
  - `test_keeps_different_links`: verifies intra-batch dedup keeps different-link messages
  - `test_empty_input_returns_empty`: verifies intra-batch dedup handles empty input
  - `test_long_message_is_truncated`: verifies _prepare_messages_text truncates long messages
  - `test_short_message_is_not_truncated`: verifies _prepare_messages_text keeps short messages
  - `test_config_reads_summary_max_input_chars_from_env`: verifies `SUMMARY_MAX_INPUT_CHARS_PER_MESSAGE` env var parsing
  - `test_config_summary_max_input_chars_default`: verifies default is 3000
  - `test_config_reads_nlp_concurrent_checks_from_env`: verifies `NLP_CONCURRENT_CHECKS` env var parsing
  - `test_config_nlp_concurrent_checks_default`: verifies default is 5
- Updated test stubs in [`tests/test_process_messages_integration.py`](tests/test_process_messages_integration.py) to include `SUMMARY_MAX_INPUT_CHARS_PER_MESSAGE` and `NLP_CONCURRENT_CHECKS`.
- All 125 tests pass without errors.

## Completed in 2026-06-08 round (Lambda hardening round 12, parallel coverage, dead code removal, logging fix)

- **Parallel coverage checks**: Refactored [`process_messages()`](message_processor.py) to run coverage checks in parallel using `asyncio.Semaphore` and `asyncio.gather`, mirroring the parallel NLP check pattern. Previously, coverage checks were sequential — each NLP-related message's coverage was checked one at a time, adding N × OpenAI latency to wall-clock time. Now all coverage checks run concurrently (up to `NLP_CONCURRENT_CHECKS` at a time), reducing wall-clock time from N × latency to ~N/5 × latency. State-mutating operations (`process_covered_message`) remain sequential for safety.
- **Dead code removal**: Removed [`remove_duplicates()`](message_processor.py), [`remove_group_duplicates()`](message_processor.py), [`_remove_duplicates_generic()`](message_processor.py), and [`are_messages_duplicate()`](message_processor.py) — these functions were superseded by inline dedup logic in `process_messages()` and were only called from tests, never from the production code path. The production flow uses `_remove_intra_batch_duplicates()` (SequenceMatcher-only, no LLM) plus the parallel coverage check.
- **Dead code removal**: Removed unused `SIMILARITY_LLM_LOWER` import from [`message_processor.py`](message_processor.py) — only `SIMILARITY_LLM_UPPER` is still used by `_remove_intra_batch_duplicates()`.
- **Import cleanup**: Moved `import asyncio` and `from config import SOURCE_CHANNELS` to module level in [`message_processor.py`](message_processor.py), removing local imports inside `process_messages()`.
- **Logging fix**: Replaced `traceback.print_exc()` with `exc_info=True` in [`summarizer.py`](summarizer.py) error handler — structured traceback in CloudWatch logs instead of unstructured stdout output. Removed unused `import traceback`.
- **Tests updated**: Removed 4 obsolete tests (2 three-band dedup tests that tested removed `_remove_duplicates_generic`, 2 Russian response tests that tested removed `are_messages_duplicate`). Added 2 new tests:
  - `test_coverage_checks_run_in_parallel`: verifies coverage check is called for each NLP-related message
  - `test_covered_messages_excluded_from_summary`: verifies covered messages are excluded from summary generation
  - Total test count: 123 (down from 125 — net -2 from 4 removed + 2 added)
- All 123 tests pass without errors.

## Completed in 2026-06-08 round 2 (Lambda hardening round 13, bug fixes, observability, cleanup)

- **Critical bug**: Fixed [`summarize_text()`](message_processor.py) and [`summarize_group_text()`](message_processor.py) — previously returned error string `"Ошибка: Не удалось сгенерировать обобщение"` on OpenAI failure. This error message was both posted to the Telegram channel AND saved to summary history, polluting the channel with error text and corrupting history files. Now returns `None` on failure, and [`process_messages()`](message_processor.py) skips Telegram send when summary is `None` (`if summary and send_message:` instead of `if send_message:`). The existing `if summary and unique_messages:` guard in [`_save_processing_results()`](message_processor.py) already handled `None` correctly.
- **Coverage check robustness**: Changed `_check_coverage()` in [`message_processor.py`](message_processor.py) from `result.strip().upper() == "ДА"` to `.startswith("ДА")` — consistent with `is_nlp_related()` which already uses `.startswith("да")`. Handles model variations like "ДА." or "ДА, тема совпадает" that would fail strict equality.
- **OpenAI token usage logging**: Added structured logging of token usage from API responses in [`call_openai()`](utils.py) — logs `model`, `prompt_tokens`, `completion_tokens`, and `total_tokens` at INFO level. Enables cost monitoring and trend detection in CloudWatch without additional API calls.
- **Lambda response enrichment**: Added `elapsed_seconds` field to Lambda handler response in [`lambda_handler.py`](lambda_handler.py) — present in both success and error responses. Enables CloudWatch metric filters and invocation-level performance tracking.
- **Dead config cleanup**: Removed redundant `send_message` environment variable from [`template.yaml`](template.yaml) — the `Environment.Variables.send_message` entry was never read by any code (the `send_message` flag is parsed from the event payload, not from env vars).
- **Tests added**: 9 new tests (total 132, up from 123):
  - `test_summarize_text_returns_none_on_empty_openai_response`: verifies `None` return on failure
  - `test_summarize_group_text_returns_none_on_empty_openai_response`: verifies `None` return on failure
  - `test_process_messages_skips_send_when_summary_is_none`: verifies Telegram send is skipped when summary fails
  - `test_coverage_check_matches_da_with_period`: verifies `.startswith("ДА")` handles "ДА."
  - `test_coverage_check_matches_da_with_comma`: verifies `.startswith("ДА")` handles "ДА, тема совпадает"
  - `test_coverage_check_rejects_net`: verifies "НЕТ" is correctly rejected
  - `test_call_openai_logs_token_usage`: verifies token usage logging in CloudWatch
  - `test_handler_includes_elapsed_seconds_on_success`: verifies `elapsed_seconds` in success response
  - `test_handler_includes_elapsed_seconds_on_error`: verifies `elapsed_seconds` in error response
- Updated [`tests/test_lambda_handler.py`](tests/test_lambda_handler.py) — changed response assertion from exact dict match to key-level checks (more resilient to new fields).
- All 132 tests pass without errors.

## Completed in 2026-06-09 round (Lambda hardening round 14, bug fixes, cost optimization, quality)

- **Critical bug**: Fixed [`update_existing_summary()`](history_manager.py) — `UPDATE_SUMMARY_MAX_TOKENS` was 500, but the prompt asks the LLM to return the full updated summary ("Ответь только обновлённое саммари"). For a typical 2000-char Russian summary (~1000 tokens), 500 max_tokens would truncate the response, causing data loss — the shortened text replaced the original. Increased default from 500 to 2000 in [`config.py`](config.py). Added length guard: if the LLM response is less than 80% of the original content length, falls back to the append approach instead of using the truncated response.
- **Critical bug**: Fixed [`process_messages()`](message_processor.py) — `nlp_related_messages.append(msg)` was outside the `for` loop (indentation error from the parallel NLP check refactor in round 12). Only the last message in each batch was being appended to `nlp_related_messages`, so all other NLP-related messages were silently dropped. This caused missing content in summaries whenever there were multiple messages in a batch.
- **Cost optimization**: Reordered [`process_messages()`](message_processor.py) — intra-batch dedup now runs before coverage checks. Previously, all NLP-related messages got coverage checks (expensive LLM calls), and then intra-batch dedup removed duplicates after. Now duplicates are removed first (free SequenceMatcher check), saving LLM coverage check tokens on duplicate messages.
- **Cost optimization**: Coverage context is now built once per [`process_messages()`](message_processor.py) call instead of per-message. Previously, `get_recent_summaries_context()` was called inside each `is_message_covered_in_summaries()` call, rebuilding the context string for every message. Now pre-computed once and passed to all parallel `_check_coverage()` calls.
- **Operational**: Made `ENABLE_SUMMARIES_DEDUPLICATION` and `ENABLE_SUMMARY_UPDATES` configurable via env vars in [`config.py`](config.py) (default: `true`). Previously hardcoded to `True`, making it impossible to disable coverage dedup or summary updates for testing or cost saving without code changes.
- **Link extraction**: Fixed [`extract_links()`](utils.py) — added `TRAILING_PUNCTUATION_REGEX` to strip trailing punctuation (`.`, `,`, `)`, `]`, `}`, etc.) from extracted URLs. Previously, `https://example.com/page).` would capture the closing parenthesis and period as part of the URL, causing incorrect link handling in summary generation and dedup.
- **Consistency**: Replaced `msg.message` with `msg.text` in [`_restore_summaries_from_channel()`](history_manager.py) — both refer to the same property in Telethon, but `msg.text` is the canonical accessor and consistent with the rest of the codebase.
- **SAM template updated**: Updated `UpdateSummaryMaxTokens` default from "500" to "2000" and added `EnableSummariesDeduplication` / `EnableSummaryUpdates` parameters and env vars in [`template.yaml`](template.yaml).
- **`.env.example` synced**: Updated `UPDATE_SUMMARY_MAX_TOKENS=2000` and added `ENABLE_SUMMARIES_DEDUPLICATION=true` / `ENABLE_SUMMARY_UPDATES=true` in [`.env.example`](.env.example).
- **Tests added**: 14 new tests (total 146, up from 132):
  - `test_falls_back_when_llm_response_truncated`: verifies append fallback on truncated LLM response
  - `test_keeps_llm_response_when_length_sufficient`: verifies LLM response kept when >= 80% of original
  - `test_enable_dedup_defaults_true`: verifies `ENABLE_SUMMARIES_DEDUPLICATION` default
  - `test_enable_dedup_can_be_disabled`: verifies `ENABLE_SUMMARIES_DEDUPLICATION=false` works
  - `test_enable_updates_defaults_true`: verifies `ENABLE_SUMMARY_UPDATES` default
  - `test_enable_updates_can_be_disabled`: verifies `ENABLE_SUMMARY_UPDATES=0` works
  - `test_strips_trailing_period`: verifies URL trailing period stripped
  - `test_strips_trailing_parenthesis`: verifies URL trailing parenthesis stripped
  - `test_strips_trailing_comma`: verifies URL trailing comma stripped
  - `test_preserves_clean_url`: verifies clean URLs unchanged
  - `test_strips_multiple_trailing_punctuation`: verifies multiple trailing chars stripped
  - `test_dedup_before_coverage_saves_llm_calls`: verifies coverage check only on deduped messages
  - `test_all_nlp_messages_appended`: regression test for the indentation bug
  - `test_config_update_summary_max_tokens_default`: verifies `UPDATE_SUMMARY_MAX_TOKENS` default is 2000
- Updated test stubs in [`tests/test_process_messages_integration.py`](tests/test_process_messages_integration.py) to patch `_check_coverage` instead of `is_message_covered_in_summaries` (coverage checks now use pre-built context).
- All 146 tests pass without errors.

## Next actions

- **CI/CD**: Настроить GitHub Actions CI/CD для автоматического деплоя Lambda при мердже в main.
- **Secrets management**: Перенести реальные секреты в AWS SSM Parameter Store (инфраструктура готова — `*_SSM_PATH` env vars и IAM policies в template.yaml).
- **Prompt A/B testing**: Продолжить тестирование промптов — отслеживать качество саммари после снижения `max_tokens` до 4000 и при необходимости корректировать.
- **OpenAI response streaming**: Рассмотреть streaming API для снижения perceived latency (но не стоимости — `max_tokens` уже ограничен).
- **Coverage check prompt**: Рассмотреть замену coverage check промптов на JSON mode (`response_format={"type": "json_object"}`) для ещё большей детерминистичности при сохранении стоимости gpt-4o-mini.
- **Intra-batch LLM dedup**: Рассмотреть добавление LLM-проверки для сообщений с ratio между SIMILARITY_LLM_LOWER (0.7) и SIMILARITY_LLM_UPPER (0.95) внутри `_remove_intra_batch_duplicates` — позволит ловить больше дубликатов за дополнительные токены. (Примечание: `are_messages_duplicate` и `_remove_duplicates_generic` удалены в раунде 2026-06-08 — при необходимости восстановления, см. git history.)
