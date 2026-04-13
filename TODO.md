# TODO

Живой список задач для автоматических раундов.

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

## Next actions

- Настроить GitHub Actions CI/CD для автоматического деплоя Lambda при мердже в main.
- Перенести чувствительные переменные в AWS SSM Parameter Store / Secrets Manager вместо env vars.
- Разделить `process_messages()` на более мелкие функции в [`message_processor.py`](message_processor.py) — частично выполнено, можно продолжить с `_replace_source_with_links` и `_prepare_messages_text`.
