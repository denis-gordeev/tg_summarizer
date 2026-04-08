# TODO

Живой список задач для автоматических раундов.

## Completed in 2026-04-08 round

- Восстановлен [`s3_sync.py`](s3_sync.py), без которого чистый импорт [`lambda_handler.py`](lambda_handler.py) падал с `ModuleNotFoundError`.
- Исправлен разбор Lambda event-флагов в [`lambda_handler.py`](lambda_handler.py): строковые значения `false` / `true` теперь интерпретируются корректно.
- Восстановлены и расширены smoke/regression-тесты для Lambda и OpenAI-конфига в [`tests/test_lambda_handler.py`](tests/test_lambda_handler.py), [`tests/test_s3_sync.py`](tests/test_s3_sync.py) и [`tests/test_openai_config.py`](tests/test_openai_config.py).
- Добавлены guardrails на длину итогового дайджеста в [`message_processor.py`](message_processor.py), чтобы итоговые саммари оставались лаконичными даже если модель пытается выйти за целевой объём.
- Исправлены и документированы OpenAI env-настройки с дефолтом `gpt-4o-mini` в [`config.py`](config.py), [`utils.py`](utils.py) и [`.env.example`](.env.example).
- Добавлен отдельный Lambda runbook [`docs/aws-lambda-runbook.md`](docs/aws-lambda-runbook.md), а [`README.md`](README.md) синхронизирован с текущим деревом проекта.

## Next actions

- Подключить `python3 -m unittest discover -s tests` к CI или хотя бы к локальному pre-push сценарию, чтобы проверки не оставались ручными.
- Добавить интеграционный тест на post-processing дайджеста целиком, включая замену `[1]` на HTML-ссылки и финальное ограничение длины.
- Добавить воспроизводимый infra-шаблон для AWS Lambda deploy (AWS SAM или Terraform), чтобы runbook не оставался только manual-процедурой.
- Проверить, нужно ли явно добавить `boto3` в зависимости проекта для локального container-image запуска Lambda, а не только для managed AWS runtime.
