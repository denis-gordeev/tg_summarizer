# TODO

Живой список задач для автоматических раундов.

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

## Next actions

- Настроить GitHub Actions CI/CD для автоматического деплоя Lambda при мердже в main.
- Перенести чувствительные переменные в AWS SSM Parameter Store / Secrets Manager вместо env vars.
- Добавить CloudWatch алерты на ошибки и таймауты Lambda.
