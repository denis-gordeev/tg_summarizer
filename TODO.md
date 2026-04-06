# TODO

Живой список задач для автоматических раундов.

## Completed in 2026-04-02 round

- Исправлена обработка Lambda event-флагов: строка `"false"` больше не превращается в `True` в [`lambda_handler.py`](lambda_handler.py).
- `boto3` добавлен в зависимости проекта для container image и локального запуска Lambda с S3-синхронизацией.
- Уточнены [README.md](README.md) и [docs/aws-lambda-runbook.md](docs/aws-lambda-runbook.md) по запуску Lambda, S3 state sync и формату входных флагов.

## Completed in 2026-04-03 round

- Добавлены smoke/regression-тесты для [`lambda_handler.py`](lambda_handler.py) и [`s3_sync.py`](s3_sync.py) в [`tests/test_lambda_handler.py`](tests/test_lambda_handler.py) и [`tests/test_s3_sync.py`](tests/test_s3_sync.py).
- README и Lambda runbook дополнены явной командой локальной проверки перед деплоем: `python3 -m unittest discover -s tests`.
- Подтверждено прохождение нового набора проверок: `python3 -m unittest discover -s tests`.
- Параметризованы `OPENAI_MODEL`, `OPENAI_DEFAULT_MAX_TOKENS`, `OPENAI_CHANNEL_SUMMARY_MAX_TOKENS` и `OPENAI_GROUP_SUMMARY_MAX_TOKENS` через [`config.py`](config.py) и [`.env.example`](.env.example) с дефолтами не дороже `gpt-4o-mini`.
- Добавлен unit-тест [`tests/test_openai_config.py`](tests/test_openai_config.py) на чтение OpenAI-настроек и применение дефолтного лимита токенов в [`utils.py`](utils.py).

## Completed in 2026-04-04 round

- Добавлен AWS SAM шаблон [`template.yaml`](template.yaml) для воспроизводимого zip-deploy в AWS Lambda.
- README и [docs/aws-lambda-runbook.md](docs/aws-lambda-runbook.md) дополнены SAM workflow, параметрами шаблона и пояснением про `ReservedConcurrentExecutions=1`.
- Подтверждено прохождение релевантных проверок: `python3 -m unittest discover -s tests` и синтаксический разбор `template.yaml`.
- AWS SAM шаблон расширен встроенным EventBridge Scheduler (`ScheduleV2`) с параметрами `ScheduleExpression`, `ScheduleExpressionTimezone`, `ScheduleState` и `SchedulePayload`.
- README и [docs/aws-lambda-runbook.md](docs/aws-lambda-runbook.md) уточнены по безопасному первому деплою: scheduler создаётся сразу, но по умолчанию остаётся выключенным до ручного smoke run.
- Подтверждено прохождение проверок после добавления scheduler: `python3 -m unittest discover -s tests` и `ruby -e 'require "yaml"; YAML.load_file("template.yaml")'`.

## Completed in 2026-04-05 round

- AWS SAM шаблон дополнен базовым мониторингом эксплуатации: alarm на `AWS/Lambda Errors`, retry policy Scheduler и SQS DLQ для недоставленных scheduled invoke.
- README и [docs/aws-lambda-runbook.md](docs/aws-lambda-runbook.md) обновлены с параметрами мониторинга, различием между scheduler DLQ и ошибками самой Lambda, и шагами проверки после deploy.
- Подтверждено прохождение проверок после обновления шаблона и документации: `python3 -m unittest discover -s tests` и `ruby -e 'require "yaml"; YAML.load_file("template.yaml")'`.
- AWS SAM шаблон дополнен stack output `LogGroupName`, чтобы post-deploy smoke run и просмотр логов не требовали ручного поиска log group.
- README и [docs/aws-lambda-runbook.md](docs/aws-lambda-runbook.md) дополнены CLI-командами для `describe-stacks`, ручного `aws lambda invoke`, `aws logs tail` и явным использованием stack outputs после deploy.
- Подтверждено прохождение проверок после доработки runbook и outputs: `python3 -m unittest discover -s tests` и `ruby -e 'require "yaml"; YAML.load_file("template.yaml")'`.
- README и [docs/aws-lambda-runbook.md](docs/aws-lambda-runbook.md) дополнены отдельным сценарием ротации Telegram/OpenAI секретов без пересоздания существующего SAM/Lambda стека.

## Completed in 2026-04-01 round

- Добавлена документация по запуску и эксплуатации в AWS Lambda.
- Добавлен отдельный runbook `docs/aws-lambda-runbook.md` с deploy/checklist/troubleshooting для AWS Lambda.
- Добавлен модуль `s3_sync.py`, чтобы `lambda_handler.py` работал в чистом окружении и мог синхронизировать состояние через S3.
- Исправлены расхождения в документации: актуализирован `run_summarizer.sh`, поправлена опечатка в `.env.example`.

## Next actions

- Подключить новые smoke/regression-тесты к CI или хотя бы к локальному pre-push сценарию, чтобы проверки не оставались ручными.
- Добавить явные guardrails на длину итогового саммари после генерации, чтобы модель не выходила за целевой формат даже при завышенном `max_tokens`.
- Добавить отдельный alarm на сообщения в scheduler DLQ, если потребуется автоэскалация не только по `AWS/Lambda Errors`, но и по недоставленным invoke.
