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

## Completed in 2026-04-01 round

- Добавлена документация по запуску и эксплуатации в AWS Lambda.
- Добавлен отдельный runbook `docs/aws-lambda-runbook.md` с deploy/checklist/troubleshooting для AWS Lambda.
- Добавлен модуль `s3_sync.py`, чтобы `lambda_handler.py` работал в чистом окружении и мог синхронизировать состояние через S3.
- Исправлены расхождения в документации: актуализирован `run_summarizer.sh`, поправлена опечатка в `.env.example`.

## Next actions

- Добавить инфраструктурный шаблон деплоя AWS Lambda (SAM, Serverless Framework или Terraform), чтобы запуск был воспроизводимым.
- Подключить новые smoke/regression-тесты к CI или хотя бы к локальному pre-push сценарию, чтобы проверки не оставались ручными.
- Параметризовать модель OpenAI и лимиты генерации через env/config, сохранив дефолт не дороже `gpt-4o-mini`.
