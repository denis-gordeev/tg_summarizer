# TODO

Живой список задач для автоматических раундов.

## Completed in 2026-04-01 round

- Добавлена документация по запуску и эксплуатации в AWS Lambda.
- Добавлен отдельный runbook `docs/aws-lambda-runbook.md` с deploy/checklist/troubleshooting для AWS Lambda.
- Добавлен модуль `s3_sync.py`, чтобы `lambda_handler.py` работал в чистом окружении и мог синхронизировать состояние через S3.
- Исправлены расхождения в документации: актуализирован `run_summarizer.sh`, поправлена опечатка в `.env.example`.

## Next actions

- Проверить и при необходимости добавить `boto3` в зависимости для non-AWS запусков Lambda через container image или локальную отладку.
- Добавить инфраструктурный шаблон деплоя AWS Lambda (SAM, Serverless Framework или Terraform), чтобы запуск был воспроизводимым.
- Параметризовать модель OpenAI и лимиты генерации через env/config, сохранив дефолт не дороже `gpt-4o-mini`.
