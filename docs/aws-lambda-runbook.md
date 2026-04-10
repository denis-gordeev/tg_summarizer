# AWS Lambda Runbook

Этот runbook описывает эксплуатационный запуск `tg_summarizer` в AWS Lambda без постоянного локального диска.

## Что делает Lambda entrypoint

Файл [`lambda_handler.py`](../lambda_handler.py) выполняет один цикл суммаризации:

1. Переходит в `/tmp`, чтобы Telethon-сессии и JSON-файлы были доступны для записи.
2. Вызывает `download_from_s3()` из [`s3_sync.py`](../s3_sync.py), если задан `STATE_S3_BUCKET`.
3. Запускает `run_summarizer(...)`.
4. После завершения вызывает `upload_to_s3()`.

Без `STATE_S3_BUCKET` состояние существует только внутри текущего execution environment Lambda и может пропасть при cold start.

## Минимальная конфигурация Lambda

- Runtime: `Python 3.12`
- Handler: `lambda_handler.handler`
- Trigger: EventBridge Scheduler, EventBridge Rule или ручной invoke
- Timeout: от `3` минут, дальше подбирать по фактическому объёму каналов
- Memory: от `512 MB`, дальше подбирать по длительности и объёму истории

## Обязательные переменные окружения

```env
TELEGRAM_API_ID=...
TELEGRAM_API_HASH=...
TELEGRAM_BOT_TOKEN=...
TARGET_CHANNEL=@your_target_channel
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
OPENAI_DEFAULT_MAX_TOKENS=300
OPENAI_CHANNEL_SUMMARY_MAX_TOKENS=16000
OPENAI_GROUP_SUMMARY_MAX_TOKENS=16000
```

Обычно также нужны источники:

```env
SOURCE_CHANNELS=@channel_a,@channel_b
SOURCE_GROUPS=@group_a,@group_b
```

Если нужен state между cold start, задайте:

```env
STATE_S3_BUCKET=your-bucket-name
STATE_S3_PREFIX=tg_summarizer/prod
```

При необходимости можно явно ограничить список синхронизируемых файлов:

```env
STATE_SYNC_FILES=tg_summarizer_user.session,tg_summarizer_bot.session,summaries_history.json,summarization_history.json
```

## Какие файлы обычно нужно сохранять

- `tg_summarizer_user.session`
- `tg_summarizer_bot.session`
- `channel_abbreviations.json`
- `summarization_history.json`
- `summaries_history.json`
- `discovered_channels.json`
- `group_summarization_history.json`
- `group_summaries_history.json`
- `group_last_run.json`
- `prompts.json`, если используется внешний файл с кастомными промптами

## Event payload

Пример входного события:

```json
{
  "send_message": false,
  "save_changes": true,
  "include_today_processed_groups": false,
  "include_today_processed_messages": false
}
```

Практический смысл флагов:

- `send_message=true` публикует дайджест в `TARGET_CHANNEL`
- `save_changes=true` сохраняет локальное состояние и историю
- `include_today_processed_groups=true` форсирует повторную суммаризацию групп в тот же день
- `include_today_processed_messages=true` отключает защиту от повторной обработки уже просмотренных сообщений за сегодня

`lambda_handler.py` принимает флаги как JSON-boolean и как строковые значения `true` / `false`.

## Первый deploy

1. Создайте или выберите S3 bucket для state sync, если хотите переживать cold start.
2. Соберите deployment package с зависимостями под `Python 3.12`.
3. Создайте Lambda с handler `lambda_handler.handler`.
4. Передайте env-переменные из секции выше.
5. Настройте trigger, но для первого smoke run используйте `send_message=false`.
6. После первого invoke проверьте CloudWatch Logs и содержимое S3.

## Smoke run

Для ручной проверки удобно вызывать Lambda payload-ом без публикации:

```json
{
  "send_message": false,
  "save_changes": true,
  "include_today_processed_groups": false,
  "include_today_processed_messages": false
}
```

Что проверить после smoke run:

- Lambda стартует без ошибок по обязательным env-переменным.
- В логах нет ошибок аутентификации Telegram.
- В логах нет ошибок авторизации OpenAI.
- Если включён S3 sync, `.session` и JSON-файлы появились или обновились в bucket.

## Локальная проверка перед деплоем

```bash
python3 -m unittest discover -s tests
```

## Диагностика

Если Lambda не стартует:

- проверьте обязательные env-переменные из [`config.py`](../config.py)
- убедитесь, что deployment package содержит зависимости проекта
- проверьте, что в package присутствует [`s3_sync.py`](../s3_sync.py)

Если состояние не сохраняется:

- проверьте `STATE_S3_BUCKET` и `STATE_S3_PREFIX`
- проверьте права `s3:GetObject`, `s3:PutObject`, `s3:ListBucket`
- убедитесь, что нужные файлы входят в `STATE_SYNC_FILES`

Если результат слишком длинный:

- проверьте значения `OPENAI_CHANNEL_SUMMARY_MAX_TOKENS` и `OPENAI_GROUP_SUMMARY_MAX_TOKENS`
- убедитесь, что post-generation guardrails не были отключены локальными правками

## Инфраструктура

Воспроизводимый infra-шаблон уже добавлен в репозиторий:

- [`template.yaml`](../template.yaml) — AWS SAM шаблон с полной инфраструктурой (Lambda, IAM роли, S3 bucket для state, EventBridge триггер).
- [`samconfig.toml.example`](../samconfig.toml.example) — пример конфигурации для `sam deploy --guided`.
- [`docs/aws-lambda-deployment.md`](../docs/aws-lambda-deployment.md) — полное руководство по деплою (AWS CLI + SAM).

### Следующие улучшения

- Настроить GitHub Actions CI/CD для автоматического деплоя при мердже в main.
- Перенести чувствительные переменные в AWS SSM Parameter Store / Secrets Manager вместо env vars.
- Добавить алерты CloudWatch на ошибки и таймауты Lambda.
