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
- Trigger: `EventBridge Scheduler` или `EventBridge Rule`
- Timeout: от `3` минут, дальше подбирать по реальному объёму каналов
- Memory: от `512 MB`, дальше подбирать по времени ответа и объёму истории
- Ephemeral storage: достаточно значения по умолчанию, если в `/tmp` лежат только `.session` и JSON-файлы

## Воспроизводимый деплой через AWS SAM

В репозитории есть минимальный инфраструктурный шаблон [`template.yaml`](../template.yaml) для zip-deploy через AWS SAM.

Что он делает:

- создаёт Lambda с handler `lambda_handler.handler`
- прокидывает обязательные env-переменные как CloudFormation parameters
- оставляет дефолтную модель `gpt-4o-mini`
- по умолчанию ставит `ReservedConcurrentExecutions=1`, чтобы не было параллельных инвокаций с гонками за `/tmp` и S3 state
- выдаёт CloudWatch Logs policy и, если задан `StateS3Bucket`, добавляет IAM-доступ к bucket

Базовый сценарий:

```bash
sam build
sam deploy --guided
```

На первом `sam deploy --guided` задайте:

- `TelegramApiId`, `TelegramApiHash`, `TelegramBotToken`
- `TargetChannel`
- `OpenAIApiKey`
- `SourceChannels` и/или `SourceGroups`
- `StateS3Bucket` и `StateS3Prefix`, если хотите переживать cold start

Если хотите сохранить конфигурацию SAM CLI локально, создайте некоммитящийся `samconfig.toml`. В `.gitignore` его пока нет намеренно, чтобы не навязывать конкретный workflow для всех окружений.

## Environment variables

Обязательные переменные:

```env
TELEGRAM_API_ID=...
TELEGRAM_API_HASH=...
TELEGRAM_BOT_TOKEN=...
TARGET_CHANNEL=@your_target_channel
OPENAI_API_KEY=...
```

Обычно также нужны источники:

```env
SOURCE_CHANNELS=@channel_a,@channel_b
SOURCE_GROUPS=@group_a,@group_b
```

Имена файлов состояния при необходимости можно переопределить:

```env
ABBREVIATIONS_FILE=channel_abbreviations.json
HISTORY_FILE=summarization_history.json
SUMMARIES_HISTORY_FILE=summaries_history.json
DISCOVERED_CHANNELS_FILE=discovered_channels.json
GROUP_HISTORY_FILE=group_summarization_history.json
GROUP_SUMMARIES_HISTORY_FILE=group_summaries_history.json
GROUP_LAST_RUN_FILE=group_last_run.json
PROMPTS_FILE=prompts.json
```

Переменные для синхронизации состояния через S3:

```env
STATE_S3_BUCKET=your-bucket-name
STATE_S3_PREFIX=tg_summarizer/prod
# Необязательно: явный список файлов для синка
# STATE_SYNC_FILES=tg_summarizer_user.session,tg_summarizer_bot.session,summarization_history.json,summaries_history.json,discovered_channels.json,group_summarization_history.json,group_summaries_history.json,group_last_run.json,prompts.json
```

## Какие файлы нужно сохранять

Для корректной работы между инвокациями обычно нужны:

- `tg_summarizer_user.session`
- `tg_summarizer_bot.session`
- `summarization_history.json`
- `summaries_history.json`
- `discovered_channels.json`
- `group_summarization_history.json`
- `group_summaries_history.json`
- `group_last_run.json`
- `prompts.json`, если вы используете файл с кастомными промптами

Если часть функциональности не используется, список можно сузить через `STATE_SYNC_FILES`.

## IAM-права

Lambda нужны права CloudWatch Logs:

- `logs:CreateLogGroup`
- `logs:CreateLogStream`
- `logs:PutLogEvents`

Если используется синхронизация состояния через S3, добавьте:

- `s3:GetObject`
- `s3:PutObject`
- `s3:ListBucket`

## Scheduler event

Пример входного события:

```json
{
  "send_message": true,
  "save_changes": true,
  "include_today_processed_groups": false,
  "include_today_processed_messages": false
}
```

Практический смысл флагов:

- `send_message=true` публикует дайджест в `TARGET_CHANNEL`
- `save_changes=true` сохраняет локальное состояние и историю
- `include_today_processed_groups=true` форсирует суммаризацию групп даже если сегодня они уже обрабатывались
- `include_today_processed_messages=true` игнорирует обычную защиту от повторной обработки сообщений за текущий день

`lambda_handler.py` принимает эти флаги как нормальные JSON-boolean и как строковые значения `true` / `false`, если scheduler или input transform передаёт их строками.

## Порядок первого деплоя

1. Создайте S3 bucket для состояния, если нужен стабильный state между cold start.
2. Выполните `sam build`.
3. Выполните `sam deploy --guided` и заполните параметры шаблона.
4. Выдайте Lambda доступ в интернет к Telegram API и OpenAI API.
5. Запустите функцию вручную с `send_message=false`, чтобы не публиковать тестовый дайджест.
6. Убедитесь, что в S3 появились `.session` и JSON-файлы состояния.
7. После этого включайте расписание с `send_message=true`.

## Операционные замечания

- `boto3` включён в зависимости проекта, поэтому S3-синхронизация должна одинаково работать в managed runtime, container image и локальной отладке.
- По умолчанию OpenAI-конфиг остаётся в бюджете `gpt-4o-mini`, но модель и лимиты генерации можно переопределить через env.
- Если S3 временно недоступен, функция не падает на каждом объекте синка: ошибки логируются, а обработка продолжается.
- Первый запуск может занять больше времени из-за инициализации Telethon-сессий.

## Быстрая проверка перед деплоем

Локально можно прогнать smoke/regression-тесты для Lambda entrypoint и S3 sync:

```bash
python3 -m unittest discover -s tests
```

Проверка шаблона, если установлен AWS SAM CLI:

```bash
sam validate
```

Полезные OpenAI-переменные окружения для Lambda:

- `OPENAI_MODEL` (`gpt-4o-mini` по умолчанию)
- `OPENAI_DEFAULT_MAX_TOKENS` (`300` по умолчанию для коротких служебных запросов)
- `OPENAI_CHANNEL_SUMMARY_MAX_TOKENS` (`16000` по умолчанию для канальных дайджестов)
- `OPENAI_GROUP_SUMMARY_MAX_TOKENS` (`16000` по умолчанию для групповых дайджестов)

## Диагностика

Если функция отработала, но результат не сохранился:

- проверьте, что задан `STATE_S3_BUCKET`
- проверьте права `s3:GetObject`, `s3:PutObject`, `s3:ListBucket`
- проверьте, что нужные файлы входят в `STATE_SYNC_FILES`

Если Lambda отработала без публикации:

- проверьте `TARGET_CHANNEL`
- проверьте `send_message` во входном событии
- проверьте, что бот имеет право писать в целевой канал

Если Lambda не может стартовать локально или в container image:

- проверьте, что зависимости из `requirements.txt` или `pyproject.toml` действительно установлены, включая `boto3`
- проверьте все обязательные переменные окружения из `config.py`
- если деплой через AWS SAM, проверьте финальные значения parameters и итоговый env в Lambda configuration
