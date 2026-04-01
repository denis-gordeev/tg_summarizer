# Telegram Summarizer

Автоматический сборщик и суммаризатор сообщений из Telegram-каналов с использованием OpenAI GPT.

## Возможности

- Автоматический сбор сообщений из указанных каналов
- Фильтрация релевантных сообщений с помощью AI
- Создание кратких дайджестов с использованием GPT
- Отправка саммари в целевой канал
- Дедупликация сообщений и саммари
- Поддержка групповых саммари
- **Восстановление истории саммари из канала при ошибках**

## Восстановление истории саммари

Приложение автоматически восстанавливает историю саммари из целевого канала в случае ошибки загрузки файлов истории. Это происходит в следующих случаях:

- Файл истории поврежден или отсутствует
- Ошибка при чтении JSON-файла
- Проблемы с кодировкой файла

### Как работает восстановление

1. **Автоматическое обнаружение**: При ошибке загрузки истории система автоматически пытается восстановить данные
2. **Чтение из канала**: Система читает сообщения из целевого канала за последнюю неделю
3. **Распознавание саммари**: Все сообщения в целевом канале считаются саммари
4. **Извлечение каналов**: Система извлекает названия исходных каналов из ссылок вида `https://t.me/channel_name/message_id`
5. **Создание объектов**: Восстанавливает объекты `SummaryInfo` из найденных сообщений с информацией о каналах
6. **Сохранение**: Автоматически сохраняет восстановленную историю в файл

### Типы саммари

Система различает два типа саммари:

- **Обычные саммари**: Все сообщения в целевом канале считаются обычными саммари
- **Групповые саммари**: Все сообщения в целевом канале считаются групповыми саммари (для совместимости с существующей системой)

### Извлечение каналов

При восстановлении истории система автоматически извлекает названия исходных каналов из двух источников:

1. **Ссылки в сообщениях**: `https://t.me/channel_name/message_id`
2. **Аббревиатуры в квадратных скобках**: `[AN]`, `[MLD]`, `[NR]` и т.д.

Система:
- Извлекает каналы из ссылок с помощью регулярного выражения
- Восстанавливает полные названия каналов из аббревиатур через файл `channel_abbreviations.json`
- Объединяет результаты в единый список без дубликатов
- Сохраняет каналы в поле `channels` объекта `SummaryInfo`

**Примеры извлечения:**
- Из ссылки `https://t.me/neuraldeep/1554` → `neuraldeep`
- Из аббревиатуры `[AN]` → `@ai_news`
- Из аббревиатуры `[MLD]` → `@machine_learning_daily`

### Тестирование восстановления

Для тестирования функции восстановления используйте:

```bash
# Базовый тест восстановления
python test_restore.py

# Тест извлечения каналов из ссылок
python test_extract_channels.py

# Полный тест с демонстрацией функциональности
python test_full_restore.py
```

Эти скрипты проверит:
- Восстановление обычных саммари
- Восстановление групповых саммари
- Извлечение каналов из ссылок
- Загрузку восстановленной истории из файла

### Примеры использования

**Автоматическое восстановление при ошибке:**
```python
from history_manager import load_summaries_history

# При ошибке загрузки файла система автоматически восстановит историю из канала
summaries = load_summaries_history()
print(f"Загружено {len(summaries)} саммари")
```

**Ручное восстановление:**
```python
from history_manager import restore_summaries_from_channel_sync

# Принудительное восстановление истории
summaries = restore_summaries_from_channel_sync()
print(f"Восстановлено {len(summaries)} саммари")
```

**Извлечение каналов из ссылок:**
```python
from utils import extract_telegram_channels

message = "Новости AI: https://t.me/neuraldeep/1554 и https://t.me/ai_news/123"
channels = extract_telegram_channels(message)
print(f"Каналы: {channels}")  # ['neuraldeep', 'ai_news']
```

## Установка

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd tg_summarizer
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```
или
```bash
poetry install
```

3. Создайте файл `.env` с необходимыми переменными окружения:
```bash
cp .env.example .env
```

## Настройка

### 1. Получение Telegram API credentials

1. Перейдите на https://my.telegram.org/
2. Войдите в свой аккаунт
3. Перейдите в "API development tools"
4. Создайте новое приложение и получите `API_ID` и `API_HASH`

### 2. Создание Telegram бота

1. Найдите @BotFather в Telegram
2. Отправьте команду `/newbot`
3. Следуйте инструкциям и получите `BOT_TOKEN`

### 3. Настройка каналов

1. **Добавьте бота в целевой канал как администратора** с правами на отправку сообщений
2. Добавьте бота в исходные каналы (или убедитесь, что каналы публичные)
3. Получите username каналов (например, `@channel_name`)

**Важно:** Бот должен быть администратором целевого канала с правами на отправку сообщений. Без этих прав бот не сможет постить саммари в канал.

### 4. Получение OpenAI API ключа

1. Зарегистрируйтесь на https://platform.openai.com/
2. Создайте API ключ в разделе "API Keys"

### 5. Настройка .env файла

Отредактируйте файл `.env`:

```env
# Telegram API credentials
TELEGRAM_API_ID=12345678
TELEGRAM_API_HASH=your_api_hash_here
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz

# Channel configuration
TARGET_CHANNEL=@your_target_channel
SOURCE_CHANNELS=@nlp_channel,@ml_news,@ai_research

# Group configuration (опционально)
SOURCE_GROUPS=@your_group1,@your_group2
TARGET_GROUP_CHANNEL=@your_group_target_channel  # По умолчанию тот же TARGET_CHANNEL

# OpenAI API key
OPENAI_API_KEY=sk-your_openai_api_key_here
```

#### Настройка групп

Для суммаризации сообщений из Telegram-групп добавьте следующие переменные:

- `SOURCE_GROUPS` - список групп для мониторинга (через запятую)
- `TARGET_GROUP_CHANNEL` - канал для отправки саммари групп (опционально, по умолчанию используется `TARGET_CHANNEL`)

**Особенности суммаризации групп:**
- Суммаризация групп происходит только **раз в сутки** (при первом запуске за день)
- Используется отдельная система истории и дедупликации
- Группы обрабатываются независимо от каналов
- Саммари групп содержит заголовок "👥 Обзор сообщества [название_группы]"

## Использование

### Тестирование подключения

Перед запуском основного скрипта рекомендуется протестировать подключение:

```bash
python test_connection.py
```

Этот скрипт проверит:
- Наличие всех переменных окружения
- Подключение к Telegram API
- Доступ к целевым и исходным каналам

### Запуск основного скрипта

```bash
python summarizer.py
```

Скрипт выполнит следующие действия:

**Обработка каналов:**
1. Соберет сообщения из исходных каналов за последние 24 часа
2. Отфильтрует сообщения по тематике NLP/ML
3. Удалит дубликаты
4. Сгенерирует краткий дайджест с HTML ссылками на оригинальные сообщения
5. Отправит дайджест в целевой канал

**Обработка групп (если настроены):**
1. Проверит, нужно ли запускать суммаризацию групп (раз в сутки)
2. Если да - соберет сообщения из групп за последние 24 часа
3. Отфильтрует сообщения по тематике NLP/ML
4. Удалит дубликаты с учетом истории групп
5. Сгенерирует отдельный дайджест для групп
6. Отправит дайджест в целевой канал групп
7. Обновит время последнего запуска

## Установка и запуск через Docker

### Сборка образа

```bash
# В корне проекта
docker build -t tg-summarizer:latest .
```

### Быстрый запуск (однократный)

```bash
docker run --rm -it \
  --env-file .env \
  -e TZ=Europe/Moscow \
  tg-summarizer:latest
```

### Рекомендованный запуск (c сохранением истории и сессий)

Важно: Telethon-сессии (`tg_summarizer_user.session`, `tg_summarizer_bot.session`) лучше создать заранее локально (первым запуском скрипта вне Docker) и смонтировать в контейнер. Также вынесем файлы истории в volume `/data`.

```bash
docker run -d --name tg-summarizer \
  --env-file .env \
  -e TZ=Europe/Moscow \
  -e HISTORY_FILE=/data/summarization_history.json \
  -e SUMMARIES_HISTORY_FILE=/data/summaries_history.json \
  -e DISCOVERED_CHANNELS_FILE=/data/discovered_channels.json \
  -e GROUP_HISTORY_FILE=/data/group_summarization_history.json \
  -e GROUP_SUMMARIES_HISTORY_FILE=/data/group_summaries_history.json \
  -e GROUP_LAST_RUN_FILE=/data/group_last_run.json \
  -v tg_summarizer_data:/data \
  -v $(pwd)/tg_summarizer_user.session:/app/tg_summarizer_user.session \
  -v $(pwd)/tg_summarizer_bot.session:/app/tg_summarizer_bot.session \
  tg-summarizer:latest
```

Опционально можно сохранить логи на хосте:

```bash
docker run -d --name tg-summarizer \
  --env-file .env \
  -v $(pwd)/logs:/app/logs \
  tg-summarizer:latest
```

Минимальный набор переменных в `.env`:

```env
TELEGRAM_API_ID=...
TELEGRAM_API_HASH=...
TELEGRAM_BOT_TOKEN=...
TARGET_CHANNEL=@your_target_channel
OPENAI_API_KEY=...
# Опционально
SOURCE_CHANNELS=@a,@b
SOURCE_GROUPS=@g1,@g2
```

## Автоматизация

### Использование готового скрипта

В проекте есть готовый скрипт `run_summarizer.sh`:

```bash
# Запуск вручную
./run_summarizer.sh

# Добавить в crontab для запуска каждый день в 9:00
0 9 * * * /path/to/tg_summarizer/run_summarizer.sh >> /path/to/tg_summarizer/logs/summarizer.log 2>&1
```

### Ручная настройка cron

```bash
# Открыть crontab для редактирования
crontab -e

# Добавить строку для запуска каждый день в 9:00
0 9 * * * cd /path/to/tg_summarizer && python summarizer.py

# Или для запуска каждый час
0 * * * * cd /path/to/tg_summarizer && python summarizer.py
```

## Запуск в AWS Lambda

В репозитории есть entrypoint [`lambda_handler.py`](lambda_handler.py), рассчитанный на запуск по расписанию через EventBridge. Перед выполнением он:

1. Переходит в `/tmp`, чтобы Telethon-сессии и JSON-файлы истории были доступны для записи.
2. Загружает состояние из S3 через `s3_sync.py`, если настроен `STATE_S3_BUCKET`.
3. Вызывает `run_summarizer(...)`.
4. Загружает обновлённое состояние обратно в S3.

### Что нужно положить в Lambda environment

Обязательные переменные такие же, как для обычного запуска:

```env
TELEGRAM_API_ID=...
TELEGRAM_API_HASH=...
TELEGRAM_BOT_TOKEN=...
TARGET_CHANNEL=@your_target_channel
OPENAI_API_KEY=...
```

Опциональные переменные для источников и файлов состояния:

```env
SOURCE_CHANNELS=@a,@b
SOURCE_GROUPS=@g1,@g2
ABBREVIATIONS_FILE=channel_abbreviations.json
HISTORY_FILE=summarization_history.json
SUMMARIES_HISTORY_FILE=summaries_history.json
DISCOVERED_CHANNELS_FILE=discovered_channels.json
GROUP_HISTORY_FILE=group_summarization_history.json
GROUP_SUMMARIES_HISTORY_FILE=group_summaries_history.json
GROUP_LAST_RUN_FILE=group_last_run.json
PROMPTS_FILE=prompts.json
```

Для сохранения состояния между инвокациями включите S3-синхронизацию:

```env
STATE_S3_BUCKET=your-bucket-name
STATE_S3_PREFIX=tg_summarizer/prod
# Необязательно: явный список файлов для синка
# STATE_SYNC_FILES=tg_summarizer_user.session,tg_summarizer_bot.session,summarization_history.json,summaries_history.json,discovered_channels.json,group_summarization_history.json,group_summaries_history.json,group_last_run.json,prompts.json
```

Если `STATE_S3_BUCKET` не задан, синхронизация с S3 отключается, и состояние живёт только внутри одного execution environment Lambda.

### Какие файлы надо сохранять

Для корректной работы между инвокациями нужны как минимум:

- `tg_summarizer_user.session`
- `tg_summarizer_bot.session`
- `summarization_history.json`
- `summaries_history.json`
- `discovered_channels.json`
- `group_summarization_history.json`
- `group_summaries_history.json`
- `group_last_run.json`
- `prompts.json`, если вы переопределяете промпты через файл

### IAM-права для Lambda

Функции нужны:

- `logs:CreateLogGroup`
- `logs:CreateLogStream`
- `logs:PutLogEvents`
- `s3:GetObject`
- `s3:PutObject`
- `s3:ListBucket`

Если S3-синхронизация не используется, права на S3 можно убрать.

### Handler и расписание

- Handler: `lambda_handler.handler`
- Runtime: Python 3.12
- Рекомендуемый trigger: EventBridge Scheduler или EventBridge Rule

Пример события для ручного запуска или scheduler input:

```json
{
  "send_message": true,
  "save_changes": true,
  "include_today_processed_groups": false,
  "include_today_processed_messages": false
}
```

### Практические замечания

- У Lambda должен быть выход в интернет к Telegram API и OpenAI API.
- Все рабочие файлы создаются в `/tmp`, поэтому без S3 их состояние не гарантируется между холодными стартами.
- Первый запуск удобнее сделать вручную и убедиться, что Telethon создал обе `.session`-сессии.
- Если вы деплоите через container image или запускаете Lambda entrypoint локально, проверьте наличие `boto3` в окружении. В managed AWS Lambda Python runtime он обычно доступен, но в других окружениях это может быть не так.

## Работа с историей суммаризаций

Система автоматически отслеживает уже обработанные сообщения, чтобы избежать дублирования информации в дайджестах.

### Дедупликация с учетом предыдущих саммари

Новая функциональность позволяет избежать повторения уже освещенных тем:

1. **Анализ предыдущих саммари**: Система анализирует последние 7 дней саммари
2. **LLM-проверка**: Использует GPT для определения, была ли тема уже освещена
3. **Умная фильтрация**: Исключает сообщения, темы которых уже были в предыдущих дайджестах

Это помогает:
- Избежать повторения одних и тех же новостей
- Сосредоточиться на действительно новой информации
- Улучшить качество дайджестов

### Просмотр истории

Для просмотра истории обработанных сообщений используйте:

```bash
python view_summarization_history.py
```

Это покажет:
- Общую статистику по обработанным сообщениям
- Распределение сообщений по каналам
- Последние 10 обработанных сообщений с деталями

### Просмотр истории саммари

Для просмотра истории созданных саммари используйте:

```bash
python view_summaries_history.py
```

Это покажет:
- Общую статистику по созданным саммари
- Последние 5 саммари с деталями
- Распределение саммари по каналам

### Просмотр истории групп

Для просмотра истории суммаризации групп используйте:

```bash
python view_group_summarization_history.py
```

Это покажет:
- Статистику по обработанным сообщениям из групп
- Статистику по созданным саммари групп
- Информацию о последнем запуске суммаризации групп
- Статус готовности к следующему запуску

#### Команды для работы с историей групп

```bash
# Показать статистику (по умолчанию)
python view_group_summarization_history.py

# Очистить историю обработанных сообщений
python view_group_summarization_history.py --clear-history

# Очистить историю созданных саммари
python view_group_summarization_history.py --clear-summaries

# Очистить информацию о последнем запуске
python view_group_summarization_history.py --clear-last-run

# Очистить всю историю групп
python view_group_summarization_history.py --clear-all

# Показать справку
python view_group_summarization_history.py --help
```

### Работа с обнаруженными каналами

Система автоматически обнаруживает новые каналы, которые репостят релевантный контент и прошли проверку `is_nlp_related`. Эти каналы сохраняются отдельно от основных источников и автоматически добавляются в список для мониторинга.

#### Просмотр обнаруженных каналов

```bash
python view_discovered_channels.py
```

Это покажет:
- Список всех обнаруженных каналов
- Количество обнаруженных каналов
- Дату последнего обновления

#### Очистка обнаруженных каналов

```bash
python view_discovered_channels.py --clear
```

Это очистит список обнаруженных каналов (полезно при изменении стратегии обнаружения).

#### Как это работает

1. **Автоматическое обнаружение**: При обработке сообщений система проверяет, из какого канала пришло сообщение
2. **Фильтрация**: Если канал не входит в основной список `SOURCE_CHANNELS`, но сообщение прошло проверку `is_nlp_related`
3. **Сохранение**: Канал автоматически добавляется в `discovered_channels.json`
4. **Мониторинг**: В следующих запусках система будет мониторить как основные, так и обнаруженные каналы

#### Файл обнаруженных каналов

Обнаруженные каналы сохраняются в файл `discovered_channels.json`:

```json
{
  "discovered_channels": [
    "@ai_news_channel",
    "@ml_research_updates"
  ],
  "last_updated": "2024-01-15T10:30:00"
}
```

### Очистка истории

Для полной очистки истории сообщений (например, при изменении источников):

```bash
python view_summarization_history.py --clear
```

Для очистки истории саммари:

```bash
python view_summaries_history.py --clear
```

### Файл истории

История сохраняется в файл `summarization_history.json` в корне проекта. Файл содержит:
- Список всех обработанных сообщений
- Дату последнего обновления
- Автоматически ограничивается последними 1000 сообщениями

## Структура проекта

```
tg_summarizer/
├── summarizer.py                           # Основной скрипт
├── lambda_handler.py                       # Entry point для AWS Lambda
├── s3_sync.py                              # Синхронизация сессионных и history-файлов через S3
├── view_summarization_history.py           # Скрипт для просмотра истории
├── view_summaries_history.py               # Скрипт для просмотра истории саммари
├── view_discovered_channels.py             # Скрипт для просмотра обнаруженных каналов
├── view_group_summarization_history.py     # Скрипт для просмотра истории групп
├── run_summarizer.sh                       # Локальный запуск по cron
├── requirements.txt                        # Зависимости Python
├── README.md                              # Документация
├── TODO.md                                # Живой список задач для следующих раундов
├── .env                                   # Переменные окружения (создать самостоятельно)
├── summarization_history.json             # История обработанных сообщений (создается автоматически)
├── summaries_history.json                 # История созданных саммари (создается автоматически)
├── discovered_channels.json               # Обнаруженные каналы (создается автоматически)
├── group_summarization_history.json       # История обработанных сообщений из групп (создается автоматически)
├── group_summaries_history.json           # История созданных саммари групп (создается автоматически)
├── group_last_run.json                    # Информация о последнем запуске групп (создается автоматически)
├── tg_summarizer_user.session             # Telethon user session (создается автоматически)
├── tg_summarizer_bot.session              # Telethon bot session (создается автоматически)
└── logs/                                  # Папка для логов (создается автоматически)
```

## Требования

- Python 3.8+
- Telegram API credentials
- OpenAI API ключ
- Доступ к исходным каналам
- Права администратора в целевом канале

## Права бота в Telegram

Для корректной работы системы бот должен иметь следующие права в целевом канале:

### Обязательные права:
- ✅ **Отправка сообщений** - бот должен иметь возможность постить в канал
- ✅ **Администратор** - бот должен быть добавлен как администратор канала

### Как добавить бота как администратора:

1. Откройте целевой канал в Telegram
2. Перейдите в настройки канала (нажмите на название канала)
3. Выберите "Администраторы" → "Добавить администратора"
4. Найдите вашего бота по username
5. Включите права "Отправка сообщений"
6. Сохраните изменения

### Проверка прав бота:

Если бот не может отправлять сообщения, проверьте:
- Бот добавлен как администратор канала
- У бота включены права "Отправка сообщений"
- Канал указан правильно в переменной `TARGET_CHANNEL`
- Бот не заблокирован в канале

## Устранение неполадок

### Ошибка "Missing required environment variables"
Убедитесь, что файл `.env` создан и содержит все необходимые переменные.

### Ошибка "Cannot find implementation or library stub for module named 'telethon'"
Установите зависимости: `pip install -r requirements.txt`

### Ошибки OpenAI API
Проверьте:
- Правильность API ключа
- Достаточность баланса на аккаунте OpenAI
- Статус API сервиса OpenAI

### Ошибки отправки сообщений ботом
Если бот не может отправлять сообщения в канал:
1. Убедитесь, что бот добавлен как администратор канала
2. Проверьте, что у бота включены права "Отправка сообщений"
3. Убедитесь, что канал указан правильно в переменной `TARGET_CHANNEL`
4. Попробуйте отправить тестовое сообщение вручную через бота
5. Проверьте, что бот не заблокирован в канале
