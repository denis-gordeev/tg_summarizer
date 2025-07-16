#!/bin/bash

# Скрипт для запуска суммаризатора Telegram новостей каждые 5 минут

# Переходим в директорию скрипта
cd "$(dirname "$0")"

# Настраиваем логирование
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/summarizer_$(date +%Y%m%d).log"

# Активируем conda environment (если используется)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate base

# Запускаем суммаризатор с логированием
echo "$(date): Starting Telegram news summarization..." >> "$LOG_FILE"
python3 summarizer.py >> "$LOG_FILE" 2>&1

# Проверяем результат
if [ $? -eq 0 ]; then
    echo "$(date): Summarization completed successfully" >> "$LOG_FILE"
else
    echo "$(date): Summarization failed with error code $?" >> "$LOG_FILE"
fi

# Очищаем старые логи (оставляем только за последние 7 дней)
find "$LOG_DIR" -name "summarizer_*.log" -mtime +7 -delete 
