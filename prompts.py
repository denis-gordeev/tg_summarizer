import json
import os
import logging

from config import PROMPTS_FILE

logger = logging.getLogger(__name__)


class PromptManager:
    """
    Manages loading and accessing prompts for the summarizer.

    It first loads default prompts defined in the class, and then overrides them
    with any custom prompts defined in an external `prompts.json` file.

    Prompts can be accessed as attributes, e.g., `prompts.DUPLICATE_CHECK_PROMPT`.
    """

    def __init__(self):
        self._defaults = {
            "DUPLICATE_CHECK_PROMPT": """
Описывают ли два сообщения Telegram одинаковый контент или статью?
Ответь одним словом: да или нет.
""",
            "SUMMARY_COVERAGE_CHECK_PROMPT": """
Сравни ТЕМУ и ОСНОВНУЮ ИДЕЮ нового сообщения с предыдущими дайджестами.
- Та же тема → "ДА"
- Новая тема → "НЕТ"
- Разные модели на Hugging Face = разные темы → "НЕТ"
- Существенные новые детали → "НЕТ"
- Игнорируй мелкие детали и формулировки

Примеры:
- "GPT-5" и "OpenAI анонсировал GPT-5" = ДА
- "GPT-5" и "GPT-4.5" = НЕТ
- "Акции NVIDIA выросли" и "NVIDIA показала хорошие результаты" = ДА

Ответь только "ДА" или "НЕТ".
""",
            "GROUP_SUMMARY_COVERAGE_CHECK_PROMPT": """
Сравни ТЕМУ и ОСНОВНУЮ ИДЕЮ нового сообщения с предыдущими дайджестами групп.
- Та же тема → "ДА"
- Новая тема → "НЕТ"
- Игнорируй мелкие детали и формулировки

Примеры:
- "GPT-5" и "OpenAI анонсировал GPT-5" = ДА
- "GPT-5" и "BERT" = НЕТ
- "Акции NVIDIA выросли" и "NVIDIA показала хорошие результаты" = ДА

Ответь только "ДА" или "НЕТ".
""",
            "COVERAGE_AND_MATCH_PROMPT": """
Сравни ТЕМУ нового сообщения с предыдущими дайджестами.
- Та же тема → номер дайджеста
- Новая тема или существенные новые детали → "НЕТ"
- Разные модели на Hugging Face = разные темы → "НЕТ"
- Игнорируй мелкие детали и формулировки
Ответь только номером или "НЕТ".
""",
            "NLP_RELEVANCE_PROMPT": """
Релевантен ли текст для NLP/ML/AI/Python/лингвистического дайджеста?

ПРИНИМАЙ ('да'):
- Статьи, исследования, модели, библиотеки, инструменты, бенчмарки, датасеты
- Конференции (NeurIPS, ICML, ICLR, ACL, EMNLP и т.д.), открытые проекты
- Kaggle, хакатоны, соревнования (в т.ч. робототехника с ИИ/LLM)
- Вакансии, пет-проекты, ИИ-экономика, увольнения в AI
- BigTech и AI-ассистенты (OpenAI, Anthropic, Google, Meta, Microsoft, Nvidia, DeepSeek, ChatGPT, Claude, Gemini, Grok, Midjourney и др.), vibe coding, AI-агенты
- LLM, промпт-инжиниринг, дообучение (LoRA и др.), диффузия, GPU, робототехника, стартапы, M&A, AI-мемы
- Обсуждения AI/моделей, рынка вакансий, ШАД (не реклама и не курсы)

ОТКЛОНЯЙ ('нет'):
- Курсы, обучение, платные программы, мастер-классы с сертификатами
- Реклама русскоязычных LLM (GigaChat, YandexGPT)
- Коммерческие предложения, услуги, вебинары с продажами, hiring days
- Ссылки на ботов без указания, что это бот

Ответь 'да' или 'нет'. Если 'нет' — напиши "нет, причина: " и объясни.
""",
            "CHANNEL_SUMMARY_PROMPT": """
Создай краткий дайджест сообщений Telegram по NLP. Язык — русский, термины на языке оригинала.
Структурируй по темам, указывай номера источников [1], [2]; при совпадении темы — все номера.

Правила:
- Без введения и заключения — сразу с содержания
- Нейтральный стиль: без эмоций, преувеличений и дословных рекомендаций авторов
- Заголовки: короткие, с 1–3 несинонимичных эмодзи. Пример: '<b>🧠💧 Переходы между AI-компаниями</b>'
- Только HTML: <b>текст</b>, не Markdown
- Только номера источников [1], [2], без собственных ссылок
- Максимум {max_summary_length} символов
""",
            "GROUP_SUMMARY_PROMPT": """
Создай краткий дайджест сообщений из Telegram-группы по NLP. Язык — русский, термины на языке оригинала.
Структурируй по темам, указывай номера источников [1], [2]; при совпадении темы — все номера.

Правила:
- Без введения и заключения — сразу с содержания
- Нейтральный стиль: без эмоций, преувеличений и дословных рекомендаций участников
- Заголовки: короткие, с 1–3 несинонимичных эмодзи. Пример: '<b>🧠💧 Переходы между AI-компаниями</b>'
- Только HTML: <b>текст</b>, не Markdown
- Только номера источников [1], [2], без собственных ссылок
- Максимум {max_summary_length} символов
- Не добавляй вопросы пользователей, на которые нет ответа
""",
            "FIND_RELEVANT_SUMMARY_PROMPT": """
В какое существующее саммари лучше добавить ссылку на новое сообщение?
""",
        }

        self._prompts = self._defaults.copy()

        prompts_file = PROMPTS_FILE
        file_path = os.path.join(os.path.dirname(__file__), prompts_file)
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    custom_prompts = json.load(f)
                    self._prompts.update(custom_prompts)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Could not load or parse %s: %s", prompts_file, e)

    def __getattr__(self, name):
        """Allows accessing prompts as attributes."""
        if name in self._prompts:
            return self._prompts[name]
        raise AttributeError(f"'PromptManager' object has no attribute '{name}'")


# Singleton instance to be used across the application
prompts = PromptManager()
