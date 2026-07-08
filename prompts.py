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

    Prompts can be accessed as attributes, e.g., `prompts.CHANNEL_SUMMARY_PROMPT`.
    """

    def __init__(self):
        self._defaults = {
            "COVERAGE_AND_MATCH_PROMPT": """Сравни тему сообщения с дайджестами. Та же тема → номер, новая/детали → "НЕТ", разные модели = разные темы. Ответь: номер или "НЕТ".""",
            "NLP_RELEVANCE_PROMPT": """Релевантен ли текст для NLP/ML/AI/Python/лингвистического дайджеста?
ПРИНИМАЙ: статьи, модели, библиотеки, бенчмарки, датасеты, конференции, Kaggle/хакатоны с AI, вакансии/увольнения в AI, BigTech/AI-ассистенты, LLM, промпт-инжиниринг, LoRA, диффузия, GPU, робототехника, AI-стартапы/M&A, AI-мемы, ШАД (не реклама).
ОТКЛОНЯЙ: курсы, платное обучение, реклама GigaChat/YandexGPT, коммерция, hiring days, ссылки на ботов без контекста.
Ответь только 'да' или 'нет'.""",
            "UPDATE_SUMMARY_PROMPT": """Вставь {new_link} в подходящее место саммари рядом с релевантным абзацем. Остальное не меняй.
Сохрани HTML-форматирование. Нет подходящего места — добавь "Доп. источники: {new_link}" в конец.
Только обновлённое саммари. Без вводных слов и заключений.""",
            "CHANNEL_SUMMARY_PROMPT": """Дайджест Telegram-сообщений по NLP. Русский язык, термины — на языке оригинала.
Группируй по темам с [1], [2]; совпадающие темы — все номера. Близкие подтемы — под один заголовок.
- Без введения, заключения, мета-комментариев и вводных слов
- Один факт на предложение, одна мысль на абзац. Без повторов, воды и пояснений
- Конкретные факты и результаты, нейтральный стиль без мнений и дословных рекомендаций авторов
- Сплошной текст с абзацами, один заголовок на тему. Без списков и подзаголовков
- Заголовки: короткие, 1–3 эмодзи, HTML <b>текст</b>, не Markdown. Пример: '<b>🧠💧 Переходы</b>'
- Источники: [1], [2], без своих ссылок
- ≤{max_summary_length} символов
""",
            "GROUP_SUMMARY_PROMPT": """Дайджест сообщений из Telegram-группы по NLP. Русский язык, термины — на языке оригинала.
Группируй по темам с [1], [2]; совпадающие темы — все номера. Близкие подтемы — под один заголовок.
- Без введения, заключения, мета-комментариев и вводных слов
- Один факт на предложение, одна мысль на абзац. Без повторов, воды и пояснений
- Конкретные факты и результаты, нейтральный стиль без мнений и дословных рекомендаций участников
- Сплошной текст с абзацами, один заголовок на тему. Без списков и подзаголовков
- Не добавляй вопросы пользователей, на которые нет ответа
- Заголовки: короткие, 1–3 эмодзи, HTML <b>текст</b>, не Markdown. Пример: '<b>🧠💧 Переходы</b>'
- Источники: [1], [2], без своих ссылок
- ≤{max_summary_length} символов
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
