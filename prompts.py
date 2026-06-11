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
            "COVERAGE_AND_MATCH_PROMPT": """
Сравни ТЕМУ нового сообщения с дайджестами.
- Та же тема → номер дайджеста
- Новая тема или существенные детали → "НЕТ"
- Разные модели/версии = разные темы
Ответь только номером или "НЕТ".
""",
            "NLP_RELEVANCE_PROMPT": """
Релевантен ли текст для NLP/ML/AI/Python/лингвистического дайджеста?

ПРИНИМАЙ ('да'):
- Статьи, исследования, модели, библиотеки, инструменты, бенчмарки, датасеты
- Конференции (NeurIPS, ICML, ICLR, ACL, EMNLP и т.д.), открытые проекты
- Kaggle, хакатоны, соревнования (в т.ч. робототехника с ИИ/LLM)
- Вакансии, пет-проекты, ИИ-экономика, увольнения в AI
- BigTech/AI-ассистенты (OpenAI, Anthropic, Google, Meta, DeepSeek и др.), vibe coding, AI-агенты
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
- Не повторяй одну и ту же информацию в разных разделах
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
- Не повторяй одну и ту же информацию в разных разделах
- Заголовки: короткие, с 1–3 несинонимичных эмодзи. Пример: '<b>🧠💧 Переходы между AI-компаниями</b>'
- Только HTML: <b>текст</b>, не Markdown
- Только номера источников [1], [2], без собственных ссылок
- Максимум {max_summary_length} символов
- Не добавляй вопросы пользователей, на которые нет ответа
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
