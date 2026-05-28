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
Ответь да или нет.
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
            "NLP_RELEVANCE_PROMPT": """
Определи, релевантен ли текст для NLP/ML/AI/Python/лингвистического дайджеста.

ПРИНИМАЙ ('да'):
- Научные статьи, исследования, новые модели, библиотеки, инструменты, бенчмарки
- Конференции (NeurIPS, ICML, ICLR, ACL, EMNLP и т.д.), открытые проекты, датасеты
- Kaggle, хакатоны, соревнования (в т.ч. робототехника с ИИ/LLM)
- Карьера, вакансии, пет-проекты, ИИ-экономика, увольнения
- AI/BigTech: OpenAI, Anthropic, Google, Meta, Microsoft, Nvidia, DeepMind, FAANG и т.д.
- Sam Altman, Mark Zuckerberg — любые новости
- AI-ассистенты и приложения (ChatGPT, Claude, Gemini, Grok, DeepSeek, Midjourney и т.д.)
- LLM, промпт-инжиниринг, обучение/дообучение моделей (LoRA и др.)
- Vibe coding, AI-агенты, диффузия, GPU, робототехника
- Стартапы, покупки/продажи компаний, поглощения
- Мемы из мира AI/технологий
- Новые детали про уже вышедшие модели/библиотеки
- Обсуждения AI/моделей, рынка вакансий, ШАД (не реклама и не подготовительные курсы)
- Незнакомые термины в контексте статей/релизов/компаний — скорее всего релевантно

ОТКЛОНЯЙ ('нет'):
- Курсы, обучение, платные программы, мастер-классы с сертификатами
- Реклама русскоязычных LLM (GigaChat, YandexGPT)
- Коммерческие предложения, услуги, вебинары с продажами, hiring days
- Ссылки на ботов без явного указания, что это бот

Ответь 'да' или 'нет'. Если 'нет' — напиши "нет, причина: " и объясни.
""",
            "CHANNEL_SUMMARY_PROMPT": """
Создай краткий дайджест сообщений Telegram по NLP. Язык — русский, термины и ссылки на языке оригинала.
Структурируй по темам. Указывай номера источников [1], [2], [3]; при совпадении темы — все номера.
Кратко описывай архитектуру, методологию, результаты.

Правила:
- Без введения и заключения — сразу с содержания
- Нейтральный стиль: без эмоций, преувеличений и дословных рекомендаций авторов
- Заголовки в стиле Telegram: короткие, яркие, с 1–5 эмодзи (несинонимичных). Примеры: '<b>🧠💧 Переходы между AI-компаниями</b>', '<b>🤖🎭👅 Новый подход к обучению языковых моделей</b>', '<b>🎯🪿📈 Новые результаты на Kaggle</b>'
- Только HTML: <b>текст</b>, не Markdown
- Не создавай собственные ссылки — только номера источников [1], [2]
- Максимум {max_summary_length} символов. Для коротких сообщений — развернутое саммари, но строго в лимите
""",
            "GROUP_SUMMARY_PROMPT": """
Создай краткий дайджест сообщений из Telegram-группы по NLP. Язык — русский, термины и ссылки на языке оригинала.
Структурируй по темам. Указывай номера источников [1], [2], [3]; при совпадении темы — все номера.
Кратко описывай архитектуру, методологию, результаты.

Правила:
- Без введения и заключения — сразу с содержания
- Нейтральный стиль: без эмоций, преувеличений и дословных рекомендаций участников
- Заголовки в стиле Telegram: короткие, яркие, с 1–5 эмодзи (несинонимичных). Примеры: '<b>🧠💧 Переходы между AI-компаниями</b>', '<b>🤖🎭👅 Новый подход к обучению языковых моделей</b>', '<b>🎯🪿📈 Новые результаты на Kaggle</b>'
- Только HTML: <b>текст</b>, не Markdown
- Не создавай собственные ссылки — только номера источников [1], [2]
- Максимум {max_summary_length} символов. Для коротких сообщений — развернутое саммари, но строго в лимите
- Не добавляй вопросы пользователей, на которые нет ответа
""",
            "FIND_RELEVANT_SUMMARY_PROMPT": """
Ты эксперт по анализу текстов. Определи, в какое существующее саммари лучше всего добавить ссылку на новое сообщение.
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
