from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class MessageInfo:
    """Информация о сообщении Telegram"""
    text: str
    channel: str
    message_id: int
    date: datetime
    link: str
    is_nlp_related: Optional[bool] = None
    is_nlp_related_reason: Optional[str] = None
    is_covered_in_summaries: Optional[bool] = None
    
    def get_telegram_link(self) -> str:
        """Генерирует ссылку на оригинальное сообщение в Telegram"""
        # Убираем @ из названия канала для формирования ссылки
        channel_name = self.channel.lstrip('@')
        return f"https://t.me/{channel_name}/{self.message_id}"
    
    def to_dict(self) -> dict:
        """Конвертирует объект в словарь для сохранения в JSON"""
        return {
            'text': self.text,
            'channel': self.channel,
            'message_id': self.message_id,
            'date': self.date.isoformat(),
            'link': self.link,
            'is_nlp_related': self.is_nlp_related,
            'is_covered_in_summaries': self.is_covered_in_summaries,
            'is_nlp_related_reason': self.is_nlp_related_reason
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MessageInfo':
        """Создает объект из словаря"""
        return cls(
            text=data['text'],
            channel=data['channel'],
            message_id=data['message_id'],
            date=datetime.fromisoformat(data['date']),
            link=data['link'],
            is_nlp_related=data.get('is_nlp_related'),
            is_covered_in_summaries=data.get('is_covered_in_summaries')
        )


@dataclass
class SummaryInfo:
    """Информация о созданном саммари"""
    content: str
    date: datetime
    message_count: int
    channels: List[str]
    message_id: Optional[int] = None  # ID сообщения в целевом канале
    
    def to_dict(self) -> dict:
        """Конвертирует объект в словарь для сохранения в JSON"""
        return {
            'content': self.content,
            'date': self.date.isoformat(),
            'message_count': self.message_count,
            'channels': self.channels,
            'message_id': self.message_id
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SummaryInfo':
        """Создает объект из словаря"""
        return cls(
            content=data['content'],
            date=datetime.fromisoformat(data['date']),
            message_count=data['message_count'],
            channels=data['channels'],
            message_id=data.get('message_id')
        ) 