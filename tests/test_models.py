import unittest
from datetime import datetime, timezone

from models import MessageInfo, SummaryInfo


class MessageInfoTests(unittest.TestCase):
    """Tests for MessageInfo serialization/deserialization."""

    def setUp(self):
        """Set up test data."""
        self.test_date = datetime.now(timezone.utc)
        self.test_message = MessageInfo(
            text="Test message about AI and ML",
            message_id=12345,
            channel="@ai_news",
            date=self.test_date,
            link="https://example.com/article",
            is_nlp_related=True,
            is_nlp_related_reason="Relevant topic",
            is_covered_in_summaries=False,
        )

    def test_to_dict_produces_all_fields(self):
        """Should serialize all fields to dict."""
        d = self.test_message.to_dict()
        self.assertIn("text", d)
        self.assertIn("message_id", d)
        self.assertIn("channel", d)
        self.assertIn("date", d)
        self.assertIn("is_nlp_related", d)
        self.assertIn("is_nlp_related_reason", d)
        self.assertIn("is_covered_in_summaries", d)

    def test_from_dict_roundtrip(self):
        """Should deserialize back to equivalent object."""
        d = self.test_message.to_dict()
        restored = MessageInfo.from_dict(d)
        
        self.assertEqual(restored.text, self.test_message.text)
        self.assertEqual(restored.message_id, self.test_message.message_id)
        self.assertEqual(restored.channel, self.test_message.channel)
        self.assertEqual(restored.is_nlp_related, self.test_message.is_nlp_related)
        self.assertEqual(restored.is_nlp_related_reason, self.test_message.is_nlp_related_reason)
        self.assertEqual(restored.is_covered_in_summaries, self.test_message.is_covered_in_summaries)

    def test_from_dict_restores_nlp_reason(self):
        """Should restore is_nlp_related_reason field (bug fix test)."""
        d = {
            "text": "Test",
            "message_id": 1,
            "channel": "@test",
            "date": self.test_date.isoformat(),
            "link": "https://t.me/test/1",
            "is_nlp_related": True,
            "is_nlp_related_reason": "Test reason",
            "is_covered_in_summaries": False,
        }
        msg = MessageInfo.from_dict(d)
        self.assertEqual(msg.is_nlp_related_reason, "Test reason")

    def test_get_telegram_link(self):
        """Should generate correct Telegram link."""
        link = self.test_message.get_telegram_link()
        self.assertIn("ai_news", link)  # Link strips the @
        self.assertIn("12345", link)
        self.assertTrue(link.startswith("https://t.me/"))

    def test_minimal_message(self):
        """Should create MessageInfo with minimal fields."""
        msg = MessageInfo(
            text="Minimal",
            message_id=1,
            channel="@minimal",
            date=self.test_date,
            link="https://t.me/minimal/1",
        )
        self.assertEqual(msg.text, "Minimal")
        self.assertFalse(msg.is_nlp_related)
        self.assertIsNone(msg.is_nlp_related_reason)

    def test_from_dict_handles_none_text(self):
        """Should handle None text gracefully (defensive hardening)."""
        d = {
            "text": None,
            "message_id": 1,
            "channel": "@test",
            "date": self.test_date.isoformat(),
            "link": "https://t.me/test/1",
        }
        msg = MessageInfo.from_dict(d)
        self.assertEqual(msg.text, "")

    def test_from_dict_handles_missing_text(self):
        """Should handle missing text field gracefully."""
        d = {
            "message_id": 1,
            "channel": "@test",
            "date": self.test_date.isoformat(),
            "link": "https://t.me/test/1",
        }
        msg = MessageInfo.from_dict(d)
        self.assertEqual(msg.text, "")

    def test_from_dict_handles_none_channel(self):
        """Should handle None channel gracefully."""
        d = {
            "text": "Test",
            "message_id": 1,
            "channel": None,
            "date": self.test_date.isoformat(),
            "link": "",
        }
        msg = MessageInfo.from_dict(d)
        self.assertEqual(msg.channel, "")


class SummaryInfoTests(unittest.TestCase):
    """Tests for SummaryInfo serialization/deserialization."""

    def setUp(self):
        """Set up test data."""
        self.test_date = datetime.now(timezone.utc)
        self.test_summary = SummaryInfo(
            content="Summary content with <b>HTML</b> tags",
            date=self.test_date,
            message_count=5,
            channels=["@ai_news", "@ml_daily"],
            message_id=67890,
        )

    def test_to_dict_produces_all_fields(self):
        """Should serialize all fields to dict."""
        d = self.test_summary.to_dict()
        self.assertIn("content", d)
        self.assertIn("date", d)
        self.assertIn("message_count", d)
        self.assertIn("channels", d)
        self.assertIn("message_id", d)

    def test_from_dict_roundtrip(self):
        """Should deserialize back to equivalent object."""
        d = self.test_summary.to_dict()
        restored = SummaryInfo.from_dict(d)
        
        self.assertEqual(restored.content, self.test_summary.content)
        self.assertEqual(restored.message_count, self.test_summary.message_count)
        self.assertEqual(restored.channels, self.test_summary.channels)
        self.assertEqual(restored.message_id, self.test_summary.message_id)

    def test_channels_is_list(self):
        """Should preserve channels as list."""
        d = self.test_summary.to_dict()
        restored = SummaryInfo.from_dict(d)
        self.assertIsInstance(restored.channels, list)
        self.assertEqual(len(restored.channels), 2)

    def test_empty_channels(self):
        """Should handle empty channels list."""
        summary = SummaryInfo(
            content="Summary",
            date=self.test_date,
            message_count=0,
            channels=[],
            message_id=None,
        )
        d = summary.to_dict()
        restored = SummaryInfo.from_dict(d)
        self.assertEqual(restored.channels, [])
        self.assertIsNone(restored.message_id)


if __name__ == '__main__':
    unittest.main()
