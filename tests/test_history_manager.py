import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock
import json
import os


class ShouldRunGroupSummarizationLogicTests(unittest.TestCase):
    """Tests for should_run_group_summarization logic (standalone)."""

    def test_time_calculation_logic(self):
        """Test the time calculation logic without file I/O."""
        # Simulate the logic from should_run_group_summarization
        def check_should_run(last_run_str):
            if not last_run_str:
                return True
            
            try:
                last_run = datetime.fromisoformat(last_run_str)
                if last_run.tzinfo is None:
                    last_run = last_run.replace(tzinfo=timezone.utc)
                
                now = datetime.now(timezone.utc)
                time_since_last_run = (now - last_run).total_seconds()
                
                # 24 hours = 86400 seconds
                return time_since_last_run > 86400
            except (ValueError, TypeError):
                return False
        
        # Test empty string
        self.assertTrue(check_should_run(""))
        
        # Test recent run (12 hours ago)
        recent = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()
        self.assertFalse(check_should_run(recent))
        
        # Test old run (30 hours ago)
        old = (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat()
        self.assertTrue(check_should_run(old))
        
        # Test malformed timestamp
        self.assertFalse(check_should_run("not-a-timestamp"))


class HistoryContextLogicTests(unittest.TestCase):
    """Tests for history context extraction logic."""

    def test_recent_summaries_context_logic(self):
        """Test the context extraction logic without file I/O."""
        def extract_context(summaries, days=3):
            if not summaries:
                return ""
            
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            recent = [s for s in summaries if s['date'] >= cutoff_date]
            
            if not recent:
                return ""
            
            return "\n\n".join([s['content'] for s in recent])
        
        # Test empty list
        self.assertEqual(extract_context([]), "")
        
        # Test old summaries
        old_summaries = [
            {
                'content': "Old summary",
                'date': datetime.now(timezone.utc) - timedelta(days=10)
            }
        ]
        self.assertEqual(extract_context(old_summaries, days=3), "")
        
        # Test recent summaries
        recent_summaries = [
            {
                'content': "Recent AI news",
                'date': datetime.now(timezone.utc) - timedelta(hours=12)
            },
            {
                'content': "Recent ML update",
                'date': datetime.now(timezone.utc) - timedelta(hours=6)
            }
        ]
        result = extract_context(recent_summaries, days=3)
        self.assertIn("Recent AI news", result)
        self.assertIn("Recent ML update", result)


if __name__ == '__main__':
    unittest.main()
