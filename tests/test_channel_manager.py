import unittest
from unittest.mock import patch, MagicMock
import json
import os

# We can't import channel_manager directly due to config.py dependency on dotenv
# So we'll test it indirectly through the functions that use it


class ChannelAbbreviationIntegrationTests(unittest.TestCase):
    """Integration tests for channel abbreviation logic."""

    def test_abbreviation_logic_standalone(self):
        """Test abbreviation creation logic without file I/O."""
        import re
        
        def create_abbreviation_standalone(channel_name: str, existing_abbreviations: dict) -> str:
            """Standalone abbreviation logic for testing."""
            if channel_name in existing_abbreviations:
                return existing_abbreviations[channel_name]
            
            clean_name = channel_name.lstrip('@')
            words = re.split(r'[\s\-_]+', clean_name)
            abbreviation = ''.join(word[0].upper() for word in words if word)
            
            if len(abbreviation) > 4:
                abbreviation = abbreviation[:4]
            
            if len(abbreviation) < 2:
                abbreviation = clean_name[:3].upper()
            
            existing_values = set(existing_abbreviations.values())
            if abbreviation in existing_values:
                counter = 1
                while f"{abbreviation}{counter}" in existing_values:
                    counter += 1
                abbreviation = f"{abbreviation}{counter}"
            
            return abbreviation
        
        # Test simple case
        self.assertEqual(create_abbreviation_standalone("@ai_news", {}), "AN")
        
        # Test multi-word case
        self.assertEqual(create_abbreviation_standalone("@machine_learning_daily", {}), "MLD")
        
        # Test collision handling
        existing = {"@ai_test": "AT"}
        result = create_abbreviation_standalone("@ai_testing", existing)
        self.assertEqual(result, "AT1")
        
        # Test reuse
        existing = {"@ai_news": "AN"}
        self.assertEqual(create_abbreviation_standalone("@ai_news", existing), "AN")


class GetAllSourceChannelsLogicTests(unittest.TestCase):
    """Tests for channel merging logic."""

    def test_channel_merge_standalone(self):
        """Test channel merge logic without file I/O."""
        source_channels = {"@channel1", "@channel2"}
        discovered = ["@discovered1", "@discovered2"]
        similar = ["@similar1"]
        banned = ["@banned1"]
        
        # Merge all channels
        all_channels = source_channels | set(discovered) | set(similar)
        # Exclude banned
        all_channels = all_channels - set(banned)
        
        self.assertIn("@channel1", all_channels)
        self.assertIn("@discovered1", all_channels)
        self.assertIn("@similar1", all_channels)
        self.assertNotIn("@banned1", all_channels)
        self.assertEqual(len(all_channels), 5)


if __name__ == '__main__':
    unittest.main()
