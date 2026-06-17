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


class ChannelAbbreviationCachingTests(unittest.TestCase):
    """Tests for channel abbreviation caching."""

    def test_abbreviations_cache_returns_same_dict_on_repeated_calls(self):
        """load_channel_abbreviations should return cached result on 2nd call."""
        import importlib
        import types
        import sys

        fake_utils = types.ModuleType("utils")
        call_count = 0

        def fake_load_json_file(filepath, default=None):
            nonlocal call_count
            call_count += 1
            return {"channel_abbreviations": {"@test_channel": "TC"}}

        fake_utils.load_json_file = fake_load_json_file
        fake_utils.save_json_file = lambda *a, **kw: True
        fake_utils.now_iso = lambda: "2026-01-01T00:00:00"

        fake_config = types.ModuleType("config")
        fake_config.ABBREVIATIONS_FILE = "abbrev.json"
        fake_config.DISCOVERED_CHANNELS_FILE = "discovered.json"

        sys.modules["utils"] = fake_utils
        sys.modules["config"] = fake_config
        sys.modules.pop("channel_manager", None)

        cm = importlib.import_module("channel_manager")

        try:
            cm._abbreviations_cache = None
            result1 = cm.load_channel_abbreviations()
            self.assertEqual(call_count, 1)
            result2 = cm.load_channel_abbreviations()
            self.assertEqual(call_count, 1, "Second call should use cache, not read file again")
            self.assertIs(result1, result2)
        finally:
            cm._abbreviations_cache = None
            sys.modules.pop("channel_manager", None)

    def test_save_channel_abbreviation_invalidates_cache(self):
        """save_channel_abbreviation should invalidate the cache so next load reads fresh data."""
        import importlib
        import types
        import sys

        call_count = 0

        def fake_load_json_file(filepath, default=None):
            nonlocal call_count
            call_count += 1
            return {"channel_abbreviations": {"@ch1": "C1"}}

        saved_data = {}

        def fake_save_json_file(filepath, data, msg):
            saved_data.update(data)

        fake_utils = types.ModuleType("utils")
        fake_utils.load_json_file = fake_load_json_file
        fake_utils.save_json_file = fake_save_json_file
        fake_utils.now_iso = lambda: "2026-01-01T00:00:00"

        fake_config = types.ModuleType("config")
        fake_config.ABBREVIATIONS_FILE = "abbrev.json"
        fake_config.DISCOVERED_CHANNELS_FILE = "discovered.json"

        sys.modules["utils"] = fake_utils
        sys.modules["config"] = fake_config
        sys.modules.pop("channel_manager", None)

        cm = importlib.import_module("channel_manager")

        try:
            cm._abbreviations_cache = None
            cm.load_channel_abbreviations()
            self.assertEqual(call_count, 1)

            cm.save_channel_abbreviation("@ch2", "C2")
            self.assertIsNone(cm._abbreviations_cache, "Cache should be invalidated after save")

            cm.load_channel_abbreviations()
            self.assertEqual(call_count, 2, "After invalidation, should re-read from file")
        finally:
            cm._abbreviations_cache = None
            sys.modules.pop("channel_manager", None)


class GetAllSourceChannelsSortedTests(unittest.TestCase):
    """Tests for get_all_source_channels returning sorted (deterministic) order."""

    def test_channels_returned_in_sorted_order(self):
        import importlib
        import types
        import sys

        fake_utils = types.ModuleType("utils")
        fake_utils.load_json_file = lambda filepath, default=None: {}
        fake_utils.save_json_file = lambda *a, **kw: True
        fake_utils.now_iso = lambda: "2026-01-01T00:00:00"

        fake_config = types.ModuleType("config")
        fake_config.ABBREVIATIONS_FILE = "abbrev.json"
        fake_config.DISCOVERED_CHANNELS_FILE = "discovered.json"
        fake_config.SOURCE_CHANNELS = {"@z_channel", "@a_channel", "@m_channel"}

        sys.modules["utils"] = fake_utils
        sys.modules["config"] = fake_config
        sys.modules.pop("channel_manager", None)

        cm = importlib.import_module("channel_manager")

        try:
            result = cm.get_all_source_channels()
            self.assertEqual(result, ["@a_channel", "@m_channel", "@z_channel"])
        finally:
            sys.modules.pop("channel_manager", None)


if __name__ == '__main__':
    unittest.main()
