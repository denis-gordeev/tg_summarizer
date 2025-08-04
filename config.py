import os
from dotenv import load_dotenv

load_dotenv()

# Get environment variables with proper error handling
api_id_str = os.getenv('TELEGRAM_API_ID')
if not api_id_str:
    raise ValueError("TELEGRAM_API_ID environment variable is required")
API_ID = int(api_id_str)

api_hash = os.getenv('TELEGRAM_API_HASH')
if not api_hash:
    raise ValueError("TELEGRAM_API_HASH environment variable is required")
API_HASH = api_hash

bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
if not bot_token:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")
BOT_TOKEN = bot_token

target_channel = os.getenv('TARGET_CHANNEL')
if not target_channel:
    raise ValueError("TARGET_CHANNEL environment variable is required")
TARGET_CHANNEL = target_channel

openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")
OPENAI_API_KEY = openai_api_key

source_channels_str = os.getenv('SOURCE_CHANNELS', '')
SOURCE_CHANNELS = set([c.strip() for c in source_channels_str.split(',') if c.strip()])

# Новые переменные для групп
source_groups_str = os.getenv('SOURCE_GROUPS', '')
SOURCE_GROUPS = [g.strip() for g in source_groups_str.split(',') if g.strip()]

# File paths
ABBREVIATIONS_FILE = 'channel_abbreviations.json'
HISTORY_FILE = 'summarization_history.json'
SUMMARIES_HISTORY_FILE = 'summaries_history.json'
DISCOVERED_CHANNELS_FILE = 'discovered_channels.json'
GROUP_HISTORY_FILE = 'group_summarization_history.json'
GROUP_SUMMARIES_HISTORY_FILE = 'group_summaries_history.json'
GROUP_LAST_RUN_FILE = 'group_last_run.json'

# Constants
SIMILARITY_THRESHOLD = 0.9
ENABLE_SUMMARIES_DEDUPLICATION = True
ENABLE_SUMMARY_UPDATES = True 