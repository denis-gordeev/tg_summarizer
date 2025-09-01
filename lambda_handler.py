import os
import asyncio
from typing import Any, Dict
from s3_sync import download_from_s3, upload_to_s3
from summarizer import run_summarizer


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    # Ensure we can write files (sessions, history) in Lambda
    try:
        os.chdir('/tmp')
    except Exception:
        pass

    # Pull state from S3 if configured
    download_from_s3()

    # Defaults can be overridden via EventBridge input transform
    send_message = bool(event.get('send_message', True))
    save_changes = bool(event.get('save_changes', True))
    include_today_processed_groups = bool(event.get('include_today_processed_groups', False))
    include_today_processed_messages = bool(event.get('include_today_processed_messages', False))

    # Run the summarizer
    asyncio.run(
        run_summarizer(
            send_message=send_message,
            save_changes=save_changes,
            include_today_processed_groups=include_today_processed_groups,
            include_today_processed_messages=include_today_processed_messages,
        )
    )

    # Push updated state back to S3
    upload_to_s3()

    return {
        'status': 'ok',
        'send_message': send_message,
        'save_changes': save_changes,
        'include_today_processed_groups': include_today_processed_groups,
        'include_today_processed_messages': include_today_processed_messages,
    }
