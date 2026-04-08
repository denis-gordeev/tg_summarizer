import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import s3_sync


class SyncFileConfigTests(unittest.TestCase):
    def test_get_sync_files_uses_override_and_removes_duplicates(self):
        with patch.dict(
            os.environ,
            {"STATE_SYNC_FILES": "alpha.json, beta.json, alpha.json"},
            clear=False,
        ):
            self.assertEqual(s3_sync._get_sync_files(), ["alpha.json", "beta.json"])

    def test_build_s3_key_trims_prefix_slashes(self):
        self.assertEqual(
            s3_sync._build_s3_key("/tg_summarizer/prod/", "state.json"),
            "tg_summarizer/prod/state.json",
        )
        self.assertEqual(s3_sync._build_s3_key("", "state.json"), "state.json")


class FakeS3Client:
    def __init__(self):
        self.download_calls = []
        self.upload_calls = []

    def download_file(self, bucket, key, destination):
        self.download_calls.append((bucket, key, destination))
        Path(destination).write_text("downloaded", encoding="utf-8")

    def upload_file(self, source, bucket, key):
        self.upload_calls.append((source, bucket, key))


class S3SyncOperationTests(unittest.TestCase):
    def test_download_from_s3_fetches_all_configured_files(self):
        client = FakeS3Client()

        with tempfile.TemporaryDirectory() as tmpdir, \
             patch.dict(
                 os.environ,
                 {
                     "STATE_S3_BUCKET": "bucket",
                     "STATE_S3_PREFIX": "prefix",
                     "STATE_SYNC_FILES": "state/a.json,state/b.json",
                 },
                 clear=False,
             ), \
             patch.object(s3_sync, "_get_s3_client", return_value=client):
            previous_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                s3_sync.download_from_s3()
                self.assertEqual(
                    client.download_calls,
                    [
                        ("bucket", "prefix/state/a.json", "state/a.json"),
                        ("bucket", "prefix/state/b.json", "state/b.json"),
                    ],
                )
                self.assertTrue(Path("state/a.json").exists())
                self.assertTrue(Path("state/b.json").exists())
            finally:
                os.chdir(previous_cwd)

    def test_upload_to_s3_skips_missing_files(self):
        client = FakeS3Client()

        with tempfile.TemporaryDirectory() as tmpdir, \
             patch.dict(
                 os.environ,
                 {
                     "STATE_S3_BUCKET": "bucket",
                     "STATE_S3_PREFIX": "prefix",
                     "STATE_SYNC_FILES": "state/a.json,state/missing.json",
                 },
                 clear=False,
             ), \
             patch.object(s3_sync, "_get_s3_client", return_value=client):
            previous_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                Path("state").mkdir()
                Path("state/a.json").write_text("content", encoding="utf-8")
                s3_sync.upload_to_s3()
                self.assertEqual(
                    client.upload_calls,
                    [("state/a.json", "bucket", "prefix/state/a.json")],
                )
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()
