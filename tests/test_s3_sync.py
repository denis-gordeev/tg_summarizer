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
        Path(destination).write_text("{}", encoding="utf-8")

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

    def test_download_logs_summary_counts(self):
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
             patch.object(s3_sync, "_get_s3_client", return_value=client), \
             patch.object(s3_sync.logger, "info") as mock_log:
            previous_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                s3_sync.download_from_s3()
                summary_logs = [call for call in mock_log.call_args_list
                                if "S3 download" in str(call)]
                self.assertTrue(len(summary_logs) > 0, "Expected S3 download summary log")
                self.assertEqual(summary_logs[0][0][1], 2)
                self.assertEqual(summary_logs[0][0][2], 0)
            finally:
                os.chdir(previous_cwd)

    def test_upload_logs_summary_counts(self):
        client = FakeS3Client()

        with tempfile.TemporaryDirectory() as tmpdir, \
             patch.dict(
                 os.environ,
                 {
                     "STATE_S3_BUCKET": "bucket",
                     "STATE_S3_PREFIX": "prefix",
                     "STATE_SYNC_FILES": "state/a.json",
                 },
                 clear=False,
             ), \
             patch.object(s3_sync, "_get_s3_client", return_value=client), \
             patch.object(s3_sync.logger, "info") as mock_log:
            previous_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                Path("state").mkdir()
                Path("state/a.json").write_text("content", encoding="utf-8")
                s3_sync.upload_to_s3()
                summary_logs = [call for call in mock_log.call_args_list
                                if "S3 upload" in str(call)]
                self.assertTrue(len(summary_logs) > 0, "Expected S3 upload summary log")
                self.assertEqual(summary_logs[0][0][1], 1)
                self.assertEqual(summary_logs[0][0][2], 0)
                self.assertEqual(summary_logs[0][0][3], 0)
            finally:
                os.chdir(previous_cwd)

    def test_upload_skips_empty_files(self):
        client = FakeS3Client()

        with tempfile.TemporaryDirectory() as tmpdir, \
             patch.dict(
                 os.environ,
                 {
                     "STATE_S3_BUCKET": "bucket",
                     "STATE_S3_PREFIX": "prefix",
                     "STATE_SYNC_FILES": "state/empty.json,state/nonempty.json",
                 },
                 clear=False,
             ), \
             patch.object(s3_sync, "_get_s3_client", return_value=client), \
             patch.object(s3_sync.logger, "warning") as mock_warn:
            previous_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                Path("state").mkdir()
                Path("state/empty.json").write_text("", encoding="utf-8")
                Path("state/nonempty.json").write_text("{}", encoding="utf-8")
                s3_sync.upload_to_s3()
                self.assertEqual(len(client.upload_calls), 1)
                self.assertIn("nonempty.json", client.upload_calls[0][0])
                warn_msgs = [str(c) for c in mock_warn.call_args_list]
                self.assertTrue(any("empty file" in w for w in warn_msgs))
            finally:
                os.chdir(previous_cwd)


class S3ClientCachingTests(unittest.TestCase):
    def test_get_s3_client_caches_client(self):
        """_get_s3_client should return the same client on subsequent calls."""
        import types
        from unittest.mock import MagicMock

        fake_boto3 = types.ModuleType("boto3")
        mock_client = MagicMock()
        fake_boto3.client = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"boto3": fake_boto3}):
            s3_sync._s3_client = None
            client1 = s3_sync._get_s3_client()
            client2 = s3_sync._get_s3_client()
            self.assertIs(client1, client2, "S3 client should be cached and reused")
            fake_boto3.client.assert_called_once_with("s3")
        s3_sync._s3_client = None


class S3DownloadJSONValidationTests(unittest.TestCase):
    def test_download_rejects_invalid_json(self):
        """Downloaded .json files that aren't valid JSON should be removed and skipped."""
        class CorruptS3Client:
            def download_file(self, bucket, key, destination):
                Path(destination).write_text("not json {{{", encoding="utf-8")

        client = CorruptS3Client()

        with tempfile.TemporaryDirectory() as tmpdir, \
             patch.dict(
                 os.environ,
                 {
                     "STATE_S3_BUCKET": "bucket",
                     "STATE_S3_PREFIX": "prefix",
                     "STATE_SYNC_FILES": "state/corrupt.json",
                 },
                 clear=False,
             ), \
             patch.object(s3_sync, "_get_s3_client", return_value=client), \
             patch.object(s3_sync.logger, "info") as mock_log:
            previous_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                s3_sync.download_from_s3()
                self.assertFalse(Path("state/corrupt.json").exists())
                summary_logs = [call for call in mock_log.call_args_list
                                if "S3 download" in str(call)]
                self.assertTrue(len(summary_logs) > 0)
                self.assertEqual(summary_logs[0][0][1], 0)
                self.assertEqual(summary_logs[0][0][2], 1)
            finally:
                os.chdir(previous_cwd)

    def test_download_accepts_valid_json(self):
        """Downloaded .json files with valid JSON should be kept."""
        class ValidS3Client:
            def download_file(self, bucket, key, destination):
                Path(destination).write_text('{"key": "value"}', encoding="utf-8")

        client = ValidS3Client()

        with tempfile.TemporaryDirectory() as tmpdir, \
             patch.dict(
                 os.environ,
                 {
                     "STATE_S3_BUCKET": "bucket",
                     "STATE_S3_PREFIX": "prefix",
                     "STATE_SYNC_FILES": "state/valid.json",
                 },
                 clear=False,
             ), \
             patch.object(s3_sync, "_get_s3_client", return_value=client):
            previous_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                s3_sync.download_from_s3()
                self.assertTrue(Path("state/valid.json").exists())
            finally:
                os.chdir(previous_cwd)

    def test_download_skips_json_validation_for_non_json(self):
        """Non-JSON files (e.g. .session) should not be validated as JSON."""
        class SessionS3Client:
            def download_file(self, bucket, key, destination):
                Path(destination).write_text("binary session data", encoding="utf-8")

        client = SessionS3Client()

        with tempfile.TemporaryDirectory() as tmpdir, \
             patch.dict(
                 os.environ,
                 {
                     "STATE_S3_BUCKET": "bucket",
                     "STATE_S3_PREFIX": "prefix",
                     "STATE_SYNC_FILES": "state/tg.session",
                 },
                 clear=False,
             ), \
             patch.object(s3_sync, "_get_s3_client", return_value=client), \
             patch.object(s3_sync.logger, "info") as mock_log:
            previous_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                s3_sync.download_from_s3()
                self.assertTrue(Path("state/tg.session").exists())
                summary_logs = [call for call in mock_log.call_args_list
                                if "S3 download" in str(call)]
                self.assertTrue(len(summary_logs) > 0)
                self.assertEqual(summary_logs[0][0][1], 1)
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()
