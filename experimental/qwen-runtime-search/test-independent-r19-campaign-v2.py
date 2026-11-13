#!/usr/bin/env python3
"""Verify actual SHA bytes, read-only cache advice, and unchanged failure policy."""
import fcntl
import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = load("cache_neutral_driver", HERE / "run-independent-r19-campaign-v2.py")
fixtures = load("independent_fixtures", HERE / "test-independent-r19-campaign.py")
fixtures.DRIVER = HERE / "run-independent-r19-campaign-v2.py"
CampaignTests = fixtures.IndependentTests


class CacheAdviceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "bytes.bin"
        self.payload = bytes(range(64))
        self.path.write_bytes(self.payload)
        driver.CACHE_ADVICE_EVENTS.clear()

    def test_hash_is_actual_bytes_and_advice_uses_same_readonly_descriptor(self):
        descriptors = []

        def advise(fd, offset, length, advice):
            self.assertEqual(fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE, os.O_RDONLY)
            self.assertEqual((offset, length, advice), (0, 0, os.POSIX_FADV_DONTNEED))
            self.assertEqual(os.fstat(fd).st_ino, self.path.stat().st_ino)
            self.assertEqual(os.lseek(fd, 0, os.SEEK_CUR), len(self.payload))
            descriptors.append(fd)

        with mock.patch.object(driver, "CACHE_ADVICE_THRESHOLD", len(self.payload)), \
                mock.patch.object(driver.os, "posix_fadvise", side_effect=advise):
            actual = driver.checksum(self.path)
        self.assertEqual(actual, hashlib.sha256(self.payload).hexdigest())
        self.assertEqual(len(descriptors), 1)
        self.assertTrue(driver.CACHE_ADVICE_EVENTS[-1]["success"])
        self.assertEqual(driver.CACHE_ADVICE_EVENTS[-1]["bytes"], len(self.payload))

    def test_small_file_keeps_actual_hash_without_advice(self):
        with mock.patch.object(driver.os, "posix_fadvise") as advise:
            actual = driver.checksum(self.path)
        advise.assert_not_called()
        self.assertEqual(actual, hashlib.sha256(self.payload).hexdigest())
        self.assertEqual(driver.CACHE_ADVICE_EVENTS, [])

    def test_unsupported_or_failed_advice_is_explicit_and_fail_closed(self):
        for error in (OSError("unsupported advice"), AttributeError("missing API")):
            with self.subTest(error=error), \
                    mock.patch.object(driver, "CACHE_ADVICE_THRESHOLD", 1), \
                    mock.patch.object(driver.os, "posix_fadvise", side_effect=error), \
                    mock.patch.object(driver.subprocess, "run") as run:
                with self.assertRaisesRegex(ValueError, "cache advice failed"):
                    driver.checksum(self.path)
                run.assert_not_called()
                self.assertFalse(driver.CACHE_ADVICE_EVENTS[-1]["success"])
                self.assertIn("error", driver.CACHE_ADVICE_EVENTS[-1])


if __name__ == "__main__":
    unittest.main()
