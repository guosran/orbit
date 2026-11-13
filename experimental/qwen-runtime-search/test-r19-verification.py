#!/usr/bin/env python3
"""Device-free checks of shared same-descriptor read-only cache advice."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

path=Path(__file__).with_name('qwen-r19-verification.py')
spec=importlib.util.spec_from_file_location('cache_verification_test',path)
verification=importlib.util.module_from_spec(spec);spec.loader.exec_module(verification)


class VerificationTests(unittest.TestCase):
    def test_large_file_advice_after_hash_same_open_descriptor(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'synthetic';path.write_bytes(b'actual checked bytes')
            calls=[]
            def advice(fd,start,length,mode):
                import os
                self.assertEqual(os.lseek(fd,0,os.SEEK_CUR),path.stat().st_size)
                calls.append((fd,start,length,mode))
            # Synthetic stat selects the large-file branch without reading a
            # model or creating cache pressure during formal NPU training.
            with patch.object(verification._driver.os,'fstat',return_value=SimpleNamespace(st_size=64*1024*1024)),patch.object(verification._driver.os,'posix_fadvise',side_effect=advice):
                actual=verification.checksum(path)
            self.assertEqual(actual,hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(len(calls),1)
            self.assertTrue(verification.receipt()['events'][-1]['success'])

    def test_advice_failure_rejects_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'synthetic';path.write_bytes(b'bytes')
            with patch.object(verification._driver.os,'fstat',return_value=SimpleNamespace(st_size=64*1024*1024)),patch.object(verification._driver.os,'posix_fadvise',side_effect=OSError('synthetic rejection')):
                with self.assertRaisesRegex(ValueError,'cache advice failed'):verification.checksum(path)
            self.assertFalse(verification.receipt()['events'][-1]['success'])


if __name__=='__main__':unittest.main()
