#!/usr/bin/env python3
"""Reuse the sealed training verifier's cache advice for all heldout paths.

Complete SHA256 reads remain mandatory. For files >=64MiB the sealed verifier
issues DONTNEED on the same read-only descriptor, then closes it. This records
successful kernel advice, not a guarantee of physical cache eviction.
"""
import hashlib
import importlib.util
from pathlib import Path

DRIVER=Path(__file__).with_name('run-independent-r19-campaign-v2.py').resolve()
DRIVER_SHA256='598a38d46a0231b263053d861916cdd20ae8de01b85e7bcd2f8e36f411ddc19c'


def bootstrap_sha(path):
    # Only the tiny verifier modules use this bootstrap, never model/proof data.
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


if bootstrap_sha(DRIVER)!=DRIVER_SHA256:raise ValueError('Sealed cache-aware training verifier changed')
_spec=importlib.util.spec_from_file_location('orbit_sealed_r19_verifier',DRIVER)
_driver=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_driver)
checksum=_driver.checksum


def files():
    return [dict(file=str(Path(__file__).resolve()),sha256=bootstrap_sha(__file__)),dict(file=str(DRIVER),sha256=DRIVER_SHA256)]


def receipt():
    return dict(policy=dict(_driver.CACHE_POLICY),code_files=files(),events=[dict(event) for event in _driver.CACHE_ADVICE_EVENTS])


def binding(row):
    """Preserve every requested alias and its separately sealed canonical blob."""
    declared=Path(row.get('file',row.get('path'))).absolute()
    requested=Path(row.get('requested_file',declared)).absolute()
    actual=requested.resolve(strict=True)
    # Legacy proofs have one requested spelling only. Normalize it once;
    # all emitted replay records retain both spellings. Sealed dependency
    # records already provide requested_file and must match their old blob.
    canonical=declared if 'requested_file' in row else declared.resolve(strict=True)
    if actual!=canonical or canonical.resolve(strict=True)!=canonical:
        raise ValueError('Sealed requested/resolved file binding changed: '+str(requested))
    return dict(file=str(canonical),requested_file=str(requested),sha256=row['sha256'])


def verify(row):
    bound=binding(row)
    if checksum(bound['requested_file'])!=bound['sha256']:
        raise ValueError('Sealed dependency bytes changed: '+bound['requested_file'])
    return bound


def file_record(path):
    requested=Path(path).absolute()
    return dict(file=str(requested.resolve(strict=True)),requested_file=str(requested),sha256=checksum(requested))
