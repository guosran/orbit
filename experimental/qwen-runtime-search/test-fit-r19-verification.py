#!/usr/bin/env python3
"""Reject retargeted requested aliases even when both blobs have equal bytes."""
import importlib.util
from pathlib import Path
import tempfile


def main():
    path=Path(__file__).with_name('fit-r19-catalogue.py')
    spec=importlib.util.spec_from_file_location('fit_verification_fixture',path)
    fit=importlib.util.module_from_spec(spec);spec.loader.exec_module(fit)
    with tempfile.TemporaryDirectory(prefix='orbit-r19-verifier-test-') as directory:
        root=Path(directory);a=root/'declared';b=root/'equal-content';link=root/'requested'
        a.write_bytes(b'identical fixture bytes');b.write_bytes(a.read_bytes());link.symlink_to(a)
        evidence=fit.record(link)
        assert evidence['file']==str(a) and evidence['requested_file']==str(link)
        assert fit.verified(evidence)==link
        link.unlink();link.symlink_to(b)
        assert fit.record(a)['sha256']==fit.record(b)['sha256']
        try:fit.verified(evidence)
        except ValueError:pass
        else:raise AssertionError('Same-content alias retarget silently admitted')
        link.unlink();link.symlink_to(a);fit.verified(evidence)
        a.write_bytes(b'changed declared bytes')
        try:fit.verified(evidence)
        except ValueError:pass
        else:raise AssertionError('Actual evidence byte mutation silently admitted')
    print('PASS: requested path retained; same-content retarget and changed bytes rejected')


if __name__=='__main__':main()
