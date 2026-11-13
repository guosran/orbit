#!/usr/bin/env python3
"""Execute a sealed actual r19 selected replay after checking its real files."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time


_verifier_spec=importlib.util.spec_from_file_location('orbit_r19_cache_verification',Path(__file__).with_name('qwen-r19-verification.py'))
verification=importlib.util.module_from_spec(_verifier_spec);_verifier_spec.loader.exec_module(verification)
sha=verification.checksum

def require(condition,message):
    if not condition:raise ValueError(message)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt',type=Path,required=True);parser.add_argument('--receipt-sha256',required=True)
    parser.add_argument('--log',type=Path,required=True);parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--timeout',type=float,default=600)
    a=parser.parse_args();require(sha(a.receipt)==a.receipt_sha256,'Selected command receipt changed')
    receipt=json.loads(a.receipt.read_text());require(receipt['schema']=='orbit.qwen-r19-command.v1','Unknown selected execution protocol')
    require(not a.log.exists() and not a.result.exists(),'Preserve previous execution evidence')
    verification_path=Path(str(a.result)+'.verification-cache.json')
    require(not verification_path.exists(),'Preserve previous verifier evidence')
    try:
        for row in receipt['files']+receipt['registered_files']:verification.verify(row)
    finally:
        # Keep actual advice/failure evidence even when preflight rejects and
        # no NPU subprocess or result measurement is ever created.
        verification_path.write_text(json.dumps(verification.receipt(),indent=2)+'\n')
    start=time.monotonic()
    with a.log.open('x') as stream:
        try:
            process=subprocess.run(receipt['command'],stdout=stream,stderr=subprocess.STDOUT,timeout=a.timeout,check=False)
            result=dict(returncode=process.returncode,timeout=False)
        except subprocess.TimeoutExpired:result=dict(returncode=None,timeout=True)
    result.update(elapsed_seconds=time.monotonic()-start,replay_id=receipt['replay_id'],program_graph_id=receipt['program_graph_id'],receipt_sha256=a.receipt_sha256)
    # Code/artifacts are immutable for the entire command, including preparation.
    drift=[]
    for row in receipt['files']+receipt['registered_files']:
        try:verification.verify(row)
        except (OSError,ValueError) as error:drift.append(dict(file=row['file'],requested_file=row.get('requested_file',row['file']),error=str(error)))
    result['file_drift']=drift
    result['verification_cache']=verification.receipt()
    verification_path.write_text(json.dumps(verification.receipt(),indent=2)+'\n')
    with a.result.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result));require(not drift,'Selected execution files drifted while running')
    raise SystemExit(result['returncode'] if result['returncode'] is not None else 124)


if __name__=='__main__':main()
