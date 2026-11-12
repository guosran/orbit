#!/usr/bin/env python3
"""Execute one already reviewed/sealed replay receipt without changing its plan.

The original runner owns the NPU lock and all measurement/memory guards. This
process checks immutable protocol/code identities, then records process status.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def require(ok,message):
    if not ok:raise ValueError(message)
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--receipt-sha256',required=True);p.add_argument('--log',type=Path,required=True);p.add_argument('--result',type=Path,required=True);p.add_argument('--timeout',type=int,default=180);a=p.parse_args()
    require(sha(a.receipt)==a.receipt_sha256,'prepared command receipt changed')
    r=json.loads(a.receipt.read_text());command=r['command'];environment=r['environment']
    require(sha(command[1])==r['frozen_runner_sha256'],'frozen runner changed')
    require(sha(Path(command[1]).with_name('qwen-node-runtime.py'))==r['frozen_node_runtime_sha256'],'frozen node runtime changed')
    require(command.count('--region-helper')==1,'missing/duplicate selector helper')
    require(sha(command[command.index('--region-helper')+1])==r['selector_sha256'],'selector helper changed')
    require(sha(environment['ORBIT_QWEN_BASE_REGION_HELPER'])==r['base_helper_sha256'],'frozen region implementation changed')
    replay=json.loads(Path(environment['ORBIT_QWEN_SELECTED_REPLAY']).read_text())
    require(replay['replay_id']==r['replay_id']==hashlib.sha256(canonical({k:v for k,v in replay.items() if k!='replay_id'})).hexdigest(),'prepared source/plan replay changed')
    for proof in replay['source_files']:require(sha(proof['file'])==proof['sha256'],'registered source/artifact changed')
    require(a.timeout>0 and not a.result.exists(),'invalid timeout or existing result')
    started=time.monotonic()
    with a.log.open('x') as log:
        result=subprocess.run(['timeout','-k','10s',str(a.timeout)+'s']+command,env=dict(os.environ,**environment),stdout=log,stderr=subprocess.STDOUT,check=False)
    with a.result.open('x') as f:f.write(json.dumps(dict(receipt_sha256=a.receipt_sha256,replay_id=replay['replay_id'],program_graph_id=r['program_graph_id'],manual_baseline=r['manual_baseline'],returncode=result.returncode,seconds=time.monotonic()-started),indent=2)+'\n')
    raise SystemExit(result.returncode)
if __name__=='__main__':main()
