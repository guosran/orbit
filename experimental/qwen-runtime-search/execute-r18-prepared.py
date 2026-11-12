#!/usr/bin/env python3
"""Execute an immutable prepared r18 protocol after rechecking its real files."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import time


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--receipt-sha256',required=True)
    p.add_argument('--log',type=Path,required=True)
    p.add_argument('--result',type=Path,required=True)
    p.add_argument('--timeout',type=float,default=240)
    a=p.parse_args()
    helper=module(Path(__file__).with_name('replay-r18-selection.py'),'r18_prepared_protocol')
    helper.require(helper.sha(a.receipt)==a.receipt_sha256,'prepared command receipt changed')
    receipt=helper.load(a.receipt);root=a.receipt.parent
    helper.require(receipt['schema']=='orbit.qwen-r18-command.v1','unknown prepared command protocol')
    for field in ('graph','catalogue'):
        path=receipt[field+'_file'];expected=receipt[field+'_sha256']
        helper.require((path is None)==(expected is None),'incomplete original '+field+' identity')
        if path is not None:helper.require(helper.sha(path)==expected,'original '+field+' changed after selection seal')
    for name,key in [('replay.json','replay_sha256'),('plan.json','plan_sha256')]:
        helper.require(helper.sha(root/name)==receipt[key],'prepared '+name+' changed')
    helper.require(helper.sha(Path(__file__).with_name('replay-r18-selection.py'))==receipt['translator_sha256'],'translator changed after selection seal')
    helper.require(helper.sha(Path(__file__).with_name('routed-r18-runtime.py'))==receipt['routing_sha256'],'routing shim changed after selection seal')
    frozen=Path(receipt['frozen_runtime']);seal=frozen/'freeze.json'
    helper.require(helper.sha(seal)==receipt['frozen_seal_sha256'],'frozen seal changed')
    for row in helper.load(seal)['files']:
        helper.require(helper.sha(row['file'])==row['sha256'],'frozen code/artifact changed: '+row['file'])
    replay=helper.load(root/'replay.json');plan=helper.load(root/'plan.json')
    helper.require(helper.digest({k:v for k,v in replay.items() if k!='replay_id'})==receipt['replay_id']==replay['replay_id'],'replay identity mismatch')
    helper.require(plan==replay['plan'],'prepared runtime plan differs from selected replay')
    for row in replay.get('source_files',[]):
        helper.require(helper.sha(row['file'])==row['sha256'],'actual registered source/artifact changed')
    source_path=Path(receipt['prefill_source']);helper.require(helper.sha(source_path)==receipt['prefill_source_sha256'],'prefill graph changed')
    adapter=module(frozen/'scripts/qwen-r18-runtime.py','r18_execute_adapter')
    routing=module(Path(__file__).with_name('routed-r18-runtime.py'),'r18_execute_routing')
    routing.adapt(adapter).validate_plan(helper.load(source_path),plan)
    context=module(frozen/'scripts/qwen-r18-decode.py','r18_execute_decode').validate(replay['decode_context']['directory'])
    helper.require(context==replay['decode_context'],'actual decode context differs from selected source protocol')
    helper.require(not a.log.exists() and not a.result.exists(),'preserve existing execution evidence')
    start=time.monotonic()
    with a.log.open('x') as log:
        try:
            process=subprocess.run(receipt['command'],stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout,check=False)
            result=dict(returncode=process.returncode,timeout=False)
        except subprocess.TimeoutExpired:result=dict(returncode=None,timeout=True)
    result.update(elapsed_seconds=time.monotonic()-start,replay_id=replay['replay_id'],
                  program_graph_id=replay['program_graph_id'],receipt_sha256=a.receipt_sha256)
    with a.result.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
    raise SystemExit(result['returncode'] if result['returncode'] is not None else 124)


if __name__=='__main__':main()
