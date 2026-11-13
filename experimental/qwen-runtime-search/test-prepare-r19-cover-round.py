#!/usr/bin/env python3
"""Small checks for actual ready/preparation command and registration assembly."""
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace


def main():
    path=Path(__file__).with_name('prepare-r19-cover-round.py')
    spec=importlib.util.spec_from_file_location('frontier_round_fixture',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    runtime=dict(preparation_entrypoint=dict(file='/sealed/prep2.py'),runnerfile=dict(file='/sealed/ready.py'),
        prefill_source_manifest=dict(file='/canonical/source.json',requested_file='/actual/source-link.json'),
        frozen_r18_runtime=dict(directory='/sealed/r18'))
    old=['/venv/python','/sealed/ready.py','--aie-root','/aie','--source','old-source','--ready-spec','old-ready',
         '--plan','old-plan','--output','old-output','--outputs','128','--warmups','1','--repeats','3',
         '--decode-variant-dir','old-context']
    command=m.ready_command(old,runtime,'new-ready','new-plan','new-output','actual-incumbent-context')
    assert command[:4]==['/venv/python','/sealed/prep2.py','--runtime-runner','/sealed/ready.py']
    assert command[command.index('--source')+1]=='/actual/source-link.json'
    assert command[command.index('--decode-variant-dir')+1]=='actual-incumbent-context'
    assert m.ready_command(command,runtime,'new-ready','new-plan','new-output','actual-incumbent-context')==command
    bad=old+['--swap-limit-gib','5']
    try:m.ready_command(bad,runtime,'r','p','o','d')
    except ValueError:pass
    else:raise AssertionError('Relaxed model memory guard silently propagated')
    digest=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    original=dict(id='original',operators=['a','b'],runtime_config=dict(backend='eager-host-group',input_policy='borrow-host',output_policy='retain-host'))
    duplicate=dict(original,id='different-id')
    staged=dict(id='stage',operators=['a','b'],runtime_config=dict(backend='eager-host-group',input_policy='stage-host-copy',output_policy='retain-host'))
    result=m.merge_cpu_registration(dict(implementations=[original]),dict(implementations=[duplicate,staged]),digest)
    assert [row['id'] for row in result['implementations']]==['original','stage']
    # Actual successful receipts retain their original execution manifests;
    # unrelated failures cannot become retained price backings.
    def require(condition,message):
        if not condition:raise ValueError(message)
    with TemporaryDirectory() as temporary:
        root=Path(temporary)
        (root/'manifest.json').write_text('{}')
        record=lambda path:dict(file=str(path),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
        old=dict(name='retained',purpose='training')
        fresh=dict(name='fresh',purpose='training')
        receipt=dict(name='fresh',manifest_sha256=record(root/'manifest.json')['sha256'],
                     returncode=0,admission_returncode=0,sealed_dependencies_unchanged=True,included_in_prices=True)
        (root/'summary.json').write_text(json.dumps([receipt]))
        fit=SimpleNamespace(require=require,record=record,load=lambda path:json.loads(Path(path).read_text()))
        prior=dict(cohort=old,execution_receipt=dict(name='retained'),marker='original-backing')
        parent=dict(cohorts=[old,fresh],retained_receipt_backing=[prior])
        audited=dict(retained=dict(actual_receipt=prior['execution_receipt']),fresh=dict(actual_receipt=receipt))
        joined=m.retain_receipt_backings(parent,audited,[root],fit)
        assert joined[0]==prior and joined[1]['execution_receipt']==receipt
        assert joined[1]['execution_manifest']==record(root/'manifest.json')
        assert joined[1]['execution_receipts']==record(root/'summary.json')
        for corrupt in [dict(retained=audited['retained']),dict(audited,failed=dict(actual_receipt=dict(name='failed')))]:
            try:m.retain_receipt_backings(parent,corrupt,[root],fit)
            except ValueError:pass
            else:raise AssertionError('Failed/unexecuted receipt promoted into retained prices')
        (root/'summary.json').write_text('[]')
        try:m.retain_receipt_backings(parent,audited,[root],fit)
        except ValueError:pass
        else:raise AssertionError('Receipt absent from actual summary admitted')
    print('PASS: actual prep2 command, requested source alias, selected decode context, unchanged guards, semantic registration dedup, strict original/fresh receipt joins and failed/missing rejection')


if __name__=='__main__':main()
