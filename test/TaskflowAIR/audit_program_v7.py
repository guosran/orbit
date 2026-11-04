#!/usr/bin/env python3
"""Independently audit saved complete-program partition execution evidence."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-diagnostic',action='store_true');p.add_argument('--completed-seeds-only',action='store_true');a=p.parse_args()
if a.completed_seeds_only and not a.allow_diagnostic:p.error('partial seed selection requires --allow-diagnostic')
root=a.directory
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
manifest=json.loads((root/'manifest.json').read_text());mode=json.loads((root/'measurement-mode.json').read_text())
assert digest(root/'manifest.json')==mode['manifest_sha256']
assert manifest['schema']=='orbit.program.partition.v1'
assert len({t['id'] for t in manifest['templates']})==len(manifest['templates'])
assert len({t['template_id'] for t in manifest['template_plans']})==len(manifest['template_plans'])
assert {t['id'] for t in manifest['templates']}=={t['template_id'] for t in manifest['template_plans']}
seeds=sorted(d for d in root.glob('seed-*') if not a.completed_seeds_only or (d/'result.json').exists());assert seeds
if not a.allow_diagnostic:assert len(seeds)>=3 and not mode['diagnostic']
if mode['long_acceptance']:assert mode['hot_samples']==1 and mode['warmups']==0
else:
 if not a.allow_diagnostic:assert mode['hot_samples']>=3 and mode['warmups']>=2
for template in manifest['templates']:
 directory=root/template['id'];compiled=json.loads((directory/'compile.json').read_text());assert compiled['status']=='compiled'
 assert digest(directory/'input.air.mlir')==compiled['source_sha256']
 assert digest(directory/'program.elf')==compiled['elf_sha256']
 identity=json.loads((directory/'source-identity.json').read_text())
 assert digest(directory/'input.air.mlir')==identity['air']['sha256']
 for name in ['taskflow','plan','abi']:assert digest(directory/(name+Path(template[name]).suffix))==identity[name]['sha256']
 command=json.loads((directory/'command.json').read_text());assert Path(command[0]).name=='aircc' and '--output-format' in command
 storage=json.loads((directory/'storage-audit.json').read_text());assert storage['status']=='passed'
 native=json.loads((directory/'plan.json').read_text())
 assert template['mapping_plan_id']==native['plan_id']
 parents=[item for item in manifest['template_plans'] if item['template_id']==template['id']]
 assert len(parents)==1 and parents[0]['plan']==native
 assert parents[0]['count']==sum(i['template_id']==template['id'] for i in manifest['invocations'])
 runtime=json.loads((directory/'runtime-transfer-audit.json').read_text());assert runtime['status']=='passed'
 assert runtime['plan_sha256']==identity['plan']['sha256'] and runtime['abi_sha256']==identity['abi']['sha256']
 if mode['long_acceptance']:assert all(stage['expected_scope']=='per-stage-native-transfers' for stage in runtime['stages'])
checks=[];input_hashes=[]
for directory in seeds:
 provenance=json.loads((directory/'input-provenance.json').read_text())
 assert digest(directory/'inputs.npz')==provenance['inputs_sha256']
 assert digest(directory/'reference.npz')==provenance['reference_sha256'];input_hashes.append(provenance['inputs_sha256'])
 record=json.loads((directory/'result.json').read_text());assert record['status']=='passed' and record['atol']<=1e-5 and record['rtol']<=1e-5
 expected=dict(np.load(directory/'reference.npz'));samples=record['samples'];seen_templates=set()
 assert len(samples)==mode['hot_samples']+mode['warmups']
 for repeat,sample in enumerate(samples):
  assert sample['repeat']==repeat and sample['warmup']==(repeat<mode['warmups']) and sample['wall_ms']>0
  assert len(sample['invocations'])==len(manifest['invocations'])
  actual=dict(np.load(directory/f'outputs-{repeat}.npz'));assert set(actual)==set(expected)
  coverage={name:np.zeros(value.shape,dtype=np.uint8) for name,value in actual.items()}
  for planned,observed in zip(manifest['invocations'],sample['invocations']):
   assert observed['index']==planned['index'] and observed['template_id']==planned['template_id']
   events=observed['api_events'];starts=[e for e in events if e['kind']=='actual_xrt_start'];waits=[e for e in events if e['kind']=='actual_xrt_wait']
   assert len(starts)==len(waits)==observed['actual_submissions']==1
   assert all(e['end_ns']>=e['start_ns']>=0 for e in events)
   assert starts[0]['end_ns']<=waits[0]['start_ns']<=waits[0]['end_ns']
   first=planned['template_id'] not in seen_templates
   expected_uploads={b['buffer_index'] for b in planned['bindings'] if first or b['mode'] in ['slice-input','slice-output']}
   actual_uploads=[e['buffer_index'] for e in events if e['kind']=='actual_bo_sync' and 'TO_DEVICE' in e['direction']]
   assert sorted(actual_uploads)==sorted(expected_uploads)
   expected_readbacks={b['buffer_index'] for b in planned['bindings'] if b['mode']=='slice-output'}
   actual_readbacks=[e['buffer_index'] for e in events if e['kind']=='actual_bo_sync' and 'FROM_DEVICE' in e['direction']]
   assert sorted(actual_readbacks)==sorted(expected_readbacks)
   seen_templates.add(planned['template_id'])
   pieces=dict(np.load(directory/f'parts-{repeat}-{planned["index"]}.npz'))
   output_bindings=[b for b in planned['bindings'] if b['mode']=='slice-output']
   assert set(pieces)=={manifest['buffers'][b['full_buffer_index']]['name'] for b in output_bindings}
   for binding in output_bindings:
    name=manifest['buffers'][binding['full_buffer_index']]['name'];slices=tuple(slice(o,o+n) for o,n in zip(binding['offsets'],binding['sizes']))
    assert not np.any(coverage[name][slices]);coverage[name][slices]=1
    assert list(pieces[name].shape)==binding['sizes'] and pieces[name].dtype==np.float32
    assert np.array_equal(actual[name][slices],pieces[name])
  assert sample['actual_submissions']==len(manifest['invocations'])
  for name,x in actual.items():
   y=expected[name];assert np.all(coverage[name]==1) and x.dtype==np.float32 and x.shape==y.shape
   assert np.isfinite(x).all() and np.isfinite(y).all() and np.allclose(x,y,atol=1e-5,rtol=1e-5)
   checks.append(dict(seed=directory.name,repeat=repeat,output=name,full_shape=list(x.shape),max_abs=float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64)))),actual_submissions=sample['actual_submissions']))
assert len(set(input_hashes))==len(input_hashes)
a.output.write_text(json.dumps(dict(status='passed',scope='partial completed-seed diagnostic' if a.completed_seeds_only else 'single-sample diagnostic' if mode['diagnostic'] else 'three-seed one-complete-run long acceptance' if mode['long_acceptance'] else 'ordinary multi-seed warm/hot acceptance',seeds_checked=len(seeds),full_outputs_checked=len(checks),manifest_sha256=mode['manifest_sha256'],plan_id=manifest['plan_id'],checks=checks),indent=2)+'\n')
