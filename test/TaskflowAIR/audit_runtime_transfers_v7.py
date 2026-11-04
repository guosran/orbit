#!/usr/bin/env python3
"""Audit static final runtime DMA producer/consumer bytes against native transfers.

This reads final installed compiler IR. It does not rewrite scheduling or claim
measured device traffic. Only static, fully expanded runtime sequences are
accepted; unsupported dynamic forms fail explicitly.
"""
import argparse,collections,hashlib,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('hardware',type=Path);p.add_argument('--plan',type=Path,required=True);p.add_argument('--abi',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
source=a.hardware/'air_project/npu.input.air.mlir';text=source.read_text();plan=json.loads(a.plan.read_text());abi=json.loads(a.abi.read_text());rows=[];errors=[];aggregate_comparisons=[]
def region(start):
 opening=text.index('{',start);depth=1;end=opening+1
 while depth:
  if text[end]=='{':depth+=1
  elif text[end]=='}':depth-=1
  end+=1
 return text[opening+1:end-1],end
try:
 explicit=bool(plan.get('transfers'))
 per_stage_summary=all('transfer_summary' in stage for stage in plan['hardware_stages'])
 aggregate_only=not explicit and not per_stage_summary
 assert explicit or per_stage_summary or plan.get('transfer_summary'),'native checked transfer summaries are required'
 aggregate_actual=collections.Counter()
 buffers=abi['buffers'];width={'f32':4,'bf16':2,'i32':4}
 devices={m[1]:region(m.start())[0] for m in re.finditer(r'aie.device\(npu2\) @([\w.$]+)\s*\{',text)}
 for index,stage in enumerate(plan['hardware_stages']):
  name='orbit_segment' if index==0 else 'orbit_segment_'+str(stage['begin_phase'])
  assert name in devices,('missing hardware stage',name)
  device=devices[name]
  channels={m[1]:('host_to_l1' if m[2]=='MM2S' else 'l1_to_host') for m in re.finditer(r'aie.shim_dma_allocation @([\w.$]+)\([^\n]*?, (MM2S|S2MM), \d+\)',device)}
  match=re.search(r'aie.runtime_sequence @'+re.escape(name)+r'_sequence\(',text)
  assert match,('missing runtime sequence',name)
  runtime,end=region(match.start())
  assert 'scf.for' not in runtime and 'scf.parallel' not in runtime,'unresolved runtime repetition requires an extended auditor'
  expected=collections.Counter();actual=collections.Counter();tasks=[]
  for transfer in (plan['transfers'] if explicit else stage.get('transfer_summary',[])):
   if not explicit or stage['begin_phase']<=transfer['phase']<stage['end_phase_exclusive']:
    expected[(transfer['storage'],transfer['direction'])]+=transfer['bytes']
  sequence_start=text.index('{',match.start())+1
  for task in re.finditer(r'aiex.dma_configure_task_for @([\w.$]+)\s*\{',runtime):
   channel=task[1];body,finish=region(sequence_start+task.start())
   suffix=text[finish:text.find('\n',finish)]
   repeats=re.search(r'repeat_count = (\d+)',suffix);repeats=int(repeats[1])+1 if repeats else 1
   assert channel in channels,('unknown runtime DMA channel',channel)
   for bd in re.finditer(r'aie.dma_bd\(%arg(\d+) : memref<[^>]+> offset = \d+ len = (\d+)',body):
    argument=int(bd[1]);assert argument<len(buffers)
    buffer=buffers[argument];amount=int(bd[2])*repeats*width[buffer['dtype']]
    actual[(buffer['name'],channels[channel])]+=amount
    tasks.append(dict(argument=argument,storage=buffer['name'],direction=channels[channel],channel=channel,bd_elements=int(bd[2]),repetitions=repeats,bytes=amount))
  aggregate_actual.update(actual)
  keys=set(expected)|set(actual) if not aggregate_only else set()
  comparison=[dict(storage=k[0],direction=k[1],expected_bytes=expected[k],actual_runtime_bytes=actual[k],passed=expected[k]==actual[k]) for k in sorted(keys)]
  rows.append(dict(stage=index,device=name,tasks=tasks,comparisons=comparison,expected_scope='aggregate-native-summary-only' if aggregate_only else 'per-stage-native-transfers'))
  for item in comparison:
   if not item['passed']:errors.append(dict(stage=index,**item))
 if aggregate_only:
  expected=collections.Counter()
  for transfer in plan['transfer_summary']:expected[(transfer['storage'],transfer['direction'])]+=transfer['bytes']
  comparison=[dict(storage=k[0],direction=k[1],expected_bytes=expected[k],actual_runtime_bytes=aggregate_actual[k],passed=expected[k]==aggregate_actual[k]) for k in sorted(set(expected)|set(aggregate_actual))]
  aggregate_comparisons=comparison
  for item in comparison:
   if not item['passed']:errors.append(dict(stage='whole_program',**item))
except (AssertionError,ValueError,KeyError,IndexError) as error:
 errors.append(dict(auditor_error=str(error)))
result=dict(status='failed' if errors else 'passed',scope='Static final runtime BD lengths times task repeats, per ABI storage and transfer direction; per-stage checked transfer records when available, aggregate-only for older compact plans with only a global native summary; not measured traffic or proof of device progress.',source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),abi_sha256=hashlib.sha256(a.abi.read_bytes()).hexdigest(),audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),stages=rows,aggregate_comparisons=aggregate_comparisons,errors=errors)
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],stages=len(rows),errors=errors)))
raise SystemExit(bool(errors))
