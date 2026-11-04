#!/usr/bin/env python3
"""Reject malformed compiler manifests before compilation/device submission."""
import argparse, copy, importlib.util, json, sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'scripts'))
spec=importlib.util.spec_from_file_location('program_executor',root/'scripts/taskflow_direct_air_program.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
original=json.loads(a.manifest.read_text());directory=a.manifest.resolve().parent
module.validate(original,directory);records=[dict(case='unmodified-native-manifest',status='accepted')]
def reject(name,change):
 value=copy.deepcopy(original);change(value)
 try:module.validate(value,directory)
 except ValueError as error:records.append(dict(case=name,status='rejected',reason=str(error)))
 else:raise AssertionError('malformed manifest accepted: '+name)
def output_binding(value):
 return next(b for b in value['invocations'][0]['bindings'] if b['mode']=='slice-output')
reject('unsupported-version',lambda v:v.update(schema='orbit.program.partition.v2'))
reject('empty-schedule',lambda v:v.update(invocations=[]))
reject('missing-output-binding',lambda v:v['invocations'][0]['bindings'].remove(output_binding(v)))
reject('duplicate-output-binding',lambda v:v['invocations'][0]['bindings'].append(copy.deepcopy(output_binding(v))))
reject('missing-output-partition',lambda v:v['invocations'].pop())
def duplicate_output(value):
 invocation=copy.deepcopy(value['invocations'][0]);invocation['index']=max(i['index'] for i in value['invocations'])+1
 value['invocations'].append(invocation)
reject('duplicate-output-partition',duplicate_output)
reject('unsatisfied-dependency',lambda v:v['invocations'][0].update(depends_on=[999999]))
reject('duplicate-invocation-index',lambda v:v['invocations'][1].update(index=v['invocations'][0]['index']))
reject('unknown-template',lambda v:v['invocations'][0].update(template_id='missing-template'))
reject('negative-invocation-range',lambda v:v['invocations'][0].update(lower=-1))
reject('wrong-template-extent',lambda v:v['invocations'][0].update(upper=v['invocations'][0]['upper']-1))
reject('negative-full-buffer-index',lambda v:v['invocations'][0]['bindings'][0].update(full_buffer_index=-1))
reject('out-of-range-full-buffer-index',lambda v:v['invocations'][0]['bindings'][0].update(full_buffer_index=len(v['buffers'])))
reject('wrong-binding-shape',lambda v:v['invocations'][0]['bindings'][0]['sizes'].__setitem__(0,1))
reject('unknown-binding-mode-version',lambda v:v['invocations'][0]['bindings'][0].update(mode='slice-input-v2'))
reject('negative-binding-offset',lambda v:v['invocations'][0]['bindings'][0]['offsets'].__setitem__(0,-1))
reject('dtype-changed-to-bf16',lambda v:v['buffers'][0].update(dtype='bf16'))
reject('wrong-buffer-name',lambda v:v['buffers'][0].update(name='wrong-source-buffer'))
reject('duplicate-template-id',lambda v:v['templates'].append(copy.deepcopy(v['templates'][0])))
reject('template-mapping-plan-id-mismatch',lambda v:v['templates'][0].update(mapping_plan_id='wrong-plan-id'))
reject('parent-template-plan-id-mismatch',lambda v:v['template_plans'][0]['plan'].update(plan_id='wrong-parent-plan-id'))
reject('parent-template-plan-body-mismatch',lambda v:v['template_plans'][0]['plan'].update(capacity_bytes=1))
reject('missing-parent-template-plan',lambda v:v['template_plans'].pop())
reject('duplicate-parent-template-plan',lambda v:v['template_plans'].append(copy.deepcopy(v['template_plans'][0])))
reject('wrong-parent-template-invocation-count',lambda v:v['template_plans'][0].update(count=999))
a.output.write_text(json.dumps(dict(status='passed',device_submissions=0,records=records),indent=2)+'\n');print(json.dumps(dict(status='passed',checks=len(records))))
