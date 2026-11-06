#!/usr/bin/env python3
"""Target binding and execution-region compiler checks; never submits NPU work."""
import argparse
import copy
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser()
p.add_argument('--tool',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--air',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
source=Path(__file__).resolve().parents[2]/'test/TaskflowAIR/fixtures/gemm-scale.mlir'
checks=[]
def run(name,config,flags=(),ok=True):
    cfg=a.output/(name+'.config.json');cfg.write_text(json.dumps(config,indent=2)+'\n')
    plan=a.output/(name+'.plan.json');air=a.output/(name+'.air.mlir')
    cmd=[str(a.tool),str(source),'--generic-map','--air-distribute-initialization','--tile-m=2','--tile-n=2','--tile-k=4',f'--target-config={cfg}',f'--mapping-output={plan}',*flags]
    if a.air:cmd+=['--lower-to-air',f'--air-output={air}']
    r=subprocess.run(cmd,capture_output=True,text=True)
    (a.output/(name+'.command.json')).write_text(json.dumps(cmd)+'\n')
    (a.output/(name+'.stdout')).write_text(r.stdout);(a.output/(name+'.stderr')).write_text(r.stderr)
    assert (r.returncode==0)==ok,(name,r.returncode,r.stderr)
    checks.append(name)
    if ok and a.air:
        c=subprocess.run([*cmd[:2],*cmd[3:cmd.index('--lower-to-air')],f'--check-air={air}'],capture_output=True,text=True)
        (a.output/(name+'.check.stderr')).write_text(c.stderr)
        assert not c.returncode,(name,c.stderr)
    return json.loads(plan.read_text()) if ok else None
base=run('baseline',{})
target=base['target_memory'];arch=copy.deepcopy(target['architecture']);binding=copy.deepcopy(target['binding'])
# AIR spaces are backend facts, excluded from the configurable identifier map.
binding.pop('host_air_memory_space');binding.pop('local_air_memory_space')
names={'system_memory_bo':'system.dram','aie_l1':'tile.sram','npu.compute':'core0','npu.dma':'dma0','host_to_l1':'import0','l1_to_host':'export0'}
for group in ('memories','resources','links'):
    for obj in arch[group]:
        for key in ('id','source','target','resource'):
            if obj.get(key) in names:obj[key]=names[obj[key]]
        if 'accessible_resources' in obj:obj['accessible_resources']=[names.get(x,x) for x in obj['accessible_resources']]
for key,val in binding.items():binding[key]=names.get(val,val)
cfg={'generic_air':{'memory_architecture':arch,'memory_binding':binding}}
renamed=run('target-bound',cfg)
assert base['actual_dma_bytes']==renamed['actual_dma_bytes'] and base['resident_bytes']==renamed['resident_bytes']
assert base['plan_id']!=renamed['plan_id']
assert {x['memory'] for x in renamed['program_communication']['allocations']}=={'system.dram','tile.sram'}
region=renamed['program_communication']['execution_regions']
assert len(region)==1 and region[0]['host_submission_count']==1
assert len(region[0]['device_stages'])==len(renamed['hardware_stages'])
# Device labels do not override explicit instance ownership, access and links.
# A target may choose arbitrary device names, including these historical names.
for name,mutate in [
    ('cpu-compute-binding',lambda c:c['generic_air']['memory_architecture']['resources'][0].update(device='host')),
    ('cpu-dma-binding',lambda c:c['generic_air']['memory_architecture']['resources'][1].update(device='host')),
    ('device-submission-binding',lambda c:c['generic_air']['memory_architecture']['resources'][2].update(device='npu')),
]:
    changed=copy.deepcopy(cfg);mutate(changed);run(name,changed)
# These are legal common target descriptions but exceed the installed AIR
# implementation capability. Core preserves them; AIR must reject, not remap.
for name,mutate in [
    ('wrong-local-kind',lambda c:c['generic_air']['memory_architecture']['memories'][1].update(kind='dram')),
    ('wrong-host-kind',lambda c:c['generic_air']['memory_architecture']['memories'][0].update(kind='local-sram')),
    ('wrong-local-owner',lambda c:c['generic_air']['memory_architecture']['memories'][1].update(domain='system')),
    ('unsupported-physical-coordinate',lambda c:c['generic_air']['memory_architecture']['domains'][-1]['coordinates'].update(column=1)),
]:
    changed=copy.deepcopy(cfg);mutate(changed);run(name,changed,ok=not a.air)
for name,mutate in [
    ('unknown-local',lambda c:c['generic_air']['memory_binding'].update(local_memory='absent')),
    ('inaccessible',lambda c:c['generic_air']['memory_architecture']['memories'][1].update(accessible_resources=['absent'])),
    ('wrong-direction',lambda c:c['generic_air']['memory_architecture']['links'][0].update(target='system.dram')),
    ('link-size-limit',lambda c:c['generic_air']['memory_architecture']['links'][0].update(max_bytes=1)),
    ('extra-scratch',lambda c:c['generic_air']['memory_architecture']['links'][0].update(temporary_bytes=32)),
    ('capacity-increase',lambda c:c['generic_air'].update(capacity_bytes=32768)),
    ('malformed-capabilities',lambda c:c['generic_air'].update(capabilities=[])),
    ('malformed-lanes',lambda c:c['generic_air'].update(capabilities={'implementations':[{'kind':'scalar-product-vector-add-f32','lanes':'8'}]})),
    ('duplicate-implementation',lambda c:c['generic_air'].update(capabilities={'implementations':[{'kind':'scalar'},{'kind':'scalar'}]})),
]:
    bad=copy.deepcopy(cfg);mutate(bad);run(name,bad,ok=False)
only={'generic_air':{'capabilities':{'implementations':[{'kind':'scalar','lanes':1}]}}}
# Explicit CLI choice must not bypass a declared target implementation subset.
run('scalar-only-target',only,flags=['--implementation=scalar-product-vector-add-f32','--vector-lanes=8'],ok=False)
(a.output/'summary.json').write_text(json.dumps({'passed':True,'checks':checks,'hardware_executed':False,'air_checked':a.air},indent=2)+'\n')
print(json.dumps({'passed':True,'checks':len(checks),'hardware_executed':False}))
