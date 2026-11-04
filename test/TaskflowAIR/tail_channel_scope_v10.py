#!/usr/bin/env python3
"""Compiler-only complete-module channel conversion, including ordinary stages.

No installed package modifications, Python IR rewrites, compilation or device
execution occur. Standard air-opt and native checkers consume generated AIR.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import resource
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--tool', required=True, type=Path)
p.add_argument('--air-opt', required=True, type=Path)
p.add_argument('--output', required=True, type=Path)
p.add_argument('--old-compiled-root', type=Path, help='optional immutable package: compare ordinary AIR bodies and separately report plan-identity changes')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
root = Path(__file__).resolve().parents[2]
text = (root/'test/TaskflowAIR/fixtures/gemm-scale.mlir').read_text()
shapes = {'4x8':'17x129','8x4':'129x80','4x4':'17x80'}
text = re.sub(r'(?<=memref<)\d+x\d+(?=xf32)', lambda m:shapes[m[0]], text)
text = re.sub(r'affine.for %(i|j|k) = 0 to \d+',
    lambda m:f'affine.for %{m[1]} = 0 to {dict(i=17,j=80,k=129)[m[1]]}', text)
checks = []

def run(directory, name, command, ok=True):
    command = list(map(str, command))
    (directory/(name+'.command.json')).write_text(json.dumps(command,indent=2)+'\n')
    def no_core_dump(): resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    result = subprocess.run(command,capture_output=True,text=True,preexec_fn=no_core_dump)
    (directory/(name+'.stdout')).write_text(result.stdout)
    (directory/(name+'.stderr')).write_text(result.stderr)
    assert (result.returncode==0)==ok,(name,result.returncode,result.stderr[-4000:])

for kind in ['scalar','scalar-product-vector-add-f32']:
    directory = a.output/kind;directory.mkdir()
    source = directory/'source.mlir';source.write_text(text)
    canonical,converted,legal = [directory/name for name in ['canonical.air.mlir','converted.air.mlir','legal.air.mlir']]
    flags = ['--air-distribute-initialization','--tile-m=8','--tile-n=32','--tile-k=32',
        '--implementation='+kind,'--vector-lanes='+('1' if kind=='scalar' else '8')]
    base = [a.tool,source,*flags]
    run(directory,'lower',[*base,'--lower-to-air',f'--air-output={canonical}',f'--mapping-output={directory}/plan.json'])
    plan = json.loads((directory/'plan.json').read_text())
    stages = plan['hardware_stages']
    assert any(s['legalize_channels'] for s in stages) and any(not s['legalize_channels'] for s in stages)
    air = canonical.read_text()
    assert 'orbit.channel_conversion_scope = "module"' in air
    assert 'orbit.async_herd_scope = "module"' in air
    assert 'orbit.local_release_scope = "static-hardware-stage"' in air
    assert len(re.findall(r'%[\w.]+ = "air.herd"',air))==air.count('"air.herd"')==len(stages)
    assert '"memref.dealloc"' not in air
    run(directory,'native-canonical-check',[*base,f'--check-air={canonical}'])
    run(directory,'standard-conversion',[a.air_opt,canonical,'--air-dma-to-channel','--canonicalize','--mlir-print-op-generic','-o',converted])
    assert '"scf.parallel"' not in converted.read_text()
    run(directory,'native-channel-legalize',[*base,f'--legalize-air-channels={converted}',f'--air-output={legal}'])
    run(directory,'native-channel-check',[*base,f'--check-air-channels={legal}'])
    run(directory,'installed-verify',[a.air_opt,legal,'-o',directory/'verified.air.mlir'])
    mutant = directory/'wrong-scope.air.mlir'
    mutant.write_text(air.replace('orbit.async_herd_scope = "module"','orbit.async_herd_scope = "stage"',1))
    run(directory,'scope-mutation-rejected',[*base,f'--check-air={mutant}'],False)
    checks.append(dict(kind=kind,stages=len(stages),plan_id=plan['plan_id'],air_sha256=hashlib.sha256(canonical.read_bytes()).hexdigest()))

if a.old_compiled_root:
    for name in ['train-m32-k128-n64-t8x16x32-scalar-phase','train-m32-k128-n64-t8x16x32-mixed8-panel']:
        old = a.old_compiled_root/name
        directory = a.output/('bridge-'+name);directory.mkdir()
        invocation = json.loads((old/'invocation.json').read_text())
        flags = invocation['flags']
        new_air = directory/'program.air.mlir'
        run(directory,'regenerate',[a.tool,old/'source.mlir',*flags,'--lower-to-air',f'--air-output={new_air}',f'--mapping-output={directory}/plan.json'])
        prior = old/'program.air.mlir'
        exact = new_air.read_bytes()==prior.read_bytes()
        diff = directory/'complete-air.diff'
        diff.write_text(''.join(difflib.unified_diff(
            prior.read_text().splitlines(keepends=True),
            new_air.read_text().splitlines(keepends=True),
            fromfile=str(prior),tofile=str(new_air))))
        # The target architecture independently acquired an explicit host
        # resource. A changed plan ID is metadata, but must not be hidden as
        # byte equality or re-execution. Never feed normalized IR to tools.
        identity = r'orbit.mapping_plan = "[0-9a-f]{64}"'
        assert re.sub(identity,'orbit.mapping_plan = "PLAN_ID"',new_air.read_text()) == re.sub(identity,'orbit.mapping_plan = "PLAN_ID"',prior.read_text()), name+' ordinary divisible AIR body changed'
        checks.append(dict(ordinary_air_bridge=name,exact_air_bytes_equal=exact,
            only_plan_identity_may_differ=True,complete_air_diff=str(diff),
            prior_air_sha256=hashlib.sha256(prior.read_bytes()).hexdigest(),
            new_air_sha256=hashlib.sha256(new_air.read_bytes()).hexdigest(),
            scope='static comparison only; old ELF identity/execution not reassigned'))
(a.output/'summary.json').write_text(json.dumps(dict(passed=True,checks=checks,hardware_executed=False),indent=2)+'\n')
print(json.dumps(dict(passed=True,checks=len(checks),hardware_executed=False)))
