#!/usr/bin/env python3
"""Native integration, checked incumbents and raw-work/time budget contracts."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]


def chain(tasks,m=4,n=16):
    ty=f'memref<{m}x{n}xf32>'
    lines=['module {',f'func.func @chain(%A: {ty} {{orbit.role="input",orbit.name="A"}}, %O: {ty} {{orbit.role="output",orbit.name="O"}}) attributes {{orbit.noalias}} {{']
    lines += [f'%v{i}=memref.alloc() {{orbit.name="v{i}"}} : {ty}' for i in range(tasks-1)]
    for i in range(tasks):
        src='%A' if i==0 else f'%done{i-1}'
        dst='%O' if i==tasks-1 else f'%v{i}'
        lines += [f'%done{i}=taskflow.task @task{i} will_reads({src} : {ty}) will_writes({dst} : {ty}) [original_read_memrefs({src} : {ty}),original_write_memrefs({dst} : {ty})] : ({ty},{ty})->{ty} {{',f'^bb0(%r: {ty},%w: {ty}):',f'affine.for %i=0 to {m} {{',f'affine.for %j=0 to {n} {{',f'%x=affine.load %r[%i,%j] : {ty}','%c=arith.constant 1.0 : f32','%y=arith.addf %x,%c : f32',f'affine.store %y,%w[%i,%j] : {ty}','}','}',f'taskflow.yield done_writes(%w : {ty})','}']
    lines += [f'memref.dealloc %v{i} : {ty}' for i in range(tasks-1)]
    return '\n'.join(lines+['return','}','}'])+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tool',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    base=json.loads((ROOT/'test/TaskflowGeneric/reuse-assumptions-v21.json').read_text())
    base['native_auto_candidates']=dict(mode='joint',algorithm='cp-sat',max_joint_tiles=4,
        max_scope_sets=16,max_catalogue_configurations=64,max_generation_attempts=256,
        max_generator_planner_calls=128,max_internal_planner_calls=512,max_cost_provider_calls=128,
        max_compute_instances=256,max_wall_time_ms=5000,exploration_slots=0)
    rows=[]
    cases=[(stem,(ROOT/'test/TaskflowAIR/fixtures'/f'{stem}-v20.mlir').read_text(),{})
        for stem in ('reuse-interleaved','reuse-interleaved-permuted','reuse-shared-input','reuse-shared-input-permuted')]
    cases += [('cost-cap',chain(4),dict(max_cost_provider_calls=3)),('wall-cap',chain(4),dict(max_wall_time_ms=.01))]
    for name,source,overrides in cases:
        folder=a.output/name;folder.mkdir();src=folder/'source.mlir';src.write_text(source)
        config=copy.deepcopy(base);config['native_auto_candidates'].update(overrides)
        cfg=folder/'config.json';cfg.write_text(json.dumps(config,indent=2)+'\n')
        search=folder/'search'
        command=[str(a.tool.resolve()),str(src.resolve()),'--search','--backend=air','--max-rounds=1',
            '--max-candidates=64','--beam-width=1','--tile-m=4','--tile-n=16','--tile-k=1',
            '--target-config='+str(cfg.resolve()),'--search-output='+str(search.resolve())]
        (folder/'command.json').write_text(json.dumps(command,indent=2)+'\n')
        start=time.monotonic();process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
        expired=False
        try:stdout,stderr=process.communicate(timeout=8)
        except subprocess.TimeoutExpired:
            expired=True;os.killpg(process.pid,signal.SIGKILL);stdout,stderr=process.communicate()
        elapsed=(time.monotonic()-start)*1000
        (folder/'stdout').write_text(stdout);(folder/'stderr').write_text(stderr)
        row=dict(name=name,elapsed_ms=elapsed,external_timeout=expired,returncode=process.returncode,
            source_sha256=hashlib.sha256(source.encode()).hexdigest(),hard_timeout_seconds=8)
        if not expired:
            assert process.returncode==0,(name,stderr)
            report=json.loads((search/'search.json').read_text());row['stats']=report['stats']
            proposals=json.loads((search/'automatic-proposals.json').read_text());row['proposals']=proposals
            passed=[r for r in report['archive'] if r['status']=='passed']
            roots=[r for r in passed if not r.get('parent')]
            if roots:
                assert min(r['score'] for r in passed)<=min(r['score'] for r in roots)
            for reports in proposals['parents'].values():
                for r in reports:
                    if r['status']=='CANDIDATE':assert r['rechecked_ns']<=r['best_local_ns'],r
            cap=config['native_auto_candidates']['max_cost_provider_calls']
            assert report['stats']['raw_cost_provider_calls']<=cap,(name,report['stats'])
            if not overrides:
                assert any('cp_sat_reuse_tile' in r['action_chain'] for r in passed),(name,report)
            if name=='wall-cap':assert report['stats']['raw_cost_provider_calls']==0
        else:raise AssertionError((name,'integration timed out'))
        rows.append(row)
        (a.output/'result.json').write_text(json.dumps(dict(status='PASS',cases=rows),indent=2)+'\n')
        print(json.dumps(dict(name=name,elapsed_ms=elapsed,external_timeout=expired)),flush=True)


if __name__=='__main__':main()
