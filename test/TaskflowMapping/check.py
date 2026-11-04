#!/usr/bin/env python3
"""Synthetic algorithm/independent-verifier tests; no hardware measurements."""
import argparse
import copy
import itertools
import json
from pathlib import Path
import sys
import hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
# The native regression harness injects the C++ tool adapters.
def map_program(*args):
    raise RuntimeError('Run cpp_differential.py with --tool and --output')
def check_plan(*args):
    raise RuntimeError('Run cpp_differential.py with --tool and --output')


PROVENANCE = {'kind':'synthetic','source':'constructed algorithm fixture, not hardware timing'}


def fixture():
    program = dict(schema='orbit.taskflow.plan.v1',candidate_id='synthetic-chain',source_id='synthetic-chain',
        buffers=[dict(id=b,shape=[2,2],dtype='f32',role=r) for b,r in [('X','input'),('A','temporary'),('Y','output')]],
        tasks=[dict(id='a',depends_on=[],operations=[dict(kind='relu',inputs=['X'],output='A',m=[0,2],n=[0,2])]),
               dict(id='b',depends_on=['a'],operations=[dict(kind='relu',inputs=['A'],output='Y',m=[0,2],n=[0,2])])],outputs=['Y'],transforms=[])
    config = dict(schema='orbit.static.config.v1',hardware=dict(
        resources=[dict(id=d,device=d,kind='compute',capacity=1) for d in ('cpu','npu')]+[dict(id='dma',kind='communication',capacity=1)],
        memories=[dict(id=d,capacity_bytes=4096,accessible_devices=[d]) for d in ('cpu','npu')],
        links=[dict(id=a+'-'+b,source=a,target=b,resource='dma',latency_ms=10,bandwidth_bytes_per_ms=1000,min_bytes=0,max_bytes=4096,provenance=PROVENANCE) for a,b in [('cpu','npu'),('npu','cpu')]],conversions=[],synchronizations=[]),
        implementations=[dict(id=d,version='fixture-v1',device=d,operations=['relu','fill','matmul','swiglu'],memory=d,layout='row_major',dtype='f32',numerical_contract='fp32',resources={d:1},temporary_bytes=0,cost=dict(kind='affine',fixed_ms=cost,per_element_ms=0,per_flop_ms=0,scope='compute_only',shape_range={a:[1,100000] for a in ('m','n','k')},provenance=PROVENANCE)) for d,cost in [('cpu',2),('npu',1)]],
        assumptions=dict(initial_locations={'X':'cpu'},output_locations={'Y':'cpu'},allowed_numerical_contracts=['fp32'],overlap='explicit',executions=1,max_assignments=256,beam_width=16))
    return program, config


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    checks=[]; negatives=[]
    def run(name,p,c):
        result=map_program(p,c)
        folder=root/name;folder.mkdir(exist_ok=True)
        for filename,obj in [('program.json',p),('config.json',c),('mapping.json',result)]:
            (folder/filename).write_text(json.dumps(obj,indent=2)+'\n')
        return result
    def require(test,label):
        assert test,label
        checks.append(label)
    p,c=fixture();r=run('communication-reversal',p,c)
    require(r['status']=='passed' and set(r['best_plan']['assignment'].values())=={'cpu'},'communication reverses isolated fastest implementation')
    # Independent closed-form exhaustive oracle for this two-op serial chain.
    oracle={}
    for a,b in itertools.product(('cpu','npu'),repeat=2):
        copies=int(a!='cpu')+int(a!=b)+int(b!='cpu')
        oracle[(a,b)]=(2 if a=='cpu' else 1)+(2 if b=='cpu' else 1)+copies*10.016
    observed={(row['assignment']['a:0'],row['assignment']['b:0']):row['score'] for row in r['alternatives']}
    require(set(observed)==set(oracle) and all(abs(observed[k]-v)<1e-8 for k,v in oracle.items()),'all four complete assignments match independent closed-form exhaustive oracle')
    c0=copy.deepcopy(c)
    for link in c0['hardware']['links']:link.update(latency_ms=0,bandwidth_bytes_per_ms=1e12)
    r0=run('communication-near-zero-sensitivity',p,c0)
    require(set(r0['best_plan']['assignment'].values())=={'npu'},'explicit synthetic link-cost sensitivity changes best mapping')
    cpu=copy.deepcopy(c);cpu['implementations']=cpu['implementations'][:1]
    rr=run('same-location-reuse',p,cpu)
    require(not any(e['kind'] in ('copy','convert','sync') for e in rr['best_plan']['events']),'same location/layout reuse has zero copies')

    # One initial region is consumed twice on NPU: upload once, two outputs
    # return separately. Full allocations remain counted at both locations.
    multi=copy.deepcopy(p);multi['buffers'][1]['role']='output';multi['tasks'][1]['operations'][0]['inputs']=['X'];multi['tasks'][1]['depends_on']=[];multi['outputs']=['A','Y']
    mc=copy.deepcopy(c);mc['implementations']=mc['implementations'][1:];mc['assumptions']['output_locations']={'A':'cpu','Y':'cpu'}
    rm=run('multiple-consumers',multi,mc);mp=rm['best_plan']
    uploads=[e for e in mp['events'] if e['kind']=='copy' and e['link']=='cpu-npu']
    require(len(uploads)==1 and uploads[0]['bytes']==16,'multiple consumers share exactly one valid input copy')
    require(mp['totals']['peak_memory_bytes']=={'cpu':48,'npu':48},'multiple consumers and distinct output copies all counted in memory')
    # Explicit control edge between independent writes carries no data.
    order=copy.deepcopy(p)
    for t in order['tasks']:t['operations'][0].update(kind='fill',inputs=[])
    ro=run('ordering-only',order,cpu)
    require(all(e['kind']=='compute' for e in ro['best_plan']['events']),'pure ordering edge creates no fictitious transfer')
    # Fast plan is infeasible because its simultaneous physical copies exceed
    # capacity; the slower CPU assignment remains feasible.
    cap=copy.deepcopy(c0);cap['hardware']['memories'][1]['capacity_bytes']=16
    rc=run('capacity-rejects-fast',p,cap)
    require(rc['status']=='passed' and set(rc['best_plan']['assignment'].values())=={'cpu'} and any(x['status']=='infeasible' for x in rc['alternatives']),'peak capacity rejects otherwise faster mapping')
    require(all(not(a['start_ms']<b['end_ms'] and b['start_ms']<a['end_ms']) for a,b in itertools.combinations([e for e in mp['events'] if e['kind']=='copy'],2)),'shared DMA resource serializes transfers')
    require(all(not(a['start_ms']<b['end_ms'] and b['start_ms']<a['end_ms']) for a,b in itertools.combinations([e for e in mp['events'] if e['kind']=='compute'],2)),'shared compute resource serializes independent consumers')

    layout=copy.deepcopy(cpu);layout['implementations'][0].update(layout='blocked',temporary_bytes=64)
    layout['hardware']['conversions']=[dict(id=a+'-'+b,memory='cpu',source_layout=a,target_layout=b,source_dtype='f32',target_dtype='f32',resource='dma',latency_ms=.5,bandwidth_bytes_per_ms=16,provenance=PROVENANCE) for a,b in [('row_major','blocked'),('blocked','row_major')]]
    rl=run('layout-conversion',p,layout);lp=rl['best_plan']
    require(rl['status']=='passed' and abs(lp['totals']['convert_ms']-3)<1e-8,'layout conversions have explicit nonzero costs and dependencies')
    require(lp['totals']['peak_memory_bytes']['cpu']==144,'layout copies and implementation temporary included in peak')
    unknown=copy.deepcopy(cpu);del unknown['implementations'][0]['cost']['fixed_ms']
    ru=run('unknown-cost',p,unknown)
    require(ru['status']!='passed' and not ru['best_plan'],'missing cost is unscored, never zero')
    missing=copy.deepcopy(cpu);missing['implementations'][0]['operations']=['fill']
    ri=run('missing-implementation',p,missing)
    require(ri['status']!='passed' and not ri['best_plan'],'missing implementation is rejected')

    # Mutate independent plan facts, not merely its summary or signature.
    base=r['best_plan']
    def reject(name,plan,prog=p,cfg=c):
        # Re-sign the tampered evidence so identity checks alone cannot make
        # semantic/resource adversarial tests pass.
        def ident(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        plan=copy.deepcopy(plan);plan['program_id']=ident(prog);plan['configuration_id']=ident(cfg)
        plan['plan_id']=ident({k:v for k,v in plan.items() if k not in ('plan_id','validation')})
        verdict=check_plan(prog,cfg,plan)
        assert not verdict['passed'],name
        negatives.append(dict(name=name,errors=verdict['errors']))
    bad=copy.deepcopy(base);bad['events'].pop();reject('omitted compute',bad)
    bad=copy.deepcopy(base);bad['events'][1]['reads'][0]['version']='forged';reject('forged input version',bad)
    bad=copy.deepcopy(base);bad['events'][1]['reads'][0]['region']=[[0,1],[0,2]];reject('partial read coverage',bad)
    bad=copy.deepcopy(base);bad['events'][1]['depends_on']=[];reject('removed producer ordering',bad)
    bad=copy.deepcopy(base);bad['events'][1].update(start_ms=0,end_ms=2);bad['totals']['makespan_ms']=2;reject('compute overlap',bad)
    bad=copy.deepcopy(base);bad['allocations'][0]['bytes']=1;reject('forged physical allocation bytes',bad)
    bad=copy.deepcopy(base);bad['totals']['peak_memory_bytes']['cpu']=1;reject('forged peak',bad)
    bad=copy.deepcopy(base);bad['totals']['makespan_ms']=.01;reject('forged makespan',bad)
    bad=copy.deepcopy(base);bad['events'][0]['resource_demands']={};reject('omitted compute demand',bad)
    bad=copy.deepcopy(base);bad['events'][0]['device']='npu';reject('forged device',bad)
    bad=copy.deepcopy(base);bad['events'][0]['reads'][0]['allocation']=bad['events'][0]['writes'][0]['allocation'];reject('wrong physical storage',bad)
    bad=copy.deepcopy(mp);next(e for e in bad['events'] if e['kind']=='copy')['bytes']=0;reject('forged transfer byte count',bad,multi,mc)
    bad=copy.deepcopy(lp);bad['allocations']=[a for a in bad['allocations'] if a.get('lifetime')!='event'];reject('omitted implementation temporary',bad,p,layout)
    bad=copy.deepcopy(lp);next(e for e in bad['events'] if e['kind']=='convert')['duration_ms']=0;reject('free layout conversion',bad,p,layout)
    precision=copy.deepcopy(c);precision['implementations'][0]['numerical_contract']='bf16_bfp16_fp32';reject('unapproved precision',base,p,precision)
    bad=copy.deepcopy(base);bad['makespan_ms']=.001;reject('forged top-level scorer makespan',bad)
    bad=copy.deepcopy(base);bad['assignment']['a:0']='npu';reject('forged assignment metadata',bad)
    bad=copy.deepcopy(base);bad['allocations'][0]['end_ms']=.1;reject('forged early allocation release',bad)
    bad=copy.deepcopy(base);bad['events'][0]['provenance']={'kind':'measured','source':'invented'};reject('forged measured provenance',bad)
    wrong_resource=copy.deepcopy(c);wrong_resource['hardware']['resources'][0]['device']='npu';reject('compute resource belongs to another device',base,p,wrong_resource)
    fractional=copy.deepcopy(c);fractional['implementations'][0]['resources']['cpu']=.5;reject('fractional compute resource units',base,p,fractional)
    missing_path=copy.deepcopy(mc);missing_path['hardware']['links']=[]
    rp=run('missing-path',multi,missing_path)
    require(rp['status']!='passed' and not rp['best_plan'],'missing communication path stays unscored')
    zero=copy.deepcopy(cpu);zero['implementations'][0]['cost']['fixed_ms']=0;zero['hardware']['memories'][0]['capacity_bytes']=47
    rz=run('zero-duration-capacity',p,zero)
    require(rz['status']!='passed','zero-duration operations still require physical graph storage')
    scratch=copy.deepcopy(layout)
    for conv in scratch['hardware']['conversions']:conv['temporary_bytes']=128
    rs=run('conversion-scratch',p,scratch)
    require(rs['status']=='passed' and rs['best_plan']['totals']['peak_memory_bytes']['cpu']==208,'conversion-specific scratch has explicit lifetime and capacity')
    bad=copy.deepcopy(rs['best_plan']);conv_ids={e['id'] for e in bad['events'] if e['kind']=='convert'};bad['allocations']=[a for a in bad['allocations'] if a.get('created_by') not in conv_ids or a.get('lifetime')!='event'];reject('omitted conversion scratch',bad,p,scratch)

    sharedp=copy.deepcopy(p);sharedp['tasks'][0]['operations'][0].update(kind='fill',inputs=[])
    shared=copy.deepcopy(c);shared['hardware']['memories']=[dict(id='shared',capacity_bytes=4096,accessible_devices=['cpu','npu'])];shared['hardware']['links']=[]
    shared['hardware']['synchronizations']=[dict(id=a+'-'+b,source_device=a,target_device=b,memory='shared',resource='dma',latency_ms=.25,provenance=PROVENANCE) for a,b in [('cpu','npu'),('npu','cpu')]]
    for impl in shared['implementations']:impl['memory']='shared';impl['operations']=['fill'] if impl['device']=='cpu' else ['relu']
    shared['assumptions'].update(initial_locations={'X':'shared'},output_locations={'Y':'shared'},output_devices={'Y':'cpu'})
    rsh=run('shared-memory-sync',sharedp,shared);sp=rsh['best_plan']
    require(rsh['status']=='passed' and sp['totals']['transfer_bytes']==0 and sp['totals']['sync_ms']==.5,'shared memory needs producer and final-consumer synchronization but no copy')
    bad=copy.deepcopy(sp);bad['events']=[e for e in bad['events'] if e['id'] not in bad['output_dependencies']];reject('missing final consumer sync',bad,sharedp,shared)
    bad=copy.deepcopy(sp);next(e for e in bad['events'] if e['kind']=='sync')['reads'][0]['version']='initial:A';reject('stale sync cannot approve new version',bad,sharedp,shared)
    nosync=copy.deepcopy(shared);nosync['hardware']['synchronizations']=[]
    rns=run('unknown-sync',sharedp,nosync);require(rns['status']!='passed','unknown shared-memory synchronization is not free')

    # Sequential K has one fill and two true accumulation operations.
    kp=copy.deepcopy(p);kp['buffers']=[dict(id=b,shape=[2,2],dtype='f32',role=r) for b,r in [('X','input'),('W','input'),('Y','output')]]
    kp['tasks']=[dict(id='k',depends_on=[],operations=[dict(kind='fill',inputs=[],output='Y',m=[0,2],n=[0,2])]+[dict(kind='matmul',inputs=['X','W'],output='Y',m=[0,2],n=[0,2],k=[lo,lo+1],accumulate=True) for lo in (0,1)])]
    kc=copy.deepcopy(cpu);kc['assumptions']['initial_locations']['W']='cpu'
    rk=run('sequential-k',kp,kc);require(rk['status']=='passed','sequential K retains fill and all ordered accumulation steps')
    bad=copy.deepcopy(rk['best_plan']);bad['events'].pop(0);reject('omitted K initialization',bad,kp,kc)
    bad=copy.deepcopy(rk['best_plan']);bad['events'].pop(1);reject('omitted K partial',bad,kp,kc)
    bad=copy.deepcopy(rk['best_plan']);bad['events'][2]['reads']=[a for a in bad['events'][2]['reads'] if a['buffer']!='Y'];reject('omitted accumulator read',bad,kp,kc)
    kn=copy.deepcopy(kc);ni=copy.deepcopy(c['implementations'][1]);ni['legality']={'full_k_only':True,'overwrite_zero_accumulator':True};ni['operations']=['matmul'];kn['implementations'].append(ni)
    rkn=run('partial-k-npu-exclusion',kp,kn)
    require(all('npu' not in row['assignment'].values() for row in rkn['alternatives']),'unsupported partial K excluded from NPU legal choices')
    full=copy.deepcopy(kp);full['tasks'][0]['operations']=full['tasks'][0]['operations'][:2];full['tasks'][0]['operations'][1]['k']=[0,2]
    rf=run('full-k-zero-copy',full,kn)
    nplan=next(x['plan'] for x in rf['alternatives'] if x['status']=='passed' and x['assignment']['k:1']=='npu')
    require(any(e['kind']=='copy' and any(a['buffer']=='Y' and a['version']=='k:0' for a in e['reads']) for e in nplan['events']),'NPU overwrite implementation still covers zero fill and explicit accumulator preparation')
    forged=copy.deepcopy(kn);forged['implementations'][0]['legality']={'full_k_only':True};reject('forged full-K legality on partial K',rk['best_plan'],kp,forged)
    result=dict(passed=True,evidence_kind='synthetic algorithm tests; not hardware performance',checks=checks,negative_checks=negatives)
    (root/'checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,positive_checks=len(checks),negative_checks=len(negatives))))


if __name__=='__main__':
    from cpp_differential import main as native_main
    if '--tool' not in sys.argv:
        sys.argv += ['--tool', str(Path(__file__).resolve().parents[2]/'build-core/tools/taskflow-opt/taskflow-opt')]
    native_main()
