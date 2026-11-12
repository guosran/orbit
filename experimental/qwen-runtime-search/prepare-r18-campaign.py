#!/usr/bin/env python3
"""Create immutable full-Qwen training/pilot commands without device execution."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen-runtime',type=Path,required=True)
    p.add_argument('--gemm-root',type=Path,required=True)
    p.add_argument('--decode-root',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--aie-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    path=a.frozen_runtime/'scripts/qwen-r18-runtime.py'
    spec=importlib.util.spec_from_file_location('r18_campaign_adapter',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    source=json.loads(a.source.read_text())
    family=['m128n256k64','m256n256k64','m128n512k256','replica2-col0','replica2-col4']
    gate=['gate-up-m128n256k32','gate-up-replica2-col0','gate-up-replica2-col4']
    gemm_backend=a.gemm_root/'frozen/npu-gemm-variants.py'
    contexts=[a.decode_root/f'decode-r0-cache{cache}-coalesce{coalesce}' for cache in (0,1) for coalesce in (0,1)]
    for ctx in contexts:
        if not (ctx/'variant.json').is_file():raise ValueError('Missing admitted rot0 context')
    def plan(name,mode='matmul',variant=None,layers=(),retention='retain',order='gate-up',only_layer=None):
        choices={op['id']:'npu' for op in source['operators'] if 'npu' in op['attributes'].get('supported_devices',[]) and
                 (mode=='all' or mode=='matmul' and op['kind']=='matmul')}
        variants={}
        if variant is not None:
            for layer in range(36):
                if only_layer is not None and layer!=only_layer:continue
                for role in ('qkv','o','gate_up','down'):
                    if layer in layers and role in ('gate_up','down'):continue
                    leaf=gate[variant%3] if role=='gate_up' else ('' if role=='qkv' else role+'-')+family[variant]
                    directory=(a.gemm_root/leaf).resolve(strict=True)
                    variants[f'prefill.L{layer}.{role}']=dict(directory=str(directory),manifest_sha256=sha(directory/'implementation.json'))
        for layer in layers:
            for role in ('gate_up','swiglu','down'):choices[f'prefill.L{layer}.{role}']='npu'
        result=m.node.make_plan(source,choices)
        manifest=a.frozen_runtime/f'compiled-{order}/manifest.json'
        result['r18']=dict(fused_layers=list(layers),rows=128,block_m=128,block_n=256,
            retention=retention,split_order=order,gemm_variants=variants,
            gemm_backend=dict(file=str(gemm_backend),sha256=sha(gemm_backend)) if variants else {},
            gemm_context_policy='close-after-call',
            ffn_artifact=dict(manifest=str(manifest),sha256=sha(manifest)) if layers else {})
        result['launch_policy']='explicit source-bound r18 GEMM/fusion implementations; strict actual compiled artifacts'
        result['plan_id']=m.node.digest({k:v for k,v in result.items() if k!='plan_id'})
        m.validate_plan(source,result)
        target=a.output/(name+'.plan.json');target.write_text(json.dumps(result,indent=2)+'\n')
        return target,result['plan_id']
    cohorts=[]
    definitions=[('cpu-original',dict(mode='cpu')),('npu-original',dict(mode='all'))]
    definitions += [('gemm-family'+str(i),dict(variant=i)) for i in range(5)]
    definitions += [('ffn-'+retention+'-'+order,dict(layers=range(36),retention=retention,order=order)) for retention in ('retain','reload','release-after-call') for order in ('gate-up','up-gate')]
    def command(plan_path,name,outputs,warmups,repeats,context):
        return ['python',str(a.frozen_runtime/'scripts/run-qwen-r18-runtime.py'),'--aie-root',str(a.aie_root),
            '--source',str(a.source),'--plan',str(plan_path),'--output',str(a.output/name),
            '--decode-variant-dir',str(context),'--outputs',str(outputs),'--warmups',str(warmups),'--repeats',str(repeats)]
    for i,(name,kwargs) in enumerate(definitions):
        plan_path,identity=plan(name,**kwargs);context=contexts[i%4] if i<8 else contexts[3]
        cohorts.append(dict(name=name,plan_id=identity,plan=str(plan_path),plan_sha256=sha(plan_path),
            decode_context=str(context),decode_manifest_sha256=sha(context/'variant.json'),
            command=command(plan_path,'train-'+name,128,1,3,context)))
    pilots=[]
    for name,kwargs in [('mixed',dict(variant=4,only_layer=0,layers=range(18,36),order='up-gate')),
                        ('gemm',dict(variant=2)),('ffn',dict(layers=range(36)))]:
        path,identity=plan('pilot-'+name,**kwargs)
        pilots.append(dict(name=name,plan_id=identity,command=command(path,'pilot-'+name,4,0,1,contexts[3])))
    result=dict(schema='orbit.qwen-r18-campaign.v1',source_id=source['source_id'],
        source_file=str(a.source),source_sha256=sha(a.source),frozen_runtime=str(a.frozen_runtime),
        freeze_sha256=sha(a.frozen_runtime/'freeze.json'),producer_sha256=sha(__file__),
        measured_prices='only these full-source inclusive hook/call records; never r17 or standalone GEMM timings',
        cohorts=cohorts,pilots=pilots,admission='pilot gates precede formal runs; all128-token original numerical gates required')
    (a.output/'campaign.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(cohorts=len(cohorts),pilots=len(pilots),campaign=str(a.output/'campaign.json'))))


if __name__=='__main__':main()
