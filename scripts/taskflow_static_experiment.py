#!/usr/bin/env python3
"""Static workload planning and original ORBIT search, without device execution."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from taskflow_common import source_plan
from taskflow_static_io import load, write, identity, digest

THEORY = dict(kind='theoretical', source='Explicit sensitivity assumptions, not fitted to whole-graph NPU measurements; no hardware performance claim')


def configuration(program, bandwidth=12_000_000):
    """Existing operation capability, deliberately hypothetical cost parameters.

NPU f32 interface denotes the existing BF16/BFP16 approximate kernel. Its
overwrite result is legal only for a full K range after a zero initializer.
CPU fill and activation remain explicit. Runtime does not execute this plan.
"""
    resources = [dict(id=x, capacity=1, kind='compute' if x in ('cpu', 'npu') else 'communication', device=x)
                 for x in ('cpu', 'npu', 'dma', 'visibility')]
    hardware = dict(resources=resources,
        memories=[dict(id='host', capacity_bytes=2**30, accessible_devices=['cpu']),
                  dict(id='shared', capacity_bytes=2**30, accessible_devices=['cpu', 'npu'])],
        links=[dict(id=a+'-to-'+b, source=a, target=b, resource='dma', latency_ms=.02,
                    bandwidth_bytes_per_ms=bandwidth, min_bytes=0, max_bytes=2**30, provenance=THEORY)
               for a,b in [('host','shared'),('shared','host')]],
        conversions=[], synchronizations=[dict(id=a+'-to-'+b+'-visibility', source_device=a,
            target_device=b, memory='shared', resource='visibility', latency_ms=.01, provenance=THEORY)
            for a,b in [('cpu','npu'),('npu','cpu')]])
    implementations=[]
    def impl(name, device, operations, memory, fixed, elements, flops, numerical, scratch=0, **extra):
        implementations.append(dict(id=name,version='static-catalogue-1', device=device,
            operations=operations,memory=memory,layout='row_major',dtype='f32',numerical_contract=numerical,
            resources={device:1},temporary_bytes=scratch,cost=dict(kind='affine',scope='compute_only',
                fixed_ms=fixed,per_element_ms=elements,per_flop_ms=flops,provenance=THEORY,
                shape_range=dict(m=[1,128],n=[1,22016],k=[1,11008])),**extra))
    impl('cpu-fp32-matmul','cpu',['matmul'],'host',.01,0,1/100_000_000,'fp32',
         capability_source='scripts/taskflow_common.py:execute_cpu; compiled MLIR CPU replay')
    impl('cpu-fill','cpu',['fill'],'host',.002,1/3_000_000,0,'fp32',capability_source='explicit CPU zero fill')
    impl('cpu-relu','cpu',['relu'],'host',.002,1/3_000_000,0,'fp32',capability_source='existing CPU ReLU')
    impl('cpu-swiglu','cpu',['swiglu'],'host',.005,1/300_000,0,'fp32',capability_source='existing CPU SiLU(gate)*up')
    impl('npu-full-k-zero','npu',['matmul'],'shared',.12,0,1/1_000_000_000,'bf16-bfp16-ebs8-fp32',scratch=512*2**20,
         legality=dict(full_k_only=True,overwrite_zero_accumulator=True),
         capability_source='python/orbit_npu/kernels.py existing GEMM primitive; full-K zero-initialized result only')
    return dict(schema='orbit.static.config.v1', hardware=hardware, implementations=implementations,
        assumptions=dict(program_source_id=program['source_id'],initial_locations={b['id']:'host' for b in program['buffers'] if b['role']=='input'},
            output_locations={b:'host' for b in program['outputs']}, output_devices={b:'cpu' for b in program['outputs']},
            allowed_numerical_contracts=['fp32','bf16-bfp16-ebs8-fp32'],
            numerical_acceptance=dict(atol=.015,rtol=.06,normalized_rms_max=.03,peak_scaled_max=.08,
                note='Explicit approximate implementation permission from existing NPU validation; static planning is not a new numerical execution check'),
            overlap='forbid', executions=1,max_assignments=64,beam_width=8),
        provenance=dict(costs='theoretical_sensitivity', runtime_execution='not_executed',
            previous_measurements='build/search-npu-v2/final-report.json is whole-graph evidence only; excluded from compute coefficients',
            memory_model='host and NPU-accessible shared system-memory allocation pools, not AIE SRAM',
            scratch_model='512 MiB assumed workspace reservation per active NPU operation for these source-bound workloads, including local conversion/padding copies. Not a proven upper bound or measured runtime allocation; physical capacity feasibility is not certified.',
            compute_boundary='kernel computation, internal precision conversion/padding and local loads/stores; separately charged pool copies/visibility; affine costs are assumptions'))


def materialize(source, dest, flags=()):
    dest.mkdir(parents=True, exist_ok=True)
    cmd=[str(ROOT/'build-core/tools/taskflow-opt/taskflow-opt'),str(source),*flags,
         '--emit-plan='+str(dest/'plan.json'),'--emit-mlir='+str(dest/'taskflow.mlir')]
    p=subprocess.run(cmd,capture_output=True,text=True)
    (dest/'materialize.log').write_text(p.stdout+p.stderr)
    write(dest/'materialize-command.json',cmd)
    if p.returncode: raise RuntimeError(p.stderr)
    return load(dest/'plan.json')


def prepare(out):
    cases=[('gemm_relu',source_plan('gemm_relu')),('ffn',source_plan('ffn')),
           *[(f'ffn_m{m}',source_plan('ffn',m=m,d=2048,i=11008)) for m in (5,128)]]
    for name,program in cases:
        d=out/'workloads'/name;write(d/'source.json',program)
        materialize(d/'source.json',d/'base')
        flags=['--tile-n=5632'] if name.startswith('ffn_m') else ['--tile-n=11']
        materialize(d/'source.json',d/'n',flags)
        flags=['--tile-k=5504'] if name.startswith('ffn_m') else ['--tile-k=7']
        materialize(d/'source.json',d/'k',flags)
        if name=='gemm_relu': materialize(d/'source.json',d/'fused',['--fuse'])
        write(d/'config.json',configuration(program))
    write(out/'cost-boundary.json',dict(kind='theoretical_sensitivity', executed=False,
        assumptions='Single cold graph. All original inputs begin in host. No free preloading or repeated steady state.',
        rates=dict(cpu_gflops=100,npu_gflops=1000,copy_decimal_gbytes_s=12),
        limitations='These are explicit scenario parameters, not calibrated performance of this machine. Prior full-graph measurements are not decomposable compute costs. Existing primitive availability restricts candidate implementations.'))


def plan(out):
    def check_plan(program, config, plan, mlir_path):
        import tempfile
        with tempfile.TemporaryDirectory() as temporary:
            cfg=Path(temporary)/"config.json"; candidate=Path(temporary)/"plan.json"
            write(cfg,config);write(candidate,plan)
            run=subprocess.run([str(ROOT/"build-core/tools/taskflow-opt/taskflow-opt"),str(mlir_path),"--target-config="+str(cfg),"--check-mapping="+str(candidate)],capture_output=True,text=True)
            if not run.stdout: raise RuntimeError(run.stderr)
            return json.loads(run.stdout)
    def map_program(program, config, mlir_path):
        import tempfile
        with tempfile.TemporaryDirectory() as temporary:
            cfg=Path(temporary)/'config.json'; output=Path(temporary)/'mapping.json'
            write(cfg,config)
            command=[str(ROOT/'build-core/tools/taskflow-opt/taskflow-opt'),str(mlir_path),
                '--static-map','--target-config='+str(cfg),'--mapping-output='+str(output)]
            run=subprocess.run(command,capture_output=True,text=True)
            if not output.exists(): raise RuntimeError(run.stderr)
            return load(output)
    for directory in sorted((out/'workloads').iterdir()):
        config=load(directory/'config.json')
        for variant in ('base','n','k','fused'):
            dest=directory/variant
            if not (dest/'plan.json').exists(): continue
            p=load(dest/'plan.json');result=map_program(p,config,dest/'taskflow.mlir')
            write(dest/'mapping.json',result)
            if result['status']!='passed': raise RuntimeError(str(result.get('reasons')))
            checked=check_plan(p,config,result['best_plan'],dest/'taskflow.mlir');write(dest/'check.json',checked)
            if not checked['passed']: raise RuntimeError(str(checked))
            print(directory.name,variant,result['best_plan']['makespan_ms'],flush=True)
        # Isolate the effect of communication while keeping every compute cost.
        slow=copy.deepcopy(config)
        for link in slow['hardware']['links']: link['bandwidth_bytes_per_ms']=1_000_000
        write(directory/'slow-link-config.json',slow)
        write(directory/'slow-link-mapping.json',map_program(load(directory/'base/plan.json'),slow,directory/'base/taskflow.mlir'))


def search(out, case, resume=False):
    directory=out/'search'/case;directory.mkdir(parents=True,exist_ok=True)
    workload=out/'workloads'/case
    files=[str(ROOT/'build-amoeba/tools/mlir-amoeba-opt/mlir-amoeba-opt')]
    code={p:digest(p) for p in files}
    target=dict(mapping_config=str(workload/'config.json'),cache_dir=str(out/'cache'),
        implementation_files=files,implementation_sha256=identity(code),
        search_preferences=dict(m_alignment=64,n_alignment=256,k_alignment=64))
    target_path=directory/'target.json'
    if target_path.exists() and load(target_path)!=target: raise ValueError('implementation/config changed; use new output')
    write(target_path,target)
    args=dict(**{'decision-flow':'structured','backend':'static','structured-plan':str(workload/'source.json'),
        'evaluator':'native','target-config':str(target_path),
        'output-dir':str(directory),'max-rounds':2,'max-candidates':9,'beam-width':2})
    if resume: args['resume']='true'
    elif (directory/'checkpoint.json').exists(): raise ValueError('existing search; pass --resume')
    if any(any(c.isspace() for c in str(v)) for v in args.values()): raise ValueError('paths must not contain whitespace')
    cmd=[str(ROOT/'build-amoeba/tools/mlir-amoeba-opt/mlir-amoeba-opt'),str(workload/'base/taskflow.mlir'),
         '--search-joint-neighborhood='+' '.join(f'{k}={v}' for k,v in args.items()),'-o',str(directory/'output.mlir')]
    write(directory/'launch.json',cmd)
    with (directory/('resume.log' if resume else 'search.log')).open('w') as stream:
        r=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
    if r.returncode: raise RuntimeError('original ORBIT search failed: '+str(directory))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','plan','search','all'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',default='ffn_m5')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();out=a.output.resolve()
    if a.stage in ('prepare','all'):
        if (out/'workloads').exists(): raise ValueError('refusing to overwrite existing workload evidence')
        prepare(out)
    if a.stage in ('plan','all'): plan(out)
    if a.stage in ('search','all'):
        for case in ([a.case] if a.stage=='search' else ['gemm_relu','ffn','ffn_m5','ffn_m128']): search(out,case,a.resume)


if __name__=='__main__': main()
