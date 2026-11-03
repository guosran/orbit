#!/usr/bin/env python3
"""Describe exact measured NPUChain regions without importing opaque costs as compute.

Held-out prediction belongs to the independent hardware auditor. This tool
publishes capability/identity/cost boundaries and references that audit.
"""
from __future__ import annotations
import argparse, hashlib, json, math, statistics
from pathlib import Path

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text())

def classify(program):
    """Recognize the narrowly supported two-GEMM region; no runtime fallback."""
    try:
        if program['schema']!='orbit.taskflow.plan.v1': raise ValueError('unsupported public schema')
        if any(op['kind'] not in ('fill','matmul') for task in program['tasks'] for op in task['operations']):
            raise ValueError('CPU activation/other operation boundary prevents this two-GEMM chain; FFN SwiGLU is not removed')
        tasks=program['tasks'];buffers={b['id']:b for b in program['buffers']}
        if len(buffers)!=len(program['buffers']) or any('alias_of' in b for b in buffers.values()): raise ValueError('duplicate or aliased storage unsupported')
        if len(tasks)!=2: raise ValueError('this catalogue supports exactly two dependent GEMM tasks')
        for t in tasks:
            if len(t['operations'])!=2: raise ValueError('expected explicit zero-fill then full-K matmul')
            fill,op=t['operations']
            if fill['kind']!='fill' or op['kind']!='matmul' or not op.get('accumulate') or any(fill[k]!=op[k] for k in ('output','m','n')):
                raise ValueError('kernel zero initialization does not match public accumulator')
            if op['m']!=[0,buffers[op['output']]['shape'][0]] or op['n']!=[0,buffers[op['output']]['shape'][1]] or op['k']!=[0,buffers[op['inputs'][0]]['shape'][1]]:
                raise ValueError('only whole-matrix full-K operations in this measured catalogue')
        a,b=[t['operations'][1] for t in tasks]
        if tasks[0]['depends_on'] or tasks[1]['depends_on']!=[tasks[0]['id']] or b['inputs'][0]!=a['output']:
            raise ValueError('not the supported producer/consumer chain')
        if buffers[a['inputs'][0]]['role']!='input' or buffers[a['inputs'][1]]['role']!='input' or buffers[b['inputs'][1]]['role']!='input' or buffers[a['output']]['role']!='temporary' or buffers[b['output']]['role']!='output' or program['outputs']!=[b['output']]:
            raise ValueError('input/weight/intermediate/final storage roles do not match catalogue')
        if any(buf['dtype']!='f32' for buf in buffers.values()): raise ValueError('public semantic dtype must be f32')
        shape=dict(m=a['m'][1],k=a['k'][1],h=a['n'][1],n=b['n'][1])
        expected={a['inputs'][0]:[shape['m'],shape['k']],a['inputs'][1]:[shape['k'],shape['h']],a['output']:[shape['m'],shape['h']],b['inputs'][1]:[shape['h'],shape['n']],b['output']:[shape['m'],shape['n']]}
        if len(expected)!=5 or set(expected)!=set(buffers) or any(buffers[name]['shape']!=size for name,size in expected.items()): raise ValueError('full matrix storage shapes do not match catalogue')
        if b['m']!=a['m'] or b['k'][1]!=shape['h']: raise ValueError('incompatible intermediate region')
        if shape not in [dict(m=128,k=256,h=256,n=256),dict(m=128,k=256,h=512,n=256)]:
            raise ValueError('shape outside demonstrated catalogue; no measured cost or executable claim')
        return dict(compatible=True,shape=shape,source_id=program['source_id'],candidate_id=program['candidate_id'],
                    numerical_permission_required='BF16 inputs and both output stores, BFP16 EBS8 MAC, FP32 accumulators; final host representation f32',
                    meaning='target catalogue eligibility only; not new runtime execution or independent numerical certification')
    except (KeyError,TypeError,ValueError,IndexError) as error:
        return dict(compatible=False,reason=str(error))

def describe(chain_root,audit=None):
    audit_data=load(audit) if audit and Path(audit).exists() else {}
    certified={str(Path(row['source']).resolve()) for row in audit_data.get('chains',[]) if row.get('passed')}
    result=dict(schema='orbit.chain.capability-cost.v1',kind='backend_target_implementation',
        common_taskflow_modified=False,gate_up_merge='different transformation; not represented by chain packing',
        supports=['two reused independent one-op contexts','one existing two-op NPUChain with explicit shared BF16 intermediate'],
        host_boundaries='CPU SwiGLU splits standard FFN; whole FFN rejected by this catalogue',
        measured_regions=[],failures=[],
        cost_boundary=dict(scope='opaque_complete_synchronous_region_wall_ms',includes=['host input staging','necessary BO synchronization','host submission and wait','intermediate host readback/staging in independent mode','final readback and BF16-to-FP32 output conversion'],
            excludes=['compilation','context initialization','first weight upload','first execution','extra warmup','numerical checking','evidence writes'],
            import_into_static_compute_only='unsupported: staging/synchronization and host overhead are inseparable and would be double-counted',
            allowed_use='exact known region identity as separate target evaluation score; not general per-task compute coefficient'),
        prediction=dict(owner='independent hardware audit',model='train-only fixed saving conditional on held-out independent wall; no duplicate model fitted here'))
    for case in ('train','holdout'):
        path=Path(chain_root)/case/'result.json'
        if not path.exists(): result['failures'].append(dict(case=case,status='missing'));continue
        run=load(path)
        if run.get('status')!='passed':
            result['failures'].append(dict(case=case,status=run.get('status'),error=run.get('error'),evidence=str(path)));continue
        program=load(path.parent/'plan.json');capability=classify(program)
        if not capability['compatible']:
            result['failures'].append(dict(case=case,status='unsupported',reason=capability['reason']));continue
        for mode,variant in run['variants'].items():
            samples=[s['wall_ms'] for s in variant['samples']]
            if len(samples)<3 or not all(isinstance(x,(int,float)) and math.isfinite(x) and x>0 for x in samples) or not all(s['check']['passed'] for s in variant['samples']):
                result['failures'].append(dict(case=case,mode=mode,status='unscored'));continue
            result['measured_regions'].append(dict(case=case,split=run['split'],mode=mode,capability=capability,
                score=dict(value=statistics.median(samples),unit='ms',kind='measured_complete_region_wall') if str(path.resolve()) in certified else None,observed_median_ms=statistics.median(samples),samples_ms=samples,
                scope='exact saved inputs, precision, physical mapping, object reuse and instrumentation',
                source_sha256=sha(path.parent/'source.json'),program_sha256=sha(path.parent/'plan.json'),target_sha256=sha(path.parent/'target-plan.json'),
                evidence=str(path),evidence_sha256=sha(path),implementation_sha256=run['implementation_sha256'],input_identity=run['input_identity'],
                actual_submissions_per_sample=[s['actual_submissions'] for s in variant['samples']],
                configuration_switch_cost='included in wall, not independently identified',
                intermediate_location='system DRAM; additionally host-visible between calls' if mode=='independent' else 'system DRAM shared argument; no midpoint host readback',
                validation='independent audit passed for this exact evidence path' if str(path.resolve()) in certified else 'experiment checks only; no matching independent audit; unscored'))
    if audit:
        path=Path(audit)
        result['prediction']['audit']=dict(path=str(path),exists=path.exists())
        if path.exists(): result['prediction']['audit'].update(sha256=sha(path),content=load(path))
    result['status']='audited_measured_catalogue' if any(r['score'] is not None for r in result['measured_regions']) else 'unscored'
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--chain-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--audit',type=Path);p.add_argument('--program',type=Path)
    args=p.parse_args();result=describe(args.chain_root,args.audit)
    if args.program: result['requested_program']=classify(load(args.program))
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(result['status'])
if __name__=='__main__':main()
