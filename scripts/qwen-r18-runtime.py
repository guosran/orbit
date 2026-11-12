#!/usr/bin/env python3
"""r18 executable FFN fusion adapter, separate from the sealed r17 hooks.

Costs are measured at actual implementation boundaries. Inline host operations
remain one explicitly measured residual group, never invented per-node prices.
"""
import importlib.util
from pathlib import Path
import time
import ast
import sys


def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


node=load(Path(__file__).with_name('qwen-node-runtime.py'),'orbit_r18_source')
backend=load(Path(__file__).resolve().parents[1]/'experimental/qwen-runtime-search/qwen-r18-ffn.py','orbit_r18_ffn')
extract_prefill_source=node.extract_prefill_source


def validate_plan(source,plan):
    choices=node.validate_plan(source,plan)
    if plan.get('plan_id')!=node.digest({k:v for k,v in plan.items() if k!='plan_id'}):
        raise ValueError('Invalid fixed r18 plan identity')
    config=plan.get('r18')
    if not isinstance(config,dict) or set(config)!={'fused_layers','rows','block_m','block_n','retention','split_order','gemm_variants','gemm_backend','gemm_context_policy','ffn_artifact'}:
        raise ValueError('An explicit complete r18 backend configuration is required')
    layers=config['fused_layers']
    if not isinstance(layers,list) or len(set(layers))!=len(layers) or any(type(i)!=int or i<0 or i>=36 for i in layers):
        raise ValueError('Invalid source layer coverage')
    for layer in layers:
        for role in ('gate_up','swiglu','down'):
            if choices[f'prefill.L{layer}.{role}']!='npu':
                raise ValueError('A fused region requires all three arithmetic choices on NPU')
    if layers:
        import json
        proof=config['ffn_artifact']
        if set(proof)!={'manifest','sha256'} or node.file_sha(proof['manifest'])!=proof['sha256']:raise ValueError('FFN compiled manifest changed')
        artifact=json.loads(Path(proof['manifest']).read_text())
        if not artifact.get('compiled') or node.file_sha(artifact['artifact'])!=artifact['artifact_sha256']:raise ValueError('FFN compiled ELF changed')
        for key in ('row_tile','block_m','block_n','split_order'):
            if artifact[key]!=config['rows' if key=='row_tile' else key]:raise ValueError('FFN compiled execution parameters differ')
        for row in artifact['source_files']:
            if node.file_sha(row['file'])!=row['sha256']:raise ValueError('FFN source changed since compilation')
    if config['gemm_context_policy']!='close-after-call':raise ValueError('Only checked exclusive GEMM context scope is supported')
    operators={op['id']:op for op in source['operators']}
    for name,variant in config['gemm_variants'].items():
        if name not in operators or operators[name]['kind']!='matmul' or choices[name]!='npu':
            raise ValueError('GEMM variant requires an actual NPU source projection')
        if operators[name]['attributes']['layer'] in layers and name.rsplit('.',1)[1] in ('gate_up','down'):
            raise ValueError('An individual GEMM variant overlaps a fused source cover')
        if set(variant)!={'directory','manifest_sha256'} or node.file_sha(Path(variant['directory'])/'implementation.json')!=variant['manifest_sha256']:
            raise ValueError('GEMM artifact manifest changed')
    if config['gemm_variants']:
        if set(config['gemm_backend'])!={'file','sha256'} or node.file_sha(config['gemm_backend']['file'])!=config['gemm_backend']['sha256']:
            raise ValueError('GEMM backend code changed')
    return choices


def make_prefiller_class(base,source,plan):
    choices=validate_plan(source,plan);config=plan['r18']
    root=Path(next(b['file'] for b in source['bindings'] if b['symbol']=='Qwen25Prefill._layer')).parents[2]
    ffn=backend.QwenFFN(root,config['rows'],config['block_m'],config['block_n'],config['retention'],config['split_order'])
    if config['fused_layers']:
        import json
        proof=json.loads(Path(config['ffn_artifact']['manifest']).read_text())
        chain=ffn.build();chain._elf_path=proof['artifact'];chain._kernel_name=proof['entrypoint']
    fused_base=backend.make_prefiller_class(base,ffn,config['fused_layers'])
    source_ids={op['id'] for op in source['operators']}
    binding_files={row['symbol']:Path(row['file']) for row in source['bindings']}
    def statements(path,owner,name):
        fn=node.method(ast.parse(path.read_text()),owner,name)
        return [x for x in fn.body if not (isinstance(x,ast.Expr) and isinstance(x.value,ast.Constant) and isinstance(x.value.value,str))]
    layer_statements=statements(binding_files['Qwen25Prefill._layer'],'Qwen25Prefill','_layer')
    prefill_statements=statements(binding_files['LlamaPrefill.prefill'],'LlamaPrefill','prefill')
    layer_offset=base._layer.__code__.co_firstlineno-1
    groups=[(['attn_norm'],2),(['qkv','qkv_bias'],3),(['qkv_split'],4),(['q_rope'],5),(['k_rope'],6),(['kv_store'],7),(['attention'],8),(['o','residual1'],9),(['ffn_norm'],10),(['gate_up','gate_split'],11),(['swiglu','down','residual2'],12)]
    gemm_backend=load(config['gemm_backend']['file'],'orbit_r18_gemm') if config['gemm_variants'] else None
    class Prefiller(fused_base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self._orbit_active_layer=None;self.orbit_trace=[];self.orbit_preparation=[]
            self._orbit_gemm={name:gemm_backend.CompiledGemm(row['directory']) for name,row in config['gemm_variants'].items()}
            self._orbit_gemm_weights={}
        def make_npu_resident(self):
            import torch
            import kernels
            if getattr(self,'_orbit_prepared',False):raise ValueError('Plan preparation must run once')
            for layer,weights in enumerate(self._w):
                if layer in self._orbit_fused_layers:
                    ffn.add_layer(layer,weights['gate_up'],weights['down'])
                    weights['gate_up']=None;weights['down']=None
                    self.orbit_preparation.append(dict(layer=layer,action='fission-original-weight-bits',
                        released_original=['gate_up','down'],abi=ffn.describe()))
                for role in ('qkv','o','gate_up','down'):
                    if weights[role] is None:continue
                    name=f'prefill.L{layer}.{role}'
                    if name in self._orbit_gemm:
                        weight=weights[role]
                        array=weight.view(torch.uint16).numpy().view(__import__('ml_dtypes').bfloat16)
                        if not torch.equal(torch.from_numpy(array.view('uint16')).view(torch.bfloat16).view(torch.int16),weight.view(torch.int16)):
                            raise ValueError('Compiled GEMM numpy binding changed original weight bits')
                        self._orbit_gemm_weights[name]=array
                        self.orbit_preparation.append(dict(layer=layer,role=role,action='actual-source-weight-view-no-copy',bits_preserved=True))
                        continue
                    if choices[f'prefill.L{layer}.{role}']!='npu':continue
                    original=weights[role];prepared=kernels.resident_weight(original,exact=True)
                    if kernels.shared_bo(prepared.padded) is None:
                        shared=kernels.shared_empty(prepared.padded.shape,prepared.padded.dtype)
                        if kernels.shared_bo(shared) is None:raise ValueError('NPU static path requires actual shared BO')
                        shared.copy_(prepared.padded);prepared.padded=shared
                    if not torch.equal(original.view(torch.int16),prepared.padded[:prepared.K,:prepared.N].view(torch.int16)):
                        raise ValueError('Preparation changed source weight bits')
                    weights[role]=prepared
                    self.orbit_preparation.append(dict(layer=layer,role=role,action='original-static-weight',bits_preserved=True))
            self._orbit_prepared=True
            return self
        def _layer(self,x,L,N,keep=None):
            if self._orbit_active_layer is not None:raise ValueError('Nested source layer')
            self._orbit_active_layer=L
            try:return fused_base._layer(self,x,L,N,keep)
            finally:self._orbit_active_layer=None
        def _record(self,operator,call):
            if operator not in source_ids:raise ValueError('Unbound source operator')
            stamp=time.perf_counter_ns();result=call()
            self.orbit_trace.append(dict(operators=[operator],exclusive_wall_ns=time.perf_counter_ns()-stamp))
            return result
        def _matmul(self,x,w,backend=None,**kwargs):
            layer=self._orbit_active_layer
            roles=[role for role in ('qkv','o','gate_up','down') if self._w[layer][role] is w]
            if len(roles)!=1:raise ValueError('Projection source weight identity mismatch')
            name=f'prefill.L{layer}.{roles[0]}'
            if name in self._orbit_gemm:
                def compiled():
                    import torch
                    from ml_dtypes import bfloat16
                    native=self._orbit_gemm[name]
                    # Explicit exclusive context scope includes open/close and
                    # restaging in this measured call. No stale context/BO lease
                    # is silently retained across other firmware programs.
                    a=x.to(torch.bfloat16).contiguous().view(torch.uint16).numpy().view(bfloat16)
                    try:result=native.run(a,self._orbit_gemm_weights[name],name)
                    finally:
                        native.close()
                        # The context cannot reuse static BOs after close.
                        # Keeping all 144 padded host packs would add ~5GB
                        # without reducing the next invocation's device work.
                        # Repacking/release belongs to this measured hook.
                        native.weights.clear()
                    return torch.from_numpy(result)
                result=self._record(name,compiled)
                record=self.orbit_trace[-1];native=self._orbit_gemm[name]
                event=dict(native.events[-1]);bridge=record['exclusive_wall_ns']-event['wall_ns']
                if bridge<0:raise ValueError('GEMM implementation clock overlaps enclosing hook')
                record.update(gemm_variant=config['gemm_variants'][name],native_phases=event,
                    source_bridge_and_close_ns=bridge,context_policy=config['gemm_context_policy'],
                    host_pack_cache_policy='clear-after-call')
                return result
            if choices[name]=='npu':kwargs['stage_key']='r18_'+name
            return self._record(name,lambda:base._matmul(self,x,w,backend=choices[name],**kwargs))
        def _rms_norm(self,x,w,eps,backend=None):
            layer=self._orbit_active_layer
            if layer is None:name='prefill.final_norm'
            else:
                roles=[role for role in ('attn_norm','ffn_norm') if self._w[layer][role] is w]
                if len(roles)!=1:raise ValueError('Norm source weight identity mismatch')
                name=f'prefill.L{layer}.{roles[0]}'
            return self._record(name,lambda:base._rms_norm(self,x,w,eps,backend=choices[name]))
        def _swiglu(self,g,u,backend=None):
            name=f'prefill.L{self._orbit_active_layer}.swiglu'
            return self._record(name,lambda:base._swiglu(self,g,u,backend=choices[name]))
        def _attention(self,*args,**kwargs):
            name=f'prefill.L{self._orbit_active_layer}.attention';kwargs['backend']='cpu'
            return self._record(name,lambda:base._attention(self,*args,**kwargs))
        def _rope(self,x,lut,heads,backend=None):
            if heads not in (16,2):raise ValueError('Source head ABI mismatch')
            name=f'prefill.L{self._orbit_active_layer}.'+('q_rope' if heads==16 else 'k_rope')
            return self._record(name,lambda:base._rope(self,x,lut,heads,backend='cpu'))
        def _kv_store(self,L,k,v,keep):
            if L!=self._orbit_active_layer:raise ValueError('KV source layer mismatch')
            return self._record(f'prefill.L{L}.kv_store',lambda:base._kv_store(self,L,k,v,keep))
        def _lm_head(self,x,w,backend=None):
            return self._record('prefill.lm_head',lambda:base._lm_head(self,x,w,backend='cpu'))
        def prefill(self,ids):
            self.orbit_trace=[];self._orbit_ffn.events=[];self._orbit_ffn.call_events=[]
            self._orbit_lines={};active={};stack=[]
            allowed={base.prefill.__code__:'prefill',fused_base._layer.__code__:'layer'}
            def trace(frame,event,arg):
                if frame.f_code not in allowed:return None
                now=time.perf_counter_ns();key=id(frame)
                if event=='call':
                    if stack:
                        parent=stack[-1];line,stamp,owner,kind=active[parent]
                        self._orbit_lines[(kind,line,owner)]=self._orbit_lines.get((kind,line,owner),0)+now-stamp
                    active[key]=(frame.f_lineno,now,frame.f_locals.get('L') if allowed[frame.f_code]=='layer' else None,allowed[frame.f_code]);stack.append(key)
                else:
                    if key not in active:return trace
                    line,stamp,owner,kind=active[key]
                    self._orbit_lines[(kind,line,owner)]=self._orbit_lines.get((kind,line,owner),0)+now-stamp
                    if event=='return':
                        del active[key]
                        if stack.pop()!=key:raise ValueError('Source trace nesting changed')
                        if stack:
                            parent=stack[-1];line,stamp,owner,kind=active[parent];active[parent]=(line,now,owner,kind)
                    else:active[key]=(frame.f_lineno,now,frame.f_locals.get('L') if allowed[frame.f_code]=='layer' else None,allowed[frame.f_code])
                return trace
            if sys.gettrace() is not None:raise ValueError('Refuse to overwrite an existing source tracer')
            start=time.perf_counter_ns();sys.settrace(trace)
            try:out=base.prefill(self,ids)
            finally:sys.settrace(None)
            self._orbit_wall_ns=time.perf_counter_ns()-start
            return out
        def source_profile(self):
            records=list(self.orbit_trace);costs={}
            calls={row['operators'][0]:row['exclusive_wall_ns'] for row in records}
            def line_total(statement,kind,layer):
                lo=statement.lineno-(layer_offset if kind=='layer' else 0)
                hi=statement.end_lineno-(layer_offset if kind=='layer' else 0)
                return sum(ns for (k,line,owner),ns in self._orbit_lines.items() if k==kind and owner==layer and lo<=line<=hi)
            def group(names,statement,kind,layer):
                total=line_total(statement,kind,layer)
                inline=[name for name in names if name not in calls]
                if not inline:
                    if len(names)!=1:raise ValueError('Ambiguous source statement cost')
                    costs[names[0]]=total
                else:
                    if len(inline)!=1:raise ValueError('Need an actual inline source owner')
                    remainder=total-sum(calls[name] for name in names if name in calls)
                    if remainder<0:raise ValueError('Source line/call timers overlap')
                    costs.update({name:calls[name] for name in names if name in calls});costs[inline[0]]=remainder
            for layer in range(36):
                for roles,index in groups:
                    if layer in self._orbit_fused_layers and index in (11,12):continue
                    group([f'prefill.L{layer}.'+role for role in roles],layer_statements[index],'layer',layer)
            costs['prefill.embedding']=sum(line_total(prefill_statements[i],'prefill',None) for i in (4,5))
            group(['prefill.final_norm','prefill.last_row'],prefill_statements[8],'prefill',None)
            group(['prefill.lm_head','prefill.logits_row'],prefill_statements[9],'prefill',None)
            for layer in sorted(self._orbit_fused_layers):
                events=[e for e in ffn.call_events if e['layer']==layer]
                if len(events)!=1:raise ValueError('Each fused source layer must execute once per prefill')
                operators=[f'prefill.L{layer}.'+role for role in ('gate_up','gate_split','swiglu','down')]
                records.append(dict(operators=operators,exclusive_wall_ns=sum(e['exclusive_wall_ns'] for e in events),
                    implementation='actual-four-configuration-FFN-dispatch',events=events))
                residual2=line_total(layer_statements[12],'layer',layer)-sum(e['exclusive_wall_ns'] for e in events)
                if residual2<0:raise ValueError('Fused FFN source line/call timers overlap')
                costs[f'prefill.L{layer}.residual2']=residual2
            covers=[r for r in records if len(r['operators'])>1]
            covered=[name for row in covers for name in row['operators']]
            if set(costs)&set(covered) or set(costs)|set(covered)!=source_ids:raise ValueError('Source timing coverage mismatch')
            residual=self._orbit_wall_ns-sum(costs.values())-sum(r['exclusive_wall_ns'] for r in covers)
            if residual<0:raise ValueError('Runtime phase accounting overlaps')
            return dict(schema='orbit.qwen-r18-implementation-profile.v1',source_id=source['source_id'],plan_id=plan['plan_id'],
                wall_ns=self._orbit_wall_ns,implementation_records=records,node_cost_ns=costs,fused_covers=covers,host_control_ns=residual,
                exclusive_sum_equals_wall=sum(costs.values())+sum(r['exclusive_wall_ns'] for r in covers)+residual==self._orbit_wall_ns,
                instrumentation='r18 source-line/call boundaries with parent source frames paused during bound child source execution',
                limitation='Python control measured as exclusive residual; selection-independent treatment is a model approximation')
    return Prefiller


def main():
    import argparse
    import json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--base-plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--layers',default='all',help='all, none, or comma-separated source layer indices')
    p.add_argument('--rows',type=int,default=128)
    p.add_argument('--block-m',type=int,default=128)
    p.add_argument('--block-n',type=int,default=256)
    p.add_argument('--retention',choices=['retain','reload','release-after-call'],default='retain')
    p.add_argument('--split-order',choices=['gate-up','up-gate'],default='gate-up')
    p.add_argument('--gemm-variants',type=Path,help='JSON mapping exact source op ids to compiled artifact directories')
    p.add_argument('--gemm-backend',type=Path,default=Path(__file__).resolve().parents[1]/'experimental/qwen-runtime-search/npu-gemm-variants.py')
    p.add_argument('--ffn-manifest',type=Path,help='Required exact compiled shape/order manifest for fused layers')
    a=p.parse_args();source=json.loads(a.source.read_text());original=json.loads(a.base_plan.read_text())
    node.validate_plan(source,original)
    layers=list(range(36)) if a.layers=='all' else [] if a.layers=='none' else [int(x) for x in a.layers.split(',')]
    placements=dict(original['placements'])
    for layer in layers:
        for role in ('gate_up','swiglu','down'):placements[f'prefill.L{layer}.{role}']='npu'
    plan=node.make_plan(source,placements)
    mapping=json.loads(a.gemm_variants.read_text()) if a.gemm_variants else {}
    variants={name:dict(directory=str(Path(directory).resolve(strict=True)),manifest_sha256=node.file_sha(Path(directory)/'implementation.json')) for name,directory in mapping.items()}
    plan['r18']=dict(fused_layers=layers,rows=a.rows,block_m=a.block_m,block_n=a.block_n,retention=a.retention,split_order=a.split_order,
        gemm_variants=variants,gemm_backend=dict(file=str(a.gemm_backend.resolve(strict=True)),sha256=node.file_sha(a.gemm_backend)) if variants else {},gemm_context_policy='close-after-call',
        ffn_artifact=dict(manifest=str(a.ffn_manifest.resolve(strict=True)),sha256=node.file_sha(a.ffn_manifest)) if a.ffn_manifest else {})
    plan['launch_policy']='r18 exact FFN source covers use one compiled chain; other operators use original implementations'
    plan['plan_id']=node.digest({k:v for k,v in plan.items() if k!='plan_id'})
    validate_plan(source,plan)
    with a.output.open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
    print(json.dumps(dict(plan_id=plan['plan_id'],fused_layers=layers)))


if __name__=='__main__':main()
