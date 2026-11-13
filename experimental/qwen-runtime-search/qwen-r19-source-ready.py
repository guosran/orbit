#!/usr/bin/env python3
"""Execute the real prefill source DAG with checked implementation covers.

This interpreter consumes source values/dependencies, not Python line order.
Actual CPU regions are supplied by the owned CPU lowerer; all original NPU
methods and fused FFN providers run on the single device owner. It preserves
original persistent KV objects for the unchanged complete decode handoff.
"""
import hashlib
import ctypes
import gc
import json
from pathlib import Path
import time


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def make_class(prefiller,base,source,plan,config,ready,xrt,cpu_module):
    operators={op['id']:op for op in source['operators']};program=cpu_module.Program(source)
    byvalue={v['id']:v for v in source['values']}
    mapping={};definitions=[]
    def add(identity,names,device,kind,parameters,proof=None):
        if any(n in mapping for n in names):raise ValueError('Overlapping runtime source covers')
        if any(n not in operators for n in names):raise ValueError('Unknown covered source operator')
        for name in names:mapping[name]=identity
        definitions.append(dict(id=identity,source_operators=names,device=device,
            implementation_kind=kind,runtime_parameters=parameters,proof=proof))
    for row in config.get('cpu_regions',[]):
        proof=cpu_module.region(program,row['operators'])
        if any(plan['placements'][n]!='cpu' for n in row['operators']):raise ValueError('CPU cover conflicts with prepared implementation')
        params={k:row.get(k,default) for k,default in [('backend','eager-host-group'),('input_policy','borrow-host'),('output_policy','retain-host')]}
        add(row.get('id','cpu-region.'+proof['region_id']),proof['operators'],'cpu','cpu-region',params,proof)
    for layer in plan['r18']['fused_layers']:
        names=[f'prefill.L{layer}.{role}' for role in ('gate_up','gate_split','swiglu','down')]
        route=plan.get('r18_layer_ffn',{}).get(str(layer),{k:plan['r18'][k] for k in ('retention','split_order','ffn_artifact')})
        add(config.get('implementation_ids',{}).get(names[0],f'prefill.L{layer}.fused-ffn'),names,'npu','prefill-fused-ffn',dict(layer=layer,**route))
    for op in source['operators']:
        if op['id'] in mapping:continue
        name=op['id'];device=plan['placements'][name]
        params=dict(operator=name,device=device)
        kind='prefill-original'
        if name in config['operators']:
            kind='heterogeneous-gate-up';params=dict(operator=name,**config['operators'][name])
        elif name in plan['r18']['gemm_variants']:
            kind='prefill-gemm';params.update(plan['r18']['gemm_variants'][name])
        add(config.get('implementation_ids',{}).get(name,name+'.'+kind+'.'+device),[name],device,kind,params)
    if set(mapping)!=set(operators):raise ValueError('Runtime implementations do not cover complete source')
    priorities=config.get('ready_priority',{})
    if not isinstance(priorities,dict):raise ValueError('ready_priority must map actual implementation IDs to integer priority')
    if any(name not in {d['id'] for d in definitions} for name in priorities):raise ValueError('Priority names unselected implementation')
    # Region boundaries depend only on the immutable source/cover, never on
    # tensor values. Compute once before entering any inference timer.
    consumers={name:set() for name in byvalue}
    for op in operators.values():
        for value in op['inputs']:consumers[value].add(op['id'])
    bindings={b['id']:b for b in source['bindings']}
    cpu_identity=hashlib.sha256(Path(cpu_module.__file__).read_bytes()).hexdigest()
    for definition in definitions:
        names=definition['source_operators'];chosen=set(names)
        produced={value for name in names for value in operators[name]['outputs']}
        ins=list(dict.fromkeys(value for name in names for value in operators[name]['inputs'] if value not in produced))
        outs=[value for name in names for value in operators[name]['outputs'] if byvalue[value]['role']=='output' or consumers[value]-chosen]
        deps={dep for name in names for dep in operators[name]['dependencies'] if dep not in chosen}
        definition['boundary']=(ins,outs,deps)
        artifact=dict(source_bindings=[bindings[key] for key in sorted({operators[name]['binding'] for name in names})])
        if definition['device']=='cpu':artifact['cpu_lowerer_sha256']=cpu_identity
        if definition['proof'] is not None:artifact['region_id']=definition['proof']['region_id']
        artifact['runtime_parameters']=definition['runtime_parameters']
        definition['artifact_key']=digest(artifact)
    class SourceReadyPrefiller(prefiller):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self._r19_cpu={d['id']:cpu_module.ExecutableRegion(program,d['source_operators'],**d['runtime_parameters']) for d in definitions if d['implementation_kind']=='cpu-region'}
            self._r19_single_cpu={d['id']:cpu_module.ExecutableRegion(program,d['source_operators']) for d in definitions if d['device']=='cpu' and d['implementation_kind']=='prefill-original' and operators[d['source_operators'][0]]['kind'] in cpu_module.KINDS}
            self._r19_half={d['id']:xrt.PrivateGemm(d['runtime_parameters']['directory']) for d in definitions if d['implementation_kind']=='heterogeneous-gate-up'}
            self._r19_last_profile=None;self._r19_preparing=False;self._r19_compile_proofs={}
            self._r19_program,self._r19_record_parts=self._r19_build_operations()
        def make_npu_resident(self):
            result=super().make_npu_resident()
            compiled=[d for d in definitions if d['implementation_kind']=='cpu-region' and d['runtime_parameters']['backend']=='inductor-cpu']
            if compiled:
                tokens=config.get('prepare_tokens')
                if not isinstance(tokens,list) or len(tokens)!=byvalue['input.tokens']['shape'][0]:raise ValueError('Compiled regions require exact untimed preparation tokens')
                directory=Path(config['cpu_prepare_directory']);directory.mkdir(parents=True,exist_ok=False)
                self._r19_preparing=True
                try:self.prefill(tokens)
                finally:self._r19_preparing=False
                self.clear_context()
                (directory/'preparation-profile.json').write_text(json.dumps(self._r19_last_profile,indent=2)+'\n')
                gc.collect();trim=ctypes.CDLL('libc.so.6').malloc_trim(0)
                self.orbit_preparation.append(dict(action='r19-actual-input-cpu-region-compilation',inside_request_timing=False,compile_proofs=self._r19_compile_proofs,malloc_trim_result=int(trim)))
            return result
        def _r19_inputs(self,ids):
            import torch
            values={'input.tokens':ids,'weights.embedding':self.embed,'input.rope_lut':self._lut,
                'weights.final_norm':self.final_norm,'weights.lm_head':self.lm_head}
            for name,row in byvalue.items():
                if name.startswith('weights.L'):
                    layer,role=name[len('weights.L'):].split('.',1)
                    # Fused weight originals are deliberately released; those
                    # inputs are consumed by the actual prepared FFN provider.
                    values[name]=self._w[int(layer)][role]
            return values
        def _r19_build_operations(self):
            import torch
            import numpy as np
            from ml_dtypes import bfloat16
            node_defs=[];source_end={};record_parts={};regions=self._r19_cpu
            def original(d,inputs):
                op=operators[d['source_operators'][0]];names=op['inputs'];args=[inputs[n] for n in names];attrs=op['attributes'];kind=op['kind'];device=d['device'];name=op['id']
                if kind=='embedding-gather':
                    output=torch.from_numpy(np.asarray(args[1][np.asarray(args[0])],dtype=np.float32))
                elif kind=='kv-prefix-store':
                    layer=attrs['layer'];base._kv_store(self,layer,*args,attrs['keep'])
                    return dict(zip(op['outputs'],[self.kv_k[layer][:byvalue['input.tokens']['shape'][0]],self.kv_v[layer][:byvalue['input.tokens']['shape'][0]]]))
                elif kind=='slice-view':output=args[0][attrs['offsets'][0]:attrs['offsets'][0]+attrs['sizes'][0]]
                elif kind=='select-row-view':output=args[0][attrs['row']]
                elif kind=='lm-head':output=base._lm_head(self,*args,backend='cpu')
                elif device=='npu':
                    if kind=='matmul':
                        kw=dict(self._wide) if name.rsplit('.',1)[-1] in ('qkv','o','gate_up') else {}
                        output=base._matmul(self,*args,backend='npu',stage_key='r19_'+name,**kw)
                    elif kind=='rms-norm':output=base._rms_norm(self,*args,attrs['eps'],backend='npu')
                    elif kind=='swiglu':output=base._swiglu(self,*args,backend='npu')
                    else:raise ValueError('No registered original NPU lowering: '+kind)
                else:
                    if d['id'] not in self._r19_single_cpu:
                        proof=cpu_module.region(program,[name]);self._r19_single_cpu[d['id']]=cpu_module.ExecutableRegion(program,[name])
                    return self._r19_single_cpu[d['id']].run(inputs)
                return {op['outputs'][0]:output}
            for d in definitions:
                identity=d['id'];names=d['source_operators'];ins,outs,deps=d['boundary'];priority=priorities.get(identity,0)
                if type(priority)!=int:raise ValueError('Priority must be an integer')
                kind=d['implementation_kind']
                if kind=='heterogeneous-gate-up':
                    op=operators[names[0]];h,w=op['inputs'];params=d['runtime_parameters'];side=params['cpu_half'];other='up' if side=='gate' else 'gate'
                    gate=identity+'.gate';up=identity+'.up';assembly=identity+'.assemble'
                    def half(values,side,device,d=d,h=h,w=w):
                        index=0 if side=='gate' else 11008;weight=values[w][:,index:index+11008]
                        if device=='cpu':return {d['id']+'.'+side:base._matmul(self,values[h],weight,backend='cpu')}
                        a=values[h].to(torch.bfloat16).contiguous().view(torch.uint16).numpy().view(bfloat16)
                        b=weight.contiguous().view(torch.uint16).numpy().view(bfloat16);native=self._r19_half[d['id']]
                        output=native.run(a,b,d['id']+'.'+side)
                        return ready.Result({d['id']+'.'+side:torch.from_numpy(output)},[native.events[-1]])
                    for part,device in ((side,'cpu'),(other,'npu')):
                        node_defs.append((ready.Operation(identity+'.'+part,[h,w],[identity+'.'+part],device,lambda values,part=part,device=device,fn=half:fn(values,part,device),priority=priority),deps))
                    def assemble(values,identity=identity,out=outs[0]):return {out:torch.cat((values[identity+'.gate'],values[identity+'.up']),dim=1)}
                    node_defs.append((ready.Operation(assembly,[gate,up],outs,'cpu',assemble,priority=priority),set()))
                    source_end.update({n:assembly for n in names});record_parts[identity]=[gate,up,assembly]
                    continue
                def call(values,d=d,ins=ins,outs=outs):
                    kind=d['implementation_kind']
                    if kind=='cpu-region':
                        region=regions[d['id']]
                        if region.backend=='inductor-cpu' and d['id'] not in self._r19_compile_proofs:
                            if not self._r19_preparing:raise RuntimeError('Unprepared CPU compiler specialization in timed request')
                            _,proof=region.prepare(values,Path(config['cpu_prepare_directory'])/digest(d['id']))
                            self._r19_compile_proofs[d['id']]=proof
                        result=region.run(values)
                        return ready.Result(result,[region.events[-1]])
                    if kind=='prefill-fused-ffn':
                        layer=d['runtime_parameters']['layer'];h=values[f'prefill.L{layer}.ffn_norm.value']
                        out=self._orbit_ffn.run(layer,h)
                        return ready.Result({f'prefill.L{layer}.down.value':out},[self._orbit_ffn.call_events[-1]])
                    if kind=='prefill-gemm':
                        op=operators[d['source_operators'][0]];a=values[op['inputs'][0]].to(torch.bfloat16).contiguous().view(torch.uint16).numpy().view(bfloat16)
                        native=self._orbit_gemm[op['id']]
                        try:out=native.run(a,self._orbit_gemm_weights[op['id']],op['id'])
                        finally:native.close();native.weights.clear()
                        return ready.Result({op['outputs'][0]:torch.from_numpy(out)},[native.events[-1]])
                    return original(d,values)
                node_defs.append((ready.Operation(identity,ins,outs,d['device'],call,priority=priority),deps))
                source_end.update({n:identity for n in names});record_parts[identity]=[identity]
            for op,deps in node_defs:op.dependencies.update(source_end[n] for n in deps)
            return ready.PreparedProgram([v[0] for v in node_defs],
                [v['id'] for v in source['values'] if v['role']=='input'],['prefill.logits']),record_parts
        def prefill(self,ids):
            if len(ids)!=byvalue['input.tokens']['shape'][0] or self.SEQ_BUCKET!=1:raise ValueError('Ready source requires its exact unpadded prompt shape')
            self.orbit_trace=[];self._orbit_ffn.events=[];self._orbit_ffn.call_events=[]
            begin=time.perf_counter_ns();result=ready.ReadyExecutor().run_prepared(self._r19_program,self._r19_inputs(ids));end=time.perf_counter_ns()
            record_parts=self._r19_record_parts
            self.current_context_length=len(ids);trace=result['trace'];events={e['operator']:e for e in trace['events']}
            records=[]
            for d in definitions:
                parts=[events[n] for n in record_parts[d['id']]];lo=min(e['start_ns'] for e in parts);hi=max(e['end_ns'] for e in parts)
                params=d['runtime_parameters'];price_key=digest(dict(source_id=source['source_id'],source_operators=d['source_operators'],kind=d['implementation_kind'],parameters=params,artifact_key=d['artifact_key']))
                records.append(dict(implementation_id=d['id'],source_operators=d['source_operators'],implementation_kind=d['implementation_kind'],runtime_parameters=params,
                    local_price_key=price_key,artifact_key=d['artifact_key'],start_ns=lo,end_ns=hi,wall_ns=hi-lo,events=parts))
            covered=[n for r in records for n in r['source_operators']]
            if len(covered)!=len(set(covered)) or set(covered)!=set(operators):raise RuntimeError('Incomplete actual source implementation cover')
            intervals=sorted((e['start_ns'],e['end_ns']) for e in trace['events']);union=0;left=right=None
            for lo,hi in intervals:
                if right is None:left,right=lo,hi
                elif lo<=right:right=max(right,hi)
                else:union+=right-left;left,right=lo,hi
            if right is not None:union+=right-left
            control=end-begin-union
            if control<0:raise RuntimeError('Source event union exceeds enclosing prefill')
            self._r19_last_profile=dict(schema='orbit.qwen-r19-ready-source-profile.v1',source_id=source['source_id'],plan_id=plan['plan_id'],wall_ns=end-begin,
                implementation_records=records,source_operators=len(covered),execution_policy='source-dependency-ready',source_ordering_identity=config.get('whole_source',{}).get('source_id',source['source_id']),async_runtime_proof_identity=config.get('async_runtime_proof_identity'),
                execution_trace=trace,callback_interval_union_ns=union,host_control_ns=control,exclusive_sum_equals_wall=union+control==end-begin,
                concurrent_callback_wall_sum_ns=sum(e['wall_ns'] for e in trace['events']),
                accounting='Exact callback interval union plus exclusive host control; overlapping local prices are never directly summed as request wall')
            return result['values']['prefill.logits']
        def source_profile(self):
            if self._r19_last_profile is None:raise RuntimeError('No complete ready source invocation')
            return self._r19_last_profile
    return SourceReadyPrefiller
