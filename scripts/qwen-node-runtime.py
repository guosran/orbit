#!/usr/bin/env python3
"""Source-preserving Qwen prefill operator graph and per-operator runtime hooks.

The original Qwen25Prefill._layer and LlamaPrefill.prefill execute unchanged.
Hook decisions choose existing backend implementations, never replace arithmetic.
Regions partition source operators; this interface does not fuse kernel launches.
Decode source/region implementations are a separate contract, not an opaque claim.
"""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def method(tree, owner, name):
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == owner)
    return next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == name)


def body_dump(node):
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body.pop(0)
    return ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)


# This recognizer protects the producer/consumer interpretation below. A source
# change requires an explicit new extractor, rather than reusing a hand-written
# node catalogue under a new hash.
LAYER_BODY = '''
keep = N if keep is None else keep
w = self._w[L]
h = self._rms_norm(x, w["attn_norm"], RMS_EPS)
qkv = self._matmul(h, w["qkv"], **self._wide) + w["qkv_bias"]
q, k, v = qkv.split([DQ, DK, DV], dim=1)
q = self._rope(q, self._lut[:N], N_Q_HEADS)
k = self._rope(k, self._lut[:N], N_KV_HEADS)
self._kv_store(L, k, v, keep)
a = self._attention(q, k, v, N_Q_HEADS, N_KV_HEADS, DH)
x = x + self._matmul(a, w["o"], **self._wide)
h = self._rms_norm(x, w["ffn_norm"], RMS_EPS)
g, u = self._matmul(h, w["gate_up"], **self._wide).split([INTER, INTER], dim=1)
return x + self._matmul(self._swiglu(g, u), w["down"])
'''


def extract_prefill_source(triton_root, prompt_tokens):
    if prompt_tokens <= 0 or prompt_tokens > 2048:
        raise ValueError('Require a static prompt length in [1,2048]')
    root = Path(triton_root).resolve(strict=True)
    shared = root / 'examples/llm_q4nx'
    paths = dict(qwen=shared / 'qwen25_prefill.py', llama=shared / 'llama_prefill.py')
    trees = {k: ast.parse(p.read_text()) for k, p in paths.items()}
    layer = method(trees['qwen'], 'Qwen25Prefill', '_layer')
    expected = ast.dump(ast.parse(LAYER_BODY), include_attributes=False)
    if body_dump(layer) != expected:
        raise ValueError('Qwen layer AST is outside the recognized source contract')
    constants = ast.parse((root / 'examples/qwen25_3b_q4nx/config.py').read_text())
    literals = {}
    for statement in constants.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            try:
                literals[statement.targets[0].id] = ast.literal_eval(statement.value)
            except (ValueError, TypeError):
                pass
    required = dict(D=2048, INTER=11008, N_LAYERS=36, N_Q_HEADS=16, N_KV_HEADS=2, DH=128, VOCAB=151936)
    if any(literals.get(k) != v for k, v in required.items()):
        raise ValueError('Unrecognized Qwen dimensions')
    bindings = []
    symbols = [('qwen', 'Qwen25Prefill', '_layer'), ('llama', 'LlamaPrefill', 'prefill')]
    symbols += [('llama', 'LlamaPrefill', n) for n in ['_matmul', '_rms_norm', '_swiglu', '_rope', '_attention', '_kv_store', '_lm_head']]
    for source, owner, name in symbols:
        fn = method(trees[source], owner, name)
        bindings.append(dict(id=owner + '.' + name, file=str(paths[source]), sha256=file_sha(paths[source]), symbol=owner + '.' + name))
    values, operators = [], []
    producer = {}
    def value(name, shape, dtype='f32', role='temporary'):
        values.append(dict(id=name, dtype=dtype, shape=shape, role=role, layout='row_major'))
        return name
    def op(name, kind, inputs, outputs, binding, **attributes):
        deps = list(dict.fromkeys(producer[x] for x in inputs if x in producer))
        operators.append(dict(id=name, kind=kind, binding=binding, source_ordinal=len(operators), inputs=inputs, outputs=outputs, dependencies=deps, attributes=attributes))
        for x in outputs:
            if x in producer:
                raise ValueError('Multiple source producers')
            producer[x] = name
    n = prompt_tokens
    tokens = value('input.tokens', [n], 'i64', 'input')
    embed = value('weights.embedding', [151936,2048], 'bf16', 'input')
    lut = value('input.rope_lut', [2048,128], 'f32', 'input')
    x = value('prefill.x0', [n,2048])
    op('prefill.embedding', 'embedding-gather', [tokens,embed], [x], 'LlamaPrefill.prefill', fixed_device='cpu', bucket=1)
    for layer_id in range(36):
        prefix = f'prefill.L{layer_id}.'
        weights = {}
        for role, shape, dtype in [('attn_norm',[2048],'f32'),('ffn_norm',[2048],'f32'),('qkv',[2048,2560],'bf16'),('o',[2048,2048],'bf16'),('gate_up',[2048,22016],'bf16'),('down',[11008,2048],'bf16'),('qkv_bias',[2560],'f32')]:
            weights[role] = value(f'weights.L{layer_id}.{role}',shape,dtype,'input')
        def result(role, shape):return value(prefix+role+'.value',shape)
        def emit(role, kind, ins, outs, hook=None, **attrs):
            if hook:
                attrs.update(runtime_hook=hook, supported_devices=['cpu','npu'] if hook in ('_matmul','_rms_norm','_swiglu') else ['cpu'], layer=layer_id)
                binding='LlamaPrefill.'+hook
            else:
                attrs['fixed_device']='cpu';binding='Qwen25Prefill._layer'
            op(prefix+role,kind,ins,outs,binding,**attrs)
        h = result('attn_norm',[n,2048]);emit('attn_norm','rms-norm',[x,weights['attn_norm']],[h],'_rms_norm',eps=1e-6)
        projection = result('qkv',[n,2560]);emit('qkv','matmul',[h,weights['qkv']],[projection],'_matmul',input_rounding='bf16-rne',accumulator='f32')
        biased = result('qkv_bias',[n,2560]);emit('qkv_bias','broadcast-add',[projection,weights['qkv_bias']],[biased])
        q, k, v = [result(role,[n,width]) for role,width in [('q',2048),('k',256),('v',256)]]
        for name, offset in [(q,0),(k,2048),(v,2304)]:
            next(item for item in values if item['id']==name)['view']=dict(base=biased,offsets=[0,offset],strides=[2560,1])
        emit('qkv_split','split-view',[biased],[q,k,v],axis=1,sizes=[2048,256,256],aliases_input=True,base_strides=[2560,1],offsets=[0,2048,2304])
        qr = result('q_rope',[n,2048]);emit('q_rope','neox-rope',[q,lut],[qr],'_rope',heads=16,head_dim=128)
        kr = result('k_rope',[n,256]);emit('k_rope','neox-rope',[k,lut],[kr],'_rope',heads=2,head_dim=128)
        kc = value(prefix+'kv.K.prefix',[n,256],role='output');vc = value(prefix+'kv.V.prefix',[n,256],role='output')
        emit('kv_store','kv-prefix-store',[kr,v],[kc,vc],'_kv_store',version='prefill-prefix',keep=n)
        attention = result('attention',[n,2048]);emit('attention','causal-gqa',[qr,kr,v],[attention],'_attention',query_heads=16,kv_heads=2,head_dim=128,scale=128**-.5)
        o = result('o',[n,2048]);emit('o','matmul',[attention,weights['o']],[o],'_matmul',input_rounding='bf16-rne',accumulator='f32')
        r1 = result('residual1',[n,2048]);emit('residual1','add',[x,o],[r1])
        h2 = result('ffn_norm',[n,2048]);emit('ffn_norm','rms-norm',[r1,weights['ffn_norm']],[h2],'_rms_norm',eps=1e-6)
        gu = result('gate_up',[n,22016]);emit('gate_up','matmul',[h2,weights['gate_up']],[gu],'_matmul',input_rounding='bf16-rne',accumulator='f32')
        g,u = [result(role,[n,11008]) for role in ('gate','up')]
        for name, offset in [(g,0),(u,11008)]:
            next(item for item in values if item['id']==name)['view']=dict(base=gu,offsets=[0,offset],strides=[22016,1])
        emit('gate_split','split-view',[gu],[g,u],axis=1,sizes=[11008,11008],aliases_input=True,base_strides=[22016,1],offsets=[0,11008])
        activation = result('swiglu',[n,11008]);emit('swiglu','swiglu',[g,u],[activation],'_swiglu')
        down = result('down',[n,2048]);emit('down','matmul',[activation,weights['down']],[down],'_matmul',input_rounding='bf16-rne',accumulator='f32')
        x = result('residual2',[n,2048]);emit('residual2','add',[r1,down],[x])
    final_weight = value('weights.final_norm',[2048],role='input')
    last = value('prefill.last_hidden',[1,2048]);next(item for item in values if item['id']==last)['view']=dict(base=x,offsets=[n-1,0],strides=[2048,1]);op('prefill.last_row','slice-view',[x],[last],'LlamaPrefill.prefill',fixed_device='cpu',aliases_input=True,offsets=[n-1,0],sizes=[1,2048])
    normalized = value('prefill.final_norm.value',[1,2048]);op('prefill.final_norm','rms-norm',[last,final_weight],[normalized],'LlamaPrefill._rms_norm',runtime_hook='_rms_norm',supported_devices=['cpu','npu'],eps=1e-6)
    lm = value('weights.lm_head',[151936,2048],'bf16','input');head = value('prefill.lm_head.value',[1,151936])
    op('prefill.lm_head','lm-head',[normalized,lm],[head],'LlamaPrefill._lm_head',runtime_hook='_lm_head',supported_devices=['cpu'],input_rounding='bf16-rne',output_dtype='f32')
    logits=value('prefill.logits',[151936],role='output')
    next(item for item in values if item['id']==logits)['view']=dict(base=head,offsets=[0],strides=[1])
    op('prefill.logits_row','select-row-view',[head],[logits],'LlamaPrefill.prefill',fixed_device='cpu',aliases_input=True,row=0)
    source = dict(schema='orbit.program-operator-source.v1', numerical_contract='original-triton-xdna-qwen25-prefill-bf16-input-f32-accumulation-v1',bindings=bindings,values=values,operators=operators)
    source['source_id'] = digest(source)
    return source


def validate_plan(source, plan):
    if source.get('schema') != 'orbit.program-operator-source.v1' or source['source_id'] != digest({k:v for k,v in source.items() if k!='source_id'}):
        raise ValueError('Invalid source identity')
    if plan.get('source_id') != source['source_id']:
        raise ValueError('Plan/source mismatch')
    operators = {o['id']:o for o in source['operators']}
    choices = plan.get('placements',{})
    if set(choices) != set(operators):
        raise ValueError('Every source operator needs exactly one explicit placement')
    for name, op in operators.items():
        devices = op['attributes'].get('supported_devices',[op['attributes'].get('fixed_device','cpu')])
        if choices[name] not in devices:
            raise ValueError('Unsupported source operator placement: '+name)
    regions = plan.get('regions')
    if not isinstance(regions,list) or not regions:
        raise ValueError('Explicit nonempty source partition required')
    covered=[]
    for region in regions:
        if set(region) != {'id','device','operators'} or not region['operators']:
            raise ValueError('Malformed execution region')
        if any(choices.get(name)!=region['device'] for name in region['operators']):
            raise ValueError('Region/device mismatch')
        covered.extend(region['operators'])
    if covered != [op['id'] for op in source['operators']] or len({r['id'] for r in regions})!=len(regions):
        raise ValueError('Regions must exactly partition the declared source order')
    return choices


def make_plan(source, overrides=None):
    choices = {o['id']:'cpu' for o in source['operators']}
    for name, device in (overrides or {}).items():
        if name not in choices:raise ValueError('Unknown source node '+name)
        choices[name]=device
    regions=[]
    for op in source['operators']:
        device=choices[op['id']]
        if not regions or regions[-1]['device']!=device:
            regions.append(dict(id='region'+str(len(regions)),device=device,operators=[]))
        regions[-1]['operators'].append(op['id'])
    plan=dict(schema='orbit.qwen-node-runtime-plan.v1',source_id=source['source_id'],placements=choices,regions=regions,
              launch_policy='each original operator launch retained; region is partition/accounting scope')
    validate_plan(source,plan)
    plan['plan_id']=digest(plan)
    return plan


def make_prefiller_class(base, source, plan):
    """Wrap original methods; caller loads weights normally and never frees them.

    Original tensors are retained for CPU GEMMs. NPU GEMMs use original kernel
    padding and per-node static stage_key; no global kernel/runtime patch occurs.
    """
    choices=validate_plan(source,plan)
    if plan.get('plan_id')!=digest({k:v for k,v in plan.items() if k!='plan_id'}):raise ValueError('Invalid plan identity')
    bindings={b['symbol']:b for b in source['bindings']}
    original_layer=base._layer;original_prefill=base.prefill
    for symbol, fn in [('Qwen25Prefill._layer',original_layer),('LlamaPrefill.prefill',original_prefill)]+[('LlamaPrefill.'+n,getattr(base,n)) for n in ['_matmul','_rms_norm','_swiglu','_rope','_attention','_kv_store','_lm_head']]:
        binding=bindings[symbol];path=Path(inspect.getsourcefile(fn)).resolve()
        if str(path)!=binding['file'] or file_sha(path)!=binding['sha256']:
            raise ValueError('Runtime method differs from source binding: '+symbol)
    hooks={o['id']:o for o in source['operators'] if 'runtime_hook' in o['attributes']}
    source_values={v['id']:v for v in source['values']}
    layer_ast=method(ast.parse(Path(bindings['Qwen25Prefill._layer']['file']).read_text()),'Qwen25Prefill','_layer')
    layer_statements=list(layer_ast.body)[1:]
    layer_groups=[([],0),([],1),(['attn_norm'],2),(['qkv','qkv_bias'],3),(['qkv_split'],4),(['q_rope'],5),(['k_rope'],6),(['kv_store'],7),(['attention'],8),(['o','residual1'],9),(['ffn_norm'],10),(['gate_up','gate_split'],11),(['swiglu','down','residual2'],12)]
    prefill_ast=method(ast.parse(Path(bindings['LlamaPrefill.prefill']['file']).read_text()),'LlamaPrefill','prefill')
    prefill_statements=list(prefill_ast.body)[1:]
    def descriptor(x):
        if hasattr(x,'padded') and hasattr(x,'K') and hasattr(x,'N'):
            physical=descriptor(x.padded)
            return dict(object_id=id(x),shape=list(x.shape),dtype=physical['dtype'],storage=physical['storage'],offset=physical['offset'],strides=physical['strides'],physical_shape=physical['shape'],physical_kind='resident-weight',physical_bytes=x.padded.numel()*x.padded.element_size())
        if hasattr(x,'untyped_storage'):
            return dict(object_id=id(x),shape=list(x.shape),dtype=str(x.dtype),storage=x.untyped_storage().data_ptr(),offset=x.storage_offset(),strides=list(x.stride()))
        return dict(object_id=id(x),shape=list(x.shape),dtype=str(x.dtype),storage=x.__array_interface__['data'][0],offset=0,strides=[stride//x.itemsize for stride in x.strides])
    class NodePrefiller(base):
        def __init__(self,*args,**kwargs):
            kwargs['ops']='all';super().__init__(*args,**kwargs)
            if self.n_layers!=36:raise ValueError('Whole source requires all 36 layers')
            self.orbit_trace=[];self._orbit_layer=None;self._orbit_calls={};self._orbit_values={};self._orbit_tensors={};self.orbit_source_lines={}
        def make_npu_resident(self):
            if getattr(self,'_orbit_prepared',False):return self
            import kernels
            import torch
            self.orbit_preparation=[]
            for L,weights in enumerate(self._w):
                for role in ('qkv','o','gate_up','down'):
                    name=f'prefill.L{L}.'+role;weight=weights[role]
                    if not hasattr(weight,'to'):raise ValueError('Preparation requires original BF16 source weights')
                    before=descriptor(weight)
                    if choices[name]=='npu':
                        prepared=kernels.resident_weight(weight,exact=True)
                        if kernels.shared_bo(prepared.padded) is None:
                            shared=kernels.shared_empty(prepared.padded.shape,prepared.padded.dtype)
                            if kernels.shared_bo(shared) is None:raise ValueError('Checked XRT static-weight plan requires original shared BO support')
                            shared.copy_(prepared.padded);prepared.padded=shared
                        # Transport preparation must preserve every original BF16
                        # bit. The original CPU weight may be released only after
                        # this source-value check succeeds.
                        if not torch.equal(weight.view(torch.int16),prepared.padded[:prepared.K,:prepared.N].view(torch.int16)):
                            raise ValueError('Resident preparation changed source weight bits')
                        weights[role]=prepared
                    self.orbit_preparation.append(dict(operator=name,device=choices[name],original=before,prepared=descriptor(weights[role]),original_bits_preserved=True))
            self._orbit_prepared=True
            return self
        def _bind(self,value_id,tensor):
            actual=descriptor(tensor)
            contract=source_values[value_id]
            types={'f32':{'torch.float32','float32'},'bf16':{'torch.bfloat16','bfloat16'},'i64':{'torch.int64','int64'}}
            if actual['shape']!=contract['shape'] or actual['dtype'] not in types[contract['dtype']]:
                raise ValueError('Tensor source ABI mismatch for '+value_id)
            view=contract.get('view')
            if view and view['base'] in self._orbit_tensors:
                base_value=self._orbit_tensors[view['base']]
                expected_offset=base_value['offset']+sum(a*b for a,b in zip(view['offsets'],view['strides']))
                if actual['storage']!=base_value['storage'] or actual['offset']!=expected_offset or actual['strides']!=view['strides']:
                    raise ValueError('Source view alias/offset mismatch for '+value_id)
            if value_id in self._orbit_tensors and self._orbit_tensors[value_id]!=actual:
                raise ValueError('Tensor producer/consumer mismatch for '+value_id)
            self._orbit_tensors[value_id]=actual
        def _visit(self,role,inputs):
            tick=time.perf_counter_ns()
            caller=sys._getframe(2)
            expected=original_prefill.__code__ if self._orbit_layer is None else original_layer.__code__
            if caller.f_code is not expected:raise ValueError('Hook was not called by the bound original source method')
            name='prefill.'+role if self._orbit_layer is None else f'prefill.L{self._orbit_layer}.'+role
            op=hooks[name]
            local=caller.f_locals
            prefix=f'prefill.L{self._orbit_layer}.'
            if role=='q_rope':
                for semantic,variable in [('qkv_bias','qkv'),('q','q'),('k','k'),('v','v')]:
                    self._bind(prefix+semantic+'.value',local[variable])
            if role=='ffn_norm':self._bind(prefix+'residual1.value',local['x'])
            if role=='swiglu':
                self._bind(prefix+'gate.value',local['g']);self._bind(prefix+'up.value',local['u'])
            if role=='final_norm':
                self._bind('prefill.last_hidden',inputs[0]);self._bind('weights.final_norm',inputs[1])
            if role=='lm_head':self._bind('weights.lm_head',inputs[1])
            for value_id,tensor in zip(op['inputs'],inputs):
                # LUT input is a prefix view into the global source LUT.
                if value_id=='input.rope_lut':continue
                self._bind(value_id,tensor)
            if name in self._orbit_values:raise ValueError('Repeated source hook')
            self._orbit_values[name]=True
            launches=0
            if choices[name]=='npu':
                import kernels
                if op['attributes']['runtime_hook']=='_rms_norm':launches=(inputs[0].shape[0]+kernels.ROW_TILE-1)//kernels.ROW_TILE
                elif op['attributes']['runtime_hook']=='_swiglu':
                    size=inputs[0].numel();chunk=kernels.glu_chunk(size,inputs[0].shape[-1],kernels.SWIGLU_BLOCK);launches=(size+chunk-1)//chunk
                else:launches=1  # one original static-weight NPUChain invocation
            self.orbit_trace.append(dict(operator=name,device=choices[name],launch_count=int(launches),inputs=[descriptor(x) for x in inputs],source_line=caller.f_lineno,source_symbol=expected.co_qualname,start_ns=tick))
            return name,caller.f_locals
        def _done(self,name,result):
            if result is not None:
                self.orbit_trace[-1]['output']=descriptor(result)
                self._bind(hooks[name]['outputs'][0],result)
            self.orbit_trace[-1]['inclusive_call_ns']=time.perf_counter_ns()-self.orbit_trace[-1]['start_ns']
            return result
        def _layer(self,x,L,N,keep=None):
            if self._orbit_layer is not None:raise ValueError('Nested layer execution')
            self._orbit_layer=L;self._orbit_calls={'rms':0,'matmul':0,'rope':0};self._orbit_values={}
            self._bind('prefill.x0' if L==0 else f'prefill.L{L-1}.residual2.value',x)
            for role,w in self._w[L].items():
                if role in ('attn_norm','ffn_norm','qkv','o','gate_up','down','qkv_bias'):self._bind(f'weights.L{L}.'+role,w)
            try:
                out=original_layer(self,x,L,N,keep=keep)
                self._bind(f'prefill.L{L}.residual2.value',out)
                if self._orbit_calls!={'rms':2,'matmul':4,'rope':2}:raise ValueError('Incomplete source layer hooks')
                return out
            finally:self._orbit_layer=None
        def prefill(self,ids):
            wall_start=time.perf_counter_ns()
            if len(ids)!=next(v['shape'][0] for v in source['values'] if v['id']=='input.tokens'):raise ValueError('Input/source length mismatch')
            if self.SEQ_BUCKET!=1:raise ValueError('Source requires bucket1')
            if self._kv_sink is not None:raise ValueError('This source implementation owns host KV; direct KV sink needs a separate implementation contract')
            self.orbit_trace=[];self._orbit_values={};self._orbit_tensors={};self.orbit_source_lines={}
            if sys.gettrace() is not None:raise ValueError('Existing trace hook conflicts with source timing instrumentation')
            active={};stack=[]
            def charge(key,now):
                line,start,layer,code=active[key]
                if start is not None:
                    slot=(code.co_name,line,layer)
                    self.orbit_source_lines[slot]=self.orbit_source_lines.get(slot,0)+now-start
            def trace(frame,event,arg):
                if frame.f_code not in (original_prefill.__code__,original_layer.__code__):return None
                now=time.perf_counter_ns();key=id(frame)
                if event=='call':
                    if stack:
                        parent=stack[-1];charge(parent,now)
                        line,start,layer,code=active[parent];active[parent]=(line,None,layer,code)
                    stack.append(key)
                elif key in active:charge(key,now)
                if event=='return':
                    active.pop(key,None)
                    if not stack or stack.pop()!=key:raise ValueError('Invalid source trace nesting')
                    if stack:
                        parent=stack[-1];line,start,layer,code=active[parent];active[parent]=(line,now,layer,code)
                else:active[key]=(frame.f_lineno,now,frame.f_locals.get('L') if frame.f_code is original_layer.__code__ else None,frame.f_code)
                return trace
            sys.settrace(trace)
            try:out=original_prefill(self,ids)
            finally:sys.settrace(None)
            if len(self.orbit_trace)!=362+36:raise ValueError('Incomplete complete-prefill source hook coverage')
            self.orbit_prefill_wall_ns=time.perf_counter_ns()-wall_start
            return out
        def source_profile(self):
            calls={row['operator']:row['inclusive_call_ns'] for row in self.orbit_trace}
            costs={}
            def group(names,statement,method_name,layer):
                total=sum(ns for (fn,line,owner),ns in self.orbit_source_lines.items() if fn==method_name and owner==layer and statement.lineno<=line<=statement.end_lineno)
                called=[name for name in names if name in calls];inline=[name for name in names if name not in calls]
                if not inline:
                    if len(names)!=1:raise ValueError('Ambiguous source statement accounting')
                    costs[names[0]]=total
                else:
                    if len(inline)!=1:raise ValueError('A source statement needs an explicitly separated single inline owner')
                    remainder=total-sum(calls[name] for name in called)
                    if remainder<0:raise ValueError('Source trace/call clocks do not admit exclusive accounting')
                    costs.update({name:calls[name] for name in called});costs[inline[0]]=remainder
            for L in range(36):
                for roles,index in layer_groups:
                    if roles:group([f'prefill.L{L}.'+role for role in roles],layer_statements[index],'_layer',L)
            # Allocate source embedding zero/fill/gather together; no arithmetic
            # replacement is made. The final slice views are separately owned.
            embedding=0
            for index in (4,5):
                stmt=prefill_statements[index]
                embedding+=sum(ns for (fn,line,owner),ns in self.orbit_source_lines.items() if fn=='prefill' and owner is None and stmt.lineno<=line<=stmt.end_lineno)
            costs['prefill.embedding']=embedding
            group(['prefill.final_norm','prefill.last_row'],prefill_statements[8],'prefill',None)
            group(['prefill.lm_head','prefill.logits_row'],prefill_statements[9],'prefill',None)
            if set(costs)!={op['id'] for op in source['operators']}:raise ValueError('Missing source-node timing')
            residual=self.orbit_prefill_wall_ns-sum(costs.values())
            if residual<0:raise ValueError('Source node/control phase accounting overlaps')
            return dict(source_id=source['source_id'],plan_id=plan['plan_id'],tensor_bindings=self._orbit_tensors,call_trace=self.orbit_trace,node_cost_ns=costs,node_launch_counts={op['id']:next((row['launch_count'] for row in self.orbit_trace if row['operator']==op['id']),0) for op in source['operators']},host_control_ns=residual,wall_ns=self.orbit_prefill_wall_ns,exclusive_sum_equals_wall=sum(costs.values())+residual==self.orbit_prefill_wall_ns,source_line_policy='parent bound frames paused while bound child source executes; hook calls already included in statement ledger',device_subphase_policy='original synchronous implementation owns launch, BO sync, wait and output copies; these are not separately measured or charged',instrumentation='same source-line and tensor-provenance hooks required for baseline and selected comparisons')
        def _rms_norm(self,x,w,eps,backend=None):
            if self._orbit_layer is None:role='final_norm'
            else:
                i=self._orbit_calls['rms'];role=['attn_norm','ffn_norm'][i];self._orbit_calls['rms']+=1
                if w is not self._w[self._orbit_layer][role]:raise ValueError('Norm weight/source identity mismatch')
            name,local=self._visit(role,[x,w]);return self._done(name,base._rms_norm(self,x,w,eps,backend=choices[name]))
        def _matmul(self,x,w,backend=None,**mm):
            role=['qkv','o','gate_up','down'][self._orbit_calls['matmul']];self._orbit_calls['matmul']+=1
            if w is not self._w[self._orbit_layer][role]:raise ValueError('Projection weight/source identity mismatch')
            name,local=self._visit(role,[x,w]);device=choices[name]
            if device=='cpu' and not hasattr(w,'to'):raise ValueError('CPU source requires original tensor, not ResidentWeight')
            if device=='npu':mm['stage_key']='orbit_'+name
            return self._done(name,base._matmul(self,x,w,backend=device,**mm))
        def _swiglu(self,g,u,backend=None):
            name,local=self._visit('swiglu',[g,u])
            if local.get('g') is not g or local.get('u') is not u:raise ValueError('SwiGLU source values mismatch')
            if g.untyped_storage().data_ptr()!=u.untyped_storage().data_ptr():raise ValueError('Expected original gate/up split views')
            return self._done(name,base._swiglu(self,g,u,backend=choices[name]))
        def _rope(self,x,lut,heads,backend=None):
            role=['q_rope','k_rope'][self._orbit_calls['rope']];self._orbit_calls['rope']+=1
            name,local=self._visit(role,[x,lut]);return self._done(name,base._rope(self,x,lut,heads,backend='cpu'))
        def _attention(self,q,k,v,nq,nkv,dh,window=None,scale=None,backend=None):
            name,local=self._visit('attention',[q,k,v]);return self._done(name,base._attention(self,q,k,v,nq,nkv,dh,window=window,scale=scale,backend='cpu'))
        def _kv_store(self,L,k,v,keep):
            if L!=self._orbit_layer:raise ValueError('KV layer/source mismatch')
            name,local=self._visit('kv_store',[k,v]);out=base._kv_store(self,L,k,v,keep)
            prefix=f'prefill.L{L}.'
            self._bind(prefix+'kv.K.prefix',self.kv_k[L][:keep]);self._bind(prefix+'kv.V.prefix',self.kv_v[L][:keep])
            return self._done(name,out)
        def _lm_head(self,x,w,backend=None):
            name,local=self._visit('lm_head',[x,w]);return self._done(name,base._lm_head(self,x,w,backend='cpu'))
    return NodePrefiller


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--triton-root',type=Path,required=True);parser.add_argument('--prompt-tokens',type=int,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=extract_prefill_source(args.triton_root,args.prompt_tokens)
    with args.output.open('x') as stream:stream.write(json.dumps(source,indent=2)+'\n')
    print(json.dumps(dict(source_id=source['source_id'],operators=len(source['operators']),variable_nodes=sum(o['attributes'].get('supported_devices')==['cpu','npu'] for o in source['operators']))))

if __name__=='__main__':main()
