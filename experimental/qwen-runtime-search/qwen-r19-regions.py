#!/usr/bin/env python3
"""Executable source-bound CPU regions with arbitrary legal fusion/fission cuts.

``eager-host-group`` changes the host dispatch boundary only. It is never
reported as device fusion. ``inductor-cpu`` delegates the actual tensor DAG to
TorchInductor; its generated kernels, not this wrapper, establish which tensor
operations fused. A region can span any supported connected convex source DAG,
including branches and exported views. KV mutation stays an explicit boundary.

All staging here is real CPU tensor copying. This module does not manufacture
NPU BO residency, DMA overlap, launch reduction or SRAM claims.
"""
import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import time
import types


_COMPILED_TEMPLATES={}
_GENERATED_CODE={}


KINDS = {'rms-norm', 'matmul', 'broadcast-add', 'add', 'split-view',
         'swiglu', 'neox-rope', 'causal-gqa'}


class Program:
    """Own an immutable-by-interface source snapshot and validate bindings once."""
    def __init__(self,source):
        self._source=copy.deepcopy(source)
        self._checked=checked_source(self._source)

    def __getitem__(self,key):
        return self._source[key]


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                     allow_nan=False).encode()).hexdigest()


def file_identity(path):
    path=Path(path).resolve(strict=True)
    return dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def require(condition,message):
    if not condition:
        raise ValueError(message)


def checked_source(source):
    """Check immutable source/binding identities before admitting a lowering."""
    if isinstance(source,Program):return source._checked
    require(source['source_id']==digest({k:v for k,v in source.items() if k!='source_id'}),
            'Source identity mismatch')
    require(source.get('numerical_contract')==
            'original-triton-xdna-qwen25-prefill-bf16-input-f32-accumulation-v1',
            'This lowering requires the original Qwen CPU numerical contract')
    operators={op['id']:op for op in source['operators']}
    values={value['id']:value for value in source['values']}
    require(len(operators)==len(source['operators']) and len(values)==len(source['values']),
            'Repeated source operator/value identity')
    bindings={binding['id']:binding for binding in source['bindings']}
    checked=set()
    for binding in bindings.values():
        key=(binding['file'],binding['sha256'])
        if key not in checked:
            require(file_identity(binding['file'])['sha256']==binding['sha256'],
                    'Original source binding changed: '+binding['file'])
            checked.add(key)
    return operators,values,bindings


def region(source,operator_ids):
    """Construct one connected convex cover and derive its complete boundary.

    Convexity forbids a path that leaves the cover and later reenters it. Such
    a cover would deadlock if replaced by one executable task, even when its
    selected operators appear close together in a source listing.
    """
    operators,values,bindings=checked_source(source)
    chosen=set(operator_ids)
    require(chosen and len(chosen)==len(operator_ids),'Empty or repeated cover')
    require(chosen<=operators.keys(),'Unknown covered source operator')
    ordered=sorted(chosen,key=lambda name:operators[name]['source_ordinal'])
    producer={value:op['id'] for op in operators.values() for value in op['outputs']}
    consumers={value:set() for value in values}
    successors={name:set() for name in operators}
    predecessors={name:set() for name in operators}
    for op in operators.values():
        for value in op['inputs']:
            require(value in values,'Undefined source input')
            consumers[value].add(op['id'])
            if value in producer:
                successors[producer[value]].add(op['id'])
                predecessors[op['id']].add(producer[value])
        for before in op.get('dependencies',[]):
            require(before in operators,'Undefined control predecessor')
            successors[before].add(op['id']);predecessors[op['id']].add(before)
    for name in chosen:
        op=operators[name]
        require(op['kind'] in KINDS,'Unsupported or effectful operator: '+name)
        require(op['binding'] in bindings,'Missing original binding')
        attributes=op.get('attributes',{})
        require(attributes.get('fixed_device','cpu')=='cpu','CPU region cannot override fixed device')
        require('cpu' in attributes.get('supported_devices',['cpu']),'CPU lowering unavailable')
        if op['kind']=='matmul':
            require(attributes.get('input_rounding')=='bf16-rne' and
                    attributes.get('accumulator')=='f32','Unsupported matmul rounding')
        if op['kind']=='split-view':
            require(attributes.get('aliases_input') and attributes.get('axis')==1,
                    'Split requires the original readonly alias contract')
        if op['kind']=='causal-gqa':
            require(attributes['query_heads']%attributes['kv_heads']==0,
                    'Invalid grouped-query geometry')
    reached={ordered[0]};pending=[ordered[0]]
    while pending:
        name=pending.pop()
        for neighbor in (successors[name]|predecessors[name])&chosen:
            if neighbor not in reached:reached.add(neighbor);pending.append(neighbor)
    require(reached==chosen,'A region must be connected by actual source edges')
    # Reachability outside a region may only continue outward, never reenter.
    outside={n for name in chosen for n in successors[name] if n not in chosen}
    pending=list(outside)
    while pending:
        name=pending.pop()
        require(not(successors[name]&chosen),'Non-convex cover crosses an external dependency')
        for child in successors[name]-chosen-outside:
            outside.add(child);pending.append(child)
    seen=set()
    for name in ordered:
        require((predecessors[name]&chosen)<=seen,'Source ordinals are not topological')
        seen.add(name)
    produced={value for name in chosen for value in operators[name]['outputs']}
    inputs=list(dict.fromkeys(value for name in ordered for value in operators[name]['inputs']
                              if value not in produced))
    outputs=[value for name in ordered for value in operators[name]['outputs']
             if consumers[value]-chosen or values[value]['role']=='output']
    require(outputs,'Region has no observable output')
    covered_output_producers={value:producer[value] for value in outputs}
    aliases=[]
    for value in outputs:
        op=operators[covered_output_producers[value]]
        if op['kind']=='split-view':
            aliases.append(dict(value=value,base=op['inputs'][0],
                                part=op['outputs'].index(value),axis=1,
                                sizes=op['attributes']['sizes'],
                                expected_strides=op['attributes']['base_strides'],
                                expected_offset=op['attributes']['offsets'][op['outputs'].index(value)]))
    proof=dict(schema='orbit.qwen-r19-cpu-region.v1',source_id=source['source_id'],
        operators=ordered,input_values=inputs,output_values=outputs,
        input_specs={value:values[value] for value in inputs},
        output_specs={value:values[value] for value in outputs},
        source_bindings=[bindings[name] for name in sorted({operators[n]['binding'] for n in chosen})],
        semantics='original CPU tensor operations in source order; compiled arithmetic requires numerical admission',
        effects='readonly inputs; internal temporary ownership; exported split views retain bases',
        exported_views=aliases,device='cpu',npu_launches=0,
        transfer='no implicit chip transfer or NPU BO alias',
        convex=True,connected=True)
    proof['region_id']=digest(proof)
    return proof


def enumerate_regions(source,max_operators,min_operators=2):
    """Enumerate legal covers from graph adjacency, without operator templates.

    max_operators is an explicit neighborhood granularity, not a search timer
    or evaluation cap. Setting it to the supported graph size enumerates the
    complete finite connected-cover domain, which can be exponentially large.
    The generator emits each exact cover once and prunes non-convex covers.
    """
    require(1<=min_operators<=max_operators,'Invalid region size range')
    program=source if isinstance(source,Program) else Program(source)
    operators,_,_=checked_source(program)
    supported={name for name,op in operators.items() if op['kind'] in KINDS and
               op.get('attributes',{}).get('fixed_device','cpu')=='cpu' and
               'cpu' in op.get('attributes',{}).get('supported_devices',['cpu'])}
    producer={v:op['id'] for op in operators.values() for v in op['outputs']}
    adjacency={name:set() for name in supported}
    for name in supported:
        op=operators[name]
        before=set(op.get('dependencies',[]))|{producer[v] for v in op['inputs'] if v in producer}
        for other in before&supported:
            adjacency[name].add(other);adjacency[other].add(name)
    rank={name:op['source_ordinal'] for name,op in operators.items()}
    frontier={frozenset([name]) for name in supported}
    for size in range(1,min(max_operators,len(supported))+1):
        following=set()
        for cover in sorted(frontier,key=lambda c:tuple(sorted(rank[n] for n in c))):
            try:proof=region(program,list(cover))
            except ValueError:continue
            if size>=min_operators:yield proof
            if size<max_operators:
                for neighbor in set().union(*(adjacency[n] for n in cover))-cover:
                    following.add(cover|{neighbor})
        frontier=following
        if not frontier:break


def fission(source,cover,partitions):
    """Split a checked cover into exact disjoint legal subregions."""
    parent=region(source,cover)
    flat=[name for part in partitions for name in part]
    require(len(flat)==len(set(flat)) and set(flat)==set(cover),
            'Fission must preserve exact source coverage without duplicates')
    children=[region(source,part) for part in partitions]
    ordinals={op['id']:op['source_ordinal'] for op in source['operators']}
    children.sort(key=lambda child:min(ordinals[n] for n in child['operators']))
    owner={name:i for i,child in enumerate(children) for name in child['operators']}
    producer={v:op['id'] for op in source['operators'] for v in op['outputs']}
    for index,child in enumerate(children):
        for value in child['input_values']:
            if producer.get(value) in owner:
                require(owner[producer[value]]<index,'Fission introduces cyclic region dependencies')
    result=dict(schema='orbit.qwen-r19-cpu-fission.v1',source_id=source['source_id'],
                parent_region_id=parent['region_id'],regions=children,
                input_values=parent['input_values'],output_values=parent['output_values'],
                inter_region_residence='host materialized tensors; exported views retain their base',
                npu_launches=0)
    result['plan_id']=digest(result)
    return result


def registration(executable,compile_proof=None):
    """Export a pending backend choice; whole-Qwen gates/prices admit it later."""
    require(executable.backend!='inductor-cpu' or compile_proof,
            'Inductor registration requires actual generated-code evidence')
    proof=executable.proof
    producer=file_identity(__file__)
    artifacts=(compile_proof['generated_code'] if compile_proof else [producer])
    artifact=digest(dict(producer=producer,artifacts=artifacts,
                         backend=executable.backend,torch_version=executable.torch.__version__))
    runtime=dict(operators=proof['operators'],backend=executable.backend,
                 input_policy=executable.input_policy,output_policy=executable.output_policy,
                 region_id=proof['region_id'],template_id=executable.template_id)
    identity=digest(dict(source_id=proof['source_id'],runtime=runtime,artifact=artifact))
    # Heap buffers are read from the actual compiled wrapper when available;
    # eager mode conservatively reserves every produced source tensor. Source
    # ABI inputs/outputs remain independently graph-accounted, so this may
    # overcount outputs rather than silently omit a private compiler buffer.
    internal=executable._managed_source_temporary_bytes
    if compile_proof:
        internal=compile_proof['generated_kernel_audit']['managed_heap_allocation_bytes']
    copies=0
    if executable.input_policy=='stage-host-copy':
        copies+=sum(math.prod(spec['shape'])*(2 if spec['dtype']=='bf16' else 4)
                    for spec in proof['input_specs'].values())
    if executable.output_policy=='copy-host':
        copies+=sum(math.prod(spec['shape'])*(2 if spec['dtype']=='bf16' else 4)
                    for spec in proof['output_specs'].values())
    return dict(id='r19-cpu-region-'+identity,source_id=proof['source_id'],
        operators=proof['operators'],input_values=proof['input_values'],output_values=proof['output_values'],
        kind='prefill-cpu-region',artifact_identity=artifact,entrypoint='ExecutableRegion.run',
        device='cpu',resource='cpu.host',memory='runtime-abi',runtime_config=runtime,
        wall_ms=None,executable=False,numeric_admitted=False,
        admission_status='compiled-pending-unchanged-complete-Qwen-gate-and-region-call-price',
        launch_count=0,cost_boundary='inclusive-region-wall',
        transport_ownership='implementation-inclusive',visibility_policy='host-materialized',
        execution_variant=dict(runtime_context_identity=executable.template_id,
            dataflow='CPU native region' if compile_proof else 'CPU eager host group',
            residence_policy=executable.input_policy+'/'+executable.output_policy,
            resource_demands={'cpu.host':1},resource_shape=[],replicas=1,
            work_distribution='single-invocation',temporary_bytes={'runtime-abi':internal+copies},
            resident_allocations={}),
        source_proof=proof,compile_proof=compile_proof,producer=producer)


def _tensor_operation(torch,op,args):
    """Match original CPU helpers; compilation may fuse, not reassociate IR."""
    kind=op['kind'];a=op.get('attributes',{})
    if kind=='matmul':
        return (args[0].to(torch.bfloat16).to(torch.float32) @ args[1].to(torch.float32),)
    if kind in ('add','broadcast-add'):
        return (args[0]+args[1],)
    if kind=='split-view':
        return tuple(args[0].split(a['sizes'],dim=a['axis']))
    if kind=='swiglu':
        return (torch.nn.functional.silu(args[0])*args[1],)
    if kind=='rms-norm':
        x=args[0].to(torch.float32)
        if hasattr(torch.nn.functional,'rms_norm'):
            return (torch.nn.functional.rms_norm(x,(x.shape[-1],),args[1],a['eps']),)
        inv=torch.rsqrt((x*x).mean(-1,keepdim=True)+a['eps'])
        return (x*inv*args[1],)
    if kind=='neox-rope':
        x,lut=args;lut=lut[:x.shape[0]];half=lut.shape[-1]//2
        cos=lut[:,:half].unsqueeze(1);sin=lut[:,half:].unsqueeze(1)
        v=x.reshape(x.shape[0],a['heads'],2*half)
        x1,x2=v[...,:half],v[...,half:]
        return (torch.cat([x1*cos-x2*sin,x1*sin+x2*cos],dim=-1).reshape(x.shape[0],-1),)
    if kind=='causal-gqa':
        q,k,v=args;n=q.shape[0];nq=a['query_heads'];nkv=a['kv_heads'];dh=a['head_dim']
        qh=q.reshape(n,nq,dh).transpose(0,1)
        kh=k.reshape(n,nkv,dh).transpose(0,1).repeat_interleave(nq//nkv,dim=0)
        vh=v.reshape(n,nkv,dh).transpose(0,1).repeat_interleave(nq//nkv,dim=0)
        out=torch.empty((nq,n,dh),dtype=qh.dtype,device=qh.device)
        for m0 in range(0,n,128):
            m1=min(m0+128,n)
            scores=(qh[:,m0:m1]@kh[:,:m1].transpose(1,2))*a['scale']
            i=torch.arange(m0,m1,device=scores.device)[:,None]
            j=torch.arange(m1,device=scores.device)[None,:]
            scores=scores.masked_fill(~(j<=i),float('-inf'))
            out[:,m0:m1]=torch.softmax(scores,dim=-1)@vh[:,:m1]
        return (out.transpose(0,1).reshape(n,nq*dh),)
    raise ValueError('Unsupported original operator: '+kind)


class ExecutableRegion:
    """One callable ABI, with observable staging/copy and ownership receipts."""
    def __init__(self,source,cover,backend='eager-host-group',input_policy='borrow-host',
                 output_policy='retain-host'):
        require(backend in ('eager-host-group','inductor-cpu'),'Unknown CPU backend')
        require(input_policy in ('borrow-host','stage-host-copy'),'Unknown input residency policy')
        require(output_policy in ('retain-host','copy-host'),'Unknown output residency policy')
        self.proof=region(source,cover)
        require(output_policy!='copy-host' or not self.proof['exported_views'],
                'Independent output copies would break an exported source view alias')
        self.backend=backend;self.input_policy=input_policy;self.output_policy=output_policy
        operators={op['id']:op for op in source['operators']}
        ops=[operators[name] for name in self.proof['operators']]
        self._operation_kinds=[op['kind'] for op in ops]
        self._managed_source_temporary_bytes=sum(
            math.prod(value['shape'])*(2 if value['dtype']=='bf16' else 4)
            for value in source['values'] if value['id'] in {v for op in ops for v in op['outputs']})
        specs={value['id']:value for value in source['values']}
        for op in ops:
            first=specs[op['inputs'][0]];elements=math.prod(first['shape'])
            if op['kind']=='matmul':
                self._managed_source_temporary_bytes+=6*elements+4*math.prod(specs[op['inputs'][1]]['shape'])
            elif op['kind'] in ('rms-norm','swiglu','neox-rope'):
                self._managed_source_temporary_bytes+=16*elements
            elif op['kind']=='causal-gqa':
                n=first['shape'][0];heads=op['attributes']['query_heads']
                self._managed_source_temporary_bytes+=16*elements+16*heads*n*n
        names=self.proof['input_values'];outputs=self.proof['output_values']
        # Return private bases solely to validate the physical view ABI. They
        # do not become new observable source outputs. Holding a Python view
        # already retains its base; this makes that ownership explicit to the
        # checker and prevents a compiler from silently materializing the view.
        self._kernel_outputs=outputs+list(dict.fromkeys(
            alias['base'] for alias in self.proof['exported_views'] if alias['base'] not in outputs))
        # Tensor programs are indexed structurally, not by layer-specific SSA
        # strings. Identical covers in 36 layers then share Dynamo/Inductor
        # geometry guards instead of exhausting its recompilation limit.
        positions={name:i for i,name in enumerate(names)}
        plan=[]
        arithmetic_keys={'eps','heads','head_dim','query_heads','kv_heads','scale','axis','sizes'}
        for op in ops:
            ins=[positions[name] for name in op['inputs']]
            outs=[]
            for name in op['outputs']:
                positions[name]=len(positions);outs.append(positions[name])
            canonical=dict(kind=op['kind'],attributes={key:value for key,value in op.get('attributes',{}).items()
                                                     if key in arithmetic_keys})
            plan.append((canonical,ins,outs))
        output_positions=[positions[name] for name in self._kernel_outputs]
        number_values=len(positions)
        uses={positions[v]:sum(op['inputs'].count(v) for op in ops)
              for op in ops for v in op['outputs']}
        import torch
        self.torch=torch
        def kernel(*arguments):
            values=list(arguments)+[None]*(number_values-len(arguments));remaining=dict(uses)
            for op,ins,outs in plan:
                result=_tensor_operation(torch,op,[values[v] for v in ins])
                for index,value in zip(outs,result):values[index]=value
                # Internal ownership really ends at the final use in eager
                # lowering. Compiler lowering performs its own buffer planning.
                for index in ins:
                    if index in remaining:
                        remaining[index]-=1
                        if remaining[index]==0 and index not in output_positions:values[index]=None
            return tuple(values[index] for index in output_positions)
        self.eager=kernel
        self.template_id=digest(dict(plan=plan,outputs=output_positions,
            inputs=[dict(shape=spec['shape'],dtype=spec['dtype']) for spec in self.proof['input_specs'].values()]))
        if backend=='inductor-cpu':
            if self.template_id not in _COMPILED_TEMPLATES:
                # One code object per structural template avoids Dynamo's
                # per-code recompilation cap without changing global settings.
                specialized=types.FunctionType(kernel.__code__.replace(co_name='region_'+self.template_id),
                    kernel.__globals__,kernel.__name__,kernel.__defaults__,kernel.__closure__)
                _COMPILED_TEMPLATES[self.template_id]=torch.compile(
                    specialized,backend='inductor',fullgraph=True,dynamic=False)
            self.kernel=_COMPILED_TEMPLATES[self.template_id]
        else:self.kernel=kernel
        self.events=[]
        self.generated_kernel_audit=None

    def _check_outputs(self,result):
        values=dict(zip(self._kernel_outputs,result))
        require(len(result)==len(self._kernel_outputs),'Compiled result arity mismatch')
        witnesses=[]
        for alias in self.proof['exported_views']:
            value=values[alias['value']];base=values[alias['base']]
            shared=value.untyped_storage().data_ptr()==base.untyped_storage().data_ptr()
            require(shared,'Lowering materialized an exported source view')
            require(list(value.stride())==alias['expected_strides'],'Exported source view strides changed')
            require(value.storage_offset()==base.storage_offset()+alias['expected_offset'],
                    'Exported source view offset changed')
            witnesses.append(dict(value=alias['value'],base=alias['base'],same_physical_storage=shared,
                strides=list(value.stride()),relative_offset=value.storage_offset()-base.storage_offset()))
        return {name:values[name] for name in self.proof['output_values']},witnesses

    def prepare(self,inputs,output_directory):
        """Compile/warm outside request timing and save actual generated code.

        Compiled code is evidence of what Inductor emitted. Registration still
        needs numerical gates and local complete-region measurements; a Python
        callable or fused-looking FX graph is not a device-kernel-count proof.
        """
        require(self.backend=='inductor-cpu','Generated-code evidence requires Inductor')
        from torch._inductor.graph import GraphLowering
        from unittest import mock
        output_directory=Path(output_directory)
        output_directory.mkdir(parents=True,exist_ok=False)
        args=[inputs[name].clone() if self.input_policy=='stage-host-copy' else inputs[name]
              for name in self.proof['input_values']]
        signature=digest(dict(template=self.template_id,strides=[list(value.stride()) for value in args]))
        code=[]
        with self.torch.inference_mode():
            # run_and_get_code() resets Dynamo globally. That would invalidate
            # earlier prepared regions and move compilation into request time.
            # Capture the same emission hook without resetting other kernels.
            with mock.patch.object(GraphLowering,'save_output_code',lambda text:code.append(text)):
                result=self.kernel(*args)
        visible,_=self._check_outputs(result)
        if code:_GENERATED_CODE[signature]=list(code)
        else:code=_GENERATED_CODE.get(signature,[])
        require(code,'No actual compiler-generated code was captured for this specialization')
        records=[]
        native_calls=[];external_calls=[];allocations=[]
        for index,text in enumerate(code):
            path=output_directory/f'generated-{index}.py'
            path.write_text(text);records.append(file_identity(path))
            tree=ast.parse(text)
            native_names={target.id for node in ast.walk(tree) if isinstance(node,ast.Assign)
                and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute)
                and node.value.func.attr=='cpp_pybinding' for target in node.targets if isinstance(target,ast.Name)}
            for method in [n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='call']:
                for node in ast.walk(method):
                    if not isinstance(node,ast.Call):continue
                    if isinstance(node.func,ast.Name) and node.func.id in native_names:
                        native_calls.append(node.func.id)
                    if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='extern_kernels':
                        external_calls.append(node.func.attr)
                    if isinstance(node.func,ast.Name) and node.func.id=='empty_strided_cpu':
                        shape=ast.literal_eval(node.args[0]);strides=ast.literal_eval(node.args[1])
                        dtype=node.args[2]
                        require(isinstance(dtype,ast.Attribute) and dtype.attr in ('float32','bfloat16','int64','bool'),
                                'Unknown generated allocation dtype')
                        width={'float32':4,'bfloat16':2,'int64':8,'bool':1}[dtype.attr]
                        size=(1+sum((extent-1)*stride for extent,stride in zip(shape,strides)))*width if math.prod(shape) else 0
                        allocations.append(dict(shape=list(shape),strides=list(strides),dtype=dtype.attr,bytes=size))
        # The proof explicitly counts source arithmetic operators separately
        # from views; folding a view around one operation is not evidence of
        # inter-operation device fusion.
        arithmetic=sum(kind!='split-view' for kind in self._operation_kinds)
        self.generated_kernel_audit=dict(native_kernel_calls=native_calls,external_kernel_calls=external_calls,
            managed_heap_allocations=allocations,managed_heap_allocation_bytes=sum(row['bytes'] for row in allocations),
            source_arithmetic_operators=arithmetic,
            native_fusion_established=len(native_calls)==1 and not external_calls and arithmetic>=2,
            evidence='AST of actual generated wrapper call method, not a host-region name')
        proof=dict(schema='orbit.qwen-r19-inductor-compile.v1',region=self.proof,
            backend=self.backend,torch_version=self.torch.__version__,
            template_id=self.template_id,specialization_id=signature,
            producer=file_identity(__file__),generated_code=records,
            generated_kernel_audit=self.generated_kernel_audit,
            device_fusion_claim=self.generated_kernel_audit['native_fusion_established'],
            admission='compiled-pending-generated-kernel-audit-and-numerical-gate')
        (output_directory/'compile.json').write_text(json.dumps(proof,indent=2)+'\n')
        return visible,proof

    def run(self,inputs):
        torch=self.torch;begin=time.perf_counter_ns()
        require(set(inputs)==set(self.proof['input_values']),'Region input ABI mismatch')
        for name,spec in self.proof['input_specs'].items():
            value=inputs[name]
            require(isinstance(value,torch.Tensor) and value.device.type=='cpu','CPU tensor input required')
            require(list(value.shape)==spec['shape'],'Source input shape mismatch: '+name)
            require(str(value.dtype).split('.')[-1]=={'f32':'float32','bf16':'bfloat16'}[spec['dtype']],
                    'Source input dtype mismatch: '+name)
        copied_in=0;args=[]
        for name in self.proof['input_values']:
            value=inputs[name]
            if self.input_policy=='stage-host-copy':
                copied_in+=value.numel()*value.element_size();value=value.clone()
            args.append(value)
        staged=time.perf_counter_ns()
        with torch.inference_mode():result=self.kernel(*args)
        output,alias_witnesses=self._check_outputs(result)
        executed=time.perf_counter_ns()
        copied_out=0
        if self.output_policy=='copy-host':
            copied_out=sum(value.numel()*value.element_size() for value in output.values())
            output={name:value.clone() for name,value in output.items()}
        for name,spec in self.proof['output_specs'].items():
            require(list(output[name].shape)==spec['shape'],'Lowered output shape mismatch')
            require(str(output[name].dtype).split('.')[-1]=={'f32':'float32','bf16':'bfloat16'}[spec['dtype']],
                    'Lowered output dtype mismatch')
        end=time.perf_counter_ns()
        event=dict(region_id=self.proof['region_id'],backend=self.backend,
            source_operators=self.proof['operators'],input_policy=self.input_policy,
            output_policy=self.output_policy,wall_ns=end-begin,
            validation_and_staging_ns=staged-begin,kernel_inclusive_ns=executed-staged,
            output_copy_and_validation_ns=end-executed,
            copied_input_bytes=copied_in,copied_output_bytes=copied_out,
            exported_view_witnesses=alias_witnesses,
            chip_transfer_bytes=0,npu_launches=0,host_callable_invocations=1,
            generated_kernel_audit=self.generated_kernel_audit,
            device_fusion_claim=bool(self.generated_kernel_audit and self.generated_kernel_audit['native_fusion_established']),
            compiler_fusion_status=('requires generated-kernel audit; host grouping alone is no proof'
                                    if self.backend=='inductor-cpu' else 'host grouping only'),
            exclusive_sum_equals_wall=True)
        self.events.append(event)
        return output


class ExecutableFission:
    """Execute a real cut plan, retaining only values needed across boundaries."""
    def __init__(self,source,cover,partitions,**lowering):
        self.proof=fission(source,cover,partitions)
        self.regions=[ExecutableRegion(source,child['operators'],**lowering)
                      for child in self.proof['regions']]
        self.events=[]

    def run(self,inputs):
        require(set(inputs)==set(self.proof['input_values']),'Fission input ABI mismatch')
        values=dict(inputs);boundaries=[];begin=time.perf_counter_ns()
        for index,child in enumerate(self.regions):
            result=child.run({name:values[name] for name in child.proof['input_values']})
            values.update(result)
            required=set(self.proof['output_values'])
            for later in self.regions[index+1:]:required.update(later.proof['input_values'])
            for name in list(values):
                if name not in required:del values[name]
            boundaries.append(child.events[-1])
        result={name:values[name] for name in self.proof['output_values']}
        self.events.append(dict(plan_id=self.proof['plan_id'],wall_ns=time.perf_counter_ns()-begin,
            host_callable_invocations=len(self.regions),npu_launches=0,
            actual_cut_outputs=[child.proof['output_values'] for child in self.regions],
            regions=boundaries,device_fusion_claim=False))
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--operators',required=True,help='Comma-separated exact source operator IDs')
    parser.add_argument('--partitions',help='Semicolon-separated fission covers, each comma-separated')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=json.loads(args.source.read_text());cover=args.operators.split(',')
    proof=(fission(source,cover,[part.split(',') for part in args.partitions.split(';')])
           if args.partitions else region(source,cover))
    proof['producer']=file_identity(__file__)
    args.output.write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),operators=len(cover))))


if __name__=='__main__':main()
