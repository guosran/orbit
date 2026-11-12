#!/usr/bin/env python3
"""Typed operator source for the original Qwen2.5-3B fused decode.

Source operators and executable covers are separate. Existing fused kernels own
Q4/BFP16 arithmetic, online-attention recurrence and their internal tile schedule.
Their source bodies are hash-bound; this extractor does not claim an MLIR proof
of external arithmetic. Only complete wave covers are executable partitions.
"""
import argparse
import hashlib
import json
from pathlib import Path


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def extract_decode_source(triton_root, prompt_tokens, steps=1, *, prefill_source=None,
                          proof_root=None, capacity=2048):
    if not (1 <= prompt_tokens < capacity and 1 <= steps <= capacity-prompt_tokens):
        raise ValueError('decode token domain exceeds the registered KV capacity')
    root = Path(triton_root).resolve(strict=True)
    dec = root/'third_party/mlir-air-src/programming_examples/fused_decode'
    llm = root/'third_party/mlir-air-src/programming_examples/llms/qwen25_3b_q4'
    builder = dec/'fused_decode.py'
    if proof_root is None:
        raise ValueError('require actual compiled/pilot region proof root')
    proof_root = Path(proof_root)
    original = json.loads((proof_root/'w0-36-L2048/compiled.json').read_text())
    if sha(builder) != original['source']['sha256']:
        raise ValueError('actual compiled builder source changed')
    text = builder.read_text()
    for symbol in ('attn_qk_blk', 'attn_kv_blk', 'attn_kv_fin', 'rope_compute',
                   'rms_norm_aie', 'residual_add_aie', 'glu_aie'):
        if symbol not in text:
            raise ValueError('unrecognized builder operator catalogue')
    if (original['decoder_layers'], original['vocab_waves']) != (36, 25):
        raise ValueError('incomplete model')
    if prefill_source:
        src = json.loads(json.dumps(prefill_source)); src.pop('source_id', None)
    else:
        src = dict(schema='orbit.program-operator-source.v1', bindings=[], values=[], operators=[])
    bindings, values, ops = src['bindings'], src['values'], src['operators']
    producer = {v: op['id'] for op in ops for v in op['outputs']}
    value_map = {v['id']: v for v in values}
    wave_ops = {}
    def bind(name, path, symbol):
        if symbol.split('.')[-1] not in path.read_text():
            raise ValueError('missing bound external symbol '+symbol)
        name = 'decode.binding.'+name
        bindings.append(dict(id=name, file=str(path), sha256=sha(path), symbol=symbol))
        return name
    runtime = bind('runtime', llm/'qwen25_3b_q4_inference.py', 'FusedDecoder.dispatch')
    seed = bind('seed', llm/'qwen25_3b_q4_inference.py', 'FusedDecoder.seed_kv')
    generation = bind('generation', llm/'qwen25_3b_q4_inference.py', 'argmax')
    rope_host = bind('rope_lut', llm/'qwen25_3b_q4_weights.py', 'generate_rope_lut')
    kernel = {'kv_append': bind('builder', builder, 'build_module')}
    for symbol, file in [('rms_norm_aie','rms_residual.cc'),
                         ('residual_add_aie','rms_residual.cc'),
                         ('proj_qmm_acc256','proj_qmm.cc'), ('rope_compute','rope.cc'),
                         ('glu_aie','glu.cc'), ('attn_qk_blk','attn_qk.cc'),
                         ('attn_kv_blk','attn_kv.cc'), ('attn_kv_fin','attn_kv.cc')]:
        kernel[symbol] = bind(symbol, dec/'kernels'/file, symbol)
    def val(name, shape, dtype='bf16', role='temporary', alias=None, view=None, allocation_view=None):
        if name in value_map:
            raise ValueError('duplicate value '+name)
        v = dict(id=name, shape=shape, dtype=dtype, role=role, layout='row_major')
        if alias: v['state_alias_of'] = alias
        if view: v['view'] = view
        if allocation_view: v['allocation_view'] = allocation_view
        if role == 'workspace': v['initialization'] = 'uninitialized'
        values.append(v); value_map[name] = v
        return name
    def op(name, kind, ins, outs, binding, wave=None, **attrs):
        deps = list(dict.fromkeys(producer[v] for v in ins if v in producer))
        ops.append(dict(id=name, kind=kind, binding=binding, source_ordinal=len(ops),
                        inputs=ins, outputs=outs, dependencies=deps, attributes=attrs))
        producer.update({v: name for v in outs})
        if wave is not None:
            wave_ops.setdefault((current_step, wave), []).append(name)
    # Physical source ABI: one packed W BO, one RMS BO, one KV BO.
    # Logical projections are readonly views, never independent allocations.
    packed_w = val('decode.W.BO', [983040000], role='input')
    rms_state = val('decode.RMS.initial', [246272], role='input')
    weights, host_bias = {}, []
    phase_sizes = [('qkv',1638400), ('o',1310720),
                   ('gate_up',14417920), ('down',7208960)]
    for layer in range(36):
        weights[layer] = {}
        offset = layer*24576000
        for name, size in phase_sizes:
            weights[layer][name] = (offset,size)
            offset += size
        host_bias.append(val(f'decode.host_bias.L{layer}', [2560], 'f32', 'input'))
    embed = ('weights.embedding' if prefill_source else
             val('decode.weights.embedding', [151936,2048], role='input'))
    lm_weights = [(884736000+c*3932160,3932160) for c in range(25)]
    kv_state = val('decode.KV.seed', [144,capacity,128],
                   role='temporary' if prefill_source else 'input')
    if prefill_source:
        token = val('decode.first_token', [1], 'i64')
        op('decode.first_token_select', 'argmax', ['prefill.logits'], [token],
           generation, fixed_device='cpu')
    else:
        token = val('decode.first_token', [1], 'i64', 'input')
    if prefill_source:
        prefixes = [f'prefill.L{layer}.kv.{kind}.prefix'
                    for layer in range(36) for kind in ('K','V')]
        op('decode.KV.handoff', 'kv-seed-handoff', prefixes, [kv_state], seed,
           keep=prompt_tokens, decoder_layers=36,
           destination_layout='layer-major, K0,K1,V0,V1; region [capacity,128]',
           rounding='float32 to bf16', zero_fill_unused=True,
           source_memory='host', destination_memory='npu-accessible')
    y_workspace = val('decode.Y.workspace', [158208], role='workspace')
    x_workspace = val('decode.X.workspace', [2048], role='workspace')
    host_groups = ([dict(id='decode.first_token_select',kind='first-token-argmax',
                              operators=['decode.first_token_select']),
                    dict(id='decode.KV.handoff',kind='kv-handoff',
                         operators=['decode.KV.handoff'])] if prefill_source else [])
    for current_step in range(steps):
        position = prompt_tokens+current_step
        prefix = f'decode.T{current_step}.'
        pos = val(prefix+'position', [1], 'i64', 'input')
        rms_after = val(prefix+'RMS.after', [246272], alias=rms_state)
        op(prefix+'rope_patch', 'rope-rms-buffer-patch',
           [rms_state,pos]+host_bias, [rms_after], rope_host,
           position=position, fixed_device='cpu', global_arg=2,
           written_offset_words=147456, written_words=96768,
           preserves_norm_slabs=True, preserves_final_norm=True,
           numerical_contract='original rope_w_layer casts LUT and QKV bias to BF16')
        rms_state = rms_after
        x = val(prefix+'x0', [2048], allocation_view=dict(base=x_workspace,offsets=[0],strides=[1]))
        op(prefix+'embedding', 'embedding-gather', [token,embed], [x], runtime,
           fixed_device='cpu')
        for layer in range(36):
            p = prefix+f'L{layer}.'; w = {}
            pending_views = []
            for name,(offset,size) in weights[layer].items():
                w[name] = val(p+'weights.'+name,[size],
                              view=dict(base=packed_w,offsets=[offset],strides=[1]))
                pending_views.append((name,packed_w,w[name]))
            for role, offset, size in [('attn_norm',layer*4096,2048),
                                       ('ffn_norm',layer*4096+2048,2048),
                                       ('qkv_bias',147456+layer*2688+128,2560)]:
                w[role] = val(p+'RMS.'+role, [size],
                              view=dict(base=rms_state,offsets=[offset],strides=[1]))
                pending_views.append((role,rms_state,w[role]))
            lut = val(p+'rope_lut', [128],
                      view=dict(base=rms_state,offsets=[147456+layer*2688],strides=[1]))
            pending_views.append(('rope_lut',rms_state,lut))
            def result(name, shape):
                return val(p+name+'.value', shape)
            def emit(name, kind, ins, outs, symbol, **attrs):
                op(p+name, kind, ins, outs, kernel[symbol], wave=layer,
                   layer=layer, position=position,
                   implementation_boundary='within original complete decode wave', **attrs)
            for view_name,base,out in pending_views:
                emit('abi_view.'+view_name,'slice-view',[base],[out],'kv_append',
                     view_only=True,allocation=False)
            h = result('attn_norm', [2048])
            emit('attn_norm','rms-norm',[x,w['attn_norm']],[h],'rms_norm_aie',eps=1e-6)
            qkv = result('qkv',[2560])
            emit('qkv','q4-matmul',[h,w['qkv']],[qkv],'proj_qmm_acc256',
                 logical_shape=[2048,2560], packed_carrier='q4k-cascade-bf16-words',
                 numerics='original AIE BFP16/Q4 kernel contract')
            biased = result('qkv_bias',[2560])
            emit('qkv_bias','broadcast-add',[qkv,w['qkv_bias']],[biased],
                 'rope_compute',before_rotation=True)
            q = val(p+'q.value',[2048],view=dict(base=biased,offsets=[0],strides=[1]))
            k = val(p+'k.value',[256],view=dict(base=biased,offsets=[2048],strides=[1]))
            v = val(p+'v.value',[256],view=dict(base=biased,offsets=[2304],strides=[1]))
            emit('qkv_split','split-view',[biased],[q,k,v],'rope_compute',sizes=[2048,256,256])
            qr = result('q_rope',[2048]); kr = result('k_rope',[256])
            emit('q_rope','neox-rope',[q,lut],[qr],'rope_compute',heads=16,head_dim=128)
            emit('k_rope','neox-rope',[k,lut],[kr],'rope_compute',heads=2,head_dim=128)
            after = val(p+'KV.after',[144,capacity,128],alias=kv_state)
            emit('kv_append','kv-append',[kv_state,kr,v],[after],'kv_append',
                 written_position=position,global_inplace_arg=4,
                 layer_offset_words=layer*capacity*512,region_major=True)
            kv_state = after
            layer_cache = val(p+'KV.layer', [4,capacity,128],
                              view=dict(base=kv_state,offsets=[layer*4,0,0],
                                        strides=[capacity*128,128,1]))
            emit('kv_layer_view','slice-view',[kv_state],[layer_cache],'kv_append',
                 view_only=True,allocation=False)
            # Typed streams expose the block recurrence. They are not permission
            # to materialize a full score buffer or split the fused implementation.
            score = result('attention.score_stream',[2,(capacity+15)//16,192])
            emit('attention.qk_online','online-qk-softmax',[qr,layer_cache],[score],
                 'attn_qk_blk',
                 iteration_domain=dict(blocks=(capacity+15)//16,
                                       valid_length=position+1,block_size=16),
                 running_state=['max:bf16[2,16]','correction:f32[2,8]'],
                 interleaved_with=p+'attention.pv_online',
                 storage='streaming shared L1 ring; logical stream not full-score allocation')
            acc = val(p+'attention.accumulator',[2,1024],'f32')
            den = val(p+'attention.denominator',[2,16],'f32')
            emit('attention.pv_online','online-pv-accumulate',[score,layer_cache],[acc,den],
                 'attn_kv_blk',
                 iteration_domain=dict(blocks=(capacity+15)//16,
                                       valid_length=position+1,block_size=16),
                 recurrence='online max correction rescales previous y and denominator')
            a = result('attention',[2048])
            emit('attention.normalize','online-attention-normalize',[acc,den],[a],
                 'attn_kv_fin',output_heads=16,head_dim=128)
            o = result('o',[2048])
            emit('o','q4-matmul',[a,w['o']],[o],'proj_qmm_acc256',
                 logical_shape=[2048,2048],packed_carrier='q4k-cascade-bf16-words')
            r1 = result('residual1',[2048])
            emit('residual1','add',[x,o],[r1],'residual_add_aie')
            h2 = result('ffn_norm',[2048])
            emit('ffn_norm','rms-norm',[r1,w['ffn_norm']],[h2],'rms_norm_aie',eps=1e-6)
            gu = result('gate_up',[22,1024])
            emit('gate_up','q4-matmul',[h2,w['gate_up']],[gu],'proj_qmm_acc256',
                 logical_shape=[2048,22528],semantic_intermediate=11008,padded_intermediate=11264)
            g = val(p+'gate.value',[22,512],view=dict(base=gu,offsets=[0,512],strides=[1024,1]))
            u = val(p+'up.value',[22,512],view=dict(base=gu,offsets=[0,0],strides=[1024,1]))
            emit('gate_split','split-view',[gu],[g,u],'glu_aie',sizes=[512,512],axis=1,chunks=22,physical_order='up512,gate512 per original GLU slice')
            activation = result('swiglu',[11264])
            emit('swiglu','swiglu',[g,u],[activation],'glu_aie',
                 semantic_intermediate=11008,padding='original zero-padded Q4 carrier')
            down = result('down',[2048])
            emit('down','q4-matmul',[activation,w['down']],[down],'proj_qmm_acc256',
                 logical_shape=[11264,2048])
            xnext = result('residual2',[2048])
            emit('residual2','add',[r1,down],[xnext],'residual_add_aie')
            stored = val(p+'X.after', [2048], alias=x)
            emit('layer_output_store','inplace-store',[x,xnext],[stored],'kv_append',
                 global_arg=0,offset_words=0,size_words=2048,overwrite_full=True)
            x = stored
        # Original vocab waves rewrite distinct slices of the same Y buffer.
        # SSA aliases preserve destructive state without inventing copies.
        chunks = []
        final_norm = val(prefix+'RMS.final_norm', [2048],
                         view=dict(base=rms_state,offsets=[244224],strides=[1]))
        op(prefix+'RMS.final_norm_view','slice-view',[rms_state],[final_norm],
           kernel['kv_append'],wave=36,view_only=True,allocation=False)
        for chunk in range(25):
            p = prefix+f'vocab.{chunk}.'
            offset,size = lm_weights[chunk]
            vocab_weight = val(p+'weight',[size],
                               view=dict(base=packed_w,offsets=[offset],strides=[1]))
            op(p+'weight_view','slice-view',[packed_w],[vocab_weight],
               kernel['kv_append'],wave=36+chunk,view_only=True,allocation=False)
            norm = val(p+'normalized',[2048])
            op(p+'norm','rms-norm',[x,final_norm],[norm],kernel['rms_norm_aie'],
               wave=36+chunk,eps=1e-6)
            yp = val(p+'logits.chunk',[6144],allocation_view=dict(base=y_workspace,offsets=[4608+chunk*6144],strides=[1]))
            op(p+'projection','q4-lm-head-chunk',[norm,vocab_weight],[yp],
               kernel['proj_qmm_acc256'],wave=36+chunk,write_offset=4608+chunk*6144,
               write_size=6144,global_arg=3,base_output_offset_words=4608)
            chunks.append(yp)
        logits = val(prefix+'logits',[151936],'f32','output')
        op(prefix+'readback','bf16-readback-concat-trim',chunks,[logits],runtime,
           fixed_device='cpu',global_arg=3,offset_words=4608,
           synchronized_words=153600,returned_words=151936,
           transport='original y_bo.sync(FROM), np.astype(float32), then valid-vocab view',
           kernel_region_covered=False)
        next_token = val(prefix+'next_token',[1],'i64','output')
        op(prefix+'argmax','argmax',[logits],[next_token],generation,fixed_device='cpu')
        token = next_token
        host_groups.extend([
            dict(id=prefix+'host_before',kind='decode-host-before',step=current_step,
                 operators=[prefix+'rope_patch',prefix+'embedding']),
            dict(id=prefix+'host_after',kind='decode-host-after',step=current_step,
                 operators=[prefix+'readback']),
            dict(id=prefix+'argmax',kind='decode-argmax',step=current_step,
                 operators=[prefix+'argmax'])])
    src['numerical_contract'] = (
        'original-triton-xdna-prefill-and-qwen25-q4k-cascade-decode-kernel-contract-v1'
        if prefill_source else 'original-qwen25-q4k-cascade-decode-kernel-contract-v1')
    src['source_id'] = hashlib.sha256(canonical(src)).hexdigest()
    covers = dict(source_id=src['source_id'],
                  scope='Only complete actual wave covers are executable; typed source operators are not arbitrary legal cuts',
                  wave_operators=[dict(step=t,wave=w,operators=ids)
                                  for (t,w),ids in wave_ops.items()],
                  legal_runtime_partitions=[[0,61],[0,36,61],[0,18,36,61]],
                  proof_root=str(proof_root.resolve()),host_groups=host_groups,
                  request_handoff_operators=['decode.first_token_select','decode.KV.handoff'] if prefill_source else [],
                  abi_allocations=[
                      dict(arg=0,value='decode.X.workspace',words=2048),
                      dict(arg=1,value='decode.W.BO',words=983040000),
                      dict(arg=2,value='decode.RMS.initial',words=246272),
                      dict(arg=3,value='decode.Y.workspace',words=158208),
                      dict(arg=4,value='decode.KV.seed',words=37748736)])
    return src,covers


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--triton-root',type=Path,required=True)
    p.add_argument('--proof-root',type=Path,required=True)
    p.add_argument('--prompt-tokens',type=int,required=True)
    p.add_argument('--steps',type=int,default=1)
    p.add_argument('--prefill-source',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--covers',type=Path,required=True)
    a = p.parse_args()
    src,covers = extract_decode_source(
        a.triton_root,a.prompt_tokens,a.steps,
        prefill_source=json.loads(a.prefill_source.read_text()) if a.prefill_source else None,
        proof_root=a.proof_root)
    for path,data in [(a.output,src),(a.covers,covers)]:
        with path.open('x') as f:
            json.dump(data,f,indent=2);f.write('\n')
    print(json.dumps(dict(source_id=src['source_id'],operators=len(src['operators']),
                          values=len(src['values']),waves=len(covers['wave_operators']))))


if __name__ == '__main__':
    main()
