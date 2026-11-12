#!/usr/bin/env python3
"""Emit source-backed managed-buffer reservations for the r18 Qwen runner.

The on-chip fallback reserves the complete physical array. It is deliberately
an upper bound for admitted compiled programs, not a claim about their actual
placement. Runtime-ABI buffers below are distinct physical allocations unless
an explicit shared-BO alias is documented. Framework/driver allocator overhead
is outside this managed-buffer ledger and remains subject to the RSS guard.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def identity(path):
    path = Path(path).resolve(strict=True)
    return dict(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def allocation(size):
    return dict(memory='runtime-abi', bytes=size)


def array_bound():
    return {**{f'npu.sram.tile_{x}_{y}': 65536 for x in range(8) for y in range(2, 6)},
            **{f'npu.sram.mem_tile_{x}_1': 524288 for x in range(8)}}


def produce(args):
    vendor = args.vendor.resolve(strict=True)
    kernels = vendor / 'examples/llm_q4nx/kernels.py'
    decdir = vendor / 'third_party/mlir-air-src/programming_examples/fused_decode'
    inference = vendor / 'third_party/mlir-air-src/programming_examples/llms/qwen25_3b_q4/qwen25_3b_q4_inference.py'
    weights = inference.with_name('qwen25_3b_q4_weights.py')
    llama_prefill = vendor/'examples/llm_q4nx/llama_prefill.py'
    prefill_source = args.artifacts/'qwen-whole-graph-r17/frozen/qwen-node-source-r17-v4-p5.json'
    source=json.loads(prefill_source.read_text())
    prefixes=[v for v in source['values'] if v['id'].endswith(('.kv.K.prefix','.kv.V.prefix'))]
    if len(prefixes)!=72 or any(v['shape']!=[5,256] or v['dtype']!='f32' or v['role']!='output' for v in prefixes):
        raise ValueError('KV physical tail deduction requires the original 72 P5 FP32 source outputs')
    kv_prefix_bytes=sum(math.prod(v['shape'])*4 for v in prefixes)
    kv_full_bytes=36*2*2048*256*4
    # The source carries only the prefix as an output. The actual prefiller
    # eagerly allocates full max_seq caches, and clear_context zeros all rows.
    # The common graph has no explicit physical alias witness linking its
    # prefix outputs to these NumPy arrays. Charge the full cache separately;
    # the resulting 368640-byte overlap is a deliberate conservative bound.
    defaults={'prefill.base-kv.full-cache-conservative-bound':allocation(kv_full_bytes),
              # np.stack(...).astype(float32) makes copies even though its
              # inputs already have that dtype. Retain both K/V stack and cast
              # bounds conservatively; the seed routine does not alias them.
              'prefill.kv-stack-copy-upper-bound':allocation(2*kv_prefix_bytes)}
    # The runner keeps per-token lg.copy() arrays, stacks them into `full`, and
    # forms a boolean finite mask. Reserve an extra full copy for NPZ output
    # staging. Overlap with logical source logits is conservatively not removed.
    logits_elements=128*151936
    defaults['runner.logit-capture-stack-save-upper-bound']=allocation(logits_elements*(3*4+1))
    evidence = [identity(p) for p in (kernels, inference, decdir/'kv_bos.py',
                                     decdir/'decode_staircase.py', decdir/'decode_insts_gen.py',
                                     vendor/'amd_triton_npu/backend/driver.py', weights,
                                     vendor/'amd_triton_npu/backend/multilaunch.py',
                                     vendor/'examples/llm_q4nx/fused_mlp.py',
                                     llama_prefill, prefill_source)]
    original = {}
    # resident_weight(exact=True) selects K unchanged and a 256-column tile for
    # these four original shapes. Every N is divisible by 256: no padding delta.
    for role, k, n in [('qkv', 2048, 2560), ('o', 2048, 2048),
                       ('gate_up', 2048, 22016), ('down', 11008, 2048)]:
        stride = k + 64 if k & (k-1) == 0 else k
        pages = {}
        for dtype, shape, width in [('bf16', [128, stride], 2), ('f32', [128, n], 4)]:
            key = 'original.io.'+dtype+'.'+'x'.join(map(str, shape))
            pages[key] = allocation(math.prod(shape)*width)
        # _proj_chain constructs real torch.zeros arguments. NPUChain.add
        # retains tuple(args) in _specs after compilation; these are distinct
        # allocations from shared IO pages and from the prepared model weight.
        # The global chain cache keys geometry, so charge once per shape rather
        # than once per layer. No alias follows from identical tensor contents.
        specs=[([128,stride],2),([k,n],2),([128,n],4)]
        for index,(shape,width) in enumerate(specs):
            key=f'original.proj-chain.128x{k}x{n}.spec{index}'
            pages[key]=allocation(math.prod(shape)*width)
        original[role] = dict(
            # Original CPU matmul explicitly casts the complete BF16 weight
            # to f32 and rounds x through bf16/f32. BLAS-internal workspaces
            # remain allocator overhead rather than a fabricated array size.
            cpu=dict(execution_variant=dict(temporary_bytes={'runtime-abi':k*n*4+5*k*6+5*n*4})),
            npu=dict(execution_variant=dict(resident_allocations=pages,
                                           temporary_bytes=array_bound()),
                     prepared_weight=dict(logical_shape=[k,n], physical_shape=[k,n],
                         physical_strides=[n,1], dtype='bf16', bytes=k*n*2,
                         extra_physical_bytes=0,
                         alias='source physical BF16 weight replaced by one shared host/device BO; original tensor released'),
                     evidence='kernels.py resident_weight/exact_k/tile_n/io_page/triton_matmul; v4 adapter verifies shared_bo and releases original'))
    # These original non-chain kernels stage separate operands through launch.
    # Charge both host arrays and device arguments, plus conversion/result
    # tensors, even if an allocator happens to reuse pages. This bounds managed
    # storage for one synchronous call without asserting a persistent BO cache.
    swiglu_n = 5*11008
    swiglu_chunk = math.ceil(128*11008/65536)*65536
    swiglu_bytes = 6*swiglu_chunk + 8*swiglu_n
    norm_bytes = (2*128*2048*2+2048*2) + 5*2048*6 + 2048*2
    for role,size in [('swiglu',swiglu_bytes), ('attn_norm',norm_bytes),
                      ('ffn_norm',norm_bytes), ('final_norm',norm_bytes)]:
        kind='swiglu' if role=='swiglu' else 'norm'
        bo_sizes=[swiglu_chunk*2]*3 if kind=='swiglu' else [128*2048*2,2048*2,128*2048*2]
        # ELF driver Session retains one grow-only BO set per compiled module.
        # All norm roles here share one geometry/module; all layer SwiGLUs share
        # one other. Conservatively retain these sets through the whole program
        # even if the driver's eight-session LRU evicts them earlier.
        bos={f'original.driver.{kind}.arg{i}':allocation(n) for i,n in enumerate(bo_sizes)}
        cpu_bytes=2*swiglu_n*4 if kind=='swiglu' else 4*5*2048*4+5*16
        original[role] = dict(cpu=dict(execution_variant=dict(temporary_bytes={'runtime-abi':cpu_bytes})),
            npu=dict(execution_variant=dict(temporary_bytes={**array_bound(),'runtime-abi':size}, resident_allocations=bos),
                accounting='conservative simultaneous host conversions/padded arrays/results and distinct staged BO operands; logical source buffers remain separately counted',
                evidence='kernels.py triton_swiglu/glu_chunk/elem_block and triton_rms_norm; P5,D2048,FFN11008'))
    # Fixed CPU expression buffers are small at P5 but still physical storage.
    # Charge explicit tensor expressions, including returned results even when
    # the source graph separately accounts for them. These are upper bounds on
    # managed arrays; native library allocator/workspace overhead is separate.
    fixed_cpu={
        'q_rope':4*5*2048*4,
        'k_rope':4*5*256*4,
        'attention':4*5*2048*4+4*16*5*5*4+5*5*4+2*5*8,
        'lm_head':2048*2+151936*(2+4),
        'embedding':5*2048*(2+4+4),
    }
    for role,size in fixed_cpu.items():
        original[role]=dict(cpu=dict(execution_variant=dict(temporary_bytes={'runtime-abi':size}),
            evidence='LlamaPrefill CPU RoPE/GQA/embedding and BF16 LM-head expressions; P5,D2048,Qheads16,KVheads2,DH128,V151936'))
    # The two actual coalescing contexts have the same physical ABI. Read the
    # checked manifest rather than deriving weight sizes from model names.
    contexts = {}
    ledger_path = args.artifacts/'qwen-full-dim-r18/decode-resource-ledger.json'
    ledger = json.loads(ledger_path.read_text())
    for coalesce in (0, 1):
        directory = args.artifacts/f'qwen-full-dim-r18/decode-r0-cache1-coalesce{coalesce}'
        template = json.loads((directory/'variant.json').read_text())
        sizes = [2048, 983040000, 246272, 158208, 37748736]
        abi = template['abi']
        for size in sizes:
            if f'memref<{size}xbf16>' not in abi:
                raise ValueError('Decoder ABI changed; re-derive physical allocation ledger')
        resident = {f'decode.bo.arg{i}': allocation(size*2) for i,size in enumerate(sizes)}
        # Host-only BO maps alias their own device pages, but these NumPy arrays
        # are separate: Wv16 retains NPZ W; KV is explicitly np.zeros.
        resident.update({'decode.host.Wv16': allocation(sizes[1]*2),
                         'decode.host.KV': allocation(sizes[4]*2),
                         'decode.host.embedding': allocation(151936*2048*2),
                         'decode.host.final_norm': allocation(2048*2),
                         # layer_qkv_bias converts each retained bias to f32.
                         'decode.host.qkv_bias': allocation(36*2560*4),
                         'decode.host.rms_slabs': allocation(36*4096*2)})
        physical = next(x for x in ledger['implementations']
                        if x['name']==directory.name+'-L2048')
        sram = {f'npu.sram.{tile}': size+(physical['stack_bytes_per_core'] if tile.startswith('tile_') else 0)
                for tile,size in physical['compiler_sram_buffer_bytes_by_tile'].items()}
        # This package has only two non-exact linear templates. Generator base
        # u32+slope i64 and state base/slope/indices i64+buf/BO u32 consume at
        # most 44 bytes/word. Reserve 48 bytes/word for both input files even
        # though they calibrate one template, plus 32 bytes/word patch scratch.
        instructions = [dict(**identity(p), bytes=p.stat().st_size)
                        for p in sorted(directory.glob('*.insts.bin'))]
        if len(instructions)!=2 or any(directory.glob('*.rb.insts.bin')) or any(directory.glob('*.sw.insts.bin')):
            raise ValueError('Exact decoder instruction scheme needs a new allocation derivation')
        inst_bytes=sum(x['bytes'] for x in instructions)
        resident['decode.instructions.linear-generator-state-bo-bound']=allocation(12*inst_bytes)
        # First-dispatch RMS init, 36 per-layer rope slabs + concatenate/cast,
        # one layer's f32 concatenation/LUT scratch, and f32 vocabulary drain.
        # They need not coexist, but summing them is a conservative local bound.
        host_helpers = 246272*2 + 3*(36*2688*2) + 2688*8 + 128*8*4 + 151936*4
        sram['runtime-abi']=8*inst_bytes+host_helpers
        contexts[str(directory)] = dict(
            template=identity(directory/'variant.json'), variant_id=template['variant_id'],
            execution_variant=dict(resident_allocations=resident, temporary_bytes=sram),
            physical_bo_bytes=sum(sizes)*2, managed_resident_bytes=sum(v['bytes'] for v in resident.values()),
            host_device_aliases=['decode.bo.arg'+str(i) for i in range(5)],
            instruction_files=instructions,
            instruction_heap_status='conservative checked linear-template array/BO bound; Python object/allocator overhead outside managed ledger',
            ledger=identity(ledger_path))
    ffn = {}
    for order in ('gate-up','up-gate'):
        path=args.artifacts/f'qwen-whole-graph-r18/frozen-r18-v4/compiled-{order}/manifest.json'
        manifest=json.loads(path.read_text())
        if identity(manifest['artifact'])['sha256']!=manifest['artifact_sha256']:
            raise ValueError('FFN artifact identity mismatch')
        # Each _mm_op call owns separate A/B/C torch.zeros placeholders. The
        # SwiGLU (plane,plane,plane) tuple aliases ONE additional torch.empty,
        # so charge it once. RoutedFFN shares the actual NPUChain among all
        # retention policies for one artifact/context; the two orders retain
        # separate chains and therefore separate placeholder storage.
        r,h=manifest['row_tile'],manifest['padded_hidden']
        prefix='ffn.chain.'+manifest['artifact_sha256']
        placeholders={}
        for phase in ('gate','up'):
            for i,(shape,width) in enumerate([([r,2048],2),([2048,h],2),([r,h],2)]):
                placeholders[f'{prefix}.{phase}.spec{i}']=allocation(math.prod(shape)*width)
        placeholders[prefix+'.swiglu.plane']=allocation(r*h*2)
        for i,(shape,width) in enumerate([([r,h],2),([h,2048],2),([r,2048],4)]):
            placeholders[f'{prefix}.down.spec{i}']=allocation(math.prod(shape)*width)
        ffn[manifest['artifact_sha256']]=dict(
            manifest=identity(path),
            execution_variant=dict(temporary_bytes=array_bound(), resident_allocations=placeholders),
            retained_placeholder_bytes=sum(row['bytes'] for row in placeholders.values()),
            accounting='full physical SRAM capacity conservative upper bound per synchronous firmware configuration; DDR weights/arguments already in FFN packing registration, add no duplicate BO allocations')
    return dict(schema='orbit.qwen-r18-managed-capacity-supplement.v1',
                producer=identity(__file__), evidence=evidence,
                defaults=dict(execution_variant=dict(resident_allocations=defaults)), original_operators=original,
                prefill_kv_accounting=dict(source_prefix_bytes=kv_prefix_bytes,
                    physical_full_cache_bytes=kv_full_bytes,
                    added_physical_reservation_bytes=kv_full_bytes,
                    conservative_overlap_bytes=kv_prefix_bytes,
                    source=identity(prefill_source),
                    accounting='No source-prefix physical alias deduction without a common-graph witness; full physical cache charged in addition to logical prefix outputs'),
                original_npu_default=dict(execution_variant=dict(temporary_bytes=array_bound())),
                decode_contexts=contexts, ffn_contexts=ffn,
                bounds=dict(onchip='exact decode buffer+stack ledger; full-array conservative bound for admitted original NPU and FFN',
                    runtime_abi='explicit managed buffers only; source graph logical allocations must be included separately',
                    unmanaged='Python/Torch/BLAS/driver/code/generator allocator overhead is not given an unproved bound; original 20 GiB RSS/0.5 GiB swap guard remains mandatory'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vendor',type=Path,required=True)
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=produce(args)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output), decode_contexts=len(result['decode_contexts']),
                          original_roles=len(result['original_operators']), ffn_contexts=len(result['ffn_contexts']))))


if __name__=='__main__':
    main()
