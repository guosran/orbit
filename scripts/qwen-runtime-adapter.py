#!/usr/bin/env python3
"""Source-bound adapter for the existing complete Triton-XDNA Qwen runtime.

This is an external implementation contract, not native ProgramGraph lowering.
It never changes vendor sources or runs the NPU unless replay --execute is used.
The two decisions are global RMSNorm and SwiGLU placement. All other choices
are fixed. Source extraction uses AST inspection and never imports the runtime.
"""
import argparse
import ast
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

SCHEMA = 'orbit.external-qwen-runtime.v1'
FIXED = dict(mode='generate', lengths='5', outputs=128, threads=4, bucket=1,
             static_weights=True, exact_k=True, max_seq=2048, block_m=0)

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def file_identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest(), bytes=path.stat().st_size)

def save_new(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')

def method(tree, cls, name):
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)

def calls(node):
    # Line/column ordering reflects the lexical source, not ast.walk ordering.
    return [n.func.attr for n in sorted(ast.walk(node), key=lambda n: (getattr(n, 'lineno', 0), getattr(n, 'col_offset', 0)))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name) and n.func.value.id == 'self']

def register(aie, prefill_tokens=5):
    if prefill_tokens not in (5,128,512):
        raise ValueError("Supported prefill lengths are 5, 128 and 512")
    aie = Path(aie).resolve(strict=True)
    src = aie / 'build/qwen25-3b/Triton-XDNA'
    shared = src / 'examples/llm_q4nx'
    qwen = shared / 'qwen25_prefill.py'
    llama = shared / 'llama_prefill.py'
    config = src / 'examples/qwen25_3b_q4nx/config.py'
    tree = ast.parse(config.read_text())
    constants = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            try:
                constants[n.targets[0].id] = ast.literal_eval(n.value)
            except (ValueError, TypeError):
                pass
    expected = dict(D=2048, DH=128, N_Q_HEADS=16, N_KV_HEADS=2, INTER=11008, N_LAYERS=36, VOCAB=151936)
    if any(constants.get(k) != v for k, v in expected.items()):
        raise ValueError('Runtime dimensions differ from the declared complete Qwen2.5-3B contract')
    layer = method(ast.parse(qwen.read_text()), 'Qwen25Prefill', '_layer')
    expected_calls = ['_rms_norm', '_matmul', '_rope', '_rope', '_kv_store', '_attention',
                      '_matmul', '_rms_norm', '_matmul', '_matmul', '_swiglu']
    # The last down GEMM lexically surrounds its SwiGLU argument.
    if calls(layer) != expected_calls:
        raise ValueError(f'Unrecognized Qwen layer call structure: {calls(layer)}')
    prefill = method(ast.parse(llama.read_text()), 'LlamaPrefill', 'prefill')
    if calls(prefill) != ['_layer', '_rms_norm', '_lm_head']:
        raise ValueError('Unrecognized prefill control structure')
    llama_class = next(n for n in ast.parse(llama.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'LlamaPrefill')
    ops = next(ast.literal_eval(n.value) for n in llama_class.body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'NPU_OPS' for t in n.targets))
    if set(ops) != {'matmul', 'rms_norm', 'swiglu'}:
        raise ValueError('Runtime --ops capabilities differ')
    air = src / 'third_party/mlir-air-src/programming_examples/llms'
    files = {Path(__file__).resolve(), aie / 'scripts/qwen25_experiment.py', aie / 'results/qwen25-3b/model-paths.json', config}
    for directory in [shared, config.parent, air / 'qwen25_3b_q4', air / 'qwen25_3b', air.parent / 'fused_decode']:
        files.update(directory.glob('*.py'))
    # These are source/ABI identities, not a claim of native IR semantic proof.
    files.update(air.rglob('*.py'))
    source_files = [file_identity(p) for p in sorted(files)]
    decoder = air.parent / 'fused_decode'
    artifacts = sorted(set(decoder.glob('decode_L*.xclbin')) | set(decoder.glob('decode_L*.insts.bin')))
    if not artifacts:
        raise ValueError('No existing decoder artifacts; this adapter does not compile or download')
    artifact_files = [file_identity(p) for p in artifacts]
    paths = json.loads((aie / 'results/qwen25-3b/model-paths.json').read_text())
    model = file_identity(paths['model'])
    token_files = [file_identity(p) for p in sorted(Path(paths['tokenizer']).glob('*'))
                   if p.is_file() and p.suffix in ('.json', '.txt', '.model')]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(paths['tokenizer'], local_files_only=True)
    base = tokenizer.encode('The capital of France is', add_special_tokens=False)
    filler = tokenizer.encode('A computer processes information using arithmetic and memory. ', add_special_tokens=False)
    prompt = base if prefill_tokens == 5 else (filler * (prefill_tokens // len(filler) + 1))[:prefill_tokens - 5] + base
    if len(prompt) != prefill_tokens or base != [785,6722,315,9625,374]:
        raise ValueError('Runtime input tokenization changed')
    fixed = dict(FIXED, lengths=str(prefill_tokens))
    nodes = []
    def add(nid, family, path, fn, ordinal, layer_id=None, **extra):
        row = dict(id=nid, family=family, source_binding=dict(file=str(path), method=fn, call_ordinal=ordinal), **extra)
        if layer_id is not None:
            row['layer'] = layer_id
        nodes.append(row)
    sequence = [('attn_norm','rms_norm',0), ('qkv','matmul',1), ('q_rope','rope',2),
                ('k_rope','rope',3), ('kv_store','kv_store',4), ('attention','attention',5),
                ('o','matmul',6), ('ffn_norm','rms_norm',7), ('gate_up','matmul',8),
                ('swiglu','swiglu',10), ('down','matmul',9)]
    for layer_id in range(36):
        for role, family, ordinal in sequence:
            add(f'prefill.L{layer_id}.{role}',family,qwen,'Qwen25Prefill._layer',ordinal,layer_id,
                role=role, placement_scope='global-family' if family in ('rms_norm','swiglu') else 'fixed')
    add('prefill.final_norm','rms_norm',llama,'LlamaPrefill.prefill',1,placement_scope='global-family')
    add('prefill.lm_head','lm_head',llama,'LlamaPrefill.prefill',2,placement_scope='fixed')
    runner = aie / 'scripts/qwen25_experiment.py'
    add('handoff.kv','kv_handoff',runner,'main.once',0,placement_scope='fixed',ownership='kv_stack + seed_kv')
    for step in range(127):
        add(f'decode.{step}','opaque_decode',runner,'main.once',1,placement_scope='fixed',
            transformer_layers=36, position=prefill_tokens+step,
            ownership='dispatch includes complete fused model; inner placement unavailable')
    edges = []
    for before, after in zip(nodes, nodes[1:]):
        edges.append(dict(producer=before['id'], consumer=after['id'], kind='source-control',
                          order='original synchronous Python source call order',
                          pricing='owned by local call wall/residual, never an additional transfer charge'))
    for layer_id in range(36):
        edges.append(dict(producer=f'prefill.L{layer_id}.kv_store', consumer='handoff.kv',
                          kind='state', storage=f'KV.layer{layer_id}', version='prefill-prefix',
                          pricing='owned by original kv_stack/seed_kv in host residual'))
    for step in range(127):
        edges.append(dict(producer='handoff.kv' if step == 0 else f'decode.{step-1}',
                          consumer=f'decode.{step}', kind='state', storage='all-layer-KV',
                          version=f'prefix-{prefill_tokens+step}',
                          pricing='owned by inclusive opaque decoder dispatch'))
    counts = {family: sum(n['family'] == family for n in nodes) for family in sorted({n['family'] for n in nodes})}
    if counts['matmul'] != 144 or counts['rms_norm'] + counts['swiglu'] != 109:
        raise ValueError('Incomplete prefill source call coverage')
    manifest = dict(schema=SCHEMA, adapter_kind='source-bound-external-runtime',
                    native_program_graph=False, aie_root=str(aie), files=source_files,
                    model=model, tokenizer_files=token_files, model_revision=paths.get('revision'),
                    dimensions=expected, fixed=fixed, nodes=nodes, edges=edges, counts=counts, decoder_artifacts=artifact_files,
                    input=dict(prompt_token_ids=prompt, greedy_outputs=128,
                               prefill_tokens=prefill_tokens, decode_steps=127),
                    domain=[dict(id='placement.rms_norm',choices=['cpu','npu'],scope='global-family'),
                            dict(id='placement.swiglu',choices=['cpu','npu'],scope='global-family')],
                    boundaries=dict(prefill='original 36-layer Qwen25Prefill.prefill',
                                    decode='opaque complete fused decoder.dispatch',
                                    kv='original float32 KV stack then seed_kv',
                                    costs='local op wall includes launch/transport; do not double-charge edges',
                                    inline_host='bias/split/residual/alloc/control remain original source, require separate residual cost',
                                    preparation='model load/compile excluded by original runtime timer'))
    manifest['source_id'] = identity(manifest)
    return manifest

def check_manifest(m):
    if m.get('schema') != SCHEMA or m.get('native_program_graph') is not False:
        raise ValueError('Wrong external runtime contract')
    without_id = {k:v for k,v in m.items() if k != 'source_id'}
    if identity(without_id) != m.get('source_id'):
        raise ValueError('Source manifest identity mismatch')
    fresh = register(m['aie_root'], m['input']['prefill_tokens'])
    if fresh != m:
        raise ValueError('Source/model/input contract changed; register a new experiment')

def make_plan(m, rms, swiglu):
    if rms not in ('cpu','npu') or swiglu not in ('cpu','npu'):
        raise ValueError('Unsupported placement')
    plan = dict(schema='orbit.external-qwen-plan.v1', source_id=m['source_id'],
                assignment=dict(rms_norm=rms,swiglu=swiglu), fixed=m['fixed'],
                scope='two global runtime placements; not native full-graph optimization')
    plan['plan_id'] = identity(plan)
    return plan

def command(m, plan, name, repeats, warmup, measure):
    check_manifest(m)
    if set(plan) != {'schema','source_id','assignment','fixed','scope','plan_id'}:
        raise ValueError('Unexpected plan fields')
    expected = make_plan(m, **dict(rms=plan['assignment']['rms_norm'], swiglu=plan['assignment']['swiglu']))
    if expected != plan:
        raise ValueError('Plan binding/identity mismatch')
    if not name or '/' in name or name.startswith('-') or name in ('.','..'):
        raise ValueError('Experiment name must be a single filename component')
    if repeats < 1 or not math.isfinite(warmup) or not math.isfinite(measure) or warmup < 0 or measure < 0:
        raise ValueError('Invalid measurement budget')
    aie = Path(m['aie_root'])
    if (aie / 'results/qwen25-3b' / f'{name}.jsonl').exists():
        raise ValueError('Refusing to overwrite existing experiment')
    ops = ['matmul'] + [k for k in ('rms_norm','swiglu') if plan['assignment'][k] == 'npu']
    return [str(aie / '.venv/bin/python'), str(aie / 'scripts/qwen25_experiment.py'), 'generate',
            '--name',name,'--ops',','.join(ops),'--bucket','1','--threads','4','--block-m','0',
            '--static-weights','--exact-k','--lengths',m['fixed']['lengths'],'--outputs','128','--max-seq','2048',
            '--repeats',str(repeats),'--warmup-seconds',str(warmup),'--measure-seconds',str(measure)]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='action',required=True)
    reg=subs.add_parser('register');reg.add_argument('--aie-root',type=Path,required=True);reg.add_argument('--output',type=Path,required=True);reg.add_argument('--prefill-tokens',type=int,choices=[5,128,512],default=5)
    plan=subs.add_parser('plan');plan.add_argument('--manifest',type=Path,required=True);plan.add_argument('--rms',choices=['cpu','npu'],required=True);plan.add_argument('--swiglu',choices=['cpu','npu'],required=True);plan.add_argument('--output',type=Path,required=True)
    replay=subs.add_parser('replay');replay.add_argument('--manifest',type=Path,required=True);replay.add_argument('--plan',type=Path,required=True);replay.add_argument('--name',required=True);replay.add_argument('--receipt',type=Path,required=True);replay.add_argument('--repeats',type=int,default=3);replay.add_argument('--warmup-seconds',type=float,default=5);replay.add_argument('--measure-seconds',type=float,default=10);replay.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.action=='register':
        m=register(args.aie_root,args.prefill_tokens);save_new(args.output,m);print(json.dumps(dict(source_id=m['source_id'],counts=m['counts'])))
    elif args.action=='plan':
        m=json.loads(args.manifest.read_text());check_manifest(m);p=make_plan(m,args.rms,args.swiglu);save_new(args.output,p);print(p['plan_id'])
    else:
        m=json.loads(args.manifest.read_text());p=json.loads(args.plan.read_text());cmd=command(m,p,args.name,args.repeats,args.warmup_seconds,args.measure_seconds)
        receipt=dict(schema='orbit.external-qwen-replay.v1',source_id=m['source_id'],plan_id=p['plan_id'],command=cmd,executed=args.execute)
        # Write the validated command before execution; original runner owns NPU lock.
        save_new(args.receipt,receipt);print(json.dumps(receipt),flush=True)
        if args.execute:
            raise SystemExit(subprocess.run(cmd,check=False).returncode)

if __name__=='__main__':
    main()
