#!/usr/bin/env python3
"""Local sparse-marker compilation adapter; hardware execution uses v3's XRT runner.

No installed file is changed. --prepare compiles only; run mode self-locks NPU.
Markers delimit one core K-block MAC after the paired input locks.
"""
import argparse
import collections
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


def instrument(text):
    """Fail closed on the known lowered core shape, preserving actual operations."""
    lines=text.splitlines(); cores=[]; i=0
    while i<len(lines):
        m=re.search(r'%core_(\d+)_(\d+) = aie.core\(',lines[i])
        if not m:i+=1;continue
        start=i; depth=lines[i].count('{')-lines[i].count('}');i+=1
        while depth:
            depth+=lines[i].count('{')-lines[i].count('}');i+=1
        end=i-1; body=lines[start:end+1]
        # Input acquire pairs gate real K-block execution. Output acquisition
        # alone can occur before runtime trace-start, so is not a valid marker.
        pairs=[j for j in range(len(body)-1) if 'AcquireGreaterEqual' in body[j]
               and 'AcquireGreaterEqual' in body[j+1]]
        inserts={}
        for j in pairs:
            release=next(q for q in range(j+2,len(body)) if 'aie.use_lock' in body[q] and 'Release' in body[q])
            indent=' '*(len(body[j])-len(body[j].lstrip()))
            inserts[j+2]=indent+'aie.event(0)'
            inserts[release]=indent+'aie.event(1)'
        if not pairs:raise ValueError('no paired input acquires for core K-block')
        for j,line in sorted(inserts.items(),reverse=True):body.insert(j,line)
        lines[start:end+1]=body;i+=len(inserts)
        cores.append(dict(column=int(m[1]),row=int(m[2])))
    if not cores:raise ValueError('no supported actual cores')
    text='\n'.join(lines)+'\n'
    # Replace only values of trace configuration writes. Never replace a shared
    # constant globally, which could silently change computation or DMA metadata.
    constants={m[1]:int(m[2]) for m in re.finditer(r'(%[\w]+) = arith.constant (-?\d+) : i32',text)}
    serial=0;changed=collections.Counter();traced=set()
    def write(match):
        nonlocal serial
        addr=constants.get(match['addr']);row=int(match['row'])
        values={0x340E0:33|(34<<8)|(124<<16)|(125<<24),0x340E4:0,0x340D0:(121<<24)|(122<<16)}
        if row<2 or addr not in values:return match[0]
        serial+=1;changed[hex(addr)]+=1;traced.add((int(match['col']),row));name=f'%orbit_trace_v4_{serial}'
        return f'{match["indent"]}{name} = arith.constant {values[addr]} : i32\n'+match[0].replace(match['val'],name,1)
    pattern=r'(?P<indent> +)aiex.npu.write32\((?P<addr>%\w+), (?P<val>%\w+)\) \{column = (?P<col>\d+) : i32, row = (?P<row>\d+) : i32\} : i32, i32'
    text=re.sub(pattern,write,text)
    if not traced or any(changed[hex(a)]!=len(traced) for a in (0x340E0,0x340E4,0x340D0)):
        raise ValueError('trace configuration/core count mismatch')
    # Emit USER_EVENT_0 on each core after all existing output await/free ops.
    # This is the trace unit stop condition, not a claim of DDR DMA completion.
    starts=list(re.finditer(r'^    aie.runtime_sequence @[^\n]+ \{$',text,re.M))
    selected=[m for m in starts if 'aiex.dma_await_task' in text[m.end():text.find('\n    }',m.end())]]
    if len(selected)!=1:raise ValueError('expected one projection runtime sequence')
    pos=text.find('\n    }',selected[0].end())
    stop=['      %orbit_trace_stop_address = arith.constant 213000 : i32',
          '      %orbit_trace_stop_broadcast = arith.constant 213064 : i32',
          '      %orbit_trace_stop_event = arith.constant 126 : i32',
          '      aiex.npu.write32(%orbit_trace_stop_broadcast, %orbit_trace_stop_event) {column = 0 : i32, row = 0 : i32} : i32, i32',
          '      aiex.npu.write32(%orbit_trace_stop_address, %orbit_trace_stop_event) {column = 0 : i32, row = 0 : i32} : i32, i32']
    # Match the installed sparse IRON trace path: 64-byte AXI bursts rather
    # than AIR's implicit default. A sparse packet must reach system memory.
    text=re.sub(r'(aiex.npu.writebd \{[^\n]*buffer_offset = \d+ : i32,)(?! burst_length)',r'\1 burst_length = 64 : i32,',text)
    pos=text.find('\n    }',text.find('aie.runtime_sequence @',selected[0].start()))
    # Finite low-rate padding explicitly drains partial trace packets / AXI
    # bursts. USER0 ends the measured scope; USER1 is excluded padding.
    padding=['      %orbit_trace_drain_address = arith.constant 213000 : i32',
             '      %orbit_trace_scope_end = arith.constant 124 : i32',
             '      %orbit_trace_padding = arith.constant 125 : i32']
    for col,row in sorted(traced):
        prefix='      aiex.npu.write32(%orbit_trace_drain_address, '
        suffix=') {column = '+str(col)+' : i32, row = '+str(row)+' : i32} : i32, i32'
        padding.append(prefix+'%orbit_trace_scope_end'+suffix)
        padding.extend([prefix+'%orbit_trace_padding'+suffix]*64)
    text=text[:pos]+'\n'+'\n'.join(padding+stop)+text[pos:]
    return text,dict(cores=cores,traced_cores=[dict(column=c,row=r) for c,r in sorted(traced)],marker_scheme="paired_input_acquire_k_block",drain_padding_events_per_core=64,event_slots=[33,34,124,125,0,0,0,0],start_event=122,stop_event=121,shim_burst_bytes=64,scope='one core K-block MAC, after both input acquires and before input release; output initialization excluded',stop_order='after projection output DMA await/free; before XRT wait and output BO sync/readback',drain_proof='not assumed; raw marker/count and stop observation required')


def adapter(arguments):
    from taskflow_common import write_json
    source=Path(arguments[-1]);original=source.read_text();modified,metadata=instrument(original)
    folder=Path(os.environ['ORBIT_TRACE_ADAPTER_DIR']);folder.mkdir(parents=True,exist_ok=True)
    (folder/'aiecc-input-original.mlir').write_text(original)
    target=folder/'aiecc-input-marked.mlir';target.write_text(modified)
    from taskflow_npu_trace_v3 import digest
    metadata.update(original_sha256=digest(source),instrumented_sha256=digest(target),adapter_sha256=digest(__file__))
    write_json(folder/'instrumentation.json',metadata)
    return subprocess.call([os.environ['ORBIT_REAL_AIECC'],*arguments[:-1],str(target)])


def compile_shape(case,shape,out,trace_bytes):
    import taskflow_npu_trace_v3 as old
    from taskflow_common import write_json
    out.mkdir(parents=True,exist_ok=True);source=old.artifact(case,shape);shutil.copy2(source,out/'source.mlir')
    env=os.environ.copy()
    if trace_bytes:
        real=shutil.which('aiecc');assert real
        wrapper=out/'adapter-bin';wrapper.mkdir()
        launcher=wrapper/'aiecc';launcher.write_text('#!'+sys.executable+'\nimport runpy,sys\nsys.path.insert(0,'+repr(str(Path(__file__).parent))+')\nsys.argv=['+repr(str(Path(__file__).resolve()))+',"--adapter",*sys.argv[1:]]\nrunpy.run_path(sys.argv[0],run_name="__main__")\n');launcher.chmod(0o755)
        env.update(ORBIT_REAL_AIECC=real,ORBIT_TRACE_ADAPTER_DIR=str(out/'instrumentation'),PATH=str(wrapper)+os.pathsep+env['PATH'])
    flags=['-trace-size',str(trace_bytes),'-trace-offset',str(shape[0]*shape[2]*4)] if trace_bytes else []
    code='from triton.backends.amd_triton_npu.driver import _aircc_compile;import json;from pathlib import Path;a=_aircc_compile('+repr(str(out/'source.mlir'))+',"elf","npu2",'+repr(str(out/'build'))+',extra_args='+repr(flags)+');Path("artifacts.json").write_text(json.dumps(a))'
    started=time.perf_counter()
    with (out/'compile.log').open('w') as log:subprocess.run([sys.executable,'-c',code],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    meta=json.loads((out/'artifacts.json').read_text());meta.update(compile_seconds=time.perf_counter()-started,source_path=str(source),source_sha256=old.digest(source),elf_sha256=old.digest(meta['elf_path']),shape=list(shape),extra_args=flags)
    write_json(out/'compile.json',meta);return meta


def decode(raw,physical,out,device_name=None):
    import taskflow_npu_trace_v3 as old
    from taskflow_common import write_json
    manifest=ORIGINAL_DECODE(raw,physical,out,device_name)
    manifest['expected_event_patterns']=['INSTR_EVENT_0','INSTR_EVENT_1','USER_EVENT_0']
    manifest['completeness']=dict(status='partial',reason='pending independent per-core expected-count and stop-marker audit; stop configured after output DMA, no assumed drain',scope='core K-block MAC intervals after paired input acquires; not complete graph or host-clock aligned compute time')
    manifest['v4_adapter_sha256']=old.digest(__file__)
    write_json(out/'manifest.json',manifest)
    return manifest


def main():
    if len(sys.argv)>1 and sys.argv[1]=='--adapter':return adapter(sys.argv[2:])
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['gemm','ffn_m5','ffn_m128','all'],default='gemm');p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--trace-bytes',type=int,default=65536);p.add_argument('--iterations',type=int,default=3);p.add_argument('--prepare',action='store_true');a=p.parse_args()
    if a.output_dir.exists():raise ValueError('use new output directory')
    if a.trace_bytes<4096 or a.trace_bytes%4096:raise ValueError('trace capacity must be a positive 4096 multiple')
    if a.iterations<3:raise ValueError('at least three hot samples')
    import taskflow_npu_trace_v3 as old
    global ORIGINAL_DECODE
    ORIGINAL_DECODE=old.decode;old.decode=decode;old.compile_shape=compile_shape
    cases=['gemm','ffn_m5','ffn_m128'] if a.case=='all' else [a.case]
    results=[]
    for case in cases:
        out=a.output_dir.resolve()/case
        if a.prepare:
            plan=json.loads((old.ROOT/f'build/search-npu-v2/data/{case}/base.json').read_text())
            shapes=set()
            for task in plan['tasks']:
                for op in task['operations']:
                    if op['kind']=='matmul':
                        m,n,k=(op[x][1]-op[x][0] for x in ('m','n','k'));shapes.add((((m+127)//128)*128,((k+63)//64)*64,((n+255)//256)*256))
            for shape in sorted(shapes):compile_shape(case,shape,out/'on'/('shape-'+'-'.join(map(str,shape))),a.trace_bytes)
            continue
        result=old.run_case(case,out,a.iterations,a.trace_bytes)
        shutil.copy2(__file__,out/'v4-driver-source.py');results.append(result)
    print(json.dumps([dict(case=r['case'],status=r['status'],error=r.get('error')) for r in results],indent=2))
    return 0 if all(r['status']=='passed' for r in results) else 1

if __name__=='__main__':raise SystemExit(main())
