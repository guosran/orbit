#!/usr/bin/env python3
"""Sparse-event minimal probe; local vendor-derived source, existing IRON runtime."""
import argparse
import fcntl
import importlib.util
from pathlib import Path
import time
import traceback
import numpy as np
import aie.iron as iron
from aie.utils.trace import TraceConfig
from taskflow_npu_trace_v3 import AIE, decode, digest
from taskflow_common import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--trace-bytes',type=int,default=16384);a=p.parse_args();out=a.output_dir.resolve()
    if out.exists():raise ValueError('use new output directory')
    out.mkdir(parents=True);source=AIE/'vendor/mlir-aie/programming_examples/basic/event_trace/aie_trace.py';text=source.read_text()
    start=text.index('    prog.enable_trace(');end=text.index('    return prog.resolve_program()',start)
    text=text[:start]+'''    prog.enable_trace(trace_size=trace_size, workers=[worker],
        coretile_events=[CoreEvent.INSTR_EVENT_0, CoreEvent.INSTR_EVENT_1],
        coremem_events=[], memtile_events=[], shimtile_events=[])

'''+text[end:]
    local=out/'design-source.py';local.write_text(text)
    off_source=out/'design-off-source.py'
    begin=text.index('    prog.enable_trace(');finish=text.index('    return prog.resolve_program()',begin)
    off_source.write_text(text[:begin]+text[finish:])
    result=dict(schema='orbit.hardware.minimal-trace.v1',status='compile_failed',design_source=str(source),original_sha256=digest(source),local_sha256=digest(local),expected_scale_calls=4,scope='four scalar scale kernel calls, vendor emits EVENT0/EVENT1',installed_packages_modified=False)
    try:
        spec=importlib.util.spec_from_file_location('orbit_minimal_v4',local);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        off_spec=importlib.util.spec_from_file_location('orbit_minimal_v4_off',off_source);off_module=importlib.util.module_from_spec(off_spec);off_spec.loader.exec_module(off_module)
        trace=TraceConfig(a.trace_bytes,str(out/'trace.txt'));module.aie_trace.trace_config=trace
        with open('/tmp/npu.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            data=np.arange(4096,dtype=np.int32)%97-48
            x=iron.tensor(data,dtype=np.int32,device='npu');factor=iron.full((1,),3,dtype=np.int32,device='npu');y=iron.zeros(4096,dtype=np.int32,device='npu')
            yoff=iron.zeros(4096,dtype=np.int32,device='npu')
            off_module.aie_trace(x,factor,yoff,tensor_size=4096,tile_size=1024,trace_size=a.trace_bytes)
            off_output=yoff.numpy();np.save(out/'output-off.npy',off_output)
            result['trace_off_numeric_passed']=bool(np.array_equal(off_output,data*3))
            start=time.perf_counter();module.aie_trace(x,factor,y,tensor_size=4096,tile_size=1024,trace_size=a.trace_bytes);result['initial_wall_ms']=(time.perf_counter()-start)*1000
            output=y.numpy();np.save(out/'output.npy',output);np.save(out/'expected.npy',data*3);result['numeric_passed']=bool(np.array_equal(output,data*3) and np.array_equal(off_output,output))
            manifest=decode(trace.read_trace(),trace.physical_mlir_path,out)
            manifest['expected_event_patterns']=['INSTR_EVENT_0','INSTR_EVENT_1']
            manifest['completeness']=dict(status='partial',scope='four marked scale calls',expected_marker_pairs_per_core=4,reason='pending independent raw-word count and terminal-marker audit')
            write_json(out/'manifest.json',manifest);result['status']='passed' if result['numeric_passed'] else 'numerical_failed'
    except Exception as exc:result.update(error=str(exc),traceback=traceback.format_exc())
    write_json(out/'result.json',result);return 0 if result['status']=='passed' else 1

if __name__=='__main__':raise SystemExit(main())
