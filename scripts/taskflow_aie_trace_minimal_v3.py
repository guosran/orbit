#!/usr/bin/env python3
"""Reproduce installed IRON event trace using the matching vendor example."""
import argparse,fcntl,importlib.util,shutil,time,traceback
from pathlib import Path
import numpy as np
import aie.iron as iron
from aie.utils.trace import TraceConfig
from taskflow_npu_trace_v3 import AIE,decode,digest
from taskflow_common import write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--trace-bytes',type=int,default=65536);a=p.parse_args();out=a.output_dir.resolve()
    if out.exists(): raise ValueError('output directory must be new')
    out.mkdir(parents=True);source=AIE/'vendor/mlir-aie/programming_examples/basic/event_trace/aie_trace.py';shutil.copy2(source,out/'design-source.py')
    result=dict(schema='orbit.hardware.minimal-trace.v1',status='compile_failed',design_source=str(source),design_sha256=digest(source),device='npu2',architecture='aie2p',initial_timing_boundary='compile+load+dispatch+readback; not a hot latency')
    try:
        spec=importlib.util.spec_from_file_location('vendor_event_trace',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        trace=TraceConfig(a.trace_bytes,str(out/'trace.txt'));module.aie_trace.trace_config=trace
        with open('/tmp/npu.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            data=np.arange(4096,dtype=np.int32)%97-48
            x=iron.tensor(data,dtype=np.int32,device='npu');factor=iron.full((1,),3,dtype=np.int32,device='npu');y=iron.zeros(4096,dtype=np.int32,device='npu')
            start=time.perf_counter();module.aie_trace(x,factor,y,tensor_size=4096,tile_size=1024);result['initial_wall_ms']=(time.perf_counter()-start)*1000
            output=y.numpy();np.save(out/'output.npy',output);np.save(out/'expected.npy',data*3);result['numeric_passed']=bool(np.array_equal(output,data*3))
            manifest=decode(trace.read_trace(),trace.physical_mlir_path,out)
            manifest['expected_event_patterns']=['INSTR_EVENT_0','INSTR_EVENT_1','PORT_RUNNING','LOCK_STALL']
            manifest['completeness']['scope_markers']='Vendor scale kernel emits INSTR_EVENT_0/1; terminal stall tracks may remain open, so capture remains partial'
            write_json(out/'manifest.json',manifest)
            result['status']='passed' if result['numeric_passed'] else 'numerical_failed'
    except Exception as exc:result.update(error=str(exc),traceback=traceback.format_exc())
    write_json(out/'result.json',result);return 0 if result['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
