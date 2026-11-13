#!/usr/bin/env python3
"""Preparation-only release/prefault before the unchanged decoder memory guard.

The seven sealed r19 runtime files and all vendor sources remain byte-for-byte
unchanged. The owned loader inserts one call after FusedDecoder construction,
before the original decode-loaded guard. It releases dead allocator pages and
reads one byte from every page of explicit live CPU weight owners. No weight,
BO, swap setting, kernel, inference timer, numerical rule or guard is changed.
"""
import argparse
import ast
import ctypes
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

def memory():
    result={}
    for line in Path('/proc/self/status').read_text().splitlines():
        key,_,value=line.partition(':')
        if key in ('VmRSS','VmHWM','VmSwap'):result[key+'_bytes']=int(value.split()[0])*1024
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):result['available_bytes']=int(line.split()[1])*1024
    usage=resource.getrusage(resource.RUSAGE_SELF)
    result.update(minor_faults=usage.ru_minflt,major_faults=usage.ru_majflt)
    return result


def weight_ranges(prefiller,decoder):
    """Collect actual CPU storage bounds; never inspect opaque BO internals."""
    import numpy as np
    import torch
    ranges=[];owners=[];visited=set()
    def visit(value,label):
        if id(value) in visited:return
        visited.add(id(value))
        if isinstance(value,torch.Tensor):
            if value.device.type!='cpu' or value.layout!=torch.strided:return
            storage=value.untyped_storage();start=storage.data_ptr();size=storage.nbytes()
            if start and size:ranges.append((int(start),int(start+size),label));owners.extend((value,storage))
        elif isinstance(value,np.ndarray):
            if value.dtype.hasobject or not value.size:return
            pointer=int(value.__array_interface__['data'][0]);lo=hi=0
            for extent,stride in zip(value.shape,value.strides):
                delta=(extent-1)*stride;lo+=min(delta,0);hi+=max(delta,0)
            ranges.append((pointer+lo,pointer+hi+value.itemsize,label));owners.append(value)
        elif isinstance(value,dict):
            for key,item in value.items():visit(item,label+'.'+str(key))
        elif isinstance(value,(list,tuple)):
            for index,item in enumerate(value):visit(item,label+'.'+str(index))
        elif hasattr(value,'padded'):
            # Original ResidentWeight: its actual owned Tensor, not a guessed
            # device address or a newly reconstructed/cast copy.
            visit(value.padded,label+'.padded')
    for name in ('_w','_orbit_gemm_weights','embed','lm_head','final_norm','_lut'):
        if hasattr(prefiller,name):visit(getattr(prefiller,name),'prefiller.'+name)
    ffn=getattr(prefiller,'_orbit_ffn',None)
    providers=getattr(ffn,'providers',{})
    if not providers and ffn is not None:providers={'default':ffn}
    for key,provider in providers.items():
        for name in ('weights','arguments'):
            if hasattr(provider,name):visit(getattr(provider,name),'ffn.'+str(key)+'.'+name)
    for name in ('Wv16','embed','rms_slabs','final_norm','bias'):
        if hasattr(decoder,name):visit(getattr(decoder,name),'decoder.'+name)
    merged=[]
    for lo,hi,label in sorted(ranges):
        if merged and lo<=merged[-1][1]:
            oldlo,oldhi,labels=merged[-1];merged[-1]=(oldlo,max(oldhi,hi),labels+[label])
        else:merged.append((lo,hi,[label]))
    return merged,owners


def release_and_prefault(prefiller,decoder,output,source):
    start=time.perf_counter_ns();before=memory();collected=gc.collect();trim=ctypes.CDLL('libc.so.6').malloc_trim(0)
    released=memory();ranges,owners=weight_ranges(prefiller,decoder);page=os.sysconf('SC_PAGE_SIZE');touches=0;checksum=0
    for lo,hi,labels in ranges:
        checksum^=ctypes.c_ubyte.from_address(lo).value;touches+=1
        address=(lo//page+1)*page
        while address<hi:
            checksum^=ctypes.c_ubyte.from_address(address).value;touches+=1;address+=page
    after_touch=memory();gc.collect();final_trim=ctypes.CDLL('libc.so.6').malloc_trim(0);after=memory()
    proof=dict(schema='orbit.qwen-r19-preparation-memory-release.v1',inside_request_timing=False,
        action='gc/malloc_trim; readonly live CPU weight storage prefault; gc/malloc_trim; original guard follows',
        owned_shim=dict(file=str(Path(__file__).resolve()),sha256=sha(__file__)),underlying_runner=source,
        before=before,after_dead_page_release=released,after_live_weight_prefault=after_touch,after=after,
        gc_collected=collected,malloc_trim_result=int(trim),final_malloc_trim_result=int(final_trim),
        page_bytes=page,pages_touched=touches,live_storage_union_bytes=sum(hi-lo for lo,hi,_ in ranges),
        storage_roots=[dict(bytes=hi-lo,owners=labels) for lo,hi,labels in ranges],
        elapsed_ns=time.perf_counter_ns()-start,reads_only=True,bo_writes=0,
        limitations='No swap configuration changes, no model copies, no guard relaxation; original guard may still reject actual live RSS/swap')
    path=Path(output)/'owned-decode-preparation.json'
    with path.open('x') as stream:json.dump(proof,stream,indent=2);stream.write('\n')
    protocol=dict(schema='orbit.qwen-r19-preparation-protocol.v1',mode='owned-decode-release-readonly-prefault-v1',producer=proof['owned_shim'],outside_request_timing=True,changes_guard=False,changes_timed_runtime=False,proof=dict(file=str(path.resolve()),sha256=sha(path)))
    with (Path(output)/'preparation-protocol.json').open('x') as stream:json.dump(protocol,stream,indent=2);stream.write('\n')
    # Keep owner references until every page read and its receipt are complete.
    return checksum,len(owners)


def instrument(runner):
    source=Path(runner.__file__);tree=ast.parse(source.read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    original_ast=ast.dump(main,include_attributes=False)
    matches=[]
    for node in ast.walk(main):
        if not isinstance(node,ast.With):continue
        for index,statement in enumerate(node.body):
            if (isinstance(statement,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='decoder' for t in statement.targets)
                and isinstance(statement.value,ast.Call) and isinstance(statement.value.func,ast.Attribute)
                and isinstance(statement.value.func.value,ast.Name) and statement.value.func.value.id=='air' and statement.value.func.attr=='FusedDecoder'):
                following=node.body[index+1]
                if not (isinstance(following,ast.Expr) and isinstance(following.value,ast.Call) and isinstance(following.value.func,ast.Name)
                    and following.value.func.id=='check_memory' and following.value.args and isinstance(following.value.args[0],ast.Constant)
                    and following.value.args[0].value=='decode-loaded'):raise ValueError('Frozen decoder guard adjacency changed')
                matches.append((node,index))
    if len(matches)!=1:raise ValueError('Expected exactly one original decoder construction/guard boundary')
    parent,index=matches[0];call=ast.parse('_r19_decode_preparation(pf,decoder,a.output)').body[0]
    ast.copy_location(call,parent.body[index]);parent.body.insert(index+1,call)
    fragment=ast.fix_missing_locations(ast.Module(body=[main],type_ignores=[]))
    record=dict(file=str(source.resolve()),sha256=sha(source),original_main_ast_sha256=hashlib.sha256(original_ast.encode()).hexdigest(),ast_change='one owned preparation call after original FusedDecoder constructor before unchanged decode-loaded guard')
    runner.__dict__['_r19_decode_preparation']=lambda pf,decoder,output:release_and_prefault(pf,decoder,output,record)
    exec(compile(fragment,str(source)+'.owned-r19-preparation-only','exec'),runner.__dict__)
    return runner


def main():
    parser=argparse.ArgumentParser(description=__doc__,add_help=False,allow_abbrev=False)
    parser.add_argument('--runtime-runner',type=Path,required=True)
    args,rest=parser.parse_known_args();runner=module(args.runtime_runner,'orbit_r19_preparation_memory_runner');loader=runner.module
    def owned_loader(path,name):
        result=loader(path,name)
        return instrument(result) if Path(path).name=='run-qwen-r18-runtime.py' else result
    runner.module=owned_loader;sys.argv=[str(args.runtime_runner),*rest];runner.main()


if __name__=='__main__':main()
