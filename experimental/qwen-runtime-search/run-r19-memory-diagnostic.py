#!/usr/bin/env python3
"""Diagnostic-only storage/smaps attribution around unchanged preparation guards.

Runs the frozen v2 preparation first, then inspects actual Python-owned CPU
storages and optionally reads their pages. This is not a training protocol;
its outputs cannot be admitted as v2 prices. No allocator/swap policy changes.
"""
import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
import types


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def collect(prefiller,decoder):
    """Inspect root instance dictionaries and containers, never opaque methods.

    Object graphs beyond the explicit owned FFN provider dictionaries are not
    followed. In particular a pyxrt BO is not introspected or mapped here.
    """
    import numpy as np
    import torch
    ranges=[];owners=[];seen=set();skipped={}
    def visit(value,label):
        if id(value) in seen:return
        seen.add(id(value))
        if isinstance(value,torch.Tensor):
            if value.device.type!='cpu' or value.layout!=torch.strided:return
            storage=value.untyped_storage();lo=int(storage.data_ptr());size=int(storage.nbytes())
            if lo and size:ranges.append((lo,lo+size,label));owners.extend((value,storage))
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
        elif isinstance(value,(types.ModuleType,type)) or callable(value):return
        else:
            # ResidentWeight is a plain dataclass. Read its actual __dict__;
            # do not invoke properties on unknown extension/BO objects.
            fields=vars(value) if type(value).__name__=='ResidentWeight' else None
            if isinstance(fields,dict):visit(fields,label)
            elif not isinstance(value,(str,int,float,bool,type(None))):
                key=type(value).__module__+'.'+type(value).__name__;skipped[key]=skipped.get(key,0)+1
    for name,root in [('prefiller',prefiller),('decoder',decoder)]:
        if root is not None:visit(vars(root),name)
    ffn=vars(prefiller).get('_orbit_ffn')
    if ffn is not None:
        providers=vars(ffn).get('providers',{'default':ffn})
        for key,provider in providers.items():visit(vars(provider),'ffn.'+str(key))
    merged=[]
    for lo,hi,label in sorted(ranges):
        if merged and lo<=merged[-1][1]:
            left,right,labels=merged[-1];merged[-1]=(left,max(right,hi),labels+[label])
        else:merged.append((lo,hi,[label]))
    return merged,owners,skipped


def smaps(ranges,limit=40):
    regions=[];current=None
    for line in Path('/proc/self/smaps').read_text().splitlines():
        match=re.match(r'^([0-9a-f]+)-([0-9a-f]+)\s+(\S+)\s+\S+\s+\S+\s+\d+\s*(.*)$',line)
        if match:
            current=dict(start=int(match[1],16),end=int(match[2],16),permissions=match[3],path=match[4],fields={});regions.append(current)
        elif current is not None:
            key,_,value=line.partition(':')
            if key in ('Size','Rss','Pss','Private_Clean','Private_Dirty','Anonymous','Swap','SwapPss','Locked'):
                current['fields'][key+'_bytes']=int(value.split()[0])*1024
    for region in regions:
        region['storage_overlaps']=[dict(overlap_bytes=min(hi,region['end'])-max(lo,region['start']),owners=labels)
            for lo,hi,labels in ranges if lo<region['end'] and hi>region['start']]
    swapped=[r for r in regions if r['fields'].get('Swap_bytes',0)]
    return dict(total_swap_bytes=sum(r['fields'].get('Swap_bytes',0) for r in regions),
        total_rss_bytes=sum(r['fields'].get('Rss_bytes',0) for r in regions),region_count=len(regions),
        swapped_region_count=len(swapped),top_swap_regions=sorted(swapped,key=lambda r:r['fields']['Swap_bytes'],reverse=True)[:limit])


def diagnose(preparation,prefiller,decoder,output,stage,touch):
    start=time.perf_counter_ns();ranges,owners,skipped=collect(prefiller,decoder)
    explicit,explicit_owners=preparation.weight_ranges(prefiller,decoder)
    missing=[]
    for lo,hi,labels in ranges:
        covered=sum(max(0,min(hi,b)-max(lo,a)) for a,b,_ in explicit)
        if covered<hi-lo:missing.append(dict(start=lo,end=hi,bytes=hi-lo,not_in_explicit_roots_bytes=hi-lo-covered,owners=labels))
    before=preparation.memory();maps_before=smaps(ranges);pages=0;checksum=0
    if touch:
        page=os.sysconf('SC_PAGE_SIZE')
        for lo,hi,_ in ranges:
            checksum^=ctypes.c_ubyte.from_address(lo).value;pages+=1;address=(lo//page+1)*page
            while address<hi:checksum^=ctypes.c_ubyte.from_address(address).value;pages+=1;address+=page
    result=dict(schema='orbit.qwen-r19-diagnostic-memory-ownership.v1',diagnostic_only=True,stage=stage,
        producer=dict(file=str(Path(__file__).resolve()),sha256=preparation.sha(__file__)),
        preparation_shim=dict(file=str(Path(preparation.__file__).resolve()),sha256=preparation.sha(preparation.__file__)),
        outside_request_timing=True,guard_changed=False,allocator_policy_changed=False,reads_only=True,bo_writes=0,
        all_collected_storage_bytes=sum(hi-lo for lo,hi,_ in ranges),explicit_storage_bytes=sum(hi-lo for lo,hi,_ in explicit),
        missing_from_explicit_roots=missing,skipped_opaque_types=skipped,before=before,smaps_before=maps_before,
        after=preparation.memory(),smaps_after=smaps(ranges),readonly_prefault_enabled=touch,pages_touched=pages,
        elapsed_ns=time.perf_counter_ns()-start,
        limitations='smaps mapping overlap identifies owners within a VMA, not which individual pages are swapped; unlabeled anonymous pages may include allocator arenas or skipped owners')
    target=Path(output)/('diagnostic-'+stage+'.json')
    with target.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(diagnostic_memory_stage=stage,before_swap=before['VmSwap_bytes'],after_swap=result['after']['VmSwap_bytes'],missing_storage_bytes=sum(r['not_in_explicit_roots_bytes'] for r in missing))),flush=True)
    return checksum,len(owners)+len(explicit_owners)


def main():
    parser=argparse.ArgumentParser(description=__doc__,add_help=False,allow_abbrev=False)
    parser.add_argument('--runtime-runner',type=Path,required=True)
    parser.add_argument('--preparation-shim',type=Path,required=True)
    parser.add_argument('--diagnostic-readonly-prefault',action='store_true')
    args,rest=parser.parse_known_args();prep=module(args.preparation_shim,'diagnostic_sealed_preparation')
    runner=module(args.runtime_runner,'diagnostic_ready_runner');loader=runner.module
    def owned_loader(path,name):
        result=loader(path,name)
        if Path(path).name=='run-qwen-r18-runtime.py':
            result=prep.instrument(result);original=result.__dict__['_r19_owned_preparation']
            def callback(pf,decoder,output,stage):
                original(pf,decoder,output,stage)
                return diagnose(prep,pf,decoder,output,stage,args.diagnostic_readonly_prefault)
            result.__dict__['_r19_owned_preparation']=callback
        return result
    runner.module=owned_loader;sys.argv=[str(args.runtime_runner),*rest];runner.main()


if __name__=='__main__':main()
