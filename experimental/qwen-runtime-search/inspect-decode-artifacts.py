#!/usr/bin/env python3
"""Read actual decoder AIE buffers and bind SRAM/stack to physical tiles.

This uses the same static-buffer parsing rule as inspect-gemm-artifacts.py.
Only successful rotation-zero implementations are included; no compilation or
NPU invocation occurs here. SRAM buffers are invocation-local configuration
resources, not long-lived DDR weight/KV allocations.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re


def identity(path):
    return dict(file=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def inspect(directory,stack,capacities):
    primary=directory/'air_project/aie.air.mlir'
    text=primary.read_text()
    coordinates=set(tuple(map(int,x)) for x in re.findall(r'%core_(\d+)_(\d+) = aie.core',text))
    tiles=dict((name,tuple(map(int,(x,y)))) for name,x,y in re.findall(r'%(\w+) = aie.tile\((\d+), (\d+)\)',text))
    allocations=[];demands=defaultdict(int)
    for line in text.splitlines():
        if 'aie.buffer(' not in line:continue
        match=re.search(r'(%\w+) = aie.buffer\(%(\w+)\).*memref<([\dx]+)(bf16|f32|i32|i64|i16|i8)',line)
        if match is None:raise ValueError('Unparsed physical allocation: '+line)
        symbol,tile,shape,dtype=match.groups()
        dimensions=[int(d) for d in shape.rstrip('x').split('x')]
        size=math.prod(dimensions)*dict(bf16=2,f32=4,i32=4,i64=8,i16=2,i8=1)[dtype]
        demands[tile]+=size
        allocations.append(dict(symbol=symbol,tile=tile,coordinate=list(tiles[tile]),shape=dimensions,dtype=dtype,bytes=size))
    temporary={};resources={'npu.device':1,'npu.command':1,'cpu.host':1}
    for tile,amount in demands.items():
        x,y=tiles[tile];is_core=(x,y) in coordinates
        memory='npu.sram.'+tile
        capacity=capacities['core_local_memory_bytes'] if is_core else capacities['memtile_memory_bytes']
        if y==0:raise ValueError('Unexpected shim allocation')
        reserved=amount+(stack if is_core else 0)
        if reserved>capacity:raise ValueError(f'{tile} requires {reserved} bytes but target provides {capacity}')
        temporary[memory]=reserved
    for x,y in coordinates:
        resources[f'npu.core.{x}.{y}']=1
        memory=f'npu.sram.tile_{x}_{y}'
        if memory not in temporary:temporary[memory]=stack
    metadata=json.loads((directory/'compiled.json').read_text())
    return dict(name=directory.name,compiled_manifest=identity(directory/'compiled.json'),
        source_air=identity(directory/'source.air.mlir'),primary_aie=identity(primary),
        placed_aie=identity(directory/'air_project/placed.air.mlir'),
        actual_core_coordinates=[list(c) for c in sorted(coordinates)],
        compiler_sram_buffer_allocations=allocations,
        compiler_sram_buffer_bytes_by_tile=dict(sorted(demands.items())),
        stack_bytes_per_core=stack,
        execution_variant_fragment=dict(resource_demands=resources,temporary_bytes=temporary),
        max_core_buffer_bytes=max((size for tile,size in demands.items() if tiles[tile] in coordinates),default=0),
        max_memtile_buffer_bytes=max((size for tile,size in demands.items() if tiles[tile][1]==1),default=0),
        state_scope='physical decoder config SRAM active for the exclusive inclusive invocation; DDR W/RMS/X/Y/KV and host/template resident storage are separate',
        numerical_admission='root separately validated original/split logits and all KV; this ledger is physical storage evidence only')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--target-capacities',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();capacities=json.loads(a.target_capacities.read_text())
    rows=[inspect(a.root/f'decode-r0-cache{cache}-coalesce{coalesce}-L2048',10240,capacities) for cache in (0,1) for coalesce in (0,1)]
    result=dict(schema='orbit.qwen-decode-physical-resource-ledger.v1',producer=identity(Path(__file__)),
        target_capacities=identity(a.target_capacities),implementations=rows,
        admission='rotation-zero compiled configurations only; no rotation-four failed numerical variants',
        capacity_scope='known target SRAM capacities; DDR/host BO and template allocations must be added by runtime registration')
    with a.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps([dict(name=r['name'],cores=len(r['actual_core_coordinates']),max_core_buffer_bytes=r['max_core_buffer_bytes'],max_memtile_buffer_bytes=r['max_memtile_buffer_bytes']) for r in rows]))


if __name__=='__main__':main()
