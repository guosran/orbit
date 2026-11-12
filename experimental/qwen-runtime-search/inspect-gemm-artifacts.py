#!/usr/bin/env python3
"""Extract actual compiled GEMM SRAM and placement demands; never load a device."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re


def file_record(p):
    return dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def inspect(root):
    rows = []
    for manifest in sorted(root.glob('*/implementation.json')):
        implementation = json.loads(manifest.read_text())
        allocations, demands = [], defaultdict(int)
        for source in sorted((manifest.parent/'air_project').glob('aie.*.mlir')):
            for line in source.read_text().splitlines():
                if 'aie.buffer(' not in line:
                    continue
                match = re.search(r'(%\w+) = aie.buffer\(%(\w+)\).*memref<([\dx]+)(bf16|f32|i32|i64|i16|i8)', line)
                if match is None:
                    raise ValueError('unparsed physical allocation: '+line)
                symbol, tile, shape, dtype = match.groups()
                dimensions = [int(d) for d in shape.rstrip('x').split('x')]
                size = math.prod(dimensions)*dict(bf16=2,f32=4,i32=4,i64=8,i16=2,i8=1)[dtype]
                demands[tile] += size
                allocations.append(dict(symbol=symbol, tile=tile, shape=dimensions,
                                        dtype=dtype, bytes=size))
        rows.append(dict(implementation=file_record(manifest), name=manifest.parent.name,
            shape=implementation['shape'], tiling=implementation['tiling'],
            resource=implementation['resource'], abi=implementation['abi'],
            compiler_sram_buffer_allocations=allocations,
            compiler_sram_buffer_bytes_by_tile=dict(sorted(demands.items())),
            stack_bytes_per_core=2048,
            physical_evidence=implementation['physical_evidence'],
            exclusive_resources=[dict(id='npu.command', units=1)],
            valid_only_after_device_equivalence_and_price=True))
    return dict(schema='orbit.qwen-gemm-physical-resource-ledger.v1',
        producer=file_record(Path(__file__)), implementations=rows,
        resource_interpretation='core coordinates and SRAM bytes are extracted from compiled AIE; reserve the full native invocation exclusively',
        supported_placement_domain='contiguous rectangular herds at row 2, with explicit column offset; non-rectangular/row remapping is not generated',
        supported_replication_domain='one instance or two independent output-N shard instances in one launch/segment; no independent host-kernel concurrency')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    data=inspect(a.root)
    with a.output.open('x') as stream:
        json.dump(data,stream,indent=2)
        stream.write('\n')
    print(json.dumps(dict(implementations=len(data['implementations']),output=str(a.output))))


if __name__=='__main__':
    main()
