#!/usr/bin/env python3
"""Explicit CLI tiles must survive target defaults and region projection."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

from program_matrix_producer_v32 import produce


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool', type=Path, required=True)
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix='orbit-heterogeneous-cli-tiles-'))
    source = root / 'gemm32.mlir'
    source.write_text(produce(32, False))

    def export(name, inherited_tile, explicit):
        config = root / (name + '.json')
        config.write_text(json.dumps({
            'heterogeneous': {'task_devices': ['npu', 'npu'],
                              'implementation_owned': True},
            'generic_air': {'capacity_bytes': 24576,
                            'tile_sizes': [inherited_tile] * 3,
                            'capabilities': {'implementations': [
                                {'kind': 'scalar', 'lanes': 1},
                                {'kind': 'aie2p-bf16-mmul-f32acc', 'lanes': 1}]}}}))
        out = root / name
        command = [str(args.tool.resolve()), str(source),
                   '--target-config=' + str(config),
                   '--heterogeneous-output=' + str(out),
                   '--air-distribute-initialization',
                   '--implementation=aie2p-bf16-mmul-f32acc']
        if explicit is not None:
            command.extend('--tile-' + axis + '=' + str(value)
                           for axis, value in zip('mnk', explicit))
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        (root / (name + '.log')).write_text(result.stdout + result.stderr)
        return result, out

    receipts = []
    for name, inherited, selected in [('same', 16, [16]*3),
                                       ('override', 32, [16]*3),
                                       ('default', 32, None)]:
        result, out = export(name, inherited, selected)
        assert result.returncode == 0, (name, result.stderr)
        manifest = json.loads((out / 'manifest.json').read_text())
        physical = json.loads((out / 'region-0/program.air.mlir.physical.json').read_text())
        receipts.append((manifest['graph_id'], physical['graph_id']))
    # Check the actual parent and selected physical graph, not only the CLI
    # attribute: the original bug silently replaced tiles during child planning.
    assert receipts[0] == receipts[1], receipts
    assert receipts[1] != receipts[2], receipts
    result, _ = export('partial', 32, [16, 16, 0])
    assert result.returncode > 0 and 'must all be positive' in result.stderr
    print(json.dumps({'status': 'passed', 'checks': 6, 'output': str(root),
                      'hardware_executed': False}))


if __name__ == '__main__':
    main()
