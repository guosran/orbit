#!/usr/bin/env python3
"""Create a singleton receipt-export domain for the BF16 matrix+relu family.

This is not an optimization comparison. Always compare exported parent and
ordered child graph identities against the actual compiled preparation.
"""
import argparse
import json
from pathlib import Path


def configuration(base, task_count, storage_count, device, tile, distribute):
    tiles = (tile, tile, tile) if type(tile) is int else tuple(tile)
    if len(tiles) != 3 or any(type(value) is not int or value < 1 for value in tiles):
        raise ValueError('three positive integer M/N/K tiles required')
    dimensions = {}

    def fixed(identifier, binding=None):
        dimensions[identifier] = dict(id=identifier, choices=[binding])

    # Null bindings inherit the checked source/default policy. Explicitly
    # replace every automatically generated dimension, rather than cap search
    # after the first point of a much larger domain.
    for identifier in (
        'fusion.residency.traversal', 'replica.partition', 'copies.retention',
        'data.regions', 'dependencies.implementation_serialization',
        'buffering.slots', 'temporal.priority', 'layout.dtype',
    ):
        fixed(identifier)
    fixed('source.fission', dict(fission=False))
    fixed('source.initialization', dict(distribute=distribute))
    fixed('implementation', dict(implementation='aie2p-bf16-mmul-f32acc', lanes=1))
    for axis, name in enumerate(('m', 'n', 'k')):
        fixed('tiling.' + name, dict(tile_axis=axis, tile=tiles[axis]))
    for task in range(task_count):
        for name in ('implementation', 'resource', 'memory', 'count', 'shape'):
            fixed(f'task.{task}.{name}')
        fixed(f'device.task.{task}', dict(task=task, device=device))
        if task:
            fixed(f'boundary.before.{task}', dict(before_task=task, cut=False))
    for task in range(task_count + 1):
        for storage in range(storage_count):
            fixed(f'task.{task}.data.{storage}.route')
    base['native_auto_candidates'] = dict(mode='joint', algorithm='cp-sat-all')
    base['program_search'] = dict(
        heterogeneous=True, seconds=120, max_evaluations=1,
        max_source_instances=100000, max_values_per_dimension=1,
        max_routes=1, max_priorities=1, dimensions=list(dimensions.values()),
    )
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tasks', type=int, required=True)
    parser.add_argument('--storages', type=int, required=True)
    parser.add_argument('--device', choices=('cpu', 'npu'), required=True)
    tile_options = parser.add_mutually_exclusive_group(required=True)
    tile_options.add_argument('--tile', type=int,
                              help='use the same tile extent on all three axes')
    tile_options.add_argument('--tiles', type=int, nargs=3, metavar=('M', 'N', 'K'),
                              help='independent M/N/K tile extents')
    parser.add_argument('--distribute', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new output only')
    if args.tasks < 1 or args.storages < 1:
        raise ValueError('positive task/storage counts required')
    tiles = args.tiles if args.tiles is not None else args.tile
    result = configuration(json.loads(args.base.read_text()), args.tasks,
                           args.storages, args.device, tiles, args.distribute)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
