#!/usr/bin/env python3
"""CPU contracts for exact public-graph AIR replay; no hardware or full search.

Each explicit singleton domain is evaluated once, frozen, then replayed without
search. Fixture target capacities/prices are synthetic test assumptions.
"""
import argparse
import copy
import json
from pathlib import Path
import re
import subprocess
import tempfile

FIXTURES = Path(__file__).resolve().parent / 'fixtures' / 'oracle-v26'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def freeze_config(name, full=False, shared=False):
    cfg = read(FIXTURES / (name + '.json'))
    dimensions = cfg['program_search']['dimensions']
    for dim in dimensions:
        choices = dim['choices']
        selected = 0
        if full and dim['id'] in {'tiling.m', 'tiling.n'}:
            selected = 1
        if full and dim['id'] == 'fusion.residency.traversal':
            selected = 2 if shared else 1
        if full and dim['id'].endswith('.placement'):
            selected = 0 if shared and dim['id'] == 'task.0.placement' else 1
        dim['choices'] = [copy.deepcopy(choices[selected])]
    dimensions += [
        dict(id='copies.retention', choices=[dict(
            policy='retain-across-users' if full else 'phase-local')]),
        dict(id='buffering.slots', choices=[dict(reuse_slots=full)]),
    ]
    cfg['program_search'].update(seconds=5, max_evaluations=1)
    return cfg


def gemm_config():
    cfg = freeze_config('fork-shared-intermediate')
    cfg['generic_air'].pop('memory_architecture')
    cfg['generic_air']['capacity_bytes'] = 24576
    decisions = [
        ('source.fission', dict(fission=False)),
        ('source.initialization', dict(distribute=True)),
        ('implementation', dict(implementation='scalar', lanes=1)),
        ('replica.partition', dict(partition=0)),
        ('temporal.priority', dict(priority=[0])),
        ('tiling.m', dict(tile_axis=0, tile=4)),
        ('tiling.n', dict(tile_axis=1, tile=4)),
        ('tiling.k', dict(tile_axis=2, tile=4)),
        ('copies.retention', dict(policy='phase-local')),
        ('buffering.slots', dict(reuse_slots=False)),
        ('fusion.residency.traversal', dict(mapping=dict(
            residency='phase-local', grouping='task-shape-stages',
            data_organization='phase-local', groups=[]))),
    ]
    cfg['program_search']['dimensions'] = [
        dict(id=k, choices=[v]) for k, v in decisions]
    return cfg


def array(line, key):
    m = re.search(re.escape(key) + r' = array<i64: ([^>]+)>', line)
    assert m, (key, line)
    return [int(x.strip()) for x in m[1].split(',')]


def check_lower(evaluation, lowered):
    graph = evaluation['program_graph']
    ledger = graph['ledger']
    manifest = read(lowered / 'manifest.json')
    air = (lowered / 'program.air.mlir').read_text()
    assert manifest['graph_id'] == graph['id']
    assert manifest['graph_copy_compute_realization'] is True
    assert manifest['analytical_group_setups_realized'] is False
    assert manifest['final_aie_checked'] is False
    assert manifest['physical_configurations'] == 1
    assert 'no timed-dispatch claim' in manifest['scope']
    assert air.count('"air.launch"') == air.count('"air.segment"') == 1
    assert air.count('"air.herd"') == len(manifest['cores'])
    assert sorted(e['event'] for e in manifest['events']) == list(range(len(ledger['events'])))
    # Independently inspect every actual serialized DMA, with no traffic added,
    # omitted, coalesced, or inferred from the old native reference.
    expected = {x['event']: x for x in graph['communications']}
    found = {}
    for line in air.splitlines():
        if '"air.dma_memcpy_nd"' not in line:
            continue
        event = int(re.search(r'orbit.public_event = (\d+) : i64', line)[1])
        assert event not in found
        found[event] = line
        transfer = expected[event]
        assert array(line, 'static_src_sizes') == transfer['sizes']
        assert array(line, 'static_dst_sizes') == transfer['sizes']
        host_side = 'src' if 'orbit.transfer = "host_to_l1"' in line else 'dst'
        assert array(line, 'static_' + host_side + '_offsets') == transfer['offsets']
    assert set(found) == set(expected)
    # Match actual physical pool SSA roots/offsets, then prove all selected
    # logical allocation ranges are disjoint and exactly fill the same payload.
    memories = {x['id']: x for x in ledger['memory_architecture']['memories']}
    selected = {i: x for i, x in enumerate(ledger['allocations'])
                if memories[x['memory']]['kind'] == 'local-sram'}
    bindings = {x['allocation']: x for x in manifest['allocations']}
    assert bindings.keys() == selected.keys()
    physical = {}
    actual_bindings = {}
    for line in air.splitlines():
        if '"memref.alloc"' not in line:
            continue
        root = re.search(r'(%\d+) = "memref.alloc"', line)[1]
        count = int(re.search(r'memref<(\d+)xf32', line)[1])
        physical[root] = count * 4
        ids, offsets = array(line, 'orbit.allocations'), array(line, 'orbit.allocation_offsets_elements')
        assert len(ids) == len(offsets)
        ranges = []
        for allocation, offset in zip(ids, offsets):
            assert allocation not in actual_bindings
            binding = bindings[allocation]
            assert binding['bytes'] == selected[allocation]['bytes']
            assert binding['offset_elements'] == offset
            assert offset >= 0 and offset * 4 + binding['bytes'] <= count * 4
            actual_bindings[allocation] = (root, offset)
            ranges.append((offset * 4, offset * 4 + binding['bytes']))
        ranges.sort()
        assert ranges[0][0] == 0 and ranges[-1][1] == count * 4
        assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
    assert actual_bindings.keys() == selected.keys()
    assert sum(physical.values()) == sum(x['bytes'] for x in selected.values())
    for line in air.splitlines():
        if '"memref.reinterpret_cast"' not in line:
            continue
        allocation = int(re.search(r'orbit.allocation = (\d+) : i64', line)[1])
        root = re.search(r'"memref.reinterpret_cast"\((%\d+)\)', line)[1]
        assert (root, array(line, 'static_offsets')[0]) == actual_bindings[allocation]
    return manifest, air


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.output or Path(tempfile.mkdtemp(prefix='orbit-public-air-v27-')) / 'results'
    root.mkdir(parents=True, exist_ok=False)
    results = []

    def run(label, fixture, cfg, rejection=False):
        folder = root / label
        folder.mkdir()
        source = folder / 'source.mlir'
        source.write_text((FIXTURES / (fixture + '.mlir')).read_text())
        target = folder / 'target.json'
        write(target, cfg)
        search = folder / 'search'
        command = [str(args.tool.resolve()), str(source), '--search', '--backend=air',
                   '--max-rounds=1', '--max-candidates=32',
                   '--target-config=' + str(target), '--search-output=' + str(search)]
        write(folder / 'evaluation-command.json', command)
        p = subprocess.run(command, capture_output=True, text=True, timeout=30)
        (folder / 'evaluation.log').write_text(p.stdout + p.stderr)
        assert p.returncode == 0, (label, p.returncode, p.stderr)
        report = read(search / 'search.json')
        assert report['evaluations'] == 1 and report['has_best']
        assert report['physical_realization_checked'] is False
        evaluation = read(search / 'best.json')
        assert evaluation['estimated_ms'] is not None
        lowered = folder / 'lowered'
        replay = [str(args.tool.resolve()), str(search / 'best.mlir'),
                  '--target-config=' + str(target),
                  '--public-program-evaluation=' + str(search / 'best.json'),
                  '--public-program-space=' + str(search / 'space.json'),
                  '--public-program-assignment=' + str(search / 'best-assignment.json'),
                  '--implementation-output=' + str(folder / 'replay.json'),
                  '--lower-public-program-to-air=' + str(lowered)]
        write(folder / 'replay-command.json', replay)
        p = subprocess.run(replay, capture_output=True, text=True, timeout=30)
        (folder / 'lower.log').write_text(p.stdout + p.stderr)
        assert p.returncode >= 0, (label, 'compiler crashed', p.stderr)
        if rejection:
            assert p.returncode != 0 and 'static selected allocation sum exceeds' in p.stderr, p.stderr
            assert not (lowered / 'manifest.json').exists()
            results.append(dict(case=label, rejected=True))
            return None
        assert p.returncode == 0, (label, p.stderr)
        receipt = read(lowered / 'replay.json')
        assert receipt['valid'] and receipt['complete_graph_matches']
        assert receipt['analytical_timing_matches'] and receipt['search_performed'] is False
        manifest, air = check_lower(evaluation, lowered)
        results.append(dict(case=label, allocations=len(manifest['allocations']),
                            dma_count=air.count('"air.dma_memcpy_nd"'),
                            local_staging_bytes=manifest['local_staging_payload_bytes']))
        return evaluation, manifest, air

    evaluation, manifest, air = run('fork-baseline', 'fork-shared-intermediate',
                                   freeze_config('fork-shared-intermediate'))
    assert len(evaluation['program_graph']['communications']) == 12
    assert len(manifest['allocations']) == 16 and air.count('"memref.alloc"') == 4
    assert manifest['local_staging_payload_bytes'] == 0
    evaluation, manifest, air = run('fork-full-core1', 'fork-shared-intermediate',
                                   freeze_config('fork-shared-intermediate', full=True))
    assert len(manifest['cores']) == 1 and manifest['cores'][0]['column'] == 1
    assert len(evaluation['program_graph']['instances']) == 3
    assert all([d['upper'] - d['lower'] for d in i['domain']] == [4, 8]
               for i in evaluation['program_graph']['instances'])
    assert len(evaluation['program_graph']['communications']) == 3
    shared_source = (FIXTURES / 'shared-input-three-branches.mlir').read_text()
    assert 'memref.alloc' not in shared_source
    evaluation, manifest, air = run('shared-no-source-alloc', 'shared-input-three-branches',
                                   freeze_config('shared-input-three-branches', full=True, shared=True))
    assert sorted(c['column'] for c in manifest['cores']) == [0, 1]
    assert len(evaluation['program_graph']['communications']) == 5
    evaluation, manifest, air = run('gemm-initialize-rmw', 'gemm-reduction', gemm_config())
    assert manifest['local_staging_payload_bytes'] == 128
    assert manifest['analytical_local_staging_charged'] is False
    assert len(manifest['local_staging']) == 2
    assert 'orbit.source_initializer' in air and 'orbit.local_version_staging' in air
    assert any(i['kind'] == 'source-initializer' for i in evaluation['program_graph']['instances'])
    allocations = evaluation['program_graph']['ledger']['allocations']
    for staging in manifest['local_staging']:
        assert staging['bytes'] == 64 and staging['sizes'] == [4, 4]
        assert staging['source_allocation'] != staging['target_allocation']
        assert allocations[staging['source_allocation']]['storage'] == staging['storage']
        assert allocations[staging['target_allocation']]['storage'] == staging['storage']
    run('capacity-rejected', 'fork-capacity-boundary',
        freeze_config('fork-capacity-boundary'), rejection=True)
    write(root / 'summary.json', dict(status='passed', cases=results,
          hardware_executed=False, final_aie_checked=False))
    print(json.dumps(dict(status='passed', cases=len(results), output=str(root))))


if __name__ == '__main__':
    main()
