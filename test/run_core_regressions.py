#!/usr/bin/env python3
"""Compiler-only tests; no builds, NumPy, artifact runtime imports, or devices."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'Support'))
from orbit_test_paths import CORE_ROOT, taskflow_tool, air_opt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool', type=Path, default=taskflow_tool())
    parser.add_argument('--amoeba-tool', type=Path, help='also test the original native search driver')
    parser.add_argument('--air-tool', type=Path, help='also run AIR code generation and legalization tests')
    parser.add_argument('--air-opt', type=Path, default=air_opt())
    parser.add_argument('--output', type=Path, required=True, help='new directory for logs and generated test inputs')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    tool = args.tool.resolve()
    fixtures = CORE_ROOT / 'test/TaskflowAIR/fixtures'
    records = []

    def run(name, command):
        command = list(map(str, command))
        (out / (name + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
        result = subprocess.run(command, cwd=CORE_ROOT, capture_output=True, text=True)
        (out / (name + '.stdout')).write_text(result.stdout)
        (out / (name + '.stderr')).write_text(result.stderr)
        records.append(dict(name=name, exit_code=result.returncode))
        (out / 'result.json').write_text(json.dumps(dict(status='FAIL' if result.returncode else 'RUNNING', tests=records), indent=2) + '\n')
        if result.returncode:
            raise RuntimeError(f'{name} failed ({result.returncode}): {result.stderr[-3000:]}')
        print(f'{name}: PASS', flush=True)

    def python(name, relative, selected_tool=tool, *extra):
        run(name, [sys.executable, CORE_ROOT / 'test' / relative,
                   '--tool', selected_tool, '--output', out / name, *extra])

    def checker(name, *extra):
        run(name, [tool.parent / name, *extra])

    run('core', [sys.executable, CORE_ROOT / 'test/TaskflowCore/check.py', '--tool', tool])
    python('generic-analysis', 'TaskflowGeneric/analysis.py')
    python('generic-search', 'TaskflowGeneric/search.py')
    python('distribution', 'TaskflowGeneric/distribution_v6.py')
    python('unsupported-core', 'TaskflowGeneric/unsupported_v6.py', tool, '--skip-air')
    checker('taskflow-generic-mapping-test', fixtures / 'add-scale.mlir')
    checker('taskflow-tiled-mapping-test', out / 'distribution/gemm.transformed.mlir')
    checker('taskflow-program-mapping-test', fixtures / 'gemm.mlir')
    checker('taskflow-program-partition-test')
    checker('taskflow-data-organization-test')
    checker('taskflow-direct-air-cost-test', CORE_ROOT / 'test/TaskflowGeneric/add.mlir')
    python('static-mapping', 'TaskflowMapping/cpp_differential.py')
    python('static-regions', 'TaskflowMapping/regions.py')
    python('search-preferences', 'TaskflowMapping/search_preferences.py')
    python('search-contract', 'TaskflowValidation/search_contract.py')
    python('data-policy', 'TaskflowAIR/data_organization_v8.py')
    python('joint-data-search', 'TaskflowAIR/joint_data_search_v8.py')
    if args.amoeba_tool:
        python('original-native-search', 'TaskflowMapping/native_search.py', tool,
               '--amoeba-tool', args.amoeba_tool.resolve())
    if args.air_tool:
        air = args.air_tool.resolve()
        python('air-lowering', 'TaskflowAIR/lowering.py', air, '--air-opt', args.air_opt)
        python('air-tiling', 'TaskflowAIR/tiling.py', air, '--air-opt', args.air_opt)
        python('air-mapping', 'TaskflowAIR/mapping_v7.py', air, '--air-opt', args.air_opt)
        python('air-channels', 'TaskflowAIR/channels_v6.py', air, '--air-opt', args.air_opt)
        python('air-policy', 'TaskflowAIR/data_organization_v8.py', air, '--air')
        python('air-joint-data-search', 'TaskflowAIR/joint_data_search_v8.py', air, '--air')
        python('air-unsupported', 'TaskflowGeneric/unsupported_v6.py', air)
        python('air-pool-names', 'TaskflowGeneric/pool_names_v6.py', air)
        python('air-mixed-f32', 'TaskflowAIR/mixed_f32_v9.py', air, '--air-opt', args.air_opt)
        python('air-stream-capacity', 'TaskflowAIR/stream_capacity_v9.py', air)
        def shaped_source(template, m, k, n):
            import re
            text = (fixtures / template).read_text()
            shapes = {'4x8': f'{m}x{k}', '8x4': f'{k}x{n}', '4x4': f'{m}x{n}'}
            text = re.sub(r'(?<=memref<)\d+x\d+(?=xf32)', lambda match: shapes[match[0]], text)
            extents = {'i': m, 'j': n, 'k': k}
            return re.sub(r'affine.for %(i|j|k) = 0 to \d+',
                          lambda match: f'affine.for %{match[1]} = 0 to {extents[match[1]]}', text)
        tiled = out / 'tiled-source.mlir'
        tiled.write_text(shaped_source('gemm.mlir', 16, 128, 64))
        program = out / 'program-source.mlir'
        program.write_text(shaped_source('gemm-scale.mlir', 32, 128, 64))
        python('air-tiled-search', 'TaskflowGeneric/tiled_search_v6.py', air, '--source', tiled)
        python('air-program-search', 'TaskflowGeneric/program_search_v7.py', air, '--source', program)
        python('air-partition-search', 'TaskflowGeneric/program_partition_search_v7.py', air, '--source', program)
    identities = {}
    for name, path in [('core', tool), ('air', args.air_tool), ('amoeba', args.amoeba_tool)]:
        if path:
            path = Path(path).resolve()
            identities[name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (out / 'result.json').write_text(json.dumps(dict(status='PASS', hardware_executed=False,
        tests=records, tools=identities, optional_backends=dict(air=bool(args.air_tool),
        amoeba=bool(args.amoeba_tool))), indent=2) + '\n')


if __name__ == '__main__':
    main()
