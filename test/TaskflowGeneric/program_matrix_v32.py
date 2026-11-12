#!/usr/bin/env python3
"""CPU-only BF16 source/native implementation contracts; no cost assertions."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

from program_matrix_producer_v32 import produce


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', type=Path, required=True)
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix='orbit-program-matrix-v32-'))
    for name, size, chain in [('gemm16', 16, False), ('gemm64', 64, False),
                              ('chain64', 64, True), ('chain128', 128, True)]:
        source = root / (name + '.mlir')
        source.write_text(produce(size, chain))
        output = root / name
        command = [str(args.test.resolve()), str(source), str(output)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        (root / (name + '.log')).write_text(result.stdout + result.stderr)
        assert result.returncode == 0, (name, result.stdout, result.stderr)
        manifest = json.loads((output / 'manifest.json').read_text())
        receipt = manifest['matrix_realization']
        assert receipt['calls'] > 0 and receipt['physical_scratch_bytes'] > 0
        assert receipt['additional_host_transfer_bytes'] == 0
        assert not receipt['final_aie_kernel_numeric_certificate']
        assert all(k['target_arch'] == 'aie2p' for k in receipt['kernels'])
        # The rounding bridge is an actual source task and scalar instance,
        # and both matrix tasks get independently selected implementations.
        if chain:
            plan = json.loads((output / 'plan.json').read_text())
            implementations = plan['task_implementations']
            assert sum(x['kind'] == 'aie2p-bf16-mmul-f32acc'
                       for x in implementations) == 2
            assert any(x['kind'] == 'scalar' for x in implementations)
    print(json.dumps(dict(status='passed', cases=4, output=str(root),
                          hardware_executed=False,
                          scope='source/mapping/AIR only; no numerical or speed claim')))


if __name__ == '__main__':
    main()
