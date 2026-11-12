#!/usr/bin/env python3
"""CPU-only signed JSON control range contracts for the standalone CP frontend."""
import argparse
import json
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool', required=True)
    parser.add_argument('--solver-off', action='store_true')
    args = parser.parse_args()
    problem = dict(tasks=1, tiles=1, edges=[], patterns=[dict(
        id=0, tasks=[0], tile=0, cost_ns=1)])
    checks = 0
    for key, value in [('workers', -1), ('workers', 0), ('workers', 17),
                       ('workers', 4294967297), ('random_seed', -1),
                       ('random_seed', 2147483648), ('random_seed', 4294967296),
                       ('random_seed', 2147483647)]:
        payload = dict(problem, options=dict(workers=1, random_seed=0,
                                             time_limit_seconds=1))
        payload['options'][key] = value
        process = subprocess.run([args.tool], input=json.dumps(payload),
                                 text=True, capture_output=True, timeout=10)
        assert process.returncode == 0, process.stderr
        result = json.loads(process.stdout)
        valid = key == 'random_seed' and value == 2147483647
        expected = ('UNAVAILABLE' if args.solver_off else 'OPTIMAL') if valid else 'INVALID_INPUT'
        assert result['status'] == expected, (key, value, result)
        if not valid:
            assert 'workers must be in [1,16]' in result['message'], result
        checks += 1
    print(json.dumps(dict(status='passed', checks=checks, hardware_executed=False)))


if __name__ == '__main__':
    main()
