#!/usr/bin/env python3
"""Audit static calibration designs, never timing labels or device execution.

Accept an explicit feature_sets/cases declaration. Each case has id, split and
features:{path,sha256}; native features must retain their exact source family.
Convex hull is a diagnostic of joint support, not an added acceptance gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog


def audit(spec_path):
    spec_path = Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text())
    policy = spec['feature_sets']
    buckets = {}
    for case in spec['cases']:
        reference = case['features']
        path = Path(reference['path'])
        path = path if path.is_absolute() else spec_path.parent / path
        if hashlib.sha256(path.read_bytes()).hexdigest() != reference['sha256']:
            raise ValueError('feature receipt SHA changed: ' + str(path))
        data = json.loads(path.read_text())
        if data.get('schema') != 'orbit.heterogeneous.edge-features.v32':
            raise ValueError('native C++ edge feature schema required')
        def append(key, phase, names, values):
            vector = [values[name] for name in names]
            if any(type(v) not in (int, float) or not np.isfinite(v) or v < 0 for v in vector):
                raise ValueError('nonfinite, negative or nonnumeric feature')
            buckets.setdefault((key, phase, tuple(names)), []).append((case['id'], case['split'], vector))
        for phase, names in policy['application'].items():
            append(('application',), phase, names, data['application'])
        for region in data['regions']:
            key = (region['device'], region['source_family'], *sorted(region['implementations']))
            for phase, names in policy[region['device']].items():
                append(key, phase, names, region['features'])
    result = []
    for (key, phase, names), observations in sorted(buckets.items()):
        train = [v for _, split, v in observations if split == 'train']
        row = dict(local_family=list(key), phase=phase, features=list(names), rows=len(train), heldout=[])
        if not train:
            row['status'] = 'unknown: no training family'
            result.append(row)
            continue
        x = np.asarray(train, dtype=float)
        scale = np.maximum(x.max(axis=0), 1)
        scaled = x / scale
        rank = int(np.linalg.matrix_rank(scaled))
        sv = np.linalg.svd(scaled, compute_uv=False)
        row.update(unique_designs=len(np.unique(x, axis=0)), rank=rank,
                   scaled_condition=float(sv[0] / sv[-1]) if rank == len(names) else None,
                   coefficient_identifiable=rank == len(names))
        for case_id, split, vector in observations:
            if split != 'holdout':
                continue
            target = np.asarray(vector, dtype=float)
            hull = linprog(np.zeros(len(x)), A_eq=np.vstack([scaled.T, np.ones(len(x))]),
                           b_eq=np.r_[target / scale, 1], bounds=(0, None), method='highs')
            row['heldout'].append(dict(case=case_id, values=vector,
                within_individual_ranges=bool(np.all(target >= x.min(axis=0)) and np.all(target <= x.max(axis=0))),
                within_joint_convex_hull=bool(hull.success),
                convex_hull_status=int(hull.status)))
        result.append(row)
    return dict(schema='orbit.heterogeneous.static-design-audit.v32', designs=result,
                scope='static features only; rank and convex hull are diagnostics, not physical cost identification or new acceptance gates',
                timing_labels_opened=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new output only')
    args.output.write_text(json.dumps(audit(args.spec), indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
