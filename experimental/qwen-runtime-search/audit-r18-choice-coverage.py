#!/usr/bin/env python3
"""Confirm registered choices entered the real, completed ORBIT evaluator.

The initial complete neighborhood suffices to witness every local choice.
This audits recorded assignments, not just catalogue field names. The bound
below is computed after search and never feeds candidate selection.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(report_path, catalogue_path):
    report, catalogue = load(report_path), load(catalogue_path)
    require(report['selected_graph_independently_checked'], 'No checked final graph')
    search = report['report']
    require(not search['budget_exhausted'] and search['max_rounds'] is None
            and search['max_evaluations'] is None and search['wall_seconds'] is None,
            'This audit requires the completed, uncapped search')
    groups = [g for g in catalogue['choice_groups'] if len(g['choices']) > 1]
    require([g['id'] for g in groups] == [d['id'] for d in search['dimensions']],
            'Catalogue/actual search dimension order differs')
    trace = report['beam_trace']
    require(trace['complete'] and trace['rows'] == search['evaluations'],
            'Incomplete streamed evaluation evidence')
    observed = [set() for _ in groups]
    checked = 0
    # A complete initial neighborhood contains the baseline and one change for
    # every nonbaseline value. Stop only after obtaining the actual witnesses.
    expected_prefix = 1 + sum(len(g['choices']) - 1 for g in groups)
    with Path(trace['path']).open() as stream:
        for line in stream:
            row = json.loads(line)
            checked += 1
            require(row['call'] == checked and len(row['assignment']) == len(groups),
                    'Trace is discontinuous or has another assignment schema')
            for index, value in enumerate(row['assignment']):
                require(type(value) is int and 0 <= value < len(groups[index]['choices']),
                        'Observed choice lies outside registered domain')
                observed[index].add(value)
            if all(len(seen) == len(group['choices'])
                   for seen, group in zip(observed, groups)):
                break
            require(checked < expected_prefix, 'Initial neighborhood omitted a registered choice')
    require(checked > 0 and all(len(seen) == len(group['choices'])
                              for seen, group in zip(observed, groups)),
            'Recorded assignments do not witness every local choice')
    impl = {i['id']: i for i in catalogue['implementations']}
    lower = catalogue['fixed_application_invocation']['wall_ms'] + math.fsum(
        min(math.fsum(impl[name]['wall_ms'] for name in choice['implementations'])
            for choice in group['choices']) for group in catalogue['choice_groups'])
    predicted = report['full_model_predicted_ms']
    return dict(
        schema='orbit.qwen-r18-actual-choice-coverage.v1',
        search_result=dict(file=str(report_path), sha256=sha(report_path)),
        catalogue=dict(file=str(catalogue_path), sha256=sha(catalogue_path)),
        complete_trace=dict(file=trace['path'], sha256=sha(trace['path']), rows=trace['rows']),
        prefix_rows_checked=checked,
        all_registered_choices_observed=True,
        variable_groups=len(groups),
        groups=[dict(id=group['id'], registered_choices=len(group['choices']),
                     actually_evaluated_choices=sorted(seen))
                for group, seen in zip(groups, observed)],
        registered_search_scope=report['registered_search_scope'],
        unconstrained_local_fee_lower_bound_ms=lower,
        selected_prediction_ms=predicted,
        selected_minus_lower_bound_ms=predicted-lower,
        matches_serial_registered_model_lower_bound=abs(predicted-lower) < 1e-8,
        bound_used_during_search=False,
        global_abstract_optimality_proved=False,
        limitation='Coverage witnesses every registered local value, not the Cartesian product or arbitrary unsupported transformations.',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('report', 'catalogue', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.report, args.catalogue)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(covered=result['all_registered_choices_observed'],
                          variables=result['variable_groups'],
                          actual_prefix_evaluations=result['prefix_rows_checked'],
                          lower_bound_gap_ms=result['selected_minus_lower_bound_ms'])))


if __name__ == '__main__':
    main()
