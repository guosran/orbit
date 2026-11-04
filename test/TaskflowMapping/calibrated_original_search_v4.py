#!/usr/bin/env python3
"""CPU-only original optimizer calibration/search/cache acceptance."""
import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'build/static-mapping-v4/region-calibration'
OUT = BASE / 'original-optimizer-acceptance-final'
TOOL = ROOT / 'build-amoeba/tools/mlir-amoeba-opt/mlir-amoeba-opt'

def save(p, v):
    p.write_text(json.dumps(v, indent=2))

def load(p):
    return json.loads(p.read_text())

def run(label, opts, expected=0):
    directory = Path(opts['output-dir'])
    directory.mkdir(exist_ok=True)
    cmd = ['strace', '-f', '-e', 'trace=process', '-o', str(OUT / (label + '.process.log')), str(TOOL), str(BASE / 'native/source.mlir'), '--search-joint-neighborhood=' + ' '.join((k + '=' + str(v) for k, v in opts.items())), '-o', str(directory / 'output.mlir')]
    save(OUT / (label + '.command.json'), cmd)
    r = subprocess.run(cmd, capture_output=True, text=True)
    (OUT / (label + '.log')).write_text(r.stdout + r.stderr)
    assert (r.returncode == 0) == (expected == 0), (label, r.returncode, r.stderr)
    processes = (OUT / (label + '.process.log')).read_text()
    execs = [line for line in processes.splitlines() if 'execve(' in line]
    assert len(execs) == 1 and str(TOOL) in execs[0], execs
    return (r, execs)

def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    OUT = parser.parse_args().output.resolve()
    OUT.mkdir(exist_ok=False)
    config = load(BASE / 'search-config.json')
    cp = OUT / 'target.json'
    save(cp, config)
    opts = {'decision-flow': 'structured', 'backend': 'static', 'structured-plan': str(BASE / 'native/source.json'), 'target-config': str(cp), 'output-dir': str(OUT / 'search'), 'max-rounds': 2, 'max-candidates': 16, 'beam-width': 2}
    _, execs = run('original-search', opts)
    result = load(OUT / 'search/search.json')
    assert len(result['selections']) == 2 and len(result['archive']) >= 3
    original_checkpoint = (OUT / 'search/checkpoint.json').read_bytes()
    original_search = (OUT / 'search/search.json').read_bytes()
    resume = dict(opts, resume='true')
    run('resume-unchanged', resume)
    assert (OUT / 'search/search.json').read_bytes() == original_search
    invalidations = []
    for label in ['region-model-id', 'region-cost-coefficient', 'region-prepared-state']:
        changed = copy.deepcopy(config)
        entry = changed['region_implementations'][0]
        if label == 'region-model-id':
            entry['cost']['model_id'] = 'changed-unmeasured-model-identity'
        elif label == 'region-cost-coefficient':
            entry['cost']['fixed_ms'] += 0.125
        else:
            entry['prepared_state'] = 'cold_inputs_only'
        save(cp, changed)
        r, _ = run(label, resume, expected=1)
        assert 'binding differs' in r.stderr, (label, r.stderr)
        assert (OUT / 'search/checkpoint.json').read_bytes() == original_checkpoint
        invalidations.append(label)
    save(cp, config)
    changed = copy.deepcopy(config)
    for entry in changed['region_implementations']:
        if entry['pattern'] == 'matmul_pair' and entry['mode'] == 'chain':
            entry['cost']['fixed_ms'] += 2
            entry['cost']['model_id'] = 'sensitivity-chain-plus-2ms'
            entry['cost']['provenance'] = {'kind': 'theoretical', 'source': 'explicit cache/ranking sensitivity; original frozen calibration retained unchanged'}
    sensitivity = OUT / 'sensitivity-config.json'
    save(sensitivity, changed)
    sopts = dict(opts, **{'target-config': str(sensitivity), 'output-dir': str(OUT / 'sensitivity-search')})
    run('sensitivity-fresh', sopts)
    sr = load(OUT / 'sensitivity-search/search.json')

    def baseline(r):
        return next((a for a in r['archive'] if a['action'] == 'identity'))
    old, new = (baseline(result), baseline(sr))
    oldp = load(Path(old['evaluation']['evidence_path']))['best_plan']
    newp = load(Path(new['evaluation']['evidence_path']))['best_plan']
    assert oldp['events'][0]['region_mode'] == 'chain' and newp['events'][0]['region_mode'] == 'independent'
    assert result['binding'] != sr['binding'] and oldp['configuration_id'] != newp['configuration_id']
    assert old['search_key'] == new['search_key']
    report = {'passed': True, 'production_tool': str(TOOL), 'binary_sha256': hashlib.sha256(TOOL.read_bytes()).hexdigest(), 'rounds': len(result['selections']), 'evaluations': result['evaluations'], 'archive': [{'round': a['round'], 'action': a['action'], 'status': a['status'], 'score': a.get('score'), 'search_key': a['search_key']} for a in result['archive']], 'selections': result['selections'], 'resume_unchanged': True, 'cache_invalidations': invalidations, 'checkpoint_unchanged_on_rejected_resume': True, 'process_evidence': {'execve_count_per_run': 1, 'only_executable': str(TOOL), 'python_decision_children': 0, 'thread_creation_allowed': True}, 'sensitivity': {'label': 'synthetic +2ms chain fixed cost, not measured and not a model refit', 'old_mode': oldp['events'][0]['region_mode'], 'new_mode': newp['events'][0]['region_mode'], 'old_score_ms': old['score'], 'new_score_ms': new['score'], 'structural_candidate_key_unchanged': old['search_key'] == new['search_key'], 'old_binding': result['binding'], 'new_binding': sr['binding']}, 'hardware_executed': False}
    save(OUT / 'report.json', report)
    print(json.dumps({'passed': True, 'report': str(OUT / 'report.json')}))
if __name__ == '__main__':
    main()
