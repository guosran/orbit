#!/usr/bin/env python3
"""Original-driver Amoeba ProgramGraph regression; no builds or devices.

The backend performs actual model inference, mapping and scheduling. This script
only invokes native commands, validates retained identities and tests replay
rejection. It never constructs a production plan or supplies a substitute cost.
"""
import argparse
import copy
import hashlib
import json
import math
import re
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def communication_payloads(text):
    # Only decode top-level emitted trace records, not a graph or scheduler.
    rows, depth, start, quoted, escaped = [], 0, None, False, False
    for index, char in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char == '{':
            if depth == 0:
                start = index
            depth += 1
        elif char == '}':
            depth -= 1
            assert depth >= 0
            if depth == 0:
                record = text[start:index+1]
                payload = re.search(r'\bpayload_bits = (\d+) : i64',record)
                producer = re.search(r'\bproducer = "([^"]+)"',record)
                consumer = re.search(r'\bconsumer = "([^"]+)"',record)
                assert payload and producer and consumer, 'trace record lacks explicit payload endpoints'
                rows.append(dict(producer=producer[1],consumer=consumer[1],payload_bits=int(payload[1])))
    assert depth == 0 and not quoted
    return rows


def audit_backend(directory, graph, mapping, architecture):
    receipt = mapping['best_plan']['backend_receipt']
    assert receipt['status'] == 'passed' and receipt['graph_id'] == graph['id']
    assert mapping['best_plan']['cost_status'] == 'conditional-original-cgra-ml'
    assert receipt['mapper_replay_completed'] is True
    assert receipt['production_graph_projection_verified'] is True
    assert receipt['resource_binding']['physical_capacity_claim'] is False
    assert receipt['resource_binding']['public_resource'] == 'intertask.network'
    assert receipt['resource_binding']['architecture_identity'] == sha(architecture)
    assert receipt['resource_binding']['network_identity']
    assert receipt['production_communication_trace_mlir'] and receipt['production_dependency_trace_mlir']
    assert 'not backend cycle-buffer' in receipt['copy_lifetime_scope']
    ledger = graph['ledger']
    names = [task['id'] for task in graph['source']['tasks']]
    def version_producer(copy):
        return ledger['versions'][ledger['copies'][copy]['version']]['producer']
    def event_task(event):
        return names[ledger['events'][event]['task']]
    internal = {}
    for communication in graph['communications']:
        producers = {version_producer(c) for c in communication['source_copies']}
        external_target = all(ledger['allocations'][ledger['copies'][c]['allocation']]['memory'] == 'external.memory' for c in communication['target_copies'])
        if external_target or producers == {-1}:
            continue
        assert len(producers) == 1 and next(iter(producers)) >= 0
        producer = next(iter(producers))
        internal[communication['event']] = (communication,event_task(producer),event_task(communication['event']))
    observed = []
    for edge in receipt['public_edge_projection']:
        if edge['kind'] == 'raw':
            event = edge['program_graph_event']
            assert event in internal and event not in observed and edge['program_graph_order'] == -1
            communication, producer, consumer = internal[event]
            assert [edge['producer'],edge['consumer']] == [producer,consumer]
            assert edge['payload_bits'] == communication['bytes']*8
            assert edge['region_lower'] == communication['offsets']
            assert edge['region_upper'] == [a+b for a,b in zip(communication['offsets'],communication['sizes'])]
            observed.append(event)
        else:
            assert edge['kind'] == 'control' and edge['payload_bits'] == 0
            assert edge['program_graph_event'] == -1 and edge['region_lower'] == [] and edge['region_upper'] == []
            order = edge['program_graph_order']
            if order >= 0:
                actual = ledger['order_edges'][order]
                assert [edge['producer'],edge['consumer']] == [event_task(actual['producer']),event_task(actual['consumer'])]
            else:
                assert any(d['producer'] >= 0 and [edge['producer'],edge['consumer']] == [event_task(d['producer']),event_task(d['consumer'])] for d in ledger['data_edges'])
    assert sorted(observed) == sorted(internal), 'missing or duplicate priced internal copy transfer'
    assert 'unpriced' in receipt['external_io_cost_status']
    assert 'unknown' in receipt['memory_capacity_status']
    evaluation = directory/'evaluation'
    saved = json.loads((evaluation/'backend-receipt.json').read_text())
    assert saved == receipt
    catalog = evaluation/('costs-'+graph['id']+'.json')
    entries = json.loads(catalog.read_text())['entries']
    costs = receipt['task_costs']
    tasks = {task['id']: task for task in graph['source']['tasks']}
    assert len(costs) == len(tasks) and {c['task'] for c in costs} == set(tasks)
    assert len(receipt['task_schedule']) == len(tasks)
    schedule = {p['task']: p for p in receipt['task_schedule']}
    assert len(schedule) == len(tasks) and set(schedule) == set(tasks)
    assert len(receipt['dispatch_order']) == len(tasks) and set(receipt['dispatch_order']) == set(tasks)
    for cost in costs:
        matches = [entry for entry in entries if entry['task'] == cost['task'] and
            entry['mapper_tile_rows'] == cost['mapper_rows'] and entry['mapper_tile_cols'] == cost['mapper_cols']]
        assert len(matches) == 1 and matches[0]['support_status'] == 'supported'
        assert matches[0]['predicted_ii'] == cost['predicted_ii'] and matches[0]['startup_cycles'] == cost['startup_cycles']
        # Independent audit of this fixture's actual two-dimensional domain;
        # the backend owns the production firing-count/model query.
        loops = tasks[cost['task']]['loops']
        assert len(loops) == 2
        firing_count = math.prod((l['upper']-l['lower']+l['step']-1)//l['step'] for l in loops)
        assert cost['trip_count'] == firing_count
        duration = max(1, math.ceil(cost['startup_cycles']+cost['predicted_ii']*(cost['trip_count']-1)))
        assert duration == cost['duration_cycles']
        timing = schedule[cost['task']]
        assert timing['end_cycle']-timing['start_cycle'] == duration and timing['start_cycle'] >= 0
        assert [timing['rows'],timing['cols']] == [cost['cgra_rows'],cost['cgra_cols']]
    assert receipt['predicted_makespan_cycles'] == mapping['best_plan']['makespan_cycles']
    assert receipt['predicted_makespan_cycles'] == max(p['end_cycle'] for p in schedule.values())
    actual = receipt['actual_ii']
    assert len(actual) == len(tasks) and {a['task'] for a in actual} == set(tasks)
    mapped = (evaluation/'mapped.mlir').read_text()
    lowered = (evaluation/'neura-source.mlir').read_text()
    assert graph['id'] in mapped and graph['id'] in lowered
    assert 'joint_scheduling_mapper_replay_completed' in mapped
    for row in actual:
        assert isinstance(row['compiled_ii'],int) and row['compiled_ii'] > 0
        pattern = r'(?m)^.*taskflow\.task @'+re.escape(row['task'])+r'(?: |\().*?compiled_ii = (\d+) : i32'
        values = re.findall(pattern,mapped)
        assert values == [str(row['compiled_ii'])], (row['task'],'compiled II does not match actual mapped task signature',values)
    # This schedule is a fresh production-scheduler result on compiled II.
    # mapped.mlir is the preceding mapper snapshot, so do not demand that it
    # already contain fields from this later scheduling step.
    assert len(receipt['actual_task_schedule']) == len(tasks)
    assert len(receipt['actual_duration_facts']) == len(tasks)
    actual_schedule = {p['task']: p for p in receipt['actual_task_schedule']}
    actual_facts = {f['task']: f for f in receipt['actual_duration_facts']}
    actual_ii = {f['task']: f['compiled_ii'] for f in actual}
    costs_by_task = {f['task']: f for f in costs}
    assert len(actual_schedule) == len(tasks) and set(actual_schedule) == set(tasks)
    assert len(actual_facts) == len(tasks) and set(actual_facts) == set(tasks)
    assert len(receipt['actual_dispatch_order']) == len(tasks) and set(receipt['actual_dispatch_order']) == set(tasks)
    for task, fact in actual_facts.items():
        predicted = costs_by_task[task]
        assert fact['compiled_ii'] == actual_ii[task]
        assert fact['startup_cycles'] == predicted['startup_cycles']
        assert fact['trip_count'] == predicted['trip_count']
        duration = max(1, math.ceil(fact['startup_cycles']+fact['compiled_ii']*(fact['trip_count']-1)))
        assert fact['duration_cycles'] == duration
        timing = actual_schedule[task]
        assert timing['end_cycle']-timing['start_cycle'] == duration and timing['start_cycle'] >= 0
        assert [timing['rows'],timing['cols']] == [predicted['cgra_rows'],predicted['cgra_cols']]
    assert receipt['actual_makespan_cycles'] == max(p['end_cycle'] for p in actual_schedule.values()) > 0
    assert receipt['actual_production_communication_trace_mlir'] and receipt['actual_production_dependency_trace_mlir']
    assert 'not execution timing' in receipt['actual_schedule_scope']
    actual_legacy = receipt['actual_legacy_differential']
    assert actual_legacy['requested'] is True and actual_legacy['status'] in ['identical','different']
    assert actual_legacy['public_task_schedule'] == receipt['actual_task_schedule']
    identical = actual_legacy['makespan_cycles'] == receipt['actual_makespan_cycles'] and actual_legacy['task_schedule'] == receipt['actual_task_schedule']
    assert (actual_legacy['status'] == 'identical') == identical
    if len(tasks) == 2:
        assert actual_legacy['status'] == 'identical', 'untiled compiled-II schedule must preserve existing scheduler'
    actual_payloads = communication_payloads(receipt['actual_production_communication_trace_mlir'])
    actual_old_payloads = communication_payloads(actual_legacy['production_communication_trace_mlir'])
    if len(tasks) == 2:
        assert actual_payloads == actual_old_payloads, 'untiled actual schedule payload/endpoints differ from legacy'
    witness = [json.loads(line) for line in (evaluation/'mapper-score.jsonl').read_text().splitlines() if line.strip()]
    score = next(row for row in witness if row['record_type'] == 'score')
    assert score['valid'] is True and score['task_costs'] == costs
    assert score['task_schedule'] == receipt['task_schedule'] and score['dispatch_order'] == receipt['dispatch_order']
    legacy = receipt['legacy_differential']
    assert legacy['requested'] is True and legacy['status'] in ['identical','different']
    assert legacy == json.loads((evaluation/'legacy-differential.json').read_text())
    assert legacy['public_task_schedule'] == receipt['task_schedule']
    predicted_identical = legacy['makespan_cycles'] == receipt['predicted_makespan_cycles'] and legacy['task_schedule'] == receipt['task_schedule']
    assert (legacy['status'] == 'identical') == predicted_identical
    if len(tasks) == 2:
        assert legacy['status'] == 'identical', 'untiled baseline/shape must preserve the existing scheduler'
    predicted_payloads = communication_payloads(receipt['production_communication_trace_mlir'])
    predicted_old_payloads = communication_payloads(legacy['production_communication_trace_mlir'])
    if len(tasks) == 2:
        assert predicted_payloads == predicted_old_payloads, 'untiled ML schedule payload/endpoints differ from legacy'
    identities = {str(path):sha(path) for path in [catalog,evaluation/'neura-source.mlir',evaluation/'mapped.mlir',evaluation/'backend-receipt.json',evaluation/'mapper-score.jsonl',evaluation/'legacy-differential.json']}
    return dict(communication_payloads=dict(predicted_public=predicted_payloads,predicted_legacy=predicted_old_payloads,compiled_ii_public=actual_payloads,compiled_ii_legacy=actual_old_payloads),actual_task_schedule=receipt['actual_task_schedule'],actual_dispatch_order=receipt['actual_dispatch_order'],actual_duration_facts=receipt['actual_duration_facts'],actual_legacy_differential=actual_legacy,task_count=len(tasks),predicted_makespan_cycles=receipt['predicted_makespan_cycles'],actual_makespan_cycles=receipt.get('actual_makespan_cycles'),actual_ii=actual,legacy_differential=legacy,files=identities)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool', type=Path, required=True, help='Amoeba-enabled native optimizer')
    parser.add_argument('--core-tool', type=Path, required=True)
    parser.add_argument('--checker', type=Path, help='independent Core Amoeba graph checker')
    parser.add_argument('--config', type=Path, required=True, help='native backend target/model configuration')
    parser.add_argument('--architecture', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True, help='existing formal ensemble/checkpoints, not newly trained')
    parser.add_argument('--architecture-contract', default='neura-architecture-v1:amoeba-static-rectangles-up-to-four-cgras-formal')
    parser.add_argument('--source-repository', default='git@github.com:guosran/amoeba.git')
    parser.add_argument('--source-commit', default='a57376e7043b1681e64e7169c5a8cb02eb192331')
    parser.add_argument('--source', type=Path, default=ROOT/'test/TaskflowAIR/fixtures/add-scale.mlir')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--max-candidates', type=int, default=24)
    parser.add_argument('--beam-width', type=int, default=4)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    tool, core, config = args.tool.resolve(), args.core_tool.resolve(), args.config.resolve()
    source = out/'source.mlir'
    source.write_bytes(args.source.resolve().read_bytes())
    (out/'target-config.json').write_bytes(config.read_bytes())
    config = out/'target-config.json'
    architecture = args.architecture.resolve()
    models = args.models.resolve()
    model_files = [models/name for name in ['ensemble.json','baseline.json','large-operation.json','ranking.json']]
    for path in [architecture,*model_files]:
        if not path.is_file():
            parser.error('required existing architecture/model artifact missing: '+str(path))
    checks = []

    def run(name, command, success=True):
        command = list(map(str, command))
        write(out/(name+'.command.json'), command)
        utc = datetime.now(timezone.utc).isoformat()
        begin = time.perf_counter()
        result = subprocess.run(command, capture_output=True, text=True)
        (out/(name+'.stdout')).write_text(result.stdout)
        (out/(name+'.stderr')).write_text(result.stderr)
        write(out/(name+'.result.json'), dict(returncode=result.returncode, utc_start=utc,
            utc_end=datetime.now(timezone.utc).isoformat(), elapsed_wall_seconds=time.perf_counter()-begin,
            device_submissions=0))
        if (result.returncode == 0) != success:
            raise AssertionError((name, result.returncode, result.stderr[-4000:]))
        checks.append(name)
        return result

    checker = (args.checker or core.parent/'taskflow-amoeba-program-graph-test').resolve()
    run('common-amoeba-graph-checker', [checker, source])
    run('plain-source-analysis', [core, source, '--generic-analyze', '--analysis-output='+str(out/'analysis.json')])
    analysis = json.loads((out/'analysis.json').read_text())
    assert analysis['analysis_status'] == 'analyzed'
    assert len(analysis['tasks']) == 2
    assert [t['id'] for t in analysis['tasks']] == ['add', 'scale']
    assert all(len(t['loops']) == 2 and all(l['lower'] == 0 and l['upper'] == 4 and l['step'] == 1 for l in t['loops']) for t in analysis['tasks'])
    search = out/'search'
    options = dict(**{'decision-flow':'program-graph', 'backend':'amoeba', 'function':'add_scale',
        'target-config':str(config), 'output-dir':str(search), 'max-rounds':args.rounds,
        'max-candidates':args.max_candidates, 'beam-width':args.beam_width,
        'model':str(models/'ensemble.json'), 'cache':str(out/'ml-cache'),
        'architecture-path':str(architecture), 'architecture-contract':args.architecture_contract,
        'source-repository':args.source_repository, 'source-commit':args.source_commit})
    command = [tool, source, '--neura-architecture-spec='+str(architecture), '--search-joint-neighborhood='+' '.join(k+'='+str(v) for k,v in options.items()),
        '-o', out/'output.mlir']
    run('original-program-graph-search', command)
    result = json.loads((search/'search.json').read_text())
    assert result['backend'] == 'amoeba' and result['score_unit'] == 'cycles'
    assert len(result['selections']) >= 2
    archive = result['archive']
    legal = [row for row in archive if row['status'] == 'passed']
    assert len(legal) >= 3, 'need actual baseline plus structural and shape candidates'
    retained = []
    source_changes = set()
    tiled_task_counts = []
    for row in legal:
        directory = search/'candidates'/row['candidate_id']
        materialized, graph_file = directory/'taskflow.mlir', directory/'program-graph.json'
        dot_file = directory/'program-graph.dot'
        graph = json.loads(graph_file.read_text())
        evaluation = row['evaluation']
        mapping = json.loads((directory/'mapping-result.json').read_text())
        assert mapping['program_graph'] == graph and mapping['program_graph_id'] == graph['id']
        assert mapping['best_plan']['plan_id'] == evaluation['plan_id']
        backend = audit_backend(directory,graph,mapping,architecture)
        assert graph['schema'] == 'orbit.program-graph.v1'
        assert evaluation['program_graph_id'] == graph['id'] == row['program_graph_id']
        assert graph['source']['analysis_status'] == 'analyzed'
        assert graph['schedule']['estimated_makespan_ms'] is None, 'no implicit cycles-to-ms conversion'
        memories = graph['ledger']['memories']
        assert memories and all(m['capacity_bytes'] is None and m['capacity_known'] is False for m in memories), 'abstract availability is not physical byte capacity'
        assert len(graph['instances']) == len(graph['source']['tasks'])
        assert all(i['kind'] == 'source-task' for i in graph['instances'])
        assert sha(materialized) == row['candidate_id']
        assert 'digraph' in dot_file.read_text()
        score = evaluation['score']
        assert evaluation['evaluator_kind'] == 'native_cpp' and evaluation['executed'] is False
        assert score['physical_hardware_measurement'] is False
        assert score['provenance'] == 'amoeba-program-graph-provider-and-production-schedule'
        assert score['unit'] == 'cycles' and score['value'] == row['score'] and row['score'] > 0
        source_changes.add(graph['source']['body_fingerprint'])
        if any('tiling' in action for action in row['action_chain']):
            tiled_task_counts.append(len(graph['source']['tasks']))
        retained.append(dict(candidate_id=row['candidate_id'],graph_id=graph['id'],
            action_chain=row['action_chain'],backend=backend,source_sha256=sha(materialized),
            graph_sha256=sha(graph_file),dot_sha256=sha(dot_file)))
    assert tiled_task_counts and max(tiled_task_counts) > 2, 'native rectangular tiling must clone actual source tasks'
    assert len(source_changes) >= 2, 'structural child must materialize a changed actual source'
    assert any('shape' in action for row in legal for action in row['action_chain']), 'no legal native shape action'
    assert any(row['round'] >= 2 and row['parent'] for row in legal), 'no real second-round parent lineage'
    ranking = copy.deepcopy(result['ranking'])
    resume = [tool, source, '--neura-architecture-spec='+str(architecture), '--search-joint-neighborhood='+' '.join(k+'='+str(v) for k,v in dict(options,resume='true').items()), '-o', out/'output.mlir']
    run('resume-exact', resume)
    assert json.loads((search/'search.json').read_text())['ranking'] == ranking
    # Mutations of retained exported witnesses test native reconstruction and
    # replay binding. JSON is not imported as executable computation semantics.
    row = legal[0]
    graph_path = search/'candidates'/row['candidate_id']/'program-graph.json'
    original = graph_path.read_bytes()
    graph = json.loads(original)
    mutants = [('target-identity-binding', lambda g:g.__setitem__('target_identity','0'*64)),
               ('source-binding', lambda g:g['source'].__setitem__('body_fingerprint','0'*64)),
               ('version-binding', lambda g:g['ledger']['versions'][0].__setitem__('producer',999999)),
               ('edge-binding', lambda g:g['ledger']['data_edges'].pop()),
               ('reachability-binding', lambda g:g['ledger']['links'].clear())]
    for name, mutate in mutants:
        changed = copy.deepcopy(graph)
        mutate(changed)
        write(graph_path, changed)
        try:
            run('reject-'+name, resume, False)
        finally:
            graph_path.write_bytes(original)
    checkpoint = search/'checkpoint.json'
    checkpoint_bytes = checkpoint.read_bytes()
    altered = json.loads(checkpoint_bytes)
    measured = next(r for r in altered['archive'] if r['status'] == 'passed')
    measured['score'] += 1
    measured['evaluation']['score']['value'] = measured['score']
    write(checkpoint, altered)
    try:
        run('reject-coherent-checkpoint-score', resume, False)
    finally:
        checkpoint.write_bytes(checkpoint_bytes)
    source_bytes = source.read_bytes()
    source.write_bytes(source_bytes.replace(b'arith.addf', b'arith.subf', 1))
    try:
        run('reject-source-checkpoint-binding', resume, False)
    finally:
        source.write_bytes(source_bytes)
    config_bytes = config.read_bytes()
    changed_config = json.loads(config_bytes)
    backend_config = changed_config.setdefault('amoeba', {})
    backend_config['default_shape'] = [2,1] if backend_config.get('default_shape') == [1,2] else [1,2]
    write(config,changed_config)
    try:
        rejection = run('reject-target-config-checkpoint-binding', resume, False)
        assert 'checkpoint binding differs' in rejection.stderr, 'target mutation must fail identity binding, not unrelated parsing'
    finally:
        config.write_bytes(config_bytes)
    run('resume-restored', resume)
    # A valid initialized source with a cross-slice read is outside the
    # conservative tile proof. Its baseline is real; only the action rejects.
    recurrence = out/'cross-slice-read.mlir'
    text = source.read_text()
    init = """    %T_init = taskflow.task @initialize
      will_writes(%T : memref<4x4xf32>)
      [original_write_memrefs(%T : memref<4x4xf32>)]
      : (memref<4x4xf32>) -> memref<4x4xf32> {
    ^bb0(%init_out: memref<4x4xf32>):
      %zero = arith.constant 0.0 : f32
      affine.for %init_i = 0 to 4 {
        affine.for %init_j = 0 to 4 {
          affine.store %zero, %init_out[%init_i, %init_j] : memref<4x4xf32>
        }
      }
      taskflow.yield done_writes(%init_out : memref<4x4xf32>)
    }
"""
    def replace_once(value, old, new):
        assert value.count(old) == 1, ('negative source fixture changed',old)
        return value.replace(old,new,1)
    text = replace_once(text,'    %T_done = taskflow.task @add',init+'    %T_done = taskflow.task @add')
    text = replace_once(text,'will_reads(%A, %B : memref<4x4xf32>, memref<4x4xf32>)','will_reads(%A, %B, %T_init : memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>)')
    text = replace_once(text,'original_read_memrefs(%A, %B : memref<4x4xf32>, memref<4x4xf32>)','original_read_memrefs(%A, %B, %T_init : memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>)')
    text = replace_once(text,': (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>) -> memref<4x4xf32>',': (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>) -> memref<4x4xf32>')
    text = replace_once(text,'%B_b: memref<4x4xf32>, %T_b: memref<4x4xf32>):','%B_b: memref<4x4xf32>, %T_prev: memref<4x4xf32>, %T_b: memref<4x4xf32>):')
    text = replace_once(text,'          affine.store %sum, %T_b[%i, %j]', '          %old = affine.load %T_prev[%i, 0] : memref<4x4xf32>\n          %next = arith.addf %sum, %old : f32\n          affine.store %next, %T_b[%i, %j]')
    recurrence.write_text(text)
    run('cross-slice-source-analysis',[core,recurrence,'--generic-analyze','--analysis-output='+str(out/'cross-slice-analysis.json')])
    assert len(json.loads((out/'cross-slice-analysis.json').read_text())['tasks']) == 3
    negative_options = dict(options,**{'output-dir':str(out/'cross-slice-search'),'max-rounds':1,'max-candidates':1,'beam-width':1})
    run('cross-slice-tiling-rejection',[tool,recurrence,'--neura-architecture-spec='+str(architecture),'--search-joint-neighborhood='+' '.join(k+'='+str(v) for k,v in negative_options.items()),'-o',out/'cross-slice-output.mlir'])
    rejected_search = json.loads((out/'cross-slice-search/search.json').read_text())
    assert rejected_search['evaluations'] == 1
    unsupported = [row for row in rejected_search['archive'] if row['status'] == 'unsupported']
    assert unsupported and not rejected_search['attempts']
    unsupported_mapping = json.loads((out/'cross-slice-search/candidates'/unsupported[0]['candidate_id']/'mapping-result.json').read_text())
    assert 'compute copy regions omit/fabricate actual source instance accesses' in unsupported_mapping['reason']
    write(out/'cross-slice-rejection-summary.json',dict(status='passed',source_sha256=sha(recurrence),reason=unsupported_mapping['reason'],scope='source-valid but outside current adapter access coverage; baseline rejected before model/mapper and no action proof claimed'))
    # A separate source-valid and backend-legal baseline has a foldable pure
    # invariant arithmetic prologue. The finite tiler must reject that syntax
    # before cloning, even though the original Neura path can evaluate it.
    prologue = out/'pure-prologue.mlir'
    marker = '    ^bb0(%A_b: memref<4x4xf32>, %B_b: memref<4x4xf32>, %T_b: memref<4x4xf32>):\n'
    prologue.write_text(replace_once(source.read_text(),marker,marker+'      %invariant_zero = arith.constant 0.0 : f32\n      %invariant_sum = arith.addf %invariant_zero, %invariant_zero : f32\n'))
    run('pure-prologue-source-analysis',[core,prologue,'--generic-analyze','--analysis-output='+str(out/'pure-prologue-analysis.json')])
    prologue_options = dict(options,**{'output-dir':str(out/'pure-prologue-search'),'max-rounds':1,'max-candidates':2,'beam-width':1})
    run('pure-prologue-tiling-rejection',[tool,prologue,'--neura-architecture-spec='+str(architecture),'--search-joint-neighborhood='+' '.join(k+'='+str(v) for k,v in prologue_options.items()),'-o',out/'pure-prologue-output.mlir'])
    prologue_search = json.loads((out/'pure-prologue-search/search.json').read_text())
    rejected_actions = [attempt for attempt in prologue_search['attempts'] if attempt['action'] == 'amoeba-source-rectangular-tiling' and attempt['status'] == 'semantic_invalid' and 'Amoeba tiling permits only constant invariants outside the perfect loop nest' in attempt['reason']]
    assert rejected_actions, 'backend-legal prologue baseline must explicitly exercise the native action guard'
    assert 1 <= prologue_search['evaluations'] <= 2
    assert any(row['status'] == 'passed' and not row['action_chain'] for row in prologue_search['archive']), 'action negative requires a real successful baseline'
    assert not any(row['status'] == 'passed' and 'amoeba-source-rectangular-tiling' in row['action_chain'] for row in prologue_search['archive'])
    write(out/'pure-prologue-rejection-summary.json',dict(status='passed',source_sha256=sha(prologue),attempts=rejected_actions,scope='actual baseline analyzed and model/mapper/scheduler evaluated; finite tiling action rejects nonconstant invariant prologue'))
    assert json.loads((search/'search.json').read_text())['ranking'] == ranking
    write(out/'summary.json', dict(status='passed', checks=checks, candidates=retained,
        source=dict(path=str(source),sha256=sha(source),body_fingerprint=analysis['body_fingerprint']),
        tools={str(p):sha(p) for p in [tool,core,checker]},config=dict(path=str(config),sha256=sha(config)),
        test_source_sha256=sha(Path(__file__)),model_files={str(p):sha(p) for p in model_files},architecture=dict(path=str(architecture),sha256=sha(architecture),contract=args.architecture_contract),device_submissions=0,model_training=False,
        scope='original C++ driver, native callbacks, real backend model/mapper/scheduler artifacts; exported graph mutations are replay binding negatives, not imported graph programs'))
    print(json.dumps(dict(status='passed',checks=len(checks),device_submissions=0)))


if __name__ == '__main__':
    main()
