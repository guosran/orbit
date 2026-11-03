"""Bounded whole-plan mapping search and deterministic resource list scheduler."""
from __future__ import annotations
import copy
import itertools
import math
from .contract import (PLAN_SCHEMA, PlanningError, expand_program, identity,
                       implementation_cost, implementation_supports, indexed,
                       validate_config)

def schedule_graph(graph, config):
    """Earliest placement in capacity calendars, explicit overlap only.

No bandwidth sharing is inferred. Each event occupies its declared resource
units throughout its interval. Forbid overlap serializes all event kinds.
"""
    graph = copy.deepcopy(graph)
    events = indexed(graph['events'], 'events')
    capacities = {r['id']: r['capacity'] for r in config['hardware']['resources']}
    scheduled = {}
    calendars = {r: [] for r in capacities}
    serial_end = 0.0
    while len(scheduled) < len(events):
        ready = [e for e in events.values() if e['id'] not in scheduled and set(e.get('depends_on', [])) <= scheduled.keys()]
        if not ready:
            raise PlanningError('cyclic or unknown execution event dependency', 'infeasible')
        # Stable list order is a documented heuristic, not an optimal scheduler.
        event = ready[0]
        duration = event['duration_ms']
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
            raise PlanningError('unknown or invalid event cost')
        start = max([scheduled[d]['end_ms'] for d in event.get('depends_on', [])] + [0.0])
        if config['assumptions']['overlap'] == 'forbid':
            start = max(start, serial_end)
        demands = event.get('resource_demands', {})
        for resource, units in demands.items():
            if resource not in capacities or units <= 0 or units > capacities[resource]:
                raise PlanningError('event resource demand exceeds capacity', 'infeasible')
        while True:
            conflicts = []
            if duration > 0:
                for resource, units in demands.items():
                    calendar = calendars[resource]
                    points = sorted({start} | {s for s,e,u in calendar if start < s < start+duration} | {e for s,e,u in calendar if start < e < start+duration})
                    for point in points:
                        active = [(s,e,u) for s,e,u in calendar if s <= point < e]
                        if units + sum(u for s,e,u in active) > capacities[resource]:
                            conflicts.extend(e for s,e,u in active)
                            break
            if not conflicts:
                break
            start = min(conflicts)
        event['start_ms'], event['end_ms'] = start, start + duration
        scheduled[event['id']] = event
        serial_end = max(serial_end, event['end_ms'])
        for resource, units in demands.items():
            calendars[resource].append((start, event['end_ms'], units))
    makespan = max(e['end_ms'] for e in events.values()) if events else 0.0
    peaks = {}
    for memory in config['hardware']['memories']:
        changes = []
        for allocation in graph.get('allocations', []):
            if allocation['memory'] != memory['id']:
                continue
            if allocation.get('lifetime') == 'event' or allocation.get('temporary', False):
                event = events[allocation['created_by']]
                begin, end = event['start_ms'], event['end_ms']
            else:
                begin, end = 0.0, makespan
            allocation['start_ms'], allocation['end_ms'] = begin, end
            if begin < end:
                changes += [(begin, allocation['bytes']), (end, -allocation['bytes'])]
        current = peak = 0
        # Release before allocate at identical endpoints.
        for time, amount in sorted(changes, key=lambda p: (p[0], p[1])):
            current += amount; peak = max(peak, current)
        retained = sum(a['bytes'] for a in graph.get('allocations', []) if a['memory'] == memory['id'] and a.get('lifetime') != 'event' and not a.get('temporary', False))
        # Even a zero-duration model must reserve its physical buffers/scratch.
        temporary_by_event = {}
        for a in graph.get('allocations', []):
            if a['memory'] == memory['id'] and (a.get('lifetime') == 'event' or a.get('temporary', False)):
                temporary_by_event[a['created_by']] = temporary_by_event.get(a['created_by'], 0) + a['bytes']
        peak = max(peak, retained + max(temporary_by_event.values(), default=0))
        peaks[memory['id']] = peak
        if peak > memory['capacity_bytes']:
            raise PlanningError(f'memory {memory["id"]}: peak {peak} exceeds capacity {memory["capacity_bytes"]}', 'infeasible')
    graph['makespan_ms'] = makespan
    graph['totals'] = {'makespan_ms': makespan, 'compute_ms': sum(e['duration_ms'] for e in events.values() if e['kind'] == 'compute'),
                       'copy_ms': sum(e['duration_ms'] for e in events.values() if e['kind'] == 'copy'),
                       'convert_ms': sum(e['duration_ms'] for e in events.values() if e['kind'] == 'convert'),
                       'sync_ms': sum(e['duration_ms'] for e in events.values() if e['kind'] == 'sync'),
                       'peak_memory_bytes': peaks,
                       'transfer_bytes': sum(e.get('bytes', 0) for e in events.values() if e['kind'] == 'copy')}
    return graph

def legal_choices(program, config, steps):
    buffers = indexed(program['buffers'], 'buffers')
    bystep = {s['id']: s for s in steps}
    choices = {}
    for step in steps:
        op = step['operation']; accepted = []
        for impl in config['implementations']:
            if not implementation_supports(op, impl, config):
                continue
            limits = impl.get('legality', {})
            if limits.get('full_k_only') and (op['kind'] != 'matmul' or op['k'] != [0, buffers[op['inputs'][0]]['shape'][1]]):
                continue
            if limits.get('overwrite_zero_accumulator'):
                previous = bystep.get(f'{step["task_id"]}:{step["operation_index"]-1}', {}).get('operation', {})
                if op['kind'] != 'matmul' or previous.get('kind') != 'fill' or any(previous.get(k) != op.get(k) for k in ('output','m','n')):
                    continue
            accepted.append(impl)
        if not accepted:
            raise PlanningError(f'no supported implementation for {step["id"]} ({op["kind"]}), including shape/precision/cost restrictions')
        choices[step['id']] = sorted(accepted, key=lambda i: (implementation_cost(op, i), i['id']))
    return choices

def map_program(program, config):
    """Evaluate complete assignments; never rank tasks by isolated latency.

Exhaustive when the Cartesian product fits the explicit budget. Otherwise
seed homogeneous device choices and fastest-compute choices, then use bounded
best-plan neighbors. Every score is a full communication/memory schedule.
"""
    try:
        validate_config(config)
        expected_source = config['assumptions'].get('program_source_id')
        if expected_source is not None and expected_source != program.get('source_id'):
            raise PlanningError('numerical acceptance contract is bound to another program source')
        steps = expand_program(program)
        choices = legal_choices(program, config, steps)
    except (PlanningError, KeyError, TypeError, ValueError, IndexError, OverflowError, AttributeError) as exc:
        return {'status': getattr(exc, 'status', 'unsupported'), 'best_plan': None, 'alternatives': [], 'reasons': [str(exc)]}
    from .communication import build_graph
    budget = config['assumptions'].get('max_assignments', 256)
    beam_width = config['assumptions'].get('beam_width', 16)
    if type(budget) is not int or budget <= 0 or type(beam_width) is not int or beam_width <= 0:
        return {'status': 'unsupported', 'best_plan': None, 'alternatives': [], 'reasons': ['mapping budget and beam width must be positive integers']}
    ids = [s['id'] for s in steps]
    options = [[i['id'] for i in choices[s]] for s in ids]
    count = math.prod(map(len, options))
    exhaustive = count <= budget
    queue = list(itertools.product(*options)) if exhaustive else [tuple(o[0] for o in options)]
    if not exhaustive:
        devices = sorted({i['device'] for i in config['implementations']})
        for device in devices:
            queue.append(tuple(next((i['id'] for i in choices[s] if i['device'] == device), choices[s][0]['id']) for s in ids))
    seen, alternatives = set(), []
    while queue and len(seen) < budget:
        batch, queue = queue, []
        evaluated = []
        failed_selections = []
        for selection in batch:
            if selection in seen or len(seen) >= budget:
                continue
            seen.add(selection)
            assignment = dict(zip(ids, selection))
            try:
                graph = build_graph(program, config, steps, assignment)
                plan = schedule_graph(graph, config)
                plan.update({'schema': PLAN_SCHEMA, 'program_id': identity(program), 'candidate_id': program.get('candidate_id'), 'configuration_id': identity(config),
                             'assignment': assignment, 'steps': steps,
                             'prediction_kind': 'static_estimate', 'execution_status': 'not_executed', 'schedule_policy': 'deterministic_topological_list',
                             'cost_boundary': 'compute_only includes local access; explicit copy/convert/sync exclude compute; single cold graph'})
                plan['plan_id'] = identity(plan)
                from .checker import check_plan
                check = check_plan(program, config, plan)
                if not check['passed']:
                    raise PlanningError('independent checker rejected: ' + '; '.join(check['errors']), 'invalid_plan')
                plan['validation'] = check
                entry = {'status': 'passed', 'assignment': assignment, 'score': plan['totals']['makespan_ms'], 'score_unit': 'ms', 'plan': plan}
                evaluated.append((entry['score'], selection))
            except (PlanningError, KeyError, TypeError, ValueError, IndexError, OverflowError, AttributeError) as exc:
                failed_selections.append(selection)
                entry = {'status': getattr(exc, 'status', 'unsupported'), 'assignment': assignment, 'score': None, 'score_unit': None, 'reason': str(exc)}
            alternatives.append(entry)
        if not exhaustive:
            frontier = [selected for score, selected in sorted(evaluated)[:beam_width]]
            # If all seeds fail, still explore neighbors; no artificial score is assigned.
            if not frontier:
                frontier = failed_selections[:beam_width]
            for selected in frontier:
                for i, possibilities in enumerate(options):
                    for replacement in possibilities:
                        neighbor = selected[:i] + (replacement,) + selected[i+1:]
                        if neighbor not in seen:
                            queue.append(neighbor)
    passed = [x for x in alternatives if x['status'] == 'passed']
    best = min(passed, key=lambda e: (e['score'], e['plan']['plan_id'])) if passed else None
    return {'schema': 'orbit.static.mapping-result.v1', 'status': 'passed' if best else 'unscored', 'best_plan': best['plan'] if best else None,
            'alternatives': alternatives, 'reasons': [] if best else sorted({x.get('reason','unscored') for x in alternatives}),
            'search': {'algorithm': 'bounded_complete_assignment_search', 'exhaustive_assignments': exhaustive,
                       'possible_assignments': count, 'evaluated_assignments': len(seen), 'budget': budget,
                       'selection_reason': 'lowest independently checked complete makespan; no global schedule optimality claim'}}
