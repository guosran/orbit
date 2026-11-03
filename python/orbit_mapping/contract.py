"""Versioned static mapping contract; no backend/runtime imports.

Public Taskflow remains the semantic authority. This module validates only the
restricted static planning boundary; the original core authenticates MLIR.
"""
from __future__ import annotations
import hashlib
import json
import math

CONFIG_SCHEMA = 'orbit.static.config.v1'
PLAN_SCHEMA = 'orbit.static.plan.v1'

class PlanningError(ValueError):
    def __init__(self, reason, status='unsupported'):
        super().__init__(reason)
        self.status = status

def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise PlanningError(f'{name}: expected finite {"positive" if positive else "nonnegative"} number')
    return value

def indexed(items, label):
    result = {}
    for item in items:
        key = item.get('id')
        if not isinstance(key, str) or not key or key in result:
            raise PlanningError(f'{label}: missing or duplicate id {key!r}')
        result[key] = item
    return result

def provenance(value, label):
    if not isinstance(value, dict) or value.get('kind') not in ('synthetic', 'theoretical', 'calibrated', 'measured') or not value.get('source'):
        raise PlanningError(f'{label}: explicit cost provenance required')

def expand_program(program):
    if program.get('schema') != 'orbit.taskflow.plan.v1':
        raise PlanningError('unsupported public program schema')
    buffers = indexed(program.get('buffers', []), 'buffers')
    tasks = indexed(program.get('tasks', []), 'tasks')
    if not buffers or not tasks:
        raise PlanningError('empty program')
    for b in buffers.values():
        if b.get('dtype') != 'f32' or len(b.get('shape', [])) != 2 or any(type(x) is not int or x <= 0 for x in b['shape']) or b.get('alias_of'):
            raise PlanningError('only explicit, static, nonaliased rank-two f32 storage supported')
    steps, pending, emitted = [], list(tasks.values()), set()
    while pending:
        ready = [t for t in pending if set(t.get('depends_on', [])) <= emitted]
        if not ready:
            raise PlanningError('cyclic or missing task dependency')
        for task in ready:
            ops = task.get('operations', [])
            if not ops:
                raise PlanningError('empty task unsupported')
            for i, op in enumerate(ops):
                if op.get('kind') not in ('fill', 'matmul', 'relu', 'swiglu'):
                    raise PlanningError('unsupported operation ' + str(op.get('kind')))
                if op.get('output') not in buffers or any(x not in buffers for x in op.get('inputs', [])):
                    raise PlanningError('unknown operation storage')
                deps = [f'{task["id"]}:{i-1}'] if i else [f'{d}:{len(tasks[d]["operations"])-1}' for d in task.get('depends_on', [])]
                steps.append({'id': f'{task["id"]}:{i}', 'task_id': task['id'], 'operation_index': i, 'operation': op, 'depends_on': deps})
            emitted.add(task['id']); pending.remove(task)
    return steps

def validate_config(config):
    if config.get('schema') != CONFIG_SCHEMA:
        raise PlanningError('unsupported static config schema')
    hw = config.get('hardware', {})
    resources = indexed(hw.get('resources', []), 'resources')
    memories = indexed(hw.get('memories', []), 'memories')
    if not resources or not memories:
        raise PlanningError('hardware requires resources and memories')
    for r in resources.values():
        if r.get('kind') not in ('compute', 'communication') or (r['kind'] == 'compute' and not r.get('device')):
            raise PlanningError('resource kind and compute device required')
        number(r.get('capacity'), 'resource capacity', True)
        if type(r['capacity']) is not int:
            raise PlanningError('resource capacity must be integral')
    for m in memories.values():
        number(m.get('capacity_bytes'), 'memory capacity', True)
        if not m.get('accessible_devices'):
            raise PlanningError('memory accessible_devices required')
    for kind in ('links', 'conversions', 'synchronizations'):
        for entry in indexed(hw.get(kind, []), kind).values():
            if entry.get('resource') not in resources:
                raise PlanningError(f'{kind}: unknown resource')
            number(entry.get('latency_ms'), kind + ' latency')
            provenance(entry.get('provenance'), kind)
            if 'temporary_bytes' in entry:
                number(entry['temporary_bytes'], kind + ' temporary bytes')
            if kind != 'synchronizations':
                number(entry.get('bandwidth_bytes_per_ms'), kind + ' bandwidth', True)
            if kind == 'links' and (entry.get('source') not in memories or entry.get('target') not in memories):
                raise PlanningError('link references unknown memory')
    for impl in indexed(config.get('implementations', []), 'implementations').values():
        if not impl.get('version') or not impl.get('operations') or not impl.get('device'):
            raise PlanningError('implementation version, operations and device required')
        if impl.get('memory') not in memories or impl['device'] not in memories[impl['memory']]['accessible_devices']:
            raise PlanningError('implementation cannot access its required memory')
        if not impl.get('resources'):
            raise PlanningError('implementation resource demand required')
        for rid, units in impl['resources'].items():
            number(units, 'resource units', True)
            if type(units) is not int or rid not in resources or units > resources[rid]['capacity']:
                raise PlanningError('invalid implementation resource demand')
        if not any(resources[rid]['kind'] == 'compute' and resources[rid].get('device') == impl['device'] for rid in impl['resources']):
            raise PlanningError('implementation must demand compute resource on its own device')
        number(impl.get('temporary_bytes'), 'temporary bytes')
        if not impl.get('layout') or impl.get('dtype') != 'f32' or not impl.get('numerical_contract'):
            raise PlanningError('implementation layout, f32 interface and numerical contract required')
        cost = impl.get('cost', {})
        if cost.get('kind') != 'affine' or cost.get('scope') != 'compute_only':
            raise PlanningError('only affine compute_only ms costs supported; opaque whole-graph measurements are not compute costs')
        for key in ('fixed_ms', 'per_element_ms', 'per_flop_ms'):
            number(cost.get(key), 'compute cost ' + key)
        provenance(cost.get('provenance'), 'implementation cost')
        if not isinstance(cost.get('shape_range'), dict):
            raise PlanningError('cost valid shape_range required')
        for axis in ('m', 'n', 'k'):
            bounds = cost['shape_range'].get(axis)
            if not isinstance(bounds, list) or len(bounds) != 2 or any(type(v) is not int or v <= 0 for v in bounds) or bounds[0] > bounds[1]:
                raise PlanningError('cost shape_range requires positive ordered integer bounds for m/n/k')
    a = config.get('assumptions', {})
    if a.get('overlap') not in ('forbid', 'explicit'):
        raise PlanningError('explicit overlap policy required')
    if a.get('executions', 1) != 1:
        raise PlanningError('only single cold static graph supported')
    if not a.get('allowed_numerical_contracts'):
        raise PlanningError('program numerical acceptance contract required')
    for field in ('initial_locations', 'output_locations'):
        if not isinstance(a.get(field), dict) or any(m not in memories for m in a[field].values()):
            raise PlanningError(field + ': explicit valid locations required')
    for buffer, device in a.get('output_devices', {}).items():
        memory = a['output_locations'].get(buffer)
        if memory not in memories or device not in memories[memory]['accessible_devices']:
            raise PlanningError('output device cannot access specified output memory')
    return config

def operation_metrics(operation):
    m = operation['m'][1] - operation['m'][0]
    n = operation['n'][1] - operation['n'][0]
    k = operation.get('k', [0, 1])[1] - operation.get('k', [0, 1])[0]
    return {'m': m, 'n': n, 'k': k, 'elements': m*n, 'flops': 2*m*n*k if operation['kind'] == 'matmul' else 0}

def implementation_cost(operation, impl):
    metrics = operation_metrics(operation)
    c = impl['cost']
    return c['fixed_ms'] + c['per_element_ms']*metrics['elements'] + c['per_flop_ms']*metrics['flops']

def implementation_supports(operation, impl, config):
    if operation['kind'] not in impl['operations']:
        return False
    if impl['numerical_contract'] not in config['assumptions']['allowed_numerical_contracts']:
        return False
    metrics = operation_metrics(operation)
    for axis in ('m', 'n', 'k'):
        bounds = impl['cost']['shape_range'].get(axis)
        if not isinstance(bounds, list) or len(bounds) != 2 or not bounds[0] <= metrics[axis] <= bounds[1]:
            return False
    restrictions = impl.get('legality', {})
    for axis, multiple in restrictions.get('multiples', {}).items():
        if axis not in metrics or type(multiple) is not int or multiple <= 0 or metrics[axis] % multiple:
            return False
    if restrictions.get('accumulate') is False and operation.get('accumulate'):
        return False
    return True
