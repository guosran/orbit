"""Independent static-plan verifier.

This module deliberately imports neither the mapper nor communication builder.
It reconstructs semantic accesses, versions, operation costs, physical allocation
sizes and resource occupancy from the public program and target configuration.
Planner metadata is evidence to check, never an authority for correctness.
"""
from __future__ import annotations
import math
import hashlib
import json

_EPS = 1e-8


def _same(a, b):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(a-b) <= _EPS * max(1, abs(a), abs(b))


def _inside(region, bounds):
    return all(bounds[i][0] <= region[i][0] < region[i][1] <= bounds[i][1] for i in (0, 1))


def _accesses(op):
    r = [op['m'], op['n']]
    reads = []
    if op['kind'] == 'matmul':
        reads = [(op['inputs'][0], [op['m'], op['k']]),
                 (op['inputs'][1], [op['k'], op['n']])]
        if op.get('accumulate'):
            reads.append((op['output'], r))
    elif op['kind'] in ('relu', 'swiglu'):
        reads = [(b, r) for b in op['inputs']]
    elif op['kind'] != 'fill':
        raise ValueError('unknown semantic operation')
    return reads, [(op['output'], r)]


def check_plan(program, config, plan):
    """Return a JSON-serializable verdict; malformed plans fail closed."""
    errors = []
    def require(condition, message):
        if not condition:
            errors.append(message)
    try:
        result = _check(program, config, plan, require)
    except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError, OverflowError) as exc:
        errors.append('malformed plan/config/program: ' + str(exc))
        result = {}
    return dict(passed=not errors, errors=errors, checker='independent-static-v1', **result)


def _check(program, config, plan, require):
    require(plan.get('schema') == 'orbit.static.plan.v1', 'wrong plan schema')
    require(config.get('schema') == 'orbit.static.config.v1', 'wrong config schema')
    def identity(value):
        return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    require(plan['program_id'] == identity(program), 'program identity mismatch')
    require(plan['configuration_id'] == identity(config), 'configuration identity mismatch')
    require(plan['plan_id'] == identity({k:v for k,v in plan.items() if k not in ('plan_id','validation')}), 'plan identity mismatch')
    def index(rows, label):
        result = {r['id']: r for r in rows}
        require(len(result) == len(rows), 'duplicate ' + label)
        return result
    buffers = index(program['buffers'], 'semantic buffers')
    tasks = index(program['tasks'], 'semantic tasks')
    allocations = index(plan['allocations'], 'allocations')
    events = index(plan['events'], 'events')
    resources = index(config['hardware']['resources'], 'resources')
    memories = index(config['hardware']['memories'], 'memories')
    impls = index(config['implementations'], 'implementations')
    links = index(config['hardware'].get('links', []), 'links')
    conversions = index(config['hardware'].get('conversions', []), 'conversions')
    syncs = index(config['hardware'].get('synchronizations', []), 'synchronizations')
    assumptions = config['assumptions']
    if 'program_source_id' in assumptions:
        require(assumptions['program_source_id']==program['source_id'],'numerical acceptance belongs to another program')
    def numeric(value, positive=False):
        return not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(value) and (value>0 if positive else value>=0)
    def provenance(value):
        return isinstance(value,dict) and value.get('kind') in ('synthetic','theoretical','calibrated','measured') and bool(value.get('source'))
    for r in resources.values():
        require(type(r['capacity']) is int and r['capacity']>0,'invalid resource capacity')
        require(r.get('kind') in ('compute','communication'),'unknown resource kind')
        require(r['kind']!='compute' or bool(r.get('device')),'compute resource lacks device')
    for m in memories.values():
        require(type(m['capacity_bytes']) is int and m['capacity_bytes']>0,'invalid memory capacity')
    for impl in impls.values():
        require(bool(impl.get('version')),'missing implementation version')
        require(bool(impl['resources']),'missing implementation resource demand')
        require(all(r in resources and type(units) is int and 0<units<=resources[r]['capacity'] for r,units in impl['resources'].items()),'invalid implementation resource units')
        require(any(resources.get(r,{}).get('kind')=='compute' and resources[r].get('device')==impl['device'] for r in impl['resources']),'implementation lacks execution resource on its device')
        require(type(impl['temporary_bytes']) is int and impl['temporary_bytes']>=0,'invalid temporary size')
        cost=impl['cost']
        require(all(numeric(cost[k]) for k in ('fixed_ms','per_element_ms','per_flop_ms')) and provenance(cost['provenance']),'missing/invalid implementation cost')
    for desc in list(links.values())+list(conversions.values())+list(syncs.values()):
        require(numeric(desc['latency_ms']) and provenance(desc['provenance']),'missing/invalid communication cost')
        require(type(desc.get('temporary_bytes',0)) is int and desc.get('temporary_bytes',0)>=0,'invalid communication scratch')
        if desc['id'] not in syncs:
            require(numeric(desc['bandwidth_bytes_per_ms'],True),'unknown/nonpositive bandwidth')
    require(assumptions['overlap'] in ('explicit','forbid'), 'unknown overlap policy')
    require(assumptions.get('executions', 1) == 1, 'only single execution checked')
    for b in buffers.values():
        require(not b.get('alias_of') and b['dtype']=='f32' and len(b['shape'])==2 and all(type(n) is int and n>0 for n in b['shape']), 'unsupported source storage')
    # Independently topologically order public tasks and expand ordered bodies.
    steps, pending, done = {}, list(tasks.values()), set()
    while pending:
        ready = [t for t in pending if set(t.get('depends_on', [])) <= done]
        if not ready:
            raise ValueError('cyclic/missing semantic dependency')
        for task in ready:
            require(bool(task['operations']), 'empty task')
            for i, op in enumerate(task['operations']):
                key = f'{task["id"]}:{i}'
                deps = [f'{task["id"]}:{i-1}'] if i else [f'{d}:{len(tasks[d]["operations"])-1}' for d in task.get('depends_on', [])]
                steps[key] = dict(operation=op, dependencies=deps)
            pending.remove(task); done.add(task['id'])
    # Atomic rectangles are constructed from *source* access boundaries.
    cuts = {bid: [{0, b['shape'][0]}, {0, b['shape'][1]}] for bid, b in buffers.items()}
    for step in steps.values():
        for bid, region in sum(_accesses(step['operation']), []):
            require(_inside(region, [[0, n] for n in buffers[bid]['shape']]), 'source access out of bounds')
            for axis in (0, 1):
                cuts[bid][axis].update(region[axis])
    cells = {}
    for bid, axes in cuts.items():
        x, y = map(sorted, axes)
        cells[bid] = [(a, b, c, d) for a, b in zip(x, x[1:]) for c, d in zip(y, y[1:])]
    def covered(bid, region):
        require(_inside(region, [[0, n] for n in buffers[bid]['shape']]), 'event region out of bounds')
        selected = [c for c in cells[bid] if _inside([[c[0], c[1]], [c[2], c[3]]], region)]
        require(sum((c[1]-c[0])*(c[3]-c[2]) for c in selected) == (region[0][1]-region[0][0])*(region[1][1]-region[1][0]), 'event region is not semantic-cell aligned')
        return [(bid, c) for c in selected]
    semantic = {(bid, c): 'initial:'+bid for bid, b in buffers.items() if b['role'] == 'input' for c in cells[bid]}
    expected_reads, expected_writes = {}, {}
    for sid, step in steps.items():
        reads, writes = _accesses(step['operation'])
        er, ew = {}, {}
        for bid, region in reads:
            for key in covered(bid, region):
                require(key in semantic, f'{sid}: missing initialization/producer')
                er[key] = semantic.get(key, 'UNINITIALIZED')
        for bid, region in writes:
            for key in covered(bid, region):
                ew[key] = sid; semantic[key] = sid
        expected_reads[sid], expected_writes[sid] = er, ew
    compute = {}
    for event in events.values():
        if event['kind'] == 'compute':
            sid = event['step_id']
            require(sid in steps and sid not in compute, 'missing/duplicate/unknown semantic compute')
            compute[sid] = event
    require(set(compute) == set(steps), 'compute coverage omits semantic operations (including fill and K partials)')
    require(plan['assignment']=={sid:e['implementation'] for sid,e in compute.items()},'assignment differs from actual compute implementations')
    end = max([e['end_ms'] for e in events.values()] + [0])
    require(_same(plan['totals']['makespan_ms'], end), 'forged makespan')
    require(_same(plan['makespan_ms'],end),'forged top-level makespan')
    # Dependencies must form a DAG even if zero-duration events occur.
    ancestors = {}
    def parents(eid, stack=()):
        if eid in ancestors:
            return ancestors[eid]
        if eid in stack:
            raise ValueError('event dependency cycle')
        deps = set(events[eid]['depends_on'])
        require(eid not in deps, 'self event dependency')
        result = set(deps)
        for dep in deps:
            result.update(parents(dep, stack+(eid,)))
        ancestors[eid] = result
        return result
    for eid, e in events.items():
        require(all(math.isfinite(float(e[f])) and e[f] >= 0 for f in ('start_ms', 'end_ms', 'duration_ms')), 'invalid event time')
        require(_same(e['end_ms']-e['start_ms'], e['duration_ms']), 'duration/time mismatch')
        for dep in e['depends_on']:
            require(events[dep]['end_ms'] <= e['start_ms']+_EPS, 'dependency overlaps consumer')
        parents(eid)
    for sid, e in compute.items():
        for dep in steps[sid]['dependencies']:
            require(compute[dep]['id'] in ancestors[e['id']], 'missing semantic task/operation ordering')
    # Allocation sizes/lifetimes do not trust byte counts or last_users metadata.
    physical_sizes = {}
    for aid, a in allocations.items():
        require(a['memory'] in memories, 'unknown allocation memory')
        if a.get('lifetime') == 'event':
            owner = events[a['created_by']]
            require(owner['kind'] in ('compute','convert','copy'), 'invalid temporary owner')
            if owner['kind']=='compute':size=impls[owner['implementation']]['temporary_bytes']
            elif owner['kind']=='convert':size=conversions[owner['conversion']].get('temporary_bytes',0)
            else:size=links[owner['link']].get('temporary_bytes',0)
            require(_same(a['start_ms'],owner['start_ms']) and _same(a['end_ms'],owner['end_ms']), 'forged temporary lifetime')
        else:
            b = buffers[a['buffer']]
            require(a['dtype'] == 'f32', 'unsupported physical dtype')
            require(a['region'] == [[0, n] for n in b['shape']], 'data allocation must conservatively hold whole buffer')
            size = b['shape'][0]*b['shape'][1]*4
            require(a.get('lifetime') == 'graph' and _same(a['start_ms'],0) and _same(a['end_ms'],end), 'forged graph storage lifetime')
        require(a['bytes'] == size, 'forged allocation byte size')
        physical_sizes[aid] = size
    # Real values are tracked separately from planner version labels. A content
    # token derives only from an initial input or a checked semantic operation.
    values = {}
    for aid, a in allocations.items():
        if a.get('lifetime') == 'event':
            continue
        bid = a['buffer']
        if buffers[bid]['role'] == 'input' and assumptions['initial_locations'].get(bid) == a['memory'] and a['layout'] == 'row_major' and a['dtype'] == 'f32':
            for c in cells[bid]:
                values[aid, (bid, c)] = [('initial:'+bid, 0, None, None)]
    # A topological order breaks ties for declared zero-duration events.
    ordered = sorted(events.values(), key=lambda e: (e['start_ms'], len(ancestors[e['id']]), e['id']))
    for e in ordered:
        eid, kind = e['id'], e['kind']
        require(kind in ('compute', 'copy', 'convert', 'sync'), 'unknown event kind')
        read_tokens, write_tokens, read_entries = {}, {}, {}
        for label, tokens in (('reads', read_tokens), ('writes', write_tokens)):
            for access in e[label]:
                aid, bid = access['allocation'], access['buffer']
                a = allocations[aid]
                require(a.get('buffer') == bid, 'allocation logical storage mismatch')
                for cell in covered(bid, access['region']):
                    require(cell not in tokens, 'duplicate event access cell')
                    tokens[cell] = access.get('version')
                    if label == 'reads':
                        read_entries[cell] = access
        content = {}
        for cell, access in read_entries.items():
            candidates = [v for v in values.get((access['allocation'], cell), []) if v[1] <= e['start_ms']+_EPS and (v[2] is None or v[2] in ancestors[eid])]
            require(bool(candidates), 'read before production/transfer or missing dependency')
            if candidates:
                value = max(candidates, key=lambda v: (v[1], -1 if v[2] is None else len(ancestors[v[2]])))
                content[cell] = value[0]
                require(access.get('version') == value[0], 'forged read version')
                if kind == 'compute' and value[3] and value[3] != e['device']:
                    # Same physical allocation produced by another device needs
                    # an explicit visibility event, unless a copy established it.
                    visible = any(s['kind'] == 'sync' and s['id'] in ancestors[eid] and s.get('source_device') == value[3] and s.get('target_device') == e['device'] and value[2] in ancestors[s['id']] and any(r['allocation'] == access['allocation'] and r.get('version')==value[0] and cell in covered(r['buffer'], r['region']) for r in s['reads']) for s in events.values())
                    require(visible, 'cross-device shared-memory value lacks synchronization')
        if kind == 'compute':
            sid = e['step_id']; op = steps[sid]['operation']; impl = impls[e['implementation']]
            require(content == expected_reads[sid], 'compute consumes wrong/missing semantic version or region')
            require(write_tokens == expected_writes[sid], 'compute writes wrong region/version')
            require(e['device'] == impl['device'], 'forged implementation device')
            require(op['kind'] in impl['operations'], 'implementation does not support operation')
            require(impl['numerical_contract'] in assumptions['allowed_numerical_contracts'], 'numerical contract mismatch')
            m, n = op['m'][1]-op['m'][0], op['n'][1]-op['n'][0]
            k = op.get('k', [0, 1])[1]-op.get('k', [0, 1])[0]
            dims = dict(m=m, n=n, k=k)
            c = impl['cost']
            require(e['provenance'] == c['provenance'], 'forged compute cost provenance')
            require(c['scope'] == 'compute_only' and c['kind'] == 'affine', 'invalid/double-counted compute cost boundary')
            for axis, value in dims.items():
                bounds = c['shape_range'][axis]
                require(bounds[0] <= value <= bounds[1], 'implementation outside cost validity range')
            for axis, multiple in impl.get('legality', {}).get('multiples', {}).items():
                require(dims[axis] % multiple == 0, 'implementation shape illegal')
            require(not (impl.get('legality', {}).get('accumulate') is False and op.get('accumulate')), 'implementation cannot accumulate')
            limits = impl.get('legality', {})
            if limits.get('full_k_only'):
                require(op['kind'] == 'matmul' and op['k'] == [0,buffers[op['inputs'][0]]['shape'][1]], 'implementation requires full K')
            if limits.get('overwrite_zero_accumulator'):
                task, pos = sid.rsplit(':', 1)
                previous = steps.get(f'{task}:{int(pos)-1}', {}).get('operation', {})
                require(previous.get('kind') == 'fill' and all(previous.get(f) == op.get(f) for f in ('output','m','n')), 'overwrite implementation requires preceding zero fill')
            duration = c['fixed_ms']+c['per_element_ms']*m*n+c['per_flop_ms']*(2*m*n*k if op['kind']=='matmul' else 0)
            require(_same(e['duration_ms'], duration), 'forged compute duration')
            require(e['resource_demands'] == impl['resources'], 'forged compute resource demand')
            for access in e['reads']+e['writes']:
                a = allocations[access['allocation']]
                require(a['memory'] == impl['memory'] and a['layout'] == impl['layout'] and a['dtype'] == impl['dtype'], 'implementation storage/layout contract violated')
                require(e['device'] in memories[a['memory']]['accessible_devices'], 'compute cannot access allocation')
            temps = [a for a in allocations.values() if a.get('lifetime') == 'event' and a.get('created_by') == eid]
            require(sum(a['bytes'] for a in temps) == impl['temporary_bytes'], 'implementation temporary omitted or duplicated')
            require(all(a['memory'] == impl['memory'] for a in temps), 'temporary in wrong memory')
            content = expected_writes[sid]
        elif kind in ('copy', 'convert'):
            require(set(read_tokens) == set(write_tokens) and bool(read_tokens), 'transfer source/destination region mismatch')
            require(write_tokens == read_tokens, 'transfer forges content version')
            desc = (links if kind == 'copy' else conversions)[e['link'] if kind=='copy' else e['conversion']]
            require(e['provenance'] == desc['provenance'], 'forged communication cost provenance')
            size = sum((cell[1][1]-cell[1][0])*(cell[1][3]-cell[1][2])*4 for cell in read_tokens)
            require(e['bytes'] == size, 'forged transfer bytes')
            require(_same(e['duration_ms'], desc['latency_ms']+size/desc['bandwidth_bytes_per_ms']), 'forged communication cost')
            require(e['resource_demands'] == {desc['resource']: 1}, 'forged communication resource demand')
            if kind == 'copy':
                require(desc.get('min_bytes', 0) <= size <= desc.get('max_bytes', math.inf), 'transfer outside cost validity')
            for r, w in zip(sorted(e['reads'], key=lambda a:(a['buffer'],a['region'])), sorted(e['writes'], key=lambda a:(a['buffer'],a['region']))):
                a, b = allocations[r['allocation']], allocations[w['allocation']]
                if kind == 'copy':
                    require(a['memory'] == desc['source'] and b['memory'] == desc['target'], 'wrong communication path')
                    require(a['layout'] == b['layout'] and a['dtype'] == b['dtype'], 'copy silently converts layout/dtype')
                else:
                    require(a['memory'] == b['memory'] == desc['memory'], 'conversion in wrong memory')
                    require((a['layout'], b['layout'], a['dtype'], b['dtype']) == (desc['source_layout'], desc['target_layout'], desc['source_dtype'], desc['target_dtype']), 'conversion layout/dtype mismatch')
            temps=[a for a in allocations.values() if a.get('lifetime')=='event' and a.get('created_by')==eid]
            require(sum(a['bytes'] for a in temps)==desc.get('temporary_bytes',0),'communication scratch omitted or duplicated')
            require(all(a['memory']==(desc['memory'] if kind=='convert' else desc['target']) for a in temps),'communication scratch in wrong memory')
        elif kind == 'sync':
            desc = syncs[e['synchronization']]
            require(not e['writes'] and bool(e['reads']), 'sync cannot produce data')
            require(_same(e['duration_ms'], desc['latency_ms']), 'forged synchronization cost')
            require(e['resource_demands'] == {desc['resource']:1}, 'forged synchronization demand')
            require(all(e.get(f) == desc[f] for f in ('source_device', 'target_device', 'memory')), 'wrong synchronization path')
            require(e['provenance'] == desc['provenance'], 'forged sync cost provenance')
        for a in e['writes']:
            for cell in covered(a['buffer'], a['region']):
                values.setdefault((a['allocation'], cell), []).append((content.get(cell, 'INVALID'), e['end_ms'], eid, e.get('device') if kind=='compute' else None))
    # All public outputs must end in the declared externally visible location.
    for bid in program['outputs']:
        location = assumptions['output_locations'][bid]
        device = assumptions.get('output_devices',{}).get(bid)
        if device is None:
            require(len(memories[location]['accessible_devices'])==1,'ambiguous output consumer device')
            device=memories[location]['accessible_devices'][0]
        require(device in memories[location]['accessible_devices'],'output consumer cannot access memory')
        for c in cells[bid]:
            candidates = []
            for aid,a in allocations.items():
                if a.get('buffer') == bid and a['memory'] == location and a['layout']=='row_major' and a['dtype']=='f32':
                    contents=values.get((aid,(bid,c)),[])
                    if contents:
                        candidates.append(max(contents,key=lambda v:(v[1],-1 if v[2] is None else len(ancestors[v[2]]))))
            require(any(v[0] == semantic.get((bid,c)) and v[1] <= end+_EPS for v in candidates), 'missing final output/location')
            good=[v for v in candidates if v[0]==semantic.get((bid,c)) and v[1]<=end+_EPS]
            for value in good:
                if value[3] and value[3]!=device:
                    require(any(s['kind']=='sync' and s['id'] in plan.get('output_dependencies',[]) and s.get('source_device')==value[3] and s.get('target_device')==device and value[2] in ancestors[s['id']] and any(r['buffer']==bid and r.get('version')==value[0] and (bid,c) in covered(bid,r['region']) for r in s['reads']) for s in events.values()),'final output lacks consumer visibility synchronization')
    # Sweep half-open intervals, releasing at an endpoint before acquiring.
    def peak(intervals):
        points = sorted(set(x for a,b,v in intervals for x in (a,b)))
        return max([sum(v for a,b,v in intervals if a <= t < b) for t in points] + [0])
    resource_peaks = {}
    for rid, r in resources.items():
        intervals = [(e['start_ms'], e['end_ms'], e['resource_demands'].get(rid,0)) for e in events.values()]
        resource_peaks[rid] = peak(intervals)
        require(resource_peaks[rid] <= r['capacity'], 'resource capacity exceeded: '+rid)
    for e in events.values():
        require(all(r in resources and type(v) is int and v > 0 for r,v in e['resource_demands'].items()), 'invalid resource demand')
    if assumptions['overlap'] == 'forbid':
        require(peak([(e['start_ms'],e['end_ms'],1) for e in events.values()]) <= 1, 'overlap prohibited by hardware assumptions')
    memory_peaks = {}
    for mid, memory in memories.items():
        intervals = []
        graph_bytes = 0
        for aid, a in allocations.items():
            if a['memory'] != mid:
                continue
            if a.get('lifetime') == 'event':
                e = events[a['created_by']]; intervals.append((e['start_ms'],e['end_ms'],physical_sizes[aid]))
            else:
                graph_bytes += physical_sizes[aid]
        zero_duration_floor = max([v for a,b,v in intervals]+[0])
        memory_peaks[mid] = graph_bytes+max(peak(intervals),zero_duration_floor)
        require(memory_peaks[mid] <= memory['capacity_bytes'], 'memory capacity exceeded: '+mid)
    require(plan['totals']['peak_memory_bytes'] == memory_peaks, 'forged peak memory metadata')
    for kind in ('compute','copy','convert','sync'):
        require(_same(plan['totals'][kind+'_ms'],sum(e['duration_ms'] for e in events.values() if e['kind']==kind)), 'forged aggregate '+kind+' cost')
    require(plan['totals']['transfer_bytes'] == sum(e['bytes'] for e in events.values() if e['kind']=='copy'), 'forged aggregate transfer bytes')
    # Physical hazards are checked independently of the declared producer list.
    occupancy = {}
    for e in events.values():
        for mode in ('reads','writes'):
            for a in e[mode]:
                for cell in covered(a['buffer'],a['region']):
                    occupancy.setdefault((a['allocation'],cell),{}).setdefault(e['id'],set()).add(mode)
    for users in occupancy.values():
        pairs = list(users.items())
        for i,(left,lm) in enumerate(pairs):
            for right,rm in pairs[i+1:]:
                if 'writes' in lm or 'writes' in rm:
                    require(left in ancestors[right] or right in ancestors[left], 'unordered physical read/write hazard')
    return dict(operation_count=len(steps),event_count=len(events),makespan_ms=end,peak_memory_bytes=memory_peaks,resource_peaks=resource_peaks)
