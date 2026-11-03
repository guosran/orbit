"""Region-versioned static storage and communication derivation.

No runtime state is represented. Physical copies reserve whole logical buffers
conservatively for this single cold graph; validity and transferred bytes are
tracked at the exact rectangular access-cell level.
"""
from __future__ import annotations
from collections import defaultdict, deque
from .contract import PlanningError, implementation_cost, implementation_supports


def accesses(operation, buffers):
    """Return semantic reads/writes, including the real K accumulator read."""
    op = operation; kind = op['kind']; m, n = op['m'], op['n']
    def access(buffer, region):
        shape = buffers[buffer]['shape']
        if any(len(r) != 2 or any(type(v) is not int for v in r) or not 0 <= r[0] < r[1] <= shape[i] for i, r in enumerate(region)):
            raise PlanningError('invalid static access region')
        return {'buffer': buffer, 'region': [list(r) for r in region]}
    expected = {'fill': 0, 'matmul': 2, 'relu': 1, 'swiglu': 2}[kind]
    if len(op.get('inputs', [])) != expected:
        raise PlanningError('operation input count mismatch')
    reads = []
    if kind == 'matmul':
        reads = [access(op['inputs'][0], [m, op['k']]), access(op['inputs'][1], [op['k'], n])]
        if op.get('accumulate', False):
            reads.append(access(op['output'], [m, n]))
    else:
        reads = [access(b, [m, n]) for b in op.get('inputs', [])]
    return reads, [access(op['output'], [m, n])]


def build_graph(program, config, steps, assignment):
    buffers = {b['id']: b for b in program['buffers']}
    hw = config['hardware']; assumptions = config['assumptions']
    impls = {i['id']: i for i in config['implementations']}
    boundaries = {b: [set((0, v['shape'][0])), set((0, v['shape'][1]))] for b, v in buffers.items()}
    step_access = {}
    for step in steps:
        rw = accesses(step['operation'], buffers); step_access[step['id']] = rw
        for a in rw[0]+rw[1]:
            for axis in range(2): boundaries[a['buffer']][axis].update(a['region'][axis])
    cells = {}
    total_cells = 0
    max_cells = assumptions.get('max_region_cells', 100000)
    if type(max_cells) is not int or max_cells <= 0:
        raise PlanningError('max_region_cells must be a positive integer')
    for b, bounds in boundaries.items():
        x, y = map(sorted, bounds)
        total_cells += (len(x)-1)*(len(y)-1)
        if total_cells > max_cells:
            raise PlanningError('exact access partition exceeds max_region_cells planning budget')
        cells[b] = [(x[i], x[i+1], y[j], y[j+1]) for i in range(len(x)-1) for j in range(len(y)-1)]
    def covered(a):
        r = a['region']
        return [(a['buffer'], c) for c in cells[a['buffer']] if r[0][0] <= c[0] and c[1] <= r[0][1] and r[1][0] <= c[2] and c[3] <= r[1][1]]
    def entry(key, allocation, version):
        b, c = key
        return {'buffer': b, 'region': [[c[0],c[1]],[c[2],c[3]]], 'allocation': allocation, 'version': version}
    def byte_count(keys):
        return sum((c[1]-c[0])*(c[3]-c[2])*4 for _,c in keys)
    allocations = {}; alloc_keys = {}; events = []; current = {}; copies = {}; readers = defaultdict(set); serial = 0
    def allocation(b, memory, layout='row_major', dtype='f32'):
        if dtype != 'f32': raise PlanningError('dtype conversion is outside the static subset')
        key = (b, memory, layout, dtype)
        if key not in alloc_keys:
            aid = f'alloc:{len(allocations)}'; alloc_keys[key] = aid
            shape = buffers[b]['shape']
            allocations[aid] = {'id':aid,'buffer':b,'memory':memory,'layout':layout,'dtype':dtype,'region':[[0,shape[0]],[0,shape[1]]],'bytes':shape[0]*shape[1]*4,'created_by':None,'last_users':[],'lifetime':'graph'}
        return alloc_keys[key]
    def emit(kind, deps, resources, duration, reads, writes, **extra):
        nonlocal serial
        eid = extra.pop('event_id', f'{kind}:{serial}'); serial += 1
        e = {'id':eid,'kind':kind,'depends_on':sorted(set(d for d in deps if d)), 'resource_demands':dict(resources),'duration_ms':duration,'reads':reads,'writes':writes, **extra}
        events.append(e)
        for a in reads+writes:
            al = allocations[a['allocation']]
            al['last_users'].append(eid)
            if kind != 'sync' and a in writes and al['created_by'] is None and al['buffer'] not in initial_buffers:
                al['created_by'] = eid
        return eid
    initial_buffers = {b['id'] for b in buffers.values() if b.get('role') == 'input'}
    for b in initial_buffers:
        if b not in assumptions['initial_locations']: raise PlanningError('missing initial location for '+b)
        aid = allocation(b, assumptions['initial_locations'][b])
        allocations[aid]['initial'] = True
        for c in cells[b]:
            key = (b,c); current[key] = 'initial:'+b; copies[(key,aid)] = (current[key],None,None)
    def route(source, target):
        queue = deque([(source, [])]); visited = {source}
        while queue:
            here, path = queue.popleft()
            if here == target: return path
            for link in hw.get('links', []):
                if link['source'] == here and link['target'] not in visited:
                    visited.add(link['target']); queue.append((link['target'],path+[link]))
        raise PlanningError(f'no communication path {source} -> {target}')
    def transport(keys, source_aid, dest_aid, kind, descriptor):
        nbytes = byte_count(keys)
        if nbytes < descriptor.get('min_bytes', 0) or nbytes > descriptor.get('max_bytes', float('inf')):
            raise PlanningError('communication cost outside valid byte range')
        read = [entry(k,source_aid,current[k]) for k in keys]; write = [entry(k,dest_aid,current[k]) for k in keys]
        deps = [copies[(k,source_aid)][1] for k in keys]
        deps += [r for k in keys for r in readers[(k,dest_aid)]]
        eid = emit(kind,deps,{descriptor['resource']:1},descriptor['latency_ms']+nbytes/descriptor['bandwidth_bytes_per_ms'],read,write,bytes=nbytes,provenance=descriptor['provenance'],cost_boundary='transfer_or_conversion_including_completion_visibility',**{('link' if kind=='copy' else 'conversion'):descriptor['id']})
        if descriptor.get('temporary_bytes', 0):
            dest=allocations[dest_aid]; aid='temporary:'+eid
            allocations[aid]={'id':aid,'buffer':None,'memory':dest['memory'],'layout':dest['layout'],'dtype':dest['dtype'],'region':None,'bytes':descriptor['temporary_bytes'],'created_by':eid,'last_users':[eid],'lifetime':'event'}
        for k in keys:
            copies[(k,dest_aid)] = (current[k],eid,None); readers[(k,source_aid)].add(eid)
        return eid
    def ensure(keys, target_aid, device=None):
        target = allocations[target_aid]
        # Group regions sharing one current source, so a multi-consumer copy is
        # paid once and disjoint tiles never imply full-buffer transfer bytes.
        grouped = defaultdict(list)
        for key in keys:
            if key not in current: raise PlanningError('read before initialization: '+key[0])
            old = copies.get((key,target_aid))
            if old and old[0] == current[key]: continue
            valid = [(aid, state) for (k,aid),state in copies.items() if k==key and state[0]==current[key]]
            if not valid: raise PlanningError('no valid producer for '+key[0])
            valid.sort(key=lambda x:(allocations[x[0]]['memory']!=target['memory'],allocations[x[0]]['layout']!=target['layout'],x[0]))
            feasible = []
            for source_aid, state in valid:
                source = allocations[source_aid]
                if source['layout'] != target['layout'] and not any(c['memory']==source['memory'] and c['source_layout']==source['layout'] and c['target_layout']==target['layout'] and c['source_dtype']==source['dtype'] and c['target_dtype']==target['dtype'] for c in hw.get('conversions', [])):
                    continue
                try:
                    route(source['memory'],target['memory'])
                    feasible.append(source_aid)
                except PlanningError:
                    pass
            if not feasible:
                raise PlanningError('no valid communication/layout path for '+key[0])
            grouped[feasible[0]].append(key)
        for source_aid, group in grouped.items():
            source = allocations[source_aid]; b = source['buffer']
            if source['layout'] != target['layout']:
                conversion = next((c for c in hw.get('conversions',[]) if c['memory']==source['memory'] and c['source_layout']==source['layout'] and c['target_layout']==target['layout'] and c['source_dtype']==source['dtype'] and c['target_dtype']==target['dtype']),None)
                if conversion is None: raise PlanningError('missing explicit layout conversion')
                converted = allocation(b,source['memory'],target['layout'],target['dtype'])
                transport(group,source_aid,converted,'convert',conversion); source_aid=converted; source=allocations[converted]
            for link in route(source['memory'],target['memory']):
                dest = allocation(b,link['target'],target['layout'],target['dtype'])
                transport(group,source_aid,dest,'copy',link); source_aid=dest
        deps = []; sync_groups=defaultdict(list)
        for key in keys:
            state = copies[(key,target_aid)]; deps.append(state[1])
            if device and state[2] and state[2]!=device:
                sync_groups[state[2]].append(key)
        for producer_device, group in sync_groups.items():
            sync = next((s for s in hw.get('synchronizations',[]) if s['source_device']==producer_device and s['target_device']==device and s['memory']==target['memory']),None)
            if sync is None: raise PlanningError('missing explicit shared-memory synchronization cost')
            token = (target_aid,device,tuple((k,current[k],copies[(k,target_aid)][1]) for k in group))
            if token not in sync_cache:
                sync_cache[token] = emit('sync',[copies[(k,target_aid)][1] for k in group],{sync['resource']:1},sync['latency_ms'],[entry(k,target_aid,current[k]) for k in group],[],synchronization=sync['id'],source_device=producer_device,target_device=device,memory=target['memory'],provenance=sync['provenance'],bytes=0,cost_boundary='visibility_only')
            deps.append(sync_cache[token])
        return deps
    sync_cache = {}
    for step in steps:
        sid = step['id']; impl = impls[assignment[sid]]
        if not implementation_supports(step['operation'],impl,config): raise PlanningError('unsupported assigned implementation')
        reads, writes = step_access[sid]; read_entries=[]; write_entries=[]
        deps=['compute:'+d for d in step['depends_on']]
        for a in reads:
            aid=allocation(a['buffer'],impl['memory'],impl['layout'],impl['dtype']); keys=covered(a)
            deps+=ensure(keys,aid,impl['device']); read_entries += [entry(k,aid,current[k]) for k in keys]
        for a in writes:
            aid=allocation(a['buffer'],impl['memory'],impl['layout'],impl['dtype'])
            for key in covered(a):
                # WAR/WAW create ordering only; they do not request any copy.
                deps += [state[1] for (k,_),state in copies.items() if k==key and state[0]==current.get(key)]
                deps += [r for (k,_),rr in readers.items() if k==key for r in rr]
                write_entries.append(entry(key,aid,sid))
        eid=emit('compute',deps,impl['resources'],implementation_cost(step['operation'],impl),read_entries,write_entries,event_id='compute:'+sid,step_id=sid,task_id=step['task_id'],operation_index=step['operation_index'],implementation=impl['id'],implementation_id=impl['id'],device=impl['device'],provenance=impl['cost']['provenance'],cost_provenance=impl['cost']['provenance'],cost_boundary='compute_and_local_memory_access_only')
        if impl.get('legality', {}).get('overwrite_zero_accumulator'):
            events[-1]['implementation_semantics'] = 'full_k_overwrite_of_proven_zero_accumulator; semantic_initialization_and_read_retained'
        for a in read_entries:
            r=a['region']; key=(a['buffer'],(r[0][0],r[0][1],r[1][0],r[1][1])); readers[(key,a['allocation'])].add(eid)
        for a in write_entries:
            r=a['region']; key=(a['buffer'],(r[0][0],r[0][1],r[1][0],r[1][1])); current[key]=sid; copies[(key,a['allocation'])]=(sid,eid,impl['device'])
        if impl['temporary_bytes']:
            aid='temporary:'+sid
            allocations[aid]={'id':aid,'buffer':None,'memory':impl['memory'],'layout':impl['layout'],'dtype':impl['dtype'],'region':None,'bytes':impl['temporary_bytes'],'created_by':eid,'last_users':[eid],'lifetime':'event'}
    outputs=[]; output_dependencies=[]; output_consumers={}
    memories={m['id']:m for m in hw['memories']}
    for b in program.get('outputs',[]):
        if b not in assumptions['output_locations']: raise PlanningError('missing output location for '+b)
        memory=assumptions['output_locations'][b]
        devices=memories[memory]['accessible_devices']
        device=assumptions.get('output_devices',{}).get(b)
        if device is None and len(devices)==1: device=devices[0]
        if device not in devices:
            raise PlanningError('output consumer device must be explicit for shared memory: '+b)
        output_consumers[b]=device
        aid=allocation(b,memory); keys=[(b,c) for c in cells[b]]
        output_dependencies += ensure(keys,aid,device)
        outputs += [entry(k,aid,current[k]) for k in keys]
    return {'events':events,'allocations':list(allocations.values()),'outputs':outputs,'output_dependencies':sorted(set(d for d in output_dependencies if d)),'output_consumers':output_consumers,'storage_policy':'whole_buffer_per_location_layout_retained_for_single_graph','initial_state':'cold_inputs_only_visible_to_all_accessible_devices_at_declared_initial_locations','region_cell_count':total_cells,'cost_boundary':'compute includes local memory access; explicit copies/conversions include completion visibility; direct cross-device sharing requires explicit sync'}
