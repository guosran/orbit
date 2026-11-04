#!/usr/bin/env python3
"""Independent read-only raw trace audit; never imports a device runtime."""
import argparse
import collections
import importlib.util
import json
from pathlib import Path
import re
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from hardware_evidence import trace_audit, graph_trace_audit


def marker_check(events,expected_per_core,expected_cores,require_stop):
    processes={e['pid']:e.get('args',{}).get('name','') for e in events if e.get('name')=='process_name'}
    rows=[];errors=[]
    for pid,name in processes.items():
        if not name.startswith('core_trace'):continue
        start=[e['ts'] for e in events if e.get('pid')==pid and e.get('name')=='INSTR_EVENT_0' and e.get('ph')=='B']
        end=[e['ts'] for e in events if e.get('pid')==pid and e.get('name')=='INSTR_EVENT_1' and e.get('ph')=='B']
        stop=[e['ts'] for e in events if e.get('pid')==pid and e.get('name')=='USER_EVENT_0' and e.get('ph')=='B']
        valid=len(start)==len(end)==expected_per_core and all(a<b for a,b in zip(start,end)) and all(a<b for a,b in zip(end,start[1:]))
        if not valid:errors.append(name+': missing/extra/out-of-order marker pair')
        if require_stop and (len(stop)!=1 or not end or stop[0]<=end[-1]):errors.append(name+': terminal scope-end not observed after final marked interval')
        rows.append(dict(process=pid,tile=name,starts=len(start),ends=len(end),scope_end_cycles=stop,marked_cycles=[b-a for a,b in zip(start,end)],valid_pairs=valid))
    if len(rows)!=expected_cores:errors.append('traced core count differs from compiled packet configuration')
    return dict(passed=not errors,errors=errors,expected_pairs_per_core=expected_per_core,expected_cores=expected_cores,cores=rows)


def audit(path):
    manifest=json.loads(path.read_text());events=json.loads((path.parent/manifest['decoded_native_path']).read_text());base=trace_audit(path)
    if 'physical_shape' in manifest:
        m,k,n=manifest['physical_shape'];expected=m//128*(n//256)*(k//64)
        shape=path.parent.parent/('shape-'+'-'.join(map(str,(m,k,n))))
        metadata=json.loads((shape/'instrumentation/instrumentation.json').read_text())
        cores=len(metadata['traced_cores']);stop=True
        physical=(path.parent/'physical.mlir').read_text()
        # Actual compiled code must contain two event statements in every core,
        # not merely subscribed event names in configuration.
        if physical.count('aie.event(0)')<len(metadata['cores']) or physical.count('aie.event(1)')!=physical.count('aie.event(0)'):
            base['passed']=False;base['errors'].append('actual compiled physical MLIR lacks expected emitted markers')
    else:
        expected=4;cores=1;stop=False
        reference=(np.arange(4096,dtype=np.int32)%97-48)*3
        for filename in ('output.npy','expected.npy'):
            if not np.array_equal(np.load(path.parent/filename),reference):
                base['passed']=False;base['errors'].append('minimal independent reference mismatch: '+filename)
        if (path.parent/'output-off.npy').exists() and not np.array_equal(np.load(path.parent/'output-off.npy'),reference):
            base['passed']=False;base['errors'].append('minimal trace-off independent reference mismatch')
    markers=marker_check(events,expected,cores,stop)
    # Complete means only the explicitly enumerated marker-defined work scopes.
    # It never means all device events, untraced cores, host nodes or whole graph.
    complete=base['passed'] and markers['passed']
    return dict(manifest=str(path),raw_audit=base,markers=markers,marked_scope_complete=complete,scope='all expected K-block MAC intervals on traced cores only' if stop else 'four scale calls on one core',full_graph_complete=False,compute_only_calibration_eligible=False,notes=['Marker interval excludes input waits and output initialization; incomplete region boundary still prevents compute-only calibration.','Expected exact pair counts are the truncation test; trailing zeros alone are insufficient.','No host/device clock alignment or physical frequency assumption.'])


def selftest():
    events=[dict(ph='M',name='process_name',pid=0,args=dict(name='core_trace for tile2,0'))]
    for i in range(4):events.extend([dict(ph='B',name='INSTR_EVENT_0',pid=0,ts=i*10),dict(ph='B',name='INSTR_EVENT_1',pid=0,ts=i*10+5)])
    events.append(dict(ph='B',name='USER_EVENT_0',pid=0,ts=40))
    assert marker_check(events,4,1,True)['passed']
    assert not marker_check(events[:-1],4,1,True)['passed']
    assert not marker_check(events[:5]+events[6:],4,1,True)['passed']
    wrong=[dict(e) for e in events];wrong[-1]['ts']=1
    assert not marker_check(wrong,4,1,True)['passed']
    assert not marker_check(events,4,2,True)['passed']
    print('marker audit positive and four mutation tests passed')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--output',type=Path);p.add_argument('--selftest',action='store_true');a=p.parse_args()
    if a.selftest:selftest();return 0
    traces=[];graphs=[]
    for manifest in sorted(a.root.rglob('manifest.json')):
        try:traces.append(audit(manifest))
        except Exception as exc:traces.append(dict(manifest=str(manifest),marked_scope_complete=False,error=str(exc)))
    for raw in sorted(a.root.rglob('trace.uint32.npy')):
        if not (raw.parent/'manifest.json').exists():traces.append(dict(raw=str(raw),marked_scope_complete=False,error='raw capture lacks decoded manifest'))
    for path in sorted(a.root.rglob('result.json')):
        result=json.loads(path.read_text())
        if result.get('schema')=='orbit.trace.graph-experiment.v1':graphs.append(graph_trace_audit(path))
    report=dict(schema='orbit.trace.v4.audit',traces=traces,graphs=graphs,all_marked_scopes_complete=bool(traces) and all(t['marked_scope_complete'] for t in traces),all_graph_checks_passed=all(g['passed'] for g in graphs),hardware_executed_by_audit=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('traces','graphs')}))
    return 0 if report['all_marked_scopes_complete'] and report['all_graph_checks_passed'] else 1

if __name__=='__main__':raise SystemExit(main())
