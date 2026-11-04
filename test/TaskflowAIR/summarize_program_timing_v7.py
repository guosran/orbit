#!/usr/bin/env python3
"""Aggregate observed host API spans without treating them as hardware cycles."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.directory
manifest=json.loads((root/'manifest.json').read_text());rows=[]
for seed in sorted(root.glob('seed-*')):
 if not (seed/'result.json').exists():continue
 result=json.loads((seed/'result.json').read_text())
 for sample in result['samples']:
  groups={kind:[] for kind in ['actual_bo_sync','actual_xrt_start','actual_xrt_wait']}
  for invocation in sample['invocations']:
   for event in invocation['api_events']:
    if event['kind'] in groups:groups[event['kind']].append(event)
  spans={kind:sum((e['end_ns']-e['start_ns'])/1e6 for e in events) for kind,events in groups.items()}
  row=dict(seed=seed.name,repeat=sample['repeat'],warmup=sample['warmup'],status=result['status'],context_load_ms=result['load_ms'],complete_program_wall_ms=sample['wall_ms'],observed_api_ms=spans,api_counts={kind:len(events) for kind,events in groups.items()},other_host_work_ms=sample['wall_ms']-sum(spans.values()),per_command_wall_ms=[v['wall_ms'] for v in sample['invocations']])
  row['bo_sync_to_device_ms']=sum((e['end_ns']-e['start_ns'])/1e6 for e in groups['actual_bo_sync'] if 'TO_DEVICE' in e['direction'])
  row['bo_sync_from_device_ms']=sum((e['end_ns']-e['start_ns'])/1e6 for e in groups['actual_bo_sync'] if 'FROM_DEVICE' in e['direction'])
  rows.append(row)
compile_records={t['id']:json.loads((root/t['id']/'compile.json').read_text()) for t in manifest['templates']}
a.output.write_text(json.dumps(dict(status='completed-seeds-only summary',compile={key:dict(aircc_subprocess_seconds=r['seconds'],cache_state='unknown; cache hits/misses not instrumented',source_sha256=r['source_sha256'],elf_sha256=r['elf_sha256']) for key,r in compile_records.items()},samples=rows,interpretation=dict(wait='Observed host span includes wait checkpoint serialization, XRT runtime scheduling/waiting/polling; not pure kernel time or calibrated hardware clock cycles.',sync='Observed XRT BO synchronization API spans; not independently measured DDR traffic.',other_host_work='Full wall less disjoint observed API spans: allocations/staging/slice copies/argument preparation/readback copies/output assembly/evidence IO and other host overhead; not a pure transfer measure.',load='Per-seed template context creation/loading, excluded from complete program wall.',wall='Includes all compiler-issued commands, host slice/staging/sync/copy/assembly and per-command evidence IO; excludes context loading, full input/reference loading, final numerical checking/full-output serialization.')),indent=2)+'\n')
