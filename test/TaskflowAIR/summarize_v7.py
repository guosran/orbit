#!/usr/bin/env python3
"""Inventory raw v7 measurements without changing acceptance thresholds."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
records=[]
for out in sorted(p for p in a.root.iterdir() if p.is_dir() and ((p/'hardware').exists() or (p/'lower.log').exists() or (p/'measurement-mode.json').exists())):
    hardware=out if (out/'manifest.json').exists() else out/'hardware';entry=dict(attempt=out.name,scope='see measurement-mode.json; diagnostic attempts are not multi-seed acceptance',results=[])
    for name in ['measurement-mode.json','compile.json']:
        if (hardware/name).exists():entry[name.removesuffix('.json')]=json.loads((hardware/name).read_text())
    for result in sorted(hardware.glob('seed-*/result.json')):
        r=json.loads(result.read_text());entry['results'].append(dict(status=r['status'],wall_ms=r['median_ms'],checks=r['samples'][0]['checks']))
    for failure in sorted(hardware.glob('seed-*/failure.json')):entry['results'].append(json.loads(failure.read_text()))
    entry['identity']={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [out/'source.mlir',out/'plan.json',hardware/'input.air.mlir',hardware/'program.elf',out/'invocation.json',out/'manifest.json'] if f.exists()}
    records.append(entry)
(a.root/'diagnostic-summary.json').write_text(json.dumps(records,indent=2)+'\n')
(a.root/'evidence-manifest.json').write_text(json.dumps({str(f.relative_to(a.root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.root.rglob('*')) if f.is_file() and f.name!='evidence-manifest.json'},indent=2)+'\n')
