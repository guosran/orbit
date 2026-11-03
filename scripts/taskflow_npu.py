#!/usr/bin/env python3
"""Execute a materialized Taskflow plan on npu2, with explicit host placements."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from orbit_npu.runtime import run_plan

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True)
    p.add_argument('--inputs',required=True)
    p.add_argument('--reference',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--target-config',help='JSON file: orbit.npu.config.v1')
    p.add_argument('--iterations',type=int,default=3)
    p.add_argument('--atol',type=float,default=0.015)
    p.add_argument('--rtol',type=float,default=0.06)
    a=p.parse_args()
    r=run_plan(json.loads(Path(a.plan).read_text()),dict(np.load(a.inputs)),dict(np.load(a.reference)),
        a.output_dir,a.iterations,a.atol,a.rtol,
        target_config=json.loads(Path(a.target_config).read_text()) if a.target_config else None)
    print(json.dumps(r,indent=2),flush=True)
    return 0 if r['status']=='passed' else 1
if __name__=='__main__':
    raise SystemExit(main())
