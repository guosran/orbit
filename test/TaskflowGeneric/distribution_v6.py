#!/usr/bin/env python3
"""The distribution proof depends on accesses, never reduction arithmetic."""
import argparse
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser()
p.add_argument('--tool',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parents[2]
original=(root/'test/TaskflowAIR/fixtures/gemm.mlir').read_text()
cases={
    'gemm':(original,2),
    'different_arithmetic':(original.replace('arith.mulf','arith.subf'),2),
    'coupled_output':(original.replace('affine.load %C_b[%i, %j]',
                                      'affine.load %C_b[%i, 0]'),0),
}
for name,(source,count) in cases.items():
    src=a.output/(name+'.mlir');src.write_text(source)
    transformed=a.output/(name+'.transformed.mlir')
    facts=a.output/(name+'.json')
    command=[str(a.tool.resolve()),str(src),'--generic-analyze',
        '--air-distribute-initialization','--emit-mlir='+str(transformed),
        '--analysis-output='+str(facts)]
    result=subprocess.run(command,text=True,capture_output=True)
    (a.output/(name+'.log')).write_text(result.stdout+result.stderr)
    if count==0:
        assert result.returncode!=0 and 'uninitialized' in result.stderr
        continue
    assert result.returncode==0,result.stderr
    tasks=json.loads(facts.read_text())['tasks']
    assert len(tasks)==count,(name,len(tasks))
    if count==2:
        assert tasks[1]['dependencies']==[tasks[0]['id']],tasks[1]
(a.output/'summary.json').write_text(json.dumps({'passed':True,'cases':list(cases)},indent=2))
