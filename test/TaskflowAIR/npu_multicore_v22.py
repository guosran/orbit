#!/usr/bin/env python3
"""Check a real compiled multi-core AIE and reject targeted corruptions. No NPU."""
import argparse
import json
from pathlib import Path
import re
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('tool','source','config','aie','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--shards',type=int,default=2)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    text=a.aie.read_text();rows=[]
    def run(name,changed,valid=False):
        path=a.output/(name+'.mlir');path.write_text(changed)
        receipt=a.output/(name+'.json')
        command=[str(a.tool.resolve()),str(a.source.resolve()),'--check-multicore-aie='+str(path.resolve()),
            '--npu-shards='+str(a.shards),'--target-config='+str(a.config.resolve()),'--implementation-output='+str(receipt.resolve())]
        r=subprocess.run(command,capture_output=True,text=True,timeout=20)
        (a.output/(name+'.stderr')).write_text(r.stderr)
        assert r.returncode>=0,(name,'checker crashed',r.stderr)
        assert (r.returncode==0)==valid,(name,r.returncode,r.stderr)
        assert receipt.exists(),(name,'must reach semantic checker, not merely fail parsing')
        assert json.loads(receipt.read_text())['valid']==valid,(name,receipt)
        rows.append(dict(name=name,valid=valid,returncode=r.returncode,command=command))
    run('actual',text,True)
    def replace(name,old,new):
        assert old in text,(name,old)
        run(name,text.replace(old,new,1))
    replace('wrong-device','device = 8 : i32','device = 4 : i32')
    replace('wrong-arithmetic','value = 2.000000e+00 : f32','value = 3.000000e+00 : f32')
    replace('output-hole','static_offset = 0 : i32','static_offset = 1 : i32')
    match=re.search(r'static_len = (\d+) : i32',text);assert match
    replace('wrong-bd-length',match[0],f'static_len = {int(match[1])-1} : i32')
    match=re.search(r'col = [01] : i32, row = 2 : i32',text);assert match
    replace('wrong-placement',match[0],'col = 3 : i32, row = 2 : i32')
    match=re.search(r'^.*"aiex.dma_start_task".*\n',text,re.M);assert match
    replace('missing-start',match[0],'')
    match=re.search(r'^.*"aiex.dma_await_task".*\n',text,re.M);assert match
    replace('duplicate-await',match[0],match[0]*2)
    match=re.search(r'^.*"aiex.dma_free_task".*\n',text,re.M);assert match
    changed=text.replace(match[0],'',1)
    start=re.search(r'^.*"aiex.dma_start_task".*\n',changed,re.M);assert start
    run('premature-free',changed[:start.start()]+match[0]+changed[start.start():])
    replace('unknown-wrapper','      "aiex.configure"','      "orbit.unknown"() : () -> ()\n      "aiex.configure"')
    replace('unknown-core','      "cf.br"','      "orbit.unknown"() : () -> ()\n      "cf.br"')
    match=re.search(r'alloc = @\w+',text);assert match
    replace('unknown-allocation',match[0],'alloc = @missing_allocation')
    replace('missing-runtime-name','sym_name = "orbit_segment_sequence"','orbit.unknown = true')
    replace('symbol-only-external-use','}) {orbit.cost_status','}) {orbit.external_use = @__air_external_buffer, orbit.cost_status')
    # A valid SSA flow can still illegally join resources belonging to two shards.
    tile_owners={}
    for v,col,row in re.findall(r'(%\d+) = "aie.tile"\(\) <\{col = (\d+) : i32, row = (\d+) : i32',text):
        tile_owners[v]=(int(col),int(row))
    flow=re.search(r'"aie.flow"\((%\d+), (%\d+)\)',text);assert flow
    other=next(v for v,(col,row) in tile_owners.items() if row==2 and col!=tile_owners[flow[1]][0])
    replace('cross-owner-flow',flow[0],f'"aie.flow"({flow[1]}, {other})')
    (a.output/'result.json').write_text(json.dumps(dict(status='PASS',cases=rows),indent=2)+'\n')
    print(json.dumps(dict(status='PASS',cases=len(rows))))


if __name__=='__main__':main()
