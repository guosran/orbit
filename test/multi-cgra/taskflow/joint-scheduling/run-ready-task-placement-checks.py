#!/usr/bin/env python3
"""Execute real orchestration regressions, without invoking a task mapper."""
import argparse,json,re,subprocess
from pathlib import Path

def schedules(text):
    pattern=r'\{cgra_positions = \[(.*?)\], end_cycle = (\d+) : i64, start_cycle = (\d+) : i64, task = "([^"]+)"\}'
    rows=[]
    for cells,end,start,name in re.findall(pattern,text):
        rows.append({'task':name,'start':int(start),'end':int(end),'cells':set((int(r),int(c)) for c,r in re.findall(r'col = (\d+) : i32, row = (\d+) : i32',cells))})
    assert len(rows)==2,rows
    for row in rows:assert row['end']>row['start'] and row['cells']
    a,b=rows
    assert not(a['start']<b['end'] and b['start']<a['end'] and a['cells']&b['cells'])
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--optimizer',required=True);p.add_argument('--fixture',type=Path,required=True);p.add_argument('--architecture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    source=args.fixture.read_text();arch=args.architecture.read_text();results=[]
    def run(name,source,grid=(4,4)):
        src=args.output/(name+'.mlir');out=args.output/(name+'.out.mlir');config=args.output/(name+'.yaml');src.write_text(source)
        text=re.sub(r'(multi_cgra_defaults:.*?\n  rows:) \d+',rf'\g<1> {grid[0]}',arch,flags=re.S)
        text=re.sub(r'(multi_cgra_defaults:.*?\n  columns:) \d+',rf'\g<1> {grid[1]}',text,flags=re.S);config.write_text(text)
        argv=[args.optimizer,str(src),f'--architecture-spec={config}','--orchestrate-tasks-on-accelerators=orchestration-strategy=analytical-based-task-orchestration scheduling-mode=spatial-temporal dispatch-policy=fixed communication-mode=none','--mlir-print-op-generic','-o',str(out)]
        r=subprocess.run(argv,text=True,capture_output=True,timeout=30);(args.output/(name+'.stderr')).write_text(r.stderr);assert r.returncode==0,r.stderr
        (args.output/(name+'.command.json')).write_text(json.dumps(argv,indent=2)+'\n');return schedules(out.read_text())
    a,b=run('independent-two-2x2',source)
    assert a['start']==b['start']==0 and a['end']==b['end']==10 and not(a['cells']&b['cells'])
    results.append('two independent 2x2 rectangles overlap in time on4x4')
    again=run('deterministic-repeat',source);assert [a,b]==again
    a,b=run('insufficient-2x2-grid',source,(2,2));assert a['end']==b['start']==10 and b['end']==20 and a['cells']==b['cells']
    results.append('insufficient resources wait; disjoint half-open time intervals reuse identical cells')
    before,second=source.split('  %t1 =',1)
    dependent=before+'  %t1 ='+second.replace('will_reads(%a,','will_reads(%t0,',1).replace('original_read_memrefs(%a,','original_read_memrefs(%x,',1)
    a,b=run('dependency-ready',dependent);assert b['start']>=a['end']
    results.append('SSA dependency completion precedes consumer start')
    a,b=run('dependency-reuse',dependent,(2,2));assert a['cells']==b['cells'] and b['start']==a['end']
    results.append('dependent tasks reuse same static rectangle after producer completes')
    # An asymmetric footprint catches rows/columns transposition in placement.
    asymmetric=source.replace('cgra_count=4:i32,cgra_shape="2x2"','cgra_count=2:i32,cgra_shape="1x2"')
    for row in run('oriented-1x2',asymmetric):
        rr={r for r,c in row['cells']};cc={c for r,c in row['cells']};assert len(rr)==1 and len(cc)==2
        r0,c0=min(rr),min(cc)
        # NEURA shape-local x/y are PE columns/rows:1x2CGRAs ->2x4PEs.
        global_pes={(2*r0+y,2*c0+x) for y in range(2) for x in range(4)}
        assert len(global_pes)==8 and {(y//2,x//2) for y,x in global_pes}==row['cells']
        assert all(0<=y<8 and 0<=x<8 for y,x in global_pes)
    results.append('oriented1x2 footprint and localPE-to-globalCGRA units agree')
    (args.output/'checks.json').write_text(json.dumps({'status':'pass','checks':results,'deterministic':True},indent=2)+'\n');print(json.dumps(results,indent=2))
if __name__=='__main__':main()
