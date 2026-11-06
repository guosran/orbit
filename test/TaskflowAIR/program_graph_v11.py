#!/usr/bin/env python3
"""Original native graph export/search/resume regression; CPU/compiler only."""
import argparse,copy,hashlib,json,subprocess,time
from pathlib import Path
from datetime import datetime,timezone


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tool',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--checker',type=Path);p.add_argument('--source',type=Path,help='Optional plain source fixture; default grows the retained Core-only fixture');p.add_argument('--air',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[2];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);tool=a.tool.resolve();checks=[]
    def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    def run(name,cmd,ok=True):
        cmd=list(map(str,cmd));save(out/(name+'.command.json'),cmd);started=datetime.now(timezone.utc).isoformat();begin=time.perf_counter();r=subprocess.run(cmd,capture_output=True,text=True)
        (out/(name+'.stdout')).write_text(r.stdout);(out/(name+'.stderr')).write_text(r.stderr);save(out/(name+'.result.json'),dict(returncode=r.returncode,elapsed_wall_seconds=time.perf_counter()-begin,utc_start=started,utc_end=datetime.now(timezone.utc).isoformat(),hardware_executed=False))
        if (r.returncode==0)!=ok:raise AssertionError((name,r.returncode,r.stderr[-3000:]))
        checks.append(name);return r
    checker=(a.checker or tool.parent/'taskflow-program-graph-test').resolve()
    run('common-checker',[checker,root/'test/TaskflowAIR/fixtures/gemm-scale.mlir'])
    source=(root/'test/TaskflowAIR/fixtures/gemm-scale.mlir').read_text();shapes={'4x8':'32x128','8x4':'128x64','4x4':'32x64'}
    import re
    source=re.sub(r'(?<=memref<)\d+x\d+(?=xf32)',lambda m:shapes[m[0]],source)
    source=re.sub(r'affine.for %(i|j|k) = 0 to \d+',lambda m:f'affine.for %{m[1]} = 0 to '+str(dict(i=32,j=64,k=128)[m[1]]),source)
    path=out/'source.mlir'
    if a.source:path.write_bytes(a.source.resolve().read_bytes())
    else:path.write_text(source)
    save(out/'input-source.json',dict(origin=str(a.source.resolve()) if a.source else 'Core gemm-scale fixture grown to M32/K128/N64',snapshot=str(path),sha256=sha(path)))
    for name,flags in [('phase',[]),('retain',['--data-organization=retain-across-users','--resident-storage=1','--reuse-loop=0','--panel-loop=1'])]:
        directory=out/name;directory.mkdir();graph=directory/'program-graph.json';dot=directory/'program-graph.dot';plan=directory/'plan.json'
        command=[tool,path,'--generic-map','--air-distribute-initialization','--tile-m=8','--tile-n=16','--tile-k=32',*flags,'--mapping-output='+str(plan),'--emit-mlir='+str(directory/'transformed.mlir'),'--program-graph-output='+str(graph),'--program-graph-dot='+str(dot)]
        if a.air:command+=['--lower-to-air','--air-output='+str(directory/'program.air.mlir')]
        run(name,command);native=json.loads(plan.read_text());g=json.loads(graph.read_text())
        assert native['program_graph']==g and native['program_graph_id'] and 'digraph' in dot.read_text()
        assert native['body_fingerprint'] in json.dumps(g)
        if a.air:
            run(name+'-air-check',[tool,path,'--air-distribute-initialization','--tile-m=8','--tile-n=16','--tile-k=32',*flags,'--check-air='+str(directory/'program.air.mlir')])
    phase=json.loads((out/'phase/plan.json').read_text());retain=json.loads((out/'retain/plan.json').read_text())
    assert phase['body_fingerprint']==retain['body_fingerprint'] and phase['program_graph_id']!=retain['program_graph_id']
    assert phase['actual_dma_bytes']>retain['actual_dma_bytes'] and phase['resident_bytes']<retain['resident_bytes']
    config={'model_version':'v11-graph-unknown-CPU-regression','target_version':'v11-graph-CPU-regression','generic_air':{'capacity_bytes':24576,'cost_model':{'schema':'orbit.direct-air.cost.v1','kind':'features-only'}},'native_auto_candidates':{'mode':'joint','max_tile_proposals':6,'exploration_slots':4},'search_preferences':{'m_alignment':4,'n_alignment':8,'k_alignment':16}}
    cfg=out/'search-config.json';save(cfg,config);search=out/'search'
    command=[tool,path,'--search','--backend=air','--air-distribute-initialization','--tile-m=8','--tile-n=16','--tile-k=32','--target-config='+str(cfg),'--search-output='+str(search),'--max-rounds=2','--max-candidates=64','--beam-width=4']
    run('search-two-rounds',command);before=json.loads((search/'search.json').read_text());artifacts=[]
    for row in before['archive']:
        evaluation=row.get('evaluation',{});cid=row['candidate_id'];directory=search/'candidates'/cid
        if evaluation.get('status') not in ['passed','uncalibrated']:continue
        graph=directory/'program-graph.json';dot=directory/'program-graph.dot';materialized=directory/'taskflow.mlir';mapping=json.loads((directory/'mapping-result.json').read_text());g=json.loads(graph.read_text())
        assert sha(materialized)==cid and mapping['program_graph']==g
        assert evaluation['program_graph_id']==mapping['program_graph_id']
        assert mapping['best_plan']['program_graph_id']==mapping['program_graph_id'] and mapping['best_plan']['program_graph']==g
        assert 'digraph' in dot.read_text();artifacts.append((graph,dot,graph.read_bytes(),dot.read_bytes()))
    assert artifacts and any(row['round']>=2 for row in before['archive'])
    combinations=[row for row in before['archive'] if row.get('evaluation',{}).get('status') in ['passed','uncalibrated'] and any(action.startswith('auto_tiles_') for action in row['action_chain']) and any(action.startswith('mapping_retain-across-users') for action in row['action_chain'])]
    assert combinations,'native two-round archive lacks a legal materialized tile+retention combination'
    combined=next((row for row in combinations if any('storage_1_reuse_0_panel_1' in action for action in row['action_chain'])),combinations[0])
    archive={row['candidate_id']:row for row in before['archive']};parents=[];current=combined
    while True:
        parents.append({key:current[key] for key in ['candidate_id','parent','action_chain','round','status']})
        if not current['parent']:break
        current=archive[current['parent']]
    save(out/'combined-candidate-parent-chain.json',dict(candidate=combined['candidate_id'],program_graph_id=combined['evaluation']['program_graph_id'],parents=parents,scope='Existing legal native search lineage sampled only for validation; not Python production candidate selection'))

    run('resume-exact',[*command,'--resume']);after=json.loads((search/'search.json').read_text());assert before['ranking']==after['ranking']
    graph,dot,graph_bytes,dot_bytes=artifacts[0]
    for name,file,original,mutated in [('graph-json',graph,graph_bytes,b'{}\n'),('graph-dot',dot,dot_bytes,dot_bytes+b'//tampered\n')]:
        file.write_bytes(mutated)
        try:run('resume-'+name+'-tamper-reject',[*command,'--resume'],False)
        finally:file.write_bytes(original)
    run('resume-restored',[*command,'--resume'])
    save(out/'summary.json',dict(status='passed',checks=checks,graph_artifacts=len(artifacts),search_rounds=2,legal_combined_candidates=len(combinations),combined_candidate=combined['candidate_id'],combined_graph_id=combined['evaluation']['program_graph_id'],parent_chain_file='combined-candidate-parent-chain.json',hardware_executed=False,tool=dict(path=str(tool),sha256=sha(tool)),test_source_sha256=sha(Path(__file__)),scope='native C++ graph export/legality/automatic two-round search/resume; no new numerical or hardware acceptance'))
    print(json.dumps(dict(status='passed',checks=len(checks),hardware_executed=False)))


if __name__=='__main__':main()
