#!/usr/bin/env python3
"""Finite source-bound contracts for public joint search (no hardware/timing campaign)."""
import copy,itertools,json,subprocess,sys,tempfile
from pathlib import Path
BIN=Path(sys.argv[1]).resolve()
OFF="--unavailable" in sys.argv[2:]

def source():
 task=''' %d{a}=taskflow.task @{a} will_reads(%X : memref<1x1xf32>) will_writes(%{out} : memref<1x1xf32>) [original_read_memrefs(%X : memref<1x1xf32>),original_write_memrefs(%{out} : memref<1x1xf32>)] : (memref<1x1xf32>,memref<1x1xf32>)->memref<1x1xf32> {{
 ^bb0(%x: memref<1x1xf32>,%o: memref<1x1xf32>):
 affine.for %i=0 to 1 {{affine.for %j=0 to 1 {{%v=affine.load %x[%i,%j] : memref<1x1xf32>
 affine.store %v,%o[%i,%j] : memref<1x1xf32>}}}}
 taskflow.yield done_writes(%o : memref<1x1xf32>)
 }}\n'''
 return 'module {func.func @shared(%X: memref<1x1xf32> {orbit.role="input",orbit.name="X"},%Y: memref<1x1xf32> {orbit.role="output",orbit.name="Y"},%Z: memref<1x1xf32> {orbit.role="output",orbit.name="Z"}) attributes {orbit.noalias,llvm.emit_c_interface} {\n'+task.format(a='a',out='Y')+task.format(a='b',out='Z')+'return\n}}\n'

def config(capacity,independent):
 resources=[dict(id='host.worker',kind='compute',device='host',capacity=1,domain='system'),dict(id='submit',kind='submission',device='host',capacity=1,domain='system')]
 domains=[dict(id='system',kind='system',parent='')]
 memories=[dict(id='host',kind='system-bo',domain='system',capacity_bytes=1000000,accessible_resources=['host.worker'])];links=[]
 for i in range(2):
  t=f'tile{i}';r=f'core{i}';dm=f'dma{i if independent else 0}'
  domains.append(dict(id=t,parent='system',kind='tile',coordinates=dict(column=i,row=2)))
  resources.append(dict(id=r,kind='compute',device='npu',capacity=1,domain=t))
  if independent or i==0:resources.append(dict(id=dm,kind='transfer',device='npu',capacity=1,domain='system'))
  memories.append(dict(id=f'mem{i}',kind='local-sram',domain=t,capacity_bytes=capacity,accessible_resources=[r]))
  for inp in [True,False]:
   links.append(dict(id=f'{"in" if inp else "out"}{i}',source='host' if inp else f'mem{i}',target=f'mem{i}' if inp else 'host',resource=dm,domain='system',bandwidth_bytes_per_ms=500000 if inp else 4000000,latency_ms=0,provenance=dict(kind='architecture',source='finite test assumption; no hardware claim')))
 params={k:dict(value=v,unit='ns',provenance=dict(kind='assumed',source='finite unit test')) for k,v in {'compute.startup':4,'compute.operation':0,'compute.memory_access':0,'configuration.startup':0,'host.submit':0,'host.complete':0}.items()}
 return dict(generic_air=dict(cost_model=dict(kind='event-mechanism',schema='orbit.direct-air.cost.v1',mode='estimate',model_id='finite-test',implementations=['scalar'],boundary='unit-test-device',parameters=params)),program_joint=dict(architecture=dict(domains=domains,resources=resources,memories=memories,links=links),host_memory='host',chunks=[1],max_configurations=2))

def oracle(capacity,independent):
 """Enumerate structures, physical placements and all unary resource orders.

 This never calls CP-SAT or reads pre-priced candidate costs. Source fixture
 has X(4B), A/B(4ns), Y/Z(4B); DMA durations are 8ns in and 1ns out.
 """
 best=None;count=0
 for groups in [[[0],[1]],[[0,1]]]:
  if any(4*(1+len(g))>capacity for g in groups):continue
  for places in itertools.permutations([0,1],len(groups)):
   ops={};edges=set();resources={}
   for i,(g,p) in enumerate(zip(groups,places)):
    inp=f'i{i}';comp=f'c{i}';ops[inp]=8;ops[comp]=4*len(g);edges.add((inp,comp))
    resources.setdefault(('dma',p if independent else 0),[]).append(inp)
    resources.setdefault(('compute',p),[]).append(comp)
    for t in g:
     out=f'o{t}';ops[out]=1;edges.add((comp,out));resources.setdefault(('dma',p if independent else 0),[]).append(out)
   for orders in itertools.product(*(itertools.permutations(v) for v in resources.values())):
    count+=1;ordered=set(edges)
    for order in orders:ordered.update(zip(order,order[1:]))
    end={};pending=set(ops)
    while pending:
     ready=[n for n in pending if all(a in end for a,b in ordered if b==n)]
     if not ready:break
     for n in ready:end[n]=ops[n]+max([end[a] for a,b in ordered if b==n]+[0]);pending.remove(n)
    if pending:continue
    value=max(end.values())
    if best is None or value<best:best=value
 return best,count

def main():
 checks=0
 with tempfile.TemporaryDirectory(prefix='orbit-joint-contract-') as temporary:
  root=Path(temporary);src=root/'source.mlir';src.write_text(source());cfg=root/'config.json'
  def run(label,c,extra=(),expect=0):
   cfg.write_text(json.dumps(c));out=root/label
   command=[str(BIN),str(src),'--target-config='+str(cfg),'--joint-program-search='+str(out),*extra]
   r=subprocess.run(command,text=True,capture_output=True,timeout=20)
   assert (r.returncode==0)==(expect==0),(command,r.returncode,r.stderr)
   return out,json.loads((out/'solution.json').read_text()) if (out/'solution.json').exists() else None
  cases=[]
  for cap,ind in [(8,False),(12,False),(12,True)]:
   c=config(cap,ind);out,s=run(f'c{cap}-{ind}',c);exact,orders=oracle(cap,ind)
   assert exact=={(8,False):21,(12,False):18,(12,True):13}[cap,ind]
   assert s['has_plan'] and s['status']==('UNAVAILABLE' if OFF else 'OPTIMAL') and s['selection']==('baseline' if OFF else 'joint') and (s['makespan_ns']>=exact if OFF else s['makespan_ns']==exact),s
   domain=json.loads((out/'domain.json').read_text());assert all(v['source_graph']['source']['body_fingerprint'] for v in domain['variants'])
   cfg.write_text(json.dumps(c));r=subprocess.run([str(BIN),str(src),'--target-config='+str(cfg),'--joint-program-plan='+str(out/'solution.json')],capture_output=True,text=True,timeout=20);assert r.returncode==0,r.stderr
   checks+=3;cases.append(dict(capacity=cap,independent_dma=ind,optimum=exact,resource_orders=orders,solve_ms=s['solve_ms']))
  c=config(12,True)
  for label,extra,status in [('limit',['--joint-max-variables=10'],'RESOURCE_LIMIT'),('deadline',['--joint-seconds=0.000001'],'UNKNOWN')]:
   out,s=run(label,c,extra);assert s['status']==('UNAVAILABLE' if OFF else status) and s['has_plan'] and s['selection']=='baseline',s;checks+=1
  # Frozen choices must be checked without re-solving. Change each independent
  # invariant, keeping domain identity fixed, and require rejection.
  good=json.loads((root/'c8-False/solution.json').read_text()) if OFF else json.loads((root/'c12-True/solution.json').read_text());cfg.write_text(json.dumps(config(8,False) if OFF else c))
  mutations=[]
  bad=copy.deepcopy(good);bad['groups'][0]['inputs'][0]['version']=1000;mutations.append(bad)
  bad=copy.deepcopy(good);bad['groups'][0]['inputs'][0]['end_ns']+=1;mutations.append(bad)
  bad=copy.deepcopy(good);bad['groups'].pop();mutations.append(bad)
  bad=copy.deepcopy(good);bad['groups'].append(copy.deepcopy(bad['groups'][0]));mutations.append(bad)
  bad=copy.deepcopy(good);bad['makespan_ns']-=1;mutations.append(bad)
  bad=copy.deepcopy(good);bad['domain_id']='wrong';mutations.append(bad)
  bad=copy.deepcopy(good);bad['program_graph']['schedule']={'forged':True};mutations.append(bad)
  for i,bad in enumerate(mutations):
   record=root/f'bad{i}.json';record.write_text(json.dumps(bad));r=subprocess.run([str(BIN),str(src),'--target-config='+str(cfg),'--joint-program-plan='+str(record)],capture_output=True,text=True,timeout=20);assert r.returncode!=0,(i,r.stdout);checks+=1
  # Hierarchical DMA contention must reach the same oracle optimum even when
  # individual link resources have distinct IDs.
  hierarchy=config(12,True);arch=hierarchy['program_joint']['architecture'];arch['resources'].append(dict(id='bus',kind='shared',device='npu',capacity=1,domain='system'))
  for resource in arch['resources']:
   if resource['kind']=='transfer':resource['shared_resources']={'bus':1}
  out,hs=run('shared-bus',hierarchy);assert hs['has_plan'] and hs['makespan_ns']==18,hs;checks+=1
  # Static memory is occupied for the whole configuration, including groups
  # on distinct abstract compute resources that share a memory instance.
  memory=config(8,True);arch=memory['program_joint']['architecture'];arch['memories']=[m for m in arch['memories'] if m['id']!='mem1'];arch['memories'][1]['accessible_resources']=['core0','core1'];arch['links']=arch['links'][:2]
  out,ms=run('shared-memory',memory);assert ms['has_plan'] and ms['makespan_ns']==26 and len(ms['configuration_start_ns'])==2,ms;checks+=1
  # Forge an apparently faster shared-DMA schedule. Independent integer
  # calendars must reject it before comparison with the stored graph.
  bad=json.loads((root/'c8-False/solution.json').read_text());bad['makespan_ns']=13;bad['configuration_end_ns']=[13];bad['configuration_start_ns']=[0]
  for i,group in enumerate(bad['groups']):
   group['configuration']=0;group['place']=i
   group['compute_start_ns']=[8]
   for transfer in group['inputs']:transfer.update(start_ns=0,end_ns=8,link=2*i)
   for transfer in group['outputs']:transfer.update(start_ns=12,end_ns=13,link=2*i+1)
  cfg.write_text(json.dumps(config(8,False)));record=root/'overlap.json';record.write_text(json.dumps(bad));r=subprocess.run([str(BIN),str(src),'--target-config='+str(cfg),'--joint-program-plan='+str(record)],capture_output=True,text=True,timeout=20);assert r.returncode!=0 and 'exact integer resource' in r.stderr,r.stderr;checks+=1
  native=copy.deepcopy(c);native['native_auto_candidates']=dict(mode='joint',algorithm='cp-sat-program');cfg.write_text(json.dumps(native));out=root/'native-entry';r=subprocess.run([str(BIN),str(src),'--target-config='+str(cfg),'--search','--backend=air','--search-output='+str(out)],capture_output=True,text=True,timeout=20);assert r.returncode==0,r.stderr;ns=json.loads((out/'solution.json').read_text());assert ns['has_plan'] and ns['status']==('UNAVAILABLE' if OFF else 'OPTIMAL');checks+=1
  # A real dependency forces distinct configurations when the 12-byte merged
  # payload does not fit. Y remains an observable intermediate source output.
  text=source();a=text.index('%db=taskflow.task');text=text[:a]+text[a:].replace('%X : memref','%da : memref');src.write_text(text)
  out,s=run('raw',config(8,True));assert s['has_plan'] and len(s['configuration_start_ns'])==2,s;checks+=1
  if len(sys.argv)>2 and not sys.argv[2].startswith("--"):
   src.write_text(source());cfg.write_text(json.dumps(c));out=root/'off';r=subprocess.run([sys.argv[2],str(src),'--target-config='+str(cfg),'--joint-program-search='+str(out)],capture_output=True,text=True,timeout=20);assert r.returncode==0,r.stderr;s=json.loads((out/'solution.json').read_text());assert s['status']=='UNAVAILABLE' and s['selection']=='baseline' and s['has_plan'];checks+=1
 print(json.dumps(dict(passed=True,checks=checks,oracle_cases=cases),indent=2))
if __name__=='__main__':main()
