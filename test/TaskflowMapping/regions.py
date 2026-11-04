#!/usr/bin/env python3
"""Independent exact-cover oracle and re-signed opaque-plan adversarial checks."""
import argparse,copy,json,subprocess,sys
from pathlib import Path
import sys
from pathlib import Path as _TestPath
sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1] / 'Support'))
from orbit_test_paths import taskflow_tool

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'test/Support'),str(Path(__file__).parent)]
from structured_fixtures import chain_source as source
from cpp_differential import native_fingerprint

def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--tool',type=Path,default=taskflow_tool());args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
 tool=args.tool.resolve();program=source(4,4,8,4);save(out/'source.json',program)
 subprocess.run([str(tool),str(out/'source.json'),'--emit-mlir='+str(out/'source.mlir'),'--emit-plan='+str(out/'program.json')],check=True)
 program=json.loads((out/'program.json').read_text())
 def impl(name,pattern,mode,cost,axis='h'):
  item=dict(id=name,version='oracle-1',pattern=pattern,mode=mode,feature_axis=axis,device='npu',boundary_memory='host',layout='row_major',dtype='f32',numerical_contract='bf16-bfp16-bf16-mid-final',prepared_state='warm_contexts_weights',resources=dict(cpu=1,npu=1,dma=1),workspace_bytes=dict(host=128,shared=256),legality=dict(multiples=dict(m=1,h=1),power_of_two=['k']),cost=dict(kind='bilinear_m_h',scope='opaque_warm_host_to_host',fixed_ms=cost,per_m_ms=0,per_h_ms=0,per_mh_ms=0,model_id='synthetic-oracle',training_ids=['train-a','train-b'],excluded_ids=['heldout'],shape_range=dict(m=[2,8],h=[2,8],k=[4,8],n=[1,8]),provenance=dict(kind='synthetic',source='independent exact-cover fixture; not measured')))
  if pattern=='matmul_pair' or axis=='n':item['cost']['shape_range']['k']=[4,4]
  if pattern=='matmul_pair' or axis=='k':item['cost']['shape_range']['n']=[4,4]
  if pattern=='matmul_single':item['cost']['shape_range']['h']=list(item['cost']['shape_range'][axis])
  return item
 config=dict(schema='orbit.static.config.v1',hardware=dict(resources=[dict(id=x,kind='compute' if x!='dma' else 'communication',device=x,capacity=1) for x in ('cpu','npu','dma')],memories=[dict(id='host',capacity_bytes=2**20,accessible_devices=['cpu']),dict(id='shared',capacity_bytes=2**20,accessible_devices=['npu'])],links=[],conversions=[],synchronizations=[]),implementations=[],region_implementations=[impl('pair-chain','matmul_pair','chain',2),impl('pair-independent','matmul_pair','independent',3),impl('single-first','matmul_single','independent',2,'n'),impl('single-second','matmul_single','independent',2,'k')],assumptions=dict(execution_state='warm_contexts_weights',executions=1,overlap='forbid',max_assignments=100,beam_width=8,program_source_id=program['source_id'],allowed_numerical_contracts=['bf16-bfp16-bf16-mid-final'],initial_locations={b['id']:'host' for b in program['buffers'] if b['role']=='input'},output_locations={'Y':'host'},output_devices={'Y':'cpu'}))
 def mapping(label,cfg,mlir=None):
  d=out/label;d.mkdir();save(d/'config.json',cfg)
  cmd=[str(tool),str(mlir or out/'source.mlir'),'--static-map','--target-config='+str(d/'config.json'),'--mapping-output='+str(d/'result.json')]
  r=subprocess.run(cmd,capture_output=True,text=True);(d/'log').write_text(r.stdout+r.stderr)
  if not (d/'result.json').exists():return dict(status='unsupported',errors=r.stderr)
  return json.loads((d/'result.json').read_text())
 result=mapping('base',config);assert result['status']=='passed',result
 # Enumerate legal covers independently for the four known source operations:
 # a single pair choice, or exactly two standalone implementations.
 expected=[2.,3.,4.];observed=sorted(x['score'] for x in result['alternatives'] if x['status']=='passed')
 assert observed==expected,(observed,expected)
 best=result['best_plan'];assert best['totals']['makespan_ms']==2 and len(best['events'])==1
 changed=copy.deepcopy(config);changed['region_implementations'][0]['cost']['fixed_ms']=5
 reverse=mapping('cost-reversal',changed);assert reverse['best_plan']['totals']['makespan_ms']==3
 assert reverse['best_plan']['events'][0]['region_mode']=='independent'
 checks=['independent exact-cover oracle {chain, independent pair, two singles}', 'changed cost reverses selected implementation']
 def negative(label,mutate):
  p=copy.deepcopy(best);mutate(p);p['plan_id']=native_fingerprint({k:v for k,v in p.items() if k not in ('plan_id','validation')})
  d=out/'negative'/label;save(d/'plan.json',p)
  r=subprocess.run([str(tool),str(out/'source.mlir'),'--target-config='+str(out/'base/config.json'),'--check-mapping='+str(d/'plan.json')],capture_output=True,text=True)
  (d/'check.log').write_text(r.stdout+r.stderr)
  assert r.returncode!=0,label
  if r.stdout:
   errors=json.loads(r.stdout).get('errors',[])
   assert any('identity' not in e for e in errors),(label,errors)
  checks.append('re-signed reject '+label)
 def event(p):return p['events'][0]
 negative('overlap',lambda p:event(p)['covered_steps'].append(event(p)['covered_steps'][0]))
 negative('missing-operation',lambda p:event(p)['covered_steps'].pop())
 negative('wrong-input-version',lambda p:event(p)['reads'][0].__setitem__('version','wrong'))
 negative('wrong-input-location',lambda p:next(a for a in p['allocations'] if a['id']==event(p)['reads'][0]['allocation']).__setitem__('memory','shared'))
 negative('wrong-layout',lambda p:next(a for a in p['allocations'] if a['buffer']=='X').__setitem__('layout','column_major'))
 negative('plan-cost-boundary',lambda p:p.__setitem__('cost_boundary','compute_only'))
 negative('cold-state',lambda p:p.__setitem__('initial_state','cold_inputs_only'))
 negative('implementation-state',lambda p:event(p).__setitem__('prepared_state','cold'))
 negative('model-identity',lambda p:event(p).__setitem__('model_id','other'))
 negative('forged-shape',lambda p:event(p)['region_shape'].__setitem__('h',7))
 negative('workspace-omitted',lambda p:p.__setitem__('allocations',[a for a in p['allocations'] if a['lifetime']!='event']))
 negative('resource-omitted',lambda p:event(p).__setitem__('resource_demands',{'npu':1}))
 negative('fake-compute-cost',lambda p:p['totals'].__setitem__('compute_ms',p['totals']['region_wall_ms']))
 negative('extra-copy-billing',lambda p:event(p).__setitem__('kind','copy'))
 negative('missing-output',lambda p:p.__setitem__('outputs',[]))
 negative('missing-final-dependency',lambda p:p.__setitem__('output_dependencies',[]))
 negative('output-consumer',lambda p:p['output_consumers'].__setitem__('Y','npu'))
 for label,alter in [('missing-cost',lambda c:c['region_implementations'][0].pop('cost')),('out-of-domain',lambda c:[i['cost']['shape_range'].__setitem__('m',[8,16]) for i in c['region_implementations']]),('no-numerical-permission',lambda c:c['assumptions'].__setitem__('allowed_numerical_contracts',['fp32'])),('cold-config',lambda c:c['assumptions'].__setitem__('execution_state','cold_inputs_only')),('capacity',lambda c:c['hardware']['memories'][0].__setitem__('capacity_bytes',1)),('leaked-split',lambda c:c['region_implementations'][0]['cost'].__setitem__('excluded_ids',['train-a']))]:
  cfg=copy.deepcopy(config);alter(cfg);r=mapping(label,cfg);assert r['status']!='passed',(label,r);checks.append(label+' unscored/rejected')
 # Exposing the middle result requires singles; packed and opaque pair cannot
 # erase the source graph's external consumer.
 exposed=copy.deepcopy(program);exposed['outputs'].append('H');next(b for b in exposed['buffers'] if b['id']=='H')['role']='output';save(out/'exposed.json',exposed)
 subprocess.run([str(tool),str(out/'exposed.json'),'--emit-mlir='+str(out/'exposed.mlir')],check=True)
 cfg=copy.deepcopy(config);cfg['assumptions']['output_locations']['H']='host';cfg['assumptions']['output_devices']['H']='cpu'
 r=mapping('external-intermediate',cfg,out/'exposed.mlir');assert r['status']=='passed' and len(r['best_plan']['events'])==2,r
 checks.append('external intermediate output retained through standalone exact cover')
 save(out/'checks.json',dict(passed=True,checks=checks,oracle='independent complete cover enumeration; native semantic checker with freshly signed malformed plans',hardware=False))
 print(json.dumps(dict(passed=True,checks=len(checks))))
if __name__=='__main__':main()
