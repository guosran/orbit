#!/usr/bin/env python3
"""Native plan validation and evidence report; no production decisions."""
import argparse,hashlib,json,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))

def load(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    rows=[]
    for case in ('gemm_relu','ffn','ffn_m5','ffn_m128'):
        w=out/'workloads'/case
        for variant in ('base','n','k','fused','slow-link'):
            cfg=w/('slow-link-config.json' if variant=='slow-link' else 'config.json')
            result=w/('slow-link-mapping.json' if variant=='slow-link' else variant+'/mapping.json')
            if not result.exists():continue
            program=load(w/('base' if variant=='slow-link' else variant)/'plan.json');config=load(cfg)
            native=load(result)
            assert native['status']=='passed',(case,variant)
            checked=0
            mlir=w/('base' if variant=='slow-link' else variant)/'taskflow.mlir'
            for number,alternative in enumerate(native['alternatives']):
                if alternative['status']!='passed':continue
                candidate=out/'validation'/f'{case}-{variant}-{number}.json'
                save(candidate,alternative['plan'])
                run=subprocess.run([str(ROOT/'build-core/tools/taskflow-opt/taskflow-opt'),str(mlir),'--target-config='+str(cfg),'--check-mapping='+str(candidate)],capture_output=True,text=True)
                validation=json.loads(run.stdout)
                assert run.returncode==0 and validation['passed'],(case,variant,validation)
                checked+=1
            rows.append(dict(case=case,variant=variant,native_ms=native['best_plan']['makespan_ms'],checked_plans=checked,assignment=native['best_plan']['assignment']))
    summary=dict(passed=True,hardware_executed=False,cost_kind='theoretical_sensitivity',workloads=rows)
    for name,file in [('differential','summary.json'),('native','checks.json'),('legacy-search','checks.json'),('legacy-cache','checks.json'),('preferences','checks.json')]:
        value=load(out/'validation'/name/file);assert value['passed'];summary[name]=value
    binary=ROOT/'build-core/tools/taskflow-opt/taskflow-opt'
    symbols=subprocess.run(['nm','-C',str(binary)],check=True,capture_output=True,text=True).stdout
    dependencies=subprocess.run(['ldd',str(binary)],check=True,capture_output=True,text=True).stdout
    commands=subprocess.run(['ninja','-C',str(ROOT/'build-core'),'-t','commands','taskflow-opt'],check=True,capture_output=True,text=True).stdout
    bad=re.findall(r'\b(?:Neura|MLIRAIR|XRT|libpython|pybind11)\w*',symbols+dependencies+commands)
    assert not bad,bad
    (out/'core-ldd.txt').write_text(dependencies);(out/'core-commands.txt').write_text(commands)
    summary['core_independence']=dict(passed=True,forbidden_symbol_or_link_matches=bad)
    sources=[*sorted((ROOT/'lib/Taskflow').glob('*.cpp')),*sorted((ROOT/'include/Taskflow').glob('*.h')),ROOT/'tools/taskflow-opt/taskflow-opt.cpp',binary,ROOT/'build-amoeba/tools/mlir-amoeba-opt/mlir-amoeba-opt']
    save(out/'source-identities.json',{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources})
    save(out/'summary.json',summary)
    lines=['# Native static planning validation', '',
           'Planning and checking run in C++/MLIR. Python only orchestrates tests and records artifacts.',
           'Costs are theoretical sensitivity parameters, not hardware measurements.', '',
           '| Workload | Variant | Estimated ms | Native checked plans |',
           '|---|---|---:|---:|']
    for row in rows:
        lines.append(f"| {row['case']} | {row['variant']} | {row['native_ms']:.9f} | {row['checked_plans']} |")
    lines += ['', 'Detailed assertions, negative tests, search tests and source hashes are recorded in summary.json and source-identities.json.',
              'The Python mapping prototype is retired. Cross-language differential reports are historical evidence; this run validates native compiler behavior.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'passed':True,'workload_variants':len(rows),'plans_checked':sum(r['checked_plans'] for r in rows)}))
if __name__=='__main__':main()
