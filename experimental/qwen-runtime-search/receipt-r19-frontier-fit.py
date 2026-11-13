#!/usr/bin/env python3
"""Inspect a completed strict fit without rerunning fitting or capture reads."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import pstats
import re


def load(path):return json.loads(Path(path).read_text())
def record(path):
    path=Path(path).resolve(strict=True)
    with path.open('rb') as stream:checksum=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(file=str(path),sha256=checksum,bytes=path.stat().st_size)
def require(condition,message):
    if not condition:raise ValueError(message)
def key(operators,config):return (tuple(operators),config['backend'],config['input_policy'],config['output_policy'])


def inspect(root,catalogue_path,frontier_path,inventory_path,profile_path,time_path,log_path):
    catalogue=load(catalogue_path);frontier=load(frontier_path);inventory=load(inventory_path)
    runtime=catalogue['runtime_registration']['implementations'];impls={row['id']:row for row in catalogue['implementations']}
    require(len(impls)==len(catalogue['implementations']),'Repeated implementation identity')
    by_config=collections.defaultdict(list)
    for row in catalogue['implementations']:
        reg=runtime[row['id']]
        if reg['kind']=='prefill-cpu-region':by_config[key(reg['operators'],reg)].append(row)
        require(row['executable'] and type(row['wall_ms']) in (int,float) and math.isfinite(row['wall_ms']) and row['wall_ms']>=0,'Missing actual executable price')
    actual=[]
    for pending in frontier['implementations']:
        matches=by_config[key(pending['operators'],pending['runtime_config'])]
        require(len(matches)==1,'New pending source/policy has ambiguous/missing fitted implementation')
        row=matches[0];price=row['training_price']
        require(type(price['samples'])==int and price['samples']>=3 and price['samples']%3==0,'New implementation lacks complete three-sample measurements')
        require(abs(row['wall_ms']*1e6-price['median_ns'])<1e-6,'Local wall price differs from actual sample median')
        require(sum(price['preparation_sample_counts'].values())==price['samples'],'Price/preparation sample counts mismatch')
        require(pending['id'] in inventory['pending_choice_coverage'],'New pending has no actual successful cohort evidence')
        actual.append(dict(pending_id=pending['id'],implementation_id=row['id'],source_operators=pending['operators'],wall_ms=row['wall_ms'],actual_samples=price['samples'],training_price=price,
            actual_cohorts=[r['cohort'] for r in inventory['pending_choice_coverage'][pending['id']]]))
    training=catalogue['training'];names=[row['name'] for row in training]
    require(len(names)==len(set(names))==inventory['strictfit_support_cohorts'],'Actual strictfit support receipt set mismatch')
    require({row['name'] for row in inventory['cohorts']}<=set(names),'Strictfit omits an actual new successful cohort')
    require(catalogue['missing_local_prices']==0 and catalogue['old_prices_reused'] is False,'Incomplete actual-price closure')
    groups=catalogue['choice_groups'];sizes=[len(group['choices']) for group in groups]
    max_group=max(groups,key=lambda g:len(g['choices']));choice=max_group['choices'][0]
    max_operators=len({o for identity in choice['implementations'] for o in impls[identity]['source_operators']})
    stats=pstats.Stats(str(profile_path));domain=[];construction=[];top=[]
    for (file,line,name),(primitive,total,self_seconds,cumulative,callers) in sorted(stats.stats.items(),key=lambda item:item[1][3],reverse=True):
        row=dict(file=file,line=line,function=name,primitive_calls=primitive,total_calls=total,self_seconds=self_seconds,cumulative_seconds=cumulative)
        if len(top)<15:top.append(row)
        if Path(file).name=='build-r19-cover-domain.py' and name in ('build','exact_cover_groups'):domain.append(row)
        if Path(file).name=='qwen-r19-regions.py' and name in ('__init__','region','_template','_specialization','prepare'):construction.append(row)
    timing=time_path.read_text()
    elapsed=re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)',timing).group(1)
    rss=int(re.search(r'Maximum resident set size \(kbytes\): (\d+)',timing).group(1))
    require(re.search(r'Exit status: 0\s*$',timing) is not None,'Fit process did not exit successfully')
    log=load(log_path);require(log['registration']==catalogue['execution_registration_identity'] and log['implementations']==len(impls),'Actual fitter log/catalogue mismatch')
    support=catalogue['ready_runtime']['async_runtime_support_proof'];support_payload=load(support['file'])
    canonical=hashlib.sha256(json.dumps({k:v for k,v in support_payload.items() if k!='identity'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    require(support_payload['identity']==support['identity']==canonical,'Actual runtime support canonical identity mismatch')
    return dict(schema='orbit.qwen-r19-frontier-actual-fit-receipt.v1',catalogue=record(catalogue_path),runtime_support=record(support['file']),producer=catalogue['producer'],
        execution_registration_identity=catalogue['execution_registration_identity'],campaign=catalogue['training_campaign'],cpu_registration=catalogue['cpu_pending_registration'],frontier=record(frontier_path),actual_receipt_inventory=record(inventory_path),
        training_cohorts=len(training),training_receipts=training,implementations=len(impls),implementation_kinds=dict(collections.Counter(r['kind'] for r in runtime.values())),missing_local_prices=0,old_prices_reused=False,
        new_pending_choices=len(actual),new_actual_price_matches=actual,previous_pending_choices=2854,previous_and_new_cpu_regions=len(by_config),
        cover_components=len(groups),variable_dimensions=sum(size>1 for size in sizes),fixed_dimensions=sum(size==1 for size in sizes),total_component_choices=sum(sizes),maximum_component_choices=max(sizes),
        maximum_component_source_operators=max_operators,choices_per_component_distribution=dict(collections.Counter(sizes)),
        producer_elapsed_wall=elapsed,producer_peak_rss_kib=rss,cprofile_total_seconds=stats.total_tt,domain_profile=domain,profile_construction=construction,top_cumulative_profile=top,
        profiling='Completed strict fitter under actual cProfile and time -v; no fitting/model/capture read repeated by this inspection',profile=record(profile_path),time=record(time_path),log=record(log_path),receipt_producer=record(__file__))


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('root','catalogue','frontier','inventory','profile','time','log','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();receipt=inspect(args.root,args.catalogue,args.frontier,args.inventory,args.profile,args.time,args.log)
    with args.output.open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    print(json.dumps({k:receipt[k] for k in ('training_cohorts','implementations','new_pending_choices','cover_components','variable_dimensions','total_component_choices','maximum_component_choices','maximum_component_source_operators','producer_elapsed_wall','producer_peak_rss_kib','cprofile_total_seconds')}));print(json.dumps(record(args.output)))


if __name__=='__main__':main()
