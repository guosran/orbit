#!/usr/bin/env python3
"""Replay independently selected per-layer FFN policies through frozen r18.

Only the owned adapter's FFN constructor is injected while creating its class;
no vendor function or frozen file is modified. The original source-line tracer,
all arithmetic, device kernels, memory guards and NPU lock remain in that runner.
Routing overhead is included in each measured FFN call and heldout model error.
"""
import argparse
import copy
import ctypes
import gc
import importlib.util
import json
from pathlib import Path
import sys
import time


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


class RoutedFFN:
    def __init__(self,native,root,config,routes,read):
        self.providers={};self.chains={};self.routes=routes;self.call_events=[]
        self.default=None
        # Compile artefacts fix the context identity; retention policies can
        # share that actual context while retaining distinct per-layer BO keys.
        for layer,row in routes.items():
            key=(row['retention'],row['split_order'],row['ffn_artifact']['sha256'])
            if key not in self.providers:
                provider=native(root,config['rows'],config['block_m'],config['block_n'],row['retention'],row['split_order'])
                proof=read(row['ffn_artifact']['manifest'])
                context=proof['artifact_sha256']
                if context not in self.chains:
                    chain=provider.build();chain._elf_path=proof['artifact'];chain._kernel_name=proof['entrypoint']
                    self.chains[context]=chain
                provider.chain=self.chains[context]
                self.providers[key]=provider
            row['_provider_key']=key
            if row['split_order']==config['split_order'] and row['retention']==config['retention']:
                self.default=self.providers[key]
        if self.default is None and self.providers:
            raise ValueError('Global FFN manifest/config must match one selected route')

    @property
    def events(self):return [event for p in self.providers.values() for event in p.events]

    @events.setter
    def events(self,value):
        if value:raise ValueError('Events can only be reset')
        for p in self.providers.values():p.events=[];p.call_events=[]

    def build(self):
        if self.default is None:raise ValueError('No fused source layer is selected')
        return self.default.chain

    def _provider(self,layer):
        row=self.routes[str(layer)]
        return self.providers[row['_provider_key']]

    def add_layer(self,layer,gate_up,down):self._provider(layer).add_layer(layer,gate_up,down)

    def describe(self):
        return dict(schema='orbit.qwen-r18-per-layer-ffn-router.v1',
            contexts=list(self.chains),layers={layer:{k:v for k,v in row.items() if k!='_provider_key'} for layer,row in self.routes.items()},
            physical_storage='native FFN per-layer prepared weights and BO keys; same-artifact policies share one real NPUChain')

    def run(self,layer,h):
        start=time.perf_counter_ns();p=self._provider(layer);chain=p.chain
        # The original backend may evict a context. Consult its actual BO
        # owner rather than claiming a previous static upload survived close.
        if chain._runner is None:p._seen_keys.clear()
        elif chain._runtime=='xrt':p._seen_keys.intersection_update(chain._runner._bos)
        result=p.run(layer,h)
        self.call_events.append(dict(layer=layer,exclusive_wall_ns=time.perf_counter_ns()-start,
            row_events=p.call_events[-1]['row_events'],runtime_config={k:v for k,v in self.routes[str(layer)].items() if k!='_provider_key'},
            boundary='complete routed FFN call including route/context ownership checks'))
        return result


def adapt(adapter):
    original_validate=adapter.validate_plan;original_make=adapter.make_prefiller_class
    native=adapter.backend.QwenFFN
    def routes_for(plan):
        config=plan['r18']
        routes=copy.deepcopy(plan.get('r18_layer_ffn',{}))
        if not routes:
            routes={str(layer):{key:copy.deepcopy(config[key]) for key in ('retention','split_order','ffn_artifact')} for layer in config['fused_layers']}
        if set(routes)!={str(layer) for layer in config['fused_layers']}:raise ValueError('Layer routes must exactly cover selected fused source layers')
        if any(set(row)!={'retention','split_order','ffn_artifact'} for row in routes.values()):raise ValueError('Unknown routed FFN implementation parameter')
        return routes
    def validate(source,plan):
        choices=original_validate(source,plan)
        seen=set()
        for layer,row in routes_for(plan).items():
            key=adapter.node.digest(row)
            if key in seen:continue
            checked=copy.deepcopy(plan);checked['r18'].update(row);checked['r18']['fused_layers']=[int(layer)]
            checked['plan_id']=adapter.node.digest({k:v for k,v in checked.items() if k!='plan_id'})
            original_validate(source,checked);seen.add(key)
        return choices
    def make(base,source,plan):
        validate(source,plan);routes=routes_for(plan)
        if not routes:
            prefiller=original_make(base,source,plan)
        else:
            config=plan['r18']
            def factory(root,*args,**kwargs):
                return RoutedFFN(native,root,config,routes,lambda p:json.loads(Path(p).read_text()))
            adapter.backend.QwenFFN=factory
            try:prefiller=original_make(base,source,plan)
            finally:adapter.backend.QwenFFN=native
        class PreparedPrefiller(prefiller):
            def make_npu_resident(self):
                result=super().make_npu_resident()
                # The source preparation frame has returned: release its dead
                # allocator pages before the frozen runner's unchanged guard.
                # This runs once, outside every measured inference request.
                def memory():
                    status={}
                    for line in Path('/proc/self/status').read_text().splitlines():
                        key,_,value=line.partition(':')
                        if key in ('VmRSS','VmHWM','VmSwap'):
                            status[key+'_bytes']=int(value.split()[0])*1024
                    return status
                before=memory()
                collected=gc.collect()
                trimmed=ctypes.CDLL('libc.so.6').malloc_trim(0)
                self.orbit_preparation.append(dict(
                    action='owned-pre-guard-dead-allocator-page-release',
                    before=before,after=memory(),gc_collected=collected,
                    malloc_trim_result=int(trimmed),inside_request_timing=False,
                    guard_policy='unchanged frozen runner RSS and swap limits'))
                return result
        return PreparedPrefiller
    adapter.validate_plan=validate;adapter.make_prefiller_class=make
    return adapter


def main():
    parser=argparse.ArgumentParser(description=__doc__,add_help=False)
    parser.add_argument('--frozen-runtime',type=Path,required=True)
    args,rest=parser.parse_known_args()
    runner=module(args.frozen_runtime/'scripts/run-qwen-r18-runtime.py','orbit_r18_routed_runner')
    loader=runner.module
    def load(path,name):
        result=loader(path,name)
        return adapt(result) if Path(path).name=='qwen-r18-runtime.py' else result
    runner.module=load
    sys.argv=[str(args.frozen_runtime/'scripts/run-qwen-r18-runtime.py'),*rest]
    runner.main()


if __name__=='__main__':main()
