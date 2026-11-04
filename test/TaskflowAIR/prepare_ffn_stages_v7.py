#!/usr/bin/env python3
"""Independent complete-main stage diagnostics; never a production planner."""
import argparse, hashlib, json, re
from pathlib import Path
import numpy as np
from prepare_v6 import fixture


def stage_source(stage, m, d, intermediate):
    source = fixture('ffn', m, d, intermediate)
    if stage == 'full':
        return source
    tasks = re.findall(r'    %\w+_done = taskflow.task .*?(?=    %\w+_done = taskflow.task |    memref.dealloc)', source, re.S)
    index = dict(gate=0, up=1, activation=2, down=3)[stage]
    body = re.sub(r'^    %\w+_done =', '    %stage_result =', tasks[index])
    body = body.replace('%G_done', '%G').replace('%U_done', '%U').replace('%H_done', '%H')
    shapes = dict(X=(m,d), W_gate=(d,intermediate), W_up=(d,intermediate),
                  G=(m,intermediate), U=(m,intermediate), H=(m,intermediate),
                  W_down=(intermediate,d), Y=(m,d))
    names = dict(gate=['X','W_gate','G'], up=['X','W_up','U'],
                 activation=['G','U','H'], down=['H','W_down','Y'])[stage]
    arguments = []
    for i, name in enumerate(names):
        a,b = shapes[name]
        role = 'output' if i == 2 else 'input'
        arguments.append(f'%{name}: memref<{a}x{b}xf32> {{orbit.role = "{role}", orbit.name = "{name}"}}')
    return 'module {\n  func.func @'+stage+'('+', '.join(arguments)+') attributes {orbit.noalias, llvm.emit_c_interface} {\n'+body+'    return\n  }\n}\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--m',type=int,required=True)
    p.add_argument('--d',type=int,default=2048)
    p.add_argument('--intermediate',type=int,default=11008)
    p.add_argument('--seeds',type=int,nargs='+',default=[11])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if min(a.m,a.d,a.intermediate)<=0:p.error('positive dimensions required')
    a.output.mkdir(parents=True,exist_ok=False)
    stages=['gate','up','activation','down','full']
    for stage in stages:
        out=a.output/stage;out.mkdir()
        source=stage_source(stage,a.m,a.d,a.intermediate)
        (out/'source.mlir').write_text(source)
        (out/'fixture.json').write_text(json.dumps(dict(stage=stage,m=a.m,d=a.d,intermediate=a.intermediate,
            seeds=a.seeds,source_sha256=hashlib.sha256(source.encode()).hexdigest(),diagnostic=True,
            reference='float64 explicit equations; isolated stage inputs rounded to f32 before its reference; full program has no intermediate rounding in reference'),indent=2)+'\n')
    for seed in a.seeds:
        rng=np.random.default_rng(seed)
        def draw(shape,scale=1):
            x=(rng.uniform(-.5,.5,shape)*scale).astype(np.float32);x.flat[0]=0;return x
        x=draw((a.m,a.d));wg=draw((a.d,a.intermediate),1/np.sqrt(a.d))
        wu=draw((a.d,a.intermediate),1/np.sqrt(a.d));wd=draw((a.intermediate,a.d),1/np.sqrt(a.intermediate))
        g=x.astype(np.float64)@wg.astype(np.float64);u=x.astype(np.float64)@wu.astype(np.float64)
        h=g/(1+np.exp(-g))*u
        gf,uf,hf=g.astype(np.float32),u.astype(np.float32),h.astype(np.float32)
        cases=dict(gate=({'X':x,'W_gate':wg},{'G':gf}),up=({'X':x,'W_up':wu},{'U':uf}),
            activation=({'G':gf,'U':uf},{'H':(gf.astype(np.float64)/(1+np.exp(-gf.astype(np.float64)))*uf.astype(np.float64)).astype(np.float32)}),
            down=({'H':hf,'W_down':wd},{'Y':(hf.astype(np.float64)@wd.astype(np.float64)).astype(np.float32)}),
            full=({'X':x,'W_gate':wg,'W_up':wu,'W_down':wd},{'Y':(h@wd.astype(np.float64)).astype(np.float32)}))
        for stage,(inputs,reference) in cases.items():
            np.savez(a.output/stage/f'inputs-{seed}.npz',**inputs)
            np.savez(a.output/stage/f'reference-{seed}.npz',**reference)

if __name__=='__main__':main()
