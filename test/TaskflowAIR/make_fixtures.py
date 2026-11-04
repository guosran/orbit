#!/usr/bin/env python3
"""Build explicit test source fixtures, not a production graph importer."""
import argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--m',type=int,default=2);p.add_argument('--d',type=int,default=8);p.add_argument('--i',type=int,default=16);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
def ty(m,n):return f'memref<{m}x{n}xf32>'
def task(name,reads,out,types,body):
 vals=reads+[out];decl=', '.join(types[v] for v in vals)
 rd=', '.join('%'+v for v in reads);rt=', '.join(types[v] for v in reads);ot=types[out]
 return f'''    %{out}_done = taskflow.task @{name} will_reads({rd} : {rt}) will_writes(%{out} : {ot}) [original_read_memrefs({rd} : {rt}), original_write_memrefs(%{out} : {ot})] : ({decl}) -> {ot} {{
    ^bb0({', '.join('%'+v+'_b: '+types[v] for v in vals)}):
{body}
      taskflow.yield done_writes(%{out}_b : {ot})
    }}\n'''
def gemm(name,x,w,out,m,k,n,types):
 return task(name,[x,w],out,types,f'''      %zero = arith.constant 0.0 : f32
      affine.for %i = 0 to {m} {{
        affine.for %j = 0 to {n} {{
          affine.store %zero, %{out}_b[%i, %j] : {types[out]}
          affine.for %k = 0 to {k} {{
            %x = affine.load %{x}_b[%i, %k] : {types[x]}
            %w = affine.load %{w}_b[%k, %j] : {types[w]}
            %old = affine.load %{out}_b[%i, %j] : {types[out]}
            %p = arith.mulf %x, %w : f32
            %sum = arith.addf %old, %p : f32
            affine.store %sum, %{out}_b[%i, %j] : {types[out]}
          }}
        }}
      }}''')
def header(name,types,inputs,outputs):
 return 'module {\n  func.func @'+name+'('+', '.join('%'+v+': '+types[v]+' {orbit.role = "'+('input' if v in inputs else 'output')+'", orbit.name = "'+v+'"}' for v in inputs+outputs)+') attributes {orbit.noalias, llvm.emit_c_interface} {\n'
types={'A':ty(4,8),'B':ty(8,4),'C':ty(4,4)}
(a.output/'gemm.mlir').write_text(header('gemm',types,['A','B'],['C'])+gemm('gemm','A','B','C',4,8,4,types)+'    return\n  }\n}\n')
types={n:ty(4,4) for n in ['A','B','T','C']}
source=header('add_scale',types,['A','B'],['C'])+'    %T = memref.alloc() {orbit.name = "T"} : '+types['T']+'\n'
source+=task('add',['A','B'],'T',types,f'''      affine.for %i = 0 to 4 {{
        affine.for %j = 0 to 4 {{
          %x = affine.load %A_b[%i, %j] : {types['A']}
          %y = affine.load %B_b[%i, %j] : {types['B']}
          %sum = arith.addf %x, %y : f32
          affine.store %sum, %T_b[%i, %j] : {types['T']}
        }}
      }}''')
# Dependency is the yielded storage SSA version, not merely a reporting edge.
source+=task('scale',['T_done'],'C',{**types,'T_done':types['T']},f'''      %two = arith.constant 2.0 : f32
      affine.for %i = 0 to 4 {{
        affine.for %j = 0 to 4 {{
          %x = affine.load %T_done_b[%i, %j] : {types['T']}
          %v = arith.mulf %x, %two : f32
          affine.store %v, %C_b[%i, %j] : {types['C']}
        }}
      }}''')
source+='    memref.dealloc %T : '+types['T']+'\n    return\n  }\n}\n';(a.output/'add-scale.mlir').write_text(source)
m,d,i=a.m,a.d,a.i
types={'X':ty(m,d),'W_gate':ty(d,i),'W_up':ty(d,i),'W_down':ty(i,d),'G':ty(m,i),'U':ty(m,i),'H':ty(m,i),'Y':ty(m,d)}
source=header('ffn',types,['X','W_gate','W_up','W_down'],['Y'])
for v in ['G','U','H']:source+=f'    %{v} = memref.alloc() {{orbit.name = "{v}"}} : {types[v]}\n'
source+=gemm('gate','X','W_gate','G',m,d,i,types)+gemm('up','X','W_up','U',m,d,i,types)
source+=task('swiglu',['G_done','U_done'],'H',{**types,'G_done':types['G'],'U_done':types['U']},f'''      %one = arith.constant 1.0 : f32
      affine.for %i = 0 to {m} {{
        affine.for %j = 0 to {i} {{
          %g = affine.load %G_done_b[%i, %j] : {types['G']}
          %u = affine.load %U_done_b[%i, %j] : {types['U']}
          %neg = arith.negf %g : f32
          %e = math.exp %neg : f32
          %den = arith.addf %one, %e : f32
          %sig = arith.divf %g, %den : f32
          %h = arith.mulf %sig, %u : f32
          affine.store %h, %H_b[%i, %j] : {types['H']}
        }}
      }}''')
source+=gemm('down','H_done','W_down','Y',m,i,d,{**types,'H_done':types['H']})
for v in ['G','U','H']:source+=f'    memref.dealloc %{v} : {types[v]}\n'
source+='    return\n  }\n}\n';(a.output/'ffn.mlir').write_text(source)
