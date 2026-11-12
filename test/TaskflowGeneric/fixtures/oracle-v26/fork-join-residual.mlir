module {
// Oracle finite-domain fixture. Sidecar JSON costs/capacities are synthetic, not hardware measurements.
func.func @forkjoin4(%A:memref<4x8xf32> {orbit.name="A",orbit.role="input"},%Y:memref<4x8xf32> {orbit.name="Y",orbit.role="output"}) attributes {orbit.noalias,llvm.emit_c_interface} {
%X=memref.alloc() {orbit.name="X"}:memref<4x8xf32>
%U=memref.alloc() {orbit.name="U"}:memref<4x8xf32>
%V=memref.alloc() {orbit.name="V"}:memref<4x8xf32>
%done_X=taskflow.task @X will_reads(%A:memref<4x8xf32>) will_writes(%X:memref<4x8xf32>) [original_read_memrefs(%A:memref<4x8xf32>),original_write_memrefs(%X:memref<4x8xf32>)] : (memref<4x8xf32>,memref<4x8xf32>)->memref<4x8xf32> {
^bb0(%a0:memref<4x8xf32>,%o:memref<4x8xf32>):
affine.for %i=0 to 4 {affine.for %j=0 to 8 {
%v0=affine.load %a0[%i,%j]:memref<4x8xf32>
%c=arith.constant 1.0:f32
%r=arith.addf %v0,%c:f32
affine.store %r,%o[%i,%j]:memref<4x8xf32>
}}
taskflow.yield done_writes(%o:memref<4x8xf32>)
}
%done_U=taskflow.task @U will_reads(%done_X:memref<4x8xf32>) will_writes(%U:memref<4x8xf32>) [original_read_memrefs(%done_X:memref<4x8xf32>),original_write_memrefs(%U:memref<4x8xf32>)] : (memref<4x8xf32>,memref<4x8xf32>)->memref<4x8xf32> {
^bb0(%a0:memref<4x8xf32>,%o:memref<4x8xf32>):
affine.for %i=0 to 4 {affine.for %j=0 to 8 {
%v0=affine.load %a0[%i,%j]:memref<4x8xf32>
%c=arith.constant 1.0:f32
%r=arith.mulf %v0,%c:f32
affine.store %r,%o[%i,%j]:memref<4x8xf32>
}}
taskflow.yield done_writes(%o:memref<4x8xf32>)
}
%done_V=taskflow.task @V will_reads(%done_X:memref<4x8xf32>) will_writes(%V:memref<4x8xf32>) [original_read_memrefs(%done_X:memref<4x8xf32>),original_write_memrefs(%V:memref<4x8xf32>)] : (memref<4x8xf32>,memref<4x8xf32>)->memref<4x8xf32> {
^bb0(%a0:memref<4x8xf32>,%o:memref<4x8xf32>):
affine.for %i=0 to 4 {affine.for %j=0 to 8 {
%v0=affine.load %a0[%i,%j]:memref<4x8xf32>
%c=arith.constant 1.0:f32
%r=arith.addf %v0,%c:f32
affine.store %r,%o[%i,%j]:memref<4x8xf32>
}}
taskflow.yield done_writes(%o:memref<4x8xf32>)
}
%done_Y=taskflow.task @Y will_reads(%done_U,%done_V:memref<4x8xf32>,memref<4x8xf32>) will_writes(%Y:memref<4x8xf32>) [original_read_memrefs(%done_U,%done_V:memref<4x8xf32>,memref<4x8xf32>),original_write_memrefs(%Y:memref<4x8xf32>)] : (memref<4x8xf32>,memref<4x8xf32>,memref<4x8xf32>)->memref<4x8xf32> {
^bb0(%a0:memref<4x8xf32>,%a1:memref<4x8xf32>,%o:memref<4x8xf32>):
affine.for %i=0 to 4 {affine.for %j=0 to 8 {
%v0=affine.load %a0[%i,%j]:memref<4x8xf32>
%v1=affine.load %a1[%i,%j]:memref<4x8xf32>
%r=arith.addf %v0,%v1:f32
affine.store %r,%o[%i,%j]:memref<4x8xf32>
}}
taskflow.yield done_writes(%o:memref<4x8xf32>)
}
memref.dealloc %X:memref<4x8xf32>
memref.dealloc %U:memref<4x8xf32>
memref.dealloc %V:memref<4x8xf32>
return
}}
