module {
// Oracle finite-domain fixture. Sidecar JSON costs/capacities are synthetic, not hardware measurements.
func.func @fork3(%A:memref<4x8xf32> {orbit.name="A",orbit.role="input"},%Y:memref<4x8xf32> {orbit.name="Y",orbit.role="output"},%Z:memref<4x8xf32> {orbit.name="Z",orbit.role="output"}) attributes {orbit.noalias,llvm.emit_c_interface} {
%X=memref.alloc() {orbit.name="X"}:memref<4x8xf32>
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
%done_Y=taskflow.task @Y will_reads(%done_X:memref<4x8xf32>) will_writes(%Y:memref<4x8xf32>) [original_read_memrefs(%done_X:memref<4x8xf32>),original_write_memrefs(%Y:memref<4x8xf32>)] : (memref<4x8xf32>,memref<4x8xf32>)->memref<4x8xf32> {
^bb0(%a0:memref<4x8xf32>,%o:memref<4x8xf32>):
affine.for %i=0 to 4 {affine.for %j=0 to 8 {
%v0=affine.load %a0[%i,%j]:memref<4x8xf32>
%c=arith.constant 1.0:f32
%r=arith.mulf %v0,%c:f32
affine.store %r,%o[%i,%j]:memref<4x8xf32>
}}
taskflow.yield done_writes(%o:memref<4x8xf32>)
}
%done_Z=taskflow.task @Z will_reads(%done_X:memref<4x8xf32>) will_writes(%Z:memref<4x8xf32>) [original_read_memrefs(%done_X:memref<4x8xf32>),original_write_memrefs(%Z:memref<4x8xf32>)] : (memref<4x8xf32>,memref<4x8xf32>)->memref<4x8xf32> {
^bb0(%a0:memref<4x8xf32>,%o:memref<4x8xf32>):
affine.for %i=0 to 4 {affine.for %j=0 to 8 {
%v0=affine.load %a0[%i,%j]:memref<4x8xf32>
%c=arith.constant 1.0:f32
%r=arith.addf %v0,%c:f32
affine.store %r,%o[%i,%j]:memref<4x8xf32>
}}
taskflow.yield done_writes(%o:memref<4x8xf32>)
}
memref.dealloc %X:memref<4x8xf32>
return
}}
