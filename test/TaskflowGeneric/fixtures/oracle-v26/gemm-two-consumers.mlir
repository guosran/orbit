module {
// Oracle finite-domain fixture. Sidecar JSON costs/capacities are synthetic, not hardware measurements.

  func.func @gemmfork(%A: memref<2x4xf32> {orbit.role = "input", orbit.name = "A"}, %B: memref<4x8xf32> {orbit.role = "input", orbit.name = "B"}, %Y: memref<2x8xf32> {orbit.role="output",orbit.name="Y"}, %Z: memref<2x8xf32> {orbit.role="output",orbit.name="Z"}) attributes {orbit.noalias, llvm.emit_c_interface} {
    %C=memref.alloc() {orbit.name="C"}:memref<2x8xf32>
    %C_done = taskflow.task @gemm will_reads(%A, %B : memref<2x4xf32>, memref<4x8xf32>) will_writes(%C : memref<2x8xf32>) [original_read_memrefs(%A, %B : memref<2x4xf32>, memref<4x8xf32>), original_write_memrefs(%C : memref<2x8xf32>)] : (memref<2x4xf32>, memref<4x8xf32>, memref<2x8xf32>) -> memref<2x8xf32> {
    ^bb0(%A_b: memref<2x4xf32>, %B_b: memref<4x8xf32>, %C_b: memref<2x8xf32>):
      %zero = arith.constant 0.0 : f32
      affine.for %i = 0 to 2 {
        affine.for %j = 0 to 8 {
          affine.store %zero, %C_b[%i, %j] : memref<2x8xf32>
          affine.for %k = 0 to 4 {
            %x = affine.load %A_b[%i, %k] : memref<2x4xf32>
            %w = affine.load %B_b[%k, %j] : memref<4x8xf32>
            %old = affine.load %C_b[%i, %j] : memref<2x8xf32>
            %p = arith.mulf %x, %w : f32
            %sum = arith.addf %old, %p : f32
            affine.store %sum, %C_b[%i, %j] : memref<2x8xf32>
          }
        }
      }
      taskflow.yield done_writes(%C_b : memref<2x8xf32>)
    }
%done_Y=taskflow.task @Y will_reads(%C_done:memref<2x8xf32>) will_writes(%Y:memref<2x8xf32>) [original_read_memrefs(%C_done:memref<2x8xf32>),original_write_memrefs(%Y:memref<2x8xf32>)] : (memref<2x8xf32>,memref<2x8xf32>)->memref<2x8xf32> {
^bb0(%a:memref<2x8xf32>,%o:memref<2x8xf32>):
%c=arith.constant 1.0:f32
affine.for %i = 0 to 2 {affine.for %j = 0 to 8 {
%v=affine.load %a[%i,%j]:memref<2x8xf32>
%r=arith.addf %v,%c:f32
affine.store %r,%o[%i,%j]:memref<2x8xf32>
}}
taskflow.yield done_writes(%o:memref<2x8xf32>)
}
%done_Z=taskflow.task @Z will_reads(%C_done:memref<2x8xf32>) will_writes(%Z:memref<2x8xf32>) [original_read_memrefs(%C_done:memref<2x8xf32>),original_write_memrefs(%Z:memref<2x8xf32>)] : (memref<2x8xf32>,memref<2x8xf32>)->memref<2x8xf32> {
^bb0(%a:memref<2x8xf32>,%o:memref<2x8xf32>):
%c=arith.constant 1.0:f32
affine.for %i = 0 to 2 {affine.for %j = 0 to 8 {
%v=affine.load %a[%i,%j]:memref<2x8xf32>
%r=arith.mulf %v,%c:f32
affine.store %r,%o[%i,%j]:memref<2x8xf32>
}}
taskflow.yield done_writes(%o:memref<2x8xf32>)
}
memref.dealloc %C:memref<2x8xf32>
    return
  }
}
