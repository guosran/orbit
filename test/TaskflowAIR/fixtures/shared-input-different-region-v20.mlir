module {
 func.func @shared_boundary(%A: memref<5x18xf32> {orbit.role="input",orbit.name="A"}, %O: memref<5x17xf32> {orbit.role="output",orbit.name="O"}, %Z: memref<5x17xf32> {orbit.role="output",orbit.name="Z"}) attributes {orbit.noalias,llvm.emit_c_interface} {
  %d0=taskflow.task @reader_0 will_reads(%A : memref<5x18xf32>) will_writes(%O : memref<5x17xf32>) [original_read_memrefs(%A : memref<5x18xf32>),original_write_memrefs(%O : memref<5x17xf32>)] : (memref<5x18xf32>,memref<5x17xf32>)->memref<5x17xf32> {
  ^bb0(%r: memref<5x18xf32>,%dst: memref<5x17xf32>):
   affine.for %i=0 to 5 {
    affine.for %j=0 to 17 {
     %v=affine.load %r[%i,%j] : memref<5x18xf32>
     %c=arith.constant 2.0 : f32
     %w=arith.addf %v,%c : f32
     affine.store %w,%dst[%i,%j] : memref<5x17xf32>
    }
   }
   taskflow.yield done_writes(%dst : memref<5x17xf32>)
  }
  %d1=taskflow.task @reader_1 will_reads(%A : memref<5x18xf32>) will_writes(%Z : memref<5x17xf32>) [original_read_memrefs(%A : memref<5x18xf32>),original_write_memrefs(%Z : memref<5x17xf32>)] : (memref<5x18xf32>,memref<5x17xf32>)->memref<5x17xf32> {
  ^bb0(%r: memref<5x18xf32>,%dst: memref<5x17xf32>):
   affine.for %i=0 to 5 {
    affine.for %j=0 to 17 {
     %v=affine.load %r[%i,%j + 1] : memref<5x18xf32>
     %c=arith.constant 2.0 : f32
     %w=arith.mulf %v,%c : f32
     affine.store %w,%dst[%i,%j] : memref<5x17xf32>
    }
   }
   taskflow.yield done_writes(%dst : memref<5x17xf32>)
  }
  return
 }
}
