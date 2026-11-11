module {
 func.func @data_reuse(%A: memref<17x136xf32> {orbit.role="input",orbit.name="A"},%D: memref<17x136xf32> {orbit.role="input",orbit.name="D"},%O: memref<17x136xf32> {orbit.role="output",orbit.name="O"},%U: memref<17x136xf32> {orbit.role="output",orbit.name="U"}) attributes {orbit.noalias,llvm.emit_c_interface} {
  %Y=memref.alloc() {orbit.name="Y"} : memref<17x136xf32>
  %Z=memref.alloc() {orbit.name="Z"} : memref<17x136xf32>
  %done_Y=taskflow.task @write_Y will_reads(%A : memref<17x136xf32>) will_writes(%Y : memref<17x136xf32>) [original_read_memrefs(%A : memref<17x136xf32>),original_write_memrefs(%Y : memref<17x136xf32>)] : (memref<17x136xf32>,memref<17x136xf32>)->memref<17x136xf32> {
  ^bb0(%r0: memref<17x136xf32>,%dst: memref<17x136xf32>):
   affine.for %i=0 to 17 {
    affine.for %j=0 to 136 {
     %v0=affine.load %r0[%i,%j] : memref<17x136xf32>
     %c=arith.constant 1.000000 : f32
     %v=arith.addf %v0,%c : f32
     affine.store %v,%dst[%i,%j] : memref<17x136xf32>
    }
   }
   taskflow.yield done_writes(%dst : memref<17x136xf32>)
  }
  %done_U=taskflow.task @write_U will_reads(%D : memref<17x136xf32>) will_writes(%U : memref<17x136xf32>) [original_read_memrefs(%D : memref<17x136xf32>),original_write_memrefs(%U : memref<17x136xf32>)] : (memref<17x136xf32>,memref<17x136xf32>)->memref<17x136xf32> {
  ^bb0(%r0: memref<17x136xf32>,%dst: memref<17x136xf32>):
   affine.for %i=0 to 17 {
    affine.for %j=0 to 136 {
     %v0=affine.load %r0[%i,%j] : memref<17x136xf32>
     %c=arith.constant 3.000000 : f32
     %v=arith.addf %v0,%c : f32
     affine.store %v,%dst[%i,%j] : memref<17x136xf32>
    }
   }
   taskflow.yield done_writes(%dst : memref<17x136xf32>)
  }
  %done_Z=taskflow.task @write_Z will_reads(%A : memref<17x136xf32>) will_writes(%Z : memref<17x136xf32>) [original_read_memrefs(%A : memref<17x136xf32>),original_write_memrefs(%Z : memref<17x136xf32>)] : (memref<17x136xf32>,memref<17x136xf32>)->memref<17x136xf32> {
  ^bb0(%r0: memref<17x136xf32>,%dst: memref<17x136xf32>):
   affine.for %i=0 to 17 {
    affine.for %j=0 to 136 {
     %v0=affine.load %r0[%i,%j] : memref<17x136xf32>
     %c=arith.constant 2.000000 : f32
     %v=arith.mulf %v0,%c : f32
     affine.store %v,%dst[%i,%j] : memref<17x136xf32>
    }
   }
   taskflow.yield done_writes(%dst : memref<17x136xf32>)
  }
  %done_O=taskflow.task @write_O will_reads(%done_Y,%done_Z : memref<17x136xf32>,memref<17x136xf32>) will_writes(%O : memref<17x136xf32>) [original_read_memrefs(%done_Y,%done_Z : memref<17x136xf32>,memref<17x136xf32>),original_write_memrefs(%O : memref<17x136xf32>)] : (memref<17x136xf32>,memref<17x136xf32>,memref<17x136xf32>)->memref<17x136xf32> {
  ^bb0(%r0: memref<17x136xf32>,%r1: memref<17x136xf32>,%dst: memref<17x136xf32>):
   affine.for %i=0 to 17 {
    affine.for %j=0 to 136 {
     %v0=affine.load %r0[%i,%j] : memref<17x136xf32>
     %v1=affine.load %r1[%i,%j] : memref<17x136xf32>
     %v=arith.addf %v0,%v1 : f32
     affine.store %v,%dst[%i,%j] : memref<17x136xf32>
    }
   }
   taskflow.yield done_writes(%dst : memref<17x136xf32>)
  }
  memref.dealloc %Y : memref<17x136xf32>
  memref.dealloc %Z : memref<17x136xf32>
  return
 }
}
