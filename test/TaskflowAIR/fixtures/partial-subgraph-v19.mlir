module {
 func.func @partial_subgraph_v19(%A: memref<17x136xf32> {orbit.role="input",orbit.name="A"},%Z: memref<17x136xf32> {orbit.role="output",orbit.name="Z"},%O: memref<17x136xf32> {orbit.role="output",orbit.name="O"}) attributes {orbit.noalias,llvm.emit_c_interface} {
  %U=memref.alloc() {orbit.name="U"} : memref<17x136xf32>
  %X=memref.alloc() {orbit.name="X"} : memref<17x136xf32>
  %Y=memref.alloc() {orbit.name="Y"} : memref<17x136xf32>
  %d0=taskflow.task @stage_0 will_reads(%A : memref<17x136xf32>) will_writes(%U : memref<17x136xf32>) [original_read_memrefs(%A : memref<17x136xf32>),original_write_memrefs(%U : memref<17x136xf32>)] : (memref<17x136xf32>,memref<17x136xf32>)->memref<17x136xf32> {
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
  %d1=taskflow.task @stage_1 will_reads(%d0 : memref<17x136xf32>) will_writes(%X : memref<17x136xf32>) [original_read_memrefs(%d0 : memref<17x136xf32>),original_write_memrefs(%X : memref<17x136xf32>)] : (memref<17x136xf32>,memref<17x136xf32>)->memref<17x136xf32> {
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
  %d2=taskflow.task @stage_2 will_reads(%d1 : memref<17x136xf32>) will_writes(%Y : memref<17x136xf32>) [original_read_memrefs(%d1 : memref<17x136xf32>),original_write_memrefs(%Y : memref<17x136xf32>)] : (memref<17x136xf32>,memref<17x136xf32>)->memref<17x136xf32> {
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
  %d3=taskflow.task @stage_3 will_reads(%d1 : memref<17x136xf32>) will_writes(%Z : memref<17x136xf32>) [original_read_memrefs(%d1 : memref<17x136xf32>),original_write_memrefs(%Z : memref<17x136xf32>)] : (memref<17x136xf32>,memref<17x136xf32>)->memref<17x136xf32> {
  ^bb0(%r0: memref<17x136xf32>,%dst: memref<17x136xf32>):
   affine.for %i=0 to 17 {
    affine.for %j=0 to 136 {
     %v0=affine.load %r0[%i,%j] : memref<17x136xf32>
     %c=arith.constant 1.000000 : f32
     %v=arith.subf %v0,%c : f32
     affine.store %v,%dst[%i,%j] : memref<17x136xf32>
    }
   }
   taskflow.yield done_writes(%dst : memref<17x136xf32>)
  }
  %d4=taskflow.task @stage_4 will_reads(%d2,%d3 : memref<17x136xf32>,memref<17x136xf32>) will_writes(%O : memref<17x136xf32>) [original_read_memrefs(%d2,%d3 : memref<17x136xf32>,memref<17x136xf32>),original_write_memrefs(%O : memref<17x136xf32>)] : (memref<17x136xf32>,memref<17x136xf32>,memref<17x136xf32>)->memref<17x136xf32> {
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
  memref.dealloc %U : memref<17x136xf32>
  memref.dealloc %X : memref<17x136xf32>
  memref.dealloc %Y : memref<17x136xf32>
  return
 }
}
