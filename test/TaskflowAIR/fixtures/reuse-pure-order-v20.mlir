module {
 func.func @data_reuse(%A: memref<17x136xf32> {orbit.role="input",orbit.name="A"},%D: memref<17x136xf32> {orbit.role="input",orbit.name="D"},%O: memref<17x136xf32> {orbit.role="output",orbit.name="O"},%H: memref<17x136xf32> {orbit.role="output",orbit.name="H"}) attributes {orbit.noalias,llvm.emit_c_interface} {
  %X=memref.alloc() {orbit.name="X"} : memref<17x136xf32>
  %V=memref.alloc() {orbit.name="V"} : memref<17x136xf32>
  %Y=memref.alloc() {orbit.name="Y"} : memref<17x136xf32>
  %Z=memref.alloc() {orbit.name="Z"} : memref<17x136xf32>
  %done_X=taskflow.task @write_X will_reads(%A : memref<17x136xf32>) will_writes(%X : memref<17x136xf32>) [original_read_memrefs(%A : memref<17x136xf32>),original_write_memrefs(%X : memref<17x136xf32>)] : (memref<17x136xf32>,memref<17x136xf32>)->memref<17x136xf32> {
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
  %done_V=taskflow.task @write_V will_reads(%D : memref<17x136xf32>) will_writes(%V : memref<17x136xf32>) [original_read_memrefs(%D : memref<17x136xf32>),original_write_memrefs(%V : memref<17x136xf32>)] : (memref<17x136xf32>,memref<17x136xf32>)->memref<17x136xf32> {
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
  %done_Y=taskflow.task @write_Y will_reads(%done_X,%A,%done_V : memref<17x136xf32>,memref<17x136xf32>,memref<17x136xf32>) will_writes(%Y : memref<17x136xf32>) [original_read_memrefs(%done_X,%A,%done_V : memref<17x136xf32>,memref<17x136xf32>,memref<17x136xf32>),original_write_memrefs(%Y : memref<17x136xf32>)] : (memref<17x136xf32>,memref<17x136xf32>,memref<17x136xf32>,memref<17x136xf32>)->memref<17x136xf32> {
  ^bb0(%r0: memref<17x136xf32>,%r1: memref<17x136xf32>,%order_only: memref<17x136xf32>,%dst: memref<17x136xf32>):
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
  %done_Z=taskflow.task @write_Z will_reads(%done_X : memref<17x136xf32>) will_writes(%Z : memref<17x136xf32>) [original_read_memrefs(%done_X : memref<17x136xf32>),original_write_memrefs(%Z : memref<17x136xf32>)] : (memref<17x136xf32>,memref<17x136xf32>)->memref<17x136xf32> {
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
  %done_H=taskflow.task @write_H will_reads(%done_V : memref<17x136xf32>) will_writes(%H : memref<17x136xf32>) [original_read_memrefs(%done_V : memref<17x136xf32>),original_write_memrefs(%H : memref<17x136xf32>)] : (memref<17x136xf32>,memref<17x136xf32>)->memref<17x136xf32> {
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
  memref.dealloc %X : memref<17x136xf32>
  memref.dealloc %V : memref<17x136xf32>
  memref.dealloc %Y : memref<17x136xf32>
  memref.dealloc %Z : memref<17x136xf32>
  return
 }
}
