module {
 func.func @diamond(%A: memref<64x128xf32> {orbit.role="input",orbit.name="A"},%B: memref<64x128xf32> {orbit.role="output",orbit.name="B"},%C: memref<64x128xf32> {orbit.role="output",orbit.name="C"}) attributes {orbit.noalias,llvm.emit_c_interface} {
  %X=memref.alloc() {orbit.name="X"} : memref<64x128xf32>
  %xd=taskflow.task @produce will_reads(%A : memref<64x128xf32>) will_writes(%X : memref<64x128xf32>) [original_read_memrefs(%A : memref<64x128xf32>),original_write_memrefs(%X : memref<64x128xf32>)] : (memref<64x128xf32>,memref<64x128xf32>)->memref<64x128xf32> {
  ^bb0(%in: memref<64x128xf32>,%out: memref<64x128xf32>):
   affine.for %i=0 to 64 {
    affine.for %j=0 to 128 {
     %v=affine.load %in[%i,%j] : memref<64x128xf32>
     %bias=arith.constant 1.0 : f32
     %result=arith.addf %v,%bias : f32
     affine.store %result,%out[%i,%j] : memref<64x128xf32>
    }
   }
   taskflow.yield done_writes(%out : memref<64x128xf32>)
  }
  %bd=taskflow.task @first will_reads(%xd : memref<64x128xf32>) will_writes(%B : memref<64x128xf32>) [original_read_memrefs(%xd : memref<64x128xf32>),original_write_memrefs(%B : memref<64x128xf32>)] : (memref<64x128xf32>,memref<64x128xf32>)->memref<64x128xf32> {
  ^bb0(%in: memref<64x128xf32>,%out: memref<64x128xf32>):
   affine.for %i=0 to 64 {
    affine.for %j=0 to 128 {
     %v=affine.load %in[%i,%j] : memref<64x128xf32>
     %bias=arith.constant 2.0 : f32
     %result=arith.addf %v,%bias : f32
     affine.store %result,%out[%i,%j] : memref<64x128xf32>
    }
   }
   taskflow.yield done_writes(%out : memref<64x128xf32>)
  }
  %cd=taskflow.task @second will_reads(%xd : memref<64x128xf32>) will_writes(%C : memref<64x128xf32>) [original_read_memrefs(%xd : memref<64x128xf32>),original_write_memrefs(%C : memref<64x128xf32>)] : (memref<64x128xf32>,memref<64x128xf32>)->memref<64x128xf32> {
  ^bb0(%in: memref<64x128xf32>,%out: memref<64x128xf32>):
   affine.for %i=0 to 64 {
    affine.for %j=0 to 128 {
     %v=affine.load %in[%i,%j] : memref<64x128xf32>
     %result=arith.addf %v,%v : f32
     affine.store %result,%out[%i,%j] : memref<64x128xf32>
    }
   }
   taskflow.yield done_writes(%out : memref<64x128xf32>)
  }
  memref.dealloc %X : memref<64x128xf32>
  return
 }
}
