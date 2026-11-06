module {
  func.func @fanout(%a: memref<8x64xf32> {orbit.role="input",orbit.name="A"}, %b: memref<8x64xf32> {orbit.role="output",orbit.name="B"}, %c: memref<8x64xf32> {orbit.role="output",orbit.name="C"}) attributes {orbit.noalias, llvm.emit_c_interface} {
    %x = memref.alloc() {orbit.name="X"} : memref<8x64xf32>
    %xd = taskflow.task @produce will_reads(%a : memref<8x64xf32>) will_writes(%x : memref<8x64xf32>) [original_read_memrefs(%a : memref<8x64xf32>), original_write_memrefs(%x : memref<8x64xf32>)] : (memref<8x64xf32>,memref<8x64xf32>) -> memref<8x64xf32> {
    ^bb0(%in: memref<8x64xf32>,%out: memref<8x64xf32>):
      affine.for %i=0 to 8 {
        affine.for %j=0 to 64 {
        %v=affine.load %in[%i, %j] : memref<8x64xf32>
        %two=arith.constant 2.0 : f32
        %r=arith.mulf %v,%two : f32
        affine.store %r,%out[%i, %j] : memref<8x64xf32>
      }
      }
      taskflow.yield done_writes(%out : memref<8x64xf32>)
    }
    %bd = taskflow.task @first will_reads(%xd : memref<8x64xf32>) will_writes(%b : memref<8x64xf32>) [original_read_memrefs(%xd : memref<8x64xf32>), original_write_memrefs(%b : memref<8x64xf32>)] : (memref<8x64xf32>,memref<8x64xf32>) -> memref<8x64xf32> {
    ^bb0(%in: memref<8x64xf32>,%out: memref<8x64xf32>):
      affine.for %i=0 to 8 {
        affine.for %j=0 to 64 {
        %v=affine.load %in[%i, %j] : memref<8x64xf32>
        affine.store %v,%out[%i, %j] : memref<8x64xf32>
      }
      }
      taskflow.yield done_writes(%out : memref<8x64xf32>)
    }
    %cd = taskflow.task @second will_reads(%xd,%bd : memref<8x64xf32>,memref<8x64xf32>) will_writes(%c : memref<8x64xf32>) [original_read_memrefs(%xd,%bd : memref<8x64xf32>,memref<8x64xf32>), original_write_memrefs(%c : memref<8x64xf32>)] : (memref<8x64xf32>,memref<8x64xf32>,memref<8x64xf32>) -> memref<8x64xf32> {
    ^bb0(%in: memref<8x64xf32>,%orderOnly: memref<8x64xf32>,%out: memref<8x64xf32>):
      affine.for %i=0 to 8 {
        affine.for %j=0 to 64 {
        %v=affine.load %in[%i, %j] : memref<8x64xf32>
        affine.store %v,%out[%i, %j] : memref<8x64xf32>
      }
      }
      taskflow.yield done_writes(%out : memref<8x64xf32>)
    }
    memref.dealloc %x : memref<8x64xf32>
    return
  }
}
