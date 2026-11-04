module {
  func.func @add(%a: memref<16xf32> {orbit.role = "input", orbit.name = "A"}, %b: memref<16xf32> {orbit.role = "input", orbit.name = "B"}, %c: memref<16xf32> {orbit.role = "output", orbit.name = "C"}) attributes {orbit.noalias, llvm.emit_c_interface} {
    %out = taskflow.task @add will_reads(%a, %b : memref<16xf32>, memref<16xf32>) will_writes(%c : memref<16xf32>) [original_read_memrefs(%a, %b : memref<16xf32>, memref<16xf32>), original_write_memrefs(%c : memref<16xf32>)] : (memref<16xf32>, memref<16xf32>, memref<16xf32>) -> memref<16xf32> {
    ^bb0(%x: memref<16xf32>, %y: memref<16xf32>, %z: memref<16xf32>):
      affine.for %i = 0 to 16 {
        %u = affine.load %x[%i] : memref<16xf32>
        %v = affine.load %y[%i] : memref<16xf32>
        %s = arith.addf %u, %v : f32
        affine.store %s, %z[%i] : memref<16xf32>
      }
      taskflow.yield done_writes(%z : memref<16xf32>)
    }
    return
  }
}
