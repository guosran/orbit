module {
  func.func @add_scale(%A: memref<4x4xf32> {orbit.role = "input", orbit.name = "A"}, %B: memref<4x4xf32> {orbit.role = "input", orbit.name = "B"}, %C: memref<4x4xf32> {orbit.role = "output", orbit.name = "C"}) attributes {orbit.noalias, llvm.emit_c_interface} {
    %T = memref.alloc() {orbit.name = "T"} : memref<4x4xf32>
    %T_done = taskflow.task @add will_reads(%A, %B : memref<4x4xf32>, memref<4x4xf32>) will_writes(%T : memref<4x4xf32>) [original_read_memrefs(%A, %B : memref<4x4xf32>, memref<4x4xf32>), original_write_memrefs(%T : memref<4x4xf32>)] : (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>) -> memref<4x4xf32> {
    ^bb0(%A_b: memref<4x4xf32>, %B_b: memref<4x4xf32>, %T_b: memref<4x4xf32>):
      affine.for %i = 0 to 4 {
        affine.for %j = 0 to 4 {
          %x = affine.load %A_b[%i, %j] : memref<4x4xf32>
          %y = affine.load %B_b[%i, %j] : memref<4x4xf32>
          %sum = arith.addf %x, %y : f32
          affine.store %sum, %T_b[%i, %j] : memref<4x4xf32>
        }
      }
      taskflow.yield done_writes(%T_b : memref<4x4xf32>)
    }
    %C_done = taskflow.task @scale will_reads(%T_done : memref<4x4xf32>) will_writes(%C : memref<4x4xf32>) [original_read_memrefs(%T_done : memref<4x4xf32>), original_write_memrefs(%C : memref<4x4xf32>)] : (memref<4x4xf32>, memref<4x4xf32>) -> memref<4x4xf32> {
    ^bb0(%T_done_b: memref<4x4xf32>, %C_b: memref<4x4xf32>):
      %two = arith.constant 2.0 : f32
      affine.for %i = 0 to 4 {
        affine.for %j = 0 to 4 {
          %x = affine.load %T_done_b[%i, %j] : memref<4x4xf32>
          %v = arith.mulf %x, %two : f32
          affine.store %v, %C_b[%i, %j] : memref<4x4xf32>
        }
      }
      taskflow.yield done_writes(%C_b : memref<4x4xf32>)
    }
    memref.dealloc %T : memref<4x4xf32>
    return
  }
}
