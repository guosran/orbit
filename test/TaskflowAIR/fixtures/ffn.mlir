module {
  func.func @ffn(%X: memref<2x8xf32> {orbit.role = "input", orbit.name = "X"}, %W_gate: memref<8x16xf32> {orbit.role = "input", orbit.name = "W_gate"}, %W_up: memref<8x16xf32> {orbit.role = "input", orbit.name = "W_up"}, %W_down: memref<16x8xf32> {orbit.role = "input", orbit.name = "W_down"}, %Y: memref<2x8xf32> {orbit.role = "output", orbit.name = "Y"}) attributes {orbit.noalias, llvm.emit_c_interface} {
    %G = memref.alloc() {orbit.name = "G"} : memref<2x16xf32>
    %U = memref.alloc() {orbit.name = "U"} : memref<2x16xf32>
    %H = memref.alloc() {orbit.name = "H"} : memref<2x16xf32>
    %G_done = taskflow.task @gate will_reads(%X, %W_gate : memref<2x8xf32>, memref<8x16xf32>) will_writes(%G : memref<2x16xf32>) [original_read_memrefs(%X, %W_gate : memref<2x8xf32>, memref<8x16xf32>), original_write_memrefs(%G : memref<2x16xf32>)] : (memref<2x8xf32>, memref<8x16xf32>, memref<2x16xf32>) -> memref<2x16xf32> {
    ^bb0(%X_b: memref<2x8xf32>, %W_gate_b: memref<8x16xf32>, %G_b: memref<2x16xf32>):
      %zero = arith.constant 0.0 : f32
      affine.for %i = 0 to 2 {
        affine.for %j = 0 to 16 {
          affine.store %zero, %G_b[%i, %j] : memref<2x16xf32>
          affine.for %k = 0 to 8 {
            %x = affine.load %X_b[%i, %k] : memref<2x8xf32>
            %w = affine.load %W_gate_b[%k, %j] : memref<8x16xf32>
            %old = affine.load %G_b[%i, %j] : memref<2x16xf32>
            %p = arith.mulf %x, %w : f32
            %sum = arith.addf %old, %p : f32
            affine.store %sum, %G_b[%i, %j] : memref<2x16xf32>
          }
        }
      }
      taskflow.yield done_writes(%G_b : memref<2x16xf32>)
    }
    %U_done = taskflow.task @up will_reads(%X, %W_up : memref<2x8xf32>, memref<8x16xf32>) will_writes(%U : memref<2x16xf32>) [original_read_memrefs(%X, %W_up : memref<2x8xf32>, memref<8x16xf32>), original_write_memrefs(%U : memref<2x16xf32>)] : (memref<2x8xf32>, memref<8x16xf32>, memref<2x16xf32>) -> memref<2x16xf32> {
    ^bb0(%X_b: memref<2x8xf32>, %W_up_b: memref<8x16xf32>, %U_b: memref<2x16xf32>):
      %zero = arith.constant 0.0 : f32
      affine.for %i = 0 to 2 {
        affine.for %j = 0 to 16 {
          affine.store %zero, %U_b[%i, %j] : memref<2x16xf32>
          affine.for %k = 0 to 8 {
            %x = affine.load %X_b[%i, %k] : memref<2x8xf32>
            %w = affine.load %W_up_b[%k, %j] : memref<8x16xf32>
            %old = affine.load %U_b[%i, %j] : memref<2x16xf32>
            %p = arith.mulf %x, %w : f32
            %sum = arith.addf %old, %p : f32
            affine.store %sum, %U_b[%i, %j] : memref<2x16xf32>
          }
        }
      }
      taskflow.yield done_writes(%U_b : memref<2x16xf32>)
    }
    %H_done = taskflow.task @swiglu will_reads(%G_done, %U_done : memref<2x16xf32>, memref<2x16xf32>) will_writes(%H : memref<2x16xf32>) [original_read_memrefs(%G_done, %U_done : memref<2x16xf32>, memref<2x16xf32>), original_write_memrefs(%H : memref<2x16xf32>)] : (memref<2x16xf32>, memref<2x16xf32>, memref<2x16xf32>) -> memref<2x16xf32> {
    ^bb0(%G_done_b: memref<2x16xf32>, %U_done_b: memref<2x16xf32>, %H_b: memref<2x16xf32>):
      %one = arith.constant 1.0 : f32
      affine.for %i = 0 to 2 {
        affine.for %j = 0 to 16 {
          %g = affine.load %G_done_b[%i, %j] : memref<2x16xf32>
          %u = affine.load %U_done_b[%i, %j] : memref<2x16xf32>
          %neg = arith.negf %g : f32
          %e = math.exp %neg : f32
          %den = arith.addf %one, %e : f32
          %sig = arith.divf %g, %den : f32
          %h = arith.mulf %sig, %u : f32
          affine.store %h, %H_b[%i, %j] : memref<2x16xf32>
        }
      }
      taskflow.yield done_writes(%H_b : memref<2x16xf32>)
    }
    %Y_done = taskflow.task @down will_reads(%H_done, %W_down : memref<2x16xf32>, memref<16x8xf32>) will_writes(%Y : memref<2x8xf32>) [original_read_memrefs(%H_done, %W_down : memref<2x16xf32>, memref<16x8xf32>), original_write_memrefs(%Y : memref<2x8xf32>)] : (memref<2x16xf32>, memref<16x8xf32>, memref<2x8xf32>) -> memref<2x8xf32> {
    ^bb0(%H_done_b: memref<2x16xf32>, %W_down_b: memref<16x8xf32>, %Y_b: memref<2x8xf32>):
      %zero = arith.constant 0.0 : f32
      affine.for %i = 0 to 2 {
        affine.for %j = 0 to 8 {
          affine.store %zero, %Y_b[%i, %j] : memref<2x8xf32>
          affine.for %k = 0 to 16 {
            %x = affine.load %H_done_b[%i, %k] : memref<2x16xf32>
            %w = affine.load %W_down_b[%k, %j] : memref<16x8xf32>
            %old = affine.load %Y_b[%i, %j] : memref<2x8xf32>
            %p = arith.mulf %x, %w : f32
            %sum = arith.addf %old, %p : f32
            affine.store %sum, %Y_b[%i, %j] : memref<2x8xf32>
          }
        }
      }
      taskflow.yield done_writes(%Y_b : memref<2x8xf32>)
    }
    memref.dealloc %G : memref<2x16xf32>
    memref.dealloc %U : memref<2x16xf32>
    memref.dealloc %H : memref<2x16xf32>
    return
  }
}
