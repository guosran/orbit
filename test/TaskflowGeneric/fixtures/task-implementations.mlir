module {
  func.func @matrix_program(%A2: memref<16x8xbf16> {orbit.role = "input", orbit.name = "A2"}, %B2: memref<8x16xbf16> {orbit.role = "input", orbit.name = "B2"}, %A: memref<16x16xbf16> {orbit.role = "input", orbit.name = "A"}, %B: memref<16x16xbf16> {orbit.role = "input", orbit.name = "B"}, %Y: memref<16x16xf32> {orbit.role = "output", orbit.name = "Y"}) attributes {orbit.noalias, llvm.emit_c_interface, orbit.matrix_contract = "bf16-f32-accumulate-v1"} {
    %C = memref.alloc() {orbit.name = "C"} : memref<16x16xf32>
    %C_done = taskflow.task @C will_reads(%A, %B : memref<16x16xbf16>, memref<16x16xbf16>) will_writes(%C : memref<16x16xf32>) [original_read_memrefs(%A, %B : memref<16x16xbf16>, memref<16x16xbf16>), original_write_memrefs(%C : memref<16x16xf32>)] : (memref<16x16xbf16>, memref<16x16xbf16>, memref<16x16xf32>) -> memref<16x16xf32> {
    ^bb0(%a: memref<16x16xbf16>, %b: memref<16x16xbf16>, %c: memref<16x16xf32>):
      %zero = arith.constant 0.0 : f32
      affine.for %i = 0 to 16 {
        affine.for %j = 0 to 16 {
          affine.store %zero, %c[%i, %j] : memref<16x16xf32>
          affine.for %k = 0 to 16 {
            %av = affine.load %a[%i, %k] : memref<16x16xbf16>
            %bv = affine.load %b[%k, %j] : memref<16x16xbf16>
            %af = arith.extf %av : bf16 to f32
            %bf = arith.extf %bv : bf16 to f32
            %old = affine.load %c[%i, %j] : memref<16x16xf32>
            %product = arith.mulf %af, %bf : f32
            %sum = arith.addf %old, %product : f32
            affine.store %sum, %c[%i, %j] : memref<16x16xf32>
          }
        }
      }
      taskflow.yield done_writes(%c : memref<16x16xf32>)
    }
    %C_done2 = taskflow.task @C_second will_reads(%A2, %B2 : memref<16x8xbf16>, memref<8x16xbf16>) will_writes(%C_done : memref<16x16xf32>) [original_read_memrefs(%A2, %B2 : memref<16x8xbf16>, memref<8x16xbf16>), original_write_memrefs(%C_done : memref<16x16xf32>)] : (memref<16x8xbf16>, memref<8x16xbf16>, memref<16x16xf32>) -> memref<16x16xf32> {
    ^bb0(%a: memref<16x8xbf16>, %b: memref<8x16xbf16>, %c: memref<16x16xf32>):
      %zero = arith.constant 0.0 : f32
      affine.for %i = 0 to 16 {
        affine.for %j = 0 to 16 {
          affine.store %zero, %c[%i, %j] : memref<16x16xf32>
          affine.for %k = 0 to 8 {
            %av = affine.load %a[%i, %k] : memref<16x8xbf16>
            %bv = affine.load %b[%k, %j] : memref<8x16xbf16>
            %af = arith.extf %av : bf16 to f32
            %bf = arith.extf %bv : bf16 to f32
            %old = affine.load %c[%i, %j] : memref<16x16xf32>
            %product = arith.mulf %af, %bf : f32
            %sum = arith.addf %old, %product : f32
            affine.store %sum, %c[%i, %j] : memref<16x16xf32>
          }
        }
      }
      taskflow.yield done_writes(%c : memref<16x16xf32>)
    }
    %relu_done = taskflow.task @relu will_reads(%C_done2 : memref<16x16xf32>) will_writes(%Y : memref<16x16xf32>) [original_read_memrefs(%C_done2 : memref<16x16xf32>), original_write_memrefs(%Y : memref<16x16xf32>)] : (memref<16x16xf32>, memref<16x16xf32>) -> memref<16x16xf32> {
    ^bb0(%c: memref<16x16xf32>, %y: memref<16x16xf32>):
      %zero = arith.constant 0.0 : f32
      affine.for %i = 0 to 16 {
        affine.for %j = 0 to 16 {
          %v = affine.load %c[%i, %j] : memref<16x16xf32>
          %r = arith.maximumf %v, %zero : f32
          affine.store %r, %y[%i, %j] : memref<16x16xf32>
        }
      }
      taskflow.yield done_writes(%y : memref<16x16xf32>)
    }
    memref.dealloc %C : memref<16x16xf32>
    return
  }
}
