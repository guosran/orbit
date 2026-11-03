// RUN: mlir-amoeba-opt %s --prepare-structured-taskflow-for-neura --verify-each | FileCheck %s
// RUN: mlir-amoeba-opt %s --prepare-structured-taskflow-for-neura --lower-affine --lower-joint-taskflow-to-host-scf --verify-each | FileCheck %s --check-prefix=HOST
// A target-side implementation stage is sequenced by the actual write state;
// it must not repeat fill at each K block or lose the original completion.
module {
  func.func @main(%A: memref<2x3xf32>, %B: memref<3x2xf32>, %C: memref<2x2xf32>) {
    %done, %token = taskflow.task @gemm
      will_reads(%A, %B : memref<2x3xf32>, memref<3x2xf32>)
      will_writes(%C : memref<2x2xf32>)
      [original_read_memrefs(%A, %B : memref<2x3xf32>, memref<3x2xf32>),
       original_write_memrefs(%C : memref<2x2xf32>)]
      : (memref<2x3xf32>, memref<3x2xf32>, memref<2x2xf32>) -> (memref<2x2xf32>, i1) {
    ^bb0(%a: memref<2x3xf32>, %b: memref<3x2xf32>, %c: memref<2x2xf32>):
      %zero = arith.constant 0.0 : f32
      %ready = arith.constant true
      affine.for %i = 0 to 2 {
        affine.for %j = 0 to 2 {
          affine.store %zero, %c[%i, %j] : memref<2x2xf32>
        }
      }
      affine.for %i = 0 to 2 {
        affine.for %j = 0 to 2 {
          affine.for %k = 0 to 3 {
            %x = affine.load %a[%i, %k] : memref<2x3xf32>
            %y = affine.load %b[%k, %j] : memref<3x2xf32>
            %old = affine.load %c[%i, %j] : memref<2x2xf32>
            %p = arith.mulf %x, %y : f32
            %s = arith.addf %old, %p : f32
            affine.store %s, %c[%i, %j] : memref<2x2xf32>
          }
        }
      }
      taskflow.yield done_writes(%c : memref<2x2xf32>) values(%ready : i1)
    }
    return
  }
}
// CHECK: %[[FIRST:[a-zA-Z0-9_]+]], {{.*}} = taskflow.task @gemm_neura_stage_0
// CHECK: orbit.neura.source_task = "gemm"
// CHECK: taskflow.task @gemm_neura_stage_1
// CHECK-SAME: will_writes(%[[FIRST]] : memref<2x2xf32>)
// CHECK: orbit.neura.stage = 1 : i64
// HOST-NOT: taskflow.
// HOST: scf.for
// HOST: arith.mulf
// HOST: arith.addf
