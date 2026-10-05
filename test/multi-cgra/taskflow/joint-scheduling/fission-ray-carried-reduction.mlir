// RUN: mlir-amoeba-opt %s --pass-pipeline='builtin.module(func.func(fission-ray-carried-reduction))' --verify-each | FileCheck %s

module {
  func.func @ray_reduction(
      %bound: index,
      %input0: memref<1472xi32> {amoeba.logical_transfer_shape = array<i64: 1472>, amoeba.noalias},
      %matrix: memref<1472x8xi32> {amoeba.logical_transfer_shape = array<i64: 1472, 8>, amoeba.noalias},
      %output_index: memref<1472xi32> {amoeba.logical_transfer_shape = array<i64: 1472>, amoeba.noalias},
      %output_best: memref<1472xi32> {amoeba.logical_transfer_shape = array<i64: 1472>, amoeba.noalias},
      %threshold: i32, %fallback: i32, %initial_index: i32)
      attributes {amoeba.input0_caller_noalias_proven,
                  amoeba.input0_caller_noalias_proof_mode = "prepared-external-caller",
                  amoeba.input0_caller_evidence_path = "caller-evidence.json",
                  amoeba.input0_caller_prepared_input_path = "prepared.mlir",
                  amoeba.input0_caller_static_bound = 1472 : i64,
                  amoeba.static_bound.arg.0 = 1472 : i64} {
    %result:2 = taskflow.task @Task_13
        will_reads(%input0, %matrix : memref<1472xi32>, memref<1472x8xi32>)
        will_writes(%output_index, %output_best : memref<1472xi32>, memref<1472xi32>)
        value_inputs(%bound, %threshold, %fallback, %initial_index : index, i32, i32, i32)
        [original_read_memrefs(%input0, %matrix : memref<1472xi32>, memref<1472x8xi32>),
         original_write_memrefs(%output_index, %output_best : memref<1472xi32>, memref<1472xi32>)]
        : (memref<1472xi32>, memref<1472x8xi32>, memref<1472xi32>, memref<1472xi32>, index, i32, i32, i32) -> (memref<1472xi32>, memref<1472xi32>) {
      ^bb0(%read_input: memref<1472xi32>, %read_matrix: memref<1472x8xi32>,
           %write_index: memref<1472xi32>, %write_best: memref<1472xi32>,
           %n: index, %thresh: i32, %default: i32, %seed_index: i32):
        affine.for %row = 0 to %n {
          %seed = affine.load %read_input[%row] : memref<1472xi32>
          %seed_ok = arith.cmpi sgt, %seed, %thresh : i32
          %seed_best = arith.select %seed_ok, %seed, %default : i32
          %best, %index = affine.for %lane = 0 to 8
              iter_args(%best_init = %seed_best, %index_init = %seed_index) -> (i32, i32) {
            %lane_i32 = arith.index_cast %lane : index to i32
            %candidate = affine.load %read_matrix[%row, %lane] : memref<1472x8xi32>
            %candidate_ok = arith.cmpi sgt, %candidate, %thresh : i32
            %updated:2 = scf.if %candidate_ok -> (i32, i32) {
              %smaller = arith.cmpi slt, %candidate, %best_init : i32
              %next_best = arith.select %smaller, %candidate, %best_init : i32
              %next_index = arith.select %smaller, %lane_i32, %index_init : i32
              scf.yield %next_best, %next_index : i32, i32
            } else {
              scf.yield %best_init, %index_init : i32, i32
            }
            affine.yield %updated#0, %updated#1 : i32, i32
          }
          affine.store %index, %write_index[%row] : memref<1472xi32>
          affine.store %best, %write_best[%row] : memref<1472xi32>
        }
      taskflow.yield done_writes(%write_index, %write_best : memref<1472xi32>, memref<1472xi32>)
    }
    func.return
  }
}

// CHECK-LABEL: func.func @ray_reduction
// CHECK: taskflow.task @Task_13.fission.0
// CHECK-SAME: amoeba.fission.lane_interval = array<i64: 0, 4>
// CHECK-SAME: amoeba.fission.parent_source_work_count = 11776 : i64
// CHECK-SAME: amoeba.fission.stage_source_work_count = 5888 : i64
// CHECK: affine.for {{.*}} = 0 to %{{.*}}
// CHECK: affine.for {{.*}} = 0 to 4 iter_args
// CHECK: taskflow.task @Task_13
// CHECK-SAME: amoeba.fission.lane_interval = array<i64: 4, 8>
// CHECK-SAME: amoeba.fission.parent_source_work_count = 11776 : i64
// CHECK-SAME: amoeba.fission.stage_source_work_count = 5888 : i64
// CHECK: affine.for {{.*}} = 4 to 8 iter_args
