// RUN: mlir-amoeba-opt %s --verify-each \
// RUN:   '--prove-static-active-transfer-shapes=function=mixed_root_join arguments=0' \
// RUN:   -o /dev/null > %t-mixed.log 2>&1; test $? -ne 0
// RUN: FileCheck %s --check-prefix=MIXED --input-file=%t-mixed.log

module {
  func.func @mixed_root_join(
      %left: memref<?xi32> {
        amoeba.logical_transfer_shape = array<i64: 4>, amoeba.noalias
      },
      %right: memref<?xi32> {
        amoeba.logical_transfer_shape = array<i64: 4>, amoeba.noalias
      }) attributes {
        amoeba.input0_caller_evidence_path = "fixture-caller.mlir",
        amoeba.input0_caller_function = "main",
        amoeba.input0_caller_noalias_proof_mode = "prepared-external-caller",
        amoeba.input0_caller_noalias_proven,
        amoeba.input0_caller_prepared_input_path = "fixture-prepared.mlir"
      } {
    %base = memref.cast %left : memref<?xi32> to memref<4xi32>
    %tile0 = taskflow.task @Tile0
        will_writes(%left : memref<?xi32>)
        [original_write_memrefs(%left : memref<?xi32>)]
        : (memref<?xi32>) -> (memref<?xi32>) {
      ^bb0(%write_state: memref<?xi32>):
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %zero = arith.constant 0 : i32
        %index = taskflow.counter from %c0 to %c2 step %c1
            attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} : index
        memref.store %zero, %write_state[%index] : memref<?xi32>
        taskflow.yield done_writes(%write_state : memref<?xi32>)
    }
    %tile1 = taskflow.task @Tile1
        will_writes(%right : memref<?xi32>)
        [original_write_memrefs(%right : memref<?xi32>)]
        : (memref<?xi32>) -> (memref<?xi32>) {
      ^bb0(%write_state: memref<?xi32>):
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c4 = arith.constant 4 : index
        %zero = arith.constant 0 : i32
        %index = taskflow.counter from %c2 to %c4 step %c1
            attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} : index
        memref.store %zero, %write_state[%index] : memref<?xi32>
        taskflow.yield done_writes(%write_state : memref<?xi32>)
    }
    %tile0_static = memref.cast %tile0 : memref<?xi32> to memref<4xi32>
    %tile1_static = memref.cast %tile1 : memref<?xi32> to memref<4xi32>
    %joined = taskflow.join states(%tile0_static, %tile1_static)
        base(%base) axis(0) region([0], [4])
        {amoeba.replica.completion_only, amoeba.semantic.completion_only}
        : memref<4xi32>
    return
  }
}

// MIXED: cannot prove a tile state's static region and base
