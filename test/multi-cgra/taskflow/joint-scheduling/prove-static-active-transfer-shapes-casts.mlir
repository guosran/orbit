// RUN: mlir-amoeba-opt %s --verify-each \
// RUN:   '--prove-static-active-transfer-shapes=function=cast_completion_join arguments=0 report-output=%t-join.json' \
// RUN:   -o %t-join.mlir
// RUN: python3 -c 'import json,sys; p=json.load(open(sys.argv[1]))["proofs"][0]; assert p["status"] == "proven" and p["active_shape"] == [4] and p["capacity_shape"] == [4] and {a["task"] for a in p["accesses"]} == {"Tile0", "Tile1", "Consumer"}' %t-join.json
// RUN: FileCheck %s --check-prefix=CAST --input-file=%t-join.mlir
// RUN: mlir-amoeba-opt %s --verify-each \
// RUN:   '--prove-static-active-transfer-shapes=function=wrong_extent_cast arguments=0 report-output=%t-type.json' \
// RUN:   -o %t-type.mlir
// RUN: python3 -c 'import json,sys; p=json.load(open(sys.argv[1]))["proofs"][0]; assert p["status"] == "unknown" and p["capacity_shape"] == [4] and "active_shape" not in p' %t-type.json
// RUN: mlir-amoeba-opt %s --verify-each \
// RUN:   '--prove-static-active-transfer-shapes=function=strided_subview arguments=0 report-output=%t-layout.json' \
// RUN:   -o %t-layout.mlir
// RUN: python3 -c 'import json,sys; p=json.load(open(sys.argv[1]))["proofs"][0]; assert p["status"] == "unknown" and p["capacity_shape"] == [4] and "active_shape" not in p' %t-layout.json

module {
  func.func @cast_completion_join(
      %data: memref<?xi32> {
        amoeba.logical_transfer_shape = array<i64: 4>, amoeba.noalias
      }) attributes {
        amoeba.input0_caller_evidence_path = "fixture-caller.mlir",
        amoeba.input0_caller_function = "main",
        amoeba.input0_caller_noalias_proof_mode = "prepared-external-caller",
        amoeba.input0_caller_noalias_proven,
        amoeba.input0_caller_prepared_input_path = "fixture-prepared.mlir"
      } {
    %base = memref.cast %data : memref<?xi32> to memref<4xi32>

    %tile0 = taskflow.task @Tile0
        will_writes(%data : memref<?xi32>)
        [original_write_memrefs(%data : memref<?xi32>)]
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
        will_writes(%data : memref<?xi32>)
        [original_write_memrefs(%data : memref<?xi32>)]
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
    %joined_dynamic = memref.cast %joined : memref<4xi32> to memref<?xi32>

    %read_done, %write_done = taskflow.task @Consumer
        will_reads(%joined_dynamic : memref<?xi32>)
        will_writes(%joined_dynamic : memref<?xi32>)
        [original_read_memrefs(%data : memref<?xi32>),
         original_write_memrefs(%data : memref<?xi32>)]
        : (memref<?xi32>, memref<?xi32>) -> (memref<?xi32>, memref<?xi32>) {
      ^bb0(%read_state: memref<?xi32>, %write_state: memref<?xi32>):
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %index = taskflow.counter from %c0 to %c4 step %c1
            attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} : index
        %value = memref.load %read_state[%index] : memref<?xi32>
        memref.store %value, %write_state[%index] : memref<?xi32>
        taskflow.yield done_reads(%read_state : memref<?xi32>)
                         done_writes(%write_state : memref<?xi32>)
    }
    return
  }

  func.func @wrong_extent_cast(
      %data: memref<?xi32> {
        amoeba.logical_transfer_shape = array<i64: 4>, amoeba.noalias
      }) attributes {
        amoeba.input0_caller_evidence_path = "fixture-caller.mlir",
        amoeba.input0_caller_function = "main",
        amoeba.input0_caller_noalias_proof_mode = "prepared-external-caller",
        amoeba.input0_caller_noalias_proven,
        amoeba.input0_caller_prepared_input_path = "fixture-prepared.mlir"
      } {
    %wrong = memref.cast %data : memref<?xi32> to memref<5xi32>
    %c0 = arith.constant 0 : index
    %value = memref.load %wrong[%c0] : memref<5xi32>
    return
  }

  func.func @strided_subview(
      %data: memref<?xi32> {
        amoeba.logical_transfer_shape = array<i64: 4>, amoeba.noalias
      }) attributes {
        amoeba.input0_caller_evidence_path = "fixture-caller.mlir",
        amoeba.input0_caller_function = "main",
        amoeba.input0_caller_noalias_proof_mode = "prepared-external-caller",
        amoeba.input0_caller_noalias_proven,
        amoeba.input0_caller_prepared_input_path = "fixture-prepared.mlir"
      } {
    %view = memref.subview %data[0] [4] [2]
        : memref<?xi32> to memref<4xi32, strided<[2], offset: 0>>
    %c0 = arith.constant 0 : index
    %value = memref.load %view[%c0]
        : memref<4xi32, strided<[2], offset: 0>>
    return
  }
}

// CAST: amoeba.active_transfer_capacity_shape = array<i64: 4>
// CAST: amoeba.active_transfer_shape = array<i64: 4>
