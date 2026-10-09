// RUN: python3 %S/check-static-active-transfer-payload.py mlir-amoeba-opt %s %t
// Static and dynamic capacities must honor the same authenticated contract.
module {
  func.func @active_payload(%data: memref<4xi32> {
      amoeba.logical_transfer_shape = array<i64: 4>, amoeba.noalias
    }) attributes {
      amoeba.input0_caller_evidence_path = "fixture-caller.mlir",
      amoeba.input0_caller_function = "main",
      amoeba.input0_caller_noalias_proof_mode = "prepared-external-caller",
      amoeba.input0_caller_noalias_proven,
      amoeba.input0_caller_prepared_input_path = "fixture-prepared.mlir"
    } {
    %done = taskflow.task @Producer
      will_writes(%data : memref<4xi32>)
      [original_write_memrefs(%data : memref<4xi32>)]
      : (memref<4xi32>) -> memref<4xi32> {
    ^bb0(%write: memref<4xi32>):
      neura.kernel inputs(%write : memref<4xi32>)
        attributes {accelerator = "neura", dataflow_mode = "predicate"} {
      ^bb0(%kernel_write: memref<4xi32>):
        %c0 = arith.constant 0 : index
        %one = arith.constant 1 : i32
        memref.store %one, %kernel_write[%c0] : memref<4xi32>
        neura.yield {yield_type = "void"}
      }
      taskflow.yield done_writes(%write : memref<4xi32>)
    }
    %read_done = taskflow.task @Consumer
      will_reads(%done : memref<4xi32>)
      [original_read_memrefs(%data : memref<4xi32>)]
      : (memref<4xi32>) -> memref<4xi32> {
    ^bb0(%read: memref<4xi32>):
      neura.kernel inputs(%read : memref<4xi32>)
        attributes {accelerator = "neura", dataflow_mode = "predicate"} {
      ^bb0(%kernel_read: memref<4xi32>):
        %c0 = arith.constant 0 : index
        %value = memref.load %kernel_read[%c0] : memref<4xi32>
        neura.yield {yield_type = "void"}
      }
      taskflow.yield done_reads(%read : memref<4xi32>)
    }
    return
  }
}
