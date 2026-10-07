// RUN: not mlir-amoeba-opt %s --verify-each \
// RUN:   --pass-pipeline='builtin.module(extract-joint-task-graph-facts{function=payload_tamper output=%t.json fact-only=true})' \
// RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=PAYLOAD

// A consumer cannot replace the payload independently derived from a proven
// producer/output and consumer/input rectangle.
module {
  func.func @payload_tamper(
      %state: memref<2x2xi32> {amoeba.noalias},
      %output: memref<2x2xi32> {amoeba.noalias}) {
    %producer = taskflow.task @Producer
        will_reads(%state : memref<2x2xi32>)
        will_writes(%state : memref<2x2xi32>)
        [original_read_memrefs(%state : memref<2x2xi32>),
         original_write_memrefs(%state : memref<2x2xi32>)]
        {amoeba.tiling.output_region_lowers = [array<i64: 0, 0>],
         amoeba.tiling.output_region_uppers = [array<i64: 1, 2>]}
        : (memref<2x2xi32>, memref<2x2xi32>) -> memref<2x2xi32> {
      ^bb0(%read: memref<2x2xi32>, %write: memref<2x2xi32>):
        taskflow.yield done_writes(%write : memref<2x2xi32>)
    }

    %consumer = taskflow.task @Consumer
        will_reads(%producer : memref<2x2xi32>)
        will_writes(%output : memref<2x2xi32>)
        [original_read_memrefs(%state : memref<2x2xi32>),
         original_write_memrefs(%output : memref<2x2xi32>)]
        {amoeba.tiling.input_region_lowers = [array<i64: 0, 0>],
         amoeba.tiling.input_region_uppers = [array<i64: 1, 2>],
         amoeba.tiling.input_region_reasons = ["in_place_output_coordinate_proof"],
         amoeba.edge_payload_bits = array<i64: 63>}
        : (memref<2x2xi32>, memref<2x2xi32>) -> memref<2x2xi32> {
      ^bb0(%read: memref<2x2xi32>, %write: memref<2x2xi32>):
        taskflow.yield done_writes(%write : memref<2x2xi32>)
    }
    return
  }
}

// PAYLOAD: amoeba.edge_payload_bits disagrees with the proven transferred region
