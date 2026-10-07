// Focused check for safe common-read elimination during post-Neura sibling fusion.
// RUN: python3 %S/run-neura-sibling-load-dedup-checks.py %s mlir-amoeba-opt

module {
  func.func @sibling_fusion(
      %input: memref<4x5xi32> {amoeba.noalias},
      %out0: memref<4x5xi32> {amoeba.noalias},
      %out1: memref<4x5xi32> {amoeba.noalias},
      %bias: i32) -> (memref<4x5xi32>, memref<4x5xi32>) {
    %done0 = taskflow.task @sibling_a
        will_reads(%input : memref<4x5xi32>)
        will_writes(%out0 : memref<4x5xi32>)
        value_inputs(%bias : i32)
        [original_read_memrefs(%input : memref<4x5xi32>),
         original_write_memrefs(%out0 : memref<4x5xi32>)]
        : (memref<4x5xi32>, memref<4x5xi32>, i32)
       -> (memref<4x5xi32>) {
    ^bb0(%read: memref<4x5xi32>, %write: memref<4x5xi32>, %value: i32):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c1 = arith.constant 1 : index
      %i = taskflow.counter from %c0 to %c4 step %c1 : index
      %j = taskflow.counter parent(%i : index) from %c0 to %c5 step %c1 : index
      "taskflow.hyperblock"(%i, %j) <{operandSegmentSizes = array<i32: 2, 0>}> ({
      ^bb0(%ii: index, %jj: index):
        %element = memref.load %read[%ii, %jj] : memref<4x5xi32>
        %sum = arith.addi %element, %value : i32
        memref.store %sum, %write[%ii, %jj] : memref<4x5xi32>
        taskflow.hyperblock.yield
      }) : (index, index) -> ()
      taskflow.yield done_writes(%write : memref<4x5xi32>)
    }
    %done1 = taskflow.task @sibling_b
        will_reads(%input : memref<4x5xi32>)
        will_writes(%out1 : memref<4x5xi32>)
        value_inputs(%bias : i32)
        [original_read_memrefs(%input : memref<4x5xi32>),
         original_write_memrefs(%out1 : memref<4x5xi32>)]
        : (memref<4x5xi32>, memref<4x5xi32>, i32)
       -> (memref<4x5xi32>) {
    ^bb0(%read: memref<4x5xi32>, %write: memref<4x5xi32>, %value: i32):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c1 = arith.constant 1 : index
      %i = taskflow.counter from %c0 to %c4 step %c1 : index
      %j = taskflow.counter parent(%i : index) from %c0 to %c5 step %c1 : index
      "taskflow.hyperblock"(%i, %j) <{operandSegmentSizes = array<i32: 2, 0>}> ({
      ^bb0(%ii: index, %jj: index):
        %element = memref.load %read[%ii, %jj] : memref<4x5xi32>
        %sum = arith.subi %element, %value : i32
        memref.store %sum, %write[%ii, %jj] : memref<4x5xi32>
        taskflow.hyperblock.yield
      }) : (index, index) -> ()
      taskflow.yield done_writes(%write : memref<4x5xi32>)
    }
    return %done0, %done1 : memref<4x5xi32>, memref<4x5xi32>
  }
}
