// RUN: not mlir-amoeba-opt %s '--fission-task=task-name=Cycle left-nodes=0:1:2' 2>&1 | FileCheck %s
// The shared recurrence is used on both sides of a forward-closed cut. A
// scalar scratch edge does not legalize this reverse loop-carried dependence.
module {
  func.func @cycle(%a: memref<4xi32> {amoeba.noalias}, %y: memref<4xi32> {amoeba.noalias}, %z: memref<4xi32> {amoeba.noalias}) {
    %zero_value = arith.constant 0 : i32
    %out:2 = taskflow.task @Cycle will_reads(%a : memref<4xi32>) will_writes(%y, %z : memref<4xi32>, memref<4xi32>) value_inputs(%zero_value : i32)
      [original_read_memrefs(%a : memref<4xi32>), original_write_memrefs(%y, %z : memref<4xi32>, memref<4xi32>)]
      : (memref<4xi32>, memref<4xi32>, memref<4xi32>, i32) -> (memref<4xi32>, memref<4xi32>) {
    ^bb0(%read: memref<4xi32>, %wy: memref<4xi32>, %wz: memref<4xi32>, %init: i32):
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %four = arith.constant 4 : index
      %counter = taskflow.counter from %zero to %four step %one : index
      %unused = "taskflow.hyperblock"(%counter, %init) <{operandSegmentSizes = array<i32: 1, 1>}> ({
      ^bb0(%i: index, %carry: i32):
        %value = memref.load %read[%i] : memref<4xi32>
        %next = arith.addi %carry, %value : i32
        memref.store %next, %wy[%i] : memref<4xi32>
        %other = arith.muli %carry, %carry : i32
        memref.store %other, %wz[%i] : memref<4xi32>
        "taskflow.hyperblock.yield"(%next, %next) <{operandSegmentSizes = array<i32: 1, 1>}> : (i32, i32) -> ()
      }) : (index, i32) -> i32
      taskflow.yield done_writes(%wy, %wz : memref<4xi32>, memref<4xi32>)
    }
    return
  }
}
// CHECK: fission partition crosses a loop-carried dependence
