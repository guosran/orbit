// RUN: mlir-amoeba-opt %s '--fission-task=task-name=ReductionBranches left-nodes=0:1:2:3' --verify-each | FileCheck %s
// Two recurrence cycles and their output initialization are independently
// owned. The shared load result is transported through a real cut tensor.
module {
  func.func @reductions(%a: memref<4xi32> {amoeba.noalias}, %y: memref<4xi32> {amoeba.noalias}, %z: memref<4xi32> {amoeba.noalias}) {
    %zero_value = arith.constant 0 : i32
    %out:2 = taskflow.task @ReductionBranches will_reads(%a : memref<4xi32>) will_writes(%y, %z : memref<4xi32>, memref<4xi32>) value_inputs(%zero_value : i32)
      [original_read_memrefs(%a : memref<4xi32>), original_write_memrefs(%y, %z : memref<4xi32>, memref<4xi32>)]
      : (memref<4xi32>, memref<4xi32>, memref<4xi32>, i32) -> (memref<4xi32>, memref<4xi32>) {
    ^bb0(%read: memref<4xi32>, %wy: memref<4xi32>, %wz: memref<4xi32>, %init: i32):
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %four = arith.constant 4 : index
      %counter = taskflow.counter from %zero to %four step %one : index
      %unused:2 = "taskflow.hyperblock"(%counter, %init, %init) <{operandSegmentSizes = array<i32: 1, 2>}> ({
      ^bb0(%i: index, %carry_y: i32, %carry_z: i32):
        %first = arith.cmpi eq, %i, %zero : index
        scf.if %first {
          memref.store %init, %wy[%i] : memref<4xi32>
          memref.store %init, %wz[%i] : memref<4xi32>
        }
        %value = memref.load %read[%i] : memref<4xi32>
        %square = arith.muli %value, %value : i32
        %next_y = arith.addi %carry_y, %square : i32
        memref.store %next_y, %wy[%i] : memref<4xi32>
        %next_z = arith.addi %carry_z, %value : i32
        memref.store %next_z, %wz[%i] : memref<4xi32>
        "taskflow.hyperblock.yield"(%next_y, %next_z, %next_y, %next_z) <{operandSegmentSizes = array<i32: 2, 2>}> : (i32, i32, i32, i32) -> ()
      }) : (index, i32, i32) -> (i32, i32)
      taskflow.yield done_writes(%wy, %wz : memref<4xi32>, memref<4xi32>)
    }
    return
  }
}
// CHECK: memref.alloc() : memref<4xi32>
// CHECK: taskflow.task @ReductionBranches.split.0
// CHECK: array<i32: 1, 1>
// CHECK: scf.if
// CHECK: memref.store
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: taskflow.task @ReductionBranches.split.1
// CHECK: array<i32: 1, 1>
// CHECK: scf.if
// CHECK: memref.store
// CHECK: memref.load
// CHECK: arith.addi
