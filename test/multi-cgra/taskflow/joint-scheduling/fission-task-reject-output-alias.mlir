// RUN: not mlir-amoeba-opt %s '--fission-task=task-name=Branches left-nodes=0:1:2' 2>&1 | FileCheck %s
module {
  func.func @branches(%a: memref<3x2xi32> {amoeba.noalias},
                      %b: memref<3x2xi32> {amoeba.noalias},
                      %y: memref<3x2xi32> {amoeba.noalias},
                      %z: memref<3x2xi32> {amoeba.noalias}) {
    %out:2 = taskflow.task @Branches
      will_reads(%a, %b : memref<3x2xi32>, memref<3x2xi32>)
      will_writes(%y, %y : memref<3x2xi32>, memref<3x2xi32>)
      [original_read_memrefs(%a, %b : memref<3x2xi32>, memref<3x2xi32>),
       original_write_memrefs(%y, %y : memref<3x2xi32>, memref<3x2xi32>)]
      : (memref<3x2xi32>, memref<3x2xi32>, memref<3x2xi32>, memref<3x2xi32>) -> (memref<3x2xi32>, memref<3x2xi32>) {
      ^bb0(%ra: memref<3x2xi32>, %rb: memref<3x2xi32>, %wy: memref<3x2xi32>, %wz: memref<3x2xi32>):
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %two = arith.constant 2 : index
        %three = arith.constant 3 : index
        %i = taskflow.counter from %zero to %two step %one : index
        %j = "taskflow.counter"(%i, %zero, %three, %one) : (index, index, index, index) -> index
        "taskflow.hyperblock"(%i, %j) <{operandSegmentSizes = array<i32: 2, 0>}> ({
        ^bb0(%x: index, %v: index):
          %av = memref.load %ra[%v, %x] : memref<3x2xi32>
          %twice = arith.addi %av, %av : i32
          memref.store %twice, %wy[%v, %x] : memref<3x2xi32>
          %bv = memref.load %rb[%v, %x] : memref<3x2xi32>
          %square = arith.muli %bv, %bv : i32
          memref.store %square, %wz[%v, %x] : memref<3x2xi32>
          "taskflow.hyperblock.yield"() : () -> ()
        }) : (index, index) -> ()
        taskflow.yield done_writes(%wy, %wz : memref<3x2xi32>, memref<3x2xi32>)
    }
    return
  }
}
// CHECK: fission requires pairwise disjoint output storage
