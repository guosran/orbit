// RUN: mlir-amoeba-opt %s --pass-pipeline='builtin.module(convert-affine-to-taskflow,func.func(construct-hyperblock-from-task),classify-task-and-counter,bind-source-iteration-domain)' --verify-each | FileCheck %s

// Both task bodies and their iteration domains come from affine source loops.
// The four separate allocations give the two loops independently provable
// storage roots without asserted noalias or source-domain attributes.
module {
  func.func @fission_rebase_disjoint_history() {
    %fission_input = memref.alloc() : memref<4xi32>
    %fission_output = memref.alloc() : memref<4xi32>
    %ordinary_input = memref.alloc() : memref<4xi32>
    %ordinary_output = memref.alloc() : memref<4xi32>

    affine.for %i = 0 to 4 {
      %value = affine.load %fission_input[%i] : memref<4xi32>
      %twice = arith.addi %value, %value : i32
      %square = arith.muli %twice, %twice : i32
      affine.store %square, %fission_output[%i] : memref<4xi32>
    }

    affine.for %i = 0 to 4 {
      %value = affine.load %ordinary_input[%i] : memref<4xi32>
      %one = arith.constant 1 : i32
      %next = arith.addi %value, %one : i32
      affine.store %next, %ordinary_output[%i] : memref<4xi32>
    }
    func.return
  }
}

// CHECK-LABEL: func.func @fission_rebase_disjoint_history
// CHECK: taskflow.task @Task_0
// CHECK: amoeba.source_iteration_domain
// CHECK: complete = true
// CHECK: source_multiplicity = 4 : i64
// CHECK: taskflow.task @Task_1
// CHECK: amoeba.source_iteration_domain
// CHECK: complete = true
// CHECK: source_multiplicity = 4 : i64
