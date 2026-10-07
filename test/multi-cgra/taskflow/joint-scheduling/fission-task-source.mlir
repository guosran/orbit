// RUN: mlir-amoeba-opt %s '--pass-pipeline=builtin.module(func.func(fission-task{task-name=Task_0 left-nodes=0:1}))' --verify-each | FileCheck %s
// RUN: mlir-amoeba-opt %s '--pass-pipeline=builtin.module(func.func(fission-task{task-name=Task_0 left-nodes=1}))' --verify-diagnostics

// This source-level fixture exercises static trip-count recovery through a
// task value input and an index cast. The caller-owned bound is then retained
// by both stages, and the scalar DFG edge is transported through scratch.
module {
  func.func @source_fission(
      %bound_arg: i32,
      %input: memref<4xi32> {amoeba.noalias},
      %output: memref<4xi32> {amoeba.noalias})
      attributes {amoeba.static_bound.arg.0 = 4 : i64} {
    %bound = arith.index_cast %bound_arg : i32 to index
    // expected-error @+1 {{left-nodes must be predecessor closed}}
    %done_writes = taskflow.task @Task_0
        will_reads(%input : memref<4xi32>)
        will_writes(%output : memref<4xi32>)
        value_inputs(%bound : index)
        [original_read_memrefs(%input : memref<4xi32>),
         original_write_memrefs(%output : memref<4xi32>)]
        {dlp_replicable = true, runtime_managable = true}
        : (memref<4xi32>, memref<4xi32>, index) -> (memref<4xi32>) {
      ^bb0(%read: memref<4xi32>, %write: memref<4xi32>, %task_bound: index):
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %counter = taskflow.counter from %zero to %task_bound step %one
            : index
        "taskflow.hyperblock"(%counter)
            <{operandSegmentSizes = array<i32: 1, 0>}> ({
          ^bb0(%i: index):
            %value = memref.load %read[%i] : memref<4xi32>
            %twice = arith.addi %value, %value : i32
            %square = arith.muli %twice, %twice : i32
            memref.store %square, %write[%i] : memref<4xi32>
            "taskflow.hyperblock.yield"() : () -> ()
        }) : (index) -> ()
        taskflow.yield done_writes(%write : memref<4xi32>)
    }
    func.return
  }
}

// CHECK-LABEL: func.func @source_fission
// CHECK: %[[SCRATCH:.*]] = memref.alloc() : memref<4xi32>
// CHECK: %[[DONE:.*]] = taskflow.task @Task_0.split.0
// CHECK-SAME: will_reads(%[[INPUT:[A-Za-z0-9_]+]] : memref<4xi32>)
// CHECK-SAME: will_writes(%[[SCRATCH]] : memref<4xi32>)
// CHECK-SAME: amoeba.fission.part_index = 0 : i64
// CHECK-SAME: dlp_replicable = true
// CHECK-SAME: runtime_managable = true
// CHECK: taskflow.counter
// CHECK: "taskflow.hyperblock"
// CHECK: %[[PREFIX_VALUE:.*]] = arith.addi
// CHECK: memref.store %[[PREFIX_VALUE]], %{{.*}}[%{{.*}}] : memref<4xi32>
// CHECK: taskflow.task @Task_0.split.1
// CHECK-SAME: will_reads(%[[DONE]], %[[INPUT]] : memref<4xi32>, memref<4xi32>)
// CHECK-SAME: original_read_memrefs(%[[SCRATCH]], %[[INPUT]] : memref<4xi32>, memref<4xi32>)
// CHECK-SAME: amoeba.fission.part_index = 1 : i64
// CHECK: taskflow.counter
// CHECK: "taskflow.hyperblock"
// CHECK: %[[CUT_VALUE:.*]] = memref.load %{{.*}}[%{{.*}}] : memref<4xi32>
// CHECK: %[[SQUARE:.*]] = arith.muli %[[CUT_VALUE]], %[[CUT_VALUE]] : i32
// CHECK: memref.store %[[SQUARE]], %{{.*}}[%{{.*}}] : memref<4xi32>
