// RUN: mlir-amoeba-opt %s '--pass-pipeline=builtin.module(func.func(fission-task{task-name=FissionTask left-nodes=0:1}))' --verify-each | FileCheck %s

// The source-owned fission target and an unrelated task have disjoint memory
// footprints. Joint-neighborhood integration probes use ordinary actions on
// OrdinaryTask before fissioning FissionTask, then require exact replay and
// scoring of the resulting complete history.
module {
  func.func @fission_rebase_disjoint_history(
      %bound_arg: i32,
      %fission_input: memref<4xi32> {amoeba.noalias},
      %fission_output: memref<4xi32> {amoeba.noalias},
      %ordinary_input: memref<4xi32> {amoeba.noalias},
      %ordinary_output: memref<4xi32> {amoeba.noalias})
      attributes {amoeba.static_bound.arg.0 = 4 : i64} {
    %bound = arith.index_cast %bound_arg : i32 to index
    %fission_done = taskflow.task @FissionTask
        will_reads(%fission_input : memref<4xi32>)
        will_writes(%fission_output : memref<4xi32>)
        value_inputs(%bound : index)
        [original_read_memrefs(%fission_input : memref<4xi32>),
         original_write_memrefs(%fission_output : memref<4xi32>)]
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
    %ordinary_done = taskflow.task @OrdinaryTask
        will_reads(%ordinary_input : memref<4xi32>)
        will_writes(%ordinary_output : memref<4xi32>)
        value_inputs(%bound : index)
        [original_read_memrefs(%ordinary_input : memref<4xi32>),
         original_write_memrefs(%ordinary_output : memref<4xi32>)]
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
            %one_i32 = arith.constant 1 : i32
            %next = arith.addi %value, %one_i32 : i32
            memref.store %next, %write[%i] : memref<4xi32>
            "taskflow.hyperblock.yield"() : () -> ()
        }) : (index) -> ()
        taskflow.yield done_writes(%write : memref<4xi32>)
    }
    func.return
  }
}

// CHECK-LABEL: func.func @fission_rebase_disjoint_history
// CHECK: taskflow.task @FissionTask.split.0
// CHECK: taskflow.task @FissionTask.split.1
// CHECK: taskflow.task @OrdinaryTask
