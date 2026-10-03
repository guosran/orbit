// RUN: mlir-amoeba-opt %s --verify-each \
// RUN:   '--materialize-neura-k-reduction=task-name=Task_16 k-mode=sequential tile-size=2' \
// RUN:   -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

// A compact predicate-form memory-feedback task.  Task_16's done_reads WAR
// token and done_writes state are both consumed by Task_19, so replacing the
// source task must forward both result segments from the final K block.
module {
  func.func @gcn_memory_feedback_lifetime(
      %a: memref<1x1x4xi32> {amoeba.noalias},
      %b: memref<1x1x4xi32> {amoeba.noalias},
      %state: memref<1x1x4xi32> {amoeba.noalias}) -> memref<1x1x4xi32> {
    %k = arith.constant 4 : index
    %zero = arith.constant 0 : i32
    %done_reads, %done_writes = taskflow.task @Task_16
        will_reads(%a, %b, %state : memref<1x1x4xi32>, memref<1x1x4xi32>, memref<1x1x4xi32>)
        will_writes(%state : memref<1x1x4xi32>)
        value_inputs(%k, %zero : index, i32)
        [original_read_memrefs(%a, %b, %state : memref<1x1x4xi32>, memref<1x1x4xi32>, memref<1x1x4xi32>),
         original_write_memrefs(%state : memref<1x1x4xi32>)]
        : (memref<1x1x4xi32>, memref<1x1x4xi32>, memref<1x1x4xi32>,
           memref<1x1x4xi32>, index, i32)
          -> (memref<1x1x4xi32>, memref<1x1x4xi32>) {
      ^bb0(%read_a: memref<1x1x4xi32>, %read_b: memref<1x1x4xi32>,
           %read_state: memref<1x1x4xi32>, %write_state: memref<1x1x4xi32>,
           %bound: index, %value: i32):
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %m = taskflow.counter from %c0 to %c1 step %c1
            attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} : index
        %n = taskflow.counter parent(%m : index) from %c0 to %c1 step %c1
            attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} : index
        %kk = taskflow.counter parent(%n : index) from %c0 to %bound step %c1
            attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32} : index
        neura.kernel inputs(%value, %write_state, %read_a, %read_b, %bound :
            i32, memref<1x1x4xi32>, memref<1x1x4xi32>, memref<1x1x4xi32>, index)
            attributes {accelerator = "neura", dataflow_mode = "predicate"} {
          ^bb0(%input0: !neura.data<i32, i1>,
               %input1: !neura.data<memref<1x1x4xi32>, i1>,
               %input2: !neura.data<memref<1x1x4xi32>, i1>,
               %input3: !neura.data<memref<1x1x4xi32>, i1>,
               %input4: !neura.data<index, i1>):
            %constant = "neura.constant"() <{value = 0 : index}> : () -> !neura.data<index, i1>
            %cm = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 1 : index} -> !neura.data<index, i1>
            %cn = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32, lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 1 : index} -> !neura.data<index, i1>
            %ck = neura.counter attributes {counter_dynamism = "symbol_bound", counter_hierarchy = "leaf", counter_id = 2 : i32, lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = "%input4"} -> !neura.data<index, i1>
            %lhs = neura.load_indexed [%constant, %cm, %ck : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>] {lhs_value = "%input2"} : !neura.data<i32, i1>
            %rhs = neura.load_indexed [%constant, %cm, %ck : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>] {lhs_value = "%input3"} : !neura.data<i32, i1>
            %product = "neura.mul"(%lhs, %rhs) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
            %accumulator = neura.load_indexed [%constant, %cm, %cn : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>] {lhs_value = "%input1"} : !neura.data<i32, i1>
            %sum = "neura.add"(%accumulator, %product) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
            neura.store_indexed %sum to [%constant, %cm, %cn : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>] {rhs_value = "%input1"} : !neura.data<i32, i1>
            neura.yield {yield_type = "void"}
        }
        taskflow.yield done_reads(%read_b : memref<1x1x4xi32>)
                         done_writes(%write_state : memref<1x1x4xi32>)
    }

    %consumer_done = taskflow.task @Task_19
        will_reads(%done_writes : memref<1x1x4xi32>)
        will_writes(%done_reads : memref<1x1x4xi32>)
        [original_read_memrefs(%state : memref<1x1x4xi32>),
         original_write_memrefs(%b : memref<1x1x4xi32>)]
        : (memref<1x1x4xi32>, memref<1x1x4xi32>) -> (memref<1x1x4xi32>) {
      ^bb0(%read: memref<1x1x4xi32>, %write: memref<1x1x4xi32>):
        taskflow.yield done_writes(%write : memref<1x1x4xi32>)
    }
    return %consumer_done : memref<1x1x4xi32>
  }
}

// CHECK-LABEL: func.func @gcn_memory_feedback_lifetime
// CHECK: %[[K0_READ:[A-Za-z0-9_]+]], %[[K0_WRITE:[A-Za-z0-9_]+]] = taskflow.task @Task_16.k.0
// CHECK-NEXT: ^bb0(%[[K0_A:[A-Za-z0-9_]+]]: memref<1x1x4xi32>, %[[K0_B:[A-Za-z0-9_]+]]: memref<1x1x4xi32>, %[[K0_STATE:[A-Za-z0-9_]+]]: memref<1x1x4xi32>, %[[K0_OUTPUT:[A-Za-z0-9_]+]]: memref<1x1x4xi32>,
// CHECK: taskflow.yield done_reads(%[[K0_B]] : memref<1x1x4xi32>) done_writes(%[[K0_OUTPUT]] : memref<1x1x4xi32>)
// CHECK: %[[K1_READ:[A-Za-z0-9_]+]], %[[K1_WRITE:[A-Za-z0-9_]+]] = taskflow.task @Task_16.k.1 will_reads(%arg0, %arg1, %[[K0_WRITE]] : memref<1x1x4xi32>, memref<1x1x4xi32>, memref<1x1x4xi32>)
// CHECK-NEXT: ^bb0(%[[K1_A:[A-Za-z0-9_]+]]: memref<1x1x4xi32>, %[[K1_B:[A-Za-z0-9_]+]]: memref<1x1x4xi32>, %[[K1_STATE:[A-Za-z0-9_]+]]: memref<1x1x4xi32>, %[[K1_OUTPUT:[A-Za-z0-9_]+]]: memref<1x1x4xi32>,
// CHECK: taskflow.yield done_reads(%[[K1_B]] : memref<1x1x4xi32>) done_writes(%[[K1_OUTPUT]] : memref<1x1x4xi32>)
// CHECK-NOT: taskflow.task @Task_16 will_reads
// CHECK: taskflow.task @Task_19 will_reads(%[[K1_WRITE]] : memref<1x1x4xi32>) will_writes(%[[K1_READ]] : memref<1x1x4xi32>)
