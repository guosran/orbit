// RUN: mlir-amoeba-opt %s --verify-each \
// RUN:   '--prove-static-active-transfer-shapes=function=active_boxes arguments=1,2 report-output=%t.json' \
// RUN:   -o %t.mlir
// RUN: FileCheck %s --check-prefix=ACTIVE --input-file=%t.mlir
// RUN: python3 -c 'import json,sys; p=json.load(open(sys.argv[1]))["proofs"]; assert [(x["status"],x["active_shape"],x["capacity_shape"]) for x in p] == [("proven",[4,5,5],[4,256,256]),("proven",[12,5,16],[12,256,16])]' %t.json
// RUN: mlir-amoeba-opt %s --verify-each \
// RUN:   '--prove-static-active-transfer-shapes=function=generic_memref arguments=1 report-output=%t-memref.json' \
// RUN:   -o %t-memref.mlir
// RUN: python3 -c 'import json,sys; p=json.load(open(sys.argv[1]))["proofs"][0]; assert p["status"] == "proven" and p["active_shape"] == [1] and p["capacity_shape"] == [4]' %t-memref.json

// The input-0 caller facts are stand-ins for the imported caller proof. The
// counters below still have to agree between Taskflow and Neura before either
// active box is published.
module {
  func.func @active_boxes(
      %bound: index,
      %weights: memref<?x256x256xi32> {
        amoeba.logical_transfer_shape = array<i64: 4, 256, 256>, amoeba.noalias
      },
      %states: memref<?x256x16xi32> {
        amoeba.logical_transfer_shape = array<i64: 12, 256, 16>, amoeba.noalias
      }) attributes {
        amoeba.input0_caller_evidence_path = "fixture-caller.mlir",
        amoeba.input0_caller_function = "main",
        amoeba.input0_caller_noalias_proof_mode = "prepared-external-caller",
        amoeba.input0_caller_noalias_proven,
        amoeba.input0_caller_prepared_input_path = "fixture-prepared.mlir",
        amoeba.input0_caller_static_bound = 5 : i64,
        amoeba.static_bound.arg.0 = 5 : i64
      } {
    %weights_read_done, %weights_done = taskflow.task @Weights
        will_reads(%weights : memref<?x256x256xi32>)
        will_writes(%weights : memref<?x256x256xi32>)
        value_inputs(%bound : index)
        [original_read_memrefs(%weights : memref<?x256x256xi32>),
         original_write_memrefs(%weights : memref<?x256x256xi32>)]
        : (memref<?x256x256xi32>, memref<?x256x256xi32>, index)
          -> (memref<?x256x256xi32>, memref<?x256x256xi32>) {
      ^bb0(%read_weights: memref<?x256x256xi32>,
           %write_weights: memref<?x256x256xi32>, %task_bound: index):
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %layer = taskflow.counter from %c0 to %c4 step %c1
            attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} : index
        %row = taskflow.counter parent(%layer : index) from %c0 to %task_bound step %c1
            attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} : index
        %column = taskflow.counter parent(%row : index) from %c0 to %task_bound step %c1
            attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32} : index
        neura.kernel inputs(%read_weights, %write_weights, %task_bound : memref<?x256x256xi32>, memref<?x256x256xi32>, index)
            attributes {accelerator = "neura", dataflow_mode = "predicate"} {
          ^bb0(%input0: !neura.data<memref<?x256x256xi32>, i1>,
               %input1: !neura.data<memref<?x256x256xi32>, i1>,
               %input2: !neura.data<index, i1>):
            %i = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 4 : index} -> !neura.data<index, i1>
            %j = neura.counter attributes {counter_dynamism = "symbol_bound", counter_hierarchy = "relay", counter_id = 1 : i32, lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = "%input2"} -> !neura.data<index, i1>
            %k = neura.counter attributes {counter_dynamism = "symbol_bound", counter_hierarchy = "leaf", counter_id = 2 : i32, lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = "%input2"} -> !neura.data<index, i1>
            %value = neura.load_indexed [%i, %j, %k : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>] {lhs_value = "%input0"} : !neura.data<i32, i1>
            neura.store_indexed %value to [%i, %j, %k : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>] {rhs_value = "%input1"} : !neura.data<i32, i1>
            neura.yield {yield_type = "void"}
        }
        taskflow.yield done_reads(%read_weights : memref<?x256x256xi32>)
                         done_writes(%write_weights : memref<?x256x256xi32>)
    }

    %states_read_done, %states_done = taskflow.task @States
        will_reads(%states : memref<?x256x16xi32>)
        will_writes(%states : memref<?x256x16xi32>)
        value_inputs(%bound : index)
        [original_read_memrefs(%states : memref<?x256x16xi32>),
         original_write_memrefs(%states : memref<?x256x16xi32>)]
        : (memref<?x256x16xi32>, memref<?x256x16xi32>, index)
          -> (memref<?x256x16xi32>, memref<?x256x16xi32>) {
      ^bb0(%read_states: memref<?x256x16xi32>,
           %write_states: memref<?x256x16xi32>, %task_bound: index):
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c12 = arith.constant 12 : index
        %c16 = arith.constant 16 : index
        %layer = taskflow.counter from %c0 to %c12 step %c1
            attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} : index
        %row = taskflow.counter parent(%layer : index) from %c0 to %task_bound step %c1
            attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} : index
        %column = taskflow.counter parent(%row : index) from %c0 to %c16 step %c1
            attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32} : index
        neura.kernel inputs(%read_states, %write_states, %task_bound : memref<?x256x16xi32>, memref<?x256x16xi32>, index)
            attributes {accelerator = "neura", dataflow_mode = "predicate"} {
          ^bb0(%input0: !neura.data<memref<?x256x16xi32>, i1>,
               %input1: !neura.data<memref<?x256x16xi32>, i1>,
               %input2: !neura.data<index, i1>):
            %i = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 12 : index} -> !neura.data<index, i1>
            %j = neura.counter attributes {counter_dynamism = "symbol_bound", counter_hierarchy = "relay", counter_id = 1 : i32, lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = "%input2"} -> !neura.data<index, i1>
            %k = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32, lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 16 : index} -> !neura.data<index, i1>
            %value = neura.load_indexed [%i, %j, %k : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>] {lhs_value = "%input0"} : !neura.data<i32, i1>
            neura.store_indexed %value to [%i, %j, %k : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>] {rhs_value = "%input1"} : !neura.data<i32, i1>
            neura.yield {yield_type = "void"}
        }
        taskflow.yield done_reads(%read_states : memref<?x256x16xi32>)
                         done_writes(%write_states : memref<?x256x16xi32>)
    }
    return
  }

  func.func @generic_memref(
      %bound: index,
      %data: memref<?xi32> {
        amoeba.logical_transfer_shape = array<i64: 4>, amoeba.noalias
      }) attributes {
        amoeba.input0_caller_evidence_path = "fixture-caller.mlir",
        amoeba.input0_caller_function = "main",
        amoeba.input0_caller_noalias_proof_mode = "prepared-external-caller",
        amoeba.input0_caller_noalias_proven,
        amoeba.input0_caller_prepared_input_path = "fixture-prepared.mlir",
        amoeba.input0_caller_static_bound = 5 : i64,
        amoeba.static_bound.arg.0 = 5 : i64
      } {
    %c0 = arith.constant 0 : index
    %value = memref.load %data[%c0] : memref<?xi32>
    memref.store %value, %data[%c0] : memref<?xi32>
    return
  }
}

// ACTIVE: amoeba.active_transfer_capacity_shape = array<i64: 4, 256, 256>
// ACTIVE: amoeba.active_transfer_shape = array<i64: 4, 5, 5>
// ACTIVE: amoeba.active_transfer_capacity_shape = array<i64: 12, 256, 16>
// ACTIVE: amoeba.active_transfer_shape = array<i64: 12, 5, 16>
