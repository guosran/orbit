// RUN: mlir-amoeba-opt %s --verify-each \
// RUN:   '--materialize-neura-joint-rewrite=task-name=Task_valid tile-axis=0 tile-factor=2 fusion-mode=none' \
// RUN:   -o %t.valid.mlir
// RUN: FileCheck %s --check-prefix=VALID --input-file=%t.valid.mlir
// RUN: not mlir-amoeba-opt %s --verify-each \
// RUN:   '--materialize-neura-joint-rewrite=task-name=Task_foreign tile-axis=0 tile-factor=2 fusion-mode=none' \
// RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=FOREIGN
// RUN: not mlir-amoeba-opt %s --verify-each \
// RUN:   '--materialize-neura-joint-rewrite=task-name=Task_metadata tile-axis=0 tile-factor=2 fusion-mode=none' \
// RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=METADATA
// RUN: not mlir-amoeba-opt %s --verify-each \
// RUN:   '--materialize-neura-joint-rewrite=task-name=Task_phi tile-axis=0 tile-factor=2 fusion-mode=none' \
// RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=PHI

// In-place rank-2 tiling must follow the output storage through the canonical
// constant/data_mov/grant_predicate/data_mov chain, and accept coordinates
// whose two phi arms both name the same counter.  A foreign write, a folded
// base that contradicts its SSA base, and a phi joining different counters
// must stay fail-closed.
module {
  func.func @in_place_mn_tiling_proofs(
      %state: memref<2x2xi32> {amoeba.noalias},
      %foreign: memref<2x2xi32> {amoeba.noalias}, %value: i32) {
    %valid = taskflow.task @Task_valid
        will_reads(%state, %foreign : memref<2x2xi32>, memref<2x2xi32>)
        will_writes(%state : memref<2x2xi32>)
        value_inputs(%value : i32)
        [original_read_memrefs(%state, %foreign : memref<2x2xi32>, memref<2x2xi32>),
         original_write_memrefs(%state : memref<2x2xi32>)]
        : (memref<2x2xi32>, memref<2x2xi32>, memref<2x2xi32>, i32) -> memref<2x2xi32> {
      ^bb0(%read: memref<2x2xi32>, %other: memref<2x2xi32>,
           %write: memref<2x2xi32>, %scalar: i32):
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %two = arith.constant 2 : index
        %row = taskflow.counter from %zero to %two step %one
            attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} : index
        %column = taskflow.counter parent(%row : index) from %zero to %two step %one
            attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} : index
        neura.kernel inputs(%scalar, %write, %other : i32, memref<2x2xi32>, memref<2x2xi32>)
            attributes {accelerator = "neura", dataflow_mode = "predicate"} {
          ^bb0(%input0: !neura.data<i32, i1>,
               %input1: !neura.data<memref<2x2xi32>, i1>,
               %input2: !neura.data<memref<2x2xi32>, i1>):
            %row_index = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 2 : index} -> !neura.data<index, i1>
            %column_index = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32, lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 2 : index} -> !neura.data<index, i1>
            %predicate = "neura.icmp"(%row_index) <{cmpType = "eq"}> {rhs_value = 0 : index} : (!neura.data<index, i1>) -> !neura.data<i1, i1>
            %predicate_not = "neura.not"(%predicate) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
            %row_true = "neura.grant_predicate"(%row_index, %predicate) : (!neura.data<index, i1>, !neura.data<i1, i1>) -> !neura.data<index, i1>
            %row_false = "neura.grant_predicate"(%row_index, %predicate_not) : (!neura.data<index, i1>, !neura.data<i1, i1>) -> !neura.data<index, i1>
            %column_true = "neura.grant_predicate"(%column_index, %predicate) : (!neura.data<index, i1>, !neura.data<i1, i1>) -> !neura.data<index, i1>
            %column_false = "neura.grant_predicate"(%column_index, %predicate_not) : (!neura.data<index, i1>, !neura.data<i1, i1>) -> !neura.data<index, i1>
            %row_phi = "neura.phi"(%row_true, %row_false) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
            %column_phi = "neura.phi"(%column_true, %column_false) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
            %old = neura.load_indexed [%row_phi, %column_phi : !neura.data<index, i1>, !neura.data<index, i1>] {lhs_value = "%input1"} : !neura.data<i32, i1>
            %other_old = neura.load_indexed [%row_phi, %column_phi : !neura.data<index, i1>, !neura.data<index, i1>] {lhs_value = "%input2"} : !neura.data<i32, i1>
            %base0 = "neura.constant"() <{value = "%input1"}> : () -> !neura.data<memref<2x2xi32>, i1>
            %base1 = "neura.data_mov"(%base0) : (!neura.data<memref<2x2xi32>, i1>) -> !neura.data<memref<2x2xi32>, i1>
            %base2 = "neura.grant_predicate"(%base1, %predicate) : (!neura.data<memref<2x2xi32>, i1>, !neura.data<i1, i1>) -> !neura.data<memref<2x2xi32>, i1>
            %base3 = "neura.data_mov"(%base2) : (!neura.data<memref<2x2xi32>, i1>) -> !neura.data<memref<2x2xi32>, i1>
            "neura.store_indexed"(%base3, %row_phi, %column_phi) <{operandSegmentSizes = array<i32: 0, 1, 2>}> {lhs_value = "%input0"} : (!neura.data<memref<2x2xi32>, i1>, !neura.data<index, i1>, !neura.data<index, i1>) -> ()
            neura.yield {yield_type = "void"}
        }
        taskflow.yield done_writes(%write : memref<2x2xi32>)
    }

    %foreign_done = taskflow.task @Task_foreign
        will_reads(%state, %foreign : memref<2x2xi32>, memref<2x2xi32>)
        will_writes(%state : memref<2x2xi32>)
        value_inputs(%value : i32)
        [original_read_memrefs(%state, %foreign : memref<2x2xi32>, memref<2x2xi32>),
         original_write_memrefs(%state : memref<2x2xi32>)]
        : (memref<2x2xi32>, memref<2x2xi32>, memref<2x2xi32>, i32) -> memref<2x2xi32> {
      ^bb0(%read: memref<2x2xi32>, %other: memref<2x2xi32>,
           %write: memref<2x2xi32>, %scalar: i32):
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %two = arith.constant 2 : index
        %row = taskflow.counter from %zero to %two step %one
            attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} : index
        %column = taskflow.counter parent(%row : index) from %zero to %two step %one
            attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} : index
        neura.kernel inputs(%scalar, %write, %other : i32, memref<2x2xi32>, memref<2x2xi32>)
            attributes {accelerator = "neura", dataflow_mode = "predicate"} {
          ^bb0(%input0: !neura.data<i32, i1>,
               %input1: !neura.data<memref<2x2xi32>, i1>,
               %input2: !neura.data<memref<2x2xi32>, i1>):
            %row_index = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 2 : index} -> !neura.data<index, i1>
            %column_index = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32, lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 2 : index} -> !neura.data<index, i1>
            %old = neura.load_indexed [%row_index, %column_index : !neura.data<index, i1>, !neura.data<index, i1>] {lhs_value = "%input1"} : !neura.data<i32, i1>
            neura.store_indexed %input0 to [%row_index, %column_index : !neura.data<index, i1>, !neura.data<index, i1>] {rhs_value = "%input1"} : !neura.data<i32, i1>
            %other0 = "neura.constant"() <{value = "%input2"}> : () -> !neura.data<memref<2x2xi32>, i1>
            %other1 = "neura.data_mov"(%other0) : (!neura.data<memref<2x2xi32>, i1>) -> !neura.data<memref<2x2xi32>, i1>
            %predicate = "neura.icmp"(%row_index) <{cmpType = "eq"}> {rhs_value = 0 : index} : (!neura.data<index, i1>) -> !neura.data<i1, i1>
            %other2 = "neura.grant_predicate"(%other1, %predicate) : (!neura.data<memref<2x2xi32>, i1>, !neura.data<i1, i1>) -> !neura.data<memref<2x2xi32>, i1>
            %other3 = "neura.data_mov"(%other2) : (!neura.data<memref<2x2xi32>, i1>) -> !neura.data<memref<2x2xi32>, i1>
            "neura.store_indexed"(%other3, %row_index, %column_index) <{operandSegmentSizes = array<i32: 0, 1, 2>}> {lhs_value = "%input0"} : (!neura.data<memref<2x2xi32>, i1>, !neura.data<index, i1>, !neura.data<index, i1>) -> ()
            neura.yield {yield_type = "void"}
        }
        taskflow.yield done_writes(%write : memref<2x2xi32>)
    }

    %metadata_done = taskflow.task @Task_metadata
        will_reads(%state, %foreign : memref<2x2xi32>, memref<2x2xi32>)
        will_writes(%state : memref<2x2xi32>)
        value_inputs(%value : i32)
        [original_read_memrefs(%state, %foreign : memref<2x2xi32>, memref<2x2xi32>),
         original_write_memrefs(%state : memref<2x2xi32>)]
        : (memref<2x2xi32>, memref<2x2xi32>, memref<2x2xi32>, i32) -> memref<2x2xi32> {
      ^bb0(%read: memref<2x2xi32>, %other: memref<2x2xi32>,
           %write: memref<2x2xi32>, %scalar: i32):
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %two = arith.constant 2 : index
        %row = taskflow.counter from %zero to %two step %one
            attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} : index
        %column = taskflow.counter parent(%row : index) from %zero to %two step %one
            attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} : index
        neura.kernel inputs(%scalar, %write, %other : i32, memref<2x2xi32>, memref<2x2xi32>)
            attributes {accelerator = "neura", dataflow_mode = "predicate"} {
          ^bb0(%input0: !neura.data<i32, i1>,
               %input1: !neura.data<memref<2x2xi32>, i1>,
               %input2: !neura.data<memref<2x2xi32>, i1>):
            %row_index = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 2 : index} -> !neura.data<index, i1>
            %column_index = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32, lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 2 : index} -> !neura.data<index, i1>
            %old = neura.load_indexed [%row_index, %column_index : !neura.data<index, i1>, !neura.data<index, i1>] {lhs_value = "%input1"} : !neura.data<i32, i1>
            neura.store_indexed %input0 to [%row_index, %column_index : !neura.data<index, i1>, !neura.data<index, i1>] {rhs_value = "%input1"} : !neura.data<i32, i1>
            %other0 = "neura.constant"() <{value = "%input2"}> : () -> !neura.data<memref<2x2xi32>, i1>
            %other1 = "neura.data_mov"(%other0) : (!neura.data<memref<2x2xi32>, i1>) -> !neura.data<memref<2x2xi32>, i1>
            %predicate = "neura.icmp"(%row_index) <{cmpType = "eq"}> {rhs_value = 0 : index} : (!neura.data<index, i1>) -> !neura.data<i1, i1>
            %other2 = "neura.grant_predicate"(%other1, %predicate) : (!neura.data<memref<2x2xi32>, i1>, !neura.data<i1, i1>) -> !neura.data<memref<2x2xi32>, i1>
            %other3 = "neura.data_mov"(%other2) : (!neura.data<memref<2x2xi32>, i1>) -> !neura.data<memref<2x2xi32>, i1>
            "neura.store_indexed"(%other3, %row_index, %column_index) <{operandSegmentSizes = array<i32: 0, 1, 2>}> {lhs_value = "%input0", rhs_value = "%input1"} : (!neura.data<memref<2x2xi32>, i1>, !neura.data<index, i1>, !neura.data<index, i1>) -> ()
            neura.yield {yield_type = "void"}
        }
        taskflow.yield done_writes(%write : memref<2x2xi32>)
    }

    %phi_done = taskflow.task @Task_phi
        will_reads(%state : memref<2x2xi32>)
        will_writes(%state : memref<2x2xi32>)
        value_inputs(%value : i32)
        [original_read_memrefs(%state : memref<2x2xi32>),
         original_write_memrefs(%state : memref<2x2xi32>)]
        : (memref<2x2xi32>, memref<2x2xi32>, i32) -> memref<2x2xi32> {
      ^bb0(%read: memref<2x2xi32>, %write: memref<2x2xi32>, %scalar: i32):
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %two = arith.constant 2 : index
        %row = taskflow.counter from %zero to %two step %one
            attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} : index
        %column = taskflow.counter parent(%row : index) from %zero to %two step %one
            attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} : index
        neura.kernel inputs(%scalar, %write : i32, memref<2x2xi32>)
            attributes {accelerator = "neura", dataflow_mode = "predicate"} {
          ^bb0(%input0: !neura.data<i32, i1>,
               %input1: !neura.data<memref<2x2xi32>, i1>):
            %row_index = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 2 : index} -> !neura.data<index, i1>
            %column_index = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32, lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 2 : index} -> !neura.data<index, i1>
            %bad_row = "neura.phi"(%row_index, %column_index) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
            %column_phi = "neura.phi"(%column_index, %column_index) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
            %old = neura.load_indexed [%row_index, %column_index : !neura.data<index, i1>, !neura.data<index, i1>] {lhs_value = "%input1"} : !neura.data<i32, i1>
            neura.store_indexed %input0 to [%bad_row, %column_phi : !neura.data<index, i1>, !neura.data<index, i1>] {rhs_value = "%input1"} : !neura.data<i32, i1>
            neura.yield {yield_type = "void"}
        }
        taskflow.yield done_writes(%write : memref<2x2xi32>)
    }
    return
  }
}

// VALID-LABEL: func.func @in_place_mn_tiling_proofs
// VALID: taskflow.task @Task_valid.tile.0.0
// VALID: amoeba.tiling.input_region_lowers = [array<i64: 0, 0>, unit]
// VALID: amoeba.tiling.input_region_uppers = [array<i64: 1, 2>, unit]
// VALID: amoeba.tiling.output_region_lowers = [array<i64: 0, 0>]
// VALID: amoeba.tiling.output_region_uppers = [array<i64: 1, 2>]
// VALID: taskflow.task @Task_valid.tile.0.1
// VALID: amoeba.tiling.input_region_lowers = [array<i64: 1, 0>, unit]
// VALID: amoeba.tiling.input_region_uppers = [array<i64: 2, 2>, unit]
// VALID: amoeba.tiling.output_region_lowers = [array<i64: 1, 0>]
// VALID: amoeba.tiling.output_region_uppers = [array<i64: 2, 2>]
// FOREIGN: in-place tiling rejects a store to a foreign storage root
// METADATA: in-place tiling cannot prove a rank-2 store storage root
// PHI: in-place tiling requires every output access to use the matching counter indices
