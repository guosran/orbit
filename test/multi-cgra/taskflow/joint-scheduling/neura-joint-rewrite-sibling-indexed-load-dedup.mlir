// Focused post-Neura check for shared indexed reads and predicate identity.
// RUN: python3 %S/run-neura-sibling-indexed-load-dedup-checks.py %s mlir-amoeba-opt

module {
  func.func @sibling_indexed_loads(
      %input: memref<4x4xi32> {amoeba.noalias},
      %input_alt: memref<4x4xi32> {amoeba.noalias},
      %out_a: memref<4x4xi32> {amoeba.noalias},
      %out_b: memref<4x4xi32> {amoeba.noalias}) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %extent = arith.constant 4 : index

    %done_a = taskflow.task @Task_A
        will_reads(%input : memref<4x4xi32>)
        will_writes(%out_a : memref<4x4xi32>)
        value_inputs(%extent : index)
        [original_read_memrefs(%input : memref<4x4xi32>),
         original_write_memrefs(%out_a : memref<4x4xi32>)]
        : (memref<4x4xi32>, memref<4x4xi32>, index) -> memref<4x4xi32> {
      ^bb0(%read: memref<4x4xi32>, %write: memref<4x4xi32>, %bound: index):
        %lower = arith.constant 0 : index
        %step = arith.constant 1 : index
        %row = taskflow.counter from %lower to %bound step %step
            attributes {counter_dynamism = "constant_bound",
                        counter_hierarchy = "root", counter_id = 0 : i32} : index
        %column = taskflow.counter parent(%row : index)
            from %lower to %bound step %step
            attributes {counter_dynamism = "constant_bound",
                        counter_hierarchy = "leaf", counter_id = 1 : i32} : index
        neura.kernel inputs(%read, %write, %bound : memref<4x4xi32>, memref<4x4xi32>, index)
            attributes {accelerator = "neura", dataflow_mode = "predicate"} {
          ^bb0(%input0: !neura.data<memref<4x4xi32>, i1>,
               %input1: !neura.data<memref<4x4xi32>, i1>,
               %input2: !neura.data<index, i1>):
            %row_index = neura.counter attributes {
              counter_dynamism = "symbol_bound", counter_hierarchy = "root",
              counter_id = 0 : i32, lower_bound_value = 0 : index,
              step_value = 1 : index, upper_bound_value = "%input2"}
                -> !neura.data<index, i1>
            %column_index = neura.counter attributes {
              counter_dynamism = "symbol_bound", counter_hierarchy = "leaf",
              counter_id = 1 : i32, lower_bound_value = 0 : index,
              step_value = 1 : index, upper_bound_value = "%input2"}
                -> !neura.data<index, i1>
            %load_validity = "neura.constant"() <{value = 1 : i1}> : () -> !neura.data<i1, i1>
            %predicated_row = neura.grant_predicate %row_index, %load_validity : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
            %value = neura.load_indexed [%predicated_row, %column_index : !neura.data<index, i1>, !neura.data<index, i1>]
                {lhs_value = "%input0"} : !neura.data<i32, i1>
            neura.store_indexed %value to [%row_index, %column_index : !neura.data<index, i1>, !neura.data<index, i1>]
                {rhs_value = "%input1"} : !neura.data<i32, i1>
            neura.yield {yield_type = "void"}
        }
        taskflow.yield done_writes(%write : memref<4x4xi32>)
    }

    %done_b = taskflow.task @Task_B
        will_reads(%input : memref<4x4xi32>)
        will_writes(%out_b : memref<4x4xi32>)
        value_inputs(%extent : index)
        [original_read_memrefs(%input : memref<4x4xi32>),
         original_write_memrefs(%out_b : memref<4x4xi32>)]
        : (memref<4x4xi32>, memref<4x4xi32>, index) -> memref<4x4xi32> {
      ^bb0(%read: memref<4x4xi32>, %write: memref<4x4xi32>, %bound: index):
        %lower = arith.constant 0 : index
        %step = arith.constant 1 : index
        %row = taskflow.counter from %lower to %bound step %step
            attributes {counter_dynamism = "constant_bound",
                        counter_hierarchy = "root", counter_id = 0 : i32} : index
        %column = taskflow.counter parent(%row : index)
            from %lower to %bound step %step
            attributes {counter_dynamism = "constant_bound",
                        counter_hierarchy = "leaf", counter_id = 1 : i32} : index
        neura.kernel inputs(%read, %write, %bound : memref<4x4xi32>, memref<4x4xi32>, index)
            attributes {accelerator = "neura", dataflow_mode = "predicate"} {
          ^bb0(%input0: !neura.data<memref<4x4xi32>, i1>,
               %input1: !neura.data<memref<4x4xi32>, i1>,
               %input2: !neura.data<index, i1>):
            %row_index = neura.counter attributes {
              counter_dynamism = "symbol_bound", counter_hierarchy = "root",
              counter_id = 0 : i32, lower_bound_value = 0 : index,
              step_value = 1 : index, upper_bound_value = "%input2"}
                -> !neura.data<index, i1>
            %column_index = neura.counter attributes {
              counter_dynamism = "symbol_bound", counter_hierarchy = "leaf",
              counter_id = 1 : i32, lower_bound_value = 0 : index,
              step_value = 1 : index, upper_bound_value = "%input2"}
                -> !neura.data<index, i1>
            %load_validity = "neura.constant"() <{value = 1 : i1}> : () -> !neura.data<i1, i1>
            %predicated_row = neura.grant_predicate %row_index, %load_validity : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
            %value = neura.load_indexed [%predicated_row, %column_index : !neura.data<index, i1>, !neura.data<index, i1>]
                {lhs_value = "%input0"} : !neura.data<i32, i1>
            neura.store_indexed %value to [%row_index, %column_index : !neura.data<index, i1>, !neura.data<index, i1>]
                {rhs_value = "%input1"} : !neura.data<i32, i1>
            neura.yield {yield_type = "void"}
        }
        taskflow.yield done_writes(%write : memref<4x4xi32>)
    }
    return
  }
}
