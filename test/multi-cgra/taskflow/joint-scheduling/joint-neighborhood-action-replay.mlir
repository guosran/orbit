// RUN: python3 %S/run-joint-neighborhood-action-replay-checks.py %s mlir-amoeba-opt

// A small source-certified Taskflow/Neura axis used to replay real C++ graph
// actions and exercise candidate chaining under the immutable source guard.
module {
  func.func @source_partition_lineage(%input: memref<4x1xi32> {amoeba.noalias},
                                      %output: memref<4x1xi32> {amoeba.noalias}) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %extent = arith.constant 4 : index
    %done = taskflow.task @Task
        will_reads(%input : memref<4x1xi32>)
        will_writes(%output : memref<4x1xi32>)
        value_inputs(%extent : index)
        [original_read_memrefs(%input : memref<4x1xi32>),
         original_write_memrefs(%output : memref<4x1xi32>)]
        {amoeba.source_iteration_capture_pending,
         amoeba.source_iteration_domain = {
           axes = [{carried_values = 0 : i64,
                    expanded_inside_mapper_firing = false, extent = 4 : i64,
                    kind = "taskflow-counter", lower = 0 : i64,
                    ordinal = 0 : i64, parent_counter_ordinal = -1 : i64,
                    result_uses = 0 : i64, step = 1 : i64, upper = 4 : i64},
                   {carried_values = 0 : i64,
                    expanded_inside_mapper_firing = false, extent = 1 : i64,
                    kind = "taskflow-counter", lower = 0 : i64,
                    ordinal = 1 : i64, parent_counter_ordinal = -1 : i64,
                    result_uses = 0 : i64, step = 1 : i64, upper = 1 : i64}],
           canonical_witness = "amoeba-source-iteration-domain-v1\ncomplete=1\nrepresented_multiplicity=4\ninternal_multiplicity=1\nsource_multiplicity=4\naxis_count=2\naxis=T,0,0,4,1,4,0,-1,0,0\naxis=T,1,0,1,1,1,0,-1,0,0\nreason_bytes=0:\n",
           complete = true, internal_multiplicity = 1 : i64,
           represented_multiplicity = 4 : i64,
           schema = "amoeba-source-iteration-domain-v1",
           source_multiplicity = 4 : i64}}
        : (memref<4x1xi32>, memref<4x1xi32>, index) -> memref<4x1xi32> {
      ^bb0(%read: memref<4x1xi32>, %write: memref<4x1xi32>, %bound: index):
        %lower = arith.constant 0 : index
        %step = arith.constant 1 : index
        %i = taskflow.counter from %lower to %bound step %step
            attributes {counter_dynamism = "constant_bound",
                        counter_hierarchy = "root", counter_id = 0 : i32} : index
        %j = taskflow.counter parent(%i : index) from %lower to %step step %step
            attributes {counter_dynamism = "constant_bound",
                        counter_hierarchy = "leaf", counter_id = 1 : i32} : index
        neura.kernel inputs(%read, %write, %bound : memref<4x1xi32>, memref<4x1xi32>, index)
            attributes {accelerator = "neura", dataflow_mode = "predicate"} {
          ^bb0(%input0: !neura.data<memref<4x1xi32>, i1>,
               %input1: !neura.data<memref<4x1xi32>, i1>,
               %input2: !neura.data<index, i1>):
            %index = neura.counter attributes {
              counter_dynamism = "symbol_bound", counter_hierarchy = "root",
              counter_id = 0 : i32, lower_bound_value = 0 : index,
              step_value = 1 : index, upper_bound_value = "%input2"}
                -> !neura.data<index, i1>
            %column = neura.counter attributes {
              counter_dynamism = "constant_bound", counter_hierarchy = "leaf",
              counter_id = 1 : i32, lower_bound_value = 0 : index,
              step_value = 1 : index, upper_bound_value = 1 : index}
                -> !neura.data<index, i1>
            %value = neura.load_indexed [%index, %column : !neura.data<index, i1>, !neura.data<index, i1>]
                {lhs_value = "%input0"} : !neura.data<i32, i1>
            neura.store_indexed %value to [%index, %column : !neura.data<index, i1>, !neura.data<index, i1>]
                {rhs_value = "%input1"} : !neura.data<i32, i1>
            neura.yield {yield_type = "void"}
        }
        taskflow.yield done_writes(%write : memref<4x1xi32>)
    }
    return
  }
}
