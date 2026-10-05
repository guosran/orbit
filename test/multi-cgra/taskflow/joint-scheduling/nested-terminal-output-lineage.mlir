module {
  func.func @source_partition_lineage(
      %input: memref<?x?xi32> {amoeba.noalias, amoeba.logical_transfer_shape = array<i64: 4, 4>},
      %output: memref<?x?xi32> {amoeba.noalias, amoeba.logical_transfer_shape = array<i64: 4, 4>},
      %final: memref<?x?xi32> {amoeba.noalias, amoeba.logical_transfer_shape = array<i64: 4, 4>})
      -> memref<?x?xi32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %extent = arith.constant 4 : index
    %done = taskflow.task @Task
        will_reads(%input : memref<?x?xi32>)
        will_writes(%output : memref<?x?xi32>)
        value_inputs(%extent : index)
        [original_read_memrefs(%input : memref<?x?xi32>),
         original_write_memrefs(%output : memref<?x?xi32>)]
        {amoeba.source_iteration_capture_pending,
         amoeba.source_iteration_domain = {
           axes = [{carried_values = 0 : i64,
                    expanded_inside_mapper_firing = false, extent = 4 : i64,
                    kind = "taskflow-counter", lower = 0 : i64,
                    ordinal = 0 : i64, parent_counter_ordinal = -1 : i64,
                    result_uses = 0 : i64, step = 1 : i64, upper = 4 : i64},
                   {carried_values = 0 : i64,
                    expanded_inside_mapper_firing = false, extent = 4 : i64,
                    kind = "taskflow-counter", lower = 0 : i64,
                    ordinal = 1 : i64, parent_counter_ordinal = -1 : i64,
                    result_uses = 0 : i64, step = 1 : i64, upper = 4 : i64}],
           canonical_witness = "amoeba-source-iteration-domain-v1\ncomplete=1\nrepresented_multiplicity=16\ninternal_multiplicity=1\nsource_multiplicity=16\naxis_count=2\naxis=T,0,0,4,1,4,0,-1,0,0\naxis=T,1,0,4,1,4,0,-1,0,0\nreason_bytes=0:\n",
           complete = true, internal_multiplicity = 1 : i64,
           represented_multiplicity = 16 : i64,
           schema = "amoeba-source-iteration-domain-v1",
           source_multiplicity = 16 : i64}}
        : (memref<?x?xi32>, memref<?x?xi32>, index)
          -> memref<?x?xi32> {
      ^bb0(%read: memref<?x?xi32>, %write: memref<?x?xi32>, %bound: index):
        %lower = arith.constant 0 : index
        %step = arith.constant 1 : index
        %four = arith.constant 4 : index
        %i = taskflow.counter from %lower to %four step %step
            attributes {counter_dynamism = "constant_bound",
                        counter_hierarchy = "root", counter_id = 0 : i32} : index
        %j = taskflow.counter parent(%i : index) from %lower to %four step %step
            attributes {counter_dynamism = "constant_bound",
                        counter_hierarchy = "leaf", counter_id = 1 : i32} : index
        neura.kernel inputs(%read, %write : memref<?x?xi32>, memref<?x?xi32>)
            attributes {accelerator = "neura", dataflow_mode = "predicate"} {
          ^bb0(%input0: !neura.data<memref<?x?xi32>, i1>,
               %input1: !neura.data<memref<?x?xi32>, i1>):
            %index = neura.counter attributes {
              counter_dynamism = "constant_bound", counter_hierarchy = "root",
              counter_id = 0 : i32, lower_bound_value = 0 : index,
              step_value = 1 : index, upper_bound_value = 4 : index}
                -> !neura.data<index, i1>
            %column = neura.counter attributes {
              counter_dynamism = "constant_bound", counter_hierarchy = "leaf",
              counter_id = 1 : i32, lower_bound_value = 0 : index,
              step_value = 1 : index, upper_bound_value = 4 : index}
                -> !neura.data<index, i1>
            %value = neura.load_indexed [%index, %column : !neura.data<index, i1>, !neura.data<index, i1>]
                {lhs_value = "%input0"} : !neura.data<i32, i1>
            neura.store_indexed %value to [%index, %column : !neura.data<index, i1>, !neura.data<index, i1>]
                {rhs_value = "%input1"} : !neura.data<i32, i1>
            neura.yield {yield_type = "void"}
        }
        taskflow.yield done_writes(%write : memref<?x?xi32>)
    }
    %consumer = taskflow.task @Consumer
        will_reads(%done : memref<?x?xi32>)
        will_writes(%final : memref<?x?xi32>)
        [original_read_memrefs(%output : memref<?x?xi32>),
         original_write_memrefs(%final : memref<?x?xi32>)]
        {amoeba.source_iteration_capture_pending,
         amoeba.source_iteration_domain = {
           axes = [{carried_values = 0 : i64,
                    expanded_inside_mapper_firing = false, extent = 4 : i64,
                    kind = "taskflow-counter", lower = 0 : i64,
                    ordinal = 0 : i64, parent_counter_ordinal = -1 : i64,
                    result_uses = 0 : i64, step = 1 : i64, upper = 4 : i64},
                   {carried_values = 0 : i64,
                    expanded_inside_mapper_firing = false, extent = 4 : i64,
                    kind = "taskflow-counter", lower = 0 : i64,
                    ordinal = 1 : i64, parent_counter_ordinal = -1 : i64,
                    result_uses = 0 : i64, step = 1 : i64, upper = 4 : i64}],
           canonical_witness = "amoeba-source-iteration-domain-v1\ncomplete=1\nrepresented_multiplicity=16\ninternal_multiplicity=1\nsource_multiplicity=16\naxis_count=2\naxis=T,0,0,4,1,4,0,-1,0,0\naxis=T,1,0,4,1,4,0,-1,0,0\nreason_bytes=0:\n",
           complete = true, internal_multiplicity = 1 : i64,
           represented_multiplicity = 16 : i64,
           schema = "amoeba-source-iteration-domain-v1",
           source_multiplicity = 16 : i64}}
        : (memref<?x?xi32>, memref<?x?xi32>) -> memref<?x?xi32> {
      ^bb0(%read: memref<?x?xi32>, %write: memref<?x?xi32>):
        %lower = arith.constant 0 : index
        %step = arith.constant 1 : index
        %four = arith.constant 4 : index
        %i = taskflow.counter from %lower to %four step %step
            attributes {counter_dynamism = "constant_bound",
                        counter_hierarchy = "root", counter_id = 0 : i32} : index
        %j = taskflow.counter parent(%i : index) from %lower to %four step %step
            attributes {counter_dynamism = "constant_bound",
                        counter_hierarchy = "leaf", counter_id = 1 : i32} : index
        neura.kernel inputs(%read, %write : memref<?x?xi32>, memref<?x?xi32>)
            attributes {accelerator = "neura", dataflow_mode = "predicate"} {
          ^bb0(%input0: !neura.data<memref<?x?xi32>, i1>,
               %input1: !neura.data<memref<?x?xi32>, i1>):
            %index = neura.counter attributes {
              counter_dynamism = "constant_bound", counter_hierarchy = "root",
              counter_id = 0 : i32, lower_bound_value = 0 : index,
              step_value = 1 : index, upper_bound_value = 4 : index}
                -> !neura.data<index, i1>
            %column = neura.counter attributes {
              counter_dynamism = "constant_bound", counter_hierarchy = "leaf",
              counter_id = 1 : i32, lower_bound_value = 0 : index,
              step_value = 1 : index, upper_bound_value = 4 : index}
                -> !neura.data<index, i1>
            %value = neura.load_indexed [%index, %column : !neura.data<index, i1>, !neura.data<index, i1>]
                {lhs_value = "%input0"} : !neura.data<i32, i1>
            neura.store_indexed %value to [%index, %column : !neura.data<index, i1>, !neura.data<index, i1>]
                {rhs_value = "%input1"} : !neura.data<i32, i1>
            neura.yield {yield_type = "void"}
        }
        taskflow.yield done_writes(%write : memref<?x?xi32>)
    }
    return %consumer : memref<?x?xi32>
  }
}
