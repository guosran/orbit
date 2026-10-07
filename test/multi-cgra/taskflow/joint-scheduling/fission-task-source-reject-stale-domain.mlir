// RUN: mlir-amoeba-opt %s '--pass-pipeline=builtin.module(func.func(fission-task{task-name=Stale left-nodes=0:1}))' --verify-diagnostics

module {
  func.func @source_stale_domain(
      %input: memref<4xi32> {amoeba.noalias},
      %output: memref<4xi32> {amoeba.noalias}) {
    // The self-consistent certificate says [0, 5), but the actual counter is
    // [0, 4). Fission must rederive the captured extent from the live body.
    // expected-error @+1 {{fission source iteration-domain capture disagrees with the current Taskflow counter chain}}
    %done = taskflow.task @Stale
        will_reads(%input : memref<4xi32>)
        will_writes(%output : memref<4xi32>)
        [original_read_memrefs(%input : memref<4xi32>),
         original_write_memrefs(%output : memref<4xi32>)]
        {amoeba.source_iteration_capture_pending,
         amoeba.source_iteration_domain = {
           axes = [{carried_values = 0 : i64,
                    expanded_inside_mapper_firing = false,
                    extent = 5 : i64, kind = "taskflow-counter",
                    lower = 0 : i64, ordinal = 0 : i64,
                    parent_counter_ordinal = -1 : i64,
                    result_uses = 0 : i64, step = 1 : i64,
                    upper = 5 : i64}],
           canonical_witness = "amoeba-source-iteration-domain-v1\0Acomplete=1\0Arepresented_multiplicity=5\0Ainternal_multiplicity=1\0Asource_multiplicity=5\0Aaxis_count=1\0Aaxis=T,0,0,5,1,5,0,-1,0,0\0Areason_bytes=0:\0A",
           complete = true, internal_multiplicity = 1 : i64,
           represented_multiplicity = 5 : i64,
           schema = "amoeba-source-iteration-domain-v1",
           source_multiplicity = 5 : i64},
         dlp_replicable = true, runtime_managable = true}
        : (memref<4xi32>, memref<4xi32>) -> memref<4xi32> {
      ^bb0(%read: memref<4xi32>, %write: memref<4xi32>):
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %four = arith.constant 4 : index
        %counter = taskflow.counter from %zero to %four step %one : index
        "taskflow.hyperblock"(%counter)
            <{operandSegmentSizes = array<i32: 1, 0>}> ({
          ^bb0(%i: index):
            %value = memref.load %read[%i] : memref<4xi32>
            %twice = arith.addi %value, %value : i32
            memref.store %twice, %write[%i] : memref<4xi32>
            "taskflow.hyperblock.yield"() : () -> ()
        }) : (index) -> ()
        taskflow.yield done_writes(%write : memref<4xi32>)
    }
    return
  }
}
