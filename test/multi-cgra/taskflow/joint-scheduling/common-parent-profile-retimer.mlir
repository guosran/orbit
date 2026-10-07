// Small one-task source used by the native common-parent-profile retimer
// harness. The harness first binds this source, exports all eight actual
// mapper shapes, then creates a separate all-unit F45 schedule for replay.
module {
  func.func @common_profile_mapper(%input: memref<1xi32>,
                                   %output: memref<1xi32>)
      attributes {joint_scheduling_candidate_id = "candidate-0",
                  joint_scheduling_candidate_scope = "static-shape-cartesian-product"} {
    %task = taskflow.task @Task
        will_reads(%input : memref<1xi32>)
        will_writes(%output : memref<1xi32>)
        [original_read_memrefs(%input : memref<1xi32>),
         original_write_memrefs(%output : memref<1xi32>)]
        {amoeba.source_iteration_capture_pending,
         amoeba.source_iteration_domain = {
           axes = [],
           canonical_witness = "amoeba-source-iteration-domain-v1\ncomplete=1\nrepresented_multiplicity=1\ninternal_multiplicity=1\nsource_multiplicity=1\naxis_count=0\nreason_bytes=0:\n",
           complete = true, internal_multiplicity = 1 : i64,
           represented_multiplicity = 1 : i64,
           schema = "amoeba-source-iteration-domain-v1",
           source_multiplicity = 1 : i64},
         cgra_count = 1 : i32, cgra_shape = "1x1",
         amoeba.selected_cgra_count = 1 : i64,
         amoeba.selected_cgra_shape = "1x1",
         amoeba.selected_mapper_tile_rows = 2 : i64,
         amoeba.selected_mapper_tile_cols = 2 : i64,
         amoeba.selected_trip_count = 1 : i64,
         amoeba.joint_shape_orientation_fixed}
        : (memref<1xi32>, memref<1xi32>) -> memref<1xi32> {
      ^bb0(%read: memref<1xi32>, %write: memref<1xi32>):
        neura.kernel attributes {accelerator = "neura"} {
          ^bb0:
            %constant = "neura.constant"() <{value = 1 : i32}> : () -> !neura.data<i32, i1>
            %second = "neura.constant"() <{value = 2 : i32}> : () -> !neura.data<i32, i1>
            neura.yield {yield_type = "void"}
        }
        taskflow.yield done_writes(%write : memref<1xi32>)
    }
    return
  }
}
