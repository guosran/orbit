// RUN: not mlir-amoeba-opt %s --map-joint-scheduling-tasks='function=unproved_domain candidate-id=candidate-0 all-unit-baseline=true' -o /dev/null 2>&1 | FileCheck %s --check-prefix=PROOF

// No score file is supplied. The opt-in mode reaches the source proof gate
// and fails before attempting to find a kernel or invoke the mapper.
module {
  func.func @unproved_domain(%input: memref<1xi32>, %output: memref<1xi32>)
      attributes {joint_scheduling_candidate_id = "candidate-0",
                  joint_scheduling_candidate_scope = "static-shape-cartesian-product"} {
    %task = taskflow.task @Task
        will_reads(%input : memref<1xi32>)
        will_writes(%output : memref<1xi32>)
        [original_read_memrefs(%input : memref<1xi32>),
         original_write_memrefs(%output : memref<1xi32>)]
        {cgra_count = 1 : i32, cgra_shape = "1x1",
         amoeba.selected_cgra_count = 1 : i64,
         amoeba.selected_cgra_shape = "1x1",
         amoeba.selected_mapper_tile_rows = 4 : i64,
         amoeba.selected_mapper_tile_cols = 4 : i64,
         amoeba.selected_trip_count = 1 : i64,
         amoeba.joint_shape_orientation_fixed}
        : (memref<1xi32>, memref<1xi32>) -> memref<1xi32> {
      ^bb0(%read: memref<1xi32>, %write: memref<1xi32>):
        taskflow.yield done_writes(%write : memref<1xi32>)
    }
    return
  }
}

// PROOF: all-unit baseline requires a complete, source-owned iteration-domain certificate and proven current count for task Task
