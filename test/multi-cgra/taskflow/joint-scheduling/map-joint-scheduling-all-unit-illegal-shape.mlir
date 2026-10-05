// RUN: not mlir-amoeba-opt %s --map-joint-scheduling-tasks='function=illegal_shape candidate-id=candidate-0 all-unit-baseline=true' -o /dev/null 2>&1 | FileCheck %s --check-prefix=SHAPE

// Shape validation happens before source-domain validation or any mapper run.
module {
  func.func @illegal_shape(%input: memref<1xi32>, %output: memref<1xi32>)
      attributes {joint_scheduling_candidate_id = "candidate-0",
                  joint_scheduling_candidate_scope = "static-shape-cartesian-product"} {
    %task = taskflow.task @Task
        will_reads(%input : memref<1xi32>)
        will_writes(%output : memref<1xi32>)
        [original_read_memrefs(%input : memref<1xi32>),
         original_write_memrefs(%output : memref<1xi32>)]
        {cgra_count = 2 : i32, cgra_shape = "1x2",
         amoeba.selected_cgra_count = 2 : i64,
         amoeba.selected_cgra_shape = "1x2",
         amoeba.selected_mapper_tile_rows = 1 : i64,
         amoeba.selected_mapper_tile_cols = 2 : i64,
         amoeba.selected_trip_count = 1 : i64,
         amoeba.joint_shape_orientation_fixed}
        : (memref<1xi32>, memref<1xi32>) -> memref<1xi32> {
      ^bb0(%read: memref<1xi32>, %write: memref<1xi32>):
        taskflow.yield done_writes(%write : memref<1xi32>)
    }
    return
  }
}

// SHAPE: all-unit baseline requires the identity-only 1x1 fixed-orientation shape for every task
