// RUN: mlir-amoeba-opt --neura-architecture-spec=%S/mapper-arch-4x4.yaml %s --pass-pipeline='builtin.module(bind-source-iteration-domain,map-joint-scheduling-tasks{function=all_unit_mapper candidate-id=candidate-0 all-unit-baseline=true mapping-cache-dir=%t-cache},func.func(orchestrate-tasks-on-accelerators{orchestration-strategy=analytical-based-task-orchestration communication-mode=explicit dispatch-policy=fixed}))' -o %t.first 2> %t.first-log
// RUN: mlir-amoeba-opt --neura-architecture-spec=%S/mapper-arch-4x4.yaml %s --pass-pipeline='builtin.module(bind-source-iteration-domain,map-joint-scheduling-tasks{function=all_unit_mapper candidate-id=candidate-0 all-unit-baseline=true mapping-cache-dir=%t-cache},func.func(orchestrate-tasks-on-accelerators{orchestration-strategy=analytical-based-task-orchestration communication-mode=explicit dispatch-policy=fixed}))' -o %t.second 2> %t.second-log
// RUN: FileCheck %s --check-prefix=CACHE --input-file=%t.second-log
// RUN: FileCheck %s --check-prefix=PROFILE --input-file=%t.second

// This tiny mapped fixture omits --scores on both runs. The second invocation
// must reuse the exact existing mapper cache entry and still emit a production
// fixed-dispatch schedule from actual mapper II plus source-derived startup.
module {
  func.func @all_unit_mapper(%input: memref<1xi32>, %output: memref<1xi32>)
      attributes {joint_scheduling_candidate_id = "candidate-0",
                  amoeba.graph_variant_id = "identity",
                  joint_scheduling_graph_variant_id = "identity",
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
         amoeba.selected_mapper_tile_rows = 4 : i64,
         amoeba.selected_mapper_tile_cols = 4 : i64,
         amoeba.selected_trip_count = 1 : i64,
         amoeba.joint_shape_orientation_fixed}
        : (memref<1xi32>, memref<1xi32>) -> memref<1xi32> {
      ^bb0(%read: memref<1xi32>, %write: memref<1xi32>):
        neura.kernel attributes {accelerator = "neura"} {
          ^bb0:
            %constant = "neura.constant"() <{value = 1 : i32}> : () -> !neura.data<i32, i1>
            neura.yield {yield_type = "void"}
        }
        taskflow.yield done_writes(%write : memref<1xi32>)
    }
    return
  }
}

// CACHE: [JointScheduling] mapper cache hits=1 misses=0
// PROFILE: joint_scheduling_actual_trace = {candidate_id = "candidate-0", communication_mode = "explicit"
// PROFILE: joint_scheduling_mapper_duration_provenance = "source-owned-all-unit-mapped-duration"
// PROFILE: joint_scheduling_ml_prediction_status = "unknown"
// PROFILE: joint_scheduling_production_dispatch_order = ["Task"]
// PROFILE: joint_scheduling_production_dispatch_policy = "fixed"
// PROFILE: profile_info =
// PROFILE-SAME: actual_mapper_ii =
// PROFILE-SAME: duration_provenance = "source-owned-all-unit-mapped-duration"
// PROFILE-SAME: ml_predicted_ii = "unknown"
// PROFILE-SAME: structural_startup_cycles =
