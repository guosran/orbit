// RUN: mlir-amoeba-opt %s --pass-pipeline='builtin.module(func.func(construct-hyperblock-from-task),func.func(fission-task{task-name=Captured left-nodes=0:1}))' --verify-each | FileCheck %s --check-prefix=CAPTURE
// RUN: mlir-amoeba-opt %s --pass-pipeline='builtin.module(func.func(construct-hyperblock-from-task),func.func(fission-task{task-name=Captured left-nodes=0:1}),bind-source-iteration-domain)' --verify-each | FileCheck %s --check-prefix=BOUND

// Fission may carry the source-owned pending iteration certificate to both
// children only after checking it against the current static Taskflow counters.
// The fixed lower pipeline then consumes the marker and independently binds
// each changed child body.
module {
  func.func @captured_fission(
      %input: memref<4xi32> {amoeba.noalias},
      %output: memref<4xi32> {amoeba.noalias}) {
    %done = taskflow.task @Captured
        will_reads(%input : memref<4xi32>)
        will_writes(%output : memref<4xi32>)
        [original_read_memrefs(%input : memref<4xi32>),
         original_write_memrefs(%output : memref<4xi32>)]
        {dlp_replicable = true, runtime_managable = true}
        : (memref<4xi32>, memref<4xi32>) -> memref<4xi32> {
      ^bb0(%read: memref<4xi32>, %write: memref<4xi32>):
        affine.for %i = 0 to 4 {
          %value = affine.load %read[%i] : memref<4xi32>
          %twice = arith.addi %value, %value : i32
          affine.store %twice, %write[%i] : memref<4xi32>
        }
        taskflow.yield done_writes(%write : memref<4xi32>)
    }
    return
  }
}

// CAPTURE-LABEL: func.func @captured_fission
// CAPTURE: taskflow.task @Captured.split.0
// CAPTURE-SAME: amoeba.source_iteration_capture_pending
// CAPTURE-SAME: amoeba.source_iteration_domain
// CAPTURE: taskflow.task @Captured.split.1
// CAPTURE-SAME: amoeba.source_iteration_capture_pending
// CAPTURE-SAME: amoeba.source_iteration_domain

// BOUND-LABEL: func.func @captured_fission
// BOUND: taskflow.task @Captured.split.0
// BOUND-SAME: amoeba.source_iteration_control_binding
// BOUND-SAME: amoeba.source_iteration_domain
// BOUND-SAME: amoeba.source_iteration_source_control_binding
// BOUND: taskflow.task @Captured.split.1
// BOUND-SAME: amoeba.source_iteration_control_binding
// BOUND-SAME: amoeba.source_iteration_domain
// BOUND-SAME: amoeba.source_iteration_source_control_binding
// BOUND-NOT: amoeba.source_iteration_capture_pending
