// RUN: mlir-amoeba-opt %s --pass-pipeline='builtin.module(bind-source-iteration-domain)' --verify-diagnostics

module {
  func.func @forged_source_iteration_domain(%input: memref<1xi32>,
                                             %output: memref<1xi32>) {
    // expected-error @+1 {{has a stale or inconsistent source iteration-domain witness}}
    %done = taskflow.task @Forged
        will_reads(%input : memref<1xi32>)
        will_writes(%output : memref<1xi32>)
        [original_read_memrefs(%input : memref<1xi32>),
         original_write_memrefs(%output : memref<1xi32>)]
        {amoeba.source_iteration_control_binding = "stale",
         amoeba.source_iteration_domain = {
           axes = [],
           canonical_witness = "forged",
           complete = true,
           internal_multiplicity = 1 : i64,
           represented_multiplicity = 1 : i64,
           schema = "amoeba-source-iteration-domain-v1",
           source_multiplicity = 1 : i64
         },
         amoeba.source_iteration_source_control_binding = "stale"}
        : (memref<1xi32>, memref<1xi32>) -> memref<1xi32> {
      ^bb0(%read: memref<1xi32>, %write: memref<1xi32>):
        taskflow.yield done_writes(%write : memref<1xi32>)
    }
    return
  }
}
