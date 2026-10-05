// RUN: mlir-amoeba-opt %s --pass-pipeline='builtin.module(func.func(fission-ray-carried-reduction))' --verify-diagnostics

module {
  func.func @unproven(%input: memref<1472xi32> {amoeba.noalias},
                      %output: memref<1472xi32> {amoeba.noalias},
                      %bound: index) attributes {amoeba.static_bound.arg.0 = 1472 : i64} {
    // expected-error @+1 {{fission-ray-carried-reduction: requires a source-owned caller noalias proof}}
    %result = taskflow.task @Task_13
        will_reads(%input : memref<1472xi32>)
        will_writes(%output : memref<1472xi32>)
        value_inputs(%bound : index)
        [original_read_memrefs(%input : memref<1472xi32>),
         original_write_memrefs(%output : memref<1472xi32>)]
        : (memref<1472xi32>, memref<1472xi32>, index) -> (memref<1472xi32>) {
      ^bb0(%read: memref<1472xi32>, %write: memref<1472xi32>, %n: index):
        taskflow.yield done_writes(%write : memref<1472xi32>)
    }
    func.return
  }
}
