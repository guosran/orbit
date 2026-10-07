// RUN: mlir-amoeba-opt %s '--pass-pipeline=builtin.module(func.func(fission-task{task-name=Effect left-nodes=0:1}))' --verify-diagnostics

module {
  func.func private @unknown_effect() -> i32

  func.func @source_effect(%output: memref<4xi32> {amoeba.noalias}) {
    // expected-error @+1 {{fission rejects nested regions and unknown effects}}
    %done_writes = taskflow.task @Effect
        will_writes(%output : memref<4xi32>)
        [original_write_memrefs(%output : memref<4xi32>)]
        {dlp_replicable = true, runtime_managable = true}
        : (memref<4xi32>) -> (memref<4xi32>) {
      ^bb0(%write: memref<4xi32>):
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %four = arith.constant 4 : index
        %counter = taskflow.counter from %zero to %four step %one : index
        "taskflow.hyperblock"(%counter)
            <{operandSegmentSizes = array<i32: 1, 0>}> ({
          ^bb0(%i: index):
            %value = func.call @unknown_effect() : () -> i32
            %twice = arith.addi %value, %value : i32
            memref.store %twice, %write[%i] : memref<4xi32>
            "taskflow.hyperblock.yield"() : () -> ()
        }) : (index) -> ()
        taskflow.yield done_writes(%write : memref<4xi32>)
    }
    func.return
  }
}
