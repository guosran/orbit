// RUN: mlir-amoeba-opt %s '--pass-pipeline=builtin.module(func.func(fission-task{task-name=Alias left-nodes=0}))' --verify-diagnostics

module {
  func.func @source_alias(%buffer: memref<4xi32> {amoeba.noalias}) {
    // expected-error @+1 {{fission requires a proof that each read input is disjoint from the task output}}
    %done_writes = taskflow.task @Alias
        will_reads(%buffer : memref<4xi32>)
        will_writes(%buffer : memref<4xi32>)
        [original_read_memrefs(%buffer : memref<4xi32>),
         original_write_memrefs(%buffer : memref<4xi32>)]
        {dlp_replicable = true, runtime_managable = true}
        : (memref<4xi32>, memref<4xi32>) -> (memref<4xi32>) {
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
    func.return
  }
}
