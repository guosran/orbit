// RUN: mlir-amoeba-opt %s '--fission-task=task-name=Branches left-nodes=0:1:2' --verify-each | FileCheck %s --check-prefix=INDEPENDENT
// RUN: mlir-amoeba-opt %s '--fission-task=task-name=Branches left-nodes=0:3' --verify-each | FileCheck %s --check-prefix=MULTICUT
// RUN: not mlir-amoeba-opt %s '--fission-task=task-name=Branches left-nodes=1' 2>&1 | FileCheck %s --check-prefix=BAD
// RUN: mlir-amoeba-opt %s '--verify-taskflow-fission-source-replay=enumerate-function=branches max-fission-actions-per-task=1 output=%t.capped.json' -o /dev/null
// RUN: FileCheck %s --check-prefix=CAP < %t.capped.json
// Public outputs are disjoint, stores use the same reversed counter indexing
// as Radar I/Q, and both owned result tokens remain visible to the caller.
module {
  func.func @branches(%a: memref<3x2xi32> {amoeba.noalias},
                      %b: memref<3x2xi32> {amoeba.noalias},
                      %y: memref<3x2xi32> {amoeba.noalias},
                      %z: memref<3x2xi32> {amoeba.noalias}) {
    %out:2 = taskflow.task @Branches
      will_reads(%a, %b : memref<3x2xi32>, memref<3x2xi32>)
      will_writes(%y, %z : memref<3x2xi32>, memref<3x2xi32>)
      [original_read_memrefs(%a, %b : memref<3x2xi32>, memref<3x2xi32>),
       original_write_memrefs(%y, %z : memref<3x2xi32>, memref<3x2xi32>)]
      : (memref<3x2xi32>, memref<3x2xi32>, memref<3x2xi32>, memref<3x2xi32>) -> (memref<3x2xi32>, memref<3x2xi32>) {
      ^bb0(%ra: memref<3x2xi32>, %rb: memref<3x2xi32>, %wy: memref<3x2xi32>, %wz: memref<3x2xi32>):
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %two = arith.constant 2 : index
        %three = arith.constant 3 : index
        %i = taskflow.counter from %zero to %two step %one : index
        %j = "taskflow.counter"(%i, %zero, %three, %one) : (index, index, index, index) -> index
        "taskflow.hyperblock"(%i, %j) <{operandSegmentSizes = array<i32: 2, 0>}> ({
        ^bb0(%x: index, %v: index):
          %av = memref.load %ra[%v, %x] : memref<3x2xi32>
          %twice = arith.addi %av, %av : i32
          memref.store %twice, %wy[%v, %x] : memref<3x2xi32>
          %bv = memref.load %rb[%v, %x] : memref<3x2xi32>
          %square = arith.muli %bv, %bv : i32
          memref.store %square, %wz[%v, %x] : memref<3x2xi32>
          "taskflow.hyperblock.yield"() : () -> ()
        }) : (index, index) -> ()
        taskflow.yield done_writes(%wy, %wz : memref<3x2xi32>, memref<3x2xi32>)
    }
    return
  }
}
// INDEPENDENT-NOT: memref.alloc
// INDEPENDENT: taskflow.task @Branches.split.0
// INDEPENDENT: arith.addi
// INDEPENDENT-NOT: arith.muli
// INDEPENDENT: taskflow.task @Branches.split.1
// INDEPENDENT-NOT: arith.addi
// INDEPENDENT: arith.muli
// MULTICUT: memref.alloc() : memref<2x3xi32>
// MULTICUT: memref.alloc() : memref<2x3xi32>
// MULTICUT: taskflow.task @Branches.split.0
// MULTICUT: taskflow.task @Branches.split.1
// MULTICUT: arith.addi
// MULTICUT: arith.muli
// BAD: left-nodes must be predecessor closed
// CAP: "status":"complete"
// CAP: "diagnostic":"deterministic-cut-cap: legal cut enumeration truncated at 1"
// CAP: "legal_cut_count":1
