// RUN: python3 %S/run-sibling-fusion-source-domain-checks.py %s mlir-amoeba-opt

// Construct genuine source-domain certificates from the affine loop bands.
// Task_C consumes both parent completion states so replay must prove exact
// two-to-one rewiring while retaining its body and source domain.
module {
  func.func @sibling_source_domain(
      %input: memref<4xi32> {amoeba.noalias},
      %out_a: memref<4xi32> {amoeba.noalias},
      %out_b: memref<4xi32> {amoeba.noalias},
      %out_c: memref<4xi32> {amoeba.noalias}) {
    %done_a = taskflow.task @Task_A
        will_reads(%input : memref<4xi32>)
        will_writes(%out_a : memref<4xi32>)
        [original_read_memrefs(%input : memref<4xi32>),
         original_write_memrefs(%out_a : memref<4xi32>)]
        : (memref<4xi32>, memref<4xi32>) -> memref<4xi32> {
      ^bb0(%read: memref<4xi32>, %write: memref<4xi32>):
        %one = arith.constant 1 : i32
        affine.for %i = 0 to 4 {
          %value = affine.load %read[%i] : memref<4xi32>
          %next = arith.addi %value, %one : i32
          affine.store %next, %write[%i] : memref<4xi32>
        }
        taskflow.yield done_writes(%write : memref<4xi32>)
    }
    %done_b = taskflow.task @Task_B
        will_reads(%input : memref<4xi32>)
        will_writes(%out_b : memref<4xi32>)
        [original_read_memrefs(%input : memref<4xi32>),
         original_write_memrefs(%out_b : memref<4xi32>)]
        : (memref<4xi32>, memref<4xi32>) -> memref<4xi32> {
      ^bb0(%read: memref<4xi32>, %write: memref<4xi32>):
        %two = arith.constant 2 : i32
        affine.for %i = 0 to 4 {
          %value = affine.load %read[%i] : memref<4xi32>
          %next = arith.muli %value, %two : i32
          affine.store %next, %write[%i] : memref<4xi32>
        }
        taskflow.yield done_writes(%write : memref<4xi32>)
    }
    %done_c = taskflow.task @Task_C
        will_reads(%done_a, %done_b : memref<4xi32>, memref<4xi32>)
        will_writes(%out_c : memref<4xi32>)
        [original_read_memrefs(%out_a, %out_b : memref<4xi32>, memref<4xi32>),
         original_write_memrefs(%out_c : memref<4xi32>)]
        : (memref<4xi32>, memref<4xi32>, memref<4xi32>) -> memref<4xi32> {
      ^bb0(%read_a: memref<4xi32>, %read_b: memref<4xi32>,
           %write: memref<4xi32>):
        affine.for %i = 0 to 4 {
          %a = affine.load %read_a[%i] : memref<4xi32>
          %b = affine.load %read_b[%i] : memref<4xi32>
          %sum = arith.addi %a, %b : i32
          affine.store %sum, %write[%i] : memref<4xi32>
        }
        taskflow.yield done_writes(%write : memref<4xi32>)
    }
    return
  }
}
