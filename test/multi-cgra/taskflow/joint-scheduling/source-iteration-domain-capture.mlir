// RUN: mlir-amoeba-opt %s --pass-pipeline='builtin.module(func.func(construct-hyperblock-from-task))' -o %t
// RUN: FileCheck %s --input-file=%t
// RUN: mlir-amoeba-opt %s --pass-pipeline='builtin.module(func.func(construct-hyperblock-from-task),bind-source-iteration-domain)' -o %t.bound
// RUN: FileCheck %s --check-prefix=BOUND --input-file=%t.bound

// The first task has one source axis represented by a Taskflow counter. The
// second has an imperfect outer loop with a static carried inner reduction;
// its init, carried update, and final store must survive full expansion. The
// dynamic task and the no-op task exercise fail-closed and empty-domain paths.
module {
  func.func @source_iteration_domain_coverage(
      %input: memref<4xi32>, %table: memref<4x2xi32>,
      %output: memref<4xi32>, %dynamic_bound: index) {
    %represented = taskflow.task @Represented
        will_reads(%input : memref<4xi32>)
        will_writes(%output : memref<4xi32>)
        [original_read_memrefs(%input : memref<4xi32>),
         original_write_memrefs(%output : memref<4xi32>)]
        : (memref<4xi32>, memref<4xi32>) -> memref<4xi32> {
      ^bb0(%read: memref<4xi32>, %write: memref<4xi32>):
        affine.for %i = 0 to 4 {
          %value = affine.load %read[%i] : memref<4xi32>
          affine.store %value, %write[%i] : memref<4xi32>
        }
        taskflow.yield done_writes(%write : memref<4xi32>)
    }

    %carried = taskflow.task @Carried
        will_reads(%input, %table : memref<4xi32>, memref<4x2xi32>)
        will_writes(%output : memref<4xi32>)
        [original_read_memrefs(%input, %table : memref<4xi32>, memref<4x2xi32>),
         original_write_memrefs(%output : memref<4xi32>)]
        : (memref<4xi32>, memref<4x2xi32>, memref<4xi32>) -> memref<4xi32> {
      ^bb0(%read: memref<4xi32>, %values: memref<4x2xi32>,
           %write: memref<4xi32>):
        affine.for %i = 0 to 3 {
          %init = affine.load %read[%i] : memref<4xi32>
          %outer32 = arith.index_cast %i : index to i32
          %seed = arith.addi %init, %outer32 : i32
          %sum = affine.for %k = 0 to 2 iter_args(%acc = %seed) -> (i32) {
            %element = affine.load %values[%i, %k] : memref<4x2xi32>
            %k32 = arith.index_cast %k : index to i32
            %withIndex = arith.addi %element, %k32 : i32
            %next = arith.addi %acc, %withIndex : i32
            affine.yield %next : i32
          }
          affine.store %sum, %write[%i] : memref<4xi32>
        }
        taskflow.yield done_writes(%write : memref<4xi32>)
    }

    %dynamic = taskflow.task @Dynamic
        will_reads(%input : memref<4xi32>)
        will_writes(%output : memref<4xi32>)
        value_inputs(%dynamic_bound : index)
        [original_read_memrefs(%input : memref<4xi32>),
         original_write_memrefs(%output : memref<4xi32>)]
        : (memref<4xi32>, memref<4xi32>, index) -> memref<4xi32> {
      ^bb0(%read: memref<4xi32>, %write: memref<4xi32>, %bound: index):
        affine.for %i = 0 to %bound {
          %value = affine.load %read[%i] : memref<4xi32>
          affine.store %value, %write[%i] : memref<4xi32>
        }
        taskflow.yield done_writes(%write : memref<4xi32>)
    }

    %mismatch = taskflow.task @CountMismatch
        will_reads(%input : memref<4xi32>)
        will_writes(%output : memref<4xi32>)
        [original_read_memrefs(%input : memref<4xi32>),
         original_write_memrefs(%output : memref<4xi32>)]
        {trip_count = 3 : i64}
        : (memref<4xi32>, memref<4xi32>) -> memref<4xi32> {
      ^bb0(%read: memref<4xi32>, %write: memref<4xi32>):
        affine.for %i = 0 to 2 {
          %value = affine.load %read[%i] : memref<4xi32>
          affine.store %value, %write[%i] : memref<4xi32>
        }
        taskflow.yield done_writes(%write : memref<4xi32>)
    }

    %noop = taskflow.task @Noop
        will_reads(%input : memref<4xi32>)
        will_writes(%output : memref<4xi32>)
        [original_read_memrefs(%input : memref<4xi32>),
         original_write_memrefs(%output : memref<4xi32>)]
        : (memref<4xi32>, memref<4xi32>) -> memref<4xi32> {
      ^bb0(%read: memref<4xi32>, %write: memref<4xi32>):
        taskflow.yield done_writes(%write : memref<4xi32>)
    }
    return
  }
}

// CHECK-LABEL: func.func @source_iteration_domain_coverage
// CHECK: taskflow.task @Represented
// CHECK-SAME: amoeba.source_iteration_domain
// CHECK-SAME: complete = true
// CHECK-SAME: represented_multiplicity = 4 : i64
// CHECK: taskflow.task @Carried
// CHECK-SAME: expanded_inside_mapper_firing = true
// CHECK-SAME: internal_multiplicity = 2 : i64
// CHECK-SAME: source_multiplicity = 6 : i64
// CHECK-NOT: affine.for
// CHECK: taskflow.task @Dynamic
// CHECK-SAME: complete = false
// CHECK: affine.for
// CHECK: taskflow.task @CountMismatch
// CHECK-SAME: complete = false
// CHECK: affine.for
// CHECK: taskflow.task @Noop
// CHECK-SAME: axes = []
// CHECK-SAME: complete = true

// BOUND: taskflow.task @Represented
// BOUND: amoeba.source_iteration_control_binding
// BOUND: amoeba.source_iteration_source_control_binding
// BOUND: taskflow.task @Noop
// BOUND: amoeba.source_iteration_control_binding
// BOUND: amoeba.source_iteration_source_control_binding
// BOUND-NOT: amoeba.source_iteration_capture_pending
