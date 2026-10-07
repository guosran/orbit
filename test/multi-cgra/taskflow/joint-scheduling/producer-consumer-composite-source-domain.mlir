// RUN: python3 %S/run-producer-consumer-composite-source-domain-checks.py %s mlir-amoeba-opt

// The source affine loops produce genuine domain captures and lower to
// Neura indexed reads/writes. Task_A writes a private alloca used only by
// Task_B; forwarded fusion must remove that indexed store/load pair.
module {
  func.func @source_partition_lineage(
      %input: memref<4xi32> {amoeba.noalias},
      %output: memref<4xi32> {amoeba.noalias},
      %final: memref<4xi32> {amoeba.noalias}) {
    %intermediate = memref.alloca() : memref<4xi32>
    %producer = taskflow.task @Task
        will_reads(%input : memref<4xi32>)
        will_writes(%intermediate : memref<4xi32>)
        [original_read_memrefs(%input : memref<4xi32>),
         original_write_memrefs(%intermediate : memref<4xi32>)]
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
    %channel = taskflow.channel %producer : memref<4xi32> -> memref<4xi32>
    %consumer = taskflow.task @Consumer
        will_reads(%channel : memref<4xi32>)
        will_writes(%output : memref<4xi32>)
        [original_read_memrefs(%intermediate : memref<4xi32>),
         original_write_memrefs(%output : memref<4xi32>)]
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
    %finalized = taskflow.task @Completion
        will_reads(%consumer : memref<4xi32>)
        will_writes(%final : memref<4xi32>)
        [original_read_memrefs(%output : memref<4xi32>),
         original_write_memrefs(%final : memref<4xi32>)]
        : (memref<4xi32>, memref<4xi32>) -> memref<4xi32> {
      ^bb0(%read: memref<4xi32>, %write: memref<4xi32>):
        affine.for %i = 0 to 4 {
          %value = affine.load %read[%i] : memref<4xi32>
          affine.store %value, %write[%i] : memref<4xi32>
        }
        taskflow.yield done_writes(%write : memref<4xi32>)
    }
    return
  }

  // The same memory shape with an externally owned argument as intermediate
  // must remain ineligible for forwarded fusion.
  func.func @public_intermediate_negative(
      %input: memref<4xi32> {amoeba.noalias},
      %shared: memref<4xi32> {amoeba.noalias},
      %output: memref<4xi32> {amoeba.noalias}) {
    %producer = taskflow.task @PublicProducer
        will_reads(%input : memref<4xi32>)
        will_writes(%shared : memref<4xi32>)
        [original_read_memrefs(%input : memref<4xi32>),
         original_write_memrefs(%shared : memref<4xi32>)]
        : (memref<4xi32>, memref<4xi32>) -> memref<4xi32> {
      ^bb0(%read: memref<4xi32>, %write: memref<4xi32>):
        affine.for %i = 0 to 4 {
          %value = affine.load %read[%i] : memref<4xi32>
          affine.store %value, %write[%i] : memref<4xi32>
        }
        taskflow.yield done_writes(%write : memref<4xi32>)
    }
    %channel = taskflow.channel %producer : memref<4xi32> -> memref<4xi32>
    %consumer = taskflow.task @PublicConsumer
        will_reads(%channel : memref<4xi32>)
        will_writes(%output : memref<4xi32>)
        [original_read_memrefs(%shared : memref<4xi32>),
         original_write_memrefs(%output : memref<4xi32>)]
        : (memref<4xi32>, memref<4xi32>) -> memref<4xi32> {
      ^bb0(%read: memref<4xi32>, %write: memref<4xi32>):
        affine.for %i = 0 to 4 {
          %value = affine.load %read[%i] : memref<4xi32>
          affine.store %value, %write[%i] : memref<4xi32>
        }
        taskflow.yield done_writes(%write : memref<4xi32>)
    }
    return
  }

  // Both accesses agree syntactically, but the constant address aliases every
  // firing. Retained interleaving would let each consumer firing observe an
  // earlier producer write rather than the final producer value.
  func.func @repeated_address_negative(
      %input: memref<4xi32> {amoeba.noalias},
      %output: memref<4xi32> {amoeba.noalias}) {
    %scratch = memref.alloca() : memref<4xi32>
    %producer = taskflow.task @RepeatedProducer
        will_reads(%input : memref<4xi32>)
        will_writes(%scratch : memref<4xi32>)
        [original_read_memrefs(%input : memref<4xi32>),
         original_write_memrefs(%scratch : memref<4xi32>)]
        : (memref<4xi32>, memref<4xi32>) -> memref<4xi32> {
      ^bb0(%read: memref<4xi32>, %write: memref<4xi32>):
        affine.for %i = 0 to 4 {
          %value = affine.load %read[%i] : memref<4xi32>
          affine.store %value, %write[0] : memref<4xi32>
        }
        taskflow.yield done_writes(%write : memref<4xi32>)
    }
    %channel = taskflow.channel %producer : memref<4xi32> -> memref<4xi32>
    %consumer = taskflow.task @RepeatedConsumer
        will_reads(%channel : memref<4xi32>)
        will_writes(%output : memref<4xi32>)
        [original_read_memrefs(%scratch : memref<4xi32>),
         original_write_memrefs(%output : memref<4xi32>)]
        : (memref<4xi32>, memref<4xi32>) -> memref<4xi32> {
      ^bb0(%read: memref<4xi32>, %write: memref<4xi32>):
        affine.for %i = 0 to 4 {
          %value = affine.load %read[0] : memref<4xi32>
          affine.store %value, %write[%i] : memref<4xi32>
        }
        taskflow.yield done_writes(%write : memref<4xi32>)
    }
    return
  }
}
