#loop_annotation = #llvm.loop_annotation<mustProgress = true>
"builtin.module"() ({
  "aie.device"() <{device = 8 : i32, sym_name = "orbit_public_graph_configuration"}> ({
    %0 = "aie.tile"() <{col = 0 : i32, row = 0 : i32}> : () -> index
    %1 = "aie.tile"() <{col = 0 : i32, row = 2 : i32}> : () -> index
    %2 = "aie.lock"(%1) <{init = 1 : i32, lockID = 7 : i32}> : (index) -> index
    %3 = "aie.lock"(%1) <{init = 0 : i32, lockID = 6 : i32}> : (index) -> index
    %4 = "aie.lock"(%1) <{init = 1 : i32, lockID = 5 : i32}> : (index) -> index
    %5 = "aie.lock"(%1) <{init = 0 : i32, lockID = 4 : i32}> : (index) -> index
    %6 = "aie.lock"(%1) <{init = 1 : i32, lockID = 3 : i32}> : (index) -> index
    %7 = "aie.lock"(%1) <{init = 0 : i32, lockID = 2 : i32}> : (index) -> index
    %8 = "aie.lock"(%1) <{init = 1 : i32, lockID = 1 : i32}> : (index) -> index
    %9 = "aie.lock"(%1) <{init = 0 : i32, lockID = 0 : i32}> : (index) -> index
    %10 = "aie.buffer"(%1) <{aligned = true, sym_name = "buf6"}> : (index) -> memref<256xf32, 2 : i32>
    %11 = "aie.buffer"(%1) <{aligned = true, sym_name = "buf5"}> : (index) -> memref<1024xbf16, 2 : i32>
    %12 = "aie.buffer"(%1) <{aligned = true, sym_name = "buf4"}> : (index) -> memref<1024xbf16, 2 : i32>
    %13 = "aie.buffer"(%1) <{aligned = true, sym_name = "buf3"}> : (index) -> memref<1024xbf16, 2 : i32>
    %14 = "aie.buffer"(%1) <{aligned = true, sym_name = "buf2"}> : (index) -> memref<1024xbf16, 2 : i32>
    %15 = "aie.buffer"(%1) <{aligned = true, sym_name = "buf1"}> : (index) -> memref<256xf32, 2 : i32>
    %16 = "aie.buffer"(%1) <{aligned = true, sym_name = "buf0"}> : (index) -> memref<512xf32, 2 : i32>
    %17 = "aie.external_buffer"() <{sym_name = "__air_external_buffer"}> : () -> memref<64x64xbf16>
    %18 = "aie.external_buffer"() <{sym_name = "__air_external_buffer_1"}> : () -> memref<64x64xbf16>
    %19 = "aie.external_buffer"() <{sym_name = "__air_external_buffer_2"}> : () -> memref<64x64xf32>
    %20 = "aie.mem"(%1) ({
      %132 = "arith.constant"() <{value = 1 : i32}> : () -> i32
      %133 = "aie.dma_start"()[^bb1, ^bb3] <{channel = 0 : i32, channel_dir = 1 : i32, repeat_count = 0 : i32}> : () -> i1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      "aie.use_lock"(%9, %132) <{action = 2 : i32}> : (index, i32) -> ()
      "aie.dma_bd"(%16) <{operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, static_len = 256 : i32, static_offset = 0 : i32}> {task_id = 0 : i32} : (memref<512xf32, 2 : i32>) -> ()
      "aie.use_lock"(%8, %132) <{action = 1 : i32}> : (index, i32) -> ()
      "aie.next_bd"()[^bb1] : () -> ()
    ^bb2:  // pred: ^bb5
      "aie.end"() : () -> ()
    ^bb3:  // pred: ^bb0
      %134 = "aie.dma_start"()[^bb4, ^bb5] <{channel = 1 : i32, channel_dir = 1 : i32, repeat_count = 0 : i32}> : () -> i1
    ^bb4:  // 2 preds: ^bb3, ^bb4
      "aie.use_lock"(%7, %132) <{action = 2 : i32}> : (index, i32) -> ()
      "aie.dma_bd"(%16) <{operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, static_len = 256 : i32, static_offset = 0 : i32}> {task_id = 0 : i32} : (memref<512xf32, 2 : i32>) -> ()
      "aie.use_lock"(%6, %132) <{action = 1 : i32}> : (index, i32) -> ()
      "aie.next_bd"()[^bb4] : () -> ()
    ^bb5:  // pred: ^bb3
      %135 = "aie.dma_start"()[^bb6, ^bb2] <{channel = 0 : i32, channel_dir = 0 : i32, repeat_count = 0 : i32}> : () -> i1
    ^bb6:  // 2 preds: ^bb5, ^bb7
      "aie.use_lock"(%4, %132) <{action = 2 : i32}> : (index, i32) -> ()
      "aie.dma_bd"(%13) <{operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>, static_len = 1024 : i32, static_offset = 0 : i32}> {task_id = 0 : i32} : (memref<1024xbf16, 2 : i32>) -> ()
      "aie.use_lock"(%5, %132) <{action = 1 : i32}> : (index, i32) -> ()
      "aie.next_bd"()[^bb7] : () -> ()
    ^bb7:  // pred: ^bb6
      "aie.use_lock"(%2, %132) <{action = 2 : i32}> : (index, i32) -> ()
      "aie.dma_bd"(%14) <{operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>, static_len = 1024 : i32, static_offset = 0 : i32}> {task_id = 0 : i32} : (memref<1024xbf16, 2 : i32>) -> ()
      "aie.use_lock"(%3, %132) <{action = 1 : i32}> : (index, i32) -> ()
      "aie.next_bd"()[^bb6] : () -> ()
    }) : (index) -> index
    %21 = "aie.core"(%1) <{stack_size = 4096 : i32}> ({
      %30 = "arith.constant"() <{value = 1 : i32}> : () -> i32
      %31 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %32 = "arith.constant"() <{value = 16 : index}> : () -> index
      %33 = "arith.constant"() <{value = 64 : index}> : () -> index
      %34 = "arith.constant"() <{value = 8 : index}> : () -> index
      %35 = "arith.constant"() <{value = 4 : index}> : () -> index
      %36 = "arith.constant"() <{value = 2 : index}> : () -> index
      %37 = "arith.constant"() <{value = 1 : index}> : () -> index
      %38 = "arith.constant"() <{value = 0 : index}> : () -> index
      "cf.br"()[^bb1] : () -> ()
    ^bb1:  // 2 preds: ^bb0, ^bb1
      "aie.use_lock"(%6, %30) <{action = 2 : i32}> : (index, i32) -> ()
      "aie.use_lock"(%8, %30) <{action = 2 : i32}> : (index, i32) -> ()
      "aie.use_lock"(%5, %30) <{action = 2 : i32}> : (index, i32) -> ()
      "aie.use_lock"(%3, %30) <{action = 2 : i32}> : (index, i32) -> ()
      %39 = "memref.reinterpret_cast"(%16) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 16, 16>, static_strides = array<i64: 16, 1>}> {orbit.allocation = 6 : i64, orbit.logical_offsets = array<i64: 0, 0>, orbit.logical_sizes = array<i64: 16, 16>, orbit.storage_index = 3 : i64} : (memref<512xf32, 2 : i32>) -> memref<16x16xf32, strided<[16, 1]>, 2 : i32>
      "scf.for"(%38, %32, %37) ({
      ^bb0(%arg27: index):
        "scf.for"(%38, %32, %37) ({
        ^bb0(%arg28: index):
          "memref.store"(%31, %39, %arg27, %arg28) {orbit.source_initializer = 0 : i64} : (f32, memref<16x16xf32, strided<[16, 1]>, 2 : i32>, index, index) -> ()
          "scf.yield"() : () -> ()
        }) {loop_annotation = #loop_annotation} : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) {loop_annotation = #loop_annotation} : (index, index, index) -> ()
      %40 = "memref.reinterpret_cast"(%13) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 16, 64>, static_strides = array<i64: 64, 1>}> {orbit.allocation = 4 : i64, orbit.logical_offsets = array<i64: 0, 0>, orbit.logical_sizes = array<i64: 16, 64>, orbit.storage_index = 0 : i64} : (memref<1024xbf16, 2 : i32>) -> memref<16x64xbf16, strided<[64, 1]>, 2 : i32>
      %41 = "memref.reinterpret_cast"(%14) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 64, 16>, static_strides = array<i64: 16, 1>}> {orbit.allocation = 5 : i64, orbit.logical_offsets = array<i64: 0, 0>, orbit.logical_sizes = array<i64: 64, 16>, orbit.storage_index = 1 : i64} : (memref<1024xbf16, 2 : i32>) -> memref<64x16xbf16, strided<[16, 1]>, 2 : i32>
      "scf.for"(%38, %32, %37) ({
      ^bb0(%arg25: index):
        "scf.for"(%38, %33, %37) ({
        ^bb0(%arg26: index):
          %121 = "memref.load"(%40, %arg25, %arg26) : (memref<16x64xbf16, strided<[64, 1]>, 2 : i32>, index, index) -> bf16
          %122 = "arith.divui"(%arg26, %34) : (index, index) -> index
          %123 = "arith.divui"(%arg25, %35) : (index, index) -> index
          %124 = "arith.muli"(%123, %34) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %125 = "arith.addi"(%124, %122) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %126 = "arith.remui"(%arg26, %34) : (index, index) -> index
          %127 = "arith.remui"(%arg25, %35) : (index, index) -> index
          %128 = "arith.muli"(%125, %35) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %129 = "arith.addi"(%128, %127) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %130 = "arith.muli"(%129, %34) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %131 = "arith.addi"(%130, %126) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          "memref.store"(%121, %12, %131) : (bf16, memref<1024xbf16, 2 : i32>, index) -> ()
          "scf.yield"() : () -> ()
        }) {loop_annotation = #loop_annotation} : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) {loop_annotation = #loop_annotation} : (index, index, index) -> ()
      "scf.for"(%38, %33, %37) ({
      ^bb0(%arg23: index):
        "scf.for"(%38, %32, %37) ({
        ^bb0(%arg24: index):
          %110 = "memref.load"(%41, %arg23, %arg24) : (memref<64x16xbf16, strided<[16, 1]>, 2 : i32>, index, index) -> bf16
          %111 = "arith.divui"(%arg24, %34) : (index, index) -> index
          %112 = "arith.divui"(%arg23, %34) : (index, index) -> index
          %113 = "arith.muli"(%112, %36) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %114 = "arith.addi"(%113, %111) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %115 = "arith.remui"(%arg24, %34) : (index, index) -> index
          %116 = "arith.remui"(%arg23, %34) : (index, index) -> index
          %117 = "arith.muli"(%114, %34) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %118 = "arith.addi"(%117, %116) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %119 = "arith.muli"(%118, %34) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %120 = "arith.addi"(%119, %115) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          "memref.store"(%110, %11, %120) : (bf16, memref<1024xbf16, 2 : i32>, index) -> ()
          "scf.yield"() : () -> ()
        }) {loop_annotation = #loop_annotation} : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) {loop_annotation = #loop_annotation} : (index, index, index) -> ()
      "scf.for"(%38, %32, %37) ({
      ^bb0(%arg21: index):
        "scf.for"(%38, %32, %37) ({
        ^bb0(%arg22: index):
          %99 = "memref.load"(%39, %arg21, %arg22) : (memref<16x16xf32, strided<[16, 1]>, 2 : i32>, index, index) -> f32
          %100 = "arith.divui"(%arg22, %34) : (index, index) -> index
          %101 = "arith.divui"(%arg21, %35) : (index, index) -> index
          %102 = "arith.muli"(%101, %36) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %103 = "arith.addi"(%102, %100) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %104 = "arith.remui"(%arg22, %34) : (index, index) -> index
          %105 = "arith.remui"(%arg21, %35) : (index, index) -> index
          %106 = "arith.muli"(%103, %35) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %107 = "arith.addi"(%106, %105) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %108 = "arith.muli"(%107, %34) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %109 = "arith.addi"(%108, %104) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          "memref.store"(%99, %10, %109) : (f32, memref<256xf32, 2 : i32>, index) -> ()
          "scf.yield"() : () -> ()
        }) {loop_annotation = #loop_annotation} : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) {loop_annotation = #loop_annotation} : (index, index, index) -> ()
      "func.call"(%12, %11, %10) <{callee = @orbit_bf16_mmul_16x64x16}> {orbit.matrix_contract = "bf16-f32-accumulate-v1", orbit.matrix_event = 3 : i64, orbit.matrix_local_layout_payload_bytes = 6144 : i64} : (memref<1024xbf16, 2 : i32>, memref<1024xbf16, 2 : i32>, memref<256xf32, 2 : i32>) -> ()
      "scf.for"(%38, %32, %37) ({
      ^bb0(%arg19: index):
        "scf.for"(%38, %32, %37) ({
        ^bb0(%arg20: index):
          %88 = "arith.divui"(%arg20, %34) : (index, index) -> index
          %89 = "arith.divui"(%arg19, %35) : (index, index) -> index
          %90 = "arith.muli"(%89, %36) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %91 = "arith.addi"(%90, %88) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %92 = "arith.remui"(%arg20, %34) : (index, index) -> index
          %93 = "arith.remui"(%arg19, %35) : (index, index) -> index
          %94 = "arith.muli"(%91, %35) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %95 = "arith.addi"(%94, %93) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %96 = "arith.muli"(%95, %34) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %97 = "arith.addi"(%96, %92) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %98 = "memref.load"(%10, %97) : (memref<256xf32, 2 : i32>, index) -> f32
          "memref.store"(%98, %39, %arg19, %arg20) : (f32, memref<16x16xf32, strided<[16, 1]>, 2 : i32>, index, index) -> ()
          "scf.yield"() : () -> ()
        }) {loop_annotation = #loop_annotation} : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) {loop_annotation = #loop_annotation} : (index, index, index) -> ()
      "aie.use_lock"(%4, %30) <{action = 1 : i32}> : (index, i32) -> ()
      "aie.use_lock"(%5, %30) <{action = 2 : i32}> : (index, i32) -> ()
      "aie.use_lock"(%2, %30) <{action = 1 : i32}> : (index, i32) -> ()
      "aie.use_lock"(%3, %30) <{action = 2 : i32}> : (index, i32) -> ()
      %42 = "memref.reinterpret_cast"(%16) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 16, 16>, static_strides = array<i64: 16, 1>}> {orbit.allocation = 6 : i64, orbit.logical_offsets = array<i64: 0, 16>, orbit.logical_sizes = array<i64: 16, 16>, orbit.storage_index = 3 : i64} : (memref<512xf32, 2 : i32>) -> memref<16x16xf32, strided<[16, 1]>, 2 : i32>
      "scf.for"(%38, %32, %37) ({
      ^bb0(%arg17: index):
        "scf.for"(%38, %32, %37) ({
        ^bb0(%arg18: index):
          "memref.store"(%31, %42, %arg17, %arg18) {orbit.source_initializer = 0 : i64} : (f32, memref<16x16xf32, strided<[16, 1]>, 2 : i32>, index, index) -> ()
          "scf.yield"() : () -> ()
        }) {loop_annotation = #loop_annotation} : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) {loop_annotation = #loop_annotation} : (index, index, index) -> ()
      %43 = "memref.reinterpret_cast"(%14) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 64, 16>, static_strides = array<i64: 16, 1>}> {orbit.allocation = 5 : i64, orbit.logical_offsets = array<i64: 0, 16>, orbit.logical_sizes = array<i64: 64, 16>, orbit.storage_index = 1 : i64} : (memref<1024xbf16, 2 : i32>) -> memref<64x16xbf16, strided<[16, 1]>, 2 : i32>
      "scf.for"(%38, %32, %37) ({
      ^bb0(%arg15: index):
        "scf.for"(%38, %33, %37) ({
        ^bb0(%arg16: index):
          %77 = "memref.load"(%40, %arg15, %arg16) : (memref<16x64xbf16, strided<[64, 1]>, 2 : i32>, index, index) -> bf16
          %78 = "arith.divui"(%arg16, %34) : (index, index) -> index
          %79 = "arith.divui"(%arg15, %35) : (index, index) -> index
          %80 = "arith.muli"(%79, %34) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %81 = "arith.addi"(%80, %78) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %82 = "arith.remui"(%arg16, %34) : (index, index) -> index
          %83 = "arith.remui"(%arg15, %35) : (index, index) -> index
          %84 = "arith.muli"(%81, %35) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %85 = "arith.addi"(%84, %83) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %86 = "arith.muli"(%85, %34) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %87 = "arith.addi"(%86, %82) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          "memref.store"(%77, %12, %87) : (bf16, memref<1024xbf16, 2 : i32>, index) -> ()
          "scf.yield"() : () -> ()
        }) {loop_annotation = #loop_annotation} : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) {loop_annotation = #loop_annotation} : (index, index, index) -> ()
      "scf.for"(%38, %33, %37) ({
      ^bb0(%arg13: index):
        "scf.for"(%38, %32, %37) ({
        ^bb0(%arg14: index):
          %66 = "memref.load"(%43, %arg13, %arg14) : (memref<64x16xbf16, strided<[16, 1]>, 2 : i32>, index, index) -> bf16
          %67 = "arith.divui"(%arg14, %34) : (index, index) -> index
          %68 = "arith.divui"(%arg13, %34) : (index, index) -> index
          %69 = "arith.muli"(%68, %36) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %70 = "arith.addi"(%69, %67) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %71 = "arith.remui"(%arg14, %34) : (index, index) -> index
          %72 = "arith.remui"(%arg13, %34) : (index, index) -> index
          %73 = "arith.muli"(%70, %34) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %74 = "arith.addi"(%73, %72) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %75 = "arith.muli"(%74, %34) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %76 = "arith.addi"(%75, %71) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          "memref.store"(%66, %11, %76) : (bf16, memref<1024xbf16, 2 : i32>, index) -> ()
          "scf.yield"() : () -> ()
        }) {loop_annotation = #loop_annotation} : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) {loop_annotation = #loop_annotation} : (index, index, index) -> ()
      "scf.for"(%38, %32, %37) ({
      ^bb0(%arg11: index):
        "scf.for"(%38, %32, %37) ({
        ^bb0(%arg12: index):
          %55 = "memref.load"(%42, %arg11, %arg12) : (memref<16x16xf32, strided<[16, 1]>, 2 : i32>, index, index) -> f32
          %56 = "arith.divui"(%arg12, %34) : (index, index) -> index
          %57 = "arith.divui"(%arg11, %35) : (index, index) -> index
          %58 = "arith.muli"(%57, %36) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %59 = "arith.addi"(%58, %56) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %60 = "arith.remui"(%arg12, %34) : (index, index) -> index
          %61 = "arith.remui"(%arg11, %35) : (index, index) -> index
          %62 = "arith.muli"(%59, %35) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %63 = "arith.addi"(%62, %61) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %64 = "arith.muli"(%63, %34) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %65 = "arith.addi"(%64, %60) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          "memref.store"(%55, %10, %65) : (f32, memref<256xf32, 2 : i32>, index) -> ()
          "scf.yield"() : () -> ()
        }) {loop_annotation = #loop_annotation} : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) {loop_annotation = #loop_annotation} : (index, index, index) -> ()
      "func.call"(%12, %11, %10) <{callee = @orbit_bf16_mmul_16x64x16}> {orbit.matrix_contract = "bf16-f32-accumulate-v1", orbit.matrix_event = 8 : i64, orbit.matrix_local_layout_payload_bytes = 6144 : i64} : (memref<1024xbf16, 2 : i32>, memref<1024xbf16, 2 : i32>, memref<256xf32, 2 : i32>) -> ()
      "scf.for"(%38, %32, %37) ({
      ^bb0(%arg9: index):
        "scf.for"(%38, %32, %37) ({
        ^bb0(%arg10: index):
          %44 = "arith.divui"(%arg10, %34) : (index, index) -> index
          %45 = "arith.divui"(%arg9, %35) : (index, index) -> index
          %46 = "arith.muli"(%45, %36) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %47 = "arith.addi"(%46, %44) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %48 = "arith.remui"(%arg10, %34) : (index, index) -> index
          %49 = "arith.remui"(%arg9, %35) : (index, index) -> index
          %50 = "arith.muli"(%47, %35) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %51 = "arith.addi"(%50, %49) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %52 = "arith.muli"(%51, %34) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %53 = "arith.addi"(%52, %48) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %54 = "memref.load"(%10, %53) : (memref<256xf32, 2 : i32>, index) -> f32
          "memref.store"(%54, %42, %arg9, %arg10) : (f32, memref<16x16xf32, strided<[16, 1]>, 2 : i32>, index, index) -> ()
          "scf.yield"() : () -> ()
        }) {loop_annotation = #loop_annotation} : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) {loop_annotation = #loop_annotation} : (index, index, index) -> ()
      "aie.use_lock"(%4, %30) <{action = 1 : i32}> : (index, i32) -> ()
      "aie.use_lock"(%2, %30) <{action = 1 : i32}> : (index, i32) -> ()
      "aie.use_lock"(%7, %30) <{action = 1 : i32}> : (index, i32) -> ()
      "aie.use_lock"(%9, %30) <{action = 1 : i32}> : (index, i32) -> ()
      "cf.br"()[^bb1] : () -> ()
    }) {air.herd_local_id = array<i64: 0, 0>, air.herd_name = "orbit_public_core_0", air.herd_size = array<i64: 1, 1>} : (index) -> index
    "func.func"() <{function_type = (memref<1024xbf16, 2 : i32>, memref<1024xbf16, 2 : i32>, memref<256xf32, 2 : i32>) -> (), sym_name = "orbit_bf16_mmul_16x64x16", sym_visibility = "private"}> ({
    }) {link_with = "orbit_bf16_mmul_16x64x16.o", llvm.emit_c_interface, orbit.matrix_shape = array<i64: 16, 64, 16>} : () -> ()
    "aie.packet_flow"() <{ID = 0 : i8}> ({
      "aie.packet_source"(%0) <{bundle = 1 : i32, channel = 0 : i32}> : (index) -> ()
      "aie.packet_dest"(%1) <{bundle = 1 : i32, channel = 0 : i32}> : (index) -> ()
      "aie.end"() : () -> ()
    }) : () -> ()
    "aie.packet_flow"() <{ID = 1 : i8}> ({
      "aie.packet_source"(%0) <{bundle = 1 : i32, channel = 1 : i32}> : (index) -> ()
      "aie.packet_dest"(%1) <{bundle = 1 : i32, channel = 0 : i32}> : (index) -> ()
      "aie.end"() : () -> ()
    }) : () -> ()
    "aie.flow"(%1, %0) <{dest_bundle = 1 : i32, dest_channel = 0 : i32, source_bundle = 1 : i32, source_channel = 0 : i32}> : (index, index) -> ()
    "aie.flow"(%1, %0) <{dest_bundle = 1 : i32, dest_channel = 1 : i32, source_bundle = 1 : i32, source_channel = 1 : i32}> : (index, index) -> ()
    "aie.shim_dma_allocation"(%0) <{channel_dir = 0 : i32, channel_index = 0 : i64, plio = false, sym_name = "air_channel_2"}> : (index) -> ()
    "aie.shim_dma_allocation"(%0) <{channel_dir = 0 : i32, channel_index = 1 : i64, plio = false, sym_name = "air_channel_5"}> : (index) -> ()
    "aie.shim_dma_allocation"(%0) <{channel_dir = 1 : i32, channel_index = 0 : i64, plio = false, sym_name = "air_channel_0"}> : (index) -> ()
    "aie.shim_dma_allocation"(%0) <{channel_dir = 1 : i32, channel_index = 1 : i64, plio = false, sym_name = "air_channel_1"}> : (index) -> ()
    "aie.runtime_sequence"() <{sym_name = "orbit_public_graph_configuration_sequence"}> ({
    ^bb0(%arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xf32>, %arg7: memref<64x64xf32>):
      %22 = "arith.constant"() <{value = 0 : index}> : () -> index
      %23 = "arith.constant"() <{value = 2 : index}> : () -> index
      %24 = "arith.constant"() <{value = 1 : index}> : () -> index
      "scf.for"(%22, %23, %24) ({
      ^bb0(%arg8: index):
        %29 = "aiex.dma_configure_task_for"() <{alloc = @air_channel_0}> ({
          "aie.dma_bd"(%arg4) <{operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>, static_len = 1024 : i32, static_offset = 0 : i32, static_sizes = array<i64: 16, 64>, static_strides = array<i64: 64, 1>}> : (memref<64x64xbf16>) -> ()
          "aie.end"() : () -> ()
        }) : () -> index
        "aiex.dma_start_task"(%29) : (index) -> ()
        "aiex.dma_free_task"(%29) : (index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      %25 = "aiex.dma_configure_task_for"() <{alloc = @air_channel_2, issue_token = true}> ({
        "aie.dma_bd"(%arg7) <{operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, static_len = 256 : i32, static_offset = 0 : i32, static_sizes = array<i64: 16, 16>, static_strides = array<i64: 64, 1>}> : (memref<64x64xf32>) -> ()
        "aie.end"() : () -> ()
      }) {air.preserve_shim_dma_order} : () -> index
      "aiex.dma_start_task"(%25) : (index) -> ()
      %26 = "aiex.dma_configure_task_for"() <{alloc = @air_channel_5, issue_token = true}> ({
        "aie.dma_bd"(%arg7) <{operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, static_len = 256 : i32, static_offset = 16 : i32, static_sizes = array<i64: 16, 16>, static_strides = array<i64: 64, 1>}> : (memref<64x64xf32>) -> ()
        "aie.end"() : () -> ()
      }) {air.preserve_shim_dma_order} : () -> index
      "aiex.dma_start_task"(%26) : (index) -> ()
      %27 = "aiex.dma_configure_task_for"() <{alloc = @air_channel_1}> ({
        "aie.dma_bd"(%arg5) <{operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>, static_len = 1024 : i32, static_offset = 0 : i32, static_sizes = array<i64: 64, 16>, static_strides = array<i64: 64, 1>}> : (memref<64x64xbf16>) -> ()
        "aie.end"() : () -> ()
      }) : () -> index
      "aiex.dma_start_task"(%27) : (index) -> ()
      %28 = "aiex.dma_configure_task_for"() <{alloc = @air_channel_1}> ({
        "aie.dma_bd"(%arg5) <{operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>, static_len = 1024 : i32, static_offset = 16 : i32, static_sizes = array<i64: 64, 16>, static_strides = array<i64: 64, 1>}> : (memref<64x64xbf16>) -> ()
        "aie.end"() : () -> ()
      }) : () -> index
      "aiex.dma_start_task"(%28) : (index) -> ()
      "aiex.dma_await_task"(%25) : (index) -> ()
      "aiex.dma_await_task"(%26) : (index) -> ()
      "aiex.dma_free_task"(%27) : (index) -> ()
      "aiex.dma_free_task"(%28) : (index) -> ()
    }) : () -> ()
    "aie.end"() : () -> ()
  }) {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>} : () -> ()
  "aie.device"() <{device = 8 : i32, sym_name = "main"}> ({
    "aie.runtime_sequence"() <{sym_name = "heterogeneous_region_0"}> ({
    ^bb0(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xf32>, %arg3: memref<64x64xf32>):
      "aiex.configure"() <{symbol = @orbit_public_graph_configuration}> ({
        "aiex.run"(%arg0, %arg1, %arg2, %arg3) <{runtime_sequence_symbol = @orbit_public_graph_configuration_sequence}> : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xf32>, memref<64x64xf32>) -> ()
      }) : () -> ()
    }) : () -> ()
    "aie.end"() : () -> ()
  }) : () -> ()
}) {orbit.program_graph = "a7dd0883c4039fa783daaa199eb67382bb4641746925a3cf1eed9083c60dc86e"} : () -> ()

