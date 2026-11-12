#loop_annotation = #llvm.loop_annotation<mustProgress = true>
"builtin.module"() ({
  "aie.device"() <{device = 8 : i32, sym_name = "orbit_public_graph_configuration"}> ({
    %0 = "aie.tile"() <{col = 0 : i32, row = 0 : i32}> : () -> index
    %1 = "aie.tile"() <{col = 0 : i32, row = 2 : i32}> : () -> index
    %2 = "aie.lock"(%1) <{init = 1 : i32, lockID = 3 : i32}> : (index) -> index
    %3 = "aie.lock"(%1) <{init = 0 : i32, lockID = 2 : i32}> : (index) -> index
    %4 = "aie.lock"(%1) <{init = 1 : i32, lockID = 1 : i32}> : (index) -> index
    %5 = "aie.lock"(%1) <{init = 0 : i32, lockID = 0 : i32}> : (index) -> index
    %6 = "aie.buffer"(%1) <{aligned = true, sym_name = "buf3"}> : (index) -> memref<768xbf16, 2 : i32>
    %7 = "aie.buffer"(%1) <{aligned = true, sym_name = "buf2"}> : (index) -> memref<1536xbf16, 2 : i32>
    %8 = "aie.buffer"(%1) <{aligned = true, sym_name = "buf1"}> : (index) -> memref<512xf32, 2 : i32>
    %9 = "aie.buffer"(%1) <{aligned = true, sym_name = "buf0"}> : (index) -> memref<1024xf32, 2 : i32>
    %10 = "aie.external_buffer"() <{sym_name = "__air_external_buffer"}> : () -> memref<96x96xf32>
    %11 = "aie.external_buffer"() <{sym_name = "__air_external_buffer_1"}> : () -> memref<96x96xf32>
    %12 = "aie.mem"(%1) ({
      %28 = "arith.constant"() <{value = 1 : i32}> : () -> i32
      %29 = "aie.dma_start"()[^bb1, ^bb3] <{channel = 0 : i32, channel_dir = 1 : i32, repeat_count = 0 : i32}> : () -> i1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      "aie.use_lock"(%5, %28) <{action = 2 : i32}> : (index, i32) -> ()
      "aie.dma_bd"(%8) <{operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, static_len = 512 : i32, static_offset = 0 : i32}> {task_id = 0 : i32} : (memref<512xf32, 2 : i32>) -> ()
      "aie.use_lock"(%4, %28) <{action = 1 : i32}> : (index, i32) -> ()
      "aie.next_bd"()[^bb1] : () -> ()
    ^bb2:  // pred: ^bb3
      "aie.end"() : () -> ()
    ^bb3:  // pred: ^bb0
      %30 = "aie.dma_start"()[^bb4, ^bb2] <{channel = 0 : i32, channel_dir = 0 : i32, repeat_count = 0 : i32}> : () -> i1
    ^bb4:  // 2 preds: ^bb3, ^bb4
      "aie.use_lock"(%2, %28) <{action = 2 : i32}> : (index, i32) -> ()
      "aie.dma_bd"(%9) <{operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, static_len = 512 : i32, static_offset = 512 : i32}> {task_id = 0 : i32} : (memref<1024xf32, 2 : i32>) -> ()
      "aie.use_lock"(%3, %28) <{action = 1 : i32}> : (index, i32) -> ()
      "aie.next_bd"()[^bb4] : () -> ()
    }) : (index) -> index
    %13 = "aie.core"(%1) <{stack_size = 4096 : i32}> ({
      %16 = "arith.constant"() <{value = 1 : i32}> : () -> i32
      %17 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %18 = "arith.constant"() <{value = 80 : index}> : () -> index
      %19 = "arith.constant"() <{value = 96 : index}> : () -> index
      %20 = "arith.constant"() <{value = 64 : index}> : () -> index
      %21 = "arith.constant"() <{value = 1 : index}> : () -> index
      "cf.br"()[^bb1] : () -> ()
    ^bb1:  // 2 preds: ^bb0, ^bb1
      "aie.use_lock"(%4, %16) <{action = 2 : i32}> : (index, i32) -> ()
      "aie.use_lock"(%3, %16) <{action = 2 : i32}> : (index, i32) -> ()
      %22 = "memref.reinterpret_cast"(%8) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 16, 32>, static_strides = array<i64: 32, 1>}> {orbit.allocation = 8 : i64, orbit.logical_offsets = array<i64: 80, 64>, orbit.logical_sizes = array<i64: 16, 32>, orbit.storage_index = 2 : i64} : (memref<512xf32, 2 : i32>) -> memref<16x32xf32, strided<[32, 1]>, 2 : i32>
      %23 = "memref.reinterpret_cast"(%9) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 512>, static_sizes = array<i64: 16, 32>, static_strides = array<i64: 32, 1>}> {orbit.allocation = 7 : i64, orbit.logical_offsets = array<i64: 80, 64>, orbit.logical_sizes = array<i64: 16, 32>, orbit.storage_index = 3 : i64} : (memref<1024xf32, 2 : i32>) -> memref<16x32xf32, strided<[32, 1], offset: 512>, 2 : i32>
      "scf.for"(%18, %19, %21) ({
      ^bb0(%arg8: index):
        "scf.for"(%20, %19, %21) ({
        ^bb0(%arg9: index):
          %24 = "arith.subi"(%arg8, %18) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %25 = "arith.subi"(%arg9, %20) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %26 = "memref.load"(%23, %24, %25) : (memref<16x32xf32, strided<[32, 1], offset: 512>, 2 : i32>, index, index) -> f32
          %27 = "arith.maximumf"(%26, %17) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%27, %22, %24, %25) : (f32, memref<16x32xf32, strided<[32, 1]>, 2 : i32>, index, index) -> ()
          "scf.yield"() : () -> ()
        }) {loop_annotation = #loop_annotation} : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) {loop_annotation = #loop_annotation} : (index, index, index) -> ()
      "aie.use_lock"(%2, %16) <{action = 1 : i32}> : (index, i32) -> ()
      "aie.use_lock"(%5, %16) <{action = 1 : i32}> : (index, i32) -> ()
      "cf.br"()[^bb1] : () -> ()
    }) {air.herd_local_id = array<i64: 0, 0>, air.herd_name = "orbit_public_core_0", air.herd_size = array<i64: 1, 1>} : (index) -> index
    "aie.flow"(%0, %1) <{dest_bundle = 1 : i32, dest_channel = 0 : i32, source_bundle = 1 : i32, source_channel = 0 : i32}> : (index, index) -> ()
    "aie.flow"(%1, %0) <{dest_bundle = 1 : i32, dest_channel = 0 : i32, source_bundle = 1 : i32, source_channel = 0 : i32}> : (index, index) -> ()
    "aie.shim_dma_allocation"(%0) <{channel_dir = 0 : i32, channel_index = 0 : i64, plio = false, sym_name = "air_channel_1"}> : (index) -> ()
    "aie.shim_dma_allocation"(%0) <{channel_dir = 1 : i32, channel_index = 0 : i64, plio = false, sym_name = "air_channel_0"}> : (index) -> ()
    "aie.runtime_sequence"() <{sym_name = "orbit_public_graph_configuration_sequence"}> ({
    ^bb0(%arg4: memref<96x96xbf16>, %arg5: memref<96x96xbf16>, %arg6: memref<96x96xf32>, %arg7: memref<96x96xf32>):
      %14 = "aiex.dma_configure_task_for"() <{alloc = @air_channel_1, issue_token = true}> ({
        "aie.dma_bd"(%arg6) <{operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, static_len = 512 : i32, static_offset = 7744 : i32, static_sizes = array<i64: 16, 32>, static_strides = array<i64: 96, 1>}> : (memref<96x96xf32>) -> ()
        "aie.end"() : () -> ()
      }) : () -> index
      "aiex.dma_start_task"(%14) : (index) -> ()
      %15 = "aiex.dma_configure_task_for"() <{alloc = @air_channel_0}> ({
        "aie.dma_bd"(%arg7) <{operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, static_len = 512 : i32, static_offset = 7744 : i32, static_sizes = array<i64: 16, 32>, static_strides = array<i64: 96, 1>}> : (memref<96x96xf32>) -> ()
        "aie.end"() : () -> ()
      }) : () -> index
      "aiex.dma_start_task"(%15) : (index) -> ()
      "aiex.dma_await_task"(%14) : (index) -> ()
      "aiex.dma_free_task"(%15) : (index) -> ()
    }) : () -> ()
    "aie.end"() : () -> ()
  }) {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>} : () -> ()
  "aie.device"() <{device = 8 : i32, sym_name = "main"}> ({
    "aie.runtime_sequence"() <{sym_name = "heterogeneous_region_0"}> ({
    ^bb0(%arg0: memref<96x96xbf16>, %arg1: memref<96x96xbf16>, %arg2: memref<96x96xf32>, %arg3: memref<96x96xf32>):
      "aiex.configure"() <{symbol = @orbit_public_graph_configuration}> ({
        "aiex.run"(%arg0, %arg1, %arg2, %arg3) <{runtime_sequence_symbol = @orbit_public_graph_configuration_sequence}> : (memref<96x96xbf16>, memref<96x96xbf16>, memref<96x96xf32>, memref<96x96xf32>) -> ()
      }) : () -> ()
    }) : () -> ()
    "aie.end"() : () -> ()
  }) : () -> ()
}) {orbit.program_graph = "1cbd48f862d0cee78e456ae804a46e7e32761f6e5c30fb99b38ab816e5b6c675"} : () -> ()

