// RUN: mlir-amoeba-opt %s --verify-each \
// RUN:   '--prove-static-active-transfer-shapes=function=unknown_index arguments=1 report-output=%t-unknown.json' -o %t-unknown.mlir
// RUN: python3 -c 'import json,sys; p=json.load(open(sys.argv[1]))["proofs"][0]; assert p["status"] == "unknown" and p["capacity_shape"] == [5] and "active_shape" not in p' %t-unknown.json
// RUN: mlir-amoeba-opt %s --verify-each \
// RUN:   '--prove-static-active-transfer-shapes=function=unbounded_counter arguments=1 report-output=%t-unbounded.json' -o %t-unbounded.mlir
// RUN: python3 -c 'import json,sys; p=json.load(open(sys.argv[1]))["proofs"][0]; assert p["status"] == "unknown" and p["capacity_shape"] == [8] and "active_shape" not in p' %t-unbounded.json
// RUN: mlir-amoeba-opt %s --verify-each \
// RUN:   '--prove-static-active-transfer-shapes=function=alias_mismatch arguments=1 report-output=%t-alias.json' -o %t-alias.mlir
// RUN: python3 -c 'import json,sys; p=json.load(open(sys.argv[1]))["proofs"][0]; assert p["status"] == "unknown" and p["capacity_shape"] == [5] and "active_shape" not in p' %t-alias.json
// RUN: mlir-amoeba-opt %s --verify-each \
// RUN:   '--prove-static-active-transfer-shapes=function=out_of_bounds arguments=1 report-output=%t-oob.json' -o %t-oob.mlir
// RUN: python3 -c 'import json,sys; p=json.load(open(sys.argv[1]))["proofs"][0]; assert p["status"] == "unknown" and p["capacity_shape"] == [4] and "active_shape" not in p' %t-oob.json
// RUN: mlir-amoeba-opt %s --verify-each \
// RUN:   '--prove-static-active-transfer-shapes=function=faked_counter_bound arguments=1 report-output=%t-fake-bound.json' -o %t-fake-bound.mlir
// RUN: python3 -c 'import json,sys; p=json.load(open(sys.argv[1]))["proofs"][0]; assert p["status"] == "unknown" and p["capacity_shape"] == [8] and "active_shape" not in p' %t-fake-bound.json
// RUN: mlir-amoeba-opt %s \
// RUN:   '--prove-static-active-transfer-shapes=function=faked_active_shape arguments=1' \
// RUN:   > %t-fake-active.log 2>&1; test $? -ne 0
// RUN: FileCheck %s --check-prefix=FAKE-ACTIVE --input-file=%t-fake-active.log

// Every unsupported case reports unknown and omits active_shape. The source
// capacity remains present in the input IR and in the proof report.
module {
  func.func @unknown_index(
      %bound: index,
      %data: memref<?xi32> {
        amoeba.logical_transfer_shape = array<i64: 5>, amoeba.noalias
      }) attributes {
        amoeba.input0_caller_evidence_path = "fixture-caller.mlir",
        amoeba.input0_caller_function = "main",
        amoeba.input0_caller_noalias_proof_mode = "prepared-external-caller",
        amoeba.input0_caller_noalias_proven,
        amoeba.input0_caller_prepared_input_path = "fixture-prepared.mlir",
        amoeba.input0_caller_static_bound = 5 : i64,
        amoeba.static_bound.arg.0 = 5 : i64
      } {
    %done = taskflow.task @Unknown
        will_reads(%data : memref<?xi32>)
        [original_read_memrefs(%data : memref<?xi32>)]
        : (memref<?xi32>) -> (memref<?xi32>) {
      ^bb0(%read: memref<?xi32>):
        neura.kernel inputs(%read : memref<?xi32>)
            attributes {accelerator = "neura", dataflow_mode = "predicate"} {
          ^bb0(%input0: !neura.data<memref<?xi32>, i1>):
            %start = "neura.constant"() <{value = 0 : index}> : () -> !neura.data<index, i1>
            %reserved = neura.reserve : !neura.data<index, i1>
            %index = "neura.phi"(%reserved, %start) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
            %value = neura.load_indexed [%index : !neura.data<index, i1>] {lhs_value = "%input0"} : !neura.data<i32, i1>
            neura.yield {yield_type = "void"}
        }
        taskflow.yield done_reads(%read : memref<?xi32>)
    }
    return
  }

  func.func @unbounded_counter(
      %runtime_bound: index,
      %data: memref<?xi32> {
        amoeba.logical_transfer_shape = array<i64: 8>, amoeba.noalias
      }) attributes {
        amoeba.input0_caller_evidence_path = "fixture-caller.mlir",
        amoeba.input0_caller_function = "main",
        amoeba.input0_caller_noalias_proof_mode = "prepared-external-caller",
        amoeba.input0_caller_noalias_proven,
        amoeba.input0_caller_prepared_input_path = "fixture-prepared.mlir"
      } {
    %done = taskflow.task @Unbounded
        will_reads(%data : memref<?xi32>) value_inputs(%runtime_bound : index)
        [original_read_memrefs(%data : memref<?xi32>)]
        : (memref<?xi32>, index) -> (memref<?xi32>) {
      ^bb0(%read: memref<?xi32>, %bound: index):
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %loop = taskflow.counter from %c0 to %bound step %c1
            attributes {counter_dynamism = "symbol_dynamic", counter_hierarchy = "leaf", counter_id = 0 : i32} : index
        neura.kernel inputs(%read, %bound : memref<?xi32>, index)
            attributes {accelerator = "neura", dataflow_mode = "predicate"} {
          ^bb0(%input0: !neura.data<memref<?xi32>, i1>, %input1: !neura.data<index, i1>):
            %index = neura.counter attributes {counter_dynamism = "symbol_dynamic", counter_hierarchy = "leaf", counter_id = 0 : i32, lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = "%input1"} -> !neura.data<index, i1>
            %value = neura.load_indexed [%index : !neura.data<index, i1>] {lhs_value = "%input0"} : !neura.data<i32, i1>
            neura.yield {yield_type = "void"}
        }
        taskflow.yield done_reads(%read : memref<?xi32>)
    }
    return
  }

  func.func @alias_mismatch(
      %bound: index,
      %actual: memref<?xi32> {
        amoeba.logical_transfer_shape = array<i64: 5>, amoeba.noalias
      },
      %declared: memref<?xi32> {
        amoeba.logical_transfer_shape = array<i64: 5>, amoeba.noalias
      }) attributes {
        amoeba.input0_caller_evidence_path = "fixture-caller.mlir",
        amoeba.input0_caller_function = "main",
        amoeba.input0_caller_noalias_proof_mode = "prepared-external-caller",
        amoeba.input0_caller_noalias_proven,
        amoeba.input0_caller_prepared_input_path = "fixture-prepared.mlir",
        amoeba.input0_caller_static_bound = 5 : i64,
        amoeba.static_bound.arg.0 = 5 : i64
      } {
    %done = taskflow.task @Alias
        will_reads(%actual : memref<?xi32>)
        [original_read_memrefs(%declared : memref<?xi32>)]
        : (memref<?xi32>) -> (memref<?xi32>) {
      ^bb0(%read: memref<?xi32>):
        neura.kernel inputs(%read : memref<?xi32>)
            attributes {accelerator = "neura", dataflow_mode = "predicate"} {
          ^bb0(%input0: !neura.data<memref<?xi32>, i1>):
            %index = "neura.constant"() <{value = 0 : index}> : () -> !neura.data<index, i1>
            %value = neura.load_indexed [%index : !neura.data<index, i1>] {lhs_value = "%input0"} : !neura.data<i32, i1>
            neura.yield {yield_type = "void"}
        }
        taskflow.yield done_reads(%read : memref<?xi32>)
    }
    return
  }

  func.func @out_of_bounds(
      %bound: index,
      %data: memref<?xi32> {
        amoeba.logical_transfer_shape = array<i64: 4>, amoeba.noalias
      }) attributes {
        amoeba.input0_caller_evidence_path = "fixture-caller.mlir",
        amoeba.input0_caller_function = "main",
        amoeba.input0_caller_noalias_proof_mode = "prepared-external-caller",
        amoeba.input0_caller_noalias_proven,
        amoeba.input0_caller_prepared_input_path = "fixture-prepared.mlir",
        amoeba.input0_caller_static_bound = 5 : i64,
        amoeba.static_bound.arg.0 = 5 : i64
      } {
    %done = taskflow.task @OutOfBounds
        will_reads(%data : memref<?xi32>)
        [original_read_memrefs(%data : memref<?xi32>)]
        : (memref<?xi32>) -> (memref<?xi32>) {
      ^bb0(%read: memref<?xi32>):
        neura.kernel inputs(%read : memref<?xi32>)
            attributes {accelerator = "neura", dataflow_mode = "predicate"} {
          ^bb0(%input0: !neura.data<memref<?xi32>, i1>):
            %index = "neura.constant"() <{value = 4 : index}> : () -> !neura.data<index, i1>
            %value = neura.load_indexed [%index : !neura.data<index, i1>] {lhs_value = "%input0"} : !neura.data<i32, i1>
            neura.yield {yield_type = "void"}
        }
        taskflow.yield done_reads(%read : memref<?xi32>)
    }
    return
  }

  func.func @faked_counter_bound(
      %bound: index,
      %data: memref<?xi32> {
        amoeba.logical_transfer_shape = array<i64: 8>, amoeba.noalias
      }) attributes {
        amoeba.input0_caller_evidence_path = "fixture-caller.mlir",
        amoeba.input0_caller_function = "main",
        amoeba.input0_caller_noalias_proof_mode = "prepared-external-caller",
        amoeba.input0_caller_noalias_proven,
        amoeba.input0_caller_prepared_input_path = "fixture-prepared.mlir",
        amoeba.input0_caller_static_bound = 5 : i64,
        amoeba.static_bound.arg.0 = 5 : i64
      } {
    %done = taskflow.task @FakeBound
        will_reads(%data : memref<?xi32>) value_inputs(%bound : index)
        [original_read_memrefs(%data : memref<?xi32>)]
        : (memref<?xi32>, index) -> (memref<?xi32>) {
      ^bb0(%read: memref<?xi32>, %task_bound: index):
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %loop = taskflow.counter from %c0 to %c4 step %c1
            attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 0 : i32} : index
        neura.kernel inputs(%read, %task_bound : memref<?xi32>, index)
            attributes {accelerator = "neura", dataflow_mode = "predicate"} {
          ^bb0(%input0: !neura.data<memref<?xi32>, i1>, %input1: !neura.data<index, i1>):
            %index = neura.counter attributes {counter_dynamism = "symbol_bound", counter_hierarchy = "leaf", counter_id = 0 : i32, lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 5 : index} -> !neura.data<index, i1>
            %value = neura.load_indexed [%index : !neura.data<index, i1>] {lhs_value = "%input0"} : !neura.data<i32, i1>
            neura.yield {yield_type = "void"}
        }
        taskflow.yield done_reads(%read : memref<?xi32>)
    }
    return
  }

  func.func @faked_active_shape(
      %bound: index,
      %data: memref<?xi32> {
        amoeba.logical_transfer_shape = array<i64: 8>,
        amoeba.active_transfer_shape = array<i64: 4>, amoeba.noalias
      }) attributes {
        amoeba.input0_caller_evidence_path = "fixture-caller.mlir",
        amoeba.input0_caller_function = "main",
        amoeba.input0_caller_noalias_proof_mode = "prepared-external-caller",
        amoeba.input0_caller_noalias_proven,
        amoeba.input0_caller_prepared_input_path = "fixture-prepared.mlir",
        amoeba.input0_caller_static_bound = 5 : i64,
        amoeba.static_bound.arg.0 = 5 : i64
      } {
    return
  }
}

// FAKE-ACTIVE: already carries a stale or unverified active transfer fact
