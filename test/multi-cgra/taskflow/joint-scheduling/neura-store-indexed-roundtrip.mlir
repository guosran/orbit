// RUN: mlir-amoeba-opt %s --mlir-print-op-generic -o %t.generic
// RUN: mlir-amoeba-opt %t.generic -o %t.custom
// RUN: mlir-amoeba-opt %t.custom --mlir-print-op-generic -o %t.roundtrip
// RUN: diff %t.generic %t.roundtrip
// The omitted value form is emitted by FoldConstantPass and appears in the
// real LLaMA gate/up source proof. Generic -> custom -> generic must preserve
// operand segmentation, types, indices and folded attributes exactly.
module {
  func.func @indexed_store_forms(%value: !neura.data<i32, i1>, %base: !neura.data<memref<512x256xi32>, i1>, %i: !neura.data<index, i1>, %j: !neura.data<index, i1>) {
    "neura.store_indexed"(%value, %base, %i, %j) <{operandSegmentSizes = array<i32: 1, 1, 2>}> : (!neura.data<i32, i1>, !neura.data<memref<512x256xi32>, i1>, !neura.data<index, i1>, !neura.data<index, i1>) -> ()
    "neura.store_indexed"(%base, %i, %j) <{operandSegmentSizes = array<i32: 0, 1, 2>}> {lhs_value = "%input0"} : (!neura.data<memref<512x256xi32>, i1>, !neura.data<index, i1>, !neura.data<index, i1>) -> ()
    "neura.store_indexed"(%value, %i, %j) <{operandSegmentSizes = array<i32: 1, 0, 2>}> {rhs_value = "%input1"} : (!neura.data<i32, i1>, !neura.data<index, i1>, !neura.data<index, i1>) -> ()
    "neura.store_indexed"(%i, %j) <{operandSegmentSizes = array<i32: 0, 0, 2>}> {lhs_value = 7 : i32, rhs_value = "%input1"} : (!neura.data<index, i1>, !neura.data<index, i1>) -> ()
    return
  }
}
