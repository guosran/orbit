#include "Taskflow/ProgramMatrixImplementation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"
#include <array>
#include <limits>

using namespace mlir;
using namespace orbit::taskflow;
namespace {
bool fail(std::string &error, const char *message) {
  error = std::string("matrix AIR: ") + message;
  return false;
}
Value constant(OpBuilder &b, int64_t value) {
  return b.create<arith::ConstantIndexOp>(b.getUnknownLoc(), value);
}
Value add(OpBuilder &b, Value a, Value c) {
  return b.create<arith::AddIOp>(b.getUnknownLoc(), a, c);
}
Value mul(OpBuilder &b, Value a, int64_t c) {
  return b.create<arith::MulIOp>(b.getUnknownLoc(), a, constant(b, c));
}
Value div(OpBuilder &b, Value a, int64_t c) {
  return b.create<arith::DivUIOp>(b.getUnknownLoc(), a, constant(b, c));
}
Value rem(OpBuilder &b, Value a, int64_t c) {
  return b.create<arith::RemUIOp>(b.getUnknownLoc(), a, constant(b, c));
}
// Row-major microtiles: (row/R, col/S, row%R, col%S). This is the
// exact layout consumed by mm_aie2p.h, not the source row-major layout.
Value packedIndex(OpBuilder &b, Value row, Value col, int64_t columns,
                  int64_t r, int64_t s) {
  Value tile = add(b, mul(b, div(b, row, r), columns / s), div(b, col, s));
  return add(b, mul(b, add(b, mul(b, tile, r), rem(b, row, r)), s),
             rem(b, col, s));
}
void nest(OpBuilder &b, int64_t rows, int64_t columns,
          llvm::function_ref<void(Value, Value)> body, size_t event) {
  auto outer = b.create<scf::ForOp>(b.getUnknownLoc(), constant(b, 0),
                                    constant(b, rows), constant(b, 1));
  outer->setAttr("orbit.matrix_local_layout_event", b.getI64IntegerAttr(event));
  b.setInsertionPointToStart(outer.getBody());
  auto inner = b.create<scf::ForOp>(b.getUnknownLoc(), constant(b, 0),
                                    constant(b, columns), constant(b, 1));
  b.setInsertionPointToStart(inner.getBody());
  body(outer.getInductionVar(), inner.getInductionVar());
  b.setInsertionPointAfter(outer);
}
const StaticResource *resource(const ProgramGraph &graph, StringRef id) {
  auto found =
      llvm::find_if(graph.resources, [&](const auto &r) { return r.id == id; });
  return found == graph.resources.end() ? nullptr : &*found;
}
const StaticMemory *memory(const ProgramGraph &graph, StringRef id) {
  auto found =
      llvm::find_if(graph.memories, [&](const auto &m) { return m.id == id; });
  return found == graph.memories.end() ? nullptr : &*found;
}
bool selectedRectangle(ArrayRef<MappingRegion> regions, size_t storage,
                       int64_t row, int64_t col, int64_t rows, int64_t cols) {
  auto found = llvm::find_if(
      regions, [&](const auto &r) { return r.storage == storage; });
  return found != regions.end() &&
         found->offsets == std::vector<int64_t>{row, col} &&
         found->sizes == std::vector<int64_t>{rows, cols};
}
} // namespace

bool orbit::taskflow::emitProgramMatrixInstance(
    OpBuilder &b, const ProgramGraph &graph,
    const MappingTaskInstance &instance, ArrayRef<MappingRegion> regions,
    ArrayRef<Value> locals, bool &handled, std::string &error) {
  handled = instance.implementation == programMatrixImplementation;
  if (!handled) {
    return true;
  }
  ProgramMatrixInstanceContract spec;
  if (!analyzeProgramMatrixInstance(graph, instance, spec, error)) {
    return false;
  }
  if (locals.size() != graph.source.storages.size() ||
      !selectedRectangle(regions, spec.task.a, spec.lowerM, spec.lowerK, spec.m,
                         spec.k) ||
      !selectedRectangle(regions, spec.task.b, spec.lowerK, spec.lowerN, spec.k,
                         spec.n) ||
      !selectedRectangle(regions, spec.task.c, spec.lowerM, spec.lowerN, spec.m,
                         spec.n)) {
    return fail(error, "selected copies do not equal the matrix instance tile");
  }
  for (size_t storage : {spec.task.a, spec.task.b, spec.task.c}) {
    if (!locals[storage]) {
      return fail(error, "matrix instance has no selected local buffer");
    }
  }
  Operation *herd = b.getBlock()->getParentOp();
  while (herd && herd->getName().getStringRef() != "air.herd")
    herd = herd->getParentOp();
  if (!herd) {
    return fail(error, "matrix emitter requires a selected physical herd");
  }
  auto *owner = resource(graph, graph.events[instance.event].resource);
  if (!owner || owner->kind != "compute") {
    return fail(error, "matrix compute resource is missing");
  }
  // All calls on a selected core share these three scratch pools. They are
  // physical layout refinements, not extra logical versions or host copies.
  std::array<int64_t, 3> maximum{};
  for (const auto &candidate : graph.instances) {
    if (candidate.implementation != programMatrixImplementation) {
      continue;
    }
    auto *other = resource(graph, graph.events[candidate.event].resource);
    if (!other || other->domain != owner->domain) {
      continue;
    }
    ProgramMatrixInstanceContract shape;
    if (!analyzeProgramMatrixInstance(graph, candidate, shape, error)) {
      return false;
    }
    maximum[0] = std::max(maximum[0], shape.m * shape.k);
    maximum[1] = std::max(maximum[1], shape.k * shape.n);
    maximum[2] = std::max(maximum[2], shape.m * shape.n);
  }
  int64_t selectedBytes = 0;
  const StaticMemory *localMemory = nullptr;
  for (const auto &allocation : graph.allocations) {
    auto *mem = memory(graph, allocation.memory);
    if (mem && mem->kind == "local-sram" && mem->domain == owner->domain) {
      if (allocation.bytes > INT64_MAX - selectedBytes) {
        return fail(error, "selected local byte count overflows");
      }
      selectedBytes += allocation.bytes;
      localMemory = mem;
    }
  }
  const int64_t scratchBytes = 2 * maximum[0] + 2 * maximum[1] + 4 * maximum[2];
  if (!localMemory || !localMemory->capacityKnown ||
      selectedBytes > localMemory->capacityBytes ||
      scratchBytes > localMemory->capacityBytes - selectedBytes) {
    return fail(error,
                "selected allocations plus layout scratch exceed known SRAM");
  }
  herd->setAttr("orbit.matrix_layout_scratch_bytes",
                b.getI64IntegerAttr(scratchBytes));
  std::array<Value, 3> scratch;
  for (auto alloc : herd->getRegion(0).front().getOps<memref::AllocOp>()) {
    if (auto role =
            alloc->getAttrOfType<IntegerAttr>("orbit.matrix_scratch_role")) {
      if (role.getInt() < 0 || role.getInt() >= 3 || scratch[role.getInt()]) {
        return fail(error, "duplicate matrix physical scratch role");
      }
      scratch[role.getInt()] = alloc;
    }
  }
  std::array<Type, 3> elements{b.getBF16Type(), b.getBF16Type(),
                               b.getF32Type()};
  for (size_t role = 0; role < 3; ++role) {
    auto type =
        MemRefType::get({maximum[role]}, elements[role],
                        MemRefLayoutAttrInterface{}, b.getI32IntegerAttr(2));
    if (!scratch[role]) {
      OpBuilder::InsertionGuard insertion(b);
      b.setInsertionPointToStart(&herd->getRegion(0).front());
      auto alloc = b.create<memref::AllocOp>(b.getUnknownLoc(), type);
      alloc->setAttr("orbit.matrix_scratch_role", b.getI64IntegerAttr(role));
      alloc->setAttr("alignment", b.getI64IntegerAttr(64));
      scratch[role] = alloc;
    } else if (scratch[role].getType() != type) {
      return fail(error, "matrix scratch geometry is inconsistent");
    }
  }
  auto pack = [&](size_t role, size_t storage, int64_t rows, int64_t cols,
                  int64_t r, int64_t s, bool zero) {
    nest(
        b, rows, cols,
        [&](Value row, Value col) {
          Value value =
              zero ? Value(b.create<arith::ConstantFloatOp>(
                         b.getUnknownLoc(), APFloat(0.0f), b.getF32Type()))
                   : Value(b.create<memref::LoadOp>(b.getUnknownLoc(),
                                                    locals[storage],
                                                    ValueRange{row, col}));
          Value index = packedIndex(b, row, col, cols, r, s);
          b.create<memref::StoreOp>(b.getUnknownLoc(), value, scratch[role],
                                    index);
        },
        instance.event);
  };
  pack(0, spec.task.a, spec.m, spec.k, 4, 8, false);
  pack(1, spec.task.b, spec.k, spec.n, 8, 8, false);
  pack(2, spec.task.c, spec.m, spec.n, 4, 8, spec.task.initializes);
  std::array<int64_t, 3> counts{spec.m * spec.k, spec.k * spec.n,
                                spec.m * spec.n};
  SmallVector<Value> arguments;
  SmallVector<Type> types;
  for (size_t role = 0; role < 3; ++role) {
    auto type =
        MemRefType::get({counts[role]}, elements[role],
                        MemRefLayoutAttrInterface{}, b.getI32IntegerAttr(2));
    Value value = scratch[role];
    if (counts[role] != maximum[role]) {
      value = b.create<memref::ReinterpretCastOp>(
          b.getUnknownLoc(), type, value, 0, ArrayRef<int64_t>{counts[role]},
          ArrayRef<int64_t>{1});
    }
    arguments.push_back(value);
    types.push_back(type);
  }
  auto module = herd->getParentOfType<ModuleOp>();
  auto symbol = programMatrixKernelSymbol(spec.m, spec.k, spec.n);
  if (!module.lookupSymbol<func::FuncOp>(symbol)) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(module.getBody());
    auto function =
        b.create<func::FuncOp>(b.getUnknownLoc(), symbol,
                               FunctionType::get(b.getContext(), types, {}));
    function.setPrivate();
    function->setAttr("link_with", b.getStringAttr(symbol + ".o"));
    function->setAttr("orbit.matrix_shape",
                      b.getDenseI64ArrayAttr({spec.m, spec.k, spec.n}));
  }
  auto call =
      b.create<func::CallOp>(b.getUnknownLoc(), symbol, TypeRange{}, arguments);
  call->setAttr("orbit.matrix_event", b.getI64IntegerAttr(instance.event));
  call->setAttr("orbit.matrix_contract",
                b.getStringAttr(programMatrixNumericalContract));
  call->setAttr(
      "orbit.matrix_local_layout_payload_bytes",
      b.getI64IntegerAttr(2 * counts[0] + 2 * counts[1] + 8 * counts[2]));
  nest(
      b, spec.m, spec.n,
      [&](Value row, Value col) {
        Value index = packedIndex(b, row, col, spec.n, 4, 8);
        auto value =
            b.create<memref::LoadOp>(b.getUnknownLoc(), scratch[2], index);
        b.create<memref::StoreOp>(b.getUnknownLoc(), value, locals[spec.task.c],
                                  ValueRange{row, col});
      },
      instance.event);
  return true;
}

llvm::json::Value orbit::taskflow::programMatrixAIRManifest(ModuleOp module) {
  llvm::json::Array kernels;
  int64_t scratchBytes = 0, layoutBytes = 0, calls = 0;
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "air.herd") {
      if (auto value = op->getAttrOfType<IntegerAttr>(
              "orbit.matrix_layout_scratch_bytes")) {
        scratchBytes += value.getInt();
      }
    }
    if (auto call = dyn_cast<func::CallOp>(op)) {
      if (auto value = call->getAttrOfType<IntegerAttr>(
              "orbit.matrix_local_layout_payload_bytes")) {
        int64_t repetitions = 1;
        for (Operation *parent = op->getParentOp(); parent;
             parent = parent->getParentOp())
          if (auto count = parent->getAttrOfType<IntegerAttr>(
                  "orbit.selected_loop_repetitions"))
            repetitions *= count.getInt();
        layoutBytes += value.getInt() * repetitions;
        calls += repetitions;
      }
    }
    auto function = dyn_cast<func::FuncOp>(op);
    if (!function) {
      return;
    }
    auto shape =
        function->getAttrOfType<DenseI64ArrayAttr>("orbit.matrix_shape");
    if (!shape || shape.size() != 3) {
      return;
    }
    auto dims = shape.asArrayRef();
    kernels.push_back(llvm::json::Object{
        {"symbol", function.getName().str()},
        {"object",
         function->getAttrOfType<StringAttr>("link_with").getValue().str()},
        {"m", dims[0]},
        {"k", dims[1]},
        {"n", dims[2]},
        {"target_arch", "aie2p"},
        {"generated_source_fingerprint",
         staticFingerprint(llvm::json::Value(
             programMatrixKernelSource(dims[0], dims[1], dims[2])))}});
  });
  return llvm::json::Object{
      {"schema", "orbit.matrix-air-realization.v32"},
      {"numerical_contract", programMatrixNumericalContract},
      {"kernels", std::move(kernels)},
      {"calls", calls},
      {"physical_scratch_bytes", scratchBytes},
      {"local_layout_payload_bytes", layoutBytes},
      {"local_layout_payload_scope",
       "A/B pack, C pack and unpack payload; not memory bus traffic"},
      {"additional_host_transfer_bytes", 0},
      {"analytical_layout_work_charged", false},
      {"final_aie_kernel_numeric_certificate", false},
      {"scope", "explicit selected source instances and local copies; reusable "
                "per-core matrix-layout scratch; external kernel objects "
                "require hash binding and numeric validation"}};
}
