#include "Taskflow/ProgramMatrixImplementation.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include <set>

using namespace mlir;
using namespace orbit::taskflow;
namespace {
bool fail(std::string &error, const char *message) {
  error = std::string("matrix implementation: ") + message;
  return false;
}
bool zero(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  auto literal =
      constant ? dyn_cast<FloatAttr>(constant.getValue()) : FloatAttr{};
  return literal && literal.getType().isF32() && literal.getValue().isZero() &&
         !literal.getValue().isNegative();
}
bool indexed(affine::AffineLoadOp op, Value memref, Value i, Value j) {
  return op && op.getMemRef() == memref && op.getAffineMap().isIdentity() &&
         op.getIndices().size() == 2 && op.getIndices()[0] == i &&
         op.getIndices()[1] == j;
}
bool indexed(affine::AffineStoreOp op, Value memref, Value i, Value j) {
  return op && op.getMemRef() == memref && op.getAffineMap().isIdentity() &&
         op.getIndices().size() == 2 && op.getIndices()[0] == i &&
         op.getIndices()[1] == j;
}
} // namespace

bool orbit::taskflow::analyzeProgramMatrixTask(
    const GenericAnalysis &source, size_t taskIndex,
    ProgramMatrixTaskContract &result, std::string &error) {
  result = {};
  if (!source.function || taskIndex >= source.tasks.size()) {
    return fail(error, "source task is missing");
  }
  auto contract =
      source.function->getAttrOfType<StringAttr>("orbit.matrix_contract");
  if (!contract || contract.getValue() != programMatrixNumericalContract) {
    return fail(error, "explicit BF16 numerical contract is required");
  }
  const auto &task = source.tasks[taskIndex];
  if (task.loops.size() != 3 || task.argumentStorages.size() != 3 ||
      task.operation->getNumRegions() != 1 ||
      !llvm::hasSingleElement(task.operation->getRegion(0))) {
    return fail(error,
                "expected one three-loop matrix task with three buffers");
  }
  auto &body = task.operation->getRegion(0).front();
  if (body.getNumArguments() != 3) {
    return fail(error, "scalar or extra task arguments are unsupported");
  }
  SmallVector<affine::AffineForOp> loops;
  for (const auto &loop : task.loops) {
    auto affine = dyn_cast<affine::AffineForOp>(loop.operation);
    if (!affine || !affine.hasConstantBounds() ||
        affine.getConstantLowerBound() != 0 || affine.getStepAsInt() != 1 ||
        affine.getNumIterOperands() != 0) {
      return fail(error, "matrix source requires static zero-based unit loops");
    }
    loops.push_back(affine);
  }
  if (loops[1]->getParentOp() != loops[0] ||
      loops[2]->getParentOp() != loops[1]) {
    return fail(error, "matrix source must nest M, N, K in that order");
  }
  SmallVector<affine::AffineLoadOp> loads;
  SmallVector<affine::AffineStoreOp> stores;
  SmallVector<arith::ExtFOp> extensions;
  arith::MulFOp multiply;
  arith::AddFOp add;
  bool valid = true;
  task.operation->walk([&](Operation *op) {
    if (op == task.operation ||
        isa<affine::AffineForOp, affine::AffineYieldOp>(op) ||
        op->getName().getStringRef() == "taskflow.yield") {
      return;
    }
    if (auto load = dyn_cast<affine::AffineLoadOp>(op)) {
      loads.push_back(load);
    } else if (auto store = dyn_cast<affine::AffineStoreOp>(op)) {
      stores.push_back(store);
    } else if (auto ext = dyn_cast<arith::ExtFOp>(op)) {
      extensions.push_back(ext);
    } else if (auto mul = dyn_cast<arith::MulFOp>(op)) {
      valid &= !multiply;
      multiply = mul;
    } else if (auto sum = dyn_cast<arith::AddFOp>(op)) {
      valid &= !add;
      add = sum;
    } else if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
      valid &= constant.getType().isF32() && zero(constant.getResult());
    } else {
      valid = false;
    }
  });
  if (!valid || loads.size() != 3 || extensions.size() != 2 || !multiply ||
      !add || stores.empty() || stores.size() > 2) {
    return fail(error,
                "source arithmetic is not exactly BF16 GEMM accumulation");
  }
  auto extA = multiply.getLhs().getDefiningOp<arith::ExtFOp>();
  auto extB = multiply.getRhs().getDefiningOp<arith::ExtFOp>();
  if (!extA || !extB || extA == extB || !extA.getIn().getType().isBF16() ||
      !extB.getIn().getType().isBF16() || !extA.getType().isF32() ||
      !extB.getType().isF32()) {
    return fail(error, "matrix product must extend both BF16 operands to FP32");
  }
  auto a = extA.getIn().getDefiningOp<affine::AffineLoadOp>();
  auto b = extB.getIn().getDefiningOp<affine::AffineLoadOp>();
  Value old = add.getLhs() == multiply.getResult()   ? add.getRhs()
              : add.getRhs() == multiply.getResult() ? add.getLhs()
                                                     : Value{};
  auto c =
      old ? old.getDefiningOp<affine::AffineLoadOp>() : affine::AffineLoadOp{};
  auto i = loops[0].getInductionVar();
  auto j = loops[1].getInductionVar();
  auto k = loops[2].getInductionVar();
  if (!a || !b || !c) {
    return fail(error, "matrix operands are not source affine loads");
  }
  auto argument = [&](Value value) -> int64_t {
    auto arg = dyn_cast<BlockArgument>(value);
    return arg && arg.getOwner() == &body ? int64_t(arg.getArgNumber()) : -1;
  };
  int64_t argA = argument(a.getMemRef());
  int64_t argB = argument(b.getMemRef());
  int64_t argC = argument(c.getMemRef());
  if (argA < 0 || argB < 0 || argC < 0 || argA == argB || argA == argC ||
      argB == argC || !indexed(a, a.getMemRef(), i, k) ||
      !indexed(b, b.getMemRef(), k, j) || !indexed(c, c.getMemRef(), i, j) ||
      !c.getType().isF32()) {
    return fail(error, "matrix source indexing or buffer identity differs");
  }
  size_t accumulates = 0, initializes = 0;
  for (auto store : stores) {
    if (!indexed(store, c.getMemRef(), i, j)) {
      return fail(error, "matrix source writes another region");
    }
    if (store.getValueToStore() == add.getResult() &&
        store->getParentOp() == loops[2]) {
      ++accumulates;
    } else if (zero(store.getValueToStore()) &&
               store->getParentOp() == loops[1] &&
               store->isBeforeInBlock(loops[2])) {
      ++initializes;
    } else {
      return fail(error,
                  "matrix source store is not zero-init or accumulation");
    }
  }
  if (accumulates != 1 || initializes > 1 || a->getParentOp() != loops[2] ||
      b->getParentOp() != loops[2] || c->getParentOp() != loops[2] ||
      multiply->getParentOp() != loops[2] || add->getParentOp() != loops[2]) {
    return fail(error, "matrix source arithmetic scope differs");
  }
  result.task = taskIndex;
  result.a = task.argumentStorages[argA];
  result.b = task.argumentStorages[argB];
  result.c = task.argumentStorages[argC];
  result.m = loops[0].getConstantUpperBound();
  result.n = loops[1].getConstantUpperBound();
  result.k = loops[2].getConstantUpperBound();
  result.initializes = initializes;
  for (auto entry :
       {std::make_pair(result.a, std::vector<int64_t>{result.m, result.k}),
        std::make_pair(result.b, std::vector<int64_t>{result.k, result.n}),
        std::make_pair(result.c, std::vector<int64_t>{result.m, result.n})}) {
    if (entry.first >= source.storages.size() ||
        source.storages[entry.first].shape != entry.second) {
      return fail(error, "matrix storage shape differs from the loop domain");
    }
  }
  return true;
}

bool orbit::taskflow::analyzeProgramMatrixInstance(
    const ProgramGraph &graph, const MappingTaskInstance &instance,
    ProgramMatrixInstanceContract &result, std::string &error) {
  result = {};
  if (instance.implementation != programMatrixImplementation ||
      instance.kind != "source-task" || instance.domainTask != instance.task ||
      instance.lanes != 1 || instance.resourceCount != 1 ||
      instance.domain.size() != 3 ||
      !analyzeProgramMatrixTask(graph.source, instance.task, result.task,
                                error)) {
    if (error.empty()) {
      fail(error, "invalid explicitly selected matrix instance");
    }
    return false;
  }
  int64_t lower[3] = {}, upper[3] = {};
  std::set<size_t> seen;
  int64_t extents[] = {result.task.m, result.task.n, result.task.k};
  for (const auto &domain : instance.domain) {
    if (domain.loop >= 3 || !seen.insert(domain.loop).second ||
        domain.step != 1 || domain.lower < 0 || domain.lower >= domain.upper ||
        domain.upper > extents[domain.loop]) {
      return fail(error, "matrix instance domain is not a source subdomain");
    }
    lower[domain.loop] = domain.lower;
    upper[domain.loop] = domain.upper;
  }
  result.lowerM = lower[0];
  result.lowerN = lower[1];
  result.lowerK = lower[2];
  result.m = upper[0] - lower[0];
  result.n = upper[1] - lower[1];
  result.k = upper[2] - lower[2];
  if (result.m % 8 || result.n % 16 || result.k % 8 || result.m > 128 ||
      result.n > 128 || result.k > 128) {
    return fail(error,
                "AIE2P BF16 matrix tile must be bounded and aligned 8x8x16");
  }
  if (result.task.initializes &&
      (result.lowerK != 0 || result.k != result.task.k)) {
    return fail(error,
                "partial K requires a separate proved source initializer");
  }
  return true;
}

std::string orbit::taskflow::programMatrixKernelSymbol(int64_t m, int64_t k,
                                                       int64_t n) {
  return "orbit_bf16_mmul_" + std::to_string(m) + "x" + std::to_string(k) +
         "x" + std::to_string(n);
}
std::string orbit::taskflow::programMatrixKernelSource(int64_t m, int64_t k,
                                                       int64_t n) {
  return "#define bf16_f32_ONLY\n#define matmul_bf16_f32 " +
         programMatrixKernelSymbol(m, k, n) + "_vendor_entry\n#define DIM_M " +
         std::to_string(m) + "\n#define DIM_K " + std::to_string(k) +
         "\n#define DIM_N " + std::to_string(n) +
         "\n#include <aie_kernels/linalg/mm_aie2p.h>\nextern \"C\" void " +
         programMatrixKernelSymbol(m, k, n) +
         "(const bfloat16 *a, const bfloat16 *b, float *c) {\n"
         "  matmul_vectorized_4x8x8_bf16_f32<" +
         std::to_string(m) + "," + std::to_string(k) + "," + std::to_string(n) +
         ">(a,b,c);\n}\n";
}
