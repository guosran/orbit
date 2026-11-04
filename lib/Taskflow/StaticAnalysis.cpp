#include "Taskflow/StaticMapping.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
using namespace orbit::taskflow;
using namespace mlir;
namespace {
bool fail(std::string &e, const std::string &s) {
  e = s;
  return false;
}
bool constant(Value v, double expected) {
  auto op = v.getDefiningOp<arith::ConstantOp>();
  if (!op)
    return false;
  if (auto f = dyn_cast<FloatAttr>(op.getValue()))
    return f.getValueAsDouble() == expected;
  if (auto i = dyn_cast<IntegerAttr>(op.getValue()))
    return (i.getType().isInteger(1) ? double(i.getValue().getZExtValue())
                                     : double(i.getInt())) == expected;
  return false;
}
bool loop(affine::AffineForOp op, orbit::taskflow::Range &r) {
  if (!op || !op.hasConstantLowerBound() || !op.hasConstantUpperBound() ||
      op.getStep() != 1 || op.getNumIterOperands() != 0)
    return false;
  r = {op.getConstantLowerBound(), op.getConstantUpperBound()};
  return r.lo >= 0 && r.hi > r.lo;
}
bool indices(affine::AffineLoadOp op, Value x, Value y) {
  return op && op.getAffineMap().isIdentity() &&
         op.getMapOperands().size() == 2 && op.getMapOperands()[0] == x &&
         op.getMapOperands()[1] == y;
}
bool indices(affine::AffineStoreOp op, Value x, Value y) {
  return op && op.getAffineMap().isIdentity() &&
         op.getMapOperands().size() == 2 && op.getMapOperands()[0] == x &&
         op.getMapOperands()[1] == y;
}
std::vector<Operation *> content(Block *b) {
  std::vector<Operation *> r;
  for (auto &op : *b)
    if (!isa<affine::AffineYieldOp>(op))
      r.push_back(&op);
  return r;
}
bool infer(affine::AffineForOp outer,
           const llvm::DenseMap<Value, std::string> &storage, Compute &c,
           std::string &e) {
  if (!loop(outer, c.m))
    return fail(e, "static unit-step affine M loop required");
  auto xs = content(outer.getBody());
  if (xs.size() != 1)
    return fail(e, "M body must contain one N loop");
  auto inner = dyn_cast<affine::AffineForOp>(xs[0]);
  if (!loop(inner, c.n))
    return fail(e, "static unit-step affine N loop required");
  Value m = outer.getInductionVar(), n = inner.getInductionVar();
  auto ops = content(inner.getBody());
  Block *body = inner.getBody();
  Value k;
  if (ops.size() == 1 && isa<affine::AffineForOp>(ops[0])) {
    auto reduction = cast<affine::AffineForOp>(ops[0]);
    if (!loop(reduction, c.k))
      return fail(e, "static ordered K loop required");
    k = reduction.getInductionVar();
    body = reduction.getBody();
    ops = content(body);
  }
  if (ops.empty())
    return fail(e, "empty compute loop");
  auto store = dyn_cast<affine::AffineStoreOp>(ops.back());
  if (!indices(store, m, n) || !storage.count(store.getMemRef()))
    return fail(e, "unsupported store map or aliased storage");
  c.output = storage.lookup(store.getMemRef());
  Value result = store.getValue();
  auto input = [&](Value value, Value x, Value y, std::string &name) {
    auto load = value.getDefiningOp<affine::AffineLoadOp>();
    if (!indices(load, x, y) || !storage.count(load.getMemRef()))
      return false;
    name = storage.lookup(load.getMemRef());
    return true;
  };
  if (!k && ops.size() == 1 && constant(result, 0)) {
    c.kind = "fill";
    return true;
  }
  if (k) {
    if (ops.size() != 6)
      return fail(e, "matmul must contain exactly three loads, multiply, "
                     "ordered add, store");
    auto add = result.getDefiningOp<arith::AddFOp>();
    if (!add)
      return fail(e, "K reduction missing accumulating add");
    auto mul = add.getRhs().getDefiningOp<arith::MulFOp>();
    std::string old, a, b;
    if (!mul || !input(add.getLhs(), m, n, old) || old != c.output ||
        !input(mul.getLhs(), m, k, a) || !input(mul.getRhs(), k, n, b))
      return fail(e, "unsupported matmul SSA/access pattern");
    c.kind = "matmul";
    c.inputs = {a, b};
    c.accumulate = true;
    return true;
  }
  if (auto maximum = result.getDefiningOp<arith::MaximumFOp>()) {
    std::string a;
    if (ops.size() != 3 || !input(maximum.getLhs(), m, n, a) ||
        !constant(maximum.getRhs(), 0))
      return fail(e, "unsupported ReLU SSA expression");
    c.kind = "relu";
    c.inputs = {a};
    return true;
  }
  if (auto mul = result.getDefiningOp<arith::MulFOp>()) {
    auto div = mul.getLhs().getDefiningOp<arith::DivFOp>();
    if (!div)
      return fail(e, "unsupported activation multiply");
    auto add = div.getRhs().getDefiningOp<arith::AddFOp>();
    if (!add)
      return fail(e, "unsupported SiLU denominator");
    auto exp = add.getRhs().getDefiningOp<math::ExpOp>();
    auto neg =
        exp ? exp.getOperand().getDefiningOp<arith::NegFOp>() : arith::NegFOp();
    std::string gate, up;
    if (ops.size() != 8 || !constant(add.getLhs(), 1) || !neg ||
        neg.getOperand() != div.getLhs() || !input(div.getLhs(), m, n, gate) ||
        !input(mul.getRhs(), m, n, up))
      return fail(e, "unsupported SwiGLU SSA expression");
    c.kind = "swiglu";
    c.inputs = {gate, up};
    return true;
  }
  return fail(e, "unsupported static compute expression");
}
} // namespace
bool orbit::taskflow::analyzeStaticProgram(ModuleOp module, StaticAnalysis &a,
                                           std::string &e) {
  a = StaticAnalysis{};
  if (failed(verify(module)))
    return fail(e, "invalid Taskflow MLIR");
  // Witness supplies only interface names/roles/source provenance. Every
  // Compute, loop bound, access and dependency below is independently read from
  // the body.
  auto witness = module->getAttrOfType<StringAttr>("orbit.plan");
  if (!witness)
    return fail(e, "static analysis requires orbit.plan interface metadata "
                   "(storage names/roles); body is the computation authority");
  auto parsed = llvm::json::parse(witness.getValue());
  if (!parsed)
    return fail(e, "invalid interface metadata JSON");
  Program metadata;
  if (!parsePlan(*parsed, metadata, e))
    return false;
  std::vector<func::FuncOp> functions;
  for (auto &op : module.getBody()->getOperations()) {
    auto fn = dyn_cast<func::FuncOp>(op);
    if (!fn)
      return fail(e, "unsupported module operation outside static function");
    functions.push_back(fn);
  }
  if (functions.size() != 1 || functions[0].isExternal() ||
      !llvm::hasSingleElement(functions[0].getBody()))
    return fail(e, "one defined single-block static function required");
  auto fn = functions[0];
  if (fn.getNumArguments() != metadata.buffers.size() ||
      fn.getNumResults() != 0)
    return fail(e, "interface metadata/function arity mismatch");
  a.program.sourceId = metadata.sourceId;
  a.program.candidateId = metadata.candidateId;
  a.program.outputs = metadata.outputs;
  a.program.transforms = metadata.transforms;
  llvm::DenseMap<Value, std::string> global;
  for (auto [index, arg] : llvm::enumerate(fn.getArguments())) {
    auto type = dyn_cast<MemRefType>(arg.getType());
    auto b = metadata.buffers[index];
    if (!type || type.getRank() != 2 || !type.hasStaticShape() ||
        !type.getElementType().isF32() || !type.getLayout().isIdentity() ||
        type.getMemorySpace() || type.getShape()[0] != b.rows ||
        type.getShape()[1] != b.cols)
      return fail(e, "only static nonaliased identity-layout rank-two f32 "
                     "interfaces supported");
    global[arg] = b.id;
    a.program.buffers.push_back(b);
  }
  std::map<Operation *, std::string> completed;
  std::map<std::string, unsigned> counts;
  for (auto &raw : fn.getBody().front()) {
    if (auto ret = dyn_cast<func::ReturnOp>(raw)) {
      if (ret.getNumOperands())
        return fail(e, "unexpected return operands");
      continue;
    }
    auto task = dyn_cast<mlir::taskflow::TaskflowTaskOp>(raw);
    if (!task)
      return fail(e,
                  "only taskflow tasks and return supported in function body");
    Task t;
    t.id = task.getTaskName().str();
    if (counts.count(t.id))
      return fail(e, "duplicate task identity");
    if (task.getWillReads().size() != task.getOriginalReadMemrefs().size() ||
        task.getWillWrites().size() != task.getOriginalWriteMemrefs().size())
      return fail(e, "explicit original storage identity required");
    llvm::DenseMap<Value, std::string> local;
    unsigned index = 0;
    for (auto pair :
         llvm::zip(task.getWillReads(), task.getOriginalReadMemrefs())) {
      Value state = std::get<0>(pair), original = std::get<1>(pair);
      if (state != original || !global.count(original))
        return fail(e,
                    "aliased/versioned storage outside initial static subset");
      local[task.getBody().front().getArgument(index++)] =
          global.lookup(original);
    }
    for (auto pair :
         llvm::zip(task.getWillWrites(), task.getOriginalWriteMemrefs())) {
      Value state = std::get<0>(pair), original = std::get<1>(pair);
      if (state != original || !global.count(original))
        return fail(e,
                    "aliased/versioned storage outside initial static subset");
      local[task.getBody().front().getArgument(index++)] =
          global.lookup(original);
    }
    for (Value dep : task.getValueInputs()) {
      auto producer = dep.getDefiningOp<mlir::taskflow::TaskflowTaskOp>();
      if (!producer || !dep.getType().isInteger(1) ||
          !completed.count(producer.getOperation()) ||
          !llvm::is_contained(producer.getValueOutputs(), dep))
        return fail(e, "unsupported completion dependency");
      t.dependsOn.push_back(completed[producer.getOperation()]);
    }
    for (auto &op : task.getBody().front()) {
      if (isa<arith::ConstantOp>(op))
        continue;
      if (auto yield = dyn_cast<mlir::taskflow::TaskflowYieldOp>(op)) {
        if (yield.getDoneWrites().size() != task.getWillWrites().size() ||
            yield.getValueResults().size() != 1 ||
            !constant(yield.getValueResults()[0], 1))
          return fail(e, "unsupported completion yield");
        for (auto [i, v] : llvm::enumerate(yield.getDoneWrites()))
          if (v != task.getBody().front().getArgument(
                       task.getWillReads().size() + i))
            return fail(e, "yield changes original storage identity");
        continue;
      }
      auto outer = dyn_cast<affine::AffineForOp>(op);
      if (!outer)
        return fail(e, "unknown operation in task body");
      Compute c;
      if (!infer(outer, local, c, e))
        return false;
      // A fixed supported expression rejects extra dead operations and
      // fast-math changes; no generic affine arithmetic is inferred by reading
      // its witness.
      bool bad = false;
      outer.walk([&](Operation *n) {
        if (isa<affine::AffineForOp, affine::AffineYieldOp,
                affine::AffineLoadOp, affine::AffineStoreOp, arith::MulFOp,
                arith::AddFOp, arith::MaximumFOp, arith::DivFOp, arith::NegFOp,
                math::ExpOp>(n)) {
        } else
          bad = true;
        if (auto fm = n->getAttrOfType<arith::FastMathFlagsAttr>("fastmath"))
          if (fm.getValue() != arith::FastMathFlags::none)
            bad = true;
      });
      if (bad)
        return fail(e, "unsupported/fast-math computation in static body");
      StaticStep s;
      s.taskId = t.id;
      s.operationIndex = t.operations.size();
      s.id = t.id + ":" + std::to_string(s.operationIndex);
      s.operation = c;
      s.body = outer.getOperation();
      if (s.operationIndex)
        s.dependsOn = {t.id + ":" + std::to_string(s.operationIndex - 1)};
      else
        for (auto &d : t.dependsOn)
          s.dependsOn.push_back(d + ":" + std::to_string(counts[d] - 1));
      s.reads = staticReads(c);
      s.writes = staticWrites(c);
      a.steps.push_back(s);
      t.operations.push_back(c);
    }
    if (t.operations.empty())
      return fail(e, "empty task body unsupported");
    counts[t.id] = t.operations.size();
    completed[task.getOperation()] = t.id;
    a.program.tasks.push_back(t);
  }
  if (!validate(a.program, e))
    return false;
  if (staticFingerprint(printPlan(a.program)) !=
      staticFingerprint(printPlan(metadata)))
    return fail(e, "actual Taskflow computation/accesses/dependencies disagree "
                   "with metadata witness");
  std::string body;
  llvm::raw_string_ostream os(body);
  fn.print(os);
  a.bodyFingerprint = staticFingerprint(llvm::json::Value(body));
  return true;
}
