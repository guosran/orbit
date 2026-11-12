#include "Taskflow/GenericAnalysis.h"
#include "Taskflow/StaticMapping.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
using namespace mlir;
using namespace orbit::taskflow;
namespace {
using TFTask = mlir::taskflow::TaskflowTaskOp;
using Interval = GenericInterval;
bool fail(std::string &e, const std::string &s) {
  e = s;
  return false;
}
bool constant(Value v, int64_t &n) {
  llvm::APInt x;
  if (!matchPattern(v, m_ConstantInt(&x)))
    return false;
  n = x.getSExtValue();
  return true;
}
bool covered(const std::vector<Interval> &box,
             const std::vector<std::vector<Interval>> &writes) {
  std::vector<std::vector<Interval>> holes = {box};
  for (const auto &w : writes) {
    std::vector<std::vector<Interval>> next;
    for (auto h : holes) {
      bool disjoint = false;
      for (size_t i = 0; i < h.size(); ++i)
        if (h[i].upper <= w[i].lower || w[i].upper <= h[i].lower)
          disjoint = true;
      if (disjoint) {
        next.push_back(h);
        continue;
      }
      for (size_t i = 0; i < h.size(); ++i) {
        if (h[i].lower < w[i].lower) {
          auto fragment = h;
          fragment[i].upper = w[i].lower;
          next.push_back(fragment);
          h[i].lower = w[i].lower;
        }
        if (h[i].upper > w[i].upper) {
          auto fragment = h;
          fragment[i].lower = w[i].upper;
          next.push_back(fragment);
          h[i].upper = w[i].upper;
        }
      }
    }
    holes = std::move(next);
    if (holes.empty())
      return true;
    if (holes.size() > 10000)
      return false;
  }
  return holes.empty();
}
bool overlap(const GenericAccess &a, const GenericAccess &b) {
  if (a.storage != b.storage)
    return false;
  for (size_t i = 0; i < a.region.size(); ++i)
    if (a.region[i].upper <= b.region[i].lower ||
        b.region[i].upper <= a.region[i].lower)
      return false;
  return true;
}
bool expr(AffineExpr e, ArrayRef<Interval> dims, ArrayRef<Interval> symbols,
          Interval &r) {
  if (auto c = dyn_cast<AffineConstantExpr>(e)) {
    if (c.getValue() == INT64_MAX)
      return false;
    r = {c.getValue(), c.getValue() + 1};
    return true;
  }
  if (auto d = dyn_cast<AffineDimExpr>(e)) {
    if (d.getPosition() >= dims.size())
      return false;
    r = dims[d.getPosition()];
    return true;
  }
  if (auto s = dyn_cast<AffineSymbolExpr>(e)) {
    if (s.getPosition() >= symbols.size())
      return false;
    r = symbols[s.getPosition()];
    return true;
  }
  auto b = dyn_cast<AffineBinaryOpExpr>(e);
  if (!b)
    return false;
  Interval x, y;
  if (!expr(b.getLHS(), dims, symbols, x) ||
      !expr(b.getRHS(), dims, symbols, y))
    return false;
  if (e.getKind() == AffineExprKind::Add) {
    __int128 lo = __int128(x.lower) + y.lower,
             hi = __int128(x.upper) + y.upper - 1;
    if (lo < INT64_MIN || hi > INT64_MAX)
      return false;
    r = {int64_t(lo), int64_t(hi)};
    return true;
  }
  if (e.getKind() == AffineExprKind::Mul && y.upper == y.lower + 1) {
    __int128 a = __int128(x.lower) * y.lower,
             b = __int128(x.upper - 1) * y.lower;
    if (std::min(a, b) < INT64_MIN || std::max(a, b) >= INT64_MAX)
      return false;
    r = {int64_t(std::min(a, b)), int64_t(std::max(a, b) + 1)};
    return true;
  }
  return false; // reject mod/div rather than claim an exact region
}
bool valueRange(Value v, const llvm::DenseMap<Value, Interval> &ivs,
                Interval &r) {
  auto i = ivs.find(v);
  if (i != ivs.end()) {
    r = i->second;
    return true;
  }
  int64_t c;
  if (constant(v, c)) {
    if (c == INT64_MAX)
      return false;
    r = {c, c + 1};
    return true;
  }
  if (auto a = v.getDefiningOp<affine::AffineApplyOp>()) {
    SmallVector<Interval> operands;
    for (Value x : a.getMapOperands()) {
      Interval t;
      if (!valueRange(x, ivs, t))
        return false;
      operands.push_back(t);
    }
    auto map = a.getAffineMap();
    return expr(map.getResult(0),
                ArrayRef<Interval>(operands).take_front(map.getNumDims()),
                ArrayRef<Interval>(operands).drop_front(map.getNumDims()), r);
  }
  return false;
}
} // namespace
bool orbit::taskflow::analyzeGenericProgram(ModuleOp module, GenericAnalysis &a,
                                            std::string &e) {
  a = {};
  if (failed(verify(module)))
    return fail(e, "invalid Taskflow MLIR");
  for (auto f : module.getOps<func::FuncOp>()) {
    if (f.isExternal())
      continue;
    if (a.function)
      return fail(e, "generic analysis requires one defined function");
    a.function = f;
  }
  auto fn = a.function;
  if (!fn || !llvm::hasSingleElement(fn.getBody()))
    return fail(e, "generic analysis requires a single-block function");
  if (!fn->hasAttr("orbit.noalias"))
    return fail(e, "native interface requires orbit.noalias argument-storage "
                   "contract; aliases/in-place IO unsupported");
  if (auto b = fn->getAttrOfType<BoolAttr>("orbit.noalias")) {
    if (!b.getValue())
      return fail(e, "aliased interface unsupported");
  } else if (!isa<UnitAttr>(fn->getAttr("orbit.noalias")))
    return fail(e, "orbit.noalias must be unit or true");
  llvm::DenseMap<Value, size_t> storage;
  auto addStorage = [&](Value v, std::string id, std::string role) -> bool {
    auto t = dyn_cast<MemRefType>(v.getType());
    if (!t || !t.hasStaticShape() || !t.getLayout().isIdentity() ||
        t.getMemorySpaceAsInt() != 0)
      return fail(e, "static identity-layout host memref required");
    if (role != "input" && role != "output" && role != "temporary")
      return fail(e,
                  "unsupported interface role (inout/alias rejected): " + role);
    for (auto &s : a.storages)
      if (s.id == id)
        return fail(e, "duplicate storage name");
    GenericStorage s;
    s.id = id;
    s.role = role;
    s.root = v;
    s.shape.assign(t.getShape().begin(), t.getShape().end());
    llvm::raw_string_ostream os(s.dtype);
    t.getElementType().print(os);
    for (auto d : s.shape)
      if (d <= 0 || d > INT32_MAX)
        return fail(e, "invalid/oversized storage extent");
    storage[v] = a.storages.size();
    a.storages.push_back(s);
    return true;
  };
  for (auto [i, arg] : llvm::enumerate(fn.getArguments())) {
    auto role = fn.getArgAttrOfType<StringAttr>(i, "orbit.role");
    auto name = fn.getArgAttrOfType<StringAttr>(i, "orbit.name");
    if (!role)
      return fail(e, "argument missing orbit.role=input/output");
    if (role.getValue() != "input" && role.getValue() != "output")
      return fail(
          e, "native function argument role must be input/output (inout/alias "
             "rejected; temporary reserved for allocations): " +
                 role.str());
    if (!addStorage(arg, name ? name.str() : "arg" + std::to_string(i),
                    role.str()))
      return false;
  }
  std::set<size_t> freed;
  std::vector<std::vector<std::vector<Interval>>> initialized(
      a.storages.size());
  std::vector<std::vector<Operation *>> writers(a.storages.size());
  for (size_t i = 0; i < a.storages.size(); ++i)
    if (a.storages[i].role == "input") {
      std::vector<Interval> box;
      for (auto n : a.storages[i].shape)
        box.push_back({0, n});
      initialized[i].push_back(box);
      writers[i].push_back(nullptr);
    }
  llvm::DenseMap<Operation *, size_t> taskIndex;

  std::function<bool(Value, size_t &)> root = [&](Value v, size_t &s) -> bool {
    auto it = storage.find(v);
    if (it != storage.end()) {
      s = it->second;
      return true;
    }
    if (auto c = v.getDefiningOp<mlir::taskflow::TaskflowChannelOp>())
      return root(c.getSource(), s);
    if (auto j = v.getDefiningOp<mlir::taskflow::TaskflowJoinOp>())
      return root(j.getBase(), s);
    if (auto j =
            v.getDefiningOp<mlir::taskflow::TaskflowReadCompletionJoinOp>())
      return root(j.getOriginalRoot(), s);
    if (auto t = v.getDefiningOp<TFTask>()) {
      auto y = cast<mlir::taskflow::TaskflowYieldOp>(
          t.getBody().front().getTerminator());
      auto result = cast<OpResult>(v).getResultNumber();
      if (result >= y.getNumOperands())
        return false;
      auto ba = dyn_cast<BlockArgument>(y.getOperand(result));
      if (!ba)
        return false;
      unsigned k = ba.getArgNumber();
      if (k < t.getWillReads().size())
        return root(t.getOriginalReadMemrefs()[k], s);
      k -= t.getWillReads().size();
      if (k < t.getWillWrites().size())
        return root(t.getOriginalWriteMemrefs()[k], s);
    }
    return false;
  };
  std::function<void(Value, std::set<size_t> &)> deps = [&](Value v,
                                                            std::set<size_t>
                                                                &d) {
    auto op = v.getDefiningOp();
    if (!op)
      return;
    auto t = taskIndex.find(op);
    if (t != taskIndex.end()) {
      d.insert(t->second);
      return;
    }
    if (isa<mlir::taskflow::TaskflowChannelOp, mlir::taskflow::TaskflowJoinOp,
            mlir::taskflow::TaskflowReadCompletionJoinOp>(op))
      for (auto x : op->getOperands())
        deps(x, d);
  };
  for (Operation &op : fn.front()) {
    if (auto alloc = dyn_cast<memref::AllocOp>(op)) {
      auto n = alloc->getAttrOfType<StringAttr>("orbit.name");
      if (!addStorage(alloc,
                      n ? n.str() : "alloc" + std::to_string(a.storages.size()),
                      "temporary"))
        return false;
      initialized.resize(a.storages.size());
      writers.resize(a.storages.size());
      continue;
    }
    if (isa<func::ReturnOp>(op)) {
      if (op.getNumOperands())
        return fail(e, "return-valued/aliased interface unsupported; use "
                       "output arguments");
      continue;
    }
    if (auto dealloc = dyn_cast<memref::DeallocOp>(op)) {
      size_t s;
      if (!root(dealloc.getMemref(), s) || a.storages[s].role != "temporary" ||
          !freed.insert(s).second)
        return fail(e, "invalid/double free");
      continue;
    }
    if (isa<arith::ConstantOp, mlir::taskflow::TaskflowChannelOp,
            mlir::taskflow::TaskflowJoinOp,
            mlir::taskflow::TaskflowReadCompletionJoinOp>(op))
      continue;
    auto task = dyn_cast<TFTask>(op);
    if (!task)
      return fail(e, "unsupported function operation: " +
                         op.getName().getStringRef().str());
    GenericTask t;
    t.id = task.getTaskName().str();
    t.operation = &op;
    for (auto &old : a.tasks)
      if (old.id == t.id)
        return fail(e, "duplicate task name");
    std::set<size_t> direct;
    for (Value v : task->getOperands())
      deps(v, direct);
    t.dependencies.assign(direct.begin(), direct.end());
    llvm::DenseMap<Value, size_t> local;
    std::set<size_t> declaredReads, declaredWrites;
    auto bind = [&](ValueRange states, ValueRange originals, unsigned offset,
                    std::set<size_t> &decl) -> bool {
      if (states.size() != originals.size())
        return fail(e, "missing original storage identity");
      for (auto [i, v] : llvm::enumerate(originals)) {
        size_t s, version;
        if (!root(v, s) || !root(states[i], version) || s != version)
          return fail(e, "dependency state/original storage mismatch");
        local[task.getBody().front().getArgument(offset + i)] = s;
        t.argumentStorages.push_back(s);
        decl.insert(s);
      }
      return true;
    };
    if (!bind(task.getWillReads(), task.getOriginalReadMemrefs(), 0,
              declaredReads) ||
        !bind(task.getWillWrites(), task.getOriginalWriteMemrefs(),
              task.getWillReads().size(), declaredWrites))
      return false;
    llvm::DenseMap<Value, Interval> ivs;
    bool good = true;
    task.getBody().walk<WalkOrder::PreOrder>([&](Operation *inner)
                                                 -> WalkResult {
      if (!good)
        return WalkResult::interrupt();
      auto bad = [&](std::string why) -> WalkResult {
        good = false;
        e = t.id + ": " + why;
        return WalkResult::interrupt();
      };
      if (auto l = dyn_cast<affine::AffineForOp>(inner)) {
        if (!l.hasConstantLowerBound() || !l.hasConstantUpperBound() ||
            l.getNumIterOperands())
          return bad("only static rectangular affine loops without iter_args "
                     "supported");
        int64_t lo = l.getConstantLowerBound(), hi = l.getConstantUpperBound(),
                step = l.getStep().getSExtValue();
        if (lo < 0 || hi <= lo || hi > INT32_MAX || step <= 0)
          return bad("invalid loop domain");
        ivs[l.getInductionVar()] = {lo, lo + (hi - lo - 1) / step * step + 1};
        t.loops.push_back({inner, l.getInductionVar(), lo, hi, step});
        return WalkResult::advance();
      }
      if (auto l = dyn_cast<scf::ForOp>(inner)) {
        int64_t lo, hi, step;
        if (!constant(l.getLowerBound(), lo) ||
            !constant(l.getUpperBound(), hi) || !constant(l.getStep(), step) ||
            l.getNumRegionIterArgs() || lo < 0 || hi <= lo || hi > INT32_MAX ||
            step <= 0)
          return bad(
              "only static rectangular scf.for without iter_args supported");
        ivs[l.getInductionVar()] = {lo, lo + (hi - lo - 1) / step * step + 1};
        t.loops.push_back({inner, l.getInductionVar(), lo, hi, step});
        return WalkResult::advance();
      }
      GenericAccess access;
      access.operation = inner;
      Value mem;
      SmallVector<Value> indices;
      AffineMap map;
      if (auto l = dyn_cast<affine::AffineLoadOp>(inner)) {
        mem = l.getMemRef();
        indices.append(l.getMapOperands().begin(), l.getMapOperands().end());
        map = l.getAffineMap();
        access.read = true;
      } else if (auto s = dyn_cast<affine::AffineStoreOp>(inner)) {
        mem = s.getMemRef();
        indices.append(s.getMapOperands().begin(), s.getMapOperands().end());
        map = s.getAffineMap();
        access.write = true;
      } else if (auto l = dyn_cast<memref::LoadOp>(inner)) {
        mem = l.getMemRef();
        indices.append(l.getIndices().begin(), l.getIndices().end());
        access.read = true;
      } else if (auto s = dyn_cast<memref::StoreOp>(inner)) {
        mem = s.getMemRef();
        indices.append(s.getIndices().begin(), s.getIndices().end());
        access.write = true;
      }
      if (mem) {
        auto found = local.find(mem);
        if (found == local.end())
          return bad("access through unsupported alias/local storage");
        access.storage = found->second;
        if (freed.count(access.storage))
          return bad("access after deallocation");
        if (access.write && !declaredWrites.count(access.storage))
          return bad("write absent from declared task write interface");
        if (access.read && !declaredReads.count(access.storage) &&
            !declaredWrites.count(access.storage))
          return bad("read absent from task interface");
        if (access.write && a.storages[access.storage].role == "input")
          return bad("input-only storage written");
        SmallVector<Interval> operands;
        for (Value v : indices) {
          Interval r;
          if (!valueRange(v, ivs, r))
            return bad("unbounded/non-affine access index");
          operands.push_back(r);
        }
        if (map) {
          for (auto x : map.getResults()) {
            Interval r;
            if (!expr(x,
                      ArrayRef<Interval>(operands).take_front(map.getNumDims()),
                      ArrayRef<Interval>(operands).drop_front(map.getNumDims()),
                      r))
              return bad("unsupported affine index expression");
            access.region.push_back(r);
          }
        } else
          access.region.assign(operands.begin(), operands.end());
        auto &shape = a.storages[access.storage].shape;
        if (shape.size() != access.region.size())
          return bad("access rank mismatch");
        for (size_t i = 0; i < shape.size(); ++i)
          if (access.region[i].lower < 0 || access.region[i].upper > shape[i])
            return bad("out-of-bounds access");
        // A product of independent unit-stride IVs/constants has exact
        // rectangular coverage. Other affine expressions remain analyzable
        // conservative read regions.
        access.exactRegion = true;
        llvm::DenseSet<Value> used;
        SmallVector<Value> axes;
        if (!map)
          axes = indices;
        else
          for (auto x : map.getResults()) {
            if (auto d = dyn_cast<AffineDimExpr>(x))
              axes.push_back(indices[d.getPosition()]);
            else if (auto c = dyn_cast<AffineConstantExpr>(x))
              axes.push_back(Value());
            else {
              access.exactRegion = false;
              break;
            }
          }
        for (Value x : axes)
          if (x) {
            int64_t c;
            if (constant(x, c))
              continue;
            bool unit = false;
            for (auto &l : t.loops)
              if (l.induction == x && l.step == 1)
                unit = true;
            if (!unit || !used.insert(x).second)
              access.exactRegion = false;
          }
        if (access.read) {
          std::vector<std::vector<Interval>> available;
          for (size_t wi = 0; wi < initialized[access.storage].size(); ++wi) {
            Operation *writer = writers[access.storage][wi];
            bool sharedLoop = false;
            if (writer)
              for (Operation *p = inner->getParentOp();
                   p && p != task.getOperation(); p = p->getParentOp())
                if (isa<affine::AffineForOp, scf::ForOp>(p) &&
                    p->isAncestor(writer))
                  sharedLoop = true;
            bool sameIndex = false;
            if (auto s = dyn_cast_or_null<affine::AffineStoreOp>(writer))
              sameIndex = map && map == s.getAffineMap() &&
                          llvm::equal(indices, s.getMapOperands());
            if (auto s = dyn_cast_or_null<memref::StoreOp>(writer))
              sameIndex = !map && llvm::equal(indices, s.getIndices());
            if (!sharedLoop || sameIndex)
              available.push_back(initialized[access.storage][wi]);
          }
          if (!covered(access.region, available))
            return bad("read of uninitialized output/temporary region");
        }
        if (access.write && access.exactRegion) {
          initialized[access.storage].push_back(access.region);
          writers[access.storage].push_back(inner);
        }
        if (access.read) {
          if (a.storages[access.storage].role == "input")
            access.readVersions.push_back(a.storages[access.storage].id +
                                          ":initial");
          for (const auto &previousTask : a.tasks)
            for (const auto &previous : previousTask.accesses)
              if (previous.write && overlap(previous, access))
                access.readVersions.push_back(previous.writeVersion);
          for (const auto &previous : t.accesses)
            if (previous.write && overlap(previous, access))
              access.readVersions.push_back(previous.writeVersion);
        }
        if (access.write)
          access.writeVersion =
              t.id + ":write:" + std::to_string(t.accesses.size());
        t.accesses.push_back(access);
        return WalkResult::advance();
      }
      if (isa<affine::AffineYieldOp, scf::YieldOp,
              mlir::taskflow::TaskflowYieldOp>(inner))
        return WalkResult::advance();
      if (inner->getNumRegions() == 0 && isMemoryEffectFree(inner) &&
          (inner->getName().getDialectNamespace() == "arith" ||
           inner->getName().getDialectNamespace() == "math" ||
           isa<affine::AffineApplyOp>(inner)))
        return WalkResult::advance();
      return bad("unsupported body operation: " +
                 inner->getName().getStringRef().str());
    });
    if (!good)
      return false;
    if (t.accesses.empty())
      return fail(e, "task has no analyzable memory accesses");
    std::set<size_t> ancestors = direct;
    for (auto d : direct) {
      std::vector<size_t> stack = {d};
      while (!stack.empty()) {
        auto x = stack.back();
        stack.pop_back();
        for (auto y : a.tasks[x].dependencies)
          if (ancestors.insert(y).second)
            stack.push_back(y);
      }
    }
    for (size_t j = 0; j < a.tasks.size(); ++j)
      for (auto &x : a.tasks[j].accesses)
        for (auto &y : t.accesses)
          if ((x.write || y.write) && overlap(x, y) && !ancestors.count(j))
            return fail(e, "missing RAW/WAR/WAW dependency from " +
                               a.tasks[j].id + " to " + t.id);
    taskIndex[&op] = a.tasks.size();
    a.tasks.push_back(std::move(t));
  }
  if (a.tasks.empty())
    return fail(e, "no Taskflow tasks");
  for (size_t i = 0; i < a.storages.size(); ++i)
    if (a.storages[i].role == "output") {
      std::vector<Interval> box;
      for (auto n : a.storages[i].shape)
        box.push_back({0, n});
      if (!covered(box, initialized[i]))
        return fail(e,
                    "output not completely initialized: " + a.storages[i].id);
    }
  std::string text;
  llvm::raw_string_ostream os(text);
  // Keep the historical print path byte-for-byte for ordinary sources. An
  // unattached function clone can print differently because its parent alias/
  // SSA context is missing. Search lineage is nonsemantic; remove only that
  // marker on a complete parent-module clone, never on the caller's live IR.
  bool hasSearchOrigin = false;
  fn.walk([&](Operation *op) {
    hasSearchOrigin |= op->hasAttr("orbit.search_origin");
  });
  if (!hasSearchOrigin) {
    fn.print(os);
  } else {
    OwningOpRef<ModuleOp> fingerprintModule(cast<ModuleOp>(module->clone()));
    fingerprintModule.get().walk(
        [](Operation *op) { op->removeAttr("orbit.search_origin"); });
    auto fingerprintFn =
        fingerprintModule.get().lookupSymbol<func::FuncOp>(fn.getSymName());
    if (!fingerprintFn)
      return fail(e, "source fingerprint clone lost its function");
    fingerprintFn.print(os);
  }
  a.bodyFingerprint = staticFingerprint(llvm::json::Value(text));
  return true;
}
llvm::json::Value
orbit::taskflow::serializeGenericAnalysis(const GenericAnalysis &a) {
  llvm::json::Array storages, tasks;
  for (auto &s : a.storages) {
    llvm::json::Array shape;
    for (auto n : s.shape)
      shape.push_back(n);
    storages.push_back(llvm::json::Object{{"id", s.id},
                                          {"role", s.role},
                                          {"dtype", s.dtype},
                                          {"shape", std::move(shape)}});
  }
  for (auto &t : a.tasks) {
    llvm::json::Array accesses, deps, loops;
    for (auto d : t.dependencies)
      deps.push_back(a.tasks[d].id);
    for (auto &l : t.loops)
      loops.push_back(llvm::json::Object{
          {"lower", l.lower}, {"upper", l.upper}, {"step", l.step}});
    for (auto &x : t.accesses) {
      llvm::json::Array region, versions;
      for (auto &v : x.readVersions)
        versions.push_back(v);
      for (auto r : x.region)
        region.push_back(llvm::json::Array{r.lower, r.upper});
      accesses.push_back(
          llvm::json::Object{{"storage", a.storages[x.storage].id},
                             {"read", x.read},
                             {"write", x.write},
                             {"exact_region", x.exactRegion},
                             {"write_version", x.writeVersion},
                             {"read_versions", std::move(versions)},
                             {"region", std::move(region)}});
    }
    tasks.push_back(llvm::json::Object{{"id", t.id},
                                       {"dependencies", std::move(deps)},
                                       {"loops", std::move(loops)},
                                       {"accesses", std::move(accesses)}});
  }
  return llvm::json::Object{
      {"schema", "orbit.generic.analysis.v1"},
      {"body_fingerprint", a.bodyFingerprint},
      {"region_semantics", "conservative_static_bounding_boxes"},
      {"analysis_status", "analyzed"},
      {"codegen_status", "requires_target_validation"},
      {"cost_status", "uncalibrated"},
      {"storages", std::move(storages)},
      {"tasks", std::move(tasks)}};
}
