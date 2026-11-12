#include "Taskflow/GenericMapping.h"
#include "Taskflow/GenericMechanismTiming.h"
#include "Taskflow/GenericReuse.h"
#include "Taskflow/JSONUtils.h"
#include "Taskflow/NativeTransforms.h"
#include "Taskflow/ProgramMatrixImplementation.h"
#include "Taskflow/ProgramPartition.h"
#include "Taskflow/ProgramTargetBinding.h"
#include "Taskflow/StaticMapping.h"
#include "Taskflow/StructuredProgram.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/MD5.h"
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <tuple>
using namespace orbit::taskflow;
namespace {
thread_local GenericMappingStatistics mappingStatistics;
// Extracted order-only ABI arguments preserve memory-state identity. Prove
// they are unused before omitting payload DMA; the marker alone is no proof.
bool orderOnlyInput(const GenericAnalysis &source, size_t storage) {
  const auto &item = source.storages[storage];
  auto function = source.function;
  auto argument = mlir::dyn_cast<mlir::BlockArgument>(item.root);
  if (item.role != "input" || !argument ||
      argument.getOwner() != &function.getBody().front() ||
      !function.getArgAttr(argument.getArgNumber(), "orbit.order_only"))
    return false;
  for (const auto &task : source.tasks) {
    for (const auto &access : task.accesses)
      if (access.storage == storage)
        return false;
    auto operation = mlir::cast<mlir::taskflow::TaskflowTaskOp>(task.operation);
    for (size_t index = operation.getWillReads().size();
         index < task.argumentStorages.size(); ++index)
      if (task.argumentStorages[index] == storage)
        return false;
  }
  return true;
}

std::string computeDevice(const GenericMappingPlan &p) {
  auto *resource = programResource(p.memoryBinding.architecture,
                                   p.memoryBinding.computeResource);
  return resource ? resource->device : std::string{};
}
std::string computeOwner(const GenericMappingPlan &p) {
  std::string owner, error;
  programResourceOwner(p.memoryBinding.architecture,
                       p.memoryBinding.computeResource, owner, error);
  return owner;
}
std::string memoryOwner(const GenericMappingPlan &p, llvm::StringRef memory) {
  std::string owner, error;
  programMemoryOwner(p.memoryBinding.architecture, memory, owner, error);
  return owner;
}
llvm::json::Value canonicalChosenPlan(llvm::json::Value report) {
  // Alternative proposals are diagnostics, never selected implementation facts.
  // Keep them in reports while excluding them from either execution identity.
  return copyJSONWithoutFields(report, {"discovered_mapping_opportunities"});
}
llvm::json::Value directCostReport(const GenericDirectAIRCost &);
std::string directCostProvenance(const GenericDirectAIRCost &cost) {
  const auto *model = cost.model.getAsObject();
  if (model && (model->getString("kind") == "event-mechanism" ||
                model->getString("kind") == "region-invocation"))
    return "public event mechanism provider; activity-exclusive charge "
           "ownership; assumptions and unknown coverage explicit";
  return model && model->getString("kind") == "observed-table"
             ? "exact observed whole-command feedback; no prediction or "
               "generalization; transfer/runtime included"
             : "opt-in direct-AIR composite provider; calibration/domain and "
               "charge ownership explicit";
}
bool fail(std::string &e, const std::string &s) {
  e = s;
  return false;
}
int64_t storageBytes(const GenericStorage &s) {
  auto type = mlir::cast<mlir::MemRefType>(s.root.getType());
  auto element = type.getElementType();
  if (!element.isIntOrFloat())
    return -1;
  int64_t bytes = (element.getIntOrFloatBitWidth() + 7) / 8;
  for (auto n : s.shape) {
    if (n <= 0 || bytes > INT64_MAX / n)
      return -1;
    bytes *= n;
  }
  return bytes;
}
using namespace mlir;
using Interval = GenericInterval;
bool constant(Value v, int64_t &n) {
  llvm::APInt x;
  if (!matchPattern(v, m_ConstantInt(&x)))
    return false;
  n = x.getSExtValue();
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

bool covered(const std::vector<Interval> &,
             const std::vector<std::vector<Interval>> &);
// Re-evaluate the actual affine accesses for each restricted source loop
// domain.
bool phaseRegions(const GenericAnalysis &a, GenericTilePhase &phase,
                  int64_t &bytes, std::string &error) {
  const auto &task = a.tasks[phase.task];
  llvm::DenseMap<Value, Interval> ivs;
  for (auto b : phase.loops) {
    const auto &loop = task.loops[b.loop];
    ivs[loop.induction] = {
        b.lower, b.upper - 1 - (b.upper - b.lower - 1) % loop.step + 1};
  }
  std::map<size_t, GenericTileRegion> regions;
  std::map<size_t, std::vector<std::vector<Interval>>> writes;
  for (const auto &access : task.accesses) {
    SmallVector<Value> indices;
    AffineMap map;
    auto *op = access.operation;
    if (auto l = dyn_cast<affine::AffineLoadOp>(op)) {
      indices.append(l.getMapOperands().begin(), l.getMapOperands().end());
      map = l.getAffineMap();
    } else if (auto st = dyn_cast<affine::AffineStoreOp>(op)) {
      indices.append(st.getMapOperands().begin(), st.getMapOperands().end());
      map = st.getAffineMap();
    } else if (auto l = dyn_cast<memref::LoadOp>(op))
      indices.append(l.getIndices().begin(), l.getIndices().end());
    else if (auto st = dyn_cast<memref::StoreOp>(op))
      indices.append(st.getIndices().begin(), st.getIndices().end());
    SmallVector<Interval> operands, bounds;
    for (auto v : indices) {
      Interval r;
      if (!valueRange(v, ivs, r))
        return fail(error, "tile has unsupported affine index");
      operands.push_back(r);
    }
    if (map) {
      for (auto result : map.getResults()) {
        Interval r;
        if (!expr(result,
                  ArrayRef<Interval>(operands).take_front(map.getNumDims()),
                  ArrayRef<Interval>(operands).drop_front(map.getNumDims()), r))
          return fail(error, "tile has unsupported affine expression");
        bounds.push_back(r);
      }
    } else
      bounds = operands;
    if (access.write && access.exactRegion)
      writes[access.storage].emplace_back(bounds.begin(), bounds.end());
    auto found = regions.find(access.storage);
    if (found == regions.end()) {
      GenericTileRegion r;
      r.storage = access.storage;
      for (auto b : bounds) {
        r.offsets.push_back(b.lower);
        r.sizes.push_back(b.upper - b.lower);
      }
      found = regions.emplace(access.storage, std::move(r)).first;
    } else {
      auto &r = found->second;
      for (size_t d = 0; d < bounds.size(); ++d) {
        int64_t hi = std::max(r.offsets[d] + r.sizes[d], bounds[d].upper);
        r.offsets[d] = std::min(r.offsets[d], bounds[d].lower);
        r.sizes[d] = hi - r.offsets[d];
      }
    }
    auto &r = found->second;
    // Analysis records concrete reaching source writes. A read satisfied only
    // by writes in this task needs no uninitialized host-buffer import.
    if (access.read) {
      bool external = access.readVersions.empty();
      for (const auto &version : access.readVersions)
        external |= version.rfind(task.id + ":write:", 0) != 0;
      r.read |= external;
    }
    r.write |= access.write;
  }
  bytes = 0;
  for (auto &entry : regions) {
    auto &r = entry.second;
    if (r.write && !r.read) {
      std::vector<Interval> box;
      for (size_t d = 0; d < r.sizes.size(); ++d)
        box.push_back({r.offsets[d], r.offsets[d] + r.sizes[d]});
      if (!covered(box, writes[r.storage]))
        return fail(
            error,
            "tile output rectangle contains uninitialized unwritten elements");
    }
    auto type = cast<MemRefType>(a.storages[r.storage].root.getType());
    int64_t n = (type.getElementType().getIntOrFloatBitWidth() + 7) / 8;
    for (auto size : r.sizes) {
      if (size <= 0 || n > INT64_MAX / size)
        return fail(error, "tile byte overflow");
      n *= size;
    }
    if (bytes > INT64_MAX - n)
      return fail(error, "tile byte overflow");
    bytes += n;
    phase.regions.push_back(r);
  }
  return true;
}
// Prove different outer iterations touch disjoint written storage slices.
// Every access to a written storage must project that IV on the same axis.
bool disjointSlices(const GenericTask &task, size_t loop) {
  std::set<size_t> written;
  for (auto &access : task.accesses)
    if (access.write)
      written.insert(access.storage);
  for (size_t storage : written) {
    int axis = -1;
    for (auto &access : task.accesses) {
      if (access.storage != storage)
        continue;
      SmallVector<Value> indices;
      AffineMap map;
      if (auto op = dyn_cast<affine::AffineLoadOp>(access.operation)) {
        indices.append(op.getMapOperands().begin(), op.getMapOperands().end());
        map = op.getAffineMap();
      } else if (auto op = dyn_cast<affine::AffineStoreOp>(access.operation)) {
        indices.append(op.getMapOperands().begin(), op.getMapOperands().end());
        map = op.getAffineMap();
      } else if (auto op = dyn_cast<memref::LoadOp>(access.operation))
        indices.append(op.getIndices().begin(), op.getIndices().end());
      else if (auto op = dyn_cast<memref::StoreOp>(access.operation))
        indices.append(op.getIndices().begin(), op.getIndices().end());
      int projected = -1;
      if (map) {
        for (auto item : llvm::enumerate(map.getResults()))
          if (auto dim = dyn_cast<AffineDimExpr>(item.value()))
            if (indices[dim.getPosition()] == task.loops[loop].induction)
              projected = item.index();
      } else {
        for (auto item : llvm::enumerate(indices))
          if (item.value() == task.loops[loop].induction)
            projected = item.index();
      }
      if (projected < 0 || (axis >= 0 && projected != axis))
        return false;
      axis = projected;
    }
  }
  return true;
}
bool independentInput(const GenericAnalysis &a, const GenericTask &task,
                      size_t storage, size_t loop) {
  if (storage >= a.storages.size() || loop >= task.loops.size() ||
      a.storages[storage].role != "input")
    return false;
  bool found = false;
  for (const auto &access : task.accesses) {
    if (access.storage != storage)
      continue;
    if (access.write)
      return false;
    found = true;
    // Do not infer independence from equal bounding boxes: inspect the real
    // affine SSA operands, including expressions derived by affine.apply.
    std::function<bool(Value)> depends = [&](Value value) {
      if (value == task.loops[loop].induction)
        return true;
      if (auto apply = value.getDefiningOp<affine::AffineApplyOp>())
        return llvm::any_of(apply.getMapOperands(), depends);
      return false;
    };
    SmallVector<Value> indices;
    if (auto load = dyn_cast<affine::AffineLoadOp>(access.operation))
      indices.append(load.getMapOperands().begin(),
                     load.getMapOperands().end());
    else if (auto load = dyn_cast<memref::LoadOp>(access.operation))
      indices.append(load.getIndices().begin(), load.getIndices().end());
    else
      return false;
    if (llvm::any_of(indices, depends))
      return false;
  }
  return found;
}
bool panelTask(const GenericAnalysis &a, size_t task,
               const ProgramMappingChoice &choice) {
  const auto &t = a.tasks[task];
  if (choice.storage < 0 || choice.reuseLoop < 0 || choice.panelLoop < 0 ||
      choice.reuseLoop == choice.panelLoop ||
      size_t(choice.panelLoop) >= t.loops.size() ||
      !independentInput(a, t, choice.storage, choice.reuseLoop) ||
      !disjointSlices(t, choice.reuseLoop) ||
      !disjointSlices(t, choice.panelLoop))
    return false;
  for (const auto &loop : t.loops)
    for (const auto &access : t.accesses)
      if (!loop.operation->isProperAncestor(access.operation))
        return false;
  // This finite strategy only permutes the first two independent outer axes.
  // Other loop permutations need a separate dependence/accumulator proof.
  return std::min(choice.reuseLoop, choice.panelLoop) == 0 &&
         std::max(choice.reuseLoop, choice.panelLoop) == 1;
}
int64_t vectorLoop(const GenericAnalysis &a, const GenericTask &task) {
  if (!llvm::any_of(task.accesses,
                    [](const auto &access) { return access.read; }))
    return -1; // Installed affine vectorizer does not rewrite pure fills.
  for (const auto &loop : task.loops)
    if (!isa<affine::AffineForOp>(loop.operation))
      return -1;
  bool scalarMath = false;
  task.operation->walk([&](Operation *op) {
    if (op->getName().getDialectNamespace() == "math")
      scalarMath = true;
  });
  if (scalarMath)
    return -1;
  for (size_t l = task.loops.size(); l > 0; --l) {
    size_t candidate = l - 1;
    std::function<bool(Value)> depends = [&](Value value) {
      if (value == task.loops[candidate].induction)
        return true;
      if (auto apply = value.getDefiningOp<affine::AffineApplyOp>())
        return llvm::any_of(apply.getMapOperands(), depends);
      return false;
    };
    bool legal = disjointSlices(task, candidate), written = false;
    for (const auto &access : task.accesses) {
      auto type = cast<MemRefType>(a.storages[access.storage].root.getType());
      if (!type.getElementType().isF32()) {
        legal = false;
        break;
      }
      AffineMap map;
      ValueRange operands;
      if (auto load = dyn_cast<affine::AffineLoadOp>(access.operation)) {
        map = load.getAffineMap();
        operands = load.getMapOperands();
      } else if (auto store =
                     dyn_cast<affine::AffineStoreOp>(access.operation)) {
        map = store.getAffineMap();
        operands = store.getMapOperands();
        written = true;
      } else {
        legal = false;
        break;
      }
      bool varying = false;
      for (size_t d = 0; d < operands.size(); ++d) {
        if (operands[d] != task.loops[candidate].induction) {
          // A derived index is not an invariant broadcast. Keep the current
          // direct identity-last-dimension proof conservative; a future
          // vector implementation may separately prove composed affine maps.
          if (depends(operands[d]))
            legal = false;
          continue;
        }
        varying = true;
        if (d >= map.getNumDims()) {
          legal = false;
          break;
        }
        for (size_t axis = 0; axis < map.getNumResults(); ++axis) {
          AffineExpr expression = map.getResult(axis);
          if (!expression.isFunctionOfDim(d))
            continue;
          auto dim = dyn_cast<AffineDimExpr>(expression);
          if (axis + 1 != map.getNumResults() || !dim || dim.getPosition() != d)
            legal = false;
        }
      }
      if (access.write && !varying)
        legal = false;
    }
    if (legal && written)
      return candidate;
  }
  return -1;
}
void inferReadOnlyRetentions(const GenericAnalysis &a, GenericMappingPlan &p) {
  p.readOnlyRetentions.clear();
  if (p.choices.residency != "readonly-panel")
    return;
  for (const auto &stage : p.hardwareStages) {
    size_t begin = stage.beginPhase;
    while (begin < stage.endPhase) {
      const auto &first = p.phases[begin];
      if (!panelTask(a, first.task, p.choices))
        break;
      auto bound = first.loops[p.choices.panelLoop];
      size_t end = begin + 1;
      while (end < stage.endPhase &&
             p.phases[end].loops[p.choices.panelLoop].lower == bound.lower &&
             p.phases[end].loops[p.choices.panelLoop].upper == bound.upper)
        ++end;
      GenericReadOnlyRetention scope;
      scope.beginPhase = begin;
      scope.endPhase = end;
      scope.storage = p.choices.storage;
      for (size_t index = begin; index < end; ++index)
        for (const auto &region : p.phases[index].regions)
          if (region.storage == scope.storage) {
            if (scope.sizes.empty()) {
              scope.offsets = region.offsets;
              scope.sizes = region.sizes;
            } else
              for (size_t d = 0; d < region.sizes.size(); ++d) {
                int64_t hi = std::max(scope.offsets[d] + scope.sizes[d],
                                      region.offsets[d] + region.sizes[d]);
                scope.offsets[d] =
                    std::min(scope.offsets[d], region.offsets[d]);
                scope.sizes[d] = hi - scope.offsets[d];
              }
          }
      p.readOnlyRetentions.push_back(std::move(scope));
      begin = end;
    }
  }
  std::map<size_t, const GenericReadOnlyRetention *> begins;
  for (const auto &scope : p.readOnlyRetentions)
    begins[scope.beginPhase] = &scope;
  p.transfers.erase(
      std::remove_if(p.transfers.begin(), p.transfers.end(),
                     [&](GenericTransfer &transfer) {
                       if (transfer.storage != size_t(p.choices.storage) ||
                           transfer.direction != "host_to_l1")
                         return false;
                       auto found = begins.upper_bound(transfer.phase);
                       if (found == begins.begin())
                         return false;
                       const auto &scope = *std::prev(found)->second;
                       if (transfer.phase >= scope.endPhase)
                         return false;
                       if (transfer.phase != scope.beginPhase)
                         return true;
                       transfer.offsets = scope.offsets;
                       transfer.sizes = scope.sizes;
                       auto type = cast<MemRefType>(
                           a.storages[transfer.storage].root.getType());
                       transfer.bytes =
                           (type.getElementType().getIntOrFloatBitWidth() + 7) /
                           8;
                       for (auto size : scope.sizes)
                         transfer.bytes *= size;
                       return false;
                     }),
      p.transfers.end());
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
bool sameRegion(const GenericTileRegion &x, const GenericTileRegion &y) {
  return x.storage == y.storage && x.offsets == y.offsets && x.sizes == y.sizes;
}
const GenericReadOnlyRetention *readOnlyScope(const GenericMappingPlan &p,
                                              size_t phase, size_t storage) {
  auto found = std::upper_bound(
      p.readOnlyRetentions.begin(), p.readOnlyRetentions.end(), phase,
      [](size_t index, const auto &scope) { return index < scope.beginPhase; });
  if (found == p.readOnlyRetentions.begin())
    return nullptr;
  --found;
  return found->storage == storage && phase < found->endPhase ? &*found
                                                              : nullptr;
}
// Prove fusion from source accesses, without recognizing an arithmetic kernel.
bool initializerPair(const GenericAnalysis &a, size_t producer, size_t consumer,
                     size_t storage) {
  if (producer >= a.tasks.size() || consumer >= a.tasks.size() ||
      producer >= consumer || storage >= a.storages.size())
    return false;
  const auto &init = a.tasks[producer];
  const auto &update = a.tasks[consumer];
  size_t rank = a.storages[storage].shape.size();
  if (init.loops.size() != rank || update.loops.size() < rank ||
      init.accesses.empty())
    return false;
  size_t successors = 0;
  for (size_t task = 0; task < a.tasks.size(); ++task)
    if (llvm::is_contained(a.tasks[task].dependencies, producer)) {
      ++successors;
      if (task != consumer)
        return false;
    }
  if (successors != 1)
    return false;
  for (size_t axis = 0; axis < rank; ++axis) {
    const auto &x = init.loops[axis];
    const auto &y = update.loops[axis];
    if (x.lower != 0 || x.upper != a.storages[storage].shape[axis] ||
        x.step != 1 || x.lower != y.lower || x.upper != y.upper ||
        x.step != y.step)
      return false;
  }
  auto projection = [&](const GenericTask &task, const GenericAccess &access) {
    SmallVector<Value> indices;
    AffineMap map;
    if (auto op = dyn_cast<affine::AffineLoadOp>(access.operation)) {
      indices.append(op.getMapOperands().begin(), op.getMapOperands().end());
      map = op.getAffineMap();
    } else if (auto op = dyn_cast<affine::AffineStoreOp>(access.operation)) {
      indices.append(op.getMapOperands().begin(), op.getMapOperands().end());
      map = op.getAffineMap();
    } else if (auto op = dyn_cast<memref::LoadOp>(access.operation))
      indices.append(op.getIndices().begin(), op.getIndices().end());
    else if (auto op = dyn_cast<memref::StoreOp>(access.operation))
      indices.append(op.getIndices().begin(), op.getIndices().end());
    else
      return false;
    if (map && !map.isIdentity())
      return false;
    if (indices.size() != rank)
      return false;
    for (size_t axis = 0; axis < rank; ++axis)
      if (indices[axis] != task.loops[axis].induction)
        return false;
    return access.exactRegion;
  };
  for (const auto &access : init.accesses)
    if (access.read || !access.write || access.storage != storage ||
        !projection(init, access))
      return false;
  bool reads = false, writes = false;
  for (const auto &access : update.accesses)
    if (access.storage == storage) {
      reads |= access.read;
      writes |= access.write;
      if (!projection(update, access))
        return false;
    }
  if (!reads || !writes)
    return false;
  // No intervening observation or modification can be moved past the local
  // initializer. Later tasks see the consumer's complete projected writes.
  for (size_t task = producer + 1; task < consumer; ++task)
    for (const auto &access : a.tasks[task].accesses)
      if (access.storage == storage)
        return false;
  return true;
}
std::map<std::pair<size_t, size_t>, size_t>
initializerCandidates(const GenericAnalysis &a) {
  std::map<std::pair<size_t, size_t>, size_t> result;
  for (size_t producer = 0; producer < a.tasks.size(); ++producer) {
    if (a.tasks[producer].accesses.empty())
      continue;
    size_t storage = a.tasks[producer].accesses.front().storage;
    for (size_t consumer = producer + 1; consumer < a.tasks.size(); ++consumer)
      if (initializerPair(a, producer, consumer, storage)) {
        result[{consumer, storage}] = producer;
        break;
      }
  }
  return result;
}
void inferRetentions(const GenericAnalysis &a, GenericMappingPlan &p) {
  struct Run {
    size_t begin, last, task;
    GenericTileRegion region;
  };
  std::map<size_t, Run> active;
  auto candidates = initializerCandidates(a);
  auto finish = [&](const Run &run) {
    auto candidate = candidates.find({run.task, run.region.storage});
    if (run.last > run.begin || candidate != candidates.end()) {
      int64_t initializer =
          candidate == candidates.end() ? -1 : int64_t(candidate->second);
      p.retainedRegions.push_back(
          {run.begin, run.last + 1, run.region.storage, initializer});
      if (initializer >= 0 &&
          !llvm::is_contained(p.fusedInitializerTasks, size_t(initializer)))
        p.fusedInitializerTasks.push_back(initializer);
    }
  };
  p.retainedRegions.clear();
  p.fusedInitializerTasks.clear();
  for (size_t index = 0; index < p.phases.size(); ++index) {
    const auto &phase = p.phases[index];
    for (const auto &region : phase.regions) {
      if (!region.read || !region.write)
        continue;
      auto previous = active.find(region.storage);
      if (previous != active.end() && previous->second.last + 1 == index &&
          previous->second.task == phase.task &&
          sameRegion(previous->second.region, region))
        previous->second.last = index;
      else {
        if (previous != active.end())
          finish(previous->second);
        active[region.storage] = {index, index, phase.task, region};
      }
    }
  }
  for (const auto &run : active)
    finish(run.second);
  std::sort(p.retainedRegions.begin(), p.retainedRegions.end(),
            [](const auto &x, const auto &y) {
              return std::tie(x.beginPhase, x.storage) <
                     std::tie(y.beginPhase, y.storage);
            });
  std::map<size_t, std::vector<const GenericTileRetention *>> scopes;
  for (const auto &scope : p.retainedRegions)
    scopes[scope.storage].push_back(&scope);
  std::map<size_t, size_t> cursor;
  auto remove = [&](const GenericTransfer &transfer) {
    if (llvm::is_contained(p.fusedInitializerTasks,
                           p.phases[transfer.phase].task))
      return true;
    auto &list = scopes[transfer.storage];
    auto &index = cursor[transfer.storage];
    while (index < list.size() && list[index]->endPhase <= transfer.phase)
      ++index;
    if (index >= list.size() || transfer.phase < list[index]->beginPhase)
      return false;
    return transfer.direction == "host_to_l1"
               ? list[index]->initializerTask >= 0 ||
                     transfer.phase != list[index]->beginPhase
               : transfer.phase + 1 != list[index]->endPhase;
  };
  p.transfers.erase(
      std::remove_if(p.transfers.begin(), p.transfers.end(), remove),
      p.transfers.end());
}
std::vector<int64_t> hardwareShapeSignature(const GenericMappingPlan &p,
                                            size_t phaseIndex) {
  const auto &phase = p.phases[phaseIndex];
  bool retained = llvm::any_of(phase.regions, [](const auto &region) {
    return region.read && region.write;
  });
  std::vector<int64_t> signature;
  if (retained) {
    for (const auto &region : phase.regions)
      if (region.write) {
        signature.push_back(region.storage);
        signature.push_back(region.sizes.size());
        signature.insert(signature.end(), region.sizes.begin(),
                         region.sizes.end());
      }
  } else
    for (auto bound : phase.loops)
      signature.push_back(bound.upper - bound.lower);
  return signature;
}
void inferHardwareStages(GenericMappingPlan &p) {
  p.hardwareStages.clear();
  std::map<size_t, size_t> protectedEnd;
  for (const auto &scope : p.retainedRegions)
    protectedEnd[scope.beginPhase] =
        std::max(protectedEnd[scope.beginPhase], scope.endPhase);
  size_t begin = 0;
  while (begin < p.phases.size()) {
    size_t task = p.phases[begin].task, end = begin + 1;
    while (end < p.phases.size() && p.phases[end].task == task)
      ++end;
    if (!llvm::is_contained(p.fusedInitializerTasks, task)) {
      size_t stageBegin = begin, openEnd = begin;
      auto signature = hardwareShapeSignature(p, begin);
      for (size_t phase = begin; phase < end; ++phase) {
        auto next = hardwareShapeSignature(p, phase);
        if (phase > stageBegin && next != signature && phase >= openEnd) {
          p.hardwareStages.push_back({stageBegin, phase});
          stageBegin = phase;
          signature = std::move(next);
        }
        auto found = protectedEnd.find(phase);
        if (found != protectedEnd.end())
          openEnd = std::max(openEnd, found->second);
      }
      p.hardwareStages.push_back({stageBegin, end});
    }
    begin = end;
  }
}
// Backend-declared stream limits bound this implementation realization.
// Only shape/rate conflicts require explicit legalization; all templates of
// one storage share its planned port, preserving the serial source order.
bool deriveChannelAllocations(
    const GenericAnalysis &a, const GenericMappingPlan &p,
    std::vector<bool> &required,
    std::vector<GenericAIRChannelAllocation> &allocations, std::string &error) {
  required.clear();
  allocations.clear();
  for (size_t stageIndex = 0; stageIndex < p.hardwareStages.size();
       ++stageIndex) {
    const auto &stage = p.hardwareStages[stageIndex];
    std::set<size_t> readonlyStorages;
    std::set<std::vector<int64_t>> templates;
    for (size_t index = stage.beginPhase; index < stage.endPhase; ++index) {
      const auto &phase = p.phases[index];
      for (const auto &region : phase.regions)
        if (region.read && !region.write) {
          readonlyStorages.insert(region.storage);
          std::vector<int64_t> key{int64_t(region.storage),
                                   int64_t(region.sizes.size())};
          key.insert(key.end(), region.sizes.begin(), region.sizes.end());
          for (auto bound : phase.loops)
            key.push_back(bound.upper - bound.lower);
          templates.insert(std::move(key));
        }
    }
    bool needed = templates.size() > readonlyStorages.size();
    required.push_back(needed);
    if (!needed)
      continue;
    std::set<size_t> inputs, outputs;
    std::map<std::pair<size_t, std::string>, std::pair<int64_t, int64_t>>
        traffic;
    auto transfer = std::lower_bound(p.transfers.begin(), p.transfers.end(),
                                     stage.beginPhase,
                                     [](const auto &transfer, size_t phase) {
                                       return transfer.phase < phase;
                                     });
    for (; transfer != p.transfers.end() && transfer->phase < stage.endPhase;
         ++transfer) {
      (transfer->direction == "host_to_l1" ? inputs : outputs)
          .insert(transfer->storage);
      auto &totals = traffic[{transfer->storage, transfer->direction}];
      ++totals.first;
      totals.second += transfer->bytes;
    }
    if (int64_t(inputs.size()) > p.targetCapabilities.maxGroupInputs ||
        int64_t(outputs.size()) > p.targetCapabilities.maxGroupOutputs)
      return fail(error, "planned DMA channel legalization exceeds "
                         "target-declared input/output stream limits");
    auto allocate = [&](size_t storage, llvm::StringRef direction,
                        int64_t port) {
      GenericAIRChannelAllocation allocation{stageIndex, storage,
                                             direction.str(), port};
      allocation.logicalFifo = "__orbit_fifo_stage" +
                               std::to_string(stageIndex) + "_storage" +
                               std::to_string(storage) +
                               (direction == "host_to_l1" ? "_in" : "_out");
      if (allocation.logicalFifo == mlir::func::FuncOp(a.function).getName())
        allocation.logicalFifo += "_channel";
      auto totals = traffic[{storage, direction.str()}];
      allocation.transferCount = totals.first;
      allocation.bytes = totals.second;
      allocations.push_back(std::move(allocation));
    };
    int64_t port = 0;
    for (auto storage : inputs)
      allocate(storage, "host_to_l1", port++);
    port = 0;
    for (auto storage : outputs)
      allocate(storage, "l1_to_host", port++);
  }
  return true;
}
bool localRetained(const GenericMappingPlan &, size_t, size_t);
constexpr size_t maxHardwareStageExpansion = 128;
bool isolateConflictingChannelScopes(const GenericAnalysis &a,
                                     GenericMappingPlan &p,
                                     std::string &error) {
  std::vector<bool> required;
  std::vector<GenericAIRChannelAllocation> ignored;
  if (!deriveChannelAllocations(a, p, required, ignored, error))
    return false;
  std::vector<GenericHardwareStage> stages;
  for (size_t index = 0; index < p.hardwareStages.size(); ++index) {
    const auto &stage = p.hardwareStages[index];
    if (!required[index])
      stages.push_back(stage);
    else {
      size_t cursor = stage.beginPhase;
      while (cursor < stage.endPhase) {
        size_t end = cursor;
        for (const auto &scope : p.retainedRegions)
          if (scope.beginPhase == cursor && scope.endPhase <= stage.endPhase) {
            if (end != cursor && end != scope.endPhase)
              return fail(error, "channel-conflict source output scopes have "
                                 "incompatible lifetimes");
            end = scope.endPhase;
          }
        if (end == cursor)
          return fail(error, "channel-conflict stage requires complete "
                             "retained-output scope boundaries");
        for (const auto &scope : p.retainedRegions)
          if (scope.beginPhase < end && scope.endPhase > cursor &&
              (scope.beginPhase != cursor || scope.endPhase != end))
            return fail(error, "channel-conflict hardware stage would split a "
                               "retained source lifetime");
        stages.push_back({cursor, end, true});
        cursor = end;
        if (stages.size() > maxHardwareStageExpansion)
          return fail(error,
                      "hardware-stage expansion exceeds 128 stages; "
                      "host/codegen complexity bound, not SRAM capacity");
      }
    }
    if (stages.size() > maxHardwareStageExpansion)
      return fail(error, "hardware-stage expansion exceeds 128 stages; "
                         "host/codegen complexity bound, not SRAM capacity");
  }
  p.hardwareStages = std::move(stages);
  return true;
}
// Distinct hardware stages configure local storage separately. Read-only DMA
// templates have independent pools for each actual shape and loop-rate shape.
int64_t stagePoolBytes(const GenericAnalysis &a, const GenericMappingPlan &p,
                       int64_t *phasePeak = nullptr,
                       std::vector<int64_t> *stageBytes = nullptr,
                       std::vector<llvm::json::Array> *stagePools = nullptr,
                       std::vector<MappingAllocation> *allocations = nullptr) {
  std::set<std::pair<size_t, size_t>> localInitializers;
  for (const auto &scope : p.retainedRegions)
    if (scope.initializerTask >= 0)
      localInitializers.insert(
          {p.phases[scope.beginPhase].task, scope.storage});
  int64_t allocated = 0, peakLive = 0;
  for (const auto &stage : p.hardwareStages) {
    std::map<std::vector<int64_t>, int64_t> readonlyPools;
    std::map<size_t, int64_t> panelPools;
    std::map<size_t, int64_t> writtenPools;
    std::set<size_t> staging;
    for (size_t index = stage.beginPhase; index < stage.endPhase; ++index) {
      const auto &phase = p.phases[index];
      int64_t live = 0;
      for (const auto &region : phase.regions) {
        auto type = cast<MemRefType>(a.storages[region.storage].root.getType());
        int64_t bytes = (type.getElementType().getIntOrFloatBitWidth() + 7) / 8;
        for (auto n : region.sizes)
          bytes *= n;
        if (auto scope = readOnlyScope(p, index, region.storage)) {
          int64_t panelBytes =
              (type.getElementType().getIntOrFloatBitWidth() + 7) / 8;
          for (auto n : scope->sizes)
            panelBytes *= n;
          panelPools[region.storage] =
              std::max(panelPools[region.storage], panelBytes);
        } else if (region.read && !region.write &&
                   !localRetained(p, index, region.storage)) {
          std::vector<int64_t> key{int64_t(region.storage),
                                   int64_t(region.sizes.size())};
          key.insert(key.end(), region.sizes.begin(), region.sizes.end());
          for (auto bound : phase.loops)
            key.push_back(bound.upper - bound.lower);
          readonlyPools[key] = bytes;
        } else
          writtenPools[region.storage] =
              std::max(writtenPools[region.storage], bytes);
        if (region.read && region.write &&
            !localInitializers.count({phase.task, region.storage}))
          staging.insert(region.storage);
        live += bytes;
      }
      peakLive = std::max(peakLive, live);
    }
    int64_t bytes = 0;
    llvm::json::Array poolReport;
    auto reportPool = [&](size_t storage, int64_t allocatedBytes,
                          llvm::StringRef kind,
                          const std::vector<int64_t> *templateKey = nullptr) {
      if (allocations) {
        std::string key = "storage-" + std::to_string(storage) + ".data";
        if (kind == "readonly_template") {
          key += ".input";
          size_t rank = (*templateKey)[1];
          for (size_t d = 0; d < rank; ++d)
            key += "." + std::to_string((*templateKey)[2 + d]);
          key += ".loops";
          for (size_t d = 2 + rank; d < templateKey->size(); ++d)
            key += "." + std::to_string((*templateKey)[d]);
        } else if (kind == "readonly_panel")
          key += ".readonly-panel";
        else if (kind == "incoming_staging")
          key += ".incoming";
        allocations->push_back({storage, p.memoryBinding.localMemory, key,
                                allocatedBytes, stage.beginPhase,
                                stage.endPhase});
      }
      if (!stagePools)
        return;
      auto type = cast<MemRefType>(a.storages[storage].root.getType());
      int64_t width = (type.getElementType().getIntOrFloatBitWidth() + 7) / 8;
      llvm::json::Array shape, loopExtents;
      if (templateKey) {
        size_t rank = (*templateKey)[1];
        for (size_t d = 0; d < rank; ++d)
          shape.push_back((*templateKey)[2 + d]);
        for (size_t d = 2 + rank; d < templateKey->size(); ++d)
          loopExtents.push_back((*templateKey)[d]);
      }
      poolReport.push_back(llvm::json::Object{
          {"storage_index", int64_t(storage)},
          {"storage", a.storages[storage].id},
          {"dtype", a.storages[storage].dtype},
          {"kind", kind.str()},
          {"memory", p.memoryBinding.localMemory},
          {"allocated_bytes", allocatedBytes},
          {"flat_elements", allocatedBytes / width},
          {"template_shape", std::move(shape)},
          {"template_loop_extents", std::move(loopExtents)},
          {"begin_phase", int64_t(stage.beginPhase)},
          {"end_phase_exclusive", int64_t(stage.endPhase)},
          {"lifetime",
           "whole hardware stage; reused after serial completion"}});
    };
    for (const auto &pool : readonlyPools) {
      bytes += pool.second;
      reportPool(pool.first[0], pool.second, "readonly_template", &pool.first);
    }
    for (const auto &pool : panelPools) {
      bytes += pool.second;
      reportPool(pool.first, pool.second, "readonly_panel");
    }
    for (const auto &pool : writtenPools) {
      bytes += pool.second * (staging.count(pool.first) ? 2 : 1);
      reportPool(pool.first, pool.second, "writable");
      if (staging.count(pool.first))
        reportPool(pool.first, pool.second, "incoming_staging");
    }
    if (stagePools)
      stagePools->push_back(std::move(poolReport));
    allocated = std::max(allocated, bytes);
    if (stageBytes)
      stageBytes->push_back(bytes);
  }
  if (phasePeak)
    *phasePeak = peakLive;
  return allocated;
}
std::vector<size_t> semanticGroupDependencies(const GenericAnalysis &a,
                                              const GenericMappingPlan &p,
                                              size_t index) {
  std::set<size_t> parents;
  const auto &group = p.executionGroups[index];
  for (size_t phase = group.beginPhase; phase < group.endPhase; ++phase)
    for (size_t dependency : a.tasks[p.phases[phase].task].dependencies)
      for (size_t g = 0; g < index; ++g)
        for (size_t prior = p.executionGroups[g].beginPhase;
             prior < p.executionGroups[g].endPhase; ++prior)
          if (p.phases[prior].task == dependency) {
            parents.insert(g);
            break;
          }
  return {parents.begin(), parents.end()};
}
void bindGroupDependencies(const GenericAnalysis &a, GenericMappingPlan &p) {
  for (size_t i = 0; i < p.executionGroups.size(); ++i) {
    p.executionGroups[i].dependencies = semanticGroupDependencies(a, p, i);
    p.executionGroups[i].implementationDependencies =
        i ? std::vector<size_t>{i - 1} : std::vector<size_t>{};
  }
}
bool buildTiled(const GenericAnalysis &, GenericMappingPlan &, std::string &);
bool arranged(const GenericMappingPlan &p) {
  return p.choices.grouping == "tile-major" ||
         p.choices.grouping == "shape-major";
}
bool localRetained(const GenericMappingPlan &p, size_t phase, size_t storage) {
  return genericLocallyRetained(p, phase, storage);
}
bool independentPointwise(const GenericAnalysis &a, const GenericMappingPlan &p,
                          std::string &error) {
  for (auto task : p.orderedTasks) {
    const auto &t = a.tasks[task];
    if (t.loops.empty())
      return fail(error,
                  "instance ordering requires nonempty static pointwise loops");
    for (size_t l = 0; l < t.loops.size(); ++l)
      if (!disjointSlices(t, l))
        return fail(error,
                    "instance ordering has unproven cross-tile dependence");
    for (auto &access : t.accesses)
      for (auto &loop : t.loops)
        if (!loop.operation->isProperAncestor(access.operation))
          return fail(error,
                      "instance ordering would repeat non-tile source work");
  }
  if (!p.retainedRegions.empty() || !p.fusedInitializerTasks.empty() ||
      !p.readOnlyRetentions.empty())
    return fail(error,
                "instance ordering currently excludes reduction/panel scopes");
  return true;
}
bool arrangePhases(const GenericAnalysis &a, GenericMappingPlan &p,
                   std::string &error) {
  if (!independentPointwise(a, p, error))
    return false;
  for (auto &phase : p.phases)
    for (auto &r : phase.regions)
      if (r.read && r.write)
        return fail(
            error,
            "pointwise instance grouping excludes in-place/reduction updates");
  if (p.choices.grouping == "shape-major") {
    for (size_t begin = 0; begin < p.phases.size();) {
      size_t end = begin + 1;
      while (end < p.phases.size() &&
             p.phases[end].task == p.phases[begin].task)
        ++end;
      std::stable_sort(p.phases.begin() + begin, p.phases.begin() + end,
                       [](const auto &x, const auto &y) {
                         std::vector<int64_t> a, b;
                         for (auto l : x.loops)
                           a.push_back(l.upper - l.lower);
                         for (auto l : y.loops)
                           b.push_back(l.upper - l.lower);
                         return a > b;
                       });
      begin = end;
    }
  } else {
    auto decisions = genericResidencyGroups(p.choices);
    if (decisions.empty())
      return fail(error, "tile-major requires nonempty residency groups");
    std::vector<std::vector<GenericTilePhase>> byTask(a.tasks.size());
    for (auto &phase : p.phases)
      byTask[phase.task].push_back(phase);
    std::map<size_t, ProgramResidencyGroup> scopes;
    std::set<size_t> groupedTasks, retainedData;
    size_t previousEnd = 0;
    bool first = true;
    for (const auto &decision : decisions) {
      if (decision.tasks.size() < 2 || decision.retainedStorages.empty())
        return fail(
            error, "residency group requires multiple tasks and retained data");
      auto where = std::find(p.orderedTasks.begin(), p.orderedTasks.end(),
                             decision.tasks.front());
      if (where == p.orderedTasks.end())
        return fail(error, "residency group references unknown task");
      size_t begin = where - p.orderedTasks.begin(),
             end = begin + decision.tasks.size();
      if (end > p.orderedTasks.size() || (!first && begin < previousEnd))
        return fail(error,
                    "residency groups must be ordered and nonoverlapping");
      for (size_t t = 0; t < decision.tasks.size(); ++t)
        if (decision.tasks[t] != p.orderedTasks[begin + t] ||
            !groupedTasks.insert(decision.tasks[t]).second)
          return fail(error, "residency group requires a contiguous "
                             "topological task interval");
      first = false;
      previousEnd = end;
      std::set<size_t> selected;
      for (auto storage : decision.retainedStorages) {
        if (storage >= a.storages.size() || !selected.insert(storage).second ||
            !retainedData.insert(storage).second ||
            (a.storages[storage].role != "temporary" &&
             a.storages[storage].role != "output" &&
             a.storages[storage].role != "input"))
          return fail(error,
                      "retained data must be unique storage in one scope");
        size_t writers = 0, readers = 0;
        bool writerInside = false;
        for (size_t task = 0; task < a.tasks.size(); ++task) {
          bool write = false, read = false;
          for (auto &access : a.tasks[task].accesses)
            if (access.storage == storage) {
              write |= access.write;
              read |= access.read;
            }
          bool inside = llvm::is_contained(decision.tasks, task);
          writers += write;
          writerInside |= write && inside;
          readers += read && inside;
          if (read && !inside && a.storages[storage].role != "input" &&
              std::find(p.orderedTasks.begin(), p.orderedTasks.end(), task) <
                  where)
            return fail(error,
                        "retained boundary consumer precedes producer scope");
        }
        if (a.storages[storage].role == "input") {
          if (writers || readers < 2)
            return fail(error, "shared input retention needs immutable input "
                               "and multiple member readers");
        } else if (writers != 1 || !writerInside || readers < 1)
          return fail(error, "local retention needs one source producer and "
                             "internal consumers");
      }
      for (size_t storage = 0; storage < a.storages.size(); ++storage) {
        size_t writers = 0, readers = 0;
        for (auto task : decision.tasks) {
          bool write = false, read = false;
          for (const auto &access : a.tasks[task].accesses)
            if (access.storage == storage) {
              write |= access.write;
              read |= access.read;
            }
          writers += write;
          readers += read;
        }
        if (writers > 1)
          return fail(error,
                      "scope requires one task writer per logical storage");
        if (!selected.count(storage) && readers > 1 && writers)
          return fail(
              error,
              "group-internal produced data requires explicit local retention");
      }
      for (auto task : decision.tasks)
        for (auto parent : a.tasks[task].dependencies)
          if (llvm::is_contained(decision.tasks, parent)) {
            bool actual = false;
            for (auto &read : a.tasks[task].accesses)
              if (read.read)
                for (auto &write : a.tasks[parent].accesses)
                  if (write.write && write.storage == read.storage) {
                    actual = true;
                    if (!selected.count(read.storage))
                      return fail(error, "group-internal produced data "
                                         "requires explicit local retention");
                  }
            if (!actual)
              return fail(
                  error,
                  "tile-major cannot weaken whole-task pure-order barrier");
          }
      scopes.emplace(begin, decision);
    }
    std::vector<GenericTilePhase> arrangedPhases;
    p.residencyScopes.clear();
    for (size_t position = 0; position < p.orderedTasks.size();) {
      auto found = scopes.find(position);
      if (found == scopes.end()) {
        for (auto &phase : byTask[p.orderedTasks[position]])
          arrangedPhases.push_back(phase);
        ++position;
        continue;
      }
      const auto &decision = found->second;
      const auto &reference = byTask[decision.tasks.front()];
      if (reference.empty())
        return fail(error, "empty residency tile group");
      for (auto task : decision.tasks) {
        const auto &phases = byTask[task];
        if (phases.size() != reference.size())
          return fail(error, "tile group needs matching spatial grids");
        for (size_t i = 0; i < reference.size(); ++i) {
          if (phases[i].loops.size() != reference[i].loops.size())
            return fail(error, "tile group rank mismatch");
          for (size_t l = 0; l < reference[i].loops.size(); ++l) {
            auto x = phases[i].loops[l], y = reference[i].loops[l];
            if (x.loop != y.loop || x.lower != y.lower || x.upper != y.upper)
              return fail(error, "tile group domain mismatch");
          }
        }
      }
      for (size_t i = 0; i < reference.size(); ++i)
        for (auto storage : decision.retainedStorages) {
          const GenericTileRegion *base = nullptr;
          bool produced = a.storages[storage].role == "input";
          for (auto task : decision.tasks)
            for (auto &r : byTask[task][i].regions)
              if (r.storage == storage) {
                if (!base)
                  base = &r;
                else if (!sameRegion(*base, r))
                  return fail(
                      error,
                      "retained consumer must access exact producer tile");
                if (r.read && !produced)
                  return fail(error, "retained tile read before producer");
                produced |= r.write;
              }
        }
      // A shared boundary delivery requires identical actual rectangles.
      for (size_t i = 0; i < reference.size(); ++i) {
        std::map<size_t, const GenericTileRegion *> reads;
        for (auto task : decision.tasks)
          for (const auto &r : byTask[task][i].regions)
            if (r.read &&
                !llvm::is_contained(decision.retainedStorages, r.storage)) {
              auto inserted = reads.emplace(r.storage, &r);
              if (!inserted.second && !sameRegion(*inserted.first->second, r))
                return fail(error, "shared boundary input requires identical "
                                   "rectangular regions");
            }
      }
      GenericResidencyScope scope;
      scope.beginPhase = arrangedPhases.size();
      scope.tasks = decision.tasks;
      scope.retainedStorages = decision.retainedStorages;
      for (auto storage : scope.retainedStorages) {
        if (a.storages[storage].role == "input")
          continue;
        bool boundary = a.storages[storage].role == "output";
        for (size_t task = 0; task < a.tasks.size(); ++task)
          if (!llvm::is_contained(scope.tasks, task))
            for (const auto &access : a.tasks[task].accesses)
              boundary |= access.storage == storage && access.read;
        if (boundary)
          scope.boundaryExports.push_back(storage);
      }
      std::vector<size_t> coordinates;
      for (size_t i = 0; i < reference.size(); ++i)
        coordinates.push_back(i);
      std::stable_sort(coordinates.begin(), coordinates.end(),
                       [&](size_t x, size_t y) {
                         std::vector<int64_t> a, b;
                         for (auto l : reference[x].loops)
                           a.push_back(l.upper - l.lower);
                         for (auto l : reference[y].loops)
                           b.push_back(l.upper - l.lower);
                         return a > b;
                       });
      for (size_t i : coordinates)
        for (auto task : decision.tasks)
          arrangedPhases.push_back(byTask[task][i]);
      scope.endPhase = arrangedPhases.size();
      p.residencyScopes.push_back(std::move(scope));
      position += decision.tasks.size();
    }
    p.phases = std::move(arrangedPhases);
  }
  p.transfers.clear();
  for (size_t i = 0; i < p.phases.size(); ++i) {
    auto &phase = p.phases[i];
    phase.dependencies = i ? std::vector<size_t>{i - 1} : std::vector<size_t>{};
    phase.computationEvent = -1;
    for (auto &r : phase.regions) {
      bool retained = localRetained(p, i, r.storage);
      const auto *scope = genericResidencyScope(p, i);
      bool boundary =
          scope && llvm::is_contained(scope->boundaryExports, r.storage);
      int64_t n = storageBytes(a.storages[r.storage]);
      for (auto d : a.storages[r.storage].shape)
        n /= d;
      for (auto d : r.sizes)
        n *= d;
      bool import =
          r.read && (!retained || a.storages[r.storage].role == "input");
      if (import && scope) {
        size_t begin = scope->beginPhase + (i - scope->beginPhase) /
                                               scope->tasks.size() *
                                               scope->tasks.size();
        for (size_t prior = begin; prior < i; ++prior)
          for (const auto &old : p.phases[prior].regions)
            if (old.storage == r.storage && old.read)
              import = false;
      }
      if (import)
        p.transfers.push_back(
            {r.storage, "host_to_l1", n, i, r.offsets, r.sizes});
      if (r.write && (!retained || boundary))
        p.transfers.push_back(
            {r.storage, "l1_to_host", n, i, r.offsets, r.sizes});
    }
  }
  for (auto &scope : p.residencyScopes) {
    scope.inputReleasePolicy = "group-tail";
    // Shared copies stay live through their final actual reader.
    std::map<size_t, size_t> inputReaders;
    for (size_t phase = scope.beginPhase;
         phase < scope.beginPhase + scope.tasks.size(); ++phase)
      for (const auto &r : p.phases[phase].regions)
        if (r.read &&
            (!llvm::is_contained(scope.retainedStorages, r.storage) ||
             a.storages[r.storage].role == "input") &&
            ++inputReaders[r.storage] > 1)
          scope.inputReleasePolicy = "last-reader";
    if (p.targetCapabilities.acquireAllInputsAtGroupEntry)
      for (const auto &transfer : p.transfers)
        if (transfer.direction == "host_to_l1" &&
            scope.beginPhase <= transfer.phase &&
            transfer.phase < scope.endPhase &&
            (transfer.phase - scope.beginPhase) % scope.tasks.size() != 0) {
          scope.inputReleasePolicy = "last-reader";
          break;
        }
  }
  inferHardwareStages(p);
  if (p.choices.grouping == "tile-major") {
    p.hardwareStages.clear();
    size_t begin = 0;
    while (begin < p.phases.size()) {
      const auto *scope = genericResidencyScope(p, begin);
      size_t width = scope ? scope->tasks.size() : 1;
      size_t limit = scope ? scope->endPhase : begin + 1;
      if (!scope) {
        while (limit < p.phases.size() && !genericResidencyScope(p, limit) &&
               p.phases[limit].task == p.phases[begin].task)
          ++limit;
      }
      auto signature = [&](size_t i) {
        std::vector<int64_t> v;
        for (size_t t = 0; t < width; ++t)
          for (auto b : p.phases[i + t].loops)
            v.push_back(b.upper - b.lower);
        return v;
      };
      size_t end = begin + width;
      auto sig = signature(begin);
      while (end < limit && signature(end) == sig)
        end += width;
      p.hardwareStages.push_back({begin, end});
      begin = end;
    }
  }
  if (p.hardwareStages.size() > maxHardwareStageExpansion)
    return fail(
        error,
        "grouped hardware-stage expansion exceeds checked codegen bound");
  std::vector<bool> required;
  if (!deriveChannelAllocations(a, p, required, p.channelAllocations, error))
    return false;
  p.moduleWideChannelConversion = false;
  p.executionGroups.clear();
  for (size_t i = 0; i < p.hardwareStages.size(); ++i) {
    if (required[i])
      return fail(error,
                  "instance grouping has unsupported DMA template conflict");
    auto &stage = p.hardwareStages[i];
    stage.groupIndex = i;
    stage.legalizeChannels = false;
    ProgramExecutionGroup group;
    group.beginPhase = stage.beginPhase;
    group.endPhase = stage.endPhase;
    group.resource = p.memoryBinding.computeResource;
    group.memory = p.memoryBinding.localMemory;
    group.spatialOwner = computeOwner(p);
    p.executionGroups.push_back(group);
  }
  bindGroupDependencies(a, p);
  p.residentBytes = stagePoolBytes(a, p);
  if (p.residentBytes > p.capacityBytes)
    return fail(error, "grouped tile working set exceeds target capacity");
  return true;
}
bool checkArrangedDomains(const GenericAnalysis &a, const GenericMappingPlan &p,
                          std::string &error) {
  auto expected = p;
  expected.choices.grouping = "task-shape-stages";
  expected.choices.dataOrganization = "phase-local";
  expected.choices.taskGroup.clear();
  expected.choices.retainedStorages.clear();
  expected.choices.residencyGroups.clear();
  expected.residencyScopes.clear();
  if (!buildTiled(a, expected, error))
    return false;
  expected.choices = p.choices;
  if (!arrangePhases(a, expected, error))
    return false;
  if (expected.residencyScopes.size() != p.residencyScopes.size())
    return fail(error, "residency scope coverage mismatch");
  for (size_t i = 0; i < p.residencyScopes.size(); ++i) {
    const auto &x = p.residencyScopes[i];
    const auto &y = expected.residencyScopes[i];
    if (std::tie(x.beginPhase, x.endPhase, x.tasks, x.retainedStorages,
                 x.boundaryExports, x.inputReleasePolicy) !=
        std::tie(y.beginPhase, y.endPhase, y.tasks, y.retainedStorages,
                 y.boundaryExports, y.inputReleasePolicy))
      return fail(error, "residency scope boundary/export mismatch");
  }
  if (expected.phases.size() != p.phases.size() ||
      expected.transfers.size() != p.transfers.size() ||
      expected.hardwareStages.size() != p.hardwareStages.size() ||
      expected.residentBytes != p.residentBytes || !p.retainedRegions.empty() ||
      !p.readOnlyRetentions.empty() || !p.fusedInitializerTasks.empty())
    return fail(error, "grouped execution coverage/capacity mismatch");
  for (size_t i = 0; i < p.phases.size(); ++i) {
    auto &x = p.phases[i];
    auto &y = expected.phases[i];
    if (x.task != y.task || x.dependencies != y.dependencies ||
        x.loops.size() != y.loops.size() ||
        x.regions.size() != y.regions.size())
      return fail(error, "grouped phase identity/order mismatch");
    for (size_t j = 0; j < x.loops.size(); ++j)
      if (std::tie(x.loops[j].loop, x.loops[j].lower, x.loops[j].upper) !=
          std::tie(y.loops[j].loop, y.loops[j].lower, y.loops[j].upper))
        return fail(error, "grouped source domain mismatch");
    for (size_t j = 0; j < x.regions.size(); ++j)
      if (!sameRegion(x.regions[j], y.regions[j]) ||
          x.regions[j].read != y.regions[j].read ||
          x.regions[j].write != y.regions[j].write)
        return fail(error, "grouped source access mismatch");
  }
  for (size_t i = 0; i < p.transfers.size(); ++i) {
    auto &x = p.transfers[i];
    auto &y = expected.transfers[i];
    if (std::tie(x.storage, x.direction, x.bytes, x.phase, x.offsets,
                 x.sizes) !=
        std::tie(y.storage, y.direction, y.bytes, y.phase, y.offsets, y.sizes))
      return fail(error,
                  "grouped transfer differs from selected retention/writeback");
  }
  if (p.executionGroups.size() != expected.executionGroups.size() ||
      p.channelAllocations.size() != expected.channelAllocations.size())
    return fail(error, "grouped resource coverage mismatch");
  for (size_t i = 0; i < p.channelAllocations.size(); ++i) {
    auto &x = p.channelAllocations[i];
    auto &y = expected.channelAllocations[i];
    if (std::tie(x.stage, x.storage, x.direction, x.tileChannel, x.logicalFifo,
                 x.transferCount, x.bytes) !=
        std::tie(y.stage, y.storage, y.direction, y.tileChannel, y.logicalFifo,
                 y.transferCount, y.bytes))
      return fail(error, "grouped channel binding mismatch");
  }
  for (size_t i = 0; i < p.hardwareStages.size(); ++i) {
    auto &x = p.hardwareStages[i];
    auto &y = expected.hardwareStages[i];
    auto &g = p.executionGroups[i];
    if (std::tie(x.beginPhase, x.endPhase, x.groupIndex, x.legalizeChannels) !=
            std::tie(y.beginPhase, y.endPhase, y.groupIndex,
                     y.legalizeChannels) ||
        g.beginPhase != x.beginPhase || g.endPhase != x.endPhase ||
        g.resource != p.memoryBinding.computeResource ||
        g.memory != p.memoryBinding.localMemory ||
        g.dependencies != expected.executionGroups[i].dependencies ||
        g.implementationDependencies !=
            expected.executionGroups[i].implementationDependencies)
      return fail(error, "grouped stage/resource boundaries mismatch");
  }
  return true;
}
bool checkTileDomains(const GenericAnalysis &a, const GenericMappingPlan &p,
                      std::string &error) {
  if (arranged(p))
    return checkArrangedDomains(a, p, error);
  struct DomainState {
    bool begun = false, complete = false;
    std::vector<int64_t> chunk, next;
  };
  std::vector<DomainState> domains(a.tasks.size());
  std::vector<std::vector<std::vector<Interval>>> initialized(
      a.storages.size());
  for (size_t storage = 0; storage < a.storages.size(); ++storage)
    if (a.storages[storage].role == "input") {
      std::vector<Interval> box;
      for (auto extent : a.storages[storage].shape)
        box.push_back({0, extent});
      initialized[storage].push_back(box);
    }
  auto finishTask = [&](size_t ti) {
    // A completed source iteration domain initializes exactly the proven
    // rectangular source writes, without accumulating one box per tile.
    for (const auto &access : a.tasks[ti].accesses)
      if (access.write && access.exactRegion)
        initialized[access.storage].push_back(access.region);
  };
  std::map<size_t, std::vector<const GenericTileRetention *>> scopes;
  std::map<size_t, size_t> scopeCursor;
  std::set<size_t> fusedSources;
  for (const auto &scope : p.retainedRegions) {
    if (scope.storage >= a.storages.size() ||
        scope.endPhase > p.phases.size() ||
        scope.beginPhase >= scope.endPhase ||
        (scope.endPhase - scope.beginPhase < 2 && scope.initializerTask < 0) ||
        scope.initializerTask < -1)
      return fail(error, "invalid retained local region scope");
    auto &list = scopes[scope.storage];
    if (!list.empty() && list.back()->endPhase > scope.beginPhase)
      return fail(error,
                  "overlapping or unordered retained local region scopes");
    const auto &first = p.phases[scope.beginPhase];
    if (scope.initializerTask >= 0) {
      if (!initializerPair(a, scope.initializerTask, first.task, scope.storage))
        return fail(error, "local initializer fails actual source "
                           "projection/dependence proof");
      fusedSources.insert(scope.initializerTask);
    }
    auto base = llvm::find_if(first.regions, [&](const auto &region) {
      return region.storage == scope.storage;
    });
    if (base == first.regions.end() || !base->read || !base->write)
      return fail(error, "retained scope requires read/write source storage");
    for (size_t i = scope.beginPhase; i < scope.endPhase; ++i) {
      const auto &phase = p.phases[i];
      auto region = llvm::find_if(phase.regions, [&](const auto &region) {
        return region.storage == scope.storage;
      });
      if (phase.task != first.task || region == phase.regions.end() ||
          !region->read || !region->write || !sameRegion(*base, *region))
        return fail(
            error,
            "retained local scope crosses changed task, offsets or extents");
    }
    if (scope.initializerTask >= 0) {
      auto sameNeighbor = [&](size_t index) {
        const auto &neighbor = p.phases[index];
        auto region = llvm::find_if(neighbor.regions, [&](const auto &r) {
          return r.storage == scope.storage;
        });
        return neighbor.task == first.task &&
               region != neighbor.regions.end() && sameRegion(*base, *region);
      };
      if ((scope.beginPhase > 0 && sameNeighbor(scope.beginPhase - 1)) ||
          (scope.endPhase < p.phases.size() && sameNeighbor(scope.endPhase)))
        return fail(error, "local initializer must cover full invariant "
                           "accumulation scope exactly once");
    }
    list.push_back(&scope);
  }
  std::map<std::pair<size_t, size_t>, size_t> fusedConsumers;
  for (const auto &scope : p.retainedRegions)
    if (scope.initializerTask >= 0) {
      auto key = std::make_pair(p.phases[scope.beginPhase].task, scope.storage);
      auto previous = fusedConsumers.find(key);
      if (previous != fusedConsumers.end() &&
          previous->second != size_t(scope.initializerTask))
        return fail(error,
                    "conflicting source initializers for consumer storage");
      fusedConsumers[key] = scope.initializerTask;
    }
  std::set<size_t> declaredFused(p.fusedInitializerTasks.begin(),
                                 p.fusedInitializerTasks.end());
  if (declaredFused.size() != p.fusedInitializerTasks.size() ||
      declaredFused != fusedSources)
    return fail(error,
                "fused initializer stage coverage disagrees with local scopes");
  size_t previousEnd = 0;
  for (const auto &scope : p.readOnlyRetentions) {
    if (p.choices.residency != "readonly-panel" ||
        scope.storage != size_t(p.choices.storage) ||
        scope.beginPhase < previousEnd || scope.beginPhase >= scope.endPhase ||
        scope.endPhase > p.phases.size() || scope.storage >= a.storages.size())
      return fail(error, "invalid readonly panel lifetime/choice");
    previousEnd = scope.endPhase;
    const auto &first = p.phases[scope.beginPhase];
    if (!panelTask(a, first.task, p.choices))
      return fail(error,
                  "readonly panel lacks source read-only/independence proof");
    auto panelBound = first.loops[p.choices.panelLoop];
    std::vector<int64_t> offsets, sizes;
    for (size_t index = scope.beginPhase; index < scope.endPhase; ++index) {
      const auto &phase = p.phases[index];
      if (phase.task != first.task ||
          phase.loops[p.choices.panelLoop].lower != panelBound.lower ||
          phase.loops[p.choices.panelLoop].upper != panelBound.upper)
        return fail(error, "readonly panel crosses a source panel subtree");
      auto found = llvm::find_if(phase.regions, [&](const auto &r) {
        return r.storage == scope.storage;
      });
      if (found == phase.regions.end() || !found->read || found->write)
        return fail(error,
                    "readonly panel contains missing or mutable source input");
      if (offsets.empty()) {
        offsets = found->offsets;
        sizes = found->sizes;
      } else
        for (size_t d = 0; d < sizes.size(); ++d) {
          int64_t hi = std::max(offsets[d] + sizes[d],
                                found->offsets[d] + found->sizes[d]);
          offsets[d] = std::min(offsets[d], found->offsets[d]);
          sizes[d] = hi - offsets[d];
        }
    }
    if (offsets != scope.offsets || sizes != scope.sizes)
      return fail(error,
                  "readonly panel differs from actual source demand union");
    auto stage = llvm::find_if(p.hardwareStages, [&](const auto &stage) {
      return stage.beginPhase <= scope.beginPhase &&
             scope.endPhase <= stage.endPhase;
    });
    if (stage == p.hardwareStages.end())
      return fail(error,
                  "readonly panel outlives its execution group allocation");
    if (stage->beginPhase != scope.beginPhase ||
        stage->endPhase != scope.endPhase)
      return fail(error, "readonly-panel-scope grouping must bind one complete "
                         "panel per execution group");
    auto samePanel = [&](size_t index) {
      const auto &phase = p.phases[index];
      return phase.task == first.task &&
             phase.loops[p.choices.panelLoop].lower == panelBound.lower &&
             phase.loops[p.choices.panelLoop].upper == panelBound.upper;
    };
    if ((scope.beginPhase > stage->beginPhase &&
         samePanel(scope.beginPhase - 1)) ||
        (scope.endPhase < stage->endPhase && samePanel(scope.endPhase)))
      return fail(error,
                  "readonly panel does not cover its complete chosen subtree");
  }
  size_t taskPosition = 0, transferPosition = 0;
  for (size_t index = 0; index < p.phases.size(); ++index) {
    const auto &phase = p.phases[index];
    if (phase.task >= a.tasks.size())
      return fail(error, "invalid phase task");
    if (taskPosition >= p.orderedTasks.size())
      return fail(error, "unexpected phase task");
    if (p.orderedTasks[taskPosition] != phase.task) {
      size_t previous = p.orderedTasks[taskPosition];
      if (!domains[previous].complete)
        return fail(error, "incomplete producer before next task");
      finishTask(previous);
      ++taskPosition;
      if (taskPosition >= p.orderedTasks.size() ||
          p.orderedTasks[taskPosition] != phase.task)
        return fail(error, "phase task dependency/order violation");
    }
    if (index == 0 ? !phase.dependencies.empty()
                   : phase.dependencies != std::vector<size_t>{index - 1})
      return fail(error,
                  "missing serial completion dependency before local reuse");
    const auto &task = a.tasks[phase.task];
    auto &state = domains[phase.task];
    if (state.complete || phase.loops.size() != task.loops.size())
      return fail(error, "duplicate or incomplete iteration domain");
    if (!state.begun) {
      for (const auto &loop : task.loops)
        state.next.push_back(loop.lower);
      for (const auto &bound : phase.loops)
        state.chunk.push_back(bound.upper - bound.lower);
      state.begun = true;
      for (size_t l = 0; l < task.loops.size(); ++l) {
        const auto &loop = task.loops[l];
        int64_t iterations = (loop.upper - loop.lower - 1) / loop.step + 1;
        int64_t chunkIterations = (state.chunk[l] - 1) / loop.step + 1;
        if (l < p.tileSizes.size() &&
            chunkIterations != std::min(iterations, p.tileSizes[l]))
          return fail(error,
                      "loop tile domain disagrees with explicit selection");
        if (chunkIterations < iterations) {
          for (const auto &access : task.accesses)
            if (!loop.operation->isProperAncestor(access.operation))
              return fail(
                  error,
                  "tile repeats source operations outside restricted loop");
          for (size_t outer = 0; outer < l; ++outer)
            if (state.chunk[outer] > task.loops[outer].step &&
                !disjointSlices(task, outer))
              return fail(
                  error,
                  "unproven iteration dependence across rectangular tiles");
        }
      }
    }
    // Check a Cartesian tile grid using an independent lexical cursor. This
    // proves exact coverage and disjointness in O(phases * loop depth).
    for (size_t l = 0; l < task.loops.size(); ++l) {
      auto b = phase.loops[l];
      const auto &loop = task.loops[l];
      if (b.loop != l || b.lower != state.next[l] || b.lower < loop.lower ||
          b.upper > loop.upper || b.lower >= b.upper ||
          (b.lower - loop.lower) % loop.step ||
          (b.upper != loop.upper && (b.upper - loop.lower) % loop.step) ||
          b.upper != std::min(loop.upper, b.lower + state.chunk[l]))
        return fail(error,
                    "missing, overlapping or invalid source tile loop domain");
    }
    bool carry = true;
    for (size_t l = task.loops.size(); l > 0 && carry; --l) {
      size_t d = p.taskLoopOrders[phase.task][l - 1];
      const auto &loop = task.loops[d];
      if (state.next[d] + state.chunk[d] >= loop.upper)
        state.next[d] = loop.lower;
      else {
        state.next[d] += state.chunk[d];
        carry = false;
      }
    }
    state.complete = carry;
    GenericTilePhase derived;
    derived.task = phase.task;
    derived.loops = phase.loops;
    int64_t derivedBytes;
    if (!phaseRegions(a, derived, derivedBytes, error))
      return false;
    if (derived.regions.size() != phase.regions.size())
      return fail(error, "missing source access storage region");
    for (size_t r = 0; r < derived.regions.size(); ++r) {
      const auto &x = derived.regions[r];
      const auto &y = phase.regions[r];
      if (x.storage != y.storage || x.offsets != y.offsets ||
          x.sizes != y.sizes || x.read != y.read || x.write != y.write)
        return fail(error,
                    "tile region disagrees with actual source affine accesses");
    }
    int64_t live = 0;
    std::set<size_t> storages;
    for (const auto &region : phase.regions) {
      if (region.storage >= a.storages.size() ||
          !storages.insert(region.storage).second)
        return fail(error, "duplicate or invalid local storage region");
      const auto &storage = a.storages[region.storage];
      if (region.offsets.size() != storage.shape.size() ||
          region.sizes.size() != storage.shape.size())
        return fail(error, "tile region rank mismatch");
      auto type = cast<MemRefType>(storage.root.getType());
      int64_t bytes = (type.getElementType().getIntOrFloatBitWidth() + 7) / 8;
      std::vector<Interval> bounds;
      for (size_t d = 0; d < storage.shape.size(); ++d) {
        if (region.offsets[d] < 0 || region.sizes[d] <= 0 ||
            region.sizes[d] > storage.shape[d] - region.offsets[d] ||
            bytes > INT64_MAX / region.sizes[d])
          return fail(error, "invalid tile region extent");
        bytes *= region.sizes[d];
        bounds.push_back(
            {region.offsets[d], region.offsets[d] + region.sizes[d]});
      }
      bool import = region.read, exportRegion = region.write;
      auto &list = scopes[region.storage];
      auto &scopeIndex = scopeCursor[region.storage];
      while (scopeIndex < list.size() && list[scopeIndex]->endPhase <= index)
        ++scopeIndex;
      auto fused = fusedConsumers.find({phase.task, region.storage});
      if (fused != fusedConsumers.end() &&
          (scopeIndex >= list.size() || list[scopeIndex]->beginPhase > index ||
           list[scopeIndex]->initializerTask != int64_t(fused->second)))
        return fail(
            error,
            "fused source initializer missing from consumer storage phase");
      if (scopeIndex < list.size() && list[scopeIndex]->beginPhase <= index) {
        import &= list[scopeIndex]->initializerTask < 0 &&
                  index == list[scopeIndex]->beginPhase;
        exportRegion &= index + 1 == list[scopeIndex]->endPhase;
      }
      if (declaredFused.count(phase.task)) {
        import = false;
        exportRegion = false;
      }
      auto panel = readOnlyScope(p, index, region.storage);
      if (p.choices.residency == "readonly-panel" &&
          region.storage == size_t(p.choices.storage) &&
          panelTask(a, phase.task, p.choices) && !panel)
        return fail(error, "missing readonly panel source demand coverage");
      if (panel)
        import &= index == panel->beginPhase;
      for (const char *direction : {"host_to_l1", "l1_to_host"}) {
        if ((std::string(direction) == "host_to_l1" && !import) ||
            (std::string(direction) == "l1_to_host" && !exportRegion))
          continue;
        if (transferPosition >= p.transfers.size())
          return fail(error, "missing source tile transfer");
        const auto &transfer = p.transfers[transferPosition++];
        auto offsets = panel ? panel->offsets : region.offsets;
        auto sizes = panel ? panel->sizes : region.sizes;
        int64_t transferBytes = bytes;
        if (panel) {
          auto type =
              cast<MemRefType>(a.storages[region.storage].root.getType());
          transferBytes =
              (type.getElementType().getIntOrFloatBitWidth() + 7) / 8;
          for (auto size : sizes)
            transferBytes *= size;
        }
        if (transfer.storage != region.storage ||
            transfer.direction != direction || transfer.phase != index ||
            transfer.bytes != transferBytes || transfer.offsets != offsets ||
            transfer.sizes != sizes)
          return fail(error, "transfer differs from source tile extent, "
                             "direction, phase or byte count");
      }
      if (region.read && !covered(bounds, initialized[region.storage]))
        return fail(error, "tile reload before producing storage region");
      if (live > INT64_MAX - bytes)
        return fail(error, "local residency overflow");
      live += bytes;
    }
    if (live > p.capacityBytes)
      return fail(error, "phase exceeds local capacity");
  }
  if (transferPosition != p.transfers.size())
    return fail(error, "unexpected or duplicate tile transfer");
  for (const auto &state : domains)
    if (!state.begun || !state.complete)
      return fail(error, "missing source iteration coverage");
  size_t phaseCursor = 0;
  std::set<size_t> boundaries;
  if (p.executionGroups.size() != p.hardwareStages.size())
    return fail(error,
                "hardware binding lacks public execution group coverage");
  size_t groupIndex = 0;
  for (const auto &stage : p.hardwareStages) {
    const auto &group = p.executionGroups[groupIndex];
    auto dependencies = semanticGroupDependencies(a, p, groupIndex);
    auto implementationDependencies = groupIndex
                                          ? std::vector<size_t>{groupIndex - 1}
                                          : std::vector<size_t>{};
    if (stage.groupIndex != groupIndex ||
        group.beginPhase != stage.beginPhase ||
        group.endPhase != stage.endPhase ||
        group.resource != p.memoryBinding.computeResource ||
        group.memory != p.memoryBinding.localMemory ||
        group.dependencies != dependencies ||
        group.implementationDependencies != implementationDependencies)
      return fail(error, "execution group schedule/resource binding mismatch");
    ++groupIndex;
    while (phaseCursor < p.phases.size() &&
           declaredFused.count(p.phases[phaseCursor].task))
      ++phaseCursor;
    if (stage.beginPhase != phaseCursor || stage.beginPhase >= stage.endPhase ||
        stage.endPhase > p.phases.size())
      return fail(error,
                  "hardware stages do not partition actual consumer phases");
    const auto &first = p.phases[stage.beginPhase];
    auto signature = hardwareShapeSignature(p, stage.beginPhase);
    for (size_t phase = stage.beginPhase; phase < stage.endPhase; ++phase)
      if (p.phases[phase].task != first.task ||
          hardwareShapeSignature(p, phase) != signature)
        return fail(
            error,
            "hardware stage mixes source tasks or output/loop-rate shapes");
    boundaries.insert(stage.beginPhase);
    boundaries.insert(stage.endPhase);
    phaseCursor = stage.endPhase;
  }
  while (phaseCursor < p.phases.size() &&
         declaredFused.count(p.phases[phaseCursor].task))
    ++phaseCursor;
  if (phaseCursor != p.phases.size())
    return fail(error, "missing hardware stage coverage");
  for (const auto &scope : p.retainedRegions) {
    auto next = boundaries.upper_bound(scope.beginPhase);
    if (next == boundaries.end() || *next < scope.endPhase)
      return fail(error, "hardware stage ends before retained region export");
  }
  if (p.hardwareStages.size() > maxHardwareStageExpansion)
    return fail(error, "hardware-stage expansion exceeds checked host/codegen "
                       "complexity bound");
  for (const auto &stage : p.hardwareStages)
    if (stage.legalizeChannels) {
      bool scopePresent = false;
      for (const auto &scope : p.retainedRegions)
        if (scope.beginPhase < stage.endPhase &&
            scope.endPhase > stage.beginPhase) {
          if (scope.beginPhase != stage.beginPhase ||
              scope.endPhase != stage.endPhase)
            return fail(error, "channel-conflict stage aggregates multiple "
                               "retained-output scopes");
          scopePresent = true;
        }
      if (!scopePresent)
        return fail(error,
                    "channel-conflict stage lacks a retained-output scope");
    }
  std::vector<bool> channelsRequired;
  std::vector<GenericAIRChannelAllocation> channelPlan;
  if (!deriveChannelAllocations(a, p, channelsRequired, channelPlan, error))
    return false;
  for (size_t stage = 0; stage < p.hardwareStages.size(); ++stage)
    if (p.hardwareStages[stage].legalizeChannels != channelsRequired[stage])
      return fail(error, "DMA channel legalization flag disagrees with actual "
                         "source template conflicts");
  if (p.channelAllocations.size() != channelPlan.size())
    return fail(error, "missing or duplicate planned DMA channel allocation");
  for (size_t i = 0; i < channelPlan.size(); ++i) {
    const auto &x = p.channelAllocations[i];
    const auto &y = channelPlan[i];
    if (x.stage != y.stage || x.storage != y.storage ||
        x.direction != y.direction || x.tileChannel != y.tileChannel ||
        x.logicalFifo != y.logicalFifo || x.transferCount != y.transferCount ||
        x.bytes != y.bytes)
      return fail(error, "DMA channel allocation disagrees with source "
                         "transfer groups or tile port capacity");
  }
  int64_t allocated = stagePoolBytes(a, p);
  if (allocated != p.residentBytes || allocated > p.capacityBytes)
    return fail(error, "physical per-stage storage pools exceed declared "
                       "residency or local capacity");
  return true;
}
bool buildTiled(const GenericAnalysis &a, GenericMappingPlan &p,
                std::string &error) {
  p.phases.clear();
  p.retainedRegions.clear();
  p.transfers.clear();
  p.residentBytes = 0;
  auto localInitializers = initializerCandidates(a);
  for (size_t ti : p.orderedTasks) {
    const auto &task = a.tasks[ti];
    std::vector<int64_t> chunks;
    std::vector<bool> eligible;
    for (size_t l = 0; l < task.loops.size(); ++l) {
      const auto &loop = task.loops[l];
      bool safe = true;
      for (const auto &access : task.accesses)
        safe &= loop.operation->isProperAncestor(access.operation);
      eligible.push_back(safe);
      int64_t count = (loop.upper - loop.lower - 1) / loop.step + 1;
      if (l < p.tileSizes.size()) {
        if (!safe && p.tileSizes[l] < count)
          return fail(error, "requested loop tile would repeat source "
                             "initialization; distribute initialization first");
        if (safe)
          count = std::min(count, p.tileSizes[l]);
      }
      chunks.push_back(count);
    }
    auto make = [&](const std::vector<int64_t> &starts) {
      GenericTilePhase phase;
      phase.task = ti;
      for (size_t l = 0; l < task.loops.size(); ++l) {
        auto &loop = task.loops[l];
        phase.loops.push_back(
            {l, starts[l],
             std::min(loop.upper, starts[l] + chunks[l] * loop.step)});
      }
      return phase;
    };
    std::vector<int64_t> first;
    for (auto &loop : task.loops)
      first.push_back(loop.lower);
    while (true) {
      auto phase = make(first);
      int64_t bytes;
      if (!phaseRegions(a, phase, bytes, error))
        return false;
      for (const auto &region : phase.regions)
        if (region.read && region.write &&
            !localInitializers.count({ti, region.storage})) {
          auto type =
              cast<MemRefType>(a.storages[region.storage].root.getType());
          int64_t n = (type.getElementType().getIntOrFloatBitWidth() + 7) / 8;
          for (auto size : region.sizes)
            n *= size;
          bytes += n;
        }
      if (bytes <= p.capacityBytes)
        break;
      if (!p.tileSizes.empty())
        return fail(error, "explicit tile selection exceeds local capacity");
      size_t chosen = chunks.size();
      int64_t best = 1;
      for (size_t l = 0; l < chunks.size(); ++l)
        if (eligible[l] && chunks[l] > best) {
          best = chunks[l];
          chosen = l;
        }
      if (chosen == chunks.size())
        return fail(
            error,
            "no legal source-preserving tile fits the declared local capacity");
      chunks[chosen] = (chunks[chosen] + 1) / 2;
    }
    // Rectangular blocking permutes outer iterations across inner chunks.
    // Permit it only with an actual access projection proof. Otherwise unit
    // outer chunks retain the source's lexical execution order exactly.
    for (size_t inner = 0; inner < chunks.size(); ++inner) {
      auto &loop = task.loops[inner];
      int64_t count = (loop.upper - loop.lower - 1) / loop.step + 1;
      if (chunks[inner] < count)
        for (size_t outer = 0; outer < inner; ++outer)
          if (!disjointSlices(task, outer)) {
            if (!p.tileSizes.empty() && chunks[outer] > 1)
              return fail(error, "explicit tile selection reorders unproven "
                                 "dependent source iterations");
            chunks[outer] = 1;
          }
    }
    std::function<bool(size_t, std::vector<int64_t> &)> emit;
    emit = [&](size_t l, std::vector<int64_t> &starts) {
      if (l < task.loops.size()) {
        size_t sourceLoop = p.taskLoopOrders[ti][l];
        auto &loop = task.loops[sourceLoop];
        for (int64_t lo = loop.lower; lo < loop.upper;
             lo += chunks[sourceLoop] * loop.step) {
          starts[sourceLoop] = lo;
          if (!emit(l + 1, starts))
            return false;
        }
        return true;
      }
      if (p.phases.size() >= 1048576)
        return fail(error, "serial tile expansion exceeds 1048576 phases; "
                           "compact loop lowering required");
      auto phase = make(starts);
      int64_t bytes;
      if (!phaseRegions(a, phase, bytes, error))
        return false;
      if (bytes > p.capacityBytes)
        return fail(error, "tail tile exceeds local capacity");
      p.residentBytes = std::max(p.residentBytes, bytes);
      for (auto &r : phase.regions) {
        auto type = cast<MemRefType>(a.storages[r.storage].root.getType());
        int64_t n = (type.getElementType().getIntOrFloatBitWidth() + 7) / 8;
        for (auto size : r.sizes)
          n *= size;
        if (r.read)
          p.transfers.push_back({r.storage, "host_to_l1", n, p.phases.size(),
                                 r.offsets, r.sizes});
        if (r.write)
          p.transfers.push_back({r.storage, "l1_to_host", n, p.phases.size(),
                                 r.offsets, r.sizes});
      }
      if (!p.phases.empty())
        phase.dependencies.push_back(p.phases.size() - 1);
      p.phases.push_back(std::move(phase));
      return true;
    };
    if (!emit(0, first))
      return false;
  }
  inferRetentions(a, p);
  inferHardwareStages(p);
  if (p.choices.residency == "readonly-panel") {
    std::vector<GenericHardwareStage> groups;
    for (const auto &stage : p.hardwareStages) {
      size_t begin = stage.beginPhase;
      while (begin < stage.endPhase) {
        size_t end = stage.endPhase;
        if (panelTask(a, p.phases[begin].task, p.choices)) {
          const auto &bound = p.phases[begin].loops[p.choices.panelLoop];
          end = begin + 1;
          while (end < stage.endPhase &&
                 p.phases[end].loops[p.choices.panelLoop].lower ==
                     bound.lower &&
                 p.phases[end].loops[p.choices.panelLoop].upper == bound.upper)
            ++end;
        }
        groups.push_back({begin, end});
        begin = end;
      }
    }
    p.hardwareStages = std::move(groups);
  }
  if (!isolateConflictingChannelScopes(a, p, error))
    return false;
  inferReadOnlyRetentions(a, p);
  std::vector<bool> channelsRequired;
  if (!deriveChannelAllocations(a, p, channelsRequired, p.channelAllocations,
                                error))
    return false;
  for (size_t stage = 0; stage < p.hardwareStages.size(); ++stage)
    p.hardwareStages[stage].legalizeChannels = channelsRequired[stage];
  p.moduleWideChannelConversion = llvm::any_of(
      p.hardwareStages, [](const auto &s) { return s.legalizeChannels; });
  p.executionGroups.clear();
  for (size_t index = 0; index < p.hardwareStages.size(); ++index) {
    auto &stage = p.hardwareStages[index];
    stage.groupIndex = index;
    ProgramExecutionGroup group;
    group.resource = p.memoryBinding.computeResource;
    group.memory = p.memoryBinding.localMemory;
    group.spatialOwner = computeOwner(p);
    group.beginPhase = stage.beginPhase;
    group.endPhase = stage.endPhase;

    p.executionGroups.push_back(std::move(group));
  }
  bindGroupDependencies(a, p);
  p.residentBytes = stagePoolBytes(a, p);
  if (arranged(p))
    return arrangePhases(a, p, error);
  if (p.residentBytes > p.capacityBytes)
    return fail(error,
                "physical per-stage storage pools exceed local capacity");
  return true;
}
} // namespace
GenericMappingStatistics orbit::taskflow::genericMappingStatistics() {
  return mappingStatistics;
}
void orbit::taskflow::recordGenericMappingCostProviderCall() {
  ++mappingStatistics.costProviderCalls;
}
std::vector<ProgramResidencyGroup>
orbit::taskflow::genericResidencyGroups(const ProgramMappingChoice &choice) {
  if (!choice.residencyGroups.empty())
    return choice.residencyGroups;
  if (!choice.taskGroup.empty() || !choice.retainedStorages.empty())
    return {{choice.taskGroup, choice.retainedStorages}};
  return {};
}
const GenericResidencyScope *
orbit::taskflow::genericResidencyScope(const GenericMappingPlan &p,
                                       size_t phase) {
  for (const auto &scope : p.residencyScopes)
    if (scope.beginPhase <= phase && phase < scope.endPhase)
      return &scope;
  return nullptr;
}
bool orbit::taskflow::genericLocallyRetained(const GenericMappingPlan &p,
                                             size_t phase, size_t storage) {
  auto *scope = genericResidencyScope(p, phase);
  return scope && llvm::is_contained(scope->retainedStorages, storage);
}
bool orbit::taskflow::deriveGenericPhaseRegions(const GenericAnalysis &a,
                                                GenericTilePhase &phase,
                                                int64_t &bytes,
                                                std::string &error) {
  return phaseRegions(a, phase, bytes, error);
}
void orbit::taskflow::collectGenericAllocations(
    const GenericAnalysis &a, const GenericMappingPlan &p,
    std::vector<MappingAllocation> &allocations) {
  size_t end =
      p.strategy == "serial-tiled" ? p.phases.size() : p.orderedTasks.size();
  for (size_t storage = 0; storage < a.storages.size(); ++storage) {
    bool external = isa<BlockArgument>(a.storages[storage].root);
    if ((p.strategy == "serial-tiled" &&
         llvm::any_of(p.transfers,
                      [&](const auto &t) { return t.storage == storage; })) ||
        external)
      allocations.push_back({storage, p.memoryBinding.hostMemory,
                             "host-" + std::to_string(storage),
                             storageBytes(a.storages[storage]), 0, end});
    if (p.strategy != "serial-tiled" && !orderOnlyInput(a, storage))
      allocations.push_back({storage, p.memoryBinding.localMemory,
                             "storage-" + std::to_string(storage) + ".data",
                             storageBytes(a.storages[storage]), 0, end});
  }
  if (p.strategy == "serial-tiled")
    stagePoolBytes(a, p, nullptr, nullptr, nullptr, &allocations);
  for (auto &allocation : allocations) {
    allocation.addressing = allocation.memory == p.memoryBinding.hostMemory
                                ? "region-addressed"
                                : "reusable-slot";
    allocation.spatialOwner = memoryOwner(p, allocation.memory);
  }
}
std::vector<ProgramMappingChoice>
orbit::taskflow::discoverGenericMappingChoices(const GenericAnalysis &a,
                                               size_t maxResidencyProposals,
                                               size_t maxResidencyGroups) {
  ProgramMappingChoice baseline;
  baseline.residency = "phase-local";
  baseline.grouping = "task-shape-stages";
  baseline.dataOrganization = "phase-local";
  ProgramMappingChoice tiled;
  tiled.residency = "tiled-phase-local";
  tiled.grouping = "task-shape-stages";
  tiled.dataOrganization = "phase-local";
  ProgramMappingChoice whole;
  whole.residency = "whole-program";
  whole.grouping = "whole-program";
  whole.dataOrganization = "retain-across-users";
  std::vector<ProgramMappingChoice> choices{baseline, tiled, whole};
  auto shape = tiled;
  shape.grouping = "shape-major";
  choices.push_back(shape);
  auto tile = tiled;
  tile.grouping = "tile-major";
  tile.dataOrganization = "retain-across-users";
  std::vector<ProgramResidencyGroup> discovered;
  // Source access relations, not task names or fixture indices, propose scopes.
  for (size_t storage = 0;
       storage < a.storages.size() && discovered.size() < maxResidencyProposals;
       ++storage) {
    if (a.storages[storage].role != "temporary" &&
        a.storages[storage].role != "output")
      continue;
    std::vector<size_t> writers, readers;
    for (size_t task = 0; task < a.tasks.size(); ++task) {
      bool write = false, read = false;
      for (const auto &access : a.tasks[task].accesses)
        if (access.storage == storage) {
          write |= access.write;
          read |= access.read;
        }
      if (write)
        writers.push_back(task);
      if (read)
        readers.push_back(task);
    }
    if (writers.size() != 1)
      continue;
    for (auto reader : readers) {
      if (reader <= writers.front() ||
          discovered.size() >= maxResidencyProposals)
        continue;
      ProgramResidencyGroup group;
      for (size_t task = writers.front(); task <= reader; ++task)
        group.tasks.push_back(task);
      for (size_t data = 0; data < a.storages.size(); ++data) {
        if (a.storages[data].role != "temporary" &&
            a.storages[data].role != "output")
          continue;
        size_t writes = 0, internalReads = 0;
        bool internalWriter = false;
        for (size_t task = 0; task < a.tasks.size(); ++task) {
          bool write = false, read = false;
          for (auto &access : a.tasks[task].accesses)
            if (access.storage == data) {
              write |= access.write;
              read |= access.read;
            }
          writes += write;
          internalWriter |= write && llvm::is_contained(group.tasks, task);
          internalReads += read && llvm::is_contained(group.tasks, task);
        }
        if (writes == 1 && internalWriter && internalReads)
          group.retainedStorages.push_back(data);
      }
      if (std::find(discovered.begin(), discovered.end(), group) ==
          discovered.end())
        discovered.push_back(std::move(group));
    }
  }
  ProgramMappingChoice current;
  std::string ignored;
  if (a.function)
    readNativeMappingChoice(a.function->getParentOfType<mlir::ModuleOp>(),
                            current, ignored);
  auto currentGroups = current.grouping == "tile-major"
                           ? genericResidencyGroups(current)
                           : std::vector<ProgramResidencyGroup>{};
  for (const auto &group : discovered) {
    auto choice = tile;
    choice.residencyGroups = {group};
    choices.push_back(choice);
    if (!currentGroups.empty() && currentGroups.size() < maxResidencyGroups) {
      bool disjoint = true;
      for (auto &existing : currentGroups)
        for (auto task : group.tasks)
          disjoint &= !llvm::is_contained(existing.tasks, task);
      if (disjoint) {
        choice.residencyGroups = currentGroups;
        choice.residencyGroups.push_back(group);
        std::sort(choice.residencyGroups.begin(), choice.residencyGroups.end(),
                  [](const auto &x, const auto &y) {
                    return x.tasks.front() < y.tasks.front();
                  });
        choices.push_back(std::move(choice));
      }
    }
  }
  // Preserve the legacy one-whole-source spelling as an input adapter.
  for (size_t task = 0; task < a.tasks.size(); ++task)
    tile.taskGroup.push_back(task);
  for (size_t storage = 0; storage < a.storages.size(); ++storage)
    if (a.storages[storage].role == "temporary")
      tile.retainedStorages.push_back(storage);
  if (tile.taskGroup.size() > 1 && !tile.retainedStorages.empty())
    choices.push_back(tile);
  for (size_t storage = 0; storage < a.storages.size(); ++storage)
    for (int64_t reuse = 0; reuse < 2; ++reuse) {
      ProgramMappingChoice choice{"readonly-panel", int64_t(storage), reuse,
                                  1 - reuse};
      choice.grouping = "readonly-panel-scope";
      choice.dataOrganization = "retain-across-users";
      bool matched = false;
      for (size_t task = 0; task < a.tasks.size(); ++task)
        matched |= panelTask(a, task, choice);
      if (matched)
        choices.push_back(choice);
    }
  return choices;
}
bool orbit::taskflow::checkGenericMapping(const GenericAnalysis &a,
                                          const GenericMappingPlan &p,
                                          std::string &e) {
  ++mappingStatistics.checkerCalls;
  if ((p.strategy != "single-herd-resident" && p.strategy != "serial-tiled") ||
      p.bodyFingerprint != a.bodyFingerprint)
    return fail(e, "generic plan strategy/body identity mismatch");
  if (!checkGenericMemory(p.memoryBinding, p.capacityBytes, e))
    return false;
  if (p.targetCapabilities.capacityBytes != p.capacityBytes ||
      !checkGenericCapabilities(p.memoryBinding, p.targetCapabilities, e))
    return false;
  std::vector<size_t> initializerTasks;
  if (!readNativeInitializerTasks(a, initializerTasks, e))
    return false;
  for (const auto &impl : p.taskImplementations)
    if (!llvm::any_of(
            p.targetCapabilities.implementations, [&](const auto &allowed) {
              return impl.kind == allowed.kind && impl.lanes == allowed.lanes;
            }))
      return fail(e,
                  impl.kind == "independent-lane-f32"
                      ? "implementation unavailable: strict f32 vector "
                        "multiplication is not supported by target capabilities"
                      : "implementation unavailable: requested kind/lanes are "
                        "not supported by target capabilities");
  if (p.moduleWideChannelConversion !=
      llvm::any_of(p.hardwareStages,
                   [](const auto &s) { return s.legalizeChannels; }))
    return fail(e,
                "module-wide channel conversion differs from checked stages");
  if (p.channelMergePolicy != "single-logical-fifo-per-stage-storage-direction")
    return fail(e, "unsupported generic AIR channel merge policy");
  if (p.capacityBytes <= 0)
    return fail(e, "mapping requires a positive declared local capacity");
  if (p.taskLoopOrders.size() != a.tasks.size() ||
      p.taskImplementations.size() != a.tasks.size())
    return fail(e, "missing program task scheduling/implementation coverage");
  std::vector<NativeTaskImplementationChoice> localImplementations;
  if (!readNativeTaskImplementations(
          a.function->getParentOfType<mlir::ModuleOp>(), a.tasks.size(),
          localImplementations, e))
    return false;
  for (const auto &selected : localImplementations) {
    const auto &actual = p.taskImplementations[selected.task];
    if (actual.kind != selected.kind || actual.lanes != selected.lanes)
      return fail(e,
                  "task implementation differs from explicit source selection");
  }
  for (size_t task = 0; task < a.tasks.size(); ++task)
    if (llvm::is_contained(initializerTasks, task) &&
        (p.taskImplementations[task].kind != "scalar" ||
         p.taskImplementations[task].lanes != 1))
      return fail(e, "generated initializer must use scalar implementation");
  ProgramMappingChoice requested;
  if (!readNativeMappingChoice(a.function->getParentOfType<mlir::ModuleOp>(),
                               requested, e))
    return false;
  if (std::tie(requested.residency, requested.grouping,
               requested.dataOrganization, requested.storage,
               requested.reuseLoop, requested.panelLoop, requested.taskGroup,
               requested.retainedStorages, requested.residencyGroups) !=
      std::tie(p.choices.residency, p.choices.grouping,
               p.choices.dataOrganization, p.choices.storage,
               p.choices.reuseLoop, p.choices.panelLoop, p.choices.taskGroup,
               p.choices.retainedStorages, p.choices.residencyGroups))
    return fail(e,
                "program mapping choice differs from canonical source action");
  ProgramDataPolicy policy;
  if (!parseProgramDataPolicy(p.choices.dataOrganization, policy, e))
    return false;
  if ((policy == ProgramDataPolicy::RetainAcrossUsers) !=
      (p.choices.residency == "readonly-panel" ||
       p.choices.residency == "whole-program" ||
       p.choices.grouping == "tile-major"))
    return fail(
        e,
        "public data organization differs from its checked target realization");
  bool panelApplied = false;
  if (p.choices.residency != "phase-local" &&
      p.choices.residency != "readonly-panel" &&
      p.choices.residency != "whole-program" &&
      p.choices.residency != "tiled-phase-local")
    return fail(e, "unsupported program residency choice");
  if (!arranged(p) &&
      p.choices.grouping !=
          (p.choices.residency == "whole-program"    ? "whole-program"
           : p.choices.residency == "readonly-panel" ? "readonly-panel-scope"
                                                     : "task-shape-stages"))
    return fail(e, "unsupported program execution grouping choice");
  if (p.choices.residency != "readonly-panel" &&
      (p.choices.storage != -1 || p.choices.reuseLoop != -1 ||
       p.choices.panelLoop != -1))
    return fail(e, "non-panel choice carries an undeclared panel strategy");
  if (p.choices.residency == "whole-program" &&
      p.strategy != "single-herd-resident")
    return fail(e, "whole-program choice requires resident realization");
  if (p.choices.residency == "tiled-phase-local" &&
      p.strategy != "serial-tiled")
    return fail(e, "tiled-phase-local choice requires tiled realization");
  for (size_t task = 0; task < a.tasks.size(); ++task) {
    std::vector<size_t> expected;
    bool panel = p.choices.residency == "readonly-panel" &&
                 panelTask(a, task, p.choices);
    panelApplied |= panel;
    if (panel) {
      expected.push_back(p.choices.panelLoop);
      expected.push_back(p.choices.reuseLoop);
    }
    for (size_t loop = 0; loop < a.tasks[task].loops.size(); ++loop)
      if (!llvm::is_contained(expected, loop))
        expected.push_back(loop);
    if (p.taskLoopOrders[task] != expected)
      return fail(
          e, "program loop schedule lacks actual source independence proof");
    const auto &implementation = p.taskImplementations[task];
    if (implementation.task != task)
      return fail(e, "missing or duplicated source task implementation");
    if (implementation.kind == "scalar") {
      if (implementation.parallelLoop != -1 || implementation.lanes != 1)
        return fail(e, "scalar implementation has vector resource choices");
    } else if (implementation.kind == programMatrixImplementation) {
      ProgramMatrixTaskContract matrix;
      if (implementation.parallelLoop != -1 || implementation.lanes != 1 ||
          !analyzeProgramMatrixTask(a, task, matrix, e)) {
        return fail(e, "matrix implementation fails source contract proof");
      }
      for (const auto &phase : p.phases) {
        if (phase.task != task) {
          continue;
        }
        ProgramGraph graph;
        graph.source = a;
        MappingTaskInstance instance;
        instance.task = instance.domainTask = task;
        instance.implementation = programMatrixImplementation;
        for (const auto &bound : phase.loops) {
          instance.domain.push_back({bound.loop, bound.lower, bound.upper,
                                     a.tasks[task].loops[bound.loop].step});
        }
        ProgramMatrixInstanceContract checked;
        if (!analyzeProgramMatrixInstance(graph, instance, checked, e)) {
          return false;
        }
      }
    } else if ((implementation.kind != "independent-lane-f32" &&
                implementation.kind != "scalar-product-vector-add-f32") ||
               implementation.parallelLoop != vectorLoop(a, a.tasks[task]) ||
               implementation.parallelLoop < 0 || implementation.lanes < 2 ||
               implementation.lanes > 16 ||
               (implementation.lanes & (implementation.lanes - 1)))
      return fail(e,
                  "vector implementation fails independent-lane source proof");
    if (implementation.kind == "scalar-product-vector-add-f32") {
      if ((implementation.lanes != 4 && implementation.lanes != 8) ||
          a.tasks[task].loops[implementation.parallelLoop].step != 1)
        return fail(e, "scalar-product-vector-add-f32 requires unit-step "
                       "spatial loop and lanes 4 or 8");
      for (const auto &phase : p.phases)
        if (phase.task == task) {
          auto bound = llvm::find_if(phase.loops, [&](const auto &b) {
            return b.loop == size_t(implementation.parallelLoop);
          });
          if (bound == phase.loops.end() ||
              bound->lower <
                  a.tasks[task].loops[implementation.parallelLoop].lower ||
              bound->upper >
                  a.tasks[task].loops[implementation.parallelLoop].upper ||
              bound->upper <= bound->lower ||
              (bound->upper - bound->lower) % implementation.lanes)
            return fail(e, "scalar-product-vector-add-f32 requires every "
                           "spatial tile extent divisible by lanes");
        }
    }
  }
  if (auto requested = a.function->getParentOfType<mlir::ModuleOp>()
                           ->getAttrOfType<StringAttr>("orbit.implementation"))
    if (localImplementations.empty() &&
        requested.getValue() == "scalar-product-vector-add-f32" &&
        !llvm::any_of(p.taskImplementations, [](const auto &impl) {
          return impl.kind == "scalar-product-vector-add-f32";
        }))
      return fail(e, "scalar-product-vector-add-f32 request has no "
                     "vectorizable source task");
  if (p.choices.residency == "readonly-panel" && !panelApplied)
    return fail(e, "readonly panel choice matches no proven source task");
  if (p.strategy == "single-herd-resident") {
    if (p.executionGroups.size() != 1 || p.executionGroups[0].beginPhase != 0 ||
        p.executionGroups[0].endPhase != p.orderedTasks.size() ||
        p.executionGroups[0].resource != p.memoryBinding.computeResource ||
        p.executionGroups[0].memory != p.memoryBinding.localMemory ||
        !p.executionGroups[0].dependencies.empty() ||
        !p.executionGroups[0].implementationDependencies.empty() ||
        p.executionGroups[0].spatialOwner != computeOwner(p))
      return fail(
          e,
          "resident realization requires one explicitly bound program group");
    int64_t bytes = 0;
    for (size_t storage = 0; storage < a.storages.size(); ++storage) {
      if (orderOnlyInput(a, storage))
        continue;
      int64_t n = storageBytes(a.storages[storage]);
      if (n < 0 || bytes > INT64_MAX - n)
        return fail(e, "unsupported storage extent/type or byte overflow");
      bytes += n;
    }
    if (bytes != p.residentBytes)
      return fail(e, "declared residency disagrees with actual storage bytes");
    if (bytes > p.capacityBytes)
      return fail(e, "single-herd resident capacity exceeded: " +
                         std::to_string(bytes) + " > " +
                         std::to_string(p.capacityBytes));
  }
  std::set<size_t> done;
  for (size_t t : p.orderedTasks) {
    if (t >= a.tasks.size() || done.count(t))
      return fail(e, "duplicate/invalid task coverage");
    for (size_t d : a.tasks[t].dependencies)
      if (!done.count(d))
        return fail(e, "task scheduled before dependency");
    done.insert(t);
  }
  if (done.size() != a.tasks.size())
    return fail(e, "missing task coverage");
  if (p.strategy == "serial-tiled") {
    for (auto size : p.tileSizes)
      if (size <= 0)
        return fail(e, "tile sizes must be positive");
    for (const auto &transfer : p.transfers)
      if (transfer.storage >= a.storages.size())
        return fail(e, "invalid tiled transfer storage");
    for (const auto &phase : p.phases)
      if (phase.task >= a.tasks.size())
        return fail(e, "invalid tiled phase task");
    if (!checkTileDomains(a, p, e))
      return false;
    for (const auto &stage : p.hardwareStages)
      if (stage.endPhase - stage.beginPhase > 1) {
        std::set<size_t> inputs, outputs;
        auto transfer = std::lower_bound(
            p.transfers.begin(), p.transfers.end(), stage.beginPhase,
            [](const auto &t, size_t phase) { return t.phase < phase; });
        for (;
             transfer != p.transfers.end() && transfer->phase < stage.endPhase;
             ++transfer) {
          if (transfer->direction == "host_to_l1")
            inputs.insert(transfer->storage);
          else if (transfer->direction == "l1_to_host")
            outputs.insert(transfer->storage);
        }
        if (int64_t(inputs.size()) > p.targetCapabilities.maxGroupInputs ||
            int64_t(outputs.size()) > p.targetCapabilities.maxGroupOutputs)
          return fail(e, "group realization exceeds target-declared "
                         "input/output stream limits");
      }
  } else {
    if (!p.phases.empty() || !p.tileSizes.empty() ||
        !p.retainedRegions.empty() || !p.fusedInitializerTasks.empty() ||
        !p.hardwareStages.empty() || !p.channelAllocations.empty())
      return fail(e, "resident strategy carries unexpected tile phases");
    std::set<std::pair<size_t, std::string>> transfers;
    for (auto &t : p.transfers) {
      if (t.storage >= a.storages.size() ||
          t.bytes != storageBytes(a.storages[t.storage]) ||
          !transfers.emplace(t.storage, t.direction).second)
        return fail(e, "invalid/duplicate transfer or byte extent");
      auto &s = a.storages[t.storage];
      if (!((s.role == "input" && t.direction == "host_to_l1") ||
            (s.role == "output" && t.direction == "l1_to_host")))
        return fail(e, "transfer boundary/role mismatch");
    }
    for (size_t i = 0; i < a.storages.size(); ++i) {
      auto &s = a.storages[i];
      if (s.role == "input" && !orderOnlyInput(a, i) &&
          !transfers.count({i, "host_to_l1"}))
        return fail(e, "missing input transfer");
      if (s.role == "output" && !transfers.count({i, "l1_to_host"}))
        return fail(e, "missing output transfer");
    }
  }
  if (!checkProgramCommunication(a, p, e))
    return false;
  const std::string theoryProvenance =
      "theoretical direct-AIR source scalar-operation and explicit-transfer "
      "sensitivity; not calibrated; backend local copies and internal "
      "scheduling excluded";
  if (p.directCost.enabled) {
    GenericCostFeatures features;
    GenericDirectAIRCost expected;
    const auto *model = p.directCost.model.getAsObject();
    if (model && (model->getString("kind") == "event-mechanism" ||
                  model->getString("kind") == "region-invocation")) {
      if (!evaluateGenericMechanismCost(a, p, p.directCost.model,
                                        p.directCost.target, expected, e))
        return false;
    } else if (!collectGenericCostFeatures(a, p, features, e) ||
               !evaluateGenericDirectAIRCost(features, p.directCost.model,
                                             p.directCost.target, expected, e))
      return false;
    if (staticFingerprint(directCostReport(expected)) !=
        staticFingerprint(directCostReport(p.directCost)))
      return fail(e,
                  "direct-AIR cost features/model identity/estimate mismatch");
    if (p.hasCost != expected.valueMs.has_value() ||
        p.estimatedMs != expected.valueMs.value_or(0) ||
        p.theoreticalLaunchMs != 0 || p.theoreticalPerOperationMs != 0 ||
        p.theoreticalPerByteMs != 0 ||
        p.costProvenance != directCostProvenance(p.directCost))
      return fail(e, "direct-AIR score must not mix theoretical or separate "
                     "transfer charges");
  } else if (!p.hasCost) {
    if (p.estimatedMs != 0 || p.theoreticalLaunchMs != 0 ||
        p.theoreticalPerOperationMs != 0 || p.theoreticalPerByteMs != 0 ||
        p.costProvenance != "unknown: direct AIR path not calibrated")
      return fail(e, "unscored plan must not carry a fabricated cost or "
                     "theoretical coefficients");
  } else {
    if (!std::isfinite(p.estimatedMs) || p.estimatedMs <= 0 ||
        p.costProvenance != theoryProvenance)
      return fail(e, "invalid direct-path theoretical cost/provenance");
    for (double coefficient :
         {p.theoreticalLaunchMs, p.theoreticalPerOperationMs,
          p.theoreticalPerByteMs})
      if (!std::isfinite(coefficient) || coefficient < 0)
        return fail(e, "invalid theoretical coefficient");
    // Independently evaluate the declared model from actual operations and
    // the transfer plan. This proves internal consistency, never calibration.
    double scalarOperations = 0;
    for (const auto &task : a.tasks)
      task.operation->walk([&](mlir::Operation *operation) {
        auto dialect = operation->getName().getDialectNamespace();
        if ((dialect != "arith" && dialect != "math") ||
            operation->getName().getStringRef() == "arith.constant")
          return;
        double instances = 1;
        for (const auto &loop : task.loops)
          if (loop.operation->isProperAncestor(operation))
            instances *= double((loop.upper - loop.lower - 1) / loop.step + 1);
        scalarOperations += instances;
      });
    double transferred = 0;
    for (const auto &transfer : p.transfers)
      transferred += transfer.bytes;
    double expected = p.theoreticalLaunchMs +
                      p.theoreticalPerOperationMs * scalarOperations +
                      p.theoreticalPerByteMs * transferred;
    if (!std::isfinite(expected) ||
        std::abs(expected - p.estimatedMs) >
            1e-12 * std::max(1.0, std::abs(expected)))
      return fail(e, "theoretical cost disagrees with actual "
                     "operation/transfer accounting");
  }
  return true;
}
namespace {
std::string costTypeIdentity(Type type) {
  std::string text;
  llvm::raw_string_ostream os(text);
  if (auto memory = dyn_cast<MemRefType>(type)) {
    os << "memref-rank-" << memory.getRank() << ":" << memory.getElementType()
       << ":";
    if (memory.getLayout().isIdentity())
      os << "identity";
    else
      memory.getLayout().print(os);
    os << ":";
    if (memory.getMemorySpace())
      memory.getMemorySpace().print(os);
    else
      os << "default";
  } else
    type.print(os);
  return text;
}
std::string costRegionFamily(const GenericAnalysis &a) {
  llvm::json::Array storageFacts, tasks;
  for (const auto &storage : a.storages)
    storageFacts.push_back(
        llvm::json::Object{{"role", storage.role},
                           {"type", costTypeIdentity(storage.root.getType())}});
  std::string alias;
  llvm::raw_string_ostream aliasStream(alias);
  if (auto attr = func::FuncOp(a.function)->getAttr("orbit.noalias"))
    attr.print(aliasStream);
  for (const auto &task : a.tasks) {
    std::map<Operation *, size_t> ids;
    task.operation->walk([&](Operation *op) { ids.emplace(op, ids.size()); });
    auto valueIdentity = [&](Value value) {
      for (size_t i = 0; i < task.loops.size(); ++i)
        if (value == task.loops[i].induction)
          return "iv:" + std::to_string(i);
      for (size_t i = 0; i < a.storages.size(); ++i)
        if (value == a.storages[i].root)
          return "storage:" + std::to_string(i) + ":" +
                 costTypeIdentity(value.getType());
      if (auto result = dyn_cast<OpResult>(value)) {
        auto found = ids.find(result.getOwner());
        if (found != ids.end())
          return "op:" + std::to_string(found->second) + ":" +
                 std::to_string(result.getResultNumber());
        for (size_t i = 0; i < a.tasks.size(); ++i)
          if (result.getOwner() == a.tasks[i].operation)
            return "task:" + std::to_string(i) + ":" +
                   std::to_string(result.getResultNumber());
        std::string external;
        llvm::raw_string_ostream os(external);
        os << result.getOwner()->getName() << ":"
           << result.getOwner()->getAttrDictionary();
        if (auto properties = result.getOwner()->getPropertiesAsAttribute())
          properties.print(os);
        return external;
      }
      if (auto argument = dyn_cast<BlockArgument>(value))
        return "arg:" + std::to_string(argument.getArgNumber()) + ":" +
               costTypeIdentity(value.getType());
      return std::string("unknown-value");
    };
    llvm::json::Array operations, dependencies, arguments;
    for (auto dependency : task.dependencies)
      dependencies.push_back(int64_t(dependency));
    for (auto storage : task.argumentStorages)
      arguments.push_back(int64_t(storage));
    task.operation->walk([&](Operation *op) {
      if (op == task.operation)
        return;
      llvm::json::Array operands, types;
      for (Value value : op->getOperands())
        operands.push_back(valueIdentity(value));
      for (Type type : op->getResultTypes())
        types.push_back(costTypeIdentity(type));
      llvm::json::Object fact{{"operation", op->getName().getStringRef()},
                              {"parent", int64_t(ids.at(op->getParentOp()))},
                              {"operands", std::move(operands)},
                              {"types", std::move(types)}};
      if (auto loop = dyn_cast<affine::AffineForOp>(op)) {
        fact["lower_origin"] = loop.getConstantLowerBound();
        fact["step"] = loop.getStepAsInt();
        std::string extra;
        llvm::raw_string_ostream os(extra);
        for (auto attr : op->getAttrs())
          if (attr.getName() != "lowerBoundMap" &&
              attr.getName() != "upperBoundMap" && attr.getName() != "step")
            os << attr.getName() << "=" << attr.getValue() << ";";
        fact["extra_attributes"] = extra;
        // Only the static extent is erased. Parent/source IV and all access
        // expressions keep nesting and reduction axes bound to the family.
      } else {
        std::string attributes;
        llvm::raw_string_ostream os(attributes);
        op->getAttrDictionary().print(os);
        if (auto properties = op->getPropertiesAsAttribute())
          properties.print(os);
        fact["attributes"] = attributes;
      }
      operations.push_back(std::move(fact));
    });
    tasks.push_back(
        llvm::json::Object{{"dependencies", std::move(dependencies)},
                           {"argument_storages", std::move(arguments)},
                           {"operations", std::move(operations)}});
  }
  return staticFingerprint(
      llvm::json::Object{{"schema", "orbit.direct-air.region-family.v1"},
                         {"noalias", alias},
                         {"storages", std::move(storageFacts)},
                         {"tasks", std::move(tasks)}});
}
} // namespace
llvm::json::Value
orbit::taskflow::serializeGenericCostFeatures(const GenericCostFeatures &f) {
  llvm::json::Object values, operations;
  llvm::json::Array sources, implementations;
  for (const auto &v : f.values)
    values[v.first] = v.second;
  for (const auto &v : f.operations)
    operations[v.first] = v.second;
  for (const auto &v : f.sourceFingerprints)
    sources.push_back(v);
  for (const auto &v : f.implementations)
    implementations.push_back(v);
  return llvm::json::Object{
      {"schema", "orbit.direct-air.features.v1"},
      {"execution_fingerprint", f.executionFingerprint},
      {"region_family_fingerprint", f.regionFamilyFingerprint},
      {"architecture_fingerprint", f.architectureFingerprint},
      {"values", std::move(values)},
      {"source_operations", std::move(operations)},
      {"source_fingerprints", std::move(sources)},
      {"implementations", std::move(implementations)},
      {"scope", "native source/checked plan facts; counts are not instruction "
                "counts or measured physical traffic"}};
}
bool orbit::taskflow::collectGenericCostFeatures(const GenericAnalysis &source,
                                                 const GenericMappingPlan &p,
                                                 GenericCostFeatures &f,
                                                 std::string &error) {
  if (!checkProgramGraphSource(source, p.communication, error) ||
      !checkProgramGraph(p.communication, error))
    return false;
  const auto &a = p.communication.source;
  f = {};
  f.sourceFingerprints = {a.bodyFingerprint};
  std::vector<NativeTaskImplementationChoice> explicitImplementations;
  if (!readNativeTaskImplementations(
          a.function->getParentOfType<mlir::ModuleOp>(), a.tasks.size(),
          explicitImplementations, error))
    return false;
  if (!explicitImplementations.empty())
    f.values["explicit_task_implementation_choices"] =
        explicitImplementations.size();

  f.regionFamilyFingerprint = costRegionFamily(a);
  f.architectureFingerprint =
      staticFingerprint(serializeGenericMemory(p.memoryBinding));
  auto add = [&](int64_t &value, int64_t amount) {
    if (amount < 0 || value > INT64_MAX - amount)
      return false;
    value += amount;
    return true;
  };
  auto instances = [&](const GenericTask &task, Operation *op, int64_t &n) {
    n = 1;
    for (const auto &loop : task.loops)
      if (loop.operation->isProperAncestor(op)) {
        int64_t iterations = (loop.upper - loop.lower - 1) / loop.step + 1;
        if (iterations <= 0 || n > INT64_MAX / iterations)
          return false;
        n *= iterations;
      }
    return true;
  };
  bool valid = true;
  for (size_t taskIndex = 0; taskIndex < a.tasks.size(); ++taskIndex) {
    const auto &task = a.tasks[taskIndex];
    const auto implementation =
        llvm::find_if(p.communication.instances,
                      [&](const auto &i) { return i.task == taskIndex; });
    const bool mixed =
        implementation != p.communication.instances.end() &&
        implementation->implementation == "scalar-product-vector-add-f32" &&
        implementation->lanes == 8;
    task.operation->walk([&](Operation *op) {
      auto dialect = op->getName().getDialectNamespace();
      if ((dialect != "arith" && dialect != "math") ||
          op->getName().getStringRef() == "arith.constant")
        return;
      int64_t n;
      if (!instances(task, op, n)) {
        valid = false;
        return;
      }
      bool f32 =
          llvm::any_of(op->getResultTypes(),
                       [](Type t) { return t.isF32(); }) ||
          llvm::any_of(op->getOperandTypes(), [](Type t) { return t.isF32(); });
      std::string name = op->getName().getStringRef().str();
      valid &= add(f.operations[name + (f32 ? ":f32" : ":other")], n);
      if (f32)
        valid &= add(f.values["source_scalar_f32_operations"], n);
      bool mixedOperation =
          mixed && implementation->parallelLoop >= 0 &&
          size_t(implementation->parallelLoop) < task.loops.size() &&
          task.loops[implementation->parallelLoop].operation->isProperAncestor(
              op);
      if (f32 && (name == "arith.mulf" || name == "arith.divf")) {
        valid &= add(f.values["source_f32_product_operations"], n);
        valid &=
            add(f.values[mixedOperation
                             ? "implemented_mixed8_f32_product_operations"
                             : "implemented_scalar_f32_product_operations"],
                n);
      }
      if (f32 && (name == "arith.addf" || name == "arith.subf")) {
        if (mixedOperation) {
          if (n % 8)
            valid = false;
          else
            valid &=
                add(f.values["implemented_mixed8_f32_addsub_packs"], n / 8);
        } else
          valid &= add(f.values["implemented_scalar_f32_addsub_operations"], n);
      }
      if (f32 && name == "math.exp")
        valid &= add(f.values["source_math_exp_f32_operations"], n);
    });
    for (const auto &access : task.accesses) {
      int64_t n;
      if (!instances(task, access.operation, n)) {
        valid = false;
        continue;
      }
      if (access.read)
        valid &= add(f.values["source_scalar_reads"], n);
      if (access.write)
        valid &= add(f.values["source_scalar_writes"], n);
    }
  }
  for (const auto *key :
       {"source_scalar_f32_operations", "source_f32_product_operations",
        "source_math_exp_f32_operations", "source_scalar_reads",
        "source_scalar_writes", "dma_bytes", "dma_input_bytes",
        "dma_output_bytes", "dma_transfer_count", "active_compute_events",
        "local_initialize_events", "local_staging_copy_bytes",
        "implemented_scalar_f32_product_operations",
        "implemented_mixed8_f32_product_operations",
        "implemented_scalar_f32_addsub_operations",
        "implemented_mixed8_f32_addsub_packs"})
    f.values[key] += 0;
  f.values["module_wide_channel_conversion"] =
      p.moduleWideChannelConversion ? 1 : 0;
  f.values["command_count"] = 0;
  f.values["execution_region_count"] = p.communication.executionRegions.size();
  for (const auto &region : p.communication.executionRegions)
    if (region.submissionCount <= 0 ||
        !add(f.values["command_count"], region.submissionCount))
      return fail(
          error,
          "cost execution region submission count is invalid or overflows");
  f.values["hardware_stage_count"] =
      p.strategy == "serial-tiled" ? p.hardwareStages.size() : 1;
  f.values["execution_group_count"] = p.executionGroups.size();
  f.values["source_phase_count"] = p.phases.size();
  f.values["resident_payload_bytes"] = p.residentBytes;
  f.values["readonly_retention_scopes"] = p.readOnlyRetentions.size();
  f.values["serial_tiled"] = p.strategy == "serial-tiled";
  f.values["capacity_bytes"] = p.capacityBytes;
  f.values["all_source_loops_affine"] = 1;
  f.values["all_source_loops_unit_step"] = 1;
  f.values["source_loop_depth_max"] = 0;
  f.values["source_has_tile_tail"] = 0;
  f.values["mixed8_lane_coverage_exact"] = 1;
  for (size_t axis = 0; axis < 3; ++axis)
    f.values["source_max_loop_" + std::to_string(axis) + "_iterations"] = 0;
  for (const auto &task : a.tasks)
    for (size_t axis = 0; axis < task.loops.size(); ++axis) {
      const auto &loop = task.loops[axis];
      f.values["all_source_loops_affine"] &=
          isa<affine::AffineForOp>(loop.operation);
      f.values["all_source_loops_unit_step"] &= loop.step == 1;
      f.values["source_loop_depth_max"] =
          std::max(f.values["source_loop_depth_max"], int64_t(axis + 1));
      int64_t count = (loop.upper - loop.lower - 1) / loop.step + 1;
      if (axis < 3)
        f.values["source_max_loop_" + std::to_string(axis) + "_iterations"] =
            std::max(f.values["source_max_loop_" + std::to_string(axis) +
                              "_iterations"],
                     count);
      if (axis < p.tileSizes.size() && count % p.tileSizes[axis])
        f.values["source_has_tile_tail"] = 1;
    }
  for (const auto &instance : p.communication.instances)
    if (instance.implementation == "scalar-product-vector-add-f32") {
      if (instance.lanes != 8)
        f.values["mixed8_lane_coverage_exact"] = 0;
      auto bound = llvm::find_if(instance.domain, [&](const auto &d) {
        return int64_t(d.loop) == instance.parallelLoop;
      });
      if (bound == instance.domain.end() || (bound->upper - bound->lower) % 8)
        f.values["mixed8_lane_coverage_exact"] = 0;
    }
  for (size_t axis = 0; axis < 3; ++axis)
    f.values["max_tile_loop_" + std::to_string(axis) + "_iterations"] = 0;
  for (const auto &instance : p.communication.instances) {
    if (instance.kind != "source-task")
      continue;
    for (const auto &bound : instance.domain)
      if (bound.loop < 3) {
        if (bound.step <= 0 || bound.upper <= bound.lower)
          return fail(error, "cost common instance domain is invalid");
        auto &maximum = f.values["max_tile_loop_" + std::to_string(bound.loop) +
                                 "_iterations"];
        maximum =
            std::max(maximum, (bound.upper - bound.lower - 1) / bound.step + 1);
      }
  }
  f.values["core_count_per_stage"] =
      1; // Current native target realization only.
  f.values["all_storage_f32"] = llvm::all_of(a.storages, [](const auto
                                                                &storage) {
    return cast<MemRefType>(storage.root.getType()).getElementType().isF32();
  });
  std::set<std::string> implementationSet;
  for (const auto &instance : p.communication.instances)
    implementationSet.insert(instance.implementation + ":" +
                             std::to_string(instance.lanes));
  f.implementations.assign(implementationSet.begin(), implementationSet.end());
  for (const auto &event : p.communication.events) {
    if (event.kind == "compute")
      valid &= add(f.values["active_compute_events"], 1);
    if (event.kind == "initialize")
      valid &= add(f.values["local_initialize_events"], 1);
  }
  for (const auto &event : p.communication.events)
    if (event.kind == "transfer") {
      valid &= add(f.values["dma_bytes"], event.transferBytes);
      valid &= add(f.values["dma_transfer_count"], 1);
      const bool input = event.link == p.memoryBinding.inputLink;
      if (!input && event.link != p.memoryBinding.outputLink)
        return fail(error, "AIR cost common transfer link is unsupported");
      valid &= add(f.values[input ? "dma_input_bytes" : "dma_output_bytes"],
                   event.transferBytes);
      if (p.strategy == "serial-tiled" && input &&
          llvm::any_of(event.writes, [&](size_t copy) {
            const auto &allocation =
                p.communication
                    .allocations[p.communication.copies[copy].allocation];
            return llvm::StringRef(allocation.key).ends_with(".incoming");
          }))
        valid &= add(f.values["local_staging_copy_bytes"], event.transferBytes);
    }
  if (!valid)
    return fail(error, "native direct-AIR feature count overflows int64");
  auto unpriced = p;
  unpriced.directCost = {};
  unpriced.hasCost = false;
  unpriced.estimatedMs = 0;
  unpriced.theoreticalLaunchMs = 0;
  unpriced.theoreticalPerOperationMs = 0;
  unpriced.theoreticalPerByteMs = 0;
  unpriced.costProvenance = "unknown: direct AIR path not calibrated";
  unpriced.planId.clear();
  // Serialization cannot recurse: directCost is disabled on this copy.
  // Schedule, copies, implementation, grouping and resource bindings remain in
  // the hash.
  f.executionFingerprint = staticFingerprint(
      canonicalChosenPlan(serializeGenericMapping(a, unpriced)));
  return true;
}
bool orbit::taskflow::evaluateGenericDirectAIRCost(
    const GenericCostFeatures &features, const llvm::json::Value &model,
    const llvm::json::Value &target, GenericDirectAIRCost &cost,
    std::string &error) {
  recordGenericMappingCostProviderCall();
  cost = {};
  cost.enabled = true;
  cost.features = features;
  cost.model = llvm::json::Value(model);
  cost.target = llvm::json::Value(target);
  cost.featureDigest =
      staticFingerprint(serializeGenericCostFeatures(features));
  cost.modelDigest = staticFingerprint(model);
  cost.status = "uncalibrated";
  if (features.sourceFingerprints.empty() || features.implementations.empty() ||
      llvm::any_of(features.values,
                   [](const auto &v) { return v.second < 0; }) ||
      llvm::any_of(features.operations,
                   [](const auto &v) { return v.second < 0; }))
    return fail(error, "direct-AIR features require source/implementation "
                       "identities and nonnegative native counts");
  auto unknown = [&](const std::string &status, const std::string &reason) {
    cost.status = status;
    cost.reason = reason;
    return true;
  };
  const auto *object = model.getAsObject();
  if (!object || object->getString("schema") != "orbit.direct-air.cost.v1")
    return fail(
        error,
        "direct-AIR cost model requires schema orbit.direct-air.cost.v1");
  auto digest = [](llvm::StringRef value) {
    return value.size() == 64 && llvm::all_of(value, [](char c) {
             return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
           });
  };
  if (!digest(features.executionFingerprint))
    return fail(error,
                "native execution fingerprint requires lowercase SHA256");
  auto kind = object->getString("kind");
  if (kind == "contrast-composite")
    return unknown("uncalibrated", "paired implementation differences only; "
                                   "absolute invocation cost is unknown");
  if (kind == "observed-table") {
    for (auto key : {"model_id", "model_version", "boundary"})
      if (!object->getString(key) || object->getString(key)->empty())
        return fail(error, "observed-table identity/boundary missing");
    const auto *modelTarget = object->getObject("target");
    const auto *rows = object->getArray("rows");
    if (!modelTarget || !rows)
      return fail(error, "observed-table requires target and rows");
    for (auto key : {"id", "toolchain", "runtime"})
      if (!modelTarget->getString(key) || modelTarget->getString(key)->empty())
        return fail(error, "observed-table target/toolchain/runtime missing");
    std::set<std::string> seen;
    std::optional<double> measured;
    for (const auto &row : *rows) {
      const auto *entry = row.getAsObject();
      auto identity =
          entry ? entry->getString("execution_fingerprint") : std::nullopt;
      auto raw = entry ? entry->getString("raw_manifest_sha256") : std::nullopt;
      auto median = entry ? entry->getNumber("median_ms") : std::nullopt;
      auto count = entry ? entry->getInteger("sample_count") : std::nullopt;
      if (!identity || !digest(*identity) || !raw || !digest(*raw) || !median ||
          !std::isfinite(*median) || *median <= 0 || !count || *count <= 0 ||
          !seen.insert(identity->str()).second)
        return fail(error,
                    "observed-table row requires unique execution SHA256, raw "
                    "manifest SHA256, positive finite median and sample count");
      const auto *rowTarget = entry->getObject("target");
      if (entry->getString("boundary") != object->getString("boundary") ||
          !rowTarget ||
          staticFingerprint(
              llvm::json::Value(llvm::json::Object(*rowTarget))) !=
              staticFingerprint(
                  llvm::json::Value(llvm::json::Object(*modelTarget))))
        return fail(
            error,
            "observed-table row target/boundary differs from table identity");
      if (*identity == features.executionFingerprint)
        measured = *median;
    }
    const auto *actualTarget = target.getAsObject();
    if (!actualTarget)
      return unknown("out_of_domain",
                     "observed-table evaluation target is not declared");
    for (auto key : {"id", "toolchain", "runtime"})
      if (actualTarget->getString(key) != modelTarget->getString(key))
        return unknown("out_of_domain",
                       "observed-table target/toolchain/runtime mismatch");
    if (object->getString("boundary") != "prepared-command-host-to-host" ||
        actualTarget->getString("boundary") != object->getString("boundary"))
      return unknown("out_of_domain",
                     "observed-table timing boundary mismatch");
    if (features.values.find("command_count") == features.values.end() ||
        features.values.at("command_count") != 1)
      return unknown("out_of_domain",
                     "observed-table covers complete single commands only");
    if (!measured)
      return unknown("out_of_domain",
                     "no observation for exact unpriced execution plan");
    cost.valueMs = measured;
    cost.status = "measured";
    cost.reason = "exact observed complete-command median; no prediction or "
                  "generalization";
    return true;
  }
  if (kind == "features-only")
    return unknown("uncalibrated",
                   "native feature collection only; no fitted coefficients");
  const bool implementationAware = kind == "implementation-composite";
  if (kind != "linear-composite" && !implementationAware)
    return fail(error, "unsupported direct-AIR cost provider kind");
  auto calibration = object->getString("calibration");
  if (calibration == "uncalibrated")
    return unknown("uncalibrated",
                   "direct-AIR composite coefficients are not calibrated");
  if (calibration != "calibrated")
    return fail(error,
                "direct-AIR cost model requires explicit calibration status");
  for (auto key :
       {"model_id", "model_version", "training_manifest_digest", "boundary"})
    if (!object->getString(key) || object->getString(key)->empty())
      return fail(error,
                  "direct-AIR calibrated model identity/provenance missing");
  auto training = object->getString("training_manifest_digest");
  if (training->size() != 64 || !llvm::all_of(*training, [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
      }))
    return fail(
        error, "direct-AIR training manifest requires lowercase SHA256 digest");
  const auto *modelTarget = object->getObject("target");
  if (!modelTarget)
    return fail(error, "direct-AIR calibrated model target identity missing");
  for (auto key : {"id", "toolchain", "runtime"})
    if (!modelTarget->getString(key) || modelTarget->getString(key)->empty())
      return fail(error,
                  "direct-AIR target/toolchain/runtime identity missing");
  const auto *domain = object->getObject("domain");
  const auto *terms = object->getArray("terms");
  if (!domain || !terms || terms->empty() ||
      terms->size() > (implementationAware ? 4 : 8))
    return fail(error,
                "direct-AIR composite requires domain and one to eight frozen "
                "linear terms (four for implementation-composite)");
  auto sources = domain->getArray("source_fingerprints"),
       implementations = domain->getArray("implementations"),
       operations = domain->getArray("operations");
  auto ranges = domain->getObject("feature_ranges");
  if ((!implementationAware && (!sources || sources->empty())) ||
      !implementations || implementations->empty() || !operations || !ranges)
    return fail(error, "direct-AIR model requires source, implementation, "
                       "operation and feature coverage");
  auto covered = [](const llvm::json::Array &values, llvm::StringRef key) {
    return llvm::any_of(values,
                        [&](const auto &v) { return v.getAsString() == key; });
  };
  const auto *families = domain->getArray("family_fingerprints");
  const auto *architectures = domain->getArray("architecture_fingerprints");
  auto allowTails = domain->getBoolean("allow_tails");
  if (implementationAware) {
    if (domain->get("source_fingerprints") ||
        domain->get("source_feature_ranges"))
      return fail(error, "implementation-composite uses source-family/feature "
                         "domains, not exact source-hash ranges");
    if (!families || families->empty() || !architectures ||
        architectures->empty() || !allowTails)
      return fail(error, "implementation-composite requires source families, "
                         "architectures and explicit tail policy");
    for (const auto *identities : {families, architectures})
      for (const auto &identity : *identities)
        if (!identity.getAsString() || !digest(*identity.getAsString()))
          return fail(error, "implementation-composite family/architecture "
                             "identity must be SHA256");
    for (const char *key :
         {"command_count", "implemented_scalar_f32_product_operations",
          "implemented_mixed8_f32_product_operations", "hardware_stage_count",
          "source_max_loop_0_iterations", "source_max_loop_1_iterations",
          "source_max_loop_2_iterations", "max_tile_loop_0_iterations",
          "max_tile_loop_1_iterations", "max_tile_loop_2_iterations",
          "source_phase_count", "dma_bytes", "resident_payload_bytes",
          "capacity_bytes", "core_count_per_stage"})
      if (!ranges->get(key))
        return fail(error, "implementation-composite requires explicit "
                           "shape/tile/work/stage/DMA/resource envelope");
  }
  std::set<std::string> termsSeen;
  double total = 0;
  for (const auto &term : *terms) {
    auto entry = term.getAsObject();
    auto name = entry ? entry->getString("feature") : std::nullopt;
    auto coefficient =
        entry ? entry->getNumber("coefficient_ms") : std::nullopt;
    if (!name || !coefficient || !std::isfinite(*coefficient) ||
        *coefficient < 0 || !termsSeen.insert(name->str()).second ||
        !features.values.count(name->str()))
      return fail(error, "direct-AIR composite term has unknown/duplicate "
                         "feature or invalid coefficient");
    double contribution = *coefficient * features.values.at(name->str());
    if (!std::isfinite(contribution) || !std::isfinite(total + contribution))
      return fail(error, "direct-AIR composite estimate overflows");
    total += contribution;
  }
  if (implementationAware &&
      termsSeen !=
          std::set<std::string>{"command_count",
                                "implemented_scalar_f32_product_operations",
                                "implemented_mixed8_f32_product_operations",
                                "hardware_stage_count"})
    return fail(error, "implementation-composite requires the four declared "
                       "implementation/work/fixed-stage terms");
  // A newly explicit mixed task assignment is outside a legacy aggregate
  // linear calibration unless its selected scalar/mixed work is represented.
  // The established implementation-composite has its own strict four-term
  // contract and must not be weakened or rejected by this compatibility guard.
  auto local = features.values.find("explicit_task_implementation_choices");
  if (!implementationAware && local != features.values.end() &&
      local->second > 0 && features.implementations.size() > 1) {
    for (const auto &implementation : features.implementations)
      if (implementation != "scalar:1" &&
          implementation != "scalar-product-vector-add-f32:8")
        return unknown("out_of_domain",
                       "explicit mixed task implementation work is not covered "
                       "by this linear provider");
    for (const char *key : {"implemented_scalar_f32_product_operations",
                            "implemented_mixed8_f32_product_operations",
                            "implemented_scalar_f32_addsub_operations",
                            "implemented_mixed8_f32_addsub_packs"}) {
      auto work = features.values.find(key);
      if (work != features.values.end() && work->second > 0 &&
          (!termsSeen.count(key) || !ranges->get(key)))
        return unknown("out_of_domain",
                       "explicit mixed task selection requires "
                       "implementation-resolved terms and calibrated ranges");
    }
  }
  for (const auto &range : *ranges) {
    auto limits = range.second.getAsArray();
    auto actual = features.values.find(range.first.str());
    if (actual == features.values.end() || !limits || limits->size() != 2 ||
        !(*limits)[0].getAsInteger() || !(*limits)[1].getAsInteger() ||
        *(*limits)[0].getAsInteger() < 0 ||
        *(*limits)[1].getAsInteger() < *(*limits)[0].getAsInteger())
      return fail(error, "invalid direct-AIR model feature range");
  }
  const auto *sourceRanges = domain->getObject("source_feature_ranges");
  if (domain->get("source_feature_ranges") &&
      (!sourceRanges || sourceRanges->empty()))
    return fail(
        error, "source_feature_ranges requires a nonempty source-keyed object");
  if (sourceRanges)
    for (const auto &source : *sourceRanges) {
      const auto *entries = source.second.getAsObject();
      if (!covered(*sources, source.first.str()) || !entries ||
          entries->empty())
        return fail(error, "source_feature_ranges requires declared source and "
                           "nonempty native ranges");
      for (const auto &range : *entries) {
        auto limits = range.second.getAsArray();
        if (!features.values.count(range.first.str()) || !limits ||
            limits->size() != 2 || !(*limits)[0].getAsInteger() ||
            !(*limits)[1].getAsInteger() || *(*limits)[0].getAsInteger() < 0 ||
            *(*limits)[1].getAsInteger() < *(*limits)[0].getAsInteger())
          return fail(error, "invalid per-source native feature range");
      }
    }
  const auto *actualTarget = target.getAsObject();
  if (!actualTarget)
    return unknown("out_of_domain",
                   "evaluation target identity is not declared");
  for (auto key : {"id", "toolchain", "runtime"})
    if (actualTarget->getString(key) != modelTarget->getString(key))
      return unknown(
          "out_of_domain",
          "target/toolchain/runtime differs from frozen calibration");
  if (object->getString("boundary") != "prepared-command-host-to-host" ||
      actualTarget->getString("boundary") != object->getString("boundary"))
    return unknown("out_of_domain",
                   "composite timing boundary is not the supported prepared "
                   "single-command boundary");
  if (features.values.find("command_count") == features.values.end() ||
      features.values.at("command_count") != 1)
    return unknown("out_of_domain",
                   "partitioned complete-program overhead is not calibrated by "
                   "this single-command provider");
  if (features.values.find("all_storage_f32") == features.values.end() ||
      features.values.at("all_storage_f32") != 1)
    return unknown(
        "out_of_domain",
        "calibrated direct-AIR provider currently requires f32 storage");
  if (!implementationAware)
    for (const auto &source : features.sourceFingerprints)
      if (!covered(*sources, source))
        return unknown("out_of_domain",
                       "source fingerprint outside frozen source groups");
  if (implementationAware) {
    if (!digest(features.regionFamilyFingerprint) ||
        !covered(*families, features.regionFamilyFingerprint))
      return unknown(
          "out_of_domain",
          "source SSA/access family outside implementation calibration");
    if (!digest(features.architectureFingerprint) ||
        !covered(*architectures, features.architectureFingerprint))
      return unknown("out_of_domain",
                     "target memory/link/resource architecture outside "
                     "implementation calibration");
    auto fact = [&](const char *key, int64_t value) {
      auto found = features.values.find(key);
      return found != features.values.end() && found->second == value;
    };
    if (!fact("all_source_loops_affine", 1) ||
        !fact("all_source_loops_unit_step", 1) ||
        !fact("mixed8_lane_coverage_exact", 1))
      return unknown("out_of_domain",
                     "implementation model requires static affine unit-step "
                     "source and exact supported lane coverage");
    if (!*allowTails && !fact("source_has_tile_tail", 0))
      return unknown(
          "out_of_domain",
          "source/tile tails not included in frozen calibration domain");
    for (const auto &operation : features.operations)
      if (operation.first != "arith.mulf:f32" &&
          operation.first != "arith.addf:f32" &&
          operation.first != "arith.subf:f32")
        return unknown("out_of_domain",
                       "implementation model currently covers source f32 "
                       "multiply/add/subtract only");
    for (const auto &implementation : features.implementations)
      if (implementation != "scalar:1" &&
          implementation != "scalar-product-vector-add-f32:8")
        return unknown(
            "out_of_domain",
            "implementation model covers scalar and supported mixed8 only");
  }
  for (const auto &implementation : features.implementations)
    if (!covered(*implementations, implementation))
      return unknown("out_of_domain",
                     "compute implementation outside calibrated domain");
  for (const auto &operation : features.operations)
    if (!covered(*operations, operation.first))
      return unknown("out_of_domain",
                     "source operation outside calibrated coverage");
  for (const auto &range : *ranges) {
    auto limits = range.second.getAsArray();
    int64_t actual = features.values.at(range.first.str());
    if (actual < *(*limits)[0].getAsInteger() ||
        actual > *(*limits)[1].getAsInteger())
      return unknown("out_of_domain",
                     "native feature outside calibrated range");
  }
  if (sourceRanges)
    for (const auto &source : features.sourceFingerprints) {
      const auto *entries = sourceRanges->getObject(source);
      if (!entries)
        return unknown("out_of_domain",
                       "source has no frozen configuration range");
      for (const auto &range : *entries) {
        auto limits = range.second.getAsArray();
        int64_t actual = features.values.at(range.first.str());
        if (actual < *(*limits)[0].getAsInteger() ||
            actual > *(*limits)[1].getAsInteger())
          return unknown(
              "out_of_domain",
              "source configuration outside frozen native feature range");
      }
    }
  if (total <= 0)
    return unknown(
        "uncalibrated",
        "composite model has no positive estimate for this candidate");
  cost.valueMs = total;
  cost.status = "calibrated";
  cost.reason =
      implementationAware
          ? "implementation-aware prepared-region composite; transfer/runtime "
            "included, no independently identified physical components"
          : "composite full-call estimate; terms are not independently "
            "measured physical components";
  return true;
}
namespace {
llvm::json::Value directCostReport(const GenericDirectAIRCost &cost) {
  llvm::json::Object report{
      {"status", cost.status},
      {"reason", cost.reason},
      {"value_ms", cost.valueMs ? llvm::json::Value(*cost.valueMs)
                                : llvm::json::Value(nullptr)},
      {"feature_digest", cost.featureDigest},
      {"model_digest", cost.modelDigest},
      {"features", serializeGenericCostFeatures(cost.features)},
      {"model", llvm::json::Value(cost.model)},
      {"evaluation_target", llvm::json::Value(cost.target)},
      {"charge_scope",
       "whole-command prepared host arrays through returned output; includes "
       "transfer/runtime effects; never add separate DMA charges"}};
  if (!cost.eventTiming.getAsNull()) {
    report["event_timing"] = llvm::json::Value(cost.eventTiming);
    report["charge_scope"] = "activity-exclusive public execution events; each "
                             "covered event charged once";
  }
  return report;
}
} // namespace

bool orbit::taskflow::planGenericMapping(const GenericAnalysis &a,
                                         const llvm::json::Value &config,
                                         GenericMappingPlan &p,
                                         std::string &e) {
  ++mappingStatistics.plannerCalls;
  p = GenericMappingPlan{};
  p.bodyFingerprint = a.bodyFingerprint;
  auto module = a.function->getParentOfType<mlir::ModuleOp>();
  if (!readNativeMappingChoice(module, p.choices, e))
    return false;
  int64_t lanes = 1;
  std::string requestedImplementation = "scalar";
  if (auto implementation =
          module->getAttrOfType<StringAttr>("orbit.implementation")) {
    if (implementation.getValue() != "scalar" &&
        implementation.getValue() != "independent-lane-f32" &&
        implementation.getValue() != "scalar-product-vector-add-f32" &&
        implementation.getValue() != programMatrixImplementation)
      return fail(e, "unsupported program implementation choice");
    requestedImplementation = implementation.getValue().str();
    if (implementation.getValue() != "scalar" &&
        implementation.getValue() != programMatrixImplementation) {
      auto attr = module->getAttrOfType<IntegerAttr>("orbit.vector_lanes");
      if (!attr || attr.getInt() < 2 || attr.getInt() > 16 ||
          (attr.getInt() & (attr.getInt() - 1)))
        return fail(
            e, "independent-lane-f32 requires power-of-two lanes in [2,16]");
      lanes = attr.getInt();
      if (requestedImplementation == "scalar-product-vector-add-f32" &&
          lanes != 4 && lanes != 8)
        return fail(e, "scalar-product-vector-add-f32 requires lanes 4 or 8");
    }
  }
  std::vector<size_t> initializerTasks;
  if (!readNativeInitializerTasks(a, initializerTasks, e))
    return false;
  std::vector<NativeTaskImplementationChoice> localImplementations;
  if (!readNativeTaskImplementations(module, a.tasks.size(),
                                     localImplementations, e))
    return false;
  for (size_t task = 0; task < a.tasks.size(); ++task) {
    std::vector<size_t> order;
    if (p.choices.residency == "readonly-panel" &&
        panelTask(a, task, p.choices)) {
      order.push_back(p.choices.panelLoop);
      order.push_back(p.choices.reuseLoop);
    }
    for (size_t loop = 0; loop < a.tasks[task].loops.size(); ++loop)
      if (!llvm::is_contained(order, loop))
        order.push_back(loop);
    p.taskLoopOrders.push_back(std::move(order));
    GenericTaskImplementation implementation;
    implementation.task = task;
    auto local = llvm::find_if(localImplementations, [&](const auto &entry) {
      return entry.task == task;
    });
    const bool initializer = llvm::is_contained(initializerTasks, task);
    const bool strict = local != localImplementations.end();
    if (initializer && strict && (local->kind != "scalar" || local->lanes != 1))
      return fail(e, "explicit non-scalar request conflicts with proved "
                     "initializer source");
    const auto requested = strict        ? local->kind
                           : initializer ? std::string("scalar")
                                         : requestedImplementation;
    const int64_t taskLanes = strict ? local->lanes : initializer ? 1 : lanes;
    if (requested == programMatrixImplementation) {
      ProgramMatrixTaskContract matrix;
      std::string reason;
      if (analyzeProgramMatrixTask(a, task, matrix, reason))
        implementation.kind = programMatrixImplementation;
      else if (strict)
        return fail(e, "explicit task matrix implementation unsupported: " +
                           reason);
    }
    int64_t loop = taskLanes > 1 ? vectorLoop(a, a.tasks[task]) : -1;
    if (loop >= 0 && requested == "scalar-product-vector-add-f32" &&
        a.tasks[task].loops[loop].step != 1)
      loop = -1;
    if (loop >= 0) {
      implementation.kind = requested;
      implementation.parallelLoop = loop;
      implementation.lanes = taskLanes;
    } else if (strict && taskLanes > 1)
      return fail(e, "explicit task vector implementation lacks source proof");
    p.taskImplementations.push_back(std::move(implementation));
  }
  if (localImplementations.empty() &&
      requestedImplementation == "scalar-product-vector-add-f32" &&
      !llvm::any_of(p.taskImplementations, [](const auto &impl) {
        return impl.kind == "scalar-product-vector-add-f32";
      }))
    return fail(e, "scalar-product-vector-add-f32 request has no vectorizable "
                   "source task");
  if (localImplementations.empty() &&
      requestedImplementation == programMatrixImplementation &&
      !llvm::any_of(p.taskImplementations, [](const auto &impl) {
        return impl.kind == programMatrixImplementation;
      })) {
    return fail(e,
                "matrix implementation request has no proved BF16 matrix task");
  }
  auto object = config.getAsObject();
  auto cfg = object ? object->getObject("generic_air") : nullptr;
  if (cfg && cfg->get("cost_model") && cfg->get("theoretical"))
    return fail(e, "direct-AIR composite and legacy theoretical providers are "
                   "mutually exclusive");
  if (cfg)
    if (auto capacity = cfg->getInteger("capacity_bytes"))
      p.capacityBytes = *capacity;
  if (!configureGenericMemory(config, p.capacityBytes, p.memoryBinding, e) ||
      !getGenericTargetCapabilities(config, p.targetCapabilities, e))
    return false;
  for (size_t i = 0; i < a.storages.size(); ++i) {
    if (orderOnlyInput(a, i))
      continue;
    int64_t n = storageBytes(a.storages[i]);
    if (n < 0 || p.residentBytes > INT64_MAX - n)
      return fail(e, "invalid static storage byte size");
    p.residentBytes += n;
    auto role = a.storages[i].role;
    if ((role == "input" && !orderOnlyInput(a, i)) || role == "output")
      p.transfers.push_back(
          {i, role == "input" ? "host_to_l1" : "l1_to_host", n});
  }
  std::set<size_t> scheduled;
  while (scheduled.size() < a.tasks.size()) {
    bool progress = false;
    for (size_t i = 0; i < a.tasks.size(); ++i) {
      if (scheduled.count(i))
        continue;
      bool ready = true;
      for (auto d : a.tasks[i].dependencies)
        ready &= scheduled.count(d) != 0;
      if (ready) {
        p.orderedTasks.push_back(i);
        scheduled.insert(i);
        progress = true;
      }
    }
    if (!progress)
      return fail(e, "cyclic task dependencies");
  }
  if (p.choices.grouping == "tile-major" &&
      !deriveGenericResidencyOrder(a, genericResidencyGroups(p.choices),
                                   p.orderedTasks, e))
    return false;
  bool requestTiling =
      p.residentBytes > p.capacityBytes ||
      (p.choices.residency != "phase-local" &&
       p.choices.residency != "whole-program") ||
      llvm::any_of(p.taskImplementations,
                   [](const auto &entry) { return entry.lanes > 1; });
  if (auto attr = a.function->getParentOfType<mlir::ModuleOp>()
                      ->getAttrOfType<mlir::DenseI64ArrayAttr>(
                          "orbit.air_tile_sizes")) {
    p.tileSizes.assign(attr.asArrayRef().begin(), attr.asArrayRef().end());
    requestTiling = true;
  }
  if (cfg) {
    if (auto sizes = cfg->getArray("tile_sizes")) {
      p.tileSizes.clear();
      for (const auto &size : *sizes) {
        auto n = size.getAsInteger();
        if (!n || *n <= 0)
          return fail(e, "tile_sizes requires positive integers");
        p.tileSizes.push_back(*n);
      }
      requestTiling = true;
    }
    if (auto tiled = cfg->getBoolean("tiled"))
      requestTiling |= *tiled;
  }
  if (p.capacityBytes <= 0)
    return fail(e, "mapping requires a positive declared local capacity");
  if (p.choices.residency == "whole-program") {
    if (p.residentBytes > p.capacityBytes)
      return fail(e, "whole-program residency exceeds target capacity; no "
                     "implicit tiling fallback");
    if (requestTiling || !p.tileSizes.empty())
      return fail(
          e,
          "whole-program residency contradicts explicit tiling or non-scalar "
          "implementation; choose tiled-phase-local or readonly-panel");
  }
  if (requestTiling) {
    p.strategy = "serial-tiled";
    for (auto size : p.tileSizes)
      if (size <= 0)
        return fail(e, "tile sizes must be positive");
    if (!buildTiled(a, p, e))
      return false;
  }
  if (cfg)
    if (auto theory = cfg->getObject("theoretical")) {
      auto launch = theory->getNumber("launch_ms"),
           op = theory->getNumber("per_scalar_operation_ms"),
           dma = theory->getNumber("per_transfer_byte_ms");
      if (!launch || !op || !dma || *launch < 0 || *op < 0 || *dma < 0)
        return fail(e,
                    "explicit nonnegative theoretical coefficients required");
      double count = 0;
      for (auto &task : a.tasks)
        task.operation->walk([&](mlir::Operation *operation) {
          if ((operation->getName().getDialectNamespace() != "arith" &&
               operation->getName().getDialectNamespace() != "math") ||
              operation->getName().getStringRef() == "arith.constant")
            return;
          double multiplicity = 1;
          for (auto &loop : task.loops)
            if (loop.operation->isProperAncestor(operation))
              multiplicity *=
                  double((loop.upper - loop.lower + loop.step - 1) / loop.step);
          count += multiplicity;
        });
      double transferBytes = 0;
      for (auto &t : p.transfers)
        transferBytes += t.bytes;
      p.theoreticalLaunchMs = *launch;
      p.theoreticalPerOperationMs = *op;
      p.theoreticalPerByteMs = *dma;
      p.hasCost = true;
      p.estimatedMs = *launch + *op * count + *dma * transferBytes;
      p.costProvenance = "theoretical direct-AIR source scalar-operation and "
                         "explicit-transfer "
                         "sensitivity; not calibrated; backend local copies "
                         "and internal scheduling excluded";
    }
  if (p.strategy == "single-herd-resident") {
    ProgramExecutionGroup group;
    group.beginPhase = 0;
    group.endPhase = p.orderedTasks.size();
    group.resource = p.memoryBinding.computeResource;
    group.memory = p.memoryBinding.localMemory;
    group.spatialOwner = computeOwner(p);
    p.executionGroups = {std::move(group)};
  }
  if (!buildProgramCommunication(a, p, e))
    return false;
  if (cfg && cfg->get("cost_model")) {
    GenericCostFeatures features;
    const auto *target = cfg->get("cost_target");
    llvm::json::Value unknownTarget = nullptr;
    const auto *model = cfg->get("cost_model")->getAsObject();
    if (model && (model->getString("kind") == "event-mechanism" ||
                  model->getString("kind") == "region-invocation")) {
      if (!evaluateGenericMechanismCost(a, p, *cfg->get("cost_model"),
                                        target ? *target : unknownTarget,
                                        p.directCost, e))
        return false;
    } else if (!collectGenericCostFeatures(a, p, features, e) ||
               !evaluateGenericDirectAIRCost(features, *cfg->get("cost_model"),
                                             target ? *target : unknownTarget,
                                             p.directCost, e))
      return false;
    p.hasCost = p.directCost.valueMs.has_value();
    p.estimatedMs = p.directCost.valueMs.value_or(0);
    p.costProvenance = directCostProvenance(p.directCost);
  }
  if (!checkGenericMapping(a, p, e))
    return false;
  p.planId =
      staticFingerprint(canonicalChosenPlan(serializeGenericMapping(a, p)));
  return true;
}
llvm::json::Value
orbit::taskflow::serializeGenericMapping(const GenericAnalysis &a,
                                         const GenericMappingPlan &p) {
  llvm::json::Array tasks, transfers, residency, phases, tileSizes,
      phaseDomains, transferSummary, retentions, fusedInitializers,
      hardwareStages, channelAllocations, hostBuffers, loopOrders,
      opportunities, implementations, readonlyRetentions, executionGroups;
  for (const auto &choice : discoverGenericMappingChoices(a)) {
    if (choice.residency == "phase-local")
      continue;
    if (choice.residency != "readonly-panel") {
      opportunities.push_back(llvm::json::Object{
          {"storage_index", int64_t(-1)},
          {"invariant_source_loop", int64_t(-1)},
          {"panel_source_loop", int64_t(-1)},
          {"tasks", llvm::json::Array{}},
          {"data_organization", choice.dataOrganization},
          {"realization", choice.residency},
          {"grouping", choice.grouping},
          {"fact_proof", "actual checked program accesses, copy versions and "
                         "capacity; feasibility checked after selection"},
          {"proposal", choice.residency == "whole-program"
                           ? "retain shared inputs and intermediates within "
                             "the whole program"
                       : choice.grouping == "tile-major"
                           ? "retain selected source subgraph tiles with "
                             "required boundary exports"
                           : "materialize phase-local transfer boundaries with "
                             "selected tile sizes"}});
      continue;
    }
    llvm::json::Array tasks;
    for (size_t task = 0; task < a.tasks.size(); ++task)
      if (panelTask(a, task, choice))
        tasks.push_back(int64_t(task));
    opportunities.push_back(llvm::json::Object{
        {"storage_index", choice.storage},
        {"invariant_source_loop", choice.reuseLoop},
        {"panel_source_loop", choice.panelLoop},
        {"tasks", std::move(tasks)},
        {"fact_proof", "immutable input; real affine SSA indices independent "
                       "of reuse loop; every access to written storage "
                       "projects both outer IVs onto disjoint slices"},
        {"data_organization", choice.dataOrganization},
        {"realization", "readonly-panel"},
        {"realization_scope",
         "one execution group; no retention across AIR stages"},
        {"proposal",
         "retain-across-users from actual reuse; readonly rectangular panel "
         "with source-independent outer-loop permutation; capacity/target "
         "feasibility checked after selection"}});
  }
  for (size_t task = 0; task < p.taskLoopOrders.size(); ++task) {
    llvm::json::Array order;
    for (size_t loop : p.taskLoopOrders[task])
      order.push_back(int64_t(loop));
    loopOrders.push_back(llvm::json::Object{{"task", int64_t(task)},
                                            {"loop_order", std::move(order)}});
  }
  for (const auto &implementation : p.taskImplementations)
    implementations.push_back(
        llvm::json::Object{{"task", int64_t(implementation.task)},
                           {"kind", implementation.kind},
                           {"parallel_loop", implementation.parallelLoop},
                           {"lanes", implementation.lanes}});
  for (const auto &scope : p.readOnlyRetentions) {
    llvm::json::Array offsets, sizes;
    for (auto n : scope.offsets)
      offsets.push_back(n);
    for (auto n : scope.sizes)
      sizes.push_back(n);
    readonlyRetentions.push_back(
        llvm::json::Object{{"begin_phase", int64_t(scope.beginPhase)},
                           {"end_phase_exclusive", int64_t(scope.endPhase)},
                           {"storage_index", int64_t(scope.storage)},
                           {"offsets", std::move(offsets)},
                           {"sizes", std::move(sizes)}});
  }
  for (const auto &group : p.executionGroups) {
    llvm::json::Array dependencies, implementationDependencies;
    for (auto dependency : group.implementationDependencies)
      implementationDependencies.push_back(int64_t(dependency));
    for (auto dependency : group.dependencies)
      dependencies.push_back(int64_t(dependency));
    executionGroups.push_back(llvm::json::Object{
        {"begin_phase", int64_t(group.beginPhase)},
        {"end_phase_exclusive", int64_t(group.endPhase)},
        {"resource", group.resource},
        {"memory", group.memory},
        {"depends_on", std::move(dependencies)},
        {"implementation_depends_on", std::move(implementationDependencies)}});
  }
  llvm::json::Object mappingChoices{
      {"data_organization", p.choices.dataOrganization},
      {"residency", p.choices.residency},
      {"storage", p.choices.storage},
      {"reuse_loop", p.choices.reuseLoop},
      {"panel_loop", p.choices.panelLoop},
      {"grouping", p.choices.grouping}};
  if (!p.choices.taskGroup.empty()) {
    llvm::json::Array v;
    for (auto t : p.choices.taskGroup)
      v.push_back(int64_t(t));
    mappingChoices["task_group"] = std::move(v);
  }
  if (!p.choices.retainedStorages.empty()) {
    llvm::json::Array v;
    for (auto t : p.choices.retainedStorages)
      v.push_back(int64_t(t));
    mappingChoices["retained_storages"] = std::move(v);
  }
  if (!p.choices.residencyGroups.empty()) {
    llvm::json::Array groups;
    for (const auto &g : p.choices.residencyGroups) {
      llvm::json::Array tasks, storages;
      for (auto t : g.tasks)
        tasks.push_back(int64_t(t));
      for (auto x : g.retainedStorages)
        storages.push_back(int64_t(x));
      groups.push_back(
          llvm::json::Object{{"tasks", std::move(tasks)},
                             {"retained_storages", std::move(storages)}});
    }
    mappingChoices["residency_groups"] = std::move(groups);
  }
  llvm::json::Array residencyScopes;
  for (const auto &scope : p.residencyScopes) {
    llvm::json::Array tasks, storages, exports;
    for (auto t : scope.tasks)
      tasks.push_back(int64_t(t));
    for (auto x : scope.retainedStorages)
      storages.push_back(int64_t(x));
    for (auto x : scope.boundaryExports)
      exports.push_back(int64_t(x));
    residencyScopes.push_back(
        llvm::json::Object{{"begin_phase", int64_t(scope.beginPhase)},
                           {"end_phase_exclusive", int64_t(scope.endPhase)},
                           {"tasks", std::move(tasks)},
                           {"retained_storages", std::move(storages)},
                           {"boundary_exports", std::move(exports)},
                           {"input_release_policy", scope.inputReleasePolicy}});
  }

  bool compact = p.phases.size() > 16384;
  std::vector<int64_t> stageAllocationBytes;
  std::vector<llvm::json::Array> stageBufferPools;
  if (p.strategy == "serial-tiled")
    stagePoolBytes(a, p, nullptr, &stageAllocationBytes, &stageBufferPools);
  int64_t systemMemoryPeak = 0;
  for (size_t i = 0; i < a.storages.size(); ++i) {
    const auto &storage = a.storages[i];
    if (!llvm::any_of(p.communication.allocations, [&](const auto &allocation) {
          return allocation.storage == i &&
                 allocation.memory == p.memoryBinding.hostMemory;
        }))
      continue;
    int64_t bytes = storageBytes(storage);
    systemMemoryPeak += bytes;
    llvm::json::Array shape;
    for (auto n : storage.shape)
      shape.push_back(n);
    hostBuffers.push_back(
        llvm::json::Object{{"storage_index", int64_t(i)},
                           {"storage", storage.id},
                           {"role", storage.role},
                           {"dtype", storage.dtype},
                           {"shape", std::move(shape)},
                           {"bytes", bytes},
                           {"memory", p.memoryBinding.hostMemory},
                           {"lifetime", "whole_graph_lifetime"}});
  }
  for (auto task : p.fusedInitializerTasks)
    fusedInitializers.push_back(int64_t(task));
  std::set<std::pair<size_t, size_t>> initializedLocally;
  for (const auto &scope : p.retainedRegions)
    if (scope.initializerTask >= 0)
      initializedLocally.insert(
          {p.phases[scope.beginPhase].task, scope.storage});
  llvm::MD5 scheduleHash;
  auto hashInteger = [&](int64_t value) {
    scheduleHash.update(std::to_string(value));
    scheduleHash.update(",");
  };
  auto hashIntegers = [&](const auto &values) {
    hashInteger(values.size());
    for (auto value : values)
      hashInteger(value);
  };
  hashInteger(p.phases.size());
  for (const auto &phase : p.phases) {
    hashInteger(phase.task);
    hashInteger(phase.loops.size());
    for (auto bound : phase.loops) {
      hashInteger(bound.loop);
      hashInteger(bound.lower);
      hashInteger(bound.upper);
    }
    hashIntegers(phase.dependencies);
    hashInteger(phase.regions.size());
    for (const auto &region : phase.regions) {
      hashInteger(region.storage);
      hashIntegers(region.offsets);
      hashIntegers(region.sizes);
      hashInteger(region.read);
      hashInteger(region.write);
    }
  }
  hashInteger(p.retainedRegions.size());
  for (const auto &scope : p.retainedRegions) {
    hashInteger(scope.beginPhase);
    hashInteger(scope.endPhase);
    hashInteger(scope.storage);
    hashInteger(scope.initializerTask);
    retentions.push_back(llvm::json::Object{
        {"begin_phase", int64_t(scope.beginPhase)},
        {"end_phase_exclusive", int64_t(scope.endPhase)},
        {"storage_index", int64_t(scope.storage)},
        {"initializer_task", scope.initializerTask},
        {"lifetime",
         "on-chip across invariant region scope; import first, export last"}});
  }
  hashIntegers(p.fusedInitializerTasks);
  hashInteger(p.hardwareStages.size());
  size_t stageIndex = 0;
  for (const auto &stage : p.hardwareStages) {
    hashInteger(stage.beginPhase);
    hashInteger(stage.endPhase);
    hashInteger(stage.legalizeChannels);
    std::map<std::pair<size_t, std::string>, std::pair<int64_t, int64_t>>
        stageTraffic;
    auto transfer = std::lower_bound(
        p.transfers.begin(), p.transfers.end(), stage.beginPhase,
        [](const auto &t, size_t phase) { return t.phase < phase; });
    for (; transfer != p.transfers.end() && transfer->phase < stage.endPhase;
         ++transfer) {
      auto &traffic = stageTraffic[{transfer->storage, transfer->direction}];
      ++traffic.first;
      traffic.second += transfer->bytes;
    }
    llvm::json::Array stageTransfers;
    for (const auto &traffic : stageTraffic)
      stageTransfers.push_back(
          llvm::json::Object{{"storage_index", int64_t(traffic.first.first)},
                             {"storage", a.storages[traffic.first.first].id},
                             {"direction", traffic.first.second},
                             {"count", traffic.second.first},
                             {"bytes", traffic.second.second}});
    hardwareStages.push_back(llvm::json::Object{
        {"group_index", int64_t(stage.groupIndex)},
        {"begin_phase", int64_t(stage.beginPhase)},
        {"end_phase_exclusive", int64_t(stage.endPhase)},
        {"allocated_bytes", stageAllocationBytes[stageIndex]},
        {"buffer_pools", std::move(stageBufferPools[stageIndex++])},
        {"legalize_channels", stage.legalizeChannels},
        {"transfer_summary", std::move(stageTransfers)},
        {"pool_key", "read-only: storage+region shape+loop extent signature; "
                     "writable: storage"}});
  }
  hashInteger(p.channelMergePolicy.size());
  scheduleHash.update(p.channelMergePolicy);
  hashInteger(p.channelAllocations.size());
  for (const auto &allocation : p.channelAllocations) {
    hashInteger(allocation.stage);
    hashInteger(allocation.storage);
    hashInteger(allocation.tileChannel);
    hashInteger(allocation.transferCount);
    hashInteger(allocation.bytes);
    hashInteger(allocation.logicalFifo.size());
    scheduleHash.update(allocation.logicalFifo);
    hashInteger(allocation.direction.size());
    scheduleHash.update(allocation.direction);
    channelAllocations.push_back(llvm::json::Object{
        {"stage", int64_t(allocation.stage)},
        {"storage_index", int64_t(allocation.storage)},
        {"storage", a.storages[allocation.storage].id},
        {"direction", allocation.direction},
        {"tile_channel", allocation.tileChannel},
        {"logical_fifo", allocation.logicalFifo},
        {"transfer_count", allocation.transferCount},
        {"bytes", allocation.bytes},
        {"stream_order", "checked source phase order"},
        {"tile_direction",
         allocation.direction == "host_to_l1" ? "S2MM" : "MM2S"}});
  }
  hashInteger(p.transfers.size());
  std::map<std::pair<size_t, std::string>, std::pair<int64_t, int64_t>>
      summaries;
  for (const auto &transfer : p.transfers) {
    hashInteger(transfer.storage);
    hashInteger(transfer.phase);
    hashInteger(transfer.bytes);
    hashInteger(transfer.direction.size());
    scheduleHash.update(transfer.direction);
    hashIntegers(transfer.offsets);
    hashIntegers(transfer.sizes);
    auto &summary = summaries[{transfer.storage, transfer.direction}];
    ++summary.first;
    summary.second += transfer.bytes;
  }
  llvm::MD5::MD5Result hashResult;
  scheduleHash.final(hashResult);
  llvm::SmallString<32> scheduleDigest;
  llvm::MD5::stringifyResult(hashResult, scheduleDigest);
  for (const auto &summary : summaries)
    transferSummary.push_back(
        llvm::json::Object{{"storage_index", int64_t(summary.first.first)},
                           {"storage", a.storages[summary.first.first].id},
                           {"direction", summary.first.second},
                           {"count", summary.second.first},
                           {"bytes", summary.second.second}});
  for (size_t begin = 0; begin < p.phases.size();) {
    size_t end = begin + 1;
    while (end < p.phases.size() && p.phases[end].task == p.phases[begin].task)
      ++end;
    const auto &phase = p.phases[begin];
    const auto &task = a.tasks[phase.task];
    llvm::json::Array loopDomains;
    for (size_t l = 0; l < phase.loops.size(); ++l) {
      const auto &loop = task.loops[l];
      const auto &tile = phase.loops[l];
      loopDomains.push_back(llvm::json::Object{
          {"loop", int64_t(l)},
          {"lower", loop.lower},
          {"upper", loop.upper},
          {"step", loop.step},
          {"tile_iterations", (tile.upper - tile.lower - 1) / loop.step + 1}});
    }
    phaseDomains.push_back(llvm::json::Object{
        {"task", int64_t(phase.task)},
        {"first_phase", int64_t(begin)},
        {"phase_count", int64_t(end - begin)},
        {"loop_domains", std::move(loopDomains)},
        {"order", "Cartesian tile grid in task_loop_orders; each phase depends "
                  "on its predecessor"},
        {"region_derivation",
         "actual source affine accesses restricted to each tile domain"}});
    if (llvm::is_contained(p.fusedInitializerTasks, phase.task)) {
      begin = end;
      continue;
    }
    std::map<size_t, int64_t> storagePools;
    std::set<size_t> stagingPools;
    for (size_t i = begin; i < end; ++i)
      for (const auto &region : p.phases[i].regions) {
        auto type = cast<MemRefType>(a.storages[region.storage].root.getType());
        int64_t bytes = (type.getElementType().getIntOrFloatBitWidth() + 7) / 8;
        for (auto n : region.sizes)
          bytes *= n;
        storagePools[region.storage] =
            std::max(storagePools[region.storage], bytes);
        if (region.read && region.write &&
            !initializedLocally.count({phase.task, region.storage}))
          stagingPools.insert(region.storage);
      }
    for (const auto &pool : storagePools)
      residency.push_back(llvm::json::Object{
          {"task", int64_t(phase.task)},
          {"storage", a.storages[pool.first].id},
          {"source_region_and_staging_bytes",
           pool.second * (stagingPools.count(pool.first) ? 2 : 1)},
          {"import_staging_bytes",
           stagingPools.count(pool.first) ? pool.second : 0},
          {"memory", p.memoryBinding.localMemory},
          {"lifetime", "task source region; actual shape-specific hardware "
                       "pool totals listed in hardware_stages"}});
    begin = end;
  }
  auto ints = [](const std::vector<int64_t> &values) {
    llvm::json::Array out;
    for (auto v : values)
      out.push_back(v);
    return out;
  };
  for (auto size : p.tileSizes)
    tileSizes.push_back(size);
  for (size_t i = 0; !compact && i < p.phases.size(); ++i) {
    const auto &phase = p.phases[i];
    llvm::json::Array loops, regions, deps;
    for (auto dependency : phase.dependencies)
      deps.push_back(int64_t(dependency));
    for (auto b : phase.loops)
      loops.push_back(llvm::json::Object{
          {"loop", int64_t(b.loop)}, {"lower", b.lower}, {"upper", b.upper}});
    for (auto &r : phase.regions)
      regions.push_back(
          llvm::json::Object{{"storage_index", int64_t(r.storage)},
                             {"offsets", ints(r.offsets)},
                             {"sizes", ints(r.sizes)},
                             {"read", r.read},
                             {"write", r.write}});
    phases.push_back(llvm::json::Object{
        {"phase", int64_t(i)},
        {"task", int64_t(phase.task)},
        {"dependencies", std::move(deps)},
        {"loops", std::move(loops)},
        {"regions", std::move(regions)},
        {"lifetime",
         "phase live region; backing pool allocated for task stage"}});
  }
  for (auto i : p.orderedTasks)
    tasks.push_back(
        llvm::json::Object{{"task", a.tasks[i].id},
                           {"index", int64_t(i)},
                           {"device", computeDevice(p)},
                           {"resource", p.memoryBinding.computeResource},
                           {"spatial_owner", computeOwner(p)}});
  if (!compact)
    for (auto &t : p.transfers)
      transfers.push_back(llvm::json::Object{
          {"storage", a.storages[t.storage].id},
          {"storage_index", int64_t(t.storage)},
          {"direction", t.direction},
          {"bytes", t.bytes},
          {"phase", int64_t(t.phase)},
          {"offsets", ints(t.offsets)},
          {"sizes", ints(t.sizes)},
          {"extent", p.strategy == "serial-tiled" ? "rectangular_tile"
                                                  : "full_static_buffer"},
          {"source", t.direction == "host_to_l1" ? p.memoryBinding.hostMemory
                                                 : p.memoryBinding.localMemory},
          {"target", t.direction == "host_to_l1"
                         ? p.memoryBinding.localMemory
                         : p.memoryBinding.hostMemory}});
  if (p.strategy == "single-herd-resident")
    for (size_t i = 0; i < a.storages.size(); ++i)
      residency.push_back(
          llvm::json::Object{{"storage", a.storages[i].id},
                             {"memory", p.memoryBinding.localMemory},
                             {"lifetime", "whole_herd"},
                             {"bytes", storageBytes(a.storages[i])}});
  int64_t logicalReadBytes = 0, logicalWriteBytes = 0, scalarReads = 0,
          scalarWrites = 0;
  bool accessCountsValid = true;
  auto addCount = [&](int64_t &total, int64_t count) {
    if (count < 0 || total > INT64_MAX - count)
      accessCountsValid = false;
    else
      total += count;
  };
  for (const auto &task : a.tasks)
    for (const auto &access : task.accesses) {
      int64_t count = 1;
      for (const auto &loop : task.loops)
        if (loop.operation->isProperAncestor(access.operation)) {
          int64_t iterations = (loop.upper - loop.lower - 1) / loop.step + 1;
          if (count > INT64_MAX / iterations) {
            accessCountsValid = false;
            count = 0;
            break;
          }
          count *= iterations;
        }
      auto type = cast<MemRefType>(a.storages[access.storage].root.getType());
      int64_t width = (type.getElementType().getIntOrFloatBitWidth() + 7) / 8;
      if (count > INT64_MAX / width) {
        accessCountsValid = false;
        continue;
      }
      if (access.read) {
        addCount(scalarReads, count);
        addCount(logicalReadBytes, count * width);
      }
      if (access.write) {
        addCount(scalarWrites, count);
        addCount(logicalWriteBytes, count * width);
      }
    }
  int64_t phasePeak = p.residentBytes;
  if (p.strategy == "serial-tiled")
    stagePoolBytes(a, p, &phasePeak);
  int64_t logicalBytes = 0, actualBytes = 0;
  for (const auto &storage : a.storages)
    if (storage.role == "input" || storage.role == "output")
      logicalBytes += storageBytes(storage);
  int64_t stagingCopyBytes = 0;
  for (const auto &transfer : p.transfers) {
    actualBytes += transfer.bytes;
    if (p.strategy == "serial-tiled" && transfer.direction == "host_to_l1") {
      const auto &phase = p.phases[transfer.phase];
      auto region = llvm::find_if(phase.regions, [&](const auto &region) {
        return region.storage == transfer.storage;
      });
      if (region != phase.regions.end() && region->read && region->write)
        stagingCopyBytes += transfer.bytes;
    }
  }
  llvm::json::Object report{
      {"schema", "orbit.generic.mapping.v1"},
      {"logical_boundary_bytes", logicalBytes},
      {"logical_scalar_reads", accessCountsValid
                                   ? llvm::json::Value(scalarReads)
                                   : llvm::json::Value(nullptr)},
      {"logical_scalar_writes", accessCountsValid
                                    ? llvm::json::Value(scalarWrites)
                                    : llvm::json::Value(nullptr)},
      {"logical_read_bytes", accessCountsValid
                                 ? llvm::json::Value(logicalReadBytes)
                                 : llvm::json::Value(nullptr)},
      {"logical_write_bytes", accessCountsValid
                                  ? llvm::json::Value(logicalWriteBytes)
                                  : llvm::json::Value(nullptr)},
      {"actual_dma_bytes", actualBytes},
      {"discovered_mapping_opportunities", std::move(opportunities)},
      {"plan_identity_policy",
       "orbit.chosen-plan.v19.excludes-proposal-diagnostics"},
      {"program_mapping_choices", std::move(mappingChoices)},
      {"residency_scopes", std::move(residencyScopes)},
      {"data_organization_scope",
       p.choices.residency == "whole-program"
           ? "shared input and intermediate copies across all program "
             "consumers; whole-program capacity checked"
       : p.choices.grouping == "tile-major"
           ? "scope-local producer tiles shared by internal consumers; "
             "external consumers retain explicit boundary exports; tile "
             "working-set capacity checked"
       : p.choices.dataOrganization == "retain-across-users"
           ? "readonly source-instance users within one execution group; AIR "
             "stage-local"
           : "phase-local input copies; source reduction/output retention "
             "remains required"},
      {"program_communication", serializeProgramCommunication(p)},
      {"program_graph", serializeProgramGraph(p.communication)},
      {"program_graph_id", programGraphIdentity(p.communication)},
      {"backend_realization_feature_scope",
       "phase/stage/core counters describe the checked AIR realization; "
       "source/instances/traffic/submissions come from the common program "
       "graph"},
      {"task_loop_orders", std::move(loopOrders)},
      {"task_implementations", std::move(implementations)},
      {"readonly_retentions", std::move(readonlyRetentions)},
      {"execution_groups", std::move(executionGroups)},
      {"local_staging_copy_bytes", stagingCopyBytes},
      {"retained_regions", std::move(retentions)},
      {"fused_initializer_tasks", std::move(fusedInitializers)},
      {"hardware_stages", std::move(hardwareStages)},
      {"channel_allocations", std::move(channelAllocations)},
      {"channel_merge_policy", p.channelMergePolicy},
      {"max_group_inputs", p.targetCapabilities.maxGroupInputs},
      {"max_group_outputs", p.targetCapabilities.maxGroupOutputs},
      {"max_hardware_stage_expansion", int64_t(maxHardwareStageExpansion)},
      {"hardware_stage_expansion_scope",
       "bounded host/codegen expansion; not a measured silicon or local memory "
       "resource limit"},
      {"host_buffers", std::move(hostBuffers)},
      {"system_memory_peak_bytes", systemMemoryPeak},
      {"system_memory_accounting_scope",
       "system-memory BO tensor payload; excludes runner host array mirrors, "
       "references, runtime and compiler allocations"},
      {"backend_tensor_padding_bytes", 0},
      {"local_allocation_lifetime",
       p.strategy == "serial-tiled"
           ? "per-task stage; reuse only after serial completion dependency"
           : "whole_herd"},
      {"plan_id", p.planId},
      {"body_fingerprint", p.bodyFingerprint},
      {"strategy", p.strategy},
      {"placement", "npu_single_core_serial"},
      {"capacity_bytes", p.capacityBytes},
      {"target_memory", serializeGenericMemory(p.memoryBinding)},
      {"target_capabilities",
       serializeGenericCapabilities(p.targetCapabilities)},
      {"target_capability_identity", p.targetCapabilities.identity},
      {"resident_bytes", p.residentBytes},
      {"phase_peak_live_bytes", phasePeak},
      {"allocation_lifetime",
       p.strategy == "serial-tiled"
           ? "per-task device stage; storage pools reused across serial tiles"
           : "whole_herd"},
      {"tile_sizes", std::move(tileSizes)},
      {"phases", std::move(phases)},
      {"phase_count", int64_t(p.phases.size())},
      {"transfer_count", int64_t(p.transfers.size())},
      {"phase_schedule_digest", scheduleDigest.str().str()},
      {"expanded_phase_details_omitted", compact},
      {"phase_domains", std::move(phaseDomains)},
      {"transfer_summary", std::move(transferSummary)},
      {"host_plan_representation", "explicit phases; compact JSON domains and "
                                   "compact generated AIR for large schedules"},
      {"tasks", std::move(tasks)},
      {"transfers", std::move(transfers)},
      {"residency", std::move(residency)},
      {"analysis_status", "supported"},
      {"codegen_status", "requires_backend_verification"},
      {"cost_status", p.hasCost ? "theoretical" : "uncalibrated"},
      {"estimated_ms", p.hasCost ? llvm::json::Value(p.estimatedMs)
                                 : llvm::json::Value(nullptr)},
      {"cost_provenance", p.costProvenance},
      {"theoretical_coefficients",
       p.hasCost ? llvm::json::Value(llvm::json::Object{
                       {"launch_ms", p.theoreticalLaunchMs},
                       {"per_scalar_operation_ms", p.theoreticalPerOperationMs},
                       {"per_transfer_byte_ms", p.theoreticalPerByteMs}})
                 : llvm::json::Value(nullptr)},
      {"physical_execution", false},
      {"backend_internal_transport",
       "AIR compiler may add shim/L2 staging; final IR ledger required; not "
       "measured traffic"}};
  if (p.moduleWideChannelConversion)
    report["backend_legalization"] = llvm::json::Object{
        {"channel_conversion_scope", "module"},
        {"async_herd_scope", "module"},
        {"local_release_scope", "static-hardware-stage"},
        {"extra_transfer_bytes", 0},
        {"extra_tensor_padding_bytes", 0},
        {"reason",
         "installed DMA-to-channel pass converts every stage; all compact "
         "stage loops require matching async herd token scope"}};
  if (p.directCost.enabled) {
    report["direct_air_cost"] = directCostReport(p.directCost);
    report["cost_status"] = p.directCost.status;
    report["theoretical_coefficients"] = nullptr;
  }
  return report;
}
bool orbit::taskflow::importNativeInterface(mlir::ModuleOp module,
                                            const Program &program,
                                            std::string &e) {
  auto fs = module.getOps<mlir::func::FuncOp>();
  if (fs.empty())
    return fail(e, "import missing function");
  auto fn = *fs.begin();
  std::vector<mlir::memref::AllocOp> allocs;
  for (auto alloc : fn.getOps<mlir::memref::AllocOp>())
    allocs.push_back(alloc);
  if (allocs.size() != program.buffers.size())
    return fail(e, "compatibility interface allocation count mismatch");
  mlir::Builder b(module.getContext());
  for (size_t i = 0; i < allocs.size(); ++i) {
    auto &buffer = program.buffers[i];
    auto alloc = allocs[i];
    if (buffer.role == "temporary") {
      alloc->setAttr("orbit.name", b.getStringAttr(buffer.id));
      continue;
    }
    unsigned index = fn.getNumArguments();
    auto arg =
        fn.getBody().front().addArgument(alloc.getType(), alloc.getLoc());
    llvm::SmallVector<mlir::Type> inputs(fn.getArgumentTypes());
    inputs.push_back(arg.getType());
    fn.setType(b.getFunctionType(inputs, {}));
    fn.setArgAttr(index, "orbit.role", b.getStringAttr(buffer.role));
    fn.setArgAttr(index, "orbit.name", b.getStringAttr(buffer.id));
    alloc.replaceAllUsesWith(arg);
    alloc.erase();
  }
  fn->setAttr("orbit.noalias", b.getUnitAttr());
  module->removeAttr("orbit.plan");
  return true;
}
llvm::json::Value orbit::taskflow::evaluateGenericMLIR(
    llvm::StringRef text, const llvm::json::Value &cfg, std::string &e) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::taskflow::TaskflowDialect, mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect, mlir::math::MathDialect,
                  mlir::memref::MemRefDialect, mlir::func::FuncDialect,
                  mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
  if (module &&
      module->getOperation()->hasAttr("orbit.program_partition_extent"))
    return evaluatePartitionedGenericMLIR(*module, cfg, e);
  GenericAnalysis a;
  GenericMappingPlan p;
  if (!module || mlir::failed(mlir::verify(*module)) ||
      !analyzeGenericProgram(*module, a, e) ||
      !planGenericMapping(a, cfg, p, e))
    return llvm::json::Object{{"status", "unsupported"}, {"reason", e}};
  if (llvm::any_of(p.taskImplementations, [](const auto &implementation) {
        return implementation.kind == "independent-lane-f32" ||
               (implementation.kind == "scalar-product-vector-add-f32" &&
                implementation.lanes != 8);
      }))
    return llvm::json::Object{
        {"status", "unsupported"},
        {"reason", "installed AIR target requires mixed f32 eight-lane "
                   "realization; strict vector multiply and four-lane mixed "
                   "legalization are unavailable"},
        {"analysis", serializeGenericAnalysis(a)},
        {"best_plan", serializeGenericMapping(a, p)}};
  return llvm::json::Object{
      {"status", p.hasCost ? "passed" : "uncalibrated"},
      {"analysis", serializeGenericAnalysis(a)},
      {"program_graph", serializeProgramGraph(p.communication)},
      {"program_graph_id", programGraphIdentity(p.communication)},
      {"program_graph_dot", renderProgramGraphDOT(p.communication)},
      {"best_plan", serializeGenericMapping(a, p)}};
}

std::string orbit::taskflow::emitNativeTaskflow(const Program &program,
                                                std::string &e) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::taskflow::TaskflowDialect, mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect, mlir::math::MathDialect,
                  mlir::memref::MemRefDialect, mlir::func::FuncDialect,
                  mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry);
  auto module =
      mlir::parseSourceString<mlir::ModuleOp>(emitTaskflow(program), &context);
  if (!module || !importNativeInterface(*module, program, e) ||
      mlir::failed(mlir::verify(*module)))
    return {};
  std::string out;
  llvm::raw_string_ostream stream(out);
  module->print(stream);
  return out;
}
