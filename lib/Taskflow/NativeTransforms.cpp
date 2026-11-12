#include "Taskflow/NativeTransforms.h"
#include "Taskflow/GenericAnalysis.h"
#include "Taskflow/GenericMapping.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include <algorithm>
#include <map>
#include <set>
using namespace mlir;
namespace {
SmallVector<affine::AffineForOp> loops(ModuleOp module) {
  SmallVector<affine::AffineForOp> result;
  module.walk([&](mlir::taskflow::TaskflowTaskOp task) {
    for (auto loop : task.getBody().front().getOps<affine::AffineForOp>()) {
      if (!loop.hasConstantLowerBound() || !loop.hasConstantUpperBound() ||
          loop.getNumIterOperands())
        continue;
      int64_t lower = loop.getConstantLowerBound(),
              upper = loop.getConstantUpperBound(),
              step = loop.getStep().getSExtValue();
      if (step > 0 && lower >= 0 && upper <= INT32_MAX && upper > lower &&
          (upper - lower - 1) / step + 1 >= 2)
        result.push_back(loop);
    }
  });
  return result;
}
} // namespace
bool orbit::taskflow::hasSplittableNativeLoops(ModuleOp module) {
  return !loops(module).empty();
}
bool orbit::taskflow::splitNativeOuterLoops(ModuleOp module,
                                            std::string &error) {
  GenericAnalysis before;
  if (!analyzeGenericProgram(module, before, error))
    return false;
  auto candidates = loops(module);
  if (candidates.empty()) {
    error = "native loop split requires a static outer affine loop with at "
            "least two iterations";
    return false;
  }
  for (auto loop : candidates) {
    int64_t lo = loop.getConstantLowerBound(),
            hi = loop.getConstantUpperBound(),
            step = loop.getStep().getSExtValue();
    int64_t middle = lo + (((hi - lo - 1) / step + 1) / 2) * step;
    OpBuilder builder(loop);
    builder.setInsertionPointAfter(loop);
    auto tail = cast<affine::AffineForOp>(builder.clone(*loop.getOperation()));
    loop.setConstantUpperBound(middle);
    tail.setConstantLowerBound(middle);
  }
  module->removeAttr("orbit.plan");
  if (failed(verify(module))) {
    error = "native split produced invalid IR";
    return false;
  }
  GenericAnalysis after;
  return analyzeGenericProgram(module, after, error);
}

bool orbit::taskflow::setNativeAIRTileSizes(ModuleOp module, int64_t m,
                                            int64_t n, int64_t k,
                                            std::string &error) {
  if (m <= 0 || n <= 0 || k <= 0) {
    error = "AIR tile sizes must all be positive";
    return false;
  }
  ProgramMappingChoice current;
  if (!readNativeMappingChoice(module, current, error))
    return false;
  if (current.residency == "whole-program") {
    ProgramMappingChoice tiled;
    tiled.residency = "tiled-phase-local";
    if (!setNativeMappingChoice(module, tiled, error))
      return false;
  }
  module->setAttr("orbit.air_tile_sizes",
                  Builder(module.getContext()).getDenseI64ArrayAttr({m, n, k}));
  return true;
}

namespace {
bool resolveMappingChoice(const orbit::taskflow::ProgramMappingChoice &input,
                          orbit::taskflow::ProgramMappingChoice &choice,
                          std::string &error) {
  using namespace orbit::taskflow;
  choice = input;
  // The old public CLI accepted residency. Keep it as an input adapter;
  // canonical source always distinguishes lifetime policy from realization.
  if (choice.residency == "retain-across-users")
    choice.residency = "readonly-panel";
  if (choice.dataOrganization.empty())
    choice.dataOrganization = (choice.residency == "readonly-panel" ||
                               choice.residency == "whole-program")
                                  ? "retain-across-users"
                                  : "phase-local";
  bool tileMajor = choice.grouping == "tile-major",
       shapeMajor = choice.grouping == "shape-major";
  if (tileMajor || shapeMajor) {
    if (choice.residency.empty())
      choice.residency = "tiled-phase-local";
    if (choice.residency != "tiled-phase-local" || choice.storage != -1 ||
        choice.reuseLoop != -1 || choice.panelLoop != -1) {
      error = "instance grouping requires plain tiled realization without "
              "readonly-panel scope";
      return false;
    }
    if (input.dataOrganization.empty())
      choice.dataOrganization =
          tileMajor ? "retain-across-users" : "phase-local";
    if (choice.dataOrganization !=
            (tileMajor ? "retain-across-users" : "phase-local") ||
        (tileMajor &&
         ((choice.residencyGroups.empty() &&
           (choice.taskGroup.empty() || choice.retainedStorages.empty())) ||
          (!choice.residencyGroups.empty() &&
           (!choice.taskGroup.empty() || !choice.retainedStorages.empty())))) ||
        (shapeMajor &&
         (!choice.taskGroup.empty() || !choice.retainedStorages.empty() ||
          !choice.residencyGroups.empty()))) {
      error = "instance grouping has inconsistent retention scope";
      return false;
    }
    return true;
  }
  if (!choice.taskGroup.empty() || !choice.retainedStorages.empty() ||
      !choice.residencyGroups.empty()) {
    error = "task/data groups require tile-major grouping";
    return false;
  }
  ProgramDataPolicy policy;
  if (!parseProgramDataPolicy(choice.dataOrganization, policy, error))
    return false;
  choice.dataOrganization = programDataPolicyName(policy).str();
  bool retained = policy == ProgramDataPolicy::RetainAcrossUsers;
  bool whole = choice.residency == "whole-program";
  bool tiled = choice.residency == "tiled-phase-local";
  if ((whole && !retained) || (tiled && retained)) {
    error = "explicit residency contradicts data organization";
    return false;
  }
  std::string realization = whole      ? "whole-program"
                            : tiled    ? "tiled-phase-local"
                            : retained ? "readonly-panel"
                                       : "phase-local";
  std::string grouping = whole      ? "whole-program"
                         : retained ? "readonly-panel-scope"
                                    : "task-shape-stages";
  if ((!choice.residency.empty() && choice.residency != realization) ||
      (!choice.grouping.empty() && choice.grouping != grouping)) {
    error = "data organization contradicts the requested residency realization "
            "or grouping";
    return false;
  }
  choice.residency = realization;
  choice.grouping = grouping;
  if (retained && !whole) {
    if (choice.storage < 0 || choice.reuseLoop < 0 || choice.panelLoop < 0 ||
        choice.reuseLoop == choice.panelLoop) {
      error = "retain-across-users requires storage and distinct source "
              "reuse/panel loops";
      return false;
    }
  } else if (choice.storage != -1 || choice.reuseLoop != -1 ||
             choice.panelLoop != -1) {
    error = "non-panel data organization cannot carry a retained storage/loop "
            "scope";
    return false;
  }
  return true;
}
} // namespace

bool orbit::taskflow::readNativeMappingChoice(ModuleOp module,
                                              ProgramMappingChoice &choice,
                                              std::string &error) {
  ProgramMappingChoice input;
  Attribute raw = module->getAttr("orbit.mapping");
  if (raw) {
    auto mapping = dyn_cast<DictionaryAttr>(raw);
    if (!mapping) {
      error = "orbit.mapping must be a dictionary";
      return false;
    }
    for (auto item : mapping) {
      StringRef name = item.getName().getValue();
      if (name != "data_organization" && name != "residency" &&
          name != "grouping" && name != "storage" && name != "reuse_loop" &&
          name != "panel_loop" && name != "task_group" &&
          name != "retained_storages" && name != "residency_groups") {
        error = "unknown program mapping choice field: " + name.str();
        return false;
      }
      if (name == "residency_groups") {
        auto groups = dyn_cast<ArrayAttr>(item.getValue());
        if (!groups) {
          error = "residency_groups must be an array of dictionaries";
          return false;
        }
        for (auto raw : groups) {
          auto group = dyn_cast<DictionaryAttr>(raw);
          if (!group || group.size() != 2) {
            error = "residency group requires tasks and retained_storages";
            return false;
          }
          ProgramResidencyGroup scope;
          for (auto field : group) {
            auto name = field.getName().getValue();
            if (name != "tasks" && name != "retained_storages") {
              error = "unknown residency group field";
              return false;
            }
            auto values = dyn_cast<ArrayAttr>(field.getValue());
            if (!values) {
              error = "residency group members must be integer arrays";
              return false;
            }
            auto &out = name == "tasks" ? scope.tasks : scope.retainedStorages;
            for (auto raw : values) {
              auto value = dyn_cast<IntegerAttr>(raw);
              if (!value || value.getInt() < 0) {
                error = "invalid residency group identity";
                return false;
              }
              out.push_back(size_t(value.getInt()));
            }
          }
          if (scope.tasks.size() < 2 || scope.retainedStorages.empty()) {
            error = "residency group requires multiple tasks and retained data";
            return false;
          }
          input.residencyGroups.push_back(std::move(scope));
        }
        if (input.residencyGroups.empty()) {
          error = "explicit residency_groups must be nonempty";
          return false;
        }
      } else if (name == "task_group" || name == "retained_storages") {
        auto values = dyn_cast<ArrayAttr>(item.getValue());
        if (!values) {
          error = "task/data groups must be integer arrays";
          return false;
        }
        auto &dest =
            name == "task_group" ? input.taskGroup : input.retainedStorages;
        for (auto raw : values) {
          auto value = dyn_cast<IntegerAttr>(raw);
          if (!value || value.getInt() < 0) {
            error = "negative/noninteger group identity";
            return false;
          }
          dest.push_back(size_t(value.getInt()));
        }
      } else if (name == "storage" || name == "reuse_loop" ||
                 name == "panel_loop") {
        auto value = dyn_cast<IntegerAttr>(item.getValue());
        if (!value || !value.getValue().isSignedIntN(64)) {
          error = "mapping storage/loop choices must be signed 64-bit integers";
          return false;
        }
        if (name == "storage")
          input.storage = value.getInt();
        else if (name == "reuse_loop")
          input.reuseLoop = value.getInt();
        else
          input.panelLoop = value.getInt();
      } else {
        auto value = dyn_cast<StringAttr>(item.getValue());
        if (!value || value.getValue().empty()) {
          error = "explicit mapping policy/realization/grouping choices must "
                  "be nonempty strings";
          return false;
        }
        if (name == "data_organization")
          input.dataOrganization = value.str();
        else if (name == "residency")
          input.residency = value.str();
        else
          input.grouping = value.str();
      }
    }
  }
  return resolveMappingChoice(input, choice, error);
}

bool orbit::taskflow::setNativeMappingChoice(ModuleOp module,
                                             const ProgramMappingChoice &input,
                                             std::string &error) {
  ProgramMappingChoice choice;
  if (!resolveMappingChoice(input, choice, error))
    return false;
  // Canonical default: no candidate identity for merely explicit defaults.
  if (choice.residency == "phase-local" &&
      choice.grouping == "task-shape-stages") {
    module->removeAttr("orbit.mapping");
    return true;
  }
  if (choice.residency == "whole-program")
    module->removeAttr("orbit.air_tile_sizes");
  Builder b(module.getContext());
  NamedAttrList attrs;
  attrs.set("data_organization", b.getStringAttr(choice.dataOrganization));
  attrs.set("residency", b.getStringAttr(choice.residency));
  attrs.set("grouping", b.getStringAttr(choice.grouping));
  attrs.set("storage", b.getI64IntegerAttr(choice.storage));
  attrs.set("reuse_loop", b.getI64IntegerAttr(choice.reuseLoop));
  attrs.set("panel_loop", b.getI64IntegerAttr(choice.panelLoop));
  auto list = [&](const std::vector<size_t> &values) {
    SmallVector<Attribute> result;
    for (auto v : values)
      result.push_back(b.getI64IntegerAttr(v));
    return b.getArrayAttr(result);
  };
  if (!choice.taskGroup.empty())
    attrs.set("task_group", list(choice.taskGroup));
  if (!choice.retainedStorages.empty())
    attrs.set("retained_storages", list(choice.retainedStorages));
  if (!choice.residencyGroups.empty()) {
    SmallVector<Attribute> groups;
    for (const auto &g : choice.residencyGroups) {
      NamedAttrList fields;
      fields.set("tasks", list(g.tasks));
      fields.set("retained_storages", list(g.retainedStorages));
      groups.push_back(fields.getDictionary(module.getContext()));
    }
    attrs.set("residency_groups", b.getArrayAttr(groups));
  }
  module->setAttr("orbit.mapping", attrs.getDictionary(module.getContext()));
  return true;
}

bool orbit::taskflow::normalizeNativeMappingChoice(ModuleOp module,
                                                   std::string &error) {
  ProgramMappingChoice choice;
  return readNativeMappingChoice(module, choice, error) &&
         setNativeMappingChoice(module, choice, error);
}

bool orbit::taskflow::readNativeInitializerTasks(
    const GenericAnalysis &analysis, std::vector<size_t> &tasks,
    std::string &error) {
  tasks.clear();
  auto module = analysis.function->getParentOfType<ModuleOp>();
  auto raw = module->getAttr("orbit.generated_initializer_tasks");
  if (!raw)
    return true;
  auto array = dyn_cast<DenseI64ArrayAttr>(raw);
  std::set<size_t> seen;
  if (!array) {
    error = "generated initializer table requires dense i64 task indices";
    return false;
  }
  for (int64_t index : array.asArrayRef()) {
    if (index < 0 || uint64_t(index) >= analysis.tasks.size() ||
        !seen.insert(index).second) {
      error = "generated initializer task is duplicated or out of range";
      return false;
    }
    const auto &task = analysis.tasks[index];
    bool valid = !task.accesses.empty();
    for (const auto &access : task.accesses)
      valid &= !access.read && access.write && access.exactRegion;
    task.operation->walk([&](Operation *operation) {
      if (operation == task.operation ||
          isa<affine::AffineForOp, affine::AffineYieldOp, arith::ConstantOp,
              mlir::taskflow::TaskflowYieldOp>(operation))
        return;
      if (auto store = dyn_cast<affine::AffineStoreOp>(operation)) {
        valid &= bool(store.getValue().getDefiningOp<arith::ConstantOp>());
        return;
      }
      valid = false;
    });
    // Provenance is a derived hint, not source semantics. A legitimate source
    // edit can invalidate it, for example by computing a fill value. Do not
    // reject the program or let a stale claim suppress its strict mode request.
    // Grant the scalar initializer role only while the source proof holds.
    if (valid)
      tasks.push_back(index);
  }
  return true;
}

bool orbit::taskflow::validateNativeImplementation(const std::string &kind,
                                                   int64_t lanes,
                                                   std::string &error) {
  const bool scalar = kind == "scalar" || kind == "aie2p-bf16-mmul-f32acc";
  const bool vector = kind == "independent-lane-f32";
  const bool mixed = kind == "scalar-product-vector-add-f32";
  if ((!scalar && !vector && !mixed) || (scalar && lanes != 1) ||
      ((vector || mixed) &&
       (lanes < 2 || lanes > 16 || (lanes & (lanes - 1)))) ||
      (mixed && lanes != 4 && lanes != 8)) {
    error = "implementation requires scalar/matrix lanes 1, independent lanes "
            "power-of-two 2..16, or mixed lanes 4/8";
    return false;
  }
  return true;
}

bool orbit::taskflow::readNativeTaskImplementations(
    ModuleOp module, size_t taskCount,
    std::vector<NativeTaskImplementationChoice> &choices, std::string &error) {
  choices.clear();
  auto raw = module->getAttr("orbit.task_implementations");
  if (!raw)
    return true;
  auto array = dyn_cast<ArrayAttr>(raw);
  if (!array) {
    error = "orbit.task_implementations requires an array";
    return false;
  }
  std::set<size_t> seen;
  for (auto value : array) {
    auto object = dyn_cast<DictionaryAttr>(value);
    auto task = object ? object.getAs<IntegerAttr>("task") : IntegerAttr{};
    auto kind = object ? object.getAs<StringAttr>("kind") : StringAttr{};
    auto lanes = object ? object.getAs<IntegerAttr>("lanes") : IntegerAttr{};
    if (!object || object.size() != 3 || !task || !kind || !lanes ||
        task.getValue().getBitWidth() > 64 ||
        lanes.getValue().getBitWidth() > 64 || task.getInt() < 0 ||
        uint64_t(task.getInt()) >= taskCount ||
        !seen.insert(task.getInt()).second) {
      error =
          "task implementation requires unique in-range task, kind and lanes";
      return false;
    }
    if (!validateNativeImplementation(kind.getValue().str(), lanes.getInt(),
                                      error))
      return false;
    choices.push_back(
        {size_t(task.getInt()), kind.getValue().str(), lanes.getInt()});
  }
  std::sort(choices.begin(), choices.end(),
            [](const auto &a, const auto &b) { return a.task < b.task; });
  return true;
}

bool orbit::taskflow::setNativeTaskImplementations(
    ModuleOp module, size_t taskCount,
    const std::vector<NativeTaskImplementationChoice> &choices,
    std::string &error) {
  auto sorted = choices;
  std::sort(sorted.begin(), sorted.end(),
            [](const auto &a, const auto &b) { return a.task < b.task; });
  Builder builder(module.getContext());
  SmallVector<Attribute> entries;
  for (size_t i = 0; i < sorted.size(); ++i) {
    const auto &choice = sorted[i];
    if (choice.task >= taskCount || (i && sorted[i - 1].task == choice.task)) {
      error = "task implementation has duplicate or out-of-range task";
      return false;
    }
    if (!validateNativeImplementation(choice.kind, choice.lanes, error))
      return false;
    entries.push_back(builder.getDictionaryAttr(
        {builder.getNamedAttr("task", builder.getI64IntegerAttr(choice.task)),
         builder.getNamedAttr("kind", builder.getStringAttr(choice.kind)),
         builder.getNamedAttr("lanes",
                              builder.getI64IntegerAttr(choice.lanes))}));
  }
  if (entries.empty())
    module->removeAttr("orbit.task_implementations");
  else
    module->setAttr("orbit.task_implementations",
                    builder.getArrayAttr(entries));
  return true;
}

bool orbit::taskflow::setNativeImplementation(ModuleOp module,
                                              const std::string &kind,
                                              int64_t lanes,
                                              std::string &error) {
  if (!validateNativeImplementation(kind, lanes, error))
    return false;
  // Scalar/1 is the module default. Keep one canonical representation so
  // switching back from a vector choice cannot create a second identity for
  // the same source body, checked plan and executable implementation.
  if (kind == "scalar") {
    module->removeAttr("orbit.implementation");
    module->removeAttr("orbit.vector_lanes");
    return true;
  }
  Builder b(module.getContext());
  module->setAttr("orbit.implementation", b.getStringAttr(kind));
  module->setAttr("orbit.vector_lanes", b.getI64IntegerAttr(lanes));
  return true;
}

bool orbit::taskflow::distributeNativeInitialization(ModuleOp module,
                                                     std::string &error) {
  GenericAnalysis before;
  if (!analyzeGenericProgram(module, before, error))
    return false;
  std::vector<size_t> initializerTasks;
  if (!readNativeInitializerTasks(before, initializerTasks, error))
    return false;
  std::set<Operation *> initializers;
  for (auto task : initializerTasks)
    initializers.insert(before.tasks[task].operation);
  std::vector<NativeTaskImplementationChoice> previous;
  if (!readNativeTaskImplementations(module, before.tasks.size(), previous,
                                     error))
    return false;
  std::map<Operation *, NativeTaskImplementationChoice> selections;
  for (const auto &entry : previous)
    selections[before.tasks[entry.task].operation] = entry;
  SmallVector<mlir::taskflow::TaskflowTaskOp> tasks;
  module.walk([&](mlir::taskflow::TaskflowTaskOp t) { tasks.push_back(t); });
  for (auto task : tasks) {
    if (task.getWillWrites().size() != 1 || task.getDoneWrites().size() != 1 ||
        !task.getDoneReads().empty() || !task.getValueOutputs().empty())
      continue;
    auto facts = llvm::find_if(before.tasks, [&](const auto &t) {
      return t.operation == task.getOperation();
    });
    if (facts == before.tasks.end())
      continue;
    size_t outputStorage = facts->argumentStorages[task.getWillReads().size()];
    // Two body arguments may denote the same dependence-state storage even
    // under the noalias external ABI contract. Do not distribute such tasks.
    if (llvm::count(facts->argumentStorages, outputStorage) != 1)
      continue;
    SmallVector<affine::AffineForOp> outer;
    Block *block = &task.getBody().front();
    affine::AffineStoreOp init;
    affine::AffineForOp update;
    while (true) {
      SmallVector<Operation *> ops;
      for (auto &op : block->without_terminator()) {
        if (block == &task.getBody().front() && isa<arith::ConstantOp>(op))
          continue;
        ops.push_back(&op);
      }
      if (ops.size() == 1 && isa<affine::AffineForOp>(ops[0])) {
        auto loop = cast<affine::AffineForOp>(ops[0]);
        outer.push_back(loop);
        block = loop.getBody();
        continue;
      }
      if (ops.size() == 2) {
        init = dyn_cast<affine::AffineStoreOp>(ops[0]);
        update = dyn_cast<affine::AffineForOp>(ops[1]);
      }
      break;
    }
    if (!init || !update || outer.empty())
      continue;
    auto output =
        task.getBody().front().getArgument(task.getWillReads().size());
    if (init.getMemRef() != output ||
        !init.getValue().getDefiningOp<arith::ConstantOp>())
      continue;
    // Exact projections onto every outer IV prove independence between slices.
    auto projection = [&](AffineMap map, ValueRange operands) {
      if (!map.isIdentity() || operands.size() != outer.size())
        return false;
      for (auto [i, v] : llvm::enumerate(operands))
        if (v != outer[i].getInductionVar())
          return false;
      return true;
    };
    if (!projection(init.getAffineMap(), init.getMapOperands()))
      continue;
    bool safe = true;
    update.walk([&](Operation *op) {
      if (auto store = dyn_cast<affine::AffineStoreOp>(op)) {
        if (store.getMemRef() != output ||
            !projection(store.getAffineMap(), store.getMapOperands()))
          safe = false;
      } else if (auto load = dyn_cast<affine::AffineLoadOp>(op)) {
        if (load.getMemRef() == output &&
            !projection(load.getAffineMap(), load.getMapOperands()))
          safe = false;
      } else if (op->getName().getDialectNamespace() == "memref")
        safe = false;
    });
    if (!safe)
      continue;
    OpBuilder builder(task);
    auto prologue = cast<mlir::taskflow::TaskflowTaskOp>(builder.clone(*task));
    prologue.setTaskName((task.getTaskName() + ".initialize").str());
    // A generated fill is an adapter component, not another request to run
    // the original compute implementation (for example a matrix contraction).
    initializers.insert(prologue.getOperation());
    if (selections.count(task.getOperation()))
      selections[prologue.getOperation()] = {0, "scalar", 1};
    SmallVector<affine::AffineForOp> clonedLoops;
    prologue.walk<WalkOrder::PreOrder>(
        [&](affine::AffineForOp loop) { clonedLoops.push_back(loop); });
    clonedLoops[outer.size()].erase();
    init.erase();
    task->setOperand(task.getWillReads().size(), prologue.getDoneWrites()[0]);
    task->setAttr("orbit.distributed_initialization", builder.getUnitAttr());
  }
  if (failed(verify(module))) {
    error = "initialization distribution invalid IR";
    return false;
  }
  GenericAnalysis after;
  if (!analyzeGenericProgram(module, after, error))
    return false;
  std::vector<NativeTaskImplementationChoice> expanded;
  for (size_t task = 0; task < after.tasks.size(); ++task) {
    auto selected = selections.find(after.tasks[task].operation);
    if (selected != selections.end())
      expanded.push_back({task, selected->second.kind, selected->second.lanes});
  }
  SmallVector<int64_t> initializerIndices;
  for (size_t task = 0; task < after.tasks.size(); ++task)
    if (initializers.count(after.tasks[task].operation))
      initializerIndices.push_back(task);
  if (initializerIndices.empty())
    module->removeAttr("orbit.generated_initializer_tasks");
  else
    module->setAttr(
        "orbit.generated_initializer_tasks",
        Builder(module.getContext()).getDenseI64ArrayAttr(initializerIndices));
  return setNativeTaskImplementations(module, after.tasks.size(), expanded,
                                      error);
}
